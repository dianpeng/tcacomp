// Package schedule is the final optimization pass (quick-261021-ta3,
// doc/optana/schedule.md): it brings the Level-1 sea-of-nodes back to an
// AST-style CFG (Kernel → Group/Loop blocks) with scheduling-number LICM,
// replacing the obsolete statement-level scheduler (deleted with this
// rewrite — locked decision D-02). The emitted AST CFG lives in a SEPARATE
// field on ir.Graph (locked decision D-01) and is consumed lastly by
// final_lower and code generation (doc/optana/final_lower.md).
//
// File map:
//
//   - scheduler.go — the pass contract (registry, ID/Requires/Produces/
//     Preserved), the fail-closed entry gates, and Run: the top-level
//     composition of the scheduling algorithm's phases, one call per
//     design-doc step.
//   - phases.go — the algorithm itself: the scheduleRun object (member
//     data as shared state) and one small method per doc step, each
//     carrying its ruling citations. Behavior is byte-identical to the
//     pre-refactor monolith (the test suites pin it unmodified).
//   - nest.go — the forest adapter over the shared generic loopforest
//     analysis + the pass-local placement derivations (memberSetOf: the
//     ONE IV-membership relation; completionCutOf: the C3 completion
//     cut).
//   - order.go — per-rank stable topological ordering.
//   - net.go — the post-placement cross-block ordering net.
//
// The pass inserts NO runtime-object/glue nodes (locked decision D-04):
// the prologue/epilogue Groups are the landing pads final_lower fills;
// since quick-261022-b2m the epilogues also carry the pass-placed
// after-nest consumers (the end node included).
//
// Two documented readings of the spec's numbering, both required for the
// algorithm to be correct on the real tile-lower output shape:
//
//   - "Iteration node" (spec 3.1) anchors the loop's HEADER — the ForLoop
//     node plus its iv/bound operands — at the loop's nest depth, fixed
//     during propagation. A body node only ever reaches the anchor through
//     def chains (the loop→body edge is use→def, so the loop node itself
//     is never in a body node's dependency closure), and the iv is exactly
//     what loop-varying index math depends on.
//
// PLACEMENT MODEL (quick-261022-b2m's no-clamp model, recast onto the
// fix1.md sea model by quick-261024-sdk — the body/reach derivations and
// the HI-02 floor pass are deleted; ONE relation everywhere):
//
//   - MEMBERSHIP (fix1.md §0): a node is inside loop L's iteration iff its
//     data-dep closure references L's IV ("a use of L's IV is membership
//     in L"), with ForLoop nodes OPAQUE to the walk — a reference TO a
//     loop node is completion order, never a path to its IV. capOp, capUni,
//     and cluster ownership all read this one relation (nest.go
//     memberSetOf); propagation maxes over DataDeps+EffectDeps, so the
//     effect chain is still treated identically to input edges.
//   - COMPLETION CUT (C3): "def = ForLoop node → after whole loop" — a
//     node whose data/effect closure TOUCHES a loop node, or that the end
//     node's completion chain parks after a loop (NewReturnChained's
//     trailing-operand order; P6 keeps "end.operand = final store"), is
//     ordered AFTER that loop's whole nest: both caps are cut below the
//     loop's depth (nest.go completionCutOf). This resolves D-c's
//     incoherence — uses L's IV yet ordered after L — as AFTER, sending
//     the end node, repointed post-nest stores, and kernel-level
//     reduction consumers to the epilogue while their raw IV membership
//     runs deep.
//   - TWO-CASE PLACEMENT, NO CLAMP: SN <= capUni places AT SN (earliest
//     legal point — SN 0 is the first prologue, LICM intact); SN > capUni
//     places in the epilogue Group of the contributing nest — the deps
//     that produced the uncontainable SN live in a nest this node is not
//     contained in. No special cases are needed: tile-lower ALREADY
//     encodes inside-vs-after on the edge TARGET (repointChains points
//     post-nest consumers at the nest ROOT loop because "the ForLoop node
//     as effect-def" denotes whole-loop completion), so real tile-lower
//     output lands correctly by the uniform rule alone; hand-built
//     fixtures park their after-nest consumers in the end chain the same
//     way.
//   - EFFECTFUL ACCESSES (HI-02, emergent — no separate floor pass): an
//     epoch-varying access (the fill idiom — its coordinates reference
//     the host IV) is a member whose SN already equals or exceeds its
//     loop's depth by construction, so it lands in-loop; a fully
//     invariant access references no IV, is unordered w.r.t. the loop,
//     and legally hoists (fix1.md §0: ordering is dependency-only).
//   - EPILOGUE STAMPING (ruling 6): epilogue nodes KEEP their propagated
//     SN — the DEPTH REQUIREMENT — while the epilogue Group is the landing
//     zone; SN and block are complementary, which is what final_lower's
//     reducer dual route (reducer's own SN vs its operand's SchedBlock)
//     reads.
package schedule

import (
	"errors"
	"fmt"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/tiling" // Preserved: the tiling-provenance cache entry (sibling opt package — data only, no cycle)
)

// init self-registers the scheduler in the global pass registry
// (quick-261018-l36 pattern): importing this package — directly or through
// internal/opt/all — is what makes "schedule" available to BuildPipeline.
// The ID and priority 900 are pinned by the locked decisions; Requires
// pins tile-lower whose output shape — annotated KindForLoop nests — the
// pass consumes (misconfiguration panics per the registry contract).
func init() {
	opt.Register(opt.PassSpec{
		ID:       "schedule",
		Priority: 900,
		Requires: []string{"tile-lower"},
		Factory:  func() opt.Pass { return Scheduler{} },
	})
}

// Scheduler is the sea-of-nodes → AST-style CFG + LICM pass. It is
// stateless: every Run builds a fresh scheduleRun and attaches a fresh
// ASTKernel (the graph refuses a second attach).
type Scheduler struct{}

// ID returns the pipeline name "schedule".
func (Scheduler) ID() string { return "schedule" }

// Requires pins tile-lower: the pass consumes its annotated KindForLoop
// nest shape and refuses live raw-HIR iters that only tile-lower retires.
func (Scheduler) Requires() []string { return []string{"tile-lower"} }

// Produces: builder-output stage — the pass adds no node kinds (D-04), so
// the constructed legality window is the only populated row and re-checks
// the untouched graph.
func (Scheduler) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: the tiling PROVENANCE record survives scheduling — final-lower
// fetches it for the reducer bracket loops (per-dim Between/Inside) and the
// per-LNT runtime mapping construction (P-5, quick-261022-l1o). The old
// "tiling-record" entry is deliberately dropped here: that record provably
// dies at tile-lower (tile-lower's own Preserved was empty before the
// provenance landed, so nothing ever preserved the decision record past it
// — it is keyed by BIV nodes tile-lower erases).
func (Scheduler) Preserved() []string { return []string{tiling.ProvenanceName} }

// Run schedules the sea of nodes into the AST-style CFG: the entry gates,
// then the design doc's phases composed one call per step (the
// scheduleRun object in phases.go carries the shared state). Every
// refusal is deterministic and leaves the graph's AST unattached — never
// a partial schedule.
func (Scheduler) Run(_ *opt.Context, g *ir.Graph) error {
	if err := gateGraph(g); err != nil {
		return err
	}
	run := newScheduleRun(g)
	g.ResetMarks()
	defer g.ResetMarks() // marks are Run-scoped transient state
	run.markLive()       // the live set (walk-reachable)

	if err := run.deriveForest(); err != nil { // SPEC STEP 0: loop-nest forest + shape gates
		return err
	}
	run.anchorSchedule()                               // SPEC 3.1: loop headers + SN -1 prototype anchors
	run.computeCaps()                                  // the IV-membership caps + C3 completion cuts
	run.propagate()                                    // SPEC 3.2: SN max-propagation to fixpoint
	run.classify()                                     // two-case: in-loop vs epilogue (no clamp)
	run.assignOwnership()                              // in-loop cluster ownership
	if err := run.resolveEpilogueHomes(); err != nil { // epilogue cluster homes
		return err
	}
	if err := run.buildKernel(); err != nil { // SPEC STEP 1: LBBNs + clusters
		return err
	}
	ast := run.newKernel() // SPEC STEP 2: prototype + body + metadata

	// Cross-block ordering net (locked decision 2, quick-261022-b2m): the
	// empirical guard of the placement model — every live node landed
	// (leftover, subsuming the old net) and no live node's block executes
	// before a live data/effect dependency's block. Runs post-placement,
	// pre-attach: an error leaves the AST unattached.
	if err := verifyBlockOrder(ast, run.nodes, run.live); err != nil {
		return err
	}
	return g.AttachAST(ast)
}

// gateGraph is the fail-closed entry surface: nil graph, an already
// materialized/CFG-carrying graph, an already-attached AST (one schedule
// per graph), and live raw-HIR iters/clusters (KindIter and
// KindIterCluster — ME-03 — must have been lowered by tile-lower) all
// refuse deterministically. It also performs the LO-01 stale-state reset:
// a mid-placement error on a previous Run leaves SetSched state on nodes,
// and propagation is monotone-up only, so a retried Run could inherit
// poisoned SNs — clear every allocated node's sched state up front.
func gateGraph(g *ir.Graph) error {
	if g == nil {
		return errors.New("schedule: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("schedule: graph already materialized (schedule runs on the Level-1 sea)")
	}
	if g.AST() != nil {
		return errors.New("schedule: AST CFG already attached (one schedule per graph)")
	}
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil {
			continue
		}
		n.BasePtr().SetSched(0, nil)
	}
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		if n.KindOf() == ir.KindIter || n.KindOf() == ir.KindIterCluster {
			return fmt.Errorf("schedule: live KindIter/KindIterCluster node %d — raw HIR iters must have been lowered by tile-lower",
				n.BasePtr().ID())
		}
	}
	return nil
}
