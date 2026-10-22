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
//     analysis + the pass-local body/reach placement derivations.
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
// PLACEMENT MODEL (quick-261022-b2m, replacing the old containment clamp
// that review 261021-ta3 proved unsound — CR-01/CR-02): the clamp
// "SN = min(propagated, deepest-enclosing-depth)" hoisted per-epoch SMEM
// fills into the once-only prologue (fills are loop members via EFFECT
// chains, which the operand-only containment ignored) and scheduled every
// unenclosed consumer — reduction shapes, post-nest stores, every kernel's
// end node — BEFORE its in-loop dependencies. The model is now:
//
//   - UNIFORM RELATION (user ruling): ImpactBy is treated IDENTICALLY to
//     input edges throughout the scheduling model — propagation already
//     maxes over DataDeps+EffectDeps, and the containment cap AND cluster
//     ownership now use the same uniform closure (nest.go reachSetOf, the
//     exact relation tiling's reachableFrom uses, lower_rewrite.go).
//   - TWO-CASE PLACEMENT, NO CLAMP: SN <= capUni places AT SN (earliest
//     legal point — SN 0 is the first prologue, LICM intact); SN > capUni
//     places in the epilogue Group of the contributing nest — the deps
//     that produced the uncontainable SN live in a nest this node is not
//     contained in. No special cases are needed: tile-lower ALREADY
//     encodes inside-vs-after on the edge TARGET (repointChains points
//     post-nest consumers at the nest ROOT loop because "the ForLoop node
//     as effect-def" denotes whole-loop completion), so real tile-lower
//     output lands correctly by the uniform rule alone; hand-built
//     fixtures with un-repointed in-body effect-defs (the review's hazard
//     probes) land as epilogue nodes the same way.
//   - EFFECTFUL FLOOR (HI-02): after the first fixpoint, a live
//     memory-ACCESS node (EffectOf full, non-allocation) is floored at its
//     OPERAND containment depth (capOp — data-invariant accesses stay in
//     their operand-containing loop; pure invariants still hoist), then
//     propagation re-runs so floors flow to consumers. ALLOCATIONS
//     (MemNew/MemNewSMem/FragNew) are exempt: a buffer is a shared
//     declaration consumed at multiple depths, and hoisting a declaration
//     is semantics-preserving (the pin: the buffer lands at SN 0).
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

// Preserved: the tiling-record analysis cache entry survives scheduling —
// final_lower fetches it for per-LNT runtime mapping construction.
func (Scheduler) Preserved() []string { return []string{"tiling-record"} }

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
	run.anchorSchedule() // SPEC 3.1: loop headers + SN -1 prototype anchors
	run.computeCaps()    // the operand/uniform containment caps
	run.propagate()      // SPEC 3.2: SN max-propagation to fixpoint
	run.applyEffectfulFloor()
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
