// Package schedule is the final optimization pass (quick-261021-ta3,
// doc/optana/schedule.md): it brings the Level-1 sea-of-nodes back to an
// AST-style CFG (Kernel → Group/Loop blocks) with scheduling-number LICM,
// replacing the obsolete statement-level scheduler (deleted with this
// rewrite — locked decision D-02). The emitted AST CFG lives in a SEPARATE
// field on ir.Graph (locked decision D-01) and is consumed lastly by
// final_lower and code generation (doc/optana/final_lower.md).
//
// scheduler.go is the pass proper: the fail-closed entry gates, then the
// spec's Step 0/1/2 — nest forest (nest.go), SN initialization (loops and
// their iv/bound header anchored at nest depth), max-propagation to
// fixpoint over DataDeps+EffectDeps, the effectful floor, two-case
// placement (uniform cap, epilogue class — quick-261022-b2m), per-LNT LBBN
// + prologue/epilogue Groups, per-rank stable topological placement
// (order.go), cluster concatenation, and the kernel metadata (Params/Gmem/
// LNTs). The pass inserts NO runtime-object/glue nodes (locked decision
// D-04): the prologue/epilogue Groups are the landing pads final_lower
// fills; since quick-261022-b2m the epilogues also carry the pass-placed
// after-nest consumers (the end node included).
//
// Two documented readings of the spec's numbering, both required for the
// algorithm to be correct on the real tile-lower output shape (see the
// deviations in the plan's SUMMARY):
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
//     declaration consumed at multiple depths — the buffer is an operand
//     of the shallow fill AND of deep accesses — and flooring it at its
//     deepest accessor would put it after the fill and fire the ordering
//     net; hoisting a declaration is semantics-preserving (the pin: the
//     buffer lands at SN 0).
//   - EPILOGUE STAMPING (ruling 6): epilogue nodes KEEP their propagated
//     SN — the DEPTH REQUIREMENT — while the epilogue Group is the landing
//     zone; SN and block are complementary, which is what final_lower's
//     reducer dual route (reducer's own SN vs its operand's SchedBlock)
//     reads.
package schedule

import (
	"errors"
	"fmt"
	"sort"

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
// stateless: every Run rebuilds the schedule from the graph and attaches a
// fresh ASTKernel (the graph refuses a second attach).
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

// Run gates, then schedules the sea into the AST CFG per the spec's
// Step 0/1/2. Every refusal is deterministic and leaves the graph's AST
// unattached — never a partial schedule.
func (Scheduler) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("schedule: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("schedule: graph already materialized (schedule runs on the Level-1 sea)")
	}
	if g.AST() != nil {
		return errors.New("schedule: AST CFG already attached (one schedule per graph)")
	}
	// Stale-state reset (LO-01, quick-261022-b2m): a mid-placement error on a
	// previous Run leaves SetSched state on nodes; propagation is
	// monotone-up only, so a retried Run could inherit poisoned SNs. Clear
	// every allocated node's sched state up front (the one-attach gate above
	// still prevents in-place reschedules; this covers error-path retries).
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
		// ME-03 (quick-261022-b2m): KindIterCluster is the legacy compound
		// header and must be refused exactly like KindIter — a live cluster
		// would be scheduled as an abstract HIR statement in the final CFG.
		if n.KindOf() == ir.KindIter || n.KindOf() == ir.KindIterCluster {
			return fmt.Errorf("schedule: live KindIter/KindIterCluster node %d — raw HIR iters must have been lowered by tile-lower",
				n.BasePtr().ID())
		}
	}

	// LIVE SET: mark everything reachable from the sea roots with ir.Walk
	// (the same traversal checkLegality uses); erased and unreachable
	// building debris is never scheduled.
	g.ResetMarks()
	defer g.ResetMarks()
	ir.Walk(g, ir.VisitorFunc(func(n ir.Node) ir.VisitResult {
		n.BasePtr().MarkAlive()
		return ir.Descend
	}))
	nodes := g.AllocatedNodes()
	live := make(map[ir.Node]bool, len(nodes))
	allocIndex := make(map[ir.Node]int, len(nodes))
	for i, n := range nodes {
		allocIndex[n] = i
		live[n] = n.BasePtr().Alive()
	}

	// STEP 0: the loop-nest forest. A non-erased ForLoop unreachable from
	// the sea roots is dead debris: refuse it rather than schedule garbage.
	roots := forLoopForest(g)
	nests := allNests(roots)
	for _, t := range nests {
		if !live[t.loop] {
			return fmt.Errorf("schedule: for-loop node %d is not reachable from the sea roots",
				t.loop.BasePtr().ID())
		}
	}
	sort.SliceStable(nests, func(i, j int) bool { return allocIndex[nests[i].loop] < allocIndex[nests[j].loop] })

	// Branched-nest refusal (HI-01, quick-261022-b2m): byDepth conflates
	// sibling loops at the same depth (last child stamped wins), which would
	// silently cross-product sibling bodies. Tile-lower only ever builds
	// nest CHAINS (its F3 contract), so a branch is a shape violation —
	// refuse it fail-closed, mirroring F3.
	for _, t := range nests {
		if len(t.children) > 1 {
			return fmt.Errorf("schedule: branched loop nest under for-loop %d (depth %d, %d children) — nests must be chains (tile-lower F3 contract)",
				t.loop.BasePtr().ID(), t.depth, len(t.children))
		}
	}

	// SN INITIALIZATION (spec 3.1): anchors — every ForLoop and its iv/bound
	// header at the loop's nest depth, FIXED during propagation (letting a
	// loop absorb its body's deeper rank would collapse the rank→block
	// mapping; this is the design intent of "already scheduled" control
	// nodes). A shared header node takes the MAX of its anchors'
	// depths (order-independent). Every other live node starts at SN 0.
	anchored := make(map[ir.Node]bool, len(nests)*3)
	anchor := func(n ir.Node, depth int) {
		if n == nil || !live[n] {
			return
		}
		if !anchored[n] || n.BasePtr().SchedSN() < depth {
			n.BasePtr().SetSched(depth, nil)
		}
		anchored[n] = true
	}
	for _, t := range nests {
		anchor(t.loop, t.depth)
		ops := ir.OperandsOf(t.loop)
		if len(ops) >= 2 {
			anchor(ops[0], t.depth) // iv
			anchor(ops[1], t.depth) // bound
		}
	}

	// TWO CONTAINMENT CAPS (quick-261022-b2m): capOp is the deepest loop
	// whose BODY-OPERAND subtree contains n (the HI-02 floor's reference —
	// "stays inside its operand-containing loop"); capUni is the deepest
	// loop whose body reaches n through the UNIFORM closure (data +
	// impact-predecessor edges, nest.go reachSetOf — CR-01's fix: tile-lower
	// makes fills loop members via effect chains). capUni >= capOp by
	// construction (the uniform walk includes the operand walk). Branching
	// is rejected above, so within one root the depths 1..capUni form a
	// chain and every contained node is contained at EVERY depth 1..capUni.
	capOp := make(map[ir.Node]int, len(nodes))
	capUni := make(map[ir.Node]int, len(nodes))
	for _, t := range nests {
		for n := range t.body {
			if t.depth > capOp[n] {
				capOp[n] = t.depth
			}
		}
		for n := range t.reach {
			if t.depth > capUni[n] {
				capUni[n] = t.depth
			}
		}
	}

	// SN PROPAGATION (spec 3.2): fixpoint over AllocatedNodes in allocation
	// order — SN(n) = max(SN(n), max SN over DataDeps then EffectDeps).
	// Max-over-existing-values is monotone and bounded by the deepest nest
	// depth, so it always terminates; anchored nodes never update. The
	// uniform max over DataDeps AND EffectDeps is the design intent (user
	// ruling, quick-261022-b2m): the effect chain is treated identically to
	// input edges throughout the scheduling model.
	propagate := func() {
		for changed := true; changed; {
			changed = false
			for _, n := range nodes {
				if !live[n] || anchored[n] {
					continue
				}
				sn := n.BasePtr().SchedSN()
				for _, d := range ir.DataDeps(n) {
					if live[d] && d.BasePtr().SchedSN() > sn {
						sn = d.BasePtr().SchedSN()
					}
				}
				for _, d := range ir.EffectDeps(n) {
					if live[d] && d.BasePtr().SchedSN() > sn {
						sn = d.BasePtr().SchedSN()
					}
				}
				if sn > n.BasePtr().SchedSN() {
					n.BasePtr().SetSched(sn, nil)
					changed = true
				}
			}
		}
	}
	propagate()

	// EFFECTFUL FLOOR (HI-02, quick-261022-b2m): a live memory-ACCESS node
	// (effectful, non-allocation) never hoists above its operand-containing
	// loop — floor SN at capOp. Allocations are exempt (isAllocationKind):
	// a buffer is one shared declaration consumed at several depths, and
	// hoisting a declaration is semantics-preserving. Then re-propagate:
	// floors must flow to consumers (a kernel-level consumer of a floored
	// in-loop load must become an epilogue node, not a prologue node reading
	// an in-loop value). Anchors stay fixed; the pass is monotone-up and
	// terminates.
	for _, n := range nodes {
		if !live[n] || anchored[n] || n.KindOf() == ir.KindForLoop {
			continue
		}
		if ir.EffectOf(n).Has() && !isAllocationKind(n.KindOf()) {
			if c := capOp[n]; c > n.BasePtr().SchedSN() {
				n.BasePtr().SetSched(c, nil)
			}
		}
	}
	propagate()

	// TWO-CASE CLASSIFICATION (no clamp): a live non-anchored non-ForLoop
	// node is IN-LOOP class when SN <= capUni (places AT its SN — SN 0 is
	// the first prologue, LICM intact) and EPILOGUE class when SN > capUni:
	// the deps that produced the uncontainable SN live in some nest this
	// node is not contained in, so it executes AFTER that nest (rulings 1-2,
	// CR-02's fix — the end node of every kernel, reduction shapes, and
	// post-nest stores included).
	epi := make(map[ir.Node]bool, len(nodes))
	for _, n := range nodes {
		if !live[n] || anchored[n] || n.KindOf() == ir.KindForLoop {
			continue
		}
		if n.BasePtr().SchedSN() > capUni[n] {
			epi[n] = true
		}
	}

	// UNIFORM OWNERSHIP: an in-loop node at final SN r belongs to the LNT
	// of the FIRST loop (allocation order) at depth r whose UNIFORM reach
	// contains it — first-claim wins, single placement (the old body-only
	// claim starved effect-chained members like the fills, CR-01). Map
	// order stays irrelevant: first claim per node wins.
	owner := make(map[ir.Node]*forNest, len(nodes))
	for _, t := range nests { // allocation order of the loops
		r := t.depth
		for n := range t.reach {
			if live[n] && n.KindOf() != ir.KindForLoop && !anchored[n] &&
				owner[n] == nil && n.BasePtr().SchedSN() == r {
				owner[n] = nestRoot(t)
			}
		}
	}

	// EPILOGUE HOME: an epilogue-class node lands in the epilogue Group of
	// the LAST root (forest-root cluster order) among the homes of its live
	// PosetDeps with SN >= 1 — a ForLoop dep's home is its tree's root, an
	// in-loop dep's home is its owner, an epilogue dep's home is its own
	// home. Resolution is a bounded deterministic fixpoint over passes in
	// allocation order: each pass resolves every node whose deps are all
	// resolved, so chains of epilogue consumers resolve one layer per pass
	// (bounded by len(nodes) passes as a hard terminator; until-stable is
	// the real exit). A node with no resolvable home fails closed naming
	// the ID — defensive and unreachable: SN > capUni implies some live dep
	// has SN >= 1 (propagation is a max over deps).
	rootIndexOf := make(map[*forNest]int, len(roots))
	for i, r := range roots {
		rootIndexOf[r] = i
	}
	nestOf := make(map[ir.Node]*forNest, len(nests))
	for _, t := range nests {
		nestOf[t.loop] = t
	}
	home := make(map[ir.Node]*forNest, len(epi))
	for pass := 0; pass <= len(nodes); pass++ {
		changed := false
		for _, n := range nodes { // allocation order: deterministic
			if !epi[n] || home[n] != nil {
				continue
			}
			var best *forNest
			for _, d := range ir.PosetDeps(n) {
				if d == nil || !live[d] || d.BasePtr().SchedSN() < 1 {
					continue
				}
				var dh *forNest
				switch {
				case d.KindOf() == ir.KindForLoop:
					if t, ok := nestOf[d]; ok {
						dh = nestRoot(t)
					}
				case epi[d]:
					dh = home[d] // nil on early passes: try again next pass
				default:
					dh = owner[d]
				}
				if dh == nil {
					continue
				}
				if best == nil || rootIndexOf[dh] > rootIndexOf[best] {
					best = dh
				}
			}
			if best != nil {
				home[n] = best
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	for _, n := range nodes {
		if epi[n] && home[n] == nil {
			return fmt.Errorf("schedule: node %d has no resolvable epilogue home (live deps carry no in-loop root)",
				n.BasePtr().ID())
		}
	}

	// STEP 1: per-LNT scheduling — LBBN construction, the LNT record FIRST
	// (ruling 7: the prologue/epilogue Groups point at it via Nest()), rank
	// placement for the in-loop class, then the epilogue-class collection,
	// per-rank stable topo placement with the persistent linkage stamped.
	claimed := make(map[ir.Node]bool, len(nodes))
	kernel := ir.NewASTGroup()
	var lnts []*ir.ASTLNT
	if len(roots) == 0 {
		// No ForLoop nests: the kernel body is a single Group holding every
		// live node (all at rank 0), in dependency order.
		var all []ir.Node
		for _, n := range nodes {
			if live[n] && n.KindOf() != ir.KindForLoop {
				all = append(all, n)
			}
		}
		ordered, err := topoRank(0, all, allocIndex)
		if err != nil {
			return err
		}
		for _, n := range ordered {
			kernel.AppendItem(ir.ASTItem{Node: n})
			n.BasePtr().SetSched(0, kernel)
			claimed[n] = true
		}
	} else {
		for _, r := range roots { // forest-root allocation order
			rootBlock, metas, err := buildLBBN(r)
			if err != nil {
				return err
			}
			lnt := &ir.ASTLNT{Root: r.loop, Loops: metas} // built FIRST: the Groups' Nest() target (ruling 7)
			prologue, epilogue := ir.NewASTGroupInNest(lnt), ir.NewASTGroupInNest(lnt)
			byDepth := make(map[int]*ir.ASTBlock, r.maxDepth())
			var stampDepth func(t *forNest)
			stampDepth = func(t *forNest) {
				byDepth[t.depth] = t.block
				for _, c := range t.children {
					stampDepth(c)
				}
			}
			stampDepth(r)
			for rank := 0; rank <= r.maxDepth(); rank++ {
				var collect []ir.Node
				for _, n := range nodes { // allocation order: deterministic collection
					if !live[n] || claimed[n] || n.KindOf() == ir.KindForLoop {
						continue
					}
					if epi[n] {
						continue // epilogue class: placed after the rank loop
					}
					if n.BasePtr().SchedSN() != rank {
						continue
					}
					if rank == 0 || owner[n] == r {
						collect = append(collect, n)
					}
				}
				target := prologue
				if rank > 0 {
					target = byDepth[rank] // exists: the owner loop sits at this depth
				}
				ordered, err := topoRank(rank, collect, allocIndex)
				if err != nil {
					return err
				}
				for _, n := range ordered {
					target.AppendItem(ir.ASTItem{Node: n})
					n.BasePtr().SetSched(rank, target) // persistent SN + linkage (D-03)
					claimed[n] = true
				}
			}
			// EPILOGUE CLASS (quick-261022-b2m): this root's homed after-nest
			// consumers, allocation order, topoRank'ed for intra-block order
			// (maxDepth+1 is the DIAGNOSTIC rank label only — placement keeps
			// each node's own propagated SN, ruling 6). Nodes may carry
			// DIFFERENT propagated SNs in one epilogue Group.
			var epiCollect []ir.Node
			for _, n := range nodes { // allocation order: deterministic
				if epi[n] && !claimed[n] && home[n] == r {
					epiCollect = append(epiCollect, n)
				}
			}
			epiOrdered, err := topoRank(r.maxDepth()+1, epiCollect, allocIndex)
			if err != nil {
				return err
			}
			for _, n := range epiOrdered {
				epilogue.AppendItem(ir.ASTItem{Node: n})
				n.BasePtr().SetSched(n.BasePtr().SchedSN(), epilogue) // keep the propagated SN (ruling 6)
				claimed[n] = true
			}
			kernel.AppendItem(ir.ASTItem{Block: prologue})
			kernel.AppendItem(ir.ASTItem{Block: rootBlock})
			kernel.AppendItem(ir.ASTItem{Block: epilogue})
			lnts = append(lnts, lnt)
		}
	}

	// STEP 2: kernel metadata — Params pass through; Gmem lists every live
	// GMEM allocation in allocation order with its extent (op[0]); the LNT
	// SoC metadata was assembled per nest above. Do NOT set materialized, do
	// NOT touch g.cfg (D-01).
	var gmem []ir.ASTGmemAlloc
	for _, n := range nodes {
		if live[n] && n.KindOf() == ir.KindMemNew {
			var extent ir.Node
			if ops := ir.OperandsOf(n); len(ops) > 0 {
				extent = ops[0]
			}
			gmem = append(gmem, ir.ASTGmemAlloc{New: n, Extent: extent})
		}
	}
	ast := &ir.ASTKernel{
		Body:   kernel,
		Params: g.Params(),
		Gmem:   gmem,
		LNTs:   lnts,
	}

	// Cross-block ordering net (locked decision 2, quick-261022-b2m): the
	// empirical guard of the placement model — every live node landed
	// (leftover, subsuming the old net) and no live node's block executes
	// before a live data/effect dependency's block. Runs post-placement,
	// pre-attach: an error leaves the AST unattached.
	if err := verifyBlockOrder(ast, nodes, live); err != nil {
		return err
	}
	return g.AttachAST(ast)
}

// nestRoot walks to the forest root of t's tree (the LNT identity the
// claim pass compares against, in forest-root allocation order).
func nestRoot(t *forNest) *forNest {
	for t.parent != nil {
		t = t.parent
	}
	return t
}

// isAllocationKind reports the HI-02 floor EXEMPTION (quick-261022-b2m
// ruling 5): MemNew/MemNewSMem/FragNew are effectful under the fresh-object
// rule but are shared DECLARATIONS consumed at multiple depths — the SMEM
// window buffer is an operand of the shallow per-round fill AND of the deep
// in-body accesses. Flooring an allocation at its deepest accessor would
// place it after the fill and fire the cross-block ordering net; hoisting a
// declaration is semantics-preserving (the pin: the buffer lands at SN 0,
// first prologue). Only memory-ACCESS kinds (loads/stores) take the floor.
func isAllocationKind(k ir.Kind) bool {
	switch k {
	case ir.KindMemNew, ir.KindMemNewSMem, ir.KindFragNew:
		return true
	default:
		return false
	}
}

// buildLBBN is the spec's step 1.1 for one nest: recursively create the
// Loop BB for every nested loop (mapping each LIR loop node to its Loop
// BB — the LBBN), nest child blocks as Block items of their parent, and
// stamp each loop node's persistent linkage at its own Loop block. The
// prologue/epilogue Groups around the LBBN root are created by the caller.
// Returns the root block plus the outer→inner loop metadata.
func buildLBBN(t *forNest) (*ir.ASTBlock, []ir.ASTLoopMeta, error) {
	blk, err := ir.NewASTLoop(t.loop)
	if err != nil {
		return nil, nil, err
	}
	t.block = blk
	t.loop.BasePtr().SetSched(t.depth, blk) // the loop node lives in its own Loop block
	// The anchored header (iv/bound, ops[0]/ops[1]) belongs to the Loop
	// block too — the spec's header part of the loop structure. A header
	// shared by nested loops anchors at the MAX depth (max-wins anchoring),
	// so it lands at the deepest sharing loop: only the loop whose depth
	// equals the header's anchored SN stamps it.
	if ops := ir.OperandsOf(t.loop); len(ops) >= 2 {
		for _, h := range ops[:2] {
			if h != nil && h.BasePtr().SchedSN() == t.depth && h.BasePtr().SchedBlock() == nil {
				h.BasePtr().SetSched(t.depth, blk)
			}
		}
	}
	metas := []ir.ASTLoopMeta{{
		Loop:  t.loop,
		Soc:   socOf(blk.Kind()),
		Dosoc: blk.DoSoC(),
		Loc:   blk.LoC(),
	}}
	for _, c := range t.children {
		childBlk, childMetas, cerr := buildLBBN(c)
		if cerr != nil {
			return nil, nil, cerr
		}
		blk.AppendItem(ir.ASTItem{Block: childBlk})
		metas = append(metas, childMetas...)
	}
	return blk, metas, nil
}

// socOf maps a Loop block kind back to its soc encoding (0=seq, 1=cta,
// 2=thr) for the per-LNT metadata (spec 4.3[2]). Groups do not occur here.
func socOf(k ir.BlockKind) int {
	switch k {
	case ir.BlockSeqLoop:
		return 0
	case ir.BlockCTALoop:
		return 1
	case ir.BlockThrLoop:
		return 2
	default:
		return -1
	}
}
