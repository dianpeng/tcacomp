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
// fixpoint over DataDeps+EffectDeps, containment clamping, per-LNT LBBN +
// prologue/epilogue Groups, per-rank stable topological placement
// (order.go), cluster concatenation, and the kernel metadata (Params/Gmem/
// LNTs). The pass inserts NO runtime-object/glue nodes (locked decision
// D-04): the empty prologue/epilogue Groups are the landing pads
// final_lower later fills.
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
//   - A node's final SN is clamped to its deepest ENCLOSING loop depth
//     (unenclosed nodes clamp to 0). Unclamped max-propagation would
//     misplace the cross-nest effect-chained nodes tile-lower's
//     repointChains creates (later-LNT chain heads and the end node point
//     at earlier nests' roots) at ranks their structure cannot host; the
//     clamp keeps SN and the enclosing structure consistent, which is also
//     what makes the ownership assignment provably total.
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
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		if n.KindOf() == ir.KindIter {
			return fmt.Errorf("schedule: live KindIter node %d — raw HIR iters must have been lowered by tile-lower",
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

	// Containment cap: the deepest loop whose body subtree contains n (0
	// when unenclosed). A node can never land deeper than its structure
	// allows, and an unenclosed node is a kernel-level statement (rank 0).
	cap := make(map[ir.Node]int, len(nodes))
	for _, t := range nests {
		for n := range t.body {
			if t.depth > cap[n] {
				cap[n] = t.depth
			}
		}
	}

	// SN PROPAGATION (spec 3.2): fixpoint over AllocatedNodes in allocation
	// order — SN(n) = max(SN(n), max SN over DataDeps then EffectDeps).
	// Max-over-existing-values is monotone and bounded by the deepest nest
	// depth, so it always terminates; anchored nodes never update.
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
	// Containment clamp: SN = min(SN, cap) for the non-anchored nodes (the
	// anchors are the loops' own depths and stand). See the file comment.
	for _, n := range nodes {
		if !live[n] || anchored[n] {
			continue
		}
		if c := cap[n]; n.BasePtr().SchedSN() > c {
			n.BasePtr().SetSched(c, nil)
		}
	}

	// OWNERSHIP: a rank-r (r > 0) node belongs to the LNT of the FIRST
	// loop (allocation order) at depth r whose body subtree contains it —
	// first-claim wins in allocation order, the spec's "nodes shared across
	// nests: max SN wins" with single placement, no duplication. Within
	// one nest the containing loops chain through every depth 1..cap, so
	// every placed node has exactly one owner (totality by construction).
	owner := make(map[ir.Node]*forNest, len(nodes))
	for _, t := range nests { // allocation order of the loops
		r := t.depth
		for n := range t.body { // map order irrelevant: first claim per node wins
			if live[n] && n.KindOf() != ir.KindForLoop && !anchored[n] &&
				owner[n] == nil && n.BasePtr().SchedSN() == r {
				owner[n] = nestRoot(t)
			}
		}
	}

	// STEP 1: per-LNT scheduling — LBBN construction, prologue/epilogue
	// Groups (created EMPTY: landing pads final_lower later fills, D-04),
	// per-rank stable topo placement with the persistent linkage stamped.
	claimed := make(map[ir.Node]bool, len(nodes))
	kernel := ir.NewASTGroup()
	var lnts []ir.ASTLNT
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
			prologue, epilogue := ir.NewASTGroup(), ir.NewASTGroup() // landing pads, D-04
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
			kernel.AppendItem(ir.ASTItem{Block: prologue})
			kernel.AppendItem(ir.ASTItem{Block: rootBlock})
			kernel.AppendItem(ir.ASTItem{Block: epilogue})
			lnts = append(lnts, ir.ASTLNT{Root: r.loop, Loops: metas})
		}
	}

	// Fail-closed net: every live node landed in exactly one block (loops
	// via the LBBN, statements via the rank placement). A leftover names
	// the offending IDs — never a silently partial AST.
	var leftover []uint64
	for _, n := range nodes {
		if live[n] && n.BasePtr().SchedBlock() == nil {
			leftover = append(leftover, n.BasePtr().ID())
		}
	}
	if len(leftover) > 0 {
		sort.Slice(leftover, func(i, j int) bool { return leftover[i] < leftover[j] })
		return fmt.Errorf("schedule: %d live nodes landed in no block (ids: %v)", len(leftover), leftover)
	}

	// STEP 2: kernel metadata — Params pass through; Gmem lists every live
	// GMEM allocation in allocation order with its extent (op[0]); the LNT
	// SoC metadata was assembled per nest above. Attach; do NOT set
	// materialized, do NOT touch g.cfg (D-01).
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
	return g.AttachAST(&ir.ASTKernel{
		Body:   kernel,
		Params: g.Params(),
		Gmem:   gmem,
		LNTs:   lnts,
	})
}

// nestRoot walks to the forest root of t's tree (the LNT identity the
// claim pass compares against, in forest-root allocation order).
func nestRoot(t *forNest) *forNest {
	for t.parent != nil {
		t = t.parent
	}
	return t
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
