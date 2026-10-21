package tiling

// lower_rewrite.go is the loop-tile-lower pass's PHASE 2-3 (doc/optana/
// loop_tile_lower.md §6bis): the per-LNT DoSoC numbering (§2) and the
// canonical lowered loop-forms (the loop_tiling.md "Mechanical rewrite for
// tiling" templates: 0-based round enumeration, no map index in any
// induction variable, ranks as in-body offset locals, materialized
// per-tile umin bounds, tail guarding left to the runtime loop
// materialization's whole-body no-op — the pass emits NO guard, NO branch,
// NO barrier).
//
// SSA rewrite mechanics (§3): all generated loops/dimvars/expressions are
// NEW defs; the original Iter/InductionVariable/BIV/dimvar chains are
// retired through the MutContext choke points only (RAUW, then Erase in
// dependency order — Erase refuses live users, so a surviving use fails
// the pass closed). Zero new verifier invariant classes: the extensions
// are the existing per-kind shape classes (M1 for-loop annotations, the
// intrinsic shape case) landed in PHASE 0.
//
// Effect-chain re-pointing (P6): KindForLoop is data-reachable only
// downward (loop → body), so a rewritten nest root is unreachable from the
// two-root set unless chained. Every live ImpactBy payload edge whose
// target lies inside the rewritten body is re-pointed at the nest root, so
// return.ImpactBy reaches the whole nest and the order reads "after the
// whole loop" — semantically the correct total order.

import (
	"fmt"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/ir"
)

// dimRewrite records one rewritten dim's generated nodes — the PHASE 3
// product PHASE 4 (cache insertion) consumes: the fill's round bases, the
// access coordinates, and the epilogue cross-checks all read these.
type dimRewrite struct {
	iter      ir.Node // the original iter
	biv       ir.Node // the original BIV (the record key)
	soc       SoCType // the effective space of concurrency
	loc       int     // the effective level of concurrency
	tile      int     // the effective tile size
	gridDim   int     // the GridMapping dimension, -1 = none
	threadDim int     // the ThreadMap dimension, -1 = none

	single    bool    // T==1 form: one loop, no between/inside split
	between   ir.Node // the between-round forloop (the single loop when T==1)
	inside    ir.Node // the tile-inside forloop (nil when T==1)
	betweenIV ir.Node // the round-base dimvar (nil when T==1)
	insideIV  ir.Node // the in-tile coordinate dimvar (nil when T==1)
	singleIV  ir.Node // the single loop's dimvar (nil when T>1)

	// Construction inputs buildNest consumes (bound operands are wrapped
	// in BoundedDimVar nodes at loop construction):
	singleIVNode   ir.Node   // T==1: the dimvar
	singleBound    ir.Node   // T==1: the raw extent
	singleAttrs    []ir.Node // T==1: annotations
	betweenBound   ir.Node   // T>1: the raw extent
	insideBound    ir.Node   // T>1: the materialized tile_max
	betweenAttrs   []ir.Node // T>1: between-loop annotations
	insideAttrs    []ir.Node // T>1: inside-loop annotations
	coupledThreads int       // CTA dims: the coupled walk's thread count

	indexExpr ir.Node // the dim's rewritten global index expression
	roundBase ir.Node // the fill's coordinate for this dim
}

// lntRewrite records one rewritten loop-nest tree: its dims in original
// (outer→inner) order, the generated nest root (the P6 chain anchor), the
// effectful op that became the innermost body, and the inside-set for the
// chain re-pointing.
type lntRewrite struct {
	root     *loopforest.Tree
	dims     []*dimRewrite // treeItersOf order: outer→inner
	lntRoot  ir.Node       // the outermost generated loop
	bodyRoot ir.Node       // the innermost body operand
	inside   map[*ir.Base]bool
}

// u32c allocates a u32 const node (the pass's integer vocabulary).
func (lr *lowerRun) u32c(v float64) ir.Node {
	return lr.g.NewConst(lowerOrigin, ir.InternScalar(ir.TU32), v)
}

// attr allocates one M1 KeyedAttribute annotation (u32-const value).
func (lr *lowerRun) attr(key string, v float64) ir.Node {
	return lr.g.NewKeyedAttribute(lowerOrigin, key, lr.u32c(v))
}

// rewriteLNT performs PHASE 2-3 for one forest root: numbering, per-dim
// expression generation, inside-out loop construction, body placement, IV
// RAUW, original retirement, and the P6 chain re-pointing.
//
// F3 (ruled 2026-10-21): a BRANCHED LNT — any tree node with > 1 child — is
// REJECTED fail-closed before anything is built. treeItersOf's DFS preorder
// would feed sibling iters to buildNest as one linear chain, silently
// rewriting "B1 then B2 under A" into "A{ B1{ B2{ body }}}": the body would
// execute the siblings' CROSS PRODUCT, and B2's body ops would end up
// outside the nest while their IVs were RAUW'd to loops they are not inside
// (HI-05). Per-branch sub-nests are future work.
func (lr *lowerRun) rewriteLNT(root *loopforest.Tree) error {
	iters := treeItersOf(root)
	if len(iters) == 0 {
		return nil
	}
	var branchErr error
	var walk func(t *loopforest.Tree)
	walk = func(t *loopforest.Tree) {
		if branchErr != nil {
			return
		}
		if len(t.Children) > 1 {
			branchErr = fmt.Errorf(
				"tile-lower: branched LNT not supported by the v0 rewrite: iter node %d carries %d children (sibling iters must never be silently linearized into one nested chain — their spaces would execute as a cross product, F3/HI-05)",
				t.Iter.BasePtr().ID(), len(t.Children))
			return
		}
		for _, c := range t.Children {
			walk(c)
		}
	}
	walk(root)
	if branchErr != nil {
		return branchErr
	}
	dims, err := lr.numberDims(iters)
	if err != nil {
		return err
	}
	lw := &lntRewrite{root: root, dims: dims}

	// Body BEFORE generation: the deepest effectful op owned by the LNT
	// becomes the innermost inside loop's body operand.
	body, err := lr.bodyRootOf(lw)
	if err != nil {
		return err
	}
	lw.bodyRoot = body

	if err := lr.generateDimExprs(lw); err != nil {
		return err
	}
	lw.lntRoot = lr.buildNest(lw)

	// The inside-set for P6: everything reachable from the body root over
	// data + payload edges AFTER the nest is wired (the old IV/iter chain
	// is included — edges from outside into it re-point outward below).
	lw.inside = lr.reachableFrom(body)
	if err := lr.repointChains(lw); err != nil {
		return err
	}
	if err := lr.rauwAndRetire(lw); err != nil {
		return err
	}
	lr.rewrites = append(lr.rewrites, lw)
	return nil
}

// numberDims is PHASE 2 (§2): per LNT, walk the iters outer→inner with two
// dimension counters, both restarting per LNT. A CTA dim takes the next
// GridMapping dimension AND the next ThreadMap dimension (its coupled
// inside warp — D3's counting invariant: the coupled warp is part of the
// CTA assignment, NOT double-counted as a CTA-space dim). An independent
// WARP dim takes the next ThreadMap dimension. Sequential iters get no
// dimension.
func (lr *lowerRun) numberDims(iters []ir.Node) ([]*dimRewrite, error) {
	gridC, threadC := 0, 0
	dims := make([]*dimRewrite, 0, len(iters))
	for _, it := range iters {
		d := lr.decisions[it]
		tile := lr.effectiveTile(it)
		dim := &dimRewrite{iter: it, soc: d.SoC, loc: d.LoC, tile: tile, gridDim: -1, threadDim: -1}
		biv, err := bivOf(it)
		if err != nil {
			return nil, fmt.Errorf("tile-lower: %w", err)
		}
		dim.biv = biv
		switch d.SoC {
		case SoCTA:
			dim.gridDim = gridC
			gridC++
			dim.threadDim = threadC // the coupled warp, numbered at the iter's position
			threadC++
		case SoCWARP:
			dim.threadDim = threadC
			threadC++
		case SoCSequential:
			// no dimension
		default:
			return nil, fmt.Errorf("tile-lower: iter node %d carries unknown SoC %d", it.BasePtr().ID(), int(d.SoC))
		}
		dims = append(dims, dim)
		lr.byIter[it] = dim
	}
	return dims, nil
}

// dimOfIter resolves the iter's dim InputParam: iter → BIV (operand 0) →
// dimvar (operand 0) → dim (operand 0).
func dimOfIter(iter ir.Node) (ir.Node, error) {
	biv, err := bivOf(iter)
	if err != nil {
		return nil, err
	}
	bivOps := ir.DataDeps(biv)
	if len(bivOps) < 2 || bivOps[0] == nil {
		return nil, fmt.Errorf("tile-lower: BIV node %d carries no dimvar operand", biv.BasePtr().ID())
	}
	dvOps := ir.DataDeps(bivOps[0])
	if len(dvOps) == 0 || dvOps[0] == nil {
		return nil, fmt.Errorf("tile-lower: dimvar node %d carries no dim operand", bivOps[0].BasePtr().ID())
	}
	return dvOps[0], nil
}

// extentOfBiv resolves the iter's static extent: the BIV's max operand
// (operand 1). The 0-based between loops eliminate ceil entirely — the
// bound IS the raw extent (§6bis PHASE 3 note).
//
// LO-05 (261021-o61 T3): the whole materialized-bound chain (sub, umin) is
// u32 — an extent of any other type would emit ill-typed nodes that only
// the pipeline's later typeinfer re-run catches, so a non-u32 max operand
// fails closed HERE with a named error. The type is read from the computed
// cell first (the real pipeline's TensorDimOf extents type u32 there) and
// falls back to a const's interned payload type (the fixture vocabulary).
func extentOfBiv(biv ir.Node) (ir.Node, error) {
	ops := ir.DataDeps(biv)
	if len(ops) < 2 || ops[1] == nil {
		return nil, fmt.Errorf("tile-lower: BIV node %d carries no max operand (the extent)", biv.BasePtr().ID())
	}
	extent := ops[1]
	ty := ir.TypeOf(extent)
	if ty == nil {
		ty = ir.ConstTypeOf(extent)
	}
	if ty == nil || !ty.Equal(ir.InternScalar(ir.TU32)) {
		return nil, fmt.Errorf("tile-lower: BIV node %d carries a non-u32 extent (the umin bound chain requires u32 extents, LO-05)",
			biv.BasePtr().ID())
	}
	return extent, nil
}

// coupledThreads derives the CTA dim's coupled tile-inside walk thread
// count: the remaining warp portions × quantum × 32, mirroring search.go's
// WarpPortion auto logic (the independent warp dim's portion is the pass's
// WarpPortion, or auto — AutoWarpPortion when the LNT carries a CTA dim,
// else PortionWarp; the coupled walk implicitly consumes the rest).
func (lr *lowerRun) coupledThreads(hasCTA bool) int {
	quantum := (lr.params.MaxThreadsPerCTA / 32) / PortionWarp
	warpPortion := lr.params.WarpPortion
	if warpPortion == 0 {
		if hasCTA {
			warpPortion = AutoWarpPortion
		} else {
			warpPortion = PortionWarp
		}
	}
	return (PortionWarp - warpPortion) * quantum * 32
}

// hasCTADim reports whether any dim of the LNT carries a CTA assignment.
func hasCTADim(dims []*dimRewrite) bool {
	for _, d := range dims {
		if d.soc == SoCTA {
			return true
		}
	}
	return false
}

// tileMax materializes the per-tile uniform bound (§6bis PHASE 3) in the
// SATURATING form (F1, ruled 2026-10-21):
//
//	d_tile_max = umin(T_d, extent_d − umin(d_offset, extent_d))
//
// The clamp BEFORE the subtract is load-bearing: the whole chain is u32, so
// the plain `extent − offset` WRAPS on tail rounds (an out-of-range CTA
// rank 3168 against extent 1024 computes 4294965152, the min picks T, and
// the inside loop RUNS its body out of bounds instead of no-op'ing — CR-01).
// With the clamp the arithmetic is pure u32, wraps nowhere, and
// d_tile_max == 0 exactly when the rank is out of range — bounds ONLY; tail
// guarding stays the runtime loop primitive's whole-body no-op (§4), never
// an IR branch.
func (lr *lowerRun) tileMax(tile int, extent, offset ir.Node) ir.Node {
	clamped := lr.g.NewCall(lowerOrigin, "umin", offset, extent)
	rem := lr.g.NewBinOp(lowerOrigin, ir.BinOpSub, extent, clamped)
	return lr.g.NewCall(lowerOrigin, "umin", lr.u32c(float64(tile)), rem)
}

// generateDimExprs builds every dim's dimvars, bounds, ranks, offsets, and
// index expressions (PHASE 3 per-dim products) — but NOT the loops: the
// nest is constructed inside-out by buildNest, where each loop's body
// operand is wired at construction (an unwired body slot would trip the
// verifier's nil-def check, and SetDef refuses a zero handle).
func (lr *lowerRun) generateDimExprs(lw *lntRewrite) error {
	cta := hasCTADim(lw.dims)
	for _, d := range lw.dims {
		dim, err := dimOfIter(d.iter)
		if err != nil {
			return err
		}
		extent, err := extentOfBiv(d.biv)
		if err != nil {
			return err
		}
		switch {
		case d.tile == 1:
			// T==1: NO tiling for the dim — a single annotated loop; the
			// original IV uses map directly to the loop index expression.
			d.single = true
			switch d.soc {
			case SoCSequential:
				d.singleIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(1))
				d.singleIVNode, d.singleBound = d.singleIV, extent
				d.singleAttrs = []ir.Node{lr.attr("soc", 0), lr.attr("loc", 1)}
				d.indexExpr, d.roundBase = d.singleIV, d.singleIV
			case SoCTA:
				// Single CTA loop at stride LoC: index = iv + grid_rank.
				d.singleIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(float64(d.loc)))
				d.singleIVNode, d.singleBound = d.singleIV, extent
				d.singleAttrs = []ir.Node{lr.attr("soc", 1), lr.attr("dosoc", float64(d.gridDim)), lr.attr("loc", float64(d.loc))}
				rank := lr.g.NewIntrinsicGridMap(lowerOrigin, "index", lr.u32c(float64(d.gridDim)))
				d.indexExpr = lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd, d.singleIV, rank)
				d.roundBase = d.indexExpr
			case SoCWARP:
				// Single thread-parallel loop at stride LoC×32.
				d.singleIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(float64(d.loc*32)))
				d.singleIVNode, d.singleBound = d.singleIV, extent
				d.singleAttrs = []ir.Node{lr.attr("soc", 2), lr.attr("dosoc", float64(d.threadDim)), lr.attr("loc", float64(d.loc*32))}
				rank := lr.g.NewIntrinsicThreadMap(lowerOrigin, "index", lr.u32c(float64(d.threadDim)))
				d.indexExpr = lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd, d.singleIV, rank)
				d.roundBase = d.indexExpr
			}
			continue
		}

		// T > 1: between-round + tile-inside loops.
		t := float64(d.tile)
		switch d.soc {
		case SoCSequential:
			// between for (a = 0; a < A; a += T) — element stride, LoC = 1;
			// inside sequential, bounded by the materialized tile_max.
			d.betweenIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(t))
			d.betweenBound = extent
			d.insideBound = lr.tileMax(d.tile, extent, d.betweenIV)
			d.insideIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(1))
			d.betweenAttrs = []ir.Node{lr.attr("soc", 0), lr.attr("loc", 1)}
			d.insideAttrs = []ir.Node{lr.attr("soc", 0), lr.attr("loc", 1)}
			d.indexExpr = lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd, d.betweenIV, d.insideIV)
			d.roundBase = d.betweenIV
		case SoCTA:
			// between for (a = 0; a < A; a += LoC*T), LoC = MaxCTA; in-body
			// cta_rank = GridMap(dosoc,"index"); offset = rank*T + round;
			// inside = the COUPLED warp walk at ELEMENT stride.
			d.betweenIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(float64(d.loc*d.tile)))
			d.betweenBound = extent
			rank := lr.g.NewIntrinsicGridMap(lowerOrigin, "index", lr.u32c(float64(d.gridDim)))
			offset := lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd,
				lr.g.NewBinOp(lowerOrigin, ir.BinOpMul, rank, lr.u32c(t)), d.betweenIV)
			d.insideBound = lr.tileMax(d.tile, extent, offset)
			coupled := lr.coupledThreads(cta)
			if coupled < 1 {
				// F2 defensive floor (the decide pass's Validate/evalSchedule
				// reject the configuration first; this is the rewrite-side
				// backstop): a zero-portion coupled walk emits a zero-stride
				// inside ForLoop — an infinite loop.
				return fmt.Errorf("tile-lower: CTA dim (iter node %d) coupled inside walk computes %d threads — the coupled-portion floor (F2) demands >= 1",
					d.iter.BasePtr().ID(), coupled)
			}
			d.coupledThreads = coupled
			d.insideIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(float64(coupled)))
			d.betweenAttrs = []ir.Node{lr.attr("soc", 1), lr.attr("dosoc", float64(d.gridDim)), lr.attr("loc", float64(d.loc))}
			d.insideAttrs = []ir.Node{lr.attr("soc", 2), lr.attr("dosoc", float64(d.threadDim)), lr.attr("loc", float64(coupled))}
			threadRank := lr.g.NewIntrinsicThreadMap(lowerOrigin, "index", lr.u32c(float64(d.threadDim)))
			d.indexExpr = lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd,
				lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd, offset, d.insideIV), threadRank)
			d.roundBase = offset
		case SoCWARP:
			// between for (b = 0; b < B; b += nThreads*T) — each thread owns
			// a whole T-chunk (the composition-semantics ruling); in-body
			// thread_rank = ThreadMap(dosoc,"index"); offset = rank*T + round;
			// inside = SEQUENTIAL over the owned chunk.
			stride := d.loc * 32 * d.tile
			d.betweenIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(float64(stride)))
			d.betweenBound = extent
			rank := lr.g.NewIntrinsicThreadMap(lowerOrigin, "index", lr.u32c(float64(d.threadDim)))
			offset := lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd,
				lr.g.NewBinOp(lowerOrigin, ir.BinOpMul, rank, lr.u32c(t)), d.betweenIV)
			d.insideBound = lr.tileMax(d.tile, extent, offset)
			d.insideIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(1))
			d.betweenAttrs = []ir.Node{lr.attr("soc", 2), lr.attr("dosoc", float64(d.threadDim)), lr.attr("loc", float64(d.loc*32))}
			d.insideAttrs = []ir.Node{lr.attr("soc", 0), lr.attr("loc", 1)}
			d.indexExpr = lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd, offset, d.insideIV)
			d.roundBase = offset
		}
	}
	return nil
}

// buildNest constructs the generated loops and wires the canonical nest
// (§6bis PHASE 3): ALL between/single loops nest in dim order, ALL inside
// loops nest inside them, also in dim order. Construction is inside-out so
// every body operand is wired at construction time; the loops of each
// dim are stored on the dimRewrite for PHASE 4 and the epilogue. Returns
// the nest root (the outermost loop — the P6 chain anchor).
func (lr *lowerRun) buildNest(lw *lntRewrite) ir.Node {
	inner := lw.bodyRoot
	// Inside loops, innermost first (reverse dim order).
	for i := len(lw.dims) - 1; i >= 0; i-- {
		d := lw.dims[i]
		if d.single {
			continue
		}
		d.inside = lr.g.NewForLoop(lowerOrigin, d.insideIV,
			lr.g.NewBoundedDimVar(lowerOrigin, d.insideIV, d.insideBound), inner, d.insideAttrs...)
		inner = d.inside
	}
	// Between/single loops, innermost first.
	for i := len(lw.dims) - 1; i >= 0; i-- {
		d := lw.dims[i]
		if d.single {
			d.between = lr.g.NewForLoop(lowerOrigin, d.singleIV,
				lr.g.NewBoundedDimVar(lowerOrigin, d.singleIV, d.singleBound), inner, d.singleAttrs...)
		} else {
			d.between = lr.g.NewForLoop(lowerOrigin, d.betweenIV,
				lr.g.NewBoundedDimVar(lowerOrigin, d.betweenIV, d.betweenBound), inner, d.betweenAttrs...)
		}
		inner = d.between
	}
	return inner
}

// bodyRootOf picks the LNT's innermost effectful op: every live GMEM
// load/store referencing the LNT's iters (via its axis chase), owned by
// the DEEPEST referenced level; ties break to the latest allocation (the
// chain-last position). An LNT with no effectful body fails closed — the
// rewritten nest needs a body operand.
func (lr *lowerRun) bodyRootOf(lw *lntRewrite) (ir.Node, error) {
	member := make(map[ir.Node]bool, len(lw.dims))
	depth := make(map[ir.Node]int, len(lw.dims))
	var walk func(t *loopforest.Tree)
	walk = func(t *loopforest.Tree) {
		depth[t.Iter] = t.Depth
		for _, c := range t.Children {
			walk(c)
		}
	}
	walk(lw.root)
	for _, d := range lw.dims {
		member[d.iter] = true
	}
	var best ir.Node
	bestDepth := -1
	for _, n := range lr.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		k := n.KindOf()
		if k != ir.KindMemLoad && k != ir.KindMemStore {
			continue
		}
		ops := ir.OperandsOf(n)
		axesFrom := 1 // loads: axes after the tensor
		if k == ir.KindMemStore {
			axesFrom = 2 // stores: value, tensor, then axes
		}
		d := -1
		for _, ax := range ops[axesFrom:] {
			for _, it := range findAxisIters(ax) {
				if !member[it] {
					continue
				}
				if depth[it] > d {
					d = depth[it]
				}
			}
		}
		if d < 0 {
			continue // references no iter of this LNT
		}
		if lr.claimed[n.BasePtr()] {
			return nil, fmt.Errorf("tile-lower: effectful node %d is owned by two LNTs (cross-tree body)", n.BasePtr().ID())
		}
		if d > bestDepth || (d == bestDepth && best != nil && n.BasePtr().ID() > best.BasePtr().ID()) {
			best, bestDepth = n, d
		}
	}
	if best == nil {
		return nil, fmt.Errorf("tile-lower: LNT rooted at iter node %d owns no effectful body op", lw.root.Iter.BasePtr().ID())
	}
	lr.claimed[best.BasePtr()] = true
	return best, nil
}

// reachableFrom returns the set of nodes reachable from start over data
// dependencies AND ImpactBy payload edges (the walk's child relation), in
// the sea of nodes.
func (lr *lowerRun) reachableFrom(start ir.Node) map[*ir.Base]bool {
	seen := make(map[*ir.Base]bool)
	var visit func(n ir.Node)
	visit = func(n ir.Node) {
		if n == nil || n.BasePtr() == nil || seen[n.BasePtr()] || n.BasePtr().Erased() {
			return
		}
		seen[n.BasePtr()] = true
		for _, dep := range ir.DataDeps(n) {
			visit(dep)
		}
		for _, u := range ir.ImpactBy(n) {
			visit(u.Def())
		}
	}
	visit(start)
	return seen
}

// repointChains is P6: every live ImpactBy payload edge (any bearer's
// effectIns record) whose target lies inside the LNT's rewritten body is
// re-pointed at the nest root — the rewritten loops become reachable from
// the two-root set, ordered "after the whole loop".
func (lr *lowerRun) repointChains(lw *lntRewrite) error {
	for _, n := range lr.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		ins := ir.ImpactBy(n)
		if len(ins) == 0 {
			continue
		}
		if lw.inside[n.BasePtr()] {
			continue // edges between body nodes keep their local order
		}
		for _, u := range ins {
			def := u.Def()
			if def == nil || !lw.inside[def.BasePtr()] {
				continue
			}
			if err := lr.mctx.SetDef(u, lw.lntRoot); err != nil {
				return fmt.Errorf("tile-lower: re-pointing an effect edge at nest root: %w", err)
			}
		}
	}
	return nil
}

// rauwAndRetire patches every original InductionVariable use to the dim's
// (offset + inside) index expression, then retires the original Iter
// chains through the choke points in dependency order: induction variables
// first (their uses were RAUW'd away), then iters inner→outer (a child
// iter is an operand of its parent), then the BIVs, then the old dimvars.
// Erase refuses any surviving live use, so corruption fails the pass.
func (lr *lowerRun) rauwAndRetire(lw *lntRewrite) error {
	// RAUW: every IV of every dim moves onto the dim's index expression.
	for _, d := range lw.dims {
		for _, iv := range ivsOf(d.iter) {
			if err := lr.mctx.ReplaceAllUsesWith(iv, d.indexExpr); err != nil {
				return fmt.Errorf("tile-lower: RAUW of induction variable node %d: %w", iv.BasePtr().ID(), err)
			}
		}
	}
	// Retire, innermost first.
	for i := len(lw.dims) - 1; i >= 0; i-- {
		d := lw.dims[i]
		for _, iv := range ivsOf(d.iter) {
			if err := lr.mctx.Erase(iv); err != nil {
				return fmt.Errorf("tile-lower: retiring induction variable node %d: %w", iv.BasePtr().ID(), err)
			}
		}
		if err := lr.mctx.Erase(d.iter); err != nil {
			return fmt.Errorf("tile-lower: retiring iter node %d: %w", d.iter.BasePtr().ID(), err)
		}
	}
	// Capture the old dimvars BEFORE erasing the BIVs — Erase kills the
	// operand records, so the dimvar resolution must happen first.
	oldDimVars := make([]ir.Node, 0, len(lw.dims))
	for _, d := range lw.dims {
		dv := ir.DataDeps(d.biv)
		if len(dv) == 0 || dv[0] == nil {
			return fmt.Errorf("tile-lower: BIV node %d carries no dimvar operand", d.biv.BasePtr().ID())
		}
		oldDimVars = append(oldDimVars, dv[0])
	}
	for _, d := range lw.dims {
		if err := lr.mctx.Erase(d.biv); err != nil {
			return fmt.Errorf("tile-lower: retiring BIV node %d: %w", d.biv.BasePtr().ID(), err)
		}
	}
	for _, dv := range oldDimVars {
		if err := lr.mctx.Erase(dv); err != nil {
			return fmt.Errorf("tile-lower: retiring dimvar node %d: %w", dv.BasePtr().ID(), err)
		}
	}
	return nil
}

// ivsOf collects the live InductionVariable users of an iter.
func ivsOf(iter ir.Node) []ir.Node {
	var out []ir.Node
	for _, u := range ir.UsesOf(iter) {
		if user := u.User(); user != nil && user.KindOf() == ir.KindInductionVariable && !user.BasePtr().Erased() {
			out = append(out, user)
		}
	}
	return out
}
