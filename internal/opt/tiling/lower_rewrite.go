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
	dim       ir.Node // the dim InputParam (captured pre-erasure — the provenance key, P-5)
	soc       SoCType // the effective space of concurrency
	loc       int     // the effective level of concurrency
	tile      int     // the effective tile size
	gridDim   int     // the GridMapping dimension, -1 = none
	threadDim int     // the ThreadMap dimension, -1 = none

	between   ir.Node // the between-round forloop
	inside    ir.Node // the tile-inside forloop
	betweenIV ir.Node // the round-base dimvar
	insideIV  ir.Node // the in-tile coordinate dimvar

	// Construction inputs buildNest consumes (bound operands are wrapped
	// in BoundedDimVar nodes at loop construction):
	betweenBound   ir.Node   // the raw extent
	insideBound    ir.Node   // the materialized tile_max
	betweenAttrs   []ir.Node // between-loop annotations
	insideAttrs    []ir.Node // inside-loop annotations
	coupledThreads int       // CTA dims: the coupled walk's thread count

	indexExpr ir.Node // the dim's rewritten global index expression
	roundBase ir.Node // the fill's coordinate for this dim
}

// lntRewrite records one rewritten loop-nest tree: its dims in original
// (outer→inner) order, the generated nest root (the P6 chain anchor), and
// the inside-set for the chain re-pointing.
//
// Sea-of-nodes (fix1.md §0): no body operand, no bodyRoot. The inside-set
// is derived from IV data-dep closure membership (D-g), not from a body
// reachability walk.
type lntRewrite struct {
	root    *loopforest.Tree
	dims    []*dimRewrite // treeItersOf order: outer→inner
	lntRoot ir.Node       // the outermost generated loop
	inside  map[*ir.Base]bool
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
// expression generation, inside-out loop construction, IV RAUW, original
// retirement, and the P6 chain re-pointing.
//
// F3 (ruled 2026-10-21): a BRANCHED LNT — any tree node with > 1 child — is
// REJECTED fail-closed before anything is built. treeItersOf's DFS preorder
// would feed sibling iters to buildNest as one linear chain, silently
// rewriting "B1 then B2 under A" into "A{ B1{ B2{ body }}}": the body would
// execute the siblings' CROSS PRODUCT, and B2's body ops would end up
// outside the nest while their IVs were RAUW'd to loops they are not inside
// (HI-05). Per-branch sub-nests are future work.
//
// Sea-of-nodes (fix1.md §0, D-g): there is no body operand and no
// bodyRootOf. The P6 inside-set is derived from IV data-dep closure
// membership: a node is inside the LNT iff its data-dep closure references
// any of the LNT's generated loops' IVs — the same relation loopforest
// uses (D-b).
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
	// fix4 R-SEAM fail-closed guard: the decide side now ADMITS multi-dim
	// parallel schedules (MAPM allows up to 3 CTA dims and 3 expanded THR
	// instances per LNT), but the rewrite templates below still implement
	// the single-CTA-dim / single-WARP-dim forms — the multi-dim loop
	// templates land in fix4 T4 (§3.1). A record carrying more than one
	// CTA dim or more than one WARP dim stops HERE, loudly, never as an
	// undesigned lowering.
	ctaDims, warpDims := 0, 0
	for _, it := range iters {
		switch lr.decisions[it].SoC {
		case SoCTA:
			ctaDims++
		case SoCWARP:
			warpDims++
		}
	}
	if ctaDims > 1 || warpDims > 1 {
		return fmt.Errorf(
			"tile-lower: LNT rooted at iter node %d carries %d CTA dims and %d WARP dims — multi-dim parallel schedule: template rewrite lands in fix4 T4 (the single-dim templates below must never linearize it)",
			root.Iter.BasePtr().ID(), ctaDims, warpDims)
	}
	dims, err := lr.numberDims(iters)
	if err != nil {
		return err
	}
	lw := &lntRewrite{root: root, dims: dims}

	if err := lr.generateDimExprs(lw); err != nil {
		return err
	}
	lw.lntRoot = lr.buildNest(lw)

	// The inside-set for P6 (D-g): every node whose data-dep closure
	// references any of this LNT's loops' IVs — IV-membership, the same
	// relation loopforest uses. Effect/chain edges are NOT linking input
	// (C3); the old body-reachability derivation is deleted with bodyRootOf.
	lw.inside = lr.ivMembers(lw)
	if err := lr.repointChains(lw); err != nil {
		return err
	}
	if err := lr.rauwAndRetire(lw); err != nil {
		return err
	}
	lr.rewrites = append(lr.rewrites, lw)
	return lr.recordProvenance(lw)
}

// recordProvenance publishes this LNT's P-5 provenance entries
// (quick-261022-l1o, fix4 §3.2): one per-dim DimProvenance keyed by the
// DIM InputParam (captured on the dimRewrite at expression-generation
// time — the iter chains are ERASED by the time this runs, so the dim
// cannot be re-resolved here) and one per-LNT MappingPlan keyed by the
// generated nest root. The plan's vectors are the ABSOLUTE
// factorization shapes derived from the record's own decisions — no
// share arithmetic: GridPortions = the per-CTA-dim absolute LoC_d in
// DoSoC order, ThreadPortions = the per-WARP-dim warp counts K_d in
// DoSoC order. The launch thread budget is the derived blockDim = (Π
// warp K) × 32, carried as a MAX over LNTs (final-lower's reducer
// staging extent reads #warps from it).
func (lr *lowerRun) recordProvenance(lw *lntRewrite) error {
	for _, d := range lw.dims {
		if d.dim == nil {
			return fmt.Errorf("tile-lower: dim of iter node %d was never captured (generateDimExprs corruption?)",
				d.iter.BasePtr().ID())
		}
		lr.prov.Dims[d.dim] = DimProvenance{
			Between: d.between,
			Inside:  d.inside,
			Soc:     d.soc,
			Tile:    d.tile,
		}
		// The CTA dim's coupled inside-walk facts (the flat-index ruling's
		// materialization): the ThreadMap slot the coupled walk rides and
		// its thread count — the LNT's blockDim under the fix4 model.
		// Final-lower's reducer redirect and any future ownership-gate
		// walk read these instead of re-deriving launch arithmetic (D3's
		// counting invariant lives here, at the pass that built the walk).
		if d.soc == SoCTA {
			lr.prov.Dims[d.dim] = DimProvenance{
				Between:        d.between,
				Inside:         d.inside,
				Soc:            d.soc,
				Tile:           d.tile,
				CoupledDoSoC:   d.threadDim,
				CoupledThreads: d.coupledThreads,
			}
		}
	}
	grid := make([]int, 0, 1)
	thread := make([]int, 0, 1)
	warpProduct := 1
	for _, d := range lw.dims {
		switch d.soc {
		case SoCTA:
			grid = append(grid, d.loc)
		case SoCWARP:
			thread = append(thread, d.loc)
			warpProduct *= d.loc
		}
	}
	lr.prov.LNTs[lw.lntRoot] = MappingPlan{
		GridPortions:   grid,
		ThreadPortions: thread,
	}
	if blockDim := warpProduct * 32; blockDim > lr.prov.MaxThreadsPerCTA {
		lr.prov.MaxThreadsPerCTA = blockDim
	}
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

// coupledThreads is DELETED with the share grammar (fix4 §1.2): the CTA
// dim's coupled inside walk strides over the LNT's WHOLE blockDim =
// (Π warp K) × 32 — see blockDimOf in judge.go and the per-LNT
// derivation in generateDimExprs/recordProvenance. hasCTADim survives for
// the MIRROR-GATE slot resolution.

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
	rem := lr.saturatingRemainder(extent, offset)
	return lr.g.NewCall(lowerOrigin, "umin", lr.u32c(float64(tile)), rem)
}

// saturatingRemainder materializes the F1 core extent − umin(offset,
// extent): the exact number of in-bounds rounds/elements remaining at
// offset, wrapping nowhere in u32. Its only caller is tileMax — every
// materialized per-tile bound rides it.
func (lr *lowerRun) saturatingRemainder(extent, offset ir.Node) ir.Node {
	clamped := lr.g.NewCall(lowerOrigin, "umin", offset, extent)
	return lr.g.NewBinOp(lowerOrigin, ir.BinOpSub, extent, clamped)
}

// generateDimExprs builds every dim's dimvars, bounds, ranks, offsets, and
// index expressions (PHASE 3 per-dim products) — but NOT the loops: the
// nest is constructed inside-out by buildNest, where each loop's body
// operand is wired at construction (an unwired body slot would trip the
// verifier's nil-def check, and SetDef refuses a zero handle).
func (lr *lowerRun) generateDimExprs(lw *lntRewrite) error {
	cta := hasCTADim(lw.dims)
	// The CTA dim's ThreadMap slot (the coupled walk's group id dim) — the
	// MIRROR-GATE term's DoSoC (2026-10-23 reducer redirect): an independent
	// WARP dim's strip offset gains index(ctaThreadDim)*extent so the upper
	// group mirrors (the threads whose coupled-group coordinate is nonzero)
	// saturate their tile max to zero trips — they hold the identity init
	// under a uniform reduction bracket instead of double-counting. Pure
	// offset arithmetic over the F1 tail-clamp machinery; no new query, no
	// branch. At most one CTA dim reaches the templates (the fix4 T4
	// fail-closed guard in rewriteLNT pins the multi-dim forms to T4).
	ctaThreadDim := -1
	if cta {
		for _, d := range lw.dims {
			if d.soc == SoCTA {
				ctaThreadDim = d.threadDim
				break
			}
		}
	}
	for _, d := range lw.dims {
		dim, err := dimOfIter(d.iter)
		if err != nil {
			return err
		}
		d.dim = dim // captured pre-erasure: the P-5 provenance key
		extent, err := extentOfBiv(d.biv)
		if err != nil {
			return err
		}
		// GATE 3 (the 2026-10-24 tile-disable ruling): the fail-closed
		// backstop over the EFFECTIVE tile (the record decision OR a budget
		// override). Params.Validate (gate 1) rejects TileSizes entries < 2
		// and tiling-verify (gate 2) rejects record TileSize < 2, but a
		// HAND-STAMPED Verified record or a bad override reaching the
		// rewrite dies HERE — never as a non-tiling (semantically
		// divergent) rewrite. The single-loop form is deleted; every dim
		// takes the mechanical between/inside path below.
		if d.tile < 2 {
			return fmt.Errorf("tile-lower: iter node %d carries effective tile size %d (record decision or budget override) — tile size 1 is not a legal schedule: the compiler never generates a non-tiling schedule (2026-10-24 ruling; gate 3 backstop — Params.Validate and tiling-verify are gates 1-2)",
				d.iter.BasePtr().ID(), d.tile)
		}

		// Between-round + tile-inside loops — the mechanical form, every dim.
		t := float64(d.tile)
		switch d.soc {
		case SoCSequential:
			// between for (a = 0; a < A; a += T) — element stride, LoC = 1;
			// inside sequential, bounded by the materialized tile_max.
			d.betweenIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(t))
			d.betweenBound = extent
			d.insideBound = lr.tileMax(d.tile, extent, d.betweenIV)
			d.insideIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(1))
			d.betweenAttrs = []ir.Node{lr.attr("soc", 0), lr.attr("loc", 1), lr.attr("tile", float64(d.tile))}
			d.insideAttrs = []ir.Node{lr.attr("soc", 0), lr.attr("loc", 1), lr.attr("tile", float64(d.tile))}
			d.indexExpr = lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd, d.betweenIV, d.insideIV)
			d.roundBase = d.betweenIV
		case SoCTA:
			// between for (a = 0; a < A; a += LoC*T); in-body
			// cta_rank = GridMap(dosoc,"index"); offset = rank*T + round.
			// Inside = the COUPLED walk at ELEMENT stride — the
			// FLAT-INDEX FORM (ruling 2026-10-23): the IV's INIT is the
			// no-arg flat thread query (the thread's absolute 1-D rank),
			// step = the LNT's blockDim, bound = the materialized tile
			// max; indexExpr = offset + insideIV (Index() = flatRank +
			// i*blockDim) — the separate +index(threadDim) tuple-coordinate
			// add is DELETED (it supplies only the slowest-axis coordinate,
			// never a 0..blockDim-1 element rank — finding C1). Under the
			// fix4 factorization model the coupled walk is single-owner
			// exact for ANY tile_max (init = flat rank, step = blockDim:
			// each element has exactly one owning rank), so the old
			// tile-vs-coupled bound gate is DELETED (fix4 R-HI03) — the
			// §3.3 pin itself requires T > blockDim.
			d.betweenIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(float64(d.loc*d.tile)))
			d.betweenBound = extent
			rank := lr.g.NewRuntimeGridQuery(lowerOrigin, "index", lr.u32c(float64(d.gridDim)))
			offset := lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd,
				lr.g.NewBinOp(lowerOrigin, ir.BinOpMul, rank, lr.u32c(t)), d.betweenIV)
			d.insideBound = lr.tileMax(d.tile, extent, offset)
			blockDim := blockDimOf(lr.decisions)
			d.coupledThreads = blockDim
			d.insideIV = lr.g.NewDimVar(lowerOrigin, dim,
				lr.g.NewRuntimeThreadQueryFlat(lowerOrigin), lr.u32c(float64(blockDim)))
			d.betweenAttrs = []ir.Node{lr.attr("soc", 1), lr.attr("dosoc", float64(d.gridDim)), lr.attr("loc", float64(d.loc)), lr.attr("tile", float64(d.tile))}
			d.insideAttrs = []ir.Node{lr.attr("soc", 2), lr.attr("dosoc", float64(d.threadDim)), lr.attr("loc", float64(blockDim)), lr.attr("tile", float64(d.tile))}
			d.indexExpr = lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd, offset, d.insideIV)
			d.roundBase = offset
		case SoCWARP:
			// between for (b = 0; b < B; b += nThreads*T) — each thread owns
			// a whole T-chunk (the composition-semantics ruling); in-body
			// thread_rank = ThreadMap(dosoc,"index"); offset = rank*T + round;
			// inside = SEQUENTIAL over the owned chunk.
			//
			// MIRROR GATE (2026-10-23 reducer redirect): the offset gains
			// index(ctaThreadDim)*extent — the threads of the CTA dim's
			// coupled group beyond the first (tid vs tid+groupExtent mirrors)
			// saturate their tile max to ZERO trips, so a uniform reduction
			// bracket never double-counts their (identical) strip partials.
			// Pure offset arithmetic through the F1 saturating bound — no new
			// query kind, no branch; the lower group's threads are unchanged.
			stride := d.loc * 32 * d.tile
			d.betweenIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(float64(stride)))
			d.betweenBound = extent
			rank := lr.g.NewRuntimeThreadQuery(lowerOrigin, "index", lr.u32c(float64(d.threadDim)))
			offset := lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd,
				lr.g.NewBinOp(lowerOrigin, ir.BinOpMul, rank, lr.u32c(t)), d.betweenIV)
			if ctaThreadDim >= 0 {
				gate := lr.g.NewRuntimeThreadQuery(lowerOrigin, "index", lr.u32c(float64(ctaThreadDim)))
				offset = lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd, offset,
					lr.g.NewBinOp(lowerOrigin, ir.BinOpMul, gate, extent))
			}
			d.insideBound = lr.tileMax(d.tile, extent, offset)
			d.insideIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(1))
			d.betweenAttrs = []ir.Node{lr.attr("soc", 2), lr.attr("dosoc", float64(d.threadDim)), lr.attr("loc", float64(d.loc*32)), lr.attr("tile", float64(d.tile))}
			d.insideAttrs = []ir.Node{lr.attr("soc", 0), lr.attr("loc", 1), lr.attr("tile", float64(d.tile))}
			d.indexExpr = lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd, offset, d.insideIV)
			d.roundBase = offset
		}
	}
	return nil
}

// buildNest constructs the generated loops and wires the canonical nest
// (§6bis PHASE 3): ALL between loops nest in dim order, ALL inside
// loops nest inside them, also in dim order — the chain
//
//	dims[0].between → … → dims[N-1].between → dims[0].inside → … → dims[N-1].inside
//
// Construction is inside-out (the pass has consumed dims inner-first since
// PHASE 2); the loops of each dim are stored on the dimRewrite for PHASE 4
// and the epilogue. Returns the nest root (the outermost loop — the P6 chain
// anchor).
//
// Sea-of-nodes (fix1.md §0, D-b, D-g): ForLoop has no body operand, so the
// canonical chain is carried by an explicit LOOP-TO-LOOP annotation — the
// "nest" KeyedAttribute on every non-root loop, whose value is the loop's
// canonical parent. §0 permits explicit loop-to-loop structure ("HIR iters
// already carry a parent-iter operand — loop-to-loop, permitted"); the attr
// keeps D-a's [iv, bound, attrs...] layout, never touches codegen (emit.go
// treats KeyedAttribute as a terminal), and lands the parent loop's IV in
// the child's operand closure — the exact membership edge loopforest's D-b
// derivation consumes. Because construction allocates inner loops before
// their enclosing loops, D-b's first-container-in-allocation-order rule
// resolves each loop's parent to the lexically CLOSEST ancestor: the
// candidate scan meets the wired parent before any transitive outer one.
func (lr *lowerRun) buildNest(lw *lntRewrite) ir.Node {
	// Inside loops, innermost first (reverse dim order). Header-only: each
	// loop carries [iv, bound, attrs...] — no body operand.
	for i := len(lw.dims) - 1; i >= 0; i-- {
		d := lw.dims[i]
		d.inside = lr.g.NewForLoop(lowerOrigin, d.insideIV,
			lr.g.NewBoundedDimVar(lowerOrigin, d.insideIV, d.insideBound),
			d.insideAttrs...)
	}
	// Between loops, innermost first.
	for i := len(lw.dims) - 1; i >= 0; i-- {
		d := lw.dims[i]
		d.between = lr.g.NewForLoop(lowerOrigin, d.betweenIV,
			lr.g.NewBoundedDimVar(lowerOrigin, d.betweenIV, d.betweenBound),
			d.betweenAttrs...)
	}
	// Wire the canonical chain front-to-back via the "nest" annotation: the
	// chain is betweens in dim order, then insides in dim order.
	chain := make([]ir.Node, 0, 2*len(lw.dims))
	for _, d := range lw.dims {
		chain = append(chain, d.between)
	}
	for _, d := range lw.dims {
		chain = append(chain, d.inside)
	}
	for k := 1; k < len(chain); k++ {
		nest := lr.g.NewKeyedAttribute(lowerOrigin, "nest", chain[k-1])
		lr.mctx.AddUse(chain[k], nest)
	}
	// The nest root is the outermost loop (dims[0] is outermost).
	return lw.dims[0].between
}

// ivMembers derives the P6 inside-set per fix1.md D-g: the D-g referential
// set (every node whose data-dep operand closure references any of the
// LNT's loops' IVs — the same IV-membership relation loopforest uses, D-b),
// minus the kernel's return node, closed backward over data + payload edges.
//
// The return exclusion is what keeps P6 non-vacuous, and it is D-c's own
// case: a node that uses the loop's IVs yet is ordered after the loop's
// completion — the end node's data-dep closure reaches the IVs through the
// final store, so bare membership would class it as body and its chain edge
// would keep local order, leaving the generated nest unreachable from the
// roots. The epilogue (return, and anything ordered after it — nothing, in
// the v0 single-block kernels) anchors the edges that re-point at the nest
// root; body-internal edges keep their local order.
//
// Both IV generations are linking input: body nodes still reference the
// original InductionVariable uses at this pipeline point (the RAUW onto the
// dim's index expression, which references the generated IVs, happens after
// repointing, in rauwAndRetire), while the generated loops' IVs are already
// referenced by the dims' index expressions.
func (lr *lowerRun) ivMembers(lw *lntRewrite) map[*ir.Base]bool {
	ivs := make(map[*ir.Base]bool)
	for _, d := range lw.dims {
		if d.betweenIV != nil {
			ivs[d.betweenIV.BasePtr()] = true
		}
		if d.insideIV != nil {
			ivs[d.insideIV.BasePtr()] = true
		}
	}
	iters := make(map[*ir.Base]bool)
	for _, it := range treeItersOf(lw.root) {
		if it != nil && it.BasePtr() != nil {
			iters[it.BasePtr()] = true
		}
	}
	for _, n := range lr.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		if n.KindOf() != ir.KindInductionVariable {
			continue
		}
		if deps := ir.DataDeps(n); len(deps) > 0 && deps[0] != nil && iters[deps[0].BasePtr()] {
			ivs[n.BasePtr()] = true
		}
	}
	// E: the D-g members that are not the kernel epilogue.
	seeds := make([]ir.Node, 0, 16)
	for _, n := range lr.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		if n.KindOf() == ir.KindReturn {
			continue // D-c: ordered after the loop's completion
		}
		if referencesAnyIV(n, ivs) {
			seeds = append(seeds, n)
		}
	}
	// inside = the backward (data + payload) cone of E: the chain position
	// the loop body occupies — the same extent bodyRootOf produced, restated
	// in sea vocabulary without a body operand.
	inside := make(map[*ir.Base]bool)
	for _, s := range seeds {
		for b := range lr.reachableFrom(s) {
			inside[b] = true
		}
	}
	return inside
}

// referencesAnyIV reports whether n's data-dep operand closure (transitive
// operand-slot walk, no effect edges) contains any of the given IVs.
func referencesAnyIV(n ir.Node, ivs map[*ir.Base]bool) bool {
	seen := make(map[*ir.Base]bool)
	var visit func(x ir.Node) bool
	visit = func(x ir.Node) bool {
		if x == nil || x.BasePtr() == nil || seen[x.BasePtr()] || x.BasePtr().Erased() {
			return false
		}
		if ivs[x.BasePtr()] {
			return true
		}
		seen[x.BasePtr()] = true
		for _, dep := range ir.DataDeps(x) {
			if visit(dep) {
				return true
			}
		}
		return false
	}
	return visit(n)
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
