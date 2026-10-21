package tiling

// lower_cache.go is the loop-tile-lower pass's budget resolution and PHASE
// 4 (doc/optana/loop_tile_lower.md §6 + §6bis): the SMEM/FRAG cache
// insertion — every promoted MemLoad site lowers into a buffer declared at
// the fill point, one cooperative fill per round, and buffer accesses
// replacing the load's uses. Loads-only scope: v0 candidate sites are
// MemLoads; stores NEVER route through the cache. The pass never emits a
// barrier — the runtime owns ALL synchronization (§6 sync).
//
// Fact provenance: the site facts (INS chase, tie, covered set) were
// re-derived at PHASE 1 time, BEFORE the loop rewrite retired the
// InductionVariable nodes the chase walks (lower.go Run documents why);
// here the recomputed Tie/Covered are cross-checked against the record and
// a mismatch fails closed.
//
// Fill point (§6bis PHASE 4 + M3): inside the innermost covered dim's
// between-round body, immediately before the tile-inside loops (Tie == nil
// → the same rule at the outermost position). One buffer per site,
// declared at the fill point, refilled every round; the buffer is a
// static declaration reused across rounds, never re-allocated.
//
// Effect wiring (construction-exempt, the ir env-chained constructors'
// precedent): the fill's source load chains on the source tensor exactly
// as NewMemLoadChained would (an Env primed with the erased load's chain
// predecessors); the buffer accesses chain after the fill on the buffer's
// own chain, so the fill dominates its reads in the sea's explicit effect
// order. A fill whose site load had no data users is anchored by nothing
// (exactly as the un-chained GMEM load it replaced was) — the real
// pipeline's data flow anchors every fill through its accesses.

import (
	"fmt"

	"tcacomp/internal/ir"
)

// siteFill is one promoted site's PHASE 4 product, recorded for the §8
// epilogue checks (layering, FRAG fully-unroll).
type siteFill struct {
	siteID    uint64
	load      ir.Node // the erased GMEM load (tombstone after surgery)
	src       ir.Node // the fill's tile-shaped source MemLoad
	store     ir.Node // the fill store (MemStoreSMem/FragStore)
	buffer    ir.Node // the MemNewSMem/FragNew buffer
	covered   []*dimRewrite
	annotated bool // the loadtile variant annotation rides the fill store
}

// resolveBudget implements §6's fallback BEFORE any mutation (halving a
// covered dim's tile changes the loop nest PHASE 3 builds, so the
// resolution produces per-iter tile overrides consumed by both phases).
// While the recomputed Σ SMEM bytes exceeds MaxCache: halve the offending
// site's covered dims' tile sizes; when a site's covered tiles all reach 1,
// drop it back to GMEM (its load survives unrewritten). Terminates by
// construction — halving reaches 1, dropping empties the set. With Params
// mirroring the decide pass's this is a NO-OP: tiling-verify already
// asserted Σ ≤ MaxCache.
func (p LowerPass) resolveBudget(rec *Record, facts []lntFacts, decisions map[ir.Node]IterDecision) (map[ir.Node]int, map[uint64]bool, error) {
	overrides := make(map[ir.Node]int)
	dropped := make(map[uint64]bool)
	dec := func(it ir.Node) IterDecision {
		d := decisions[it]
		if t, ok := overrides[it]; ok {
			d.TileSize = t
		}
		return d
	}
	bytesOf := func(sf SiteFacts) int {
		m := make(map[ir.Node]IterDecision, len(sf.Covered))
		for _, it := range sf.Covered {
			m[it] = dec(it)
		}
		return siteBytesPerTile(sf, m)
	}
	smemSum := func() int {
		sum := 0
		for _, lf := range facts {
			for _, sf := range lf.CS {
				sd, ok := rec.Sites[sf.Load.BasePtr().ID()]
				if !ok || !sd.Promoted || sd.Class != ClassSMEM || dropped[sf.Load.BasePtr().ID()] {
					continue
				}
				sum += bytesOf(sf)
			}
		}
		return sum
	}
	for smemSum() > p.Params.MaxCache {
		// The offending site: max bytes, ties broken by the lowest load id
		// (deterministic — never map iteration).
		var pick SiteFacts
		found := false
		for _, lf := range facts {
			for _, sf := range lf.CS {
				sd, ok := rec.Sites[sf.Load.BasePtr().ID()]
				if !ok || !sd.Promoted || sd.Class != ClassSMEM || dropped[sf.Load.BasePtr().ID()] {
					continue
				}
				id := sf.Load.BasePtr().ID()
				if !found || bytesOf(sf) > bytesOf(pick) ||
					(bytesOf(sf) == bytesOf(pick) && id < pick.Load.BasePtr().ID()) {
					pick, found = sf, true
				}
			}
		}
		if !found {
			break // nothing left to halve or drop
		}
		allOne := true
		for _, it := range pick.Covered {
			if dec(it).TileSize > 1 {
				allOne = false
				break
			}
		}
		if allOne {
			dropped[pick.Load.BasePtr().ID()] = true
			continue
		}
		for _, it := range pick.Covered {
			next := dec(it).TileSize / 2
			if next < 1 {
				next = 1
			}
			overrides[it] = next
		}
	}
	return overrides, dropped, nil
}

// lowerSites is PHASE 4: per promoted site (in the re-derived facts order —
// deterministic), the buffer/fill/access surgery.
func (lr *lowerRun) lowerSites(rec *Record, facts []lntFacts, dropped map[uint64]bool) error {
	for _, lf := range facts {
		for _, sf := range lf.CS {
			id := sf.Load.BasePtr().ID()
			sd, ok := rec.Sites[id]
			if !ok || !sd.Promoted || dropped[id] {
				continue
			}
			if !sameIter(sd.Tie, sf.Tie) {
				return fmt.Errorf("tile-lower: site %d record tie disagrees with the recomputed facts (record corruption?)", id)
			}
			if err := lr.lowerSite(sf, sd); err != nil {
				return err
			}
		}
	}
	return nil
}

// sameIter compares two tie slots: nil == nil, otherwise node identity.
func sameIter(a, b ir.Node) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return a.BasePtr() == b.BasePtr()
}

// lowerSite performs one promoted site's surgery: buffer + fill at the
// fill point, access rewrite (RAUW), chain re-pointing (P6), retirement of
// the GMEM load.
func (lr *lowerRun) lowerSite(sf SiteFacts, sd SiteDecision) error {
	id := sf.Load.BasePtr().ID()

	// Covered dims in tensor-position order (the buffer's dim order — the
	// fill coordinates and the access coordinates both use it).
	coveredSet := make(map[ir.Node]bool, len(sf.Covered))
	for _, it := range sf.Covered {
		coveredSet[it] = true
	}
	var covered []*dimRewrite
	seen := make(map[ir.Node]bool, len(sf.Covered))
	for _, axisIters := range sf.AxesIters {
		for _, it := range axisIters {
			if !coveredSet[it] || seen[it] {
				continue
			}
			d, ok := lr.byIter[it]
			if !ok {
				return fmt.Errorf("tile-lower: site %d covers iter node %d, which the rewrite did not produce", id, it.BasePtr().ID())
			}
			seen[it] = true
			covered = append(covered, d)
		}
	}
	for _, it := range sf.Covered { // defensive: any covered iter the axes never mention
		if !seen[it] {
			d, ok := lr.byIter[it]
			if !ok {
				return fmt.Errorf("tile-lower: site %d covers iter node %d, which the rewrite did not produce", id, it.BasePtr().ID())
			}
			covered = append(covered, d)
		}
	}
	if len(covered) == 0 {
		return fmt.Errorf("tile-lower: site %d promotes an empty covered set (M2: filtered at the decide pass)", id)
	}

	// Promotion preconditions, fail-closed at surgery time.
	if sd.Class == ClassFRAG {
		for _, d := range covered {
			if d.soc != SoCSequential {
				return fmt.Errorf("tile-lower: FRAG site %d covers non-sequential iter node %d (the fully-unroll precondition)", id, d.iter.BasePtr().ID())
			}
		}
	}
	for _, d := range covered {
		if d.single || d.insideIV == nil {
			return fmt.Errorf("tile-lower: site %d covers iter node %d at tile size %d (covered dims must keep T > 1)", id, d.iter.BasePtr().ID(), d.tile)
		}
	}

	elem, ok := elemTypeOf(sf.Load)
	if !ok {
		return fmt.Errorf("tile-lower: site %d carries no element type (the buffer extent is not derivable)", id)
	}

	// Buffer: extent = element bytes × the product of the covered tile
	// sizes; the tile TENSOR TYPE binds via SetType — the typeinfer
	// MemLoadSMem rule's own message names this pass as the binder
	// ("cache lowering must bind buffer types first"), the one documented
	// exception to P8's no-self-typing.
	bytes := elemBytes(elem)
	dims := make([]ir.DimRef, 0, len(covered))
	for _, d := range covered {
		bytes *= d.tile
		name := ""
		if dimNode, err := dimOfIter(d.iter); err == nil && dimNode != nil {
			name = ir.InputParamNameOf(dimNode)
		}
		dims = append(dims, ir.DimRef{Name: name, Extent: uint64(d.tile)})
	}
	var buffer ir.Node
	if sd.Class == ClassFRAG {
		buffer = lr.g.NewFragNew(lowerOrigin, lr.u32c(float64(bytes)))
	} else {
		buffer = lr.g.NewMemNewSMem(lowerOrigin, lr.u32c(float64(bytes)))
	}
	if err := lr.mctx.SetType(buffer, ir.InternTensor(elem, dims)); err != nil {
		return fmt.Errorf("tile-lower: binding site %d buffer type: %w", id, err)
	}

	// Fill source: the original tensor at the current round offsets —
	// covered dims at their round bases (the loader enumerates the tile
	// interior and clamps partial tiles at runtime, §4/§6), INS dims above
	// the tie at their enclosing rewritten index expressions (post-RAUW
	// original axes reference them already), unreferenced positions absent
	// (the partial deref keeps them as the slice's trailing dims).
	origAxes := ir.OperandsOf(sf.Load)[1:]
	fillAxes := make([]ir.Node, 0, len(origAxes))
	substitutions := 0
	for pos, axis := range origAxes {
		if pos >= len(sf.AxesIters) || len(sf.AxesIters[pos]) == 0 {
			continue // unreferenced position: absent
		}
		if !axisReferencesCovered(axis, covered) {
			fillAxes = append(fillAxes, axis) // above-tie INS index: bake as-is
			continue
		}
		rebuilt, n := lr.rebuildAxis(axis, covered)
		if n == 0 {
			return fmt.Errorf("tile-lower: site %d axis %d references a covered iter but holds no covered index expression (corrupted rewrite?)", id, pos)
		}
		substitutions += n
		fillAxes = append(fillAxes, rebuilt)
	}
	if substitutions == 0 {
		return fmt.Errorf("tile-lower: site %d fill has no covered coordinate (corrupted rewrite?)", id)
	}

	// The fill's source load chains on the source tensor exactly as
	// NewMemLoadChained would: an Env primed with the erased load's chain
	// predecessors (its ImpactBy targets — chain order and predecessor
	// liveness preserved).
	env := ir.NewEnv()
	for _, u := range ir.ImpactBy(sf.Load) {
		if pred := u.Def(); pred != nil && !pred.BasePtr().Erased() {
			env.NoteStore(sf.Tensor, pred)
		}
	}
	src := lr.g.NewMemLoadChained(lowerOrigin, env, sf.Tensor, fillAxes)

	// Fill store + variant annotation (§6's copy-variant rule): any
	// covered dim group-parallel → LoadTileCta (1); all covered dims
	// sequential → LoadTileThread (2).
	variant := 2.0
	for _, d := range covered {
		if d.soc != SoCSequential {
			variant = 1.0
			break
		}
	}
	roundBases := make([]ir.Node, 0, len(covered))
	for _, d := range covered {
		roundBases = append(roundBases, d.roundBase)
	}
	variantAttr := lr.attr("loadtile", variant)
	var fillStore ir.Node
	if sd.Class == ClassFRAG {
		fillStore = lr.g.NewFragStore(lowerOrigin, src, buffer, append(roundBases, variantAttr)...)
	} else {
		fillStore = lr.g.NewMemStoreSMem(lowerOrigin, src, buffer, append(roundBases, variantAttr)...)
	}

	// Access: every use of the erased load reads the buffer at the covered
	// dims' in-tile coordinates (M4). The accesses chain after the fill on
	// the buffer's own effect chain.
	coords := make([]ir.Node, 0, len(covered))
	for _, d := range covered {
		coords = append(coords, d.insideIV)
	}
	bufEnv := ir.NewEnv()
	bufEnv.NoteStore(buffer, fillStore)
	var access ir.Node
	if sd.Class == ClassFRAG {
		access = lr.g.NewFragLoadChained(lowerOrigin, bufEnv, buffer, coords...)
	} else {
		access = lr.g.NewMemLoadSMemChained(lowerOrigin, bufEnv, buffer, coords...)
	}

	// Body-root re-point: when the erased load WAS the LNT's innermost
	// body operand, the body slot moves to the fill store (the load's
	// in-body position is taken by its fill).
	for _, lw := range lr.rewrites {
		if lw.bodyRoot == nil || lw.bodyRoot.BasePtr() != sf.Load.BasePtr() {
			continue
		}
		innermost := lw.dims[len(lw.dims)-1]
		host := innermost.inside
		if host == nil {
			host = innermost.between
		}
		for _, u := range ir.UsesOf(sf.Load) {
			if user := u.User(); user != nil && user.BasePtr() == host.BasePtr() {
				if err := lr.mctx.SetDef(u, fillStore); err != nil {
					return fmt.Errorf("tile-lower: re-pointing the body of loop node %d at site %d fill: %w", host.BasePtr().ID(), id, err)
				}
			}
		}
	}

	// Access rewrite: RAUW every data use, then re-point the erased load's
	// incoming per-tensor chain edges at the fill source (P6 — the fill
	// source takes the load's chain position), then retire the load.
	if err := lr.mctx.ReplaceAllUsesWith(sf.Load, access); err != nil {
		return fmt.Errorf("tile-lower: RAUW of site %d load: %w", id, err)
	}
	for _, u := range ir.UsesOf(sf.Load) {
		if err := lr.mctx.SetDef(u, src); err != nil {
			return fmt.Errorf("tile-lower: re-pointing a chain edge of site %d load: %w", id, err)
		}
	}
	if err := lr.mctx.Erase(sf.Load); err != nil {
		return fmt.Errorf("tile-lower: retiring site %d load: %w", id, err)
	}

	lr.fills = append(lr.fills, &siteFill{
		siteID:    id,
		load:      sf.Load,
		src:       src,
		store:     fillStore,
		buffer:    buffer,
		covered:   covered,
		annotated: true,
	})
	return nil
}

// axisReferencesCovered reports whether the (post-RAUW) axis subtree holds
// any covered dim's index expression.
func axisReferencesCovered(axis ir.Node, covered []*dimRewrite) bool {
	for _, d := range covered {
		if subtreeContains(axis, d.indexExpr) {
			return true
		}
	}
	return false
}

// subtreeContains reports whether target appears in n's data-dependency
// subtree.
func subtreeContains(n, target ir.Node) bool {
	if n == nil || target == nil {
		return false
	}
	if n.BasePtr() == target.BasePtr() {
		return true
	}
	for _, dep := range ir.DataDeps(n) {
		if subtreeContains(dep, target) {
			return true
		}
	}
	return false
}

// rebuildAxis clones the axis expression, replacing every covered dim's
// index expression with that dim's round base (the fill holds the whole
// tile: covered dims enumerate at round bases; the loader walks the tile
// interior at runtime). Returns the rebuilt node and the substitution
// count.
func (lr *lowerRun) rebuildAxis(axis ir.Node, covered []*dimRewrite) (ir.Node, int) {
	subs := make(map[*ir.Base]ir.Node, len(covered))
	for _, d := range covered {
		subs[d.indexExpr.BasePtr()] = d.roundBase
	}
	count := 0
	var rebuild func(n ir.Node) ir.Node
	rebuild = func(n ir.Node) ir.Node {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			return n
		}
		if repl, ok := subs[n.BasePtr()]; ok {
			count++
			return repl
		}
		ops := ir.DataDeps(n)
		changed := false
		newOps := make([]ir.Node, len(ops))
		copy(newOps, ops)
		for i, op := range ops {
			r := rebuild(op)
			if r != op {
				newOps[i] = r
				changed = true
			}
		}
		if !changed {
			return n
		}
		// Clone the node with rebuilt operands (construction-exempt — the
		// same freedom the constructors exercise; only NEW nodes result).
		return lr.cloneWithOperands(n, newOps)
	}
	return rebuild(axis), count
}

// cloneWithOperands clones a scalar-expression node with new operand defs.
// Only the expression kinds that can appear in an access axis need the
// clone path; anything else fails closed (a corrupted axis expression).
func (lr *lowerRun) cloneWithOperands(n ir.Node, ops []ir.Node) ir.Node {
	switch n.KindOf() {
	case ir.KindBinOp:
		op, _ := ir.BinOpCodeOf(n)
		return lr.g.NewBinOp(lowerOrigin, op, ops[0], ops[1])
	case ir.KindUnaryOp:
		op, _ := ir.UnaryOpCodeOf(n)
		return lr.g.NewUnaryOp(lowerOrigin, op, ops[0])
	case ir.KindSelect:
		return lr.g.NewSelect(lowerOrigin, ops[0], ops[1], ops[2])
	case ir.KindCast:
		return lr.g.NewCast(lowerOrigin, ir.CastTypeOf(n), ops[0])
	case ir.KindCall:
		return lr.g.NewCall(lowerOrigin, ir.CallNameOf(n), ops...)
	default:
		return n // leaves (consts, params, dimvars, intrinsics) clone as themselves
	}
}
