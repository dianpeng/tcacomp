package tiling

// lower_cache.go is the loop-tile-lower pass's budget resolution and PHASE
// 4 (doc/optana/loop_tile_lower.md §6 + §6bis): the SMEM/FRAG cache
// insertion — every promoted MemLoad site lowers into a buffer declared at
// the fill point, one fill per fill-key epoch, and buffer accesses
// replacing the load's uses. Loads-only scope: v0 candidate sites are
// MemLoads; stores NEVER route through the cache. The pass never emits a
// barrier — the runtime owns ALL synchronization (§6 sync).
//
// Fact provenance: the site facts (INS chase, tie, covered set) were
// re-derived at PHASE 1 time, BEFORE the loop rewrite retired the
// InductionVariable nodes the chase walks (lower.go Run documents why);
// here the recomputed Tie/Covered are cross-checked against the record and
// a mismatch fails closed, and the site's CLASS is re-derived from the
// covered dims' SoC (the R2 tier rule) and must agree with the record.
//
// FILL PLACEMENT (R1, ruled 2026-10-21 — VARIANT-DEPENDENT; the former
// unconditional "immediately before the tile-inside loops" rule is
// SUPERSEDED — it mandated a position at which the fill's own per-thread
// coordinates were undefined, the HI-01 inconsistency):
//
//   - LoadTileThread (FRAG class, all covered dims sequential): the fill
//     sits INSIDE the element loop that DEFINES its per-thread coordinate —
//     the INSIDE loop of the innermost above-tie INS dim (the fill key,
//     e.g. the worked example's lhs refills per m-element inside M.inside;
//     fallback when no above-tie INS dim exists: the innermost covered
//     dim's between loop). Its coordinates are the index expressions
//     defined AT that position: covered dims ride their round bases, INS
//     dims above the fill point keep their full (post-RAUW) index
//     expressions.
//   - LoadTileCta (SMEM window class, any covered dim WARP-assigned): the
//     fill sits at the innermost covered dim's BETWEEN-round body, with
//     CTA-UNIFORM coordinates ONLY — round bases (between dimvars and
//     CTA-rank offsets); never a per-thread (ThreadMap) term. A WARP dim
//     above the tie has NO uniform coordinate and fails closed.
//
// Structural realization (the F8 splice, 261021-o61 T3): the fill is placed
// INSIDE the mandated host loop's body-operand subtree and is ordered
// before its readers. Loop membership in this IR is "inside the ForLoop
// body operand subtree"; the fill becomes a member through the pass-built
// per-buffer effect chain — every buffer access (which replaced the load's
// uses, i.e. sits where the readers sat) chains after the fill via its
// ImpactBy edge, so walking the host's body operand over data+payload
// edges reaches the fill, and the fill re-executes per fill-key epoch
// because its coordinates read that epoch's induction variable. The old
// `bodyRoot == load` re-point (HI-02's incoherent path: a cooperative fill
// INSIDE an innermost element loop with the buffer access outside the
// nest) is DELETED; when the erased load WAS a loop's body operand the
// RAUW moves that slot onto the buffer access — the coherent shape. The
// §8 self-checks (lower.go) enforce the placement structurally: reachable
// from the mandated host, never an operand-descendant of a position the
// variant forbids, coordinates in scope, uniform where required.
//
// WINDOW SIZING + ACCESS (R2, ruled 2026-10-21 — rank-in-index =>
// window-in-shape): a covered WARP dim's buffer extent is #threads_d x T_d
// (the whole group's round window — the decide side's siteWindowBytes
// prices the same product), and the access index on that dim is
// thread_map.index(d)·T_d + i (d = the dim's ThreadMap slot, i = the
// sequential inside walk). SEQ covered dims contribute T_d and index by
// their plain inside coordinate. FRAG buffers are per-thread: plain i.
//
// Effect wiring (construction-exempt, the ir env-chained constructors'
// precedent): the fill's source load chains on the source tensor exactly
// as NewMemLoadChained would (an Env primed with the erased load's chain
// predecessors); the buffer accesses chain after the fill on the buffer's
// own chain, so the fill dominates its reads in the sea's explicit effect
// order.

import (
	"fmt"

	"tcacomp/internal/ir"
)

// The loadtile copy variants (§6's tier rule / doc/tca.md §5.11): the value
// rides the fill store's trailing KeyedAttribute and the runtime selects
// the loader from it.
const (
	loadTileCta    = 1.0 // cooperative CTA loader (SMEM window class)
	loadTileThread = 2.0 // per-thread loader (FRAG class)
)

// siteFill is one promoted site's PHASE 4 product, recorded for the §8
// epilogue checks (placement, scope, uniformity, layering, FRAG
// fully-unroll).
type siteFill struct {
	siteID  uint64
	load    ir.Node // the erased GMEM load (tombstone after surgery)
	src     ir.Node // the fill's tile-shaped source MemLoad
	store   ir.Node // the fill store (MemStoreSMem/FragStore)
	buffer  ir.Node // the MemNewSMem/FragNew buffer
	covered []*dimRewrite
	host    ir.Node // the mandated fill-position loop (R1) — §8 placement anchor
	variant float64 // the loadtile variant emitted (1 = CTA, 2 = Thread)
}

// resolveBudget implements §6's fallback BEFORE any mutation (halving a
// covered dim's tile changes the loop nest PHASE 3 builds, so the
// resolution produces per-iter tile overrides consumed by both phases).
// While the recomputed KERNEL-WIDE sums exceed their budgets (R3: Σ SMEM
// window bytes over ALL LNTs' promoted SMEM sites ≤ MaxCache; F6: Σ
// per-thread FRAG bytes ≤ MaxFragPerThread): halve the offending site's
// covered dims' tile sizes; when ANY covered iter of a site reaches tile 1,
// drop the site back to GMEM (F4, ruled 2026-10-21 — the judge's promotion
// gate; the former all-covered-at-1 rule could strand a promoted site at a
// covered tile of 1 and make lowerSite fail closed on feasible input).
// Terminates by construction — halving reaches 1, dropping empties the
// set. With Params mirroring the decide pass's this is a NO-OP:
// tiling-verify already asserted both sums within budget.
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
		return siteWindowBytes(sf, m)
	}
	// kernelSum returns (smem, frag) over every promoted, non-dropped site
	// — the SAME window-bytes sums the decide feasibility and tiling-verify
	// gate (R3/F6).
	kernelSum := func() (int, int) {
		smem, frag := 0, 0
		for _, lf := range facts {
			for _, sf := range lf.CS {
				sd, ok := rec.Sites[sf.Load.BasePtr().ID()]
				if !ok || !sd.Promoted || dropped[sf.Load.BasePtr().ID()] {
					continue
				}
				switch sd.Class {
				case ClassSMEM:
					smem += bytesOf(sf)
				case ClassFRAG:
					frag += bytesOf(sf)
				}
			}
		}
		return smem, frag
	}
	// anyTileOne reports whether ANY covered iter of the site sits at tile
	// 1 under the effective decisions (F4: the drop predicate).
	anyTileOne := func(sf SiteFacts) bool {
		for _, it := range sf.Covered {
			if dec(it).TileSize <= 1 {
				return true
			}
		}
		return false
	}
	for {
		smem, frag := kernelSum()
		if smem <= p.Params.MaxCache && frag <= p.Params.MaxFragPerThread {
			break
		}
		// The offending site: max bytes, ties broken by the lowest load id
		// (deterministic — never map iteration).
		var pick SiteFacts
		found := false
		for _, lf := range facts {
			for _, sf := range lf.CS {
				sd, ok := rec.Sites[sf.Load.BasePtr().ID()]
				if !ok || !sd.Promoted || dropped[sf.Load.BasePtr().ID()] {
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
		for _, it := range pick.Covered {
			next := dec(it).TileSize / 2
			if next < 1 {
				next = 1
			}
			overrides[it] = next
		}
		// F4 sweep: ANY covered iter at tile 1 drops the site — shared
		// covered iters can drive a non-picked site to 1 while its other
		// covered dim stays > 1, and the pass must never strand a promoted
		// site into a lowerSite error. Facts order, deterministic.
		for _, lf := range facts {
			for _, sf := range lf.CS {
				id := sf.Load.BasePtr().ID()
				sd, ok := rec.Sites[id]
				if !ok || !sd.Promoted || dropped[id] {
					continue
				}
				if anyTileOne(sf) {
					dropped[id] = true
				}
			}
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
// variant-mandated fill point (R1), window-shaped buffer + rank-in-index
// access (R2), access rewrite (RAUW), retirement of the GMEM load.
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

	// Tier re-derivation (R2, ruled 2026-10-21): classify by the SoC of the
	// COVERED dims — ANY CTA-covered dim REFUSES the site (no coherent fill
	// variant under the v0 mechanics; a promoted record entry saying
	// otherwise is corruption), any WARP-covered dim makes the SMEM window
	// class, all-sequential makes the FRAG class. The record must agree.
	class := ClassFRAG
	for _, d := range covered {
		if d.soc == SoCTA {
			return fmt.Errorf("tile-lower: site %d covers CTA-assigned iter node %d — a CTA-covered tile has no coherent fill variant and is refused (R2 tier rule; the record's promoted entry is corrupted)",
				id, d.iter.BasePtr().ID())
		}
		if d.soc == SoCWARP {
			class = ClassSMEM
		}
	}
	if class != sd.Class {
		return fmt.Errorf("tile-lower: site %d record class %s disagrees with the R2 tier rule over its covered dims (%s) — record corruption",
			id, sd.Class, class)
	}

	// Covered dims keep T > 1 (the judge's promotion gate = the F4 drop
	// rule; a tile-1 covered dim reaching surgery is corruption).
	for _, d := range covered {
		if d.single || d.insideIV == nil {
			return fmt.Errorf("tile-lower: site %d covers iter node %d at tile size %d (covered dims must keep T > 1)", id, d.iter.BasePtr().ID(), d.tile)
		}
	}

	// The owning LNT rewrite + the variant-mandated fill host (R1).
	lw, err := lr.owningRewrite(covered[0].iter)
	if err != nil {
		return fmt.Errorf("tile-lower: site %d: %w", id, err)
	}
	host, err := lr.fillHost(lw, sf, covered, class)
	if err != nil {
		return fmt.Errorf("tile-lower: site %d: %w", id, err)
	}

	elem, ok := elemTypeOf(sf.Load)
	if !ok {
		return fmt.Errorf("tile-lower: site %d carries no element type (the buffer extent is not derivable)", id)
	}

	// Buffer: the R2 WINDOW model — per covered dim, WARP dims contribute
	// the whole group's round window (#threads_d = LoC_d x 32 elements of
	// T_d each), SEQ dims contribute T_d. The byte total is exactly the
	// decide side's siteWindowBytes product (one budget, both halves). The
	// tile TENSOR TYPE binds via SetType — the typeinfer MemLoadSMem rule's
	// own message names this pass as the binder ("cache lowering must bind
	// buffer types first"), the one documented exception to P8's
	// no-self-typing.
	bytes := elemBytes(elem)
	dims := make([]ir.DimRef, 0, len(covered))
	for _, d := range covered {
		extent := d.tile
		if d.soc == SoCWARP {
			extent = d.loc * 32 * d.tile // the group window, rank-in-index (R2)
		}
		bytes *= extent
		name := ""
		if dimNode, err := dimOfIter(d.iter); err == nil && dimNode != nil {
			name = ir.InputParamNameOf(dimNode)
		}
		dims = append(dims, ir.DimRef{Name: name, Extent: uint64(extent)})
	}
	var buffer ir.Node
	if class == ClassFRAG {
		buffer = lr.g.NewFragNew(lowerOrigin, lr.u32c(float64(bytes)))
	} else {
		buffer = lr.g.NewMemNewSMem(lowerOrigin, lr.u32c(float64(bytes)))
	}
	if err := lr.mctx.SetType(buffer, ir.InternTensor(elem, dims)); err != nil {
		return fmt.Errorf("tile-lower: binding site %d buffer type: %w", id, err)
	}

	// Fill source: the original tensor at the fill position's coordinates.
	// The covered dims are the TILE — the loader enumerates them internally
	// (the compiler emits no loop for the fill, §6bis), so a position
	// referencing only covered iters contributes NO axis; the tile's
	// covered round bases ride the FILL STORE's biv operands (P5). INS dims
	// above the fill point contribute the index expressions defined AT the
	// position: FRAG fills keep their full (post-RAUW) index expressions —
	// the fill key dim's inside coordinate is what re-executes the fill per
	// epoch; SMEM fills substitute EVERY referenced dim with its
	// CTA-UNIFORM base (round base: a between dimvar, or a CTA-rank offset
	// — never a per-thread ThreadMap term); a compound position mixing
	// covered and above-tie iters substitutes the covered index expressions
	// with their round bases; unreferenced positions stay absent (the
	// partial deref keeps them as the slice's trailing dims).
	origAxes := ir.OperandsOf(sf.Load)[1:]
	fillAxes := make([]ir.Node, 0, len(origAxes))
	for pos, axis := range origAxes {
		if pos >= len(sf.AxesIters) || len(sf.AxesIters[pos]) == 0 {
			continue // unreferenced position: absent
		}
		coveredHere, aboveHere := false, false
		for _, it := range sf.AxesIters[pos] {
			if coveredSet[it] {
				coveredHere = true
			} else {
				aboveHere = true // INS at or outer than the fill key
			}
		}
		switch {
		case coveredHere && !aboveHere:
			continue // the loader enumerates the covered tile dim
		case class == ClassSMEM:
			// Uniform-base substitution for EVERY referenced dim of the
			// position (R1: CTA-uniform coordinates only). A WARP dim
			// above the tie has no uniform coordinate — fail closed.
			for _, it := range sf.AxesIters[pos] {
				if coveredSet[it] {
					continue
				}
				if d, ok := lr.byIter[it]; ok && d.soc == SoCWARP {
					return fmt.Errorf("tile-lower: site %d axis %d references WARP iter node %d above the fill point — a LoadTileCta fill has no CTA-uniform coordinate for it (R1)",
						id, pos, it.BasePtr().ID())
				}
			}
			rebuilt, n := lr.rebuildAxis(axis, lr.uniformBaseSubs(lw))
			if n == 0 {
				return fmt.Errorf("tile-lower: site %d axis %d holds no substitutable index expression (corrupted rewrite?)", id, pos)
			}
			fillAxes = append(fillAxes, rebuilt)
		case coveredHere && aboveHere:
			// FRAG, mixed position: covered dims ride their round bases,
			// above-tie dims keep their full index expressions.
			rebuilt, n := lr.rebuildAxis(axis, lr.coveredBaseSubs(covered))
			if n == 0 {
				return fmt.Errorf("tile-lower: site %d axis %d mixes covered and above-tie iters but holds no covered index expression (corrupted rewrite?)", id, pos)
			}
			fillAxes = append(fillAxes, rebuilt)
		default:
			fillAxes = append(fillAxes, axis) // FRAG, above-tie only: bake as-is
		}
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

	// Fill store + variant annotation (§6's tier rule): SMEM window class →
	// cooperative LoadTileCta (1); FRAG class → per-thread LoadTileThread
	// (2). The covered round bases ride the store's trailing biv operands:
	// between dimvars only — uniform for the cooperative loader (R1/R2).
	variant := loadTileThread
	if class == ClassSMEM {
		variant = loadTileCta
	}
	roundBases := make([]ir.Node, 0, len(covered))
	for _, d := range covered {
		roundBases = append(roundBases, d.betweenIV)
	}
	variantAttr := lr.attr("loadtile", variant)
	var fillStore ir.Node
	if class == ClassFRAG {
		fillStore = lr.g.NewFragStore(lowerOrigin, src, buffer, append(roundBases, variantAttr)...)
	} else {
		fillStore = lr.g.NewMemStoreSMem(lowerOrigin, src, buffer, append(roundBases, variantAttr)...)
	}

	// Access: every use of the erased load reads the buffer at the covered
	// dims' in-tile coordinates (M4) — R2 rank-in-index: a WARP-covered
	// dim's index is thread_map.index(d)·T_d + i (d = the dim's ThreadMap
	// slot — one map per LNT, tuple coordinates, §7bis — NOT a hardcoded 0
	// and NOT the bare inside coordinate); SEQ dims index by their plain
	// inside coordinate. The accesses chain after the fill on the buffer's
	// own effect chain — this is also the F8 structural splice: the fill
	// becomes reachable from the host loop's body-operand subtree through
	// its readers and is ordered before them.
	coords := make([]ir.Node, 0, len(covered))
	for _, d := range covered {
		if d.soc == SoCWARP {
			rank := lr.g.NewIntrinsicThreadMap(lowerOrigin, "index", lr.u32c(float64(d.threadDim)))
			shifted := lr.g.NewBinOp(lowerOrigin, ir.BinOpMul, rank, lr.u32c(float64(d.tile)))
			coords = append(coords, lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd, shifted, d.insideIV))
		} else {
			coords = append(coords, d.insideIV)
		}
	}
	bufEnv := ir.NewEnv()
	bufEnv.NoteStore(buffer, fillStore)
	var access ir.Node
	if class == ClassFRAG {
		access = lr.g.NewFragLoadChained(lowerOrigin, bufEnv, buffer, coords...)
	} else {
		access = lr.g.NewMemLoadSMemChained(lowerOrigin, bufEnv, buffer, coords...)
	}

	// Access rewrite: RAUW every data use, then retire the load. There is
	// NO bodyRoot == load special case (the HI-02 path is deleted): when
	// the erased load WAS a loop's body operand, the RAUW itself moves that
	// body slot onto the buffer access — the coherent shape (the fill stays
	// at its variant-mandated position).
	if err := lr.mctx.ReplaceAllUsesWith(sf.Load, access); err != nil {
		return fmt.Errorf("tile-lower: RAUW of site %d load: %w", id, err)
	}
	if err := lr.mctx.Erase(sf.Load); err != nil {
		return fmt.Errorf("tile-lower: retiring site %d load: %w", id, err)
	}

	lr.fills = append(lr.fills, &siteFill{
		siteID:  id,
		load:    sf.Load,
		src:     src,
		store:   fillStore,
		buffer:  buffer,
		covered: covered,
		host:    host,
		variant: variant,
	})
	return nil
}

// owningRewrite resolves the LNT rewrite owning iter (the rewrite whose
// dims contain it — overlap ownership guarantees one).
func (lr *lowerRun) owningRewrite(iter ir.Node) (*lntRewrite, error) {
	for _, lw := range lr.rewrites {
		for _, d := range lw.dims {
			if d.iter.BasePtr() == iter.BasePtr() {
				return lw, nil
			}
		}
	}
	return nil, fmt.Errorf("covered iter node %d belongs to no rewritten LNT", iter.BasePtr().ID())
}

// fillHost returns the variant-mandated fill-position loop (R1, §6bis
// PHASE 4):
//
//   - ClassSMEM (cooperative LoadTileCta): the INNERMOST covered dim's
//     between-round loop — the fill runs once per round of that dim,
//     immediately before the tile-inside loops, with CTA-uniform
//     coordinates only.
//   - ClassFRAG (per-thread LoadTileThread): the INSIDE loop of the
//     INNERMOST above-tie INS dim (the fill key — the dim whose element
//     change invalidates the fragment, e.g. the worked example's lhs
//     refills per m-element inside M.inside); a T==1 fill-key dim uses its
//     single loop; with no above-tie INS dim the position degenerates to
//     the SMEM rule (the innermost covered dim's between loop).
//
// The returned loop anchors the §8 structural placement checks.
func (lr *lowerRun) fillHost(lw *lntRewrite, sf SiteFacts, covered []*dimRewrite, class CacheClass) (ir.Node, error) {
	innermost := func(pred func(d *dimRewrite) bool) *dimRewrite {
		var best *dimRewrite
		bi := -1
		for i, d := range lw.dims {
			if pred(d) && i > bi {
				best, bi = d, i
			}
		}
		return best
	}
	coveredSet := make(map[ir.Node]bool, len(covered))
	for _, d := range covered {
		coveredSet[d.iter] = true
	}
	// The innermost covered dim anchors the SMEM position (and the FRAG
	// degenerate fallback).
	hostDim := innermost(func(d *dimRewrite) bool { return coveredSet[d.iter] })
	if hostDim == nil {
		return nil, fmt.Errorf("no covered dim of the owning LNT found for the fill position")
	}
	if class == ClassSMEM {
		return hostDim.between, nil // T > 1 precondition ⇒ between exists
	}
	// FRAG path: the innermost above-tie INS dim — the fill key.
	insSet := make(map[ir.Node]bool, len(sf.INS))
	for _, it := range sf.INS {
		insSet[it] = true
	}
	keyDim := innermost(func(d *dimRewrite) bool { return insSet[d.iter] && !coveredSet[d.iter] })
	if keyDim == nil {
		return hostDim.between, nil // degenerate: no above-tie INS dim
	}
	if keyDim.inside != nil {
		return keyDim.inside, nil
	}
	return keyDim.between, nil // T==1 fill-key dim: its single loop
}

// coveredBaseSubs maps every covered dim's rewritten index expression to
// its round base (the fill holds the whole tile: covered dims enumerate at
// round bases; the loader walks the tile interior at runtime).
func (lr *lowerRun) coveredBaseSubs(covered []*dimRewrite) map[*ir.Base]ir.Node {
	subs := make(map[*ir.Base]ir.Node, len(covered))
	for _, d := range covered {
		subs[d.indexExpr.BasePtr()] = d.roundBase
	}
	return subs
}

// uniformBaseSubs maps EVERY dim of the LNT — covered and above-tie alike —
// to its CTA-UNIFORM base (R1): SEQ dims ride their between-round dimvar
// (T>1) or their single-loop IV; a CTA dim rides its rank-offset round base
// (cta_rank·T + round — uniform across the CTA: the cooperative loader is
// called by every thread of the CTA, and they all read the same grid
// rank); a WARP dim rides its between dimvar when it has one — its
// rank-offset round base carries the per-thread ThreadMap term, which a
// cooperative fill must never see. The caller fails closed on above-tie
// WARP dims before substituting; a covered WARP dim always has T > 1 (the
// surgery precondition), hence a between dimvar.
func (lr *lowerRun) uniformBaseSubs(lw *lntRewrite) map[*ir.Base]ir.Node {
	subs := make(map[*ir.Base]ir.Node, len(lw.dims))
	for _, d := range lw.dims {
		var base ir.Node
		switch d.soc {
		case SoCWARP:
			base = d.betweenIV
		default: // SEQ: between dimvar or the T==1 single IV; CTA: rank offset
			base = d.roundBase
		}
		if base != nil {
			subs[d.indexExpr.BasePtr()] = base
		}
	}
	return subs
}

// rebuildAxis clones the axis expression, replacing every mapped index
// expression with its substitution. Returns the rebuilt node and the
// substitution count.
func (lr *lowerRun) rebuildAxis(axis ir.Node, subs map[*ir.Base]ir.Node) (ir.Node, int) {
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
