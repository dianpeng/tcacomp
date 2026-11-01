package tiling

// lower_fragchunk.go is the loop-tile-lower pass's fix7 §3 slice 1a
// pre-pass (design/fix/fix7.md §3): the STREAMING REGISTER-BLOCK plan.
// The v62 pricing (judge.go siteWindowBytes — the ONE shared source,
// untouched here) pays min(raw per-thread window, MaxFragPerThread) per
// promoted FRAG site; 1a makes the emission HONOR that contract: the
// buffer's InternTensor dims BECOME the chunk, the tile-between rounds
// step by the chunk (the rounds ARE the chunk loops), and the inside
// walks clamp to the chunk via the existing umin chain.
//
// The plan runs AFTER the budget resolution and BEFORE any LNT rewrite
// (PHASE 2-3): the between-IV step and the tileMax clamp target are built
// in generateDimExprs, so the chunk must be known before the rewrite. The
// plan is per COVERED ITER (one chunk per dim — the between loop is
// per-dim, so two promoted FRAG sites sharing a covered dim must agree on
// its chunk, and a disagreement fails closed rather than streaming one
// site wrong). PHASE 4 (lowerSite) realizes the buffer dims from the same
// per-dim plan — the buffer type is the ONE source of truth the cxx decl
// prints verbatim (walk.go).
//
// CHUNK POLICY (fix7 §3.1, pinned with its arithmetic in
// internal/cg/cxx/fragchunk_test.go): chunk dims = the covered dims' RAW
// factors (the SAME factors siteWindowBytesRaw prices: SEQ -> effective
// T_d, C1b WARP -> ceil(T_d/32)) reduced, while sizeof(chunk) exceeds the
// priced block footprint, by halving the OUTERMOST covered dim whose
// chunk > 1 (floor 1). Only SEQUENTIAL dims are reducible: a WARP-covered
// dim's per-lane chunk is the lane-chunk walk's fixed width WITHIN a
// round — halving it would desynchronize the walk (the fix7 §3.3 scope
// guard: no WARP streaming machinery in 1a). If every sequential dim
// reaches 1 and the chunk still does not fit, the pass fails closed with
// a named error — never a silent clamp.
//
// SMEM sites are NEVER chunked (their cooperative windows stay
// full-size); judge.go/search.go/record.go pricing, the search budget
// gate, and the schedule pass are UNTOUCHED (fix7 §3.3).

import (
	"fmt"
	"strconv"

	"tcacomp/internal/ir"
)

// planFragChunks plans the streaming chunk for every promoted FRAG site
// (the re-derived facts order — deterministic, matching lowerSites). The
// products ride lr.fragChunk (per covered iter; read by generateDimExprs
// for the between step + inside clamp and by lowerSite for the buffer
// dims).
func (lr *lowerRun) planFragChunks(rec *Record, facts []lntFacts, dropped map[uint64]bool) error {
	for _, lf := range facts {
		for _, sf := range lf.CS {
			id := sf.Load.BasePtr().ID()
			sd, ok := rec.Sites[id]
			if !ok || !sd.Promoted || dropped[id] {
				continue
			}
			class, planned, err := lr.fragClassPlan(sf)
			if err != nil {
				return err
			}
			if !planned || class != ClassFRAG {
				continue // refusals stay lowerSite's (the pre-pass must not change WHEN they fire); SMEM windows stay full-size
			}
			if err := lr.planSiteChunk(id, sf); err != nil {
				return err
			}
		}
	}
	return nil
}

// fragClassPlan re-derives the site's cache class from the effective
// decisions — the SAME R2/C1b tier rule lowerSite applies at surgery
// (lower_cache.go): any covered CTA dim refuses the site, a thread-varying
// non-covered coordinate (CTA or WARP) forces FRAG (C1b), otherwise any
// covered WARP dim makes the SMEM window class and all-sequential makes
// FRAG. planned is false for the refusal shapes — lowerSite owns those
// errors, so the pre-pass neither re-times nor re-words them.
func (lr *lowerRun) fragClassPlan(sf SiteFacts) (class CacheClass, planned bool, err error) {
	id := sf.Load.BasePtr().ID()
	coveredSet := make(map[ir.Node]bool, len(sf.Covered))
	for _, it := range sf.Covered {
		coveredSet[it] = true
		d, ok := lr.decisions[it]
		if !ok {
			return 0, false, fmt.Errorf("tile-lower: site %d covers iter node %d, which carries no record decision (corruption)", id, it.BasePtr().ID())
		}
		if d.SoC == SoCTA {
			return 0, false, nil // lowerSite refuses at surgery (R2 tier rule)
		}
	}
	forced := false
	for _, it := range sf.INS {
		if coveredSet[it] {
			continue
		}
		d, ok := lr.decisions[it]
		if !ok {
			return 0, false, fmt.Errorf("tile-lower: site %d references iter node %d in its INS with no record decision (corruption)", id, it.BasePtr().ID())
		}
		if d.SoC == SoCTA || d.SoC == SoCWARP {
			forced = true // C1b: a thread-varying non-covered coordinate
			break
		}
	}
	if !forced {
		for _, it := range sf.Covered {
			if lr.decisions[it].SoC == SoCWARP {
				return ClassSMEM, true, nil
			}
		}
	}
	return ClassFRAG, true, nil
}

// planSiteChunk computes and records one site's chunk dims. The budget is
// READ from siteWindowBytes — the ONE shared pricing source — with the
// same effective-decisions construction resolveBudget uses (never a
// reimplementation: judge.go/search.go/record.go stay untouched).
func (lr *lowerRun) planSiteChunk(id uint64, sf SiteFacts) error {
	e, ok := elemTypeOf(sf.Load)
	if !ok {
		return nil // lowerSite fails closed on the missing element type before any buffer dims are built
	}
	cov, err := siteCoveredTensorOrder(sf, id)
	if err != nil {
		return err
	}
	chunks := make([]int, len(cov))
	for i, it := range cov {
		d, ok := lr.decisions[it]
		if !ok {
			return fmt.Errorf("tile-lower: site %d covers iter node %d, which carries no record decision (corruption)", id, it.BasePtr().ID())
		}
		switch d.SoC {
		case SoCSequential:
			chunks[i] = lr.effectiveTile(it)
		case SoCWARP:
			chunks[i] = ceilDiv(lr.effectiveTile(it), 32) // C1b: the lane's chunk
		default:
			return fmt.Errorf("tile-lower: site %d covers iter node %d at SoC %d — the chunk plan supports sequential and C1b-warp dims only (a CTA-covered site is refused at surgery)", id, it.BasePtr().ID(), int(d.SoC))
		}
	}
	// fix7 §2 pathology 2, killed by construction: a covered
	// COMPILE-TIME-CONST dim sizes its chunk by its TRUE extent, never the
	// nominal tile (an e4-at-T=256 window would allocate 252/256
	// never-touched elements). The reduction lands BEFORE the halving
	// loop: const dims are already their true size, so the loop clamps
	// the dynamic outer dims first.
	for i, it := range cov {
		if ext, ok := lr.staticIterExtent(it); ok && ext < chunks[i] {
			chunks[i] = ext
		}
	}
	// The budget: min(raw, MaxFragPerThread) from the ONE shared source.
	m := make(map[ir.Node]IterDecision, len(sf.Covered))
	for _, it := range sf.Covered {
		dec := lr.decisions[it]
		if t, ok := lr.overrides[it]; ok {
			dec.TileSize = t
		}
		m[it] = dec
	}
	budget := siteWindowBytes(sf, m, ClassFRAG, lr.params)
	// The halving loop (policy in the file header): outermost reducible
	// dim first, floor 1, fail closed when nothing can shrink.
	elem := elemBytes(e)
	for fragChunkBytes(elem, chunks) > budget {
		halved := false
		for i := range chunks {
			if lr.decisions[cov[i]].SoC != SoCSequential || chunks[i] <= 1 {
				continue // WARP dims never halve (the lane-chunk walk width is fixed within a round)
			}
			chunks[i] = max(1, chunks[i]/2)
			halved = true
			break
		}
		if !halved {
			return fmt.Errorf("tile-lower: site %d: the FRAG window chunk %v (%d B) exceeds its priced block footprint %d B with every sequential covered dim at 1 — a WARP-covered dim streams only ACROSS rounds (undesigned in 1a, fix7 §3.3); fail closed, never a silent clamp",
				id, chunks, fragChunkBytes(elem, chunks), budget)
		}
	}
	// Record the plan per covered iter: the between step, the inside
	// clamp, and the buffer dims all read this ONE map. Two promoted FRAG
	// sites sharing a covered iter must agree — a single between loop
	// realizes a single chunk per dim, so a disagreement is unrealizable.
	for i, it := range cov {
		if prev, ok := lr.fragChunk[it]; ok && prev != chunks[i] {
			return fmt.Errorf("tile-lower: site %d: promoted FRAG sites disagree on the chunk of shared iter node %d (%d vs %d) — one between loop realizes one chunk per dim; this record combination is unrealizable (fix7 §3.1)",
				id, it.BasePtr().ID(), prev, chunks[i])
		}
		lr.fragChunk[it] = chunks[i]
	}
	return nil
}

// staticIterExtent resolves a covered iter's TRUE compile-time-const
// extent (pathology 2's kill). Source precedence: (1) the BIV max
// operand's const payload (the LO-05 extentOfBiv source — the real
// pipeline's const-dim extents are literal consts); (2) the dim
// InputParam's canonical "e<extent>" static spelling (build.go dimKey —
// the c8n identity source, mirrored by staticDimNameExtent). NEVER
// ir.TypeOf: lower-input nodes carry a NIL computed type cell (the
// post-typeinfer gotcha is real and load-bearing). A non-const extent
// reports false — the dim keeps its dynamic tile chunk.
func (lr *lowerRun) staticIterExtent(iter ir.Node) (int, bool) {
	if biv, err := bivOf(iter); err == nil {
		if ops := ir.DataDeps(biv); len(ops) >= 2 && ops[1] != nil {
			if v, ok := ir.ConstValueOf(ops[1]); ok && v >= 1 {
				return int(v), true
			}
		}
	}
	if dim, err := dimOfIter(iter); err == nil && dim != nil {
		return staticDimNameExtent(ir.InputParamNameOf(dim))
	}
	return 0, false
}

// staticDimNameExtent parses a dim InputParam's canonical static spelling
// "e<extent>" (build.go dimKey) into its declared compile-time extent —
// the tiling-package twin of internal/cg/cxx/memory.go's staticDimExtent
// (the c8n identity source; cg may not be imported from opt). A dynamic
// name never matches, and extent 0 is dynamic by definition. The parse is
// 32-bit-bounded: tensor extents are u32, and the rejection keeps the
// int conversion exact.
func staticDimNameExtent(name string) (int, bool) {
	if len(name) < 2 || name[0] != 'e' {
		return 0, false
	}
	for _, c := range name[1:] {
		if c < '0' || c > '9' {
			return 0, false
		}
	}
	ext, err := strconv.ParseUint(name[1:], 10, 32)
	if err != nil || ext == 0 {
		return 0, false
	}
	return int(ext), true
}

// siteCoveredTensorOrder orders the site's covered iters in tensor
// position order (the buffer's dim order) — the SAME construction
// lowerSite uses to order its covered dimRewrites, shared so the chunk
// plan and the surgery can never disagree on the dim order.
func siteCoveredTensorOrder(sf SiteFacts, id uint64) ([]ir.Node, error) {
	coveredSet := make(map[ir.Node]bool, len(sf.Covered))
	for _, it := range sf.Covered {
		coveredSet[it] = true
	}
	seen := make(map[ir.Node]bool, len(sf.Covered))
	var ordered []ir.Node
	for _, axisIters := range sf.AxesIters {
		for _, it := range axisIters {
			if !coveredSet[it] || seen[it] {
				continue
			}
			seen[it] = true
			ordered = append(ordered, it)
		}
	}
	for _, it := range sf.Covered { // defensive: any covered iter the axes never mention
		if !seen[it] {
			ordered = append(ordered, it)
		}
	}
	if len(ordered) == 0 {
		return nil, fmt.Errorf("tile-lower: site %d promotes an empty covered set (M2: filtered at the decide pass)", id)
	}
	return ordered, nil
}

// fragChunkBytes is sizeof(chunk) in bytes: the element size times the
// chunk factors (the factor model siteWindowBytesRaw prices).
func fragChunkBytes(elem int, chunks []int) int {
	bytes := elem
	for _, c := range chunks {
		bytes *= c
	}
	return bytes
}
