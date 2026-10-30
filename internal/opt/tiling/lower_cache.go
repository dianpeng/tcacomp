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
// THE FILL IS A COMPOSITION (fix3.md R-G/D-i): the three plain verbs —
// MemNew(class, extents) + a tile-shaped MemLoad(GMEM, tensor-level, axes
// + round bases: siteFill.src) + MemStore(class, buffer, value = the tile
// load) — never a fused node. Every buffer access is a plain
// MemLoad(class, buffer, windowed coords). Ordering rides the ORDINARY
// memory-effect chain alone (C7: the source load chains on the source
// tensor exactly as NewMemLoadChained would; the accesses chain after the
// fill store on the buffer's own chain — the fill dominates its reads in
// the sea's explicit effect order, and the §8 ORDERING check (lower.go)
// verifies exactly that over the ordinary ImpactBy edges; no special
// fill-edge reading exists).
//
// FILL PLACEMENT IS DERIVED, NEVER MANDATED (fix3.md R-I): a memory
// item's position is the innermost block dominated by every surviving
// loop its coordinate closure references, still preceding its memory
// successors. The R1-era variant-dependent MANDATE is STRUCK —
// fillHost()'s mechanism is retained through fix3 D-j ONLY so placement
// does not move before the derivation lands in final-lower's element-loop
// restructure; the suite pins that the derivation (innermost referenced
// loop) picks exactly the position fillHost mandates today (the D-i
// safety net, lower_test.go assertFillPositionDerived). Historical shape
// (what the derivation reproduces on today's graphs):
//
//   - FRAG class (per-thread fill): the fill key's INSIDE loop — the
//     position whose defining coordinate the fill's closure references
//     (e.g. the worked example's lhs refills per m-element inside
//     M.inside); degenerate fallback: the innermost covered dim's between
//     loop.
//   - SMEM window class (cooperative fill): the innermost covered dim's
//     BETWEEN-round body, with CTA-UNIFORM coordinates ONLY — round bases
//     (between dimvars and CTA-rank offsets); never a per-thread
//     (ThreadMap) term. A WARP dim above the tie has NO uniform
//     coordinate and fails closed.
//
// The fill re-executes per fill-key epoch because its coordinates read
// that epoch's induction variable. Membership is via IV references
// (fix1.md §0), never a body operand.
//
// WINDOW SIZING + ACCESS (fix4 §2.4/§3.3, R-C1B): a covered WARP dim's
// SMEM buffer extent is K_d x T_d (one window per warp coordinate — the
// SAME factor the decide side's shared siteWindowBytesRaw prices; the old
// whole-group #threads_d x T_d shard window is gone, 32x smaller now),
// and the access index on that dim is thread_map.index(d)·T_d + i (d =
// the dim's ThreadMap slot in warp units, i = the lane's sequential
// inside walk). A C1b forced-FRAG site sizes its WARP-covered dim at
// ceil(T_d/32) per lane. SEQ covered dims contribute T_d and index by
// their plain inside coordinate. FRAG buffers are per-thread: plain i —
// sized by the RAW window factors (fix6 §3: the budget side chunks the
// same window to min(raw, MaxFragPerThread); the allocation itself stays
// the full window).

import (
	"fmt"

	"tcacomp/internal/ir"
)

// The loader variants (§6's tier rule / doc/tca.md §5.11), pass-internal
// facts recorded on siteFill for the §8 checks — fix3 D-h: the value rides
// NO IR payload anymore; codegen derives the loader from the fill's
// storage class (R-H/R-J).
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
	src     ir.Node // the fill's tile-shaped GMEM source MemLoad (composition verb 2)
	store   ir.Node // the fill store: a tiered MemStore of src (composition verb 3)
	buffer  ir.Node // the tiered MemNew allocation (composition verb 1)
	covered []*dimRewrite
	// host is the R1-ERA mandated position loop — NOT a mandate anymore
	// (fix3 D-i): placement is DERIVED (R-I). The mechanism is retained
	// through D-j so placement does not move before final-lower derives
	// it; the suite pins derivation == this position (the D-i safety net).
	host    ir.Node
	variant float64 // the loader variant derived from the class (1 = CTA, 2 = Thread; pass-internal fact)
}

// resolveBudget implements §6's fallback BEFORE any mutation (halving a
// covered dim's tile changes the loop nest PHASE 3 builds, so the
// resolution produces per-iter tile overrides consumed by both phases).
// While the recomputed KERNEL-WIDE sums exceed their budgets (R3: Σ SMEM
// window bytes over ALL LNTs' promoted SMEM sites ≤ MaxCache; F6: Σ
// per-thread FRAG bytes ≤ MaxFragPerThread): halve the offending site's
// covered dims' tile sizes — with the F4 halving FLOOR at 2 (the
// 2026-10-24 tile-disable ruling): a picked site whose halved covered dim
// would land below 2 is DROPPED to GMEM instead (a site that cannot fit
// at tile >= 2 is dropped, never demoted to a non-tiling schedule — tile
// 1 is not a legal schedule). Terminates by construction — halving
// reaches the floor, dropping empties the set.
//
// UNREACHABLE FAIL-CLOSED BACKSTOP (fix4 §3.3): under the fix4 model the
// record is FEASIBLE BY CONSTRUCTION — the decide side's §2.6 greedy
// eviction already brought both kernel-wide sums within budget before the
// record was stamped, and tiling-verify re-asserted them (F7) — so with
// Params mirroring the decide pass's this resolution is a NO-OP. The code
// stays as defense-in-depth over hand-stamped/corrupted records and over
// Params mismatches (the §3.3 ruling: keep the halve-then-drop machinery,
// never delete it); TestBudgetFallback exercises it on crafted records.
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
		// The record's class decides the pricing model (the C1b precedence
		// may have forced FRAG over a WARP-covered dim — per-thread bytes),
		// priced by the SAME shared budget function the decide side's
		// eviction and tiling-verify gate (siteWindowBytes — fix6 §3: FRAG
		// prices the register-block footprint, min(raw, MaxFragPerThread)).
		return siteWindowBytes(sf, m, rec.Sites[sf.Load.BasePtr().ID()].Class, p.Params)
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
	// <= 1 under the effective decisions. Belt-and-braces only (2026-10-24
	// tile-disable ruling): the F4 halving floor above owns the drop — the
	// floor never writes an override below 2 and record decisions are >= 2
	// (tiling-verify), so this predicate can no longer fire; it stays as
	// the defensive sweep over shared covered iters.
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
		// F4 halving floor (2026-10-24 tile-disable ruling): the halving
		// NEVER writes an override below 2 — when the picked site's halved
		// covered dim would compute next < 2, the SITE IS DROPPED to GMEM
		// (never demoted to a non-tiling schedule) and the outer loop
		// continues.
		floorHit := false
		for _, it := range pick.Covered {
			if dec(it).TileSize/2 < 2 {
				floorHit = true
				break
			}
		}
		if floorHit {
			dropped[pick.Load.BasePtr().ID()] = true
			continue
		}
		for _, it := range pick.Covered {
			overrides[it] = dec(it).TileSize / 2
		}
		// F4 sweep: ANY covered iter at tile <= 1 drops the site. The
		// halving floor above owns this drop now (belt-and-braces over
		// shared covered iters); facts order, deterministic.
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

	// Tier re-derivation (R2 + C1b, ruled 2026-10-21/23): classify by the SoC
	// of the COVERED dims — ANY CTA-covered dim REFUSES the site (no coherent
	// fill variant under the v0 mechanics; a promoted record entry saying
	// otherwise is corruption) — with the C1b PRECEDENCE: a THREAD-VARYING
	// non-covered source coordinate (any INS iter not covered, assigned CTA
	// or WARP) forces ClassFRAG regardless of the covered dims' SoC (a
	// cooperative SMEM fill is CTA-uniform, R1; the judge applies the same
	// predicate — the record must agree). Otherwise any WARP-covered dim
	// makes the SMEM window class, all-sequential makes the FRAG class.
	for _, d := range covered {
		if d.soc == SoCTA {
			return fmt.Errorf("tile-lower: site %d covers CTA-assigned iter node %d — a CTA-covered tile has no coherent fill variant and is refused (R2 tier rule; the record's promoted entry is corrupted)",
				id, d.iter.BasePtr().ID())
		}
	}
	forcedFrag := false
	for _, it := range sf.INS {
		if coveredSet[it] {
			continue
		}
		d, ok := lr.byIter[it]
		if !ok {
			return fmt.Errorf("tile-lower: site %d references iter node %d in its INS, which the rewrite did not produce (corrupted facts?)", id, it.BasePtr().ID())
		}
		if d.soc == SoCTA || d.soc == SoCWARP {
			forcedFrag = true // C1b: a thread-varying non-covered coordinate
			break
		}
	}
	class := ClassFRAG
	if !forcedFrag {
		for _, d := range covered {
			if d.soc == SoCWARP {
				class = ClassSMEM
				break
			}
		}
	}
	if class != sd.Class {
		return fmt.Errorf("tile-lower: site %d record class %s disagrees with the R2/C1b tier rule over its covered dims (%s) — record corruption",
			id, sd.Class, class)
	}

	// Covered dims keep T >= 2 (the judge's promotion gate + the F4 halving
	// floor; an inside-less covered dim reaching surgery is corruption —
	// every dim builds an inside loop post-ruling).
	for _, d := range covered {
		if d.insideIV == nil {
			return fmt.Errorf("tile-lower: site %d covers iter node %d at tile size %d (covered dims must keep T >= 2; a missing inside walk is rewrite corruption)", id, d.iter.BasePtr().ID(), d.tile)
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

	// Buffer: the fix4 §2.4/§3.3 WINDOW model (R-C1B) — per covered dim,
	// the SAME RAW factors the decide side's shared siteWindowBytesRaw
	// prices (the fill/buffer size — the actual allocation, NOT the
	// budget chunk: fix6 §3 prices the budget as min(raw, MaxFragPerThread)
	// via siteWindowBytes, but the over-budget remainder still fills and
	// reads — one budget, both halves; a divergence would break the
	// fail-closed record agreement): a WARP-covered dim contributes K_d x
	// T_d under the SMEM class (one window per warp coordinate — 32x
	// smaller than the old whole-group shard window) and ceil(T_d/32)
	// under a C1b forced-FRAG site (the lane's consecutive chunk); SEQ
	// dims contribute T_d either way.
	// The tile TENSOR TYPE binds via SetType — the typeinfer MemLoadSMem rule's
	// own message names this pass as the binder ("cache lowering must bind
	// buffer types first"), the one documented exception to P8's
	// no-self-typing.
	bytes := elemBytes(elem)
	dims := make([]ir.DimRef, 0, len(covered))
	for _, d := range covered {
		extent := d.tile
		if d.soc == SoCWARP {
			if class == ClassSMEM {
				extent = d.loc * d.tile // K_d windows of T_d each (rank-in-index)
			} else {
				extent = ceilDiv(d.tile, 32) // C1b: the lane's chunk (§3.3)
			}
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
	//
	// C1b (2026-10-23, corrected quick-261027-kzv): a FRAG fill whose covered
	// set includes a WARP dim rides that position's coordinate on the AXES
	// path as the dim's LANE-CHUNK window base — (rank·T + round) +
	// lane·ceil(T_d/32) — so the per-thread loader fills THIS LANE's chunk
	// (the biv path cannot carry the per-thread terms; stmtFill's bivDim
	// demands a dimvar). A SEQ-covered dim keeps the biv path (its round
	// base IS a dimvar).
	origAxes := ir.OperandsOf(sf.Load)[1:]
	fillAxes := make([]ir.Node, 0, len(origAxes))
	for pos, axis := range origAxes {
		if pos >= len(sf.AxesIters) || len(sf.AxesIters[pos]) == 0 {
			continue // unreferenced position: absent
		}
		coveredHere, aboveHere, warpCoveredHere := false, false, false
		for _, it := range sf.AxesIters[pos] {
			if coveredSet[it] {
				coveredHere = true
				if d, ok := lr.byIter[it]; ok && d.soc == SoCWARP {
					warpCoveredHere = true
				}
			} else {
				aboveHere = true // INS at or outer than the fill key
			}
		}
		switch {
		case coveredHere && !aboveHere && !(class == ClassFRAG && warpCoveredHere):
			continue // the loader enumerates the covered tile dim (biv path)
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
		case coveredHere:
			// FRAG: covered dims ride their round bases (a WARP-covered
			// position carries the per-thread chunk origin rank*T + round;
			// a mixed position keeps the above-tie dims' full index
			// expressions — the fill key dim's inside coordinate
			// re-executes the fill per epoch).
			rebuilt, n := lr.rebuildAxis(axis, lr.coveredBaseSubs(covered, class))
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

	// Fill store (§6's tier rule, fix3 D-h): a plain tiered MemStore of the
	// tile load — SMEM window class → the cooperative LoadTileCta, FRAG
	// class → the per-thread LoadTileThread, DERIVED at codegen from the
	// class (R-H/R-J; the retired v0-era loader attribute left the IR). The
	// covered round bases ride the store's trailing biv operands:
	// between dimvars only — uniform for the cooperative loader (R1/R2). A
	// C1b forced-FRAG site's WARP-covered dim is EXCLUDED — its lane-chunk
	// window base ((rank·T + round) + lane·c) carries the per-thread terms
	// and already rides the fill's axes path (stmtFill's bivDim demands
	// dimvars).
	variant := loadTileThread
	if class == ClassSMEM {
		variant = loadTileCta
	}
	roundBases := make([]ir.Node, 0, len(covered))
	for _, d := range covered {
		if class == ClassFRAG && d.soc == SoCWARP {
			continue
		}
		roundBases = append(roundBases, d.betweenIV)
	}
	var fillStore ir.Node
	if class == ClassFRAG {
		fillStore = lr.g.NewFragStore(lowerOrigin, src, buffer, roundBases...)
	} else {
		fillStore = lr.g.NewMemStoreSMem(lowerOrigin, src, buffer, roundBases...)
	}

	// Access: every use of the erased load reads the buffer at the covered
	// dims' in-tile coordinates (M4) — R2 rank-in-index: a WARP-covered
	// dim's index is thread_map.index(d)·T_d + i (d = the dim's ThreadMap
	// slot — one map per LNT, tuple coordinates, §7bis — NOT a hardcoded 0
	// and NOT the bare inside coordinate) for the SMEM window class. A C1b
	// forced-FRAG site is a per-thread window of the LANE'S CHUNK — the
	// buffer extent on a WARP-covered dim is ceil(T_d/32) (§3.3), while the
	// lane-chunk walk IV ranges [lane·c, (lane+1)·c): the access must be
	// CHUNK-LOCAL, insideIV − lane·c — the raw walk IV indexes lanes past
	// the fragment's extent (GPU-proven 2026-10-27: memcheck "Invalid
	// __local__ read" on the golden matmul, every lane ≥ 1). SEQ dims index
	// by their plain inside coordinate either way (a SEQ walk starts at 0 —
	// already local). The accesses chain after the fill on the buffer's own
	// effect chain — the ORDINARY memory chain (C7, fix3 D-i: no special
	// fill-edge reading) — so the fill dominates its readers in the sea's
	// explicit effect order and stays reachable through them.
	coords := make([]ir.Node, 0, len(covered))
	for _, d := range covered {
		switch {
		case d.soc == SoCWARP && class == ClassSMEM:
			rank := lr.g.NewRuntimeThreadQuery(lowerOrigin, "index", lr.u32c(float64(d.threadDim)))
			shifted := lr.g.NewBinOp(lowerOrigin, ir.BinOpMul, rank, lr.u32c(float64(d.tile)))
			coords = append(coords, lr.g.NewBinOp(lowerOrigin, ir.BinOpAdd, shifted, d.insideIV))
		case d.soc == SoCWARP && class == ClassFRAG:
			// C1b chunk-local: the fragment holds only this lane's chunk,
			// so subtract the chunk origin lane·ceil(T_d/32) (the same
			// lane·c term the walk IV's init carries — fix4 §3.1's chunk
			// arithmetic).
			c := ceilDiv(d.tile, 32)
			lane := lr.g.NewRuntimeThreadQueryLane(lowerOrigin)
			origin := lr.g.NewBinOp(lowerOrigin, ir.BinOpMul, lane, lr.u32c(float64(c)))
			coords = append(coords, lr.g.NewBinOp(lowerOrigin, ir.BinOpSub, d.insideIV, origin))
		default:
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

	// Access rewrite: RAUW every data use, then retire the load. Sea-of-
	// nodes (fix1.md §0): there is no body operand — the RAUW simply moves
	// every data use of the erased load onto the buffer access; placement
	// rides the per-buffer effect chain, not a body slot.
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

// fillHost returns the fill-position loop — the R1-era rule, retained
// verbatim through fix3 D-j so placement does not move before the
// derivation lands (R-I: the same position is what the derivation picks;
// the suite's assertFillPositionDerived pins that equality):
//
//   - ClassSMEM (cooperative LoadTileCta): the INNERMOST covered dim's
//     between-round loop — the fill runs once per round of that dim,
//     immediately before the tile-inside loops, with CTA-uniform
//     coordinates only.
//   - ClassFRAG (per-thread LoadTileThread): the INSIDE loop of the
//     INNERMOST above-tie INS dim (the fill key — the dim whose element
//     change invalidates the fragment, e.g. the worked example's lhs
//     refills per m-element inside M.inside); with no above-tie INS dim
//     the position degenerates to the SMEM rule (the innermost covered
//     dim's between loop).
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
		return hostDim.between, nil // every dim is tiled: between always exists
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
	// Post-ruling every dim builds an inside loop — a nil one is rewrite
	// corruption: fail closed (never fall back to the between loop).
	if keyDim.inside == nil {
		return nil, fmt.Errorf("fill-key dim (iter node %d) carries no inside loop (rewrite corruption; every dim is tiled, T >= 2)",
			keyDim.iter.BasePtr().ID())
	}
	return keyDim.inside, nil
}

// coveredBaseSubs maps every covered dim's rewritten index expression to
// its fill coordinate. A covered WARP dim under the FRAG class (C1b) rides
// its LANE-CHUNK window base — roundBase + lane·ceil(T_d/32): the per-
// thread loader fills this lane's chunk, and the bare warp origin would
// name the warp's first element for every lane (quick-261027-kzv T8b GPU
// defect). Every other dim rides its round base (the fill holds the whole
// tile; the loader walks the tile interior at runtime).
func (lr *lowerRun) coveredBaseSubs(covered []*dimRewrite, class CacheClass) map[*ir.Base]ir.Node {
	subs := make(map[*ir.Base]ir.Node, len(covered))
	for _, d := range covered {
		if class == ClassFRAG && d.soc == SoCWARP && d.chunkBase != nil {
			subs[d.indexExpr.BasePtr()] = d.chunkBase
			continue
		}
		subs[d.indexExpr.BasePtr()] = d.roundBase
	}
	return subs
}

// uniformBaseSubs maps EVERY dim of the LNT — covered and above-tie alike —
// to its CTA-UNIFORM base (R1): SEQ dims ride their between-round dimvar,
// full stop; a CTA dim rides its rank-offset round base (cta_rank·T + round
// — uniform across the CTA: the cooperative loader is called by every
// thread of the CTA, and they all read the same grid rank); a WARP dim
// rides its between dimvar when it has one — its rank-offset round base
// carries the per-thread ThreadMap term, which a cooperative fill must
// never see. The caller fails closed on above-tie WARP dims before
// substituting; a covered WARP dim always has T >= 2 (the surgery
// precondition), hence a between dimvar.
func (lr *lowerRun) uniformBaseSubs(lw *lntRewrite) map[*ir.Base]ir.Node {
	subs := make(map[*ir.Base]ir.Node, len(lw.dims))
	for _, d := range lw.dims {
		var base ir.Node
		switch d.soc {
		case SoCWARP:
			base = d.betweenIV
		default: // SEQ: between dimvar; CTA: rank offset
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
