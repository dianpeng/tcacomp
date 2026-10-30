// lower.go is the loop-tile-LOWER pass (quick-261021-jay,
// doc/optana/loop_tile_lower.md) — the lowering sequel to the decide pass
// (tiling.go): it consumes the tiling-record side table and performs the
// node surgery: every decided loop-nest tree rewrites into tile-between +
// tile-inside annotated ForLoop nests (PHASE 2-3, lower_rewrite.go), the
// decided MemLoad sites lower into SMEM/FRAG buffers + fills + buffer
// accesses (PHASE 4, lower_cache.go), and a pass-exit self-check sweep
// enforces the §8 exit conditions (PHASE 5, this file).
//
// Pipeline position: priority 116 — immediately after tiling (114) and
// tiling-verify (115), immediately before scheduling (doc/opt.md's "loop
// tiling rewrite" slot). Requires "tiling": the record-lifetime pin of
// loop_tile_lower.md §1 — the analysis cache is invalidate-all-by-default,
// so the record survives only while no pass runs between decide and lower.
//
// Preserved is EMPTY: the record describes pre-rewrite iters keyed by nodes
// this pass erases — it must not outlive the pass (the manager invalidates
// after Run).
//
// Choke-point discipline (§3): every graph edit rides MutContext
// (SetDef/AddUse/RemoveUse/ReplaceAllUsesWith/Erase); the ONLY construction
// time effect wiring is the ir package's own env-chained constructors (the
// NewMemLoadChained/NewMemStoreChained precedent, extended to the LIR
// family by the chained variants node_lir_memory.go gained in this task).
// The pass never emits a barrier (§6 sync: the runtime owns ALL
// synchronization) and never stamps types except the buffer dtype bindings
// the typeinfer rule's own message names ("cache lowering must bind buffer
// types first") — every other inserted node relies on the pipeline's
// typeinfer re-run (P8).
package tiling

import (
	"errors"
	"fmt"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// lowerOrigin is the PassOrigin stamp every node this pass constructs
// carries from birth (construction is exempt from MutContext but records
// PassOrigin, D-17).
const lowerOrigin = "tile-lower"

// init self-registers the pass: Priority 116 — doc/opt.md's "loop tiling
// rewrite" slot, immediately after tiling-verify (115). Requires BOTH
// tiling and tiling-verify (F7, ruled 2026-10-21): the per-iter feasibility
// re-checks live in tiling-verify, and a corrupted record must NOT reach
// the rewrite unvalidated — the pass-manager precondition (PASS-01) plus
// the record's Verified stamp (checked in Run, below) close the gate from
// both sides. The registry scale in internal/opt/registry.go documents the
// anchor.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "tile-lower",
		Priority: 116,
		Requires: []string{"tiling", "tiling-verify"},
		Factory:  func() opt.Pass { return LowerPass{Params: DefaultParams()} },
	})
}

// LowerPass is the loop-tile-lower pass. Params must mirror the decide
// pass's thresholds (the default factory pairs both on DefaultParams): the
// static strides and coupled thread counts derive from them.
type LowerPass struct {
	Params Params
}

// ID returns the pipeline name "tile-lower".
func (LowerPass) ID() string { return "tile-lower" }

// Requires: the decide pass AND its verification must have run earlier —
// the record this pass consumes is their product, the cache invalidation
// contract only keeps the record alive while nothing runs in between (§1),
// and F7 demands the record reach the rewrite validated.
func (LowerPass) Requires() []string { return []string{"tiling", "tiling-verify"} }

// Produces: builder-output stage — like every current pass; the new LIR
// kinds' catalog rows ARE their legality row today (P9: constructedWindow
// marks every AllKinds() kind Legal; the catalog row doc text carries the
// revision-point note).
func (LowerPass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: the tiling PROVENANCE record this pass publishes (P-5,
// quick-261022-l1o) — per-dim between/inside entries keyed by DIM
// InputParams and per-LNT mapping plans keyed by the GENERATED ForLoop
// roots, both stable across this pass and scheduling. The decision record
// itself ("tiling-record") deliberately DIES here: it is keyed by
// pre-rewrite BIV nodes this pass erases, so letting it survive would hand
// the next pass stale keys.
func (LowerPass) Preserved() []string { return []string{ProvenanceName} }

// Run lowers every decided LNT and every promoted site:
//
//   - GATES first (the tiling.go refusal pattern): a nil graph and a
//     materialized/CFG-carrying graph are deterministic errors evaluated
//     before anything is read; Params validate before any computation.
//   - PHASE 1: fetch the record FAIL CLOSED (§6bis: a nil context/cache or
//     a wrong-typed entry is an error, not a silent skip); re-derive the
//     loopforest, the side-effect analysis, and the overlap facts with the
//     SAME deterministic chase as the decide pass. The site facts are
//     derived HERE, pre-rewrite: the INS chase walks InductionVariable
//     nodes, which PHASE 3 retires — deriving them any later would chase an
//     already-rewritten graph (the spec's PHASE 4 ordering governs the
//     node SURGERY, not the fact derivation; adjacency makes the results
//     identical either way).
//   - BUDGET: resolve the SMEM budget against THIS pass's Params
//     (lower_cache.go) — with matching Params this is a no-op because
//     tiling-verify already asserted Σ ≤ MaxCache.
//   - PHASE 2-3: per LNT (Roots() order) the DoSoC numbering and the loop
//     rewrite (lower_rewrite.go).
//   - PHASE 4: per promoted site the buffer/fill/access surgery
//     (lower_cache.go).
//   - PHASE 5: the verifier plus the §8 self-check sweep (P7: the checks
//     are pass exit conditions, not a separate verify pass).
func (p LowerPass) Run(ctx *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("tile-lower: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("tile-lower: graph already materialized (tile-lower runs on the Level-1 sea)")
	}
	if err := p.Params.Validate(); err != nil {
		return fmt.Errorf("tile-lower: %w", err)
	}
	if ctx == nil || ctx.Cache == nil {
		return errors.New("tile-lower: no analysis cache (the tiling record is fetched through the cache)")
	}
	got := ctx.Cache.Get(RecordName)
	if got == nil {
		return errors.New("tile-lower: no tiling record (did the tiling pass run?)")
	}
	rec, ok := got.(*Record)
	if !ok {
		return fmt.Errorf("tile-lower: cache entry under %q is %T, want *Record", RecordName, got)
	}
	// F7 (ruled 2026-10-21): the record must carry tiling-verify's stamp —
	// the per-iter feasibility re-checks of the fix4 model (LoC preset
	// membership, the per-LNT factorization product bounds, the CTA-dim
	// presence, the tile floor, the kernel-wide budget sums, the judge
	// cross-check) live there, and an unvalidated record must not reach
	// the rewrite. Direct Run callers bypass the pass-manager
	// precondition, so the stamp is checked here too — fail closed.
	if !rec.Verified {
		return errors.New("tile-lower: the tiling record carries no tiling-verify stamp — run tiling-verify before tile-lower (F7: a corrupted record must not reach the rewrite unvalidated)")
	}

	f := loopforest.Analyze(g)
	se := sideeffect.Analyze(g)
	facts := analyzeOverlap(g, f, se, p.Params).Trees

	// Resolve every iter's decision fail-closed before any mutation
	// (adjacency guarantees completeness — a missing entry is corruption).
	decisions := make(map[ir.Node]IterDecision)
	for _, root := range f.Roots() {
		for _, it := range treeItersOf(root) {
			d, ok := rec.IterDecisionOf(it)
			if !ok {
				return fmt.Errorf("tile-lower: iter node %d has no record entry under its BIV", it.BasePtr().ID())
			}
			decisions[it] = d
		}
	}

	// Opt-out backstop (fix4 §3.5, Wave 1): when ReducedDimWarp is explicitly
	// false (the Wave-1 policy, now the opt-out mode — the Wave-2 default is
	// true), a WARP-assigned dim whose walk feeds a Reducer FAILS CLOSED —
	// the pipeline must stop loudly rather than emit a lowering outside the
	// policy. The decide-side feasible area prunes these under the same
	// flag; this backstop catches hand-stamped or corrupted records the way
	// the tile-floor gate 3 backstop catches tile 1.
	if !p.Params.ReducedDimWarp {
		if err := rejectWarpReducedDims(g, decisions); err != nil {
			return err
		}
	}
	// §3.1 fail-closed MIRROR (author ruling, 2026-10-30, iq6 candidate-1 —
	// unconditional, BOTH flag modes): a WARP-assigned dim that is NOT an
	// eliminated axis (a SURVIVING warp dim) is rejected before any
	// rewrite. GPU-proven on the R3 re-choice kernels (row_sum/softmax,
	// canary exactly 1/64 of the output space): the CTA-coupled-inside ×
	// lane-chunk composition covers only a diagonal of the cross product —
	// the iq6 §3.1 class. The decide-side §3.1 prune (warpDimOf) makes the
	// class unrepresentable; this mirror catches hand-stamped or corrupted
	// records. Lift BOTH fences when the cross-product lowering is designed.
	if err := rejectWarpSurvivingDims(g, decisions); err != nil {
		return err
	}

	// Launch-blockDim pre-pass (fix4 §3.2/§5.1): the launch blockDim is
	// the MAX over the forest's LNTs of (Π warp K_d) × 32 — never the
	// kernel-wide PRODUCT of every WARP LoC (two LNTs of K=2 and K=4
	// launch 128 threads, not 256). This REPLACES the per-call
	// blockDimOf(lr.decisions) the templates used to inline (the
	// kernel-wide product is wrong for multi-LNT graphs per §5.1);
	// blockDimOf itself stays in judge.go — the DECIDE side prices
	// per-LNT. The empty product is 1 → every LNT contributes at least 32
	// (a zero-WARP LNT launches one warp).
	launchBlockDim := 32
	for _, root := range f.Roots() {
		prod := 1
		for _, it := range treeItersOf(root) {
			if d := decisions[it]; d.SoC == SoCWARP {
				prod *= d.LoC
			}
		}
		if bd := prod * 32; bd > launchBlockDim {
			launchBlockDim = bd
		}
	}

	overrides, dropped, err := p.resolveBudget(rec, facts, decisions)
	if err != nil {
		return err
	}

	mctx := &ir.MutContext{Pass: lowerOrigin}
	lr := &lowerRun{
		g:              g,
		params:         p.Params,
		mctx:           mctx,
		decisions:      decisions,
		overrides:      overrides,
		byIter:         make(map[ir.Node]*dimRewrite),
		launchBlockDim: launchBlockDim,
		prov: &Provenance{
			Dims: make(map[ir.Node]DimProvenance),
			LNTs: make(map[ir.Node]MappingPlan),
		},
	}
	lr.prov.MaxThreadsPerCTA = launchBlockDim
	for _, root := range f.Roots() {
		if err := lr.rewriteLNT(root); err != nil {
			return err
		}
	}
	if err := lr.lowerSites(rec, facts, dropped); err != nil {
		return err
	}
	if err := lr.epilogue(rec, dropped); err != nil {
		return err
	}
	// P-5 (quick-261022-l1o): publish the tiling provenance AFTER the
	// epilogue succeeds — the per-dim between/inside entries and the
	// per-LNT mapping plans final-lower consumes. Preserved keeps it alive
	// across this pass's own invalidation and scheduling.
	if ctx.Cache == nil {
		return errors.New("tile-lower: no analysis cache to publish the tiling provenance under")
	}
	ctx.Cache.Put(lr.prov)
	return nil
}

// lowerRun carries one Run's rewrite state: the pass thresholds, the
// choke-point context, the effective per-iter decisions (record decision
// plus any budget override), the derived LAUNCH blockDim (fix4 §5.1: max
// over LNTs of (Π warp K) × 32 — the coupled walks' stride and the
// provenance's MaxThreadsPerCTA), the per-dim rewrite products PHASE 4
// consumes (filled by rewriteLNT), and the provenance record under
// construction (P-5).
type lowerRun struct {
	g              *ir.Graph
	params         Params
	mctx           *ir.MutContext
	decisions      map[ir.Node]IterDecision
	overrides      map[ir.Node]int // budget tile overrides, keyed by iter
	launchBlockDim int             // max over LNTs of (Π warp K) × 32, floor 32
	byIter         map[ir.Node]*dimRewrite
	rewrites       []*lntRewrite // per-root products, Roots() order
	fills          []*siteFill   // PHASE 4 products, for the §8 placement/scope checks
	prov           *Provenance   // the P-5 provenance under construction
}

// rejectWarpSurvivingDims is the §3.1 fail-closed mirror (author ruling,
// 2026-10-30, iq6 candidate-1): in a REDUCE-CARRYING graph, every
// WARP-assigned iter that is NOT an ELIMINATED reduction axis
// (reducedAxesOf — feeds a Reduce's value, never a store subscript) fails
// closed with a named error. A WARP digit on a SURVIVING output axis
// lowers through the CTA-coupled-inside × lane-chunk composition whose
// cross product is 63/64 UNCOVERED — GPU-proven on the R3 re-choice
// kernels (row_sum/softmax: canary exactly 1/64 of the output space; the
// iq6 §3.1 class "every (m,n) pair carries 16/1024 k values").
//
// Scope — reduce-carrying graphs ONLY (both ReducedDimWarp modes): every
// REAL v0 kernel reduces (the chain shape is map|reduce[|map]|store), so
// the mirror binds 100% of kernels that reach codegen. A crafted
// no-Reduce template graph (the lower-test corpus' mechanics fixtures)
// never occurs in the pipeline and carries no output-axis reduction to
// protect — the mirror stays silent there, leaving the template pins
// intact for the future §3.1 lowering's machinery tests. The decide-side
// §3.1 prune (warpDimOf) makes the class unrepresentable; this mirror
// catches hand-stamped or corrupted records the way the tile-floor gate
// catches tile 1. Lift BOTH fences when the tile-lower cross-product
// lowering is designed.
func rejectWarpSurvivingDims(g *ir.Graph, decisions map[ir.Node]IterDecision) error {
	eliminated := reducedAxesOf(g)
	if len(eliminated) == 0 {
		// Also short-circuit when the graph carries NO live Reduce at all
		// (see the scope note): nothing real to protect.
		hasReduce := false
		for _, n := range g.AllocatedNodes() {
			if n != nil && n.BasePtr() != nil && !n.BasePtr().Erased() && n.KindOf() == ir.KindReduce {
				hasReduce = true
				break
			}
		}
		if !hasReduce {
			return nil
		}
	}
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindIter {
			continue
		}
		d, ok := decisions[n]
		if !ok || d.SoC != SoCWARP || eliminated[n] {
			continue
		}
		return fmt.Errorf(
			"tile-lower: WARP-assigned iter node %d is a SURVIVING (non-eliminated) axis — the §3.1 CTA-coupled-inside × lane-chunk composition covers only a diagonal of the cross product (GPU-proven 1/64 coverage on row_sum/softmax, the iq6 class). Surviving-WARP vectors are pruned from the decide-side feasible area (warpDimOf §3.1 prune); design the tile-lower cross-product lowering before lifting either fence",
			n.BasePtr().ID())
	}
	return nil
}

// rejectWarpReducedDims is the fix4 §3.5 Wave-1 defense: every
// WARP-assigned iter whose InductionVariable users' forward data-dep
// closure reaches a live KindReduce fails closed with a named error. The
// walk is over USES (forward): the IV feeds a load, the load the piped
// Map, the Map the Reduce — the dim's walk demonstrably accumulates into
// a reducer, which under WARP factorization is the undesigned split-k
// shape (Wave 2, behind ReducedDimWarp=true).
func rejectWarpReducedDims(g *ir.Graph, decisions map[ir.Node]IterDecision) error {
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindIter {
			continue
		}
		if d, ok := decisions[n]; !ok || d.SoC != SoCWARP {
			continue
		}
		seen := make(map[*ir.Base]bool)
		stack := make([]ir.Node, 0, 8)
		for _, iv := range ivsOf(n) {
			stack = append(stack, iv)
		}
		for len(stack) > 0 {
			x := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if x == nil || x.BasePtr() == nil || x.BasePtr().Erased() || seen[x.BasePtr()] {
				continue
			}
			seen[x.BasePtr()] = true
			if x.KindOf() == ir.KindReduce {
				return fmt.Errorf(
					"tile-lower: WARP-assigned iter node %d feeds Reduce node %d — a WARP-factorized reduced dim is the split-k shape, undesigned in Wave 1 (fix4 3.5): the decide-side feasible-area gate prunes these while ReducedDimWarp=false, and this lower must stop loudly rather than emit an undesigned lowering",
					n.BasePtr().ID(), x.BasePtr().ID())
			}
			for _, u := range ir.UsesOf(x) {
				if usr := u.User(); usr != nil {
					stack = append(stack, usr)
				}
			}
		}
	}
	return nil
}

// effectiveTile returns the iter's tile size after budget overrides.
func (lr *lowerRun) effectiveTile(iter ir.Node) int {
	if t, ok := lr.overrides[iter]; ok {
		return t
	}
	return lr.decisions[iter].TileSize
}

// epilogue is PHASE 5: the verifier plus the §8 self-check sweep. Every
// check is read-only over the finished graph, deterministic order; any
// violation is a named pass error (P7 — the checks are pass exit
// conditions, not a separate verify pass).
func (lr *lowerRun) epilogue(rec *Record, dropped map[uint64]bool) error {
	if vs := ir.Verify(lr.g); len(vs) != 0 {
		return fmt.Errorf("tile-lower: verifier reported %d violation(s) after the rewrite: %s",
			len(vs), vs[0])
	}
	if err := lr.checkRetiredIters(); err != nil {
		return err
	}
	if err := lr.checkLoopAnnotations(); err != nil {
		return err
	}
	if err := lr.checkFillPlacement(); err != nil {
		return err
	}
	if err := lr.checkNoStaleLoads(rec, dropped); err != nil {
		return err
	}
	if err := lr.checkNoBarriers(); err != nil {
		return err
	}
	return lr.checkFragSites(rec, dropped)
}

// checkRetiredIters pins the §8 exit condition "no original
// InductionVariable use survives": after a full rewrite NO live iter or
// induction-variable node may remain anywhere in the graph.
func (lr *lowerRun) checkRetiredIters() error {
	for _, n := range lr.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		switch n.KindOf() {
		case ir.KindIter, ir.KindInductionVariable:
			return fmt.Errorf("tile-lower: node %d (%s) survived the rewrite (stale original-IV use?)",
				n.BasePtr().ID(), n.KindOf())
		}
	}
	return nil
}

// checkLoopAnnotations pins "every parallel loop carries the
// materialization annotation (thr/cta)": every generated loop's annotation
// set is re-read from the dimRewrite products and asserted well-formed —
// soc and loc present; dosoc present exactly where the fix4 §3.1/§3.2
// conventions demand it: on EVERY between loop of a parallel dim
// (soc != 0), but EXEMPT on the CTA dim's coupled INSIDE walk (soc 2, no
// dosoc — the flat-rank walk rides the NO-ARG thread index and needs no
// ThreadMap slot, the §3.2 dimensionality ruling). The pass knows each
// loop's role from its dimRewrite (between vs inside), so the exemption
// is precise, never a blanket soc-based skip.
func (lr *lowerRun) checkLoopAnnotations() error {
	for _, lw := range lr.rewrites {
		for _, d := range lw.dims {
			check := func(loop ir.Node, isBetween bool) error {
				attrs := loopAnnotations(loop)
				soc, ok := attrs["soc"]
				if !ok {
					return fmt.Errorf("tile-lower: generated loop node %d carries no soc annotation", loop.BasePtr().ID())
				}
				if _, ok := attrs["loc"]; !ok {
					return fmt.Errorf("tile-lower: generated loop node %d carries no loc annotation", loop.BasePtr().ID())
				}
				// dosoc: required on soc!=0 BETWEEN loops and on any soc==1
				// loop; exempt on the soc==2 coupled INSIDE walk and on
				// every sequential loop.
				needDosoc := (isBetween && soc != 0) || soc == 1
				if _, has := attrs["dosoc"]; needDosoc && !has {
					return fmt.Errorf("tile-lower: generated parallel loop node %d carries no dosoc annotation",
						loop.BasePtr().ID())
				}
				return nil
			}
			if err := check(d.between, true); err != nil {
				return err
			}
			if d.inside != nil {
				if err := check(d.inside, false); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// loopAnnotations collects a generated for-loop's KeyedAttribute inputs by
// key (u32-const values). The pass constructs at most one attribute per
// key per loop. Operand layout is [iv, bound, attrs...] (sea-of-nodes,
// fix1.md §0 — no body operand); attrs start at index 2.
func loopAnnotations(loop ir.Node) map[string]float64 {
	out := make(map[string]float64)
	ops := ir.OperandsOf(loop)
	for i := 2; i < len(ops); i++ {
		op := ops[i]
		if op == nil || op.KindOf() != ir.KindKeyedAttribute {
			continue
		}
		key, val := ir.KeyedAttributeOf(op)
		if key == "" {
			continue
		}
		if v, ok := ir.ConstValueOf(val); ok {
			out[key] = v
		}
	}
	return out
}

// checkFillPlacement pins the §8 fill exit conditions STRUCTURALLY (F5,
// ruled 2026-10-21: every check is computed from the emitted graph and must
// be able to fail on a plausibly-emitted graph). Post-fix1.md (D-e), the
// checks are recast around the sea-of-nodes model: no body operand, no
// body-reachability. Per emitted fill:
//
//   - ORDERING: every live buffer access of the site has the fill among
//     its chain predecessors (ImpactBy walk). The fill dominates its reads
//     in the sea's explicit effect order — this is THE structural
//     placement fact (the F8 splice wires it). A fill whose buffer has NO
//     live reads passes vacuously: it is unconsumed dead code (§0),
//     memopt's fill-liveness domain (D-f), not a placement violation.
//   - LAYERING: a cooperative LoadTileCta fill's operand-closure
//     references NO inside-loop IV — the HI-02 class: a cooperative loader
//     must be callable by every thread of the CTA the same number of
//     times, which an element-loop (inside-walk) position breaks (§4).
//     Per-thread LoadTileThread fills are exempt (R1: they ARE element-
//     level ops).
//   - SCOPE: every loop dimvar reachable from the fill's coordinates (the
//     source load's axes and the fill store's round bases) is defined by a
//     loop that encloses the fill's host — no fill reads a loop dimvar
//     from outside that loop (the CR-02/HI-01 leakage class). Enclosure
//     reads the pass-authored canonical chain (the "nest" annotations):
//     a host's own dimvar is in scope (the fill re-executes per the host's
//     epoch).
//   - UNIFORMITY: a LoadTileCta fill's coordinates carry NO ThreadMap
//     query — a per-thread term in a cooperative fill's source base is
//     incoherent (each participating thread would compute a different
//     source region into the same buffer, HI-03). GridMap queries are
//     CTA-uniform and allowed.
//
// EPOCH is no longer a separate check — the fill's coordinates are the
// promoted load's own axes by construction, so epoch membership is
// inherent (D-e).
func (lr *lowerRun) checkFillPlacement() error {
	// dimvar -> defining loop, over every rewritten dim.
	ivLoop := make(map[*ir.Base]ir.Node)
	// insideIVs collects every inside (tile-walk) dimvar — the LAYERING set.
	insideIVs := make(map[*ir.Base]bool)
	// nestPos stamps every generated loop with its position in the pass-
	// authored canonical chain (betweens in dim order, then insides in dim
	// order) and the chain's identity — the sea-of-nodes "encloses" relation
	// for the SCOPE check. The pass built the nest, so it need not re-derive
	// enclosure from closures: loop L encloses host iff both ride the same
	// chain and L's position is at or above the host's. A foreign chain's
	// loops never enclose (fail-closed: a fill references only its own
	// nest's loops by the F8 construction).
	type nestPos struct {
		chain *lntRewrite
		pos   int
	}
	positions := make(map[*ir.Base]nestPos)
	for _, lw := range lr.rewrites {
		pos := 0
		// The canonical chain order buildNest wires: ALL betweens in dim
		// order, then ALL insides in dim order.
		for _, d := range lw.dims {
			if d.between != nil {
				positions[d.between.BasePtr()] = nestPos{lw, pos}
				pos++
			}
		}
		for _, d := range lw.dims {
			// Skip collapsed dims (between == inside, one shared node —
			// already stamped in the between pass above).
			if d.inside == nil || (d.between != nil && d.inside.BasePtr() == d.between.BasePtr()) {
				continue
			}
			positions[d.inside.BasePtr()] = nestPos{lw, pos}
			pos++
		}
	}
	for _, lw := range lr.rewrites {
		for _, d := range lw.dims {
			if d.betweenIV != nil {
				ivLoop[d.betweenIV.BasePtr()] = d.between
			}
			if d.insideIV != nil {
				ivLoop[d.insideIV.BasePtr()] = d.inside
				insideIVs[d.insideIV.BasePtr()] = true
			}
		}
	}
	// encloses reports whether loop encloses host per the pass-authored
	// chain positions. A loop always encloses itself (same position).
	encloses := func(loop, host ir.Node) bool {
		if loop == nil || host == nil || loop.BasePtr() == nil || host.BasePtr() == nil {
			return false
		}
		lp, okL := positions[loop.BasePtr()]
		hp, okH := positions[host.BasePtr()]
		return okL && okH && lp.chain == hp.chain && lp.pos <= hp.pos
	}
	for _, fill := range lr.fills {
		// ORDERING: every live buffer access of the site has the fill
		// among its chain predecessors (ImpactBy walk). The buffer access
		// was RAUW'd into every data use of the promoted load; each such
		// access chains after the fill via its ImpactBy edge.
		if err := lr.checkFillOrdering(fill); err != nil {
			return err
		}
		// LAYERING: cooperative fills reference no inside-loop IV in their
		// operand closure (D-e recast — the old body-subtree check is
		// deleted with the body operand).
		if fill.variant == loadTileCta {
			fillClosure := flattenSubtree(fill.store)
			for _, n := range fillClosure {
				if insideIVs[n.BasePtr()] {
					return fmt.Errorf("tile-lower: cooperative fill for site %d references inside-loop dimvar node %d (§4 layering: a LoadTileCta op must be callable uniformly by the whole CTA — its coordinates must not reference an element-loop IV)",
						fill.siteID, n.BasePtr().ID())
				}
			}
		}
		// SCOPE + UNIFORMITY over the fill's coordinates: the source
		// load's axes and the fill store's trailing round bases.
		coords := append(append([]ir.Node(nil), ir.OperandsOf(fill.src)[1:]...), ir.OperandsOf(fill.store)[2:]...)
		for _, coord := range coords {
			for _, n := range flattenSubtree(coord) {
				if def, ok := ivLoop[n.BasePtr()]; ok && !encloses(def, fill.host) {
					return fmt.Errorf("tile-lower: fill for site %d references dimvar node %d of loop %d, which does not enclose its fill position (host loop %d) — a fill may only reference dimvars of enclosing loops",
						fill.siteID, n.BasePtr().ID(), def.BasePtr().ID(), fill.host.BasePtr().ID())
				}
				if fill.variant == loadTileCta && n.KindOf() == ir.KindRuntimeThreadQuery {
					return fmt.Errorf("tile-lower: cooperative fill for site %d carries a ThreadMap (per-thread) term in its coordinates — LoadTileCta source bases must be CTA-uniform (R1/HI-03)",
						fill.siteID)
				}
			}
		}
	}
	return nil
}

// checkFillOrdering verifies the §8 ORDERING fact (fix1.md D-e): every
// live buffer access of the site has the fill among its chain predecessors.
// The access set is derived from the fill's buffer: every FragLoad/
// MemLoadSMem that reads the fill's buffer must chain back to the fill's
// store through ImpactBy edges. Vacuous when the buffer has no live reads —
// an unconsumed fill is ordinary dead code (§0), removed by memopt's
// fill-liveness elimination (D-f), not an §8 placement violation.
func (lr *lowerRun) checkFillOrdering(fill *siteFill) error {
	// chainPreds collects every chain predecessor of n via ImpactBy walk.
	chainPreds := func(n ir.Node) map[*ir.Base]bool {
		seen := make(map[*ir.Base]bool)
		var visit func(x ir.Node)
		visit = func(x ir.Node) {
			if x == nil || x.BasePtr() == nil || seen[x.BasePtr()] || x.BasePtr().Erased() {
				return
			}
			seen[x.BasePtr()] = true
			for _, u := range ir.ImpactBy(x) {
				visit(u.Def())
			}
		}
		visit(n)
		return seen
	}
	// Every live tiered buffer access (SMEM/FRAG-class MemLoad, fix3 D-h)
	// on the fill's buffer must have fill.store among its chain
	// predecessors. No live access at all: vacuous pass (the fill is
	// unconsumed dead code — memopt's fill-liveness removal (D-f), not an
	// §8 violation).
	for _, n := range lr.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		if n.KindOf() != ir.KindMemLoad {
			continue
		}
		if sc, _ := ir.StorageClassOf(n); sc == ir.StorageGMEM {
			continue
		}
		// Confirm this access reads the fill's buffer.
		ops := ir.OperandsOf(n)
		if len(ops) == 0 || ops[0] == nil || ops[0].BasePtr() == nil ||
			fill.buffer == nil || fill.buffer.BasePtr() == nil ||
			ops[0].BasePtr() != fill.buffer.BasePtr() {
			continue
		}
		if !chainPreds(n)[fill.store.BasePtr()] {
			return fmt.Errorf("tile-lower: fill for site %d is not a chain predecessor of buffer access node %d (ORDERING: the F8 splice must wire the fill before every buffer access of the site via ImpactBy edges)",
				fill.siteID, n.BasePtr().ID())
		}
	}
	return nil
}

// operandSubtreeOf returns the operand-only descendant set of n (DataDeps
// walk, no payload/effect edges): the nodes n's operand slots reference,
// transitively. Post-fix1.md this is a generic operand-closure helper (no
// body-slot semantics); tests use it to confirm placement facts.
func (lr *lowerRun) operandSubtreeOf(n ir.Node) map[*ir.Base]bool {
	seen := make(map[*ir.Base]bool)
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || seen[x.BasePtr()] || x.BasePtr().Erased() {
			return
		}
		seen[x.BasePtr()] = true
		for _, dep := range ir.DataDeps(x) {
			visit(dep)
		}
	}
	visit(n)
	return seen
}

// flattenSubtree returns n's data-subtree nodes (self first) — the shared
// coordinate-walk helper of the §8 checks and the tests.
func flattenSubtree(n ir.Node) []ir.Node {
	var out []ir.Node
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || x.BasePtr().Erased() {
			return
		}
		out = append(out, x)
		for _, dep := range ir.DataDeps(x) {
			visit(dep)
		}
	}
	visit(n)
	return out
}

// checkNoStaleLoads pins "no stale MemLoad remains for any promoted site":
// every promoted (non-dropped) site's load is erased.
func (lr *lowerRun) checkNoStaleLoads(rec *Record, dropped map[uint64]bool) error {
	for id, sd := range rec.Sites {
		if !sd.Promoted || dropped[id] {
			continue
		}
		for _, n := range lr.g.AllocatedNodes() {
			if n == nil || n.BasePtr() == nil || n.BasePtr().ID() != id {
				continue
			}
			if !n.BasePtr().Erased() {
				return fmt.Errorf("tile-lower: promoted site %d survives un-erased (stale MemLoad)", id)
			}
		}
	}
	return nil
}

// checkNoBarriers pins "ZERO barriers emitted (§6 sync)" as a REAL check
// (F5/ME-04, ruled 2026-10-21 — a check that re-reads a baked constant is
// vacuous and forbidden): the pass constructs no barrier-emitting node
// (v0's kind suite has no sync kinds — the constructive half), and the
// operative half re-reads every emitted fill store's DERIVATION FACTS —
// since fix3 D-h the loader selection is DERIVED at codegen from the fill's
// storage class + composition shape (R-H/R-J; the retired v0-era
// attribute left the IR), so the check pins exactly those facts: the store
// is a TIERED MemStore (SMEM/FRAG class) whose value operand is a GMEM
// MemLoad — a fill that lost either half must fail the pass, not surface
// at codegen time.
func (lr *lowerRun) checkNoBarriers() error {
	for _, fill := range lr.fills {
		sc, ok := ir.StorageClassOf(fill.store)
		if fill.store.KindOf() != ir.KindMemStore || !ok || sc == ir.StorageGMEM {
			return fmt.Errorf("tile-lower: fill for site %d is not a tiered (SMEM/FRAG-class) MemStore — the codegen loader derivation has no class to read (§6, fix3 R-H)",
				fill.siteID)
		}
		ops := ir.OperandsOf(fill.store)
		if len(ops) == 0 || ops[0] == nil || ops[0].KindOf() != ir.KindMemLoad {
			return fmt.Errorf("tile-lower: fill for site %d stores no GMEM tile load — the codegen loader derivation has no source (§6, fix3 R-H)",
				fill.siteID)
		}
		if lsc, _ := ir.StorageClassOf(ops[0]); lsc != ir.StorageGMEM {
			return fmt.Errorf("tile-lower: fill for site %d stores a non-GMEM load — the loader call's source must be the tile-shaped GMEM MemLoad (§6, fix3 R-H)",
				fill.siteID)
		}
	}
	return nil
}

// checkFragSites pins "every FRAG site's covered dims are never
// CTA-assigned": sequential (the classic fully-unroll class) or WARP under
// the C1b precedence (2026-10-23 — a thread-varying non-covered coordinate
// forces FRAG; the per-thread window indexes the WARP-covered dim by its
// plain inside coordinate). A CTA-covered dim is refused at tier time and
// must never reach a FRAG record entry.
func (lr *lowerRun) checkFragSites(rec *Record, dropped map[uint64]bool) error {
	for id, sd := range rec.Sites {
		if !sd.Promoted || dropped[id] || sd.Class != ClassFRAG {
			continue
		}
		// The per-site covered-iter check rides the siteFill products.
		for _, fill := range lr.fills {
			if fill.siteID != id {
				continue
			}
			for _, d := range fill.covered {
				if lr.decisions[d.iter].SoC == SoCTA {
					return fmt.Errorf("tile-lower: FRAG site %d covers CTA-assigned iter node %d (the R2 refusal; a CTA-covered record entry is corruption)",
						id, d.iter.BasePtr().ID())
				}
			}
		}
	}
	return nil
}
