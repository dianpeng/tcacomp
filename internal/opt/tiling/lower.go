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
// rewrite" slot, immediately after tiling-verify (115). The registry scale
// in internal/opt/registry.go documents the anchor.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "tile-lower",
		Priority: 116,
		Requires: []string{"tiling"},
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

// Requires: the decide pass must have run earlier — the record this pass
// consumes is its product, and the cache invalidation contract only keeps
// the record alive while nothing runs in between (§1).
func (LowerPass) Requires() []string { return []string{"tiling"} }

// Produces: builder-output stage — like every current pass; the new LIR
// kinds' catalog rows ARE their legality row today (P9: constructedWindow
// marks every AllKinds() kind Legal; the catalog row doc text carries the
// revision-point note).
func (LowerPass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: NOTHING — the record is keyed by pre-rewrite BIV nodes this
// pass erases; letting it survive would hand the next pass stale keys.
func (LowerPass) Preserved() []string { return nil }

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

	overrides, dropped, err := p.resolveBudget(rec, facts, decisions)
	if err != nil {
		return err
	}

	mctx := &ir.MutContext{Pass: lowerOrigin}
	lr := &lowerRun{
		g:         g,
		params:    p.Params,
		mctx:      mctx,
		decisions: decisions,
		overrides: overrides,
		byIter:    make(map[ir.Node]*dimRewrite),
		claimed:   make(map[*ir.Base]bool),
	}
	for _, root := range f.Roots() {
		if err := lr.rewriteLNT(root); err != nil {
			return err
		}
	}
	if err := lr.lowerSites(rec, facts, dropped); err != nil {
		return err
	}
	return lr.epilogue(rec, dropped)
}

// lowerRun carries one Run's rewrite state: the pass thresholds, the
// choke-point context, the effective per-iter decisions (record decision
// plus any budget override), and the per-dim rewrite products PHASE 4
// consumes (filled by rewriteLNT).
type lowerRun struct {
	g         *ir.Graph
	params    Params
	mctx      *ir.MutContext
	decisions map[ir.Node]IterDecision
	overrides map[ir.Node]int // budget tile overrides, keyed by iter
	byIter    map[ir.Node]*dimRewrite
	claimed   map[*ir.Base]bool // effectful body ops already owned by an LNT
	rewrites  []*lntRewrite     // per-root products, Roots() order
	fills     []*siteFill       // PHASE 4 products, for the §8 layering checks
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
	if err := lr.checkFillLayering(); err != nil {
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

// checkLoopAnnotations pins "every parallel-inside loop carries the
// materialization annotation (thr/cta)": every generated loop's annotation
// set is re-read from the dimRewrite products and asserted well-formed —
// soc present; dosoc present exactly when soc != 0; loc present.
func (lr *lowerRun) checkLoopAnnotations() error {
	for _, lw := range lr.rewrites {
		for _, d := range lw.dims {
			loops := []ir.Node{d.between}
			if d.inside != nil {
				loops = append(loops, d.inside)
			}
			for _, loop := range loops {
				attrs := loopAnnotations(loop)
				soc, ok := attrs["soc"]
				if !ok {
					return fmt.Errorf("tile-lower: generated loop node %d carries no soc annotation", loop.BasePtr().ID())
				}
				if _, ok := attrs["loc"]; !ok {
					return fmt.Errorf("tile-lower: generated loop node %d carries no loc annotation", loop.BasePtr().ID())
				}
				if soc != 0 {
					if _, ok := attrs["dosoc"]; !ok {
						return fmt.Errorf("tile-lower: generated parallel loop node %d carries no dosoc annotation",
							loop.BasePtr().ID())
					}
				}
			}
		}
	}
	return nil
}

// loopAnnotations collects a generated for-loop's KeyedAttribute inputs by
// key (u32-const values). The pass constructs at most one attribute per
// key per loop.
func loopAnnotations(loop ir.Node) map[string]float64 {
	out := make(map[string]float64)
	ops := ir.OperandsOf(loop)
	for i := 3; i < len(ops); i++ {
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

// checkFillLayering pins "no cooperative tile op sits inside a bailing
// element loop (§4 layering)": a fill's coordinates may reference the full
// index expressions of dims OUTER than the fill point (the worked check:
// the lhs fill's m coordinate IS the enclosing m index expression) and the
// covered dims' round bases — but never the covered dims' own inside-loop
// coordinates (the fill covers a WHOLE tile; its covered coordinates ride
// round bases, placed immediately before the tile-inside loops). A
// violation means the fill was placed inside a tile-inside loop.
func (lr *lowerRun) checkFillLayering() error {
	for _, fill := range lr.fills {
		// The covered dims' inside coordinates are the reachable violation
		// class: every coordinate the fill carries comes from the load's
		// axes, whose iters are covered or above-tie (outer than the fill
		// point) by construction.
		coveredInside := make(map[*ir.Base]bool, len(fill.covered))
		for _, d := range fill.covered {
			if d.insideIV != nil {
				coveredInside[d.insideIV.BasePtr()] = true
			}
		}
		var bad ir.Node
		var visit func(n ir.Node)
		visit = func(n ir.Node) {
			if n == nil || bad != nil || n.BasePtr() == nil || ir.IsErased(n) {
				return
			}
			if coveredInside[n.BasePtr()] {
				bad = n
				return
			}
			for _, dep := range ir.DataDeps(n) {
				visit(dep)
			}
		}
		for _, ax := range ir.OperandsOf(fill.src)[1:] {
			visit(ax)
		}
		if bad != nil {
			return fmt.Errorf("tile-lower: fill for site %d references covered dim inside-loop dimvar node %d (a fill must sit outside the tile-inside loops, §4 layering)",
				fill.siteID, bad.BasePtr().ID())
		}
	}
	return nil
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

// checkNoBarriers pins "ZERO barriers emitted (§6 sync)": the pass
// constructs no barrier-emitting node — v0's suite has no sync kinds at
// all, so the constructive half is by construction; the check asserts the
// fill-store inventory carries its loadtile variant annotation (the
// runtime owns ALL synchronization — visibility, reuse safety, and store
// ordering are runtime contract items, §7bis).
func (lr *lowerRun) checkNoBarriers() error {
	for _, fill := range lr.fills {
		if !fill.annotated {
			return fmt.Errorf("tile-lower: fill for site %d carries no loadtile variant annotation", fill.siteID)
		}
	}
	return nil
}

// checkFragSites pins "every FRAG site satisfies the fully-unroll
// condition": all covered dims sequential (the decide judge's ClassFRAG
// criterion, re-asserted fail-closed at surgery time).
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
				if lr.decisions[d.iter].SoC != SoCSequential {
					return fmt.Errorf("tile-lower: FRAG site %d covers non-sequential iter node %d (the fully-unroll precondition)",
						id, d.iter.BasePtr().ID())
				}
			}
		}
	}
	return nil
}
