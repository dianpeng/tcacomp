// tiling.go is the loop-tiling DECIDE pass (doc/opt.md's "loop tiling"
// slot): for every loop-nest tree it runs the MemLoad overlap analysis
// (overlap.go) and the PSched search (search.go), and publishes the
// decision-record side table through the opt analysis cache. It occupies
// pipeline priority 114 — after memopt-verify (113), immediately before
// tiling-verify (115) — and requires simplify-lower (the pass consumes the
// Iter format with InductionVariable axes that simplify-lower produces).
//
// Ruling D2 (2026-10-20): the pass mutates NOTHING — no MutContext is
// created, no node is created/erased/rewired/field-mutated, and MemLoad
// carries no ctaCache field. The record is the only vehicle, and it
// survives the pass's own cache invalidation because its name is on this
// pass's preservation whitelist (the loop_tile_lower.md §1 record-lifetime
// mechanism; the lower pass will pin Requires when it lands).
package tiling

import (
	"errors"
	"fmt"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the pass in the global registry: Priority 114 — the
// doc/opt.md "loop tiling" slot, after the memory-optimization window
// (memopt/memopt-verify at 112/113). The registry scale in
// internal/opt/registry.go documents the anchors.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "tiling",
		Priority: 114,
		Factory:  func() opt.Pass { return Pass{Params: DefaultParams()} },
	})
}

// Pass is the loop-tiling decide pass. It is stateless: every Run
// re-derives the forest, the overlap facts, and the schedules from the
// graph, so re-runs are idempotent.
type Pass struct {
	// Params are the external thresholds; the zero value fails Validate —
	// construct via New or the registry factory (DefaultParams).
	Params Params
}

// New returns a decide-pass instance with the given thresholds.
func New(p Params) *Pass { return &Pass{Params: p} }

// ID returns the pipeline name "tiling".
func (Pass) ID() string { return "tiling" }

// Requires: simplify-lower must have run earlier — the overlap analysis
// chases InductionVariable axes, which only exist in the Iter format
// simplify-lower produces.
func (Pass) Requires() []string { return []string{"simplify-lower"} }

// Produces: builder-output stage — the pass introduces no node kinds and
// mutates nothing, so the constructed legality window re-checks the graph.
func (Pass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: the decision record. The cache is invalidate-all-by-default
// and the manager invalidates AFTER this pass runs, so the record survives
// only because its own name is on this whitelist.
func (Pass) Preserved() []string { return []string{RecordName} }

// Run decides every LNT's schedule and publishes the record:
//
//   - LEVEL GATE first (the gvn/memopt refusal pattern): a nil graph and a
//     materialized/CFG-carrying graph are deterministic errors evaluated
//     before anything is read, so a refused graph is untouched.
//   - PARAMS are validated before any computation (external input).
//   - DERIVE: loopforest.Analyze and sideeffect.Analyze once per Run (the
//     ValueEqual membership the overlap analysis threads), then per root —
//     in allocation order — the overlap analysis and the PSched search.
//   - KERNEL-WIDE BUDGETS (R3/F6, ruled 2026-10-21): the SMEM and FRAG
//     sums accumulate across LNTs — each LNT's search is gated by the
//     running totals the earlier LNTs' chosen schedules consume, and the
//     chosen schedule's sums advance them. LNTs are decided in Roots()
//     order, so the accumulation is deterministic.
//   - NON-CANDIDATES (R4): the overlap product's explicit declarations
//     (cross-LNT INS, empty INS) are published verbatim — never dropped.
//   - PUBLISH: ctx.Cache.Put(record). A nil Context (or nil Cache) is
//     tolerated — the pass is analysis-shaped and direct Run callers may
//     have no manager; the decisions simply are not cached then.
func (p Pass) Run(ctx *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("tiling: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("tiling: graph already materialized (tiling runs on the Level-1 sea)")
	}
	if err := p.Params.Validate(); err != nil {
		return err
	}

	f := loopforest.Analyze(g)
	se := sideeffect.Analyze(g)
	op := analyzeOverlap(g, f, se, p.Params)
	// R-REDUCED: the reducer-consuming iter set prunes WARP digits from
	// the feasible area when ReducedDimWarp is false (the Wave-1 policy,
	// now the explicit opt-out; fix4 §2.2/§3.5).
	// Populated here — after analyzeOverlap — so the discovery logic
	// itself stays untouched.
	reduced := reducedIters(g)
	// §3.1 (ruling 2026-10-30): the eliminated set rides beside the
	// reducer-consuming set — warpDimOf's WARP ranking consumes it.
	eliminated := reducedAxesOf(g)
	for i := range op.Trees {
		op.Trees[i].reduced = reduced
		op.Trees[i].eliminated = eliminated
	}
	record := &Record{
		Iters: make(map[ir.Node]IterDecision),
		Sites: make(map[uint64]SiteDecision, len(op.NonCandidates)),
	}
	smemFloor, fragFloor := 0, 0
	for _, lf := range op.Trees {
		res, err := searchSchedule(lf, p.Params, smemFloor, fragFloor)
		if err != nil {
			return err
		}
		smemFloor += res.smem
		fragFloor += res.frag
		for iter, d := range res.iters {
			deps := ir.DataDeps(iter)
			if len(deps) == 0 || deps[0] == nil {
				return fmt.Errorf("tiling: iter node %d carries no BIV operand", iter.BasePtr().ID())
			}
			record.Iters[deps[0]] = d
		}
		for id, sd := range res.sites {
			record.Sites[id] = sd
		}
	}
	for id, sd := range op.NonCandidates {
		record.Sites[id] = sd
	}
	if ctx != nil && ctx.Cache != nil {
		ctx.Cache.Put(record)
	}
	return nil
}
