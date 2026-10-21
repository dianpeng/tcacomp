// verify.go is the post-decide verification pass: a RECORD-CONSISTENCY
// checker that re-derives the forest and the overlap facts (the memopt
// deriveRegions pattern — the derivation is re-RUN, never shared state)
// and asserts the decision record against the graph and the bounds, so a
// corrupted record fails loudly at the next pipeline step instead of
// surfacing as bogus node surgery in the future loop-tile-lower pass. It
// does NOT re-run the search and claims nothing about optimality — green
// proves the record is consistent, not that the decide pass chose well
// (mirroring memopt-verify's stance).
//
// Scope rationale (the explicit companion decision): every opt sub-module
// so far pairs pass + verify (dce-verify 101, simplify-verify 104,
// gvn-verify 111, memopt-verify 113); the record is TRUSTED input to a
// future heavy-mutation pass, and failing loudly is this codebase's stated
// corruption policy. The registry scale in internal/opt/registry.go
// documents the 115 anchor (one authoritative comment block, shared with
// the 114 anchor Task 2 added).
package tiling

import (
	"errors"
	"fmt"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the verification pass: Priority 115 — immediately
// after tiling (114). Requires tiling: verifying an undecided graph
// asserts nothing about the tiling decisions (and would just flag a
// missing record).
func init() {
	opt.Register(opt.PassSpec{
		ID:       "tiling-verify",
		Priority: 115,
		Requires: []string{"tiling"},
		Factory:  func() opt.Pass { return VerifyPass{Params: DefaultParams()} },
	})
}

// VerifyPass asserts the record-consistency invariants. It mutates
// nothing. Configured instances must mirror the decide pass's Params —
// the bounds are re-checked against THIS pass's thresholds (the default
// factory pairs both on DefaultParams).
type VerifyPass struct {
	Params Params
}

// ID returns the pipeline name "tiling-verify".
func (VerifyPass) ID() string { return "tiling-verify" }

// Requires: the decide pass must have run earlier — the record under
// verification is its product.
func (VerifyPass) Requires() []string { return []string{"tiling"} }

// Produces: builder-output stage — the pass introduces no node kinds and
// no mutations, so the constructed legality window re-checks the graph.
func (VerifyPass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: the decision record — it must outlive THIS pass's
// invalidation too. Nothing may run between decide and lower that wipes
// it (loop_tile_lower.md §1); keeping it here is the same mechanism the
// decide pass uses, extended one step.
func (VerifyPass) Preserved() []string { return []string{RecordName} }

// Run asserts the invariants:
//
//   - PRESENCE: the cache holds a *Record under "tiling-record".
//   - COVERAGE: every live Iter of every forest tree has a per-Iter entry
//     under its BIV; every site key resolves to a live (non-erased)
//     MemLoad; every re-derived candidate site has its record entry.
//   - FEASIBILITY (per LNT, the final model): the Params re-validation at
//     the top re-checks the assert-and-trap divisibility (MaxCTA a
//     PortionCTA multiple, warps a PortionWarp multiple); at least one
//     CTA entry; the v0 policy (at most one CTA and one independent WARP
//     entry); the CTA LoC == MaxCTA (portion -1/full); the warp LoC is a
//     warp-portion multiple (LoC/quantum in [1, PortionWarp]); sequential
//     entries carry LoC 1; every LoC and TileSize >= 1; every SoC known.
//   - SITE well-formedness: a promoted site's Tie is a live iter of the
//     site's LNT or nil; its class is SMEM or FRAG; and the recomputed
//     SMEM footprint (element size x covered tile sizes) stays within
//     MaxCache.
//
// Gates mirror the decide pass: a nil graph and a materialized/CFG-carrying
// graph are deterministic errors evaluated before anything is read.
func (v VerifyPass) Run(ctx *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("tiling-verify: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("tiling-verify: graph already materialized (the record invariants are Level-1 sea invariants)")
	}
	if err := v.Params.Validate(); err != nil {
		return fmt.Errorf("tiling-verify: %w", err)
	}
	if ctx == nil || ctx.Cache == nil {
		return errors.New("tiling-verify: no analysis cache (the record is verified through the cache)")
	}
	got := ctx.Cache.Get(RecordName)
	if got == nil {
		return errors.New("tiling-verify: no tiling record (did the tiling pass run?)")
	}
	rec, ok := got.(*Record)
	if !ok {
		return fmt.Errorf("tiling-verify: cache entry under %q is %T, want *Record", RecordName, got)
	}

	// Re-derive the forest and the overlap facts — never shared state.
	f := loopforest.Analyze(g)
	se := sideeffect.Analyze(g)
	all := analyzeOverlap(g, f, se, v.Params)

	// COVERAGE (iters): every live Iter of every tree has an entry.
	for _, root := range f.Roots() {
		for _, it := range treeItersOf(root) {
			biv, err := bivOf(it)
			if err != nil {
				return fmt.Errorf("tiling-verify: %w", err)
			}
			if _, ok := rec.Iters[biv]; !ok {
				return fmt.Errorf("tiling-verify: iter node %d has no record entry under its BIV node %d",
					it.BasePtr().ID(), biv.BasePtr().ID())
			}
		}
	}

	// COVERAGE (sites): every site key resolves to a live MemLoad.
	liveByID := make(map[uint64]ir.Node, len(g.AllocatedNodes()))
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		liveByID[n.BasePtr().ID()] = n
	}
	for id := range rec.Sites {
		n, ok := liveByID[id]
		if !ok || n.KindOf() != ir.KindMemLoad {
			return fmt.Errorf("tiling-verify: site key %d does not resolve to a live MemLoad", id)
		}
	}

	// FEASIBILITY, per LNT — the final portion-grammar model. The legacy
	// CTA-product / warp-sum bounds are subsumed: the single CTA dim must
	// exhaust MaxCTA exactly (portion -1/full), and a warp LoC that is a
	// portion multiple with portion in [1, PortionWarp] cannot exceed the
	// warp space.
	warpsPerPortion := (v.Params.MaxThreadsPerCTA / 32) / PortionWarp
	for _, root := range f.Roots() {
		ctaCount, warpCount := 0, 0
		for _, it := range treeItersOf(root) {
			biv, err := bivOf(it)
			if err != nil {
				return fmt.Errorf("tiling-verify: %w", err)
			}
			d := rec.Iters[biv]
			if d.LoC < 1 || d.TileSize < 1 {
				return fmt.Errorf("tiling-verify: iter node %d carries LoC %d / TileSize %d (both must be >= 1)",
					it.BasePtr().ID(), d.LoC, d.TileSize)
			}
			switch d.SoC {
			case SoCSequential:
				if d.LoC != 1 {
					return fmt.Errorf("tiling-verify: sequential iter node %d carries LoC %d (must be 1)",
						it.BasePtr().ID(), d.LoC)
				}
			case SoCTA:
				ctaCount++
				if d.LoC != v.Params.MaxCTA {
					return fmt.Errorf("tiling-verify: iter node %d carries CTA LoC %d, want MaxCTA %d (portion -1/full exhausts the CTA space)",
						it.BasePtr().ID(), d.LoC, v.Params.MaxCTA)
				}
			case SoCWARP:
				warpCount++
				if portion := d.LoC / warpsPerPortion; d.LoC%warpsPerPortion != 0 || portion < 1 || portion > PortionWarp {
					return fmt.Errorf("tiling-verify: iter node %d carries warp LoC %d, not a warp-portion multiple (quantum %d warps, portion in [1, %d])",
						it.BasePtr().ID(), d.LoC, warpsPerPortion, PortionWarp)
				}
			default:
				return fmt.Errorf("tiling-verify: iter node %d carries unknown SoC %d",
					it.BasePtr().ID(), int(d.SoC))
			}
		}
		if ctaCount < 1 {
			return fmt.Errorf("tiling-verify: LNT rooted at iter node %d has no CTA assignment",
				root.Iter.BasePtr().ID())
		}
		if ctaCount > 1 {
			return fmt.Errorf("tiling-verify: LNT rooted at iter node %d carries %d CTA dims, violating the v0 policy (at most one)",
				root.Iter.BasePtr().ID(), ctaCount)
		}
		if warpCount > 1 {
			return fmt.Errorf("tiling-verify: LNT rooted at iter node %d carries %d independent WARP dims, violating the v0 policy (at most one)",
				root.Iter.BasePtr().ID(), warpCount)
		}
	}

	// SITE well-formedness + the recomputed SMEM budget, over the
	// re-derived candidate sites.
	smem := 0
	for _, lf := range all {
		dec := make(map[ir.Node]IterDecision, len(lf.Levels))
		for _, it := range treeItersOf(lf.Tree) {
			biv, err := bivOf(it)
			if err != nil {
				return fmt.Errorf("tiling-verify: %w", err)
			}
			dec[it] = rec.Iters[biv]
		}
		for _, sf := range lf.CS {
			sd, ok := rec.Sites[sf.Load.BasePtr().ID()]
			if !ok {
				return fmt.Errorf("tiling-verify: candidate site node %d has no record entry",
					sf.Load.BasePtr().ID())
			}
			if !sd.Promoted {
				continue
			}
			if sd.Tie != nil {
				if _, ok := f.NodeOf(sd.Tie); !ok || sd.Tie.BasePtr().Erased() {
					return fmt.Errorf("tiling-verify: promoted site node %d tie node %d is not a live iter",
						sf.Load.BasePtr().ID(), sd.Tie.BasePtr().ID())
				}
				if rootTreeOf(f, sd.Tie) != lf.Tree {
					return fmt.Errorf("tiling-verify: promoted site node %d tie node %d belongs to another LNT",
						sf.Load.BasePtr().ID(), sd.Tie.BasePtr().ID())
				}
			}
			if sd.Class != ClassSMEM && sd.Class != ClassFRAG {
				return fmt.Errorf("tiling-verify: promoted site node %d carries class %s",
					sf.Load.BasePtr().ID(), sd.Class)
			}
			if sd.Class == ClassSMEM {
				smem += siteBytesPerTile(sf, dec)
			}
		}
	}
	if smem > v.Params.MaxCache {
		return fmt.Errorf("tiling-verify: recomputed SMEM footprint %d > MaxCache %d",
			smem, v.Params.MaxCache)
	}
	return nil
}

// bivOf resolves an iter's BIV operand with a well-formedness error.
func bivOf(iter ir.Node) (ir.Node, error) {
	deps := ir.DataDeps(iter)
	if len(deps) == 0 || deps[0] == nil {
		return nil, fmt.Errorf("iter node %d carries no BIV operand", iter.BasePtr().ID())
	}
	return deps[0], nil
}

// rootTreeOf walks an iter's tree node up to its root.
func rootTreeOf(f loopforest.Forest, iter ir.Node) *loopforest.Tree {
	t, ok := f.NodeOf(iter)
	if !ok {
		return nil
	}
	for t.Parent != nil {
		t = t.Parent
	}
	return t
}
