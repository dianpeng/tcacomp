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
		// fix8 §3 slice-0 probe hook: env-gated Params overlay, zero
		// default path — mirrors the decide pass by construction (the
		// same env read, never a second source).
		Factory:  func() opt.Pass { return VerifyPass{Params: mustParamsFromEnv()} },
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
//     MemLoad; every re-derived candidate site has its record entry; and
//     every re-derived NON-CANDIDATE (R4: cross-LNT INS, empty INS, no
//     forest member) has its explicit record entry — fail-closed, a
//     silently dropped load is corruption.
//   - FEASIBILITY (per LNT, the fix4 §2.7 model): the Params
//     re-validation at the top re-checks the hyperparameter surface;
//     every iter's LoC belongs to its SoC's preset list (CTA ->
//     CtaLoCPresets, WARP -> WarpLoCPresets, SEQ -> the implicit 1); the
//     per-LNT CTA factorization product stays within MaxCTA and the warp
//     product within MaxWarpsPerCTA; every LNT carries at least one CTA
//     dim; every LoC >= 1 and every TileSize >= 2 (the compiler never
//     generates a non-tiling schedule, the 2026-10-24 ruling); every SoC
//     known.
//   - SITE well-formedness: a promoted site's Tie is a live iter of the
//     site's LNT or nil; its class is SMEM or FRAG; and the site decision
//     MATCHES the promotion judge re-run over the record's own schedule
//     under the per-LNT blockDim context (promotion flag, class, Reuse,
//     Bytes — a divergence is corruption).
//   - KERNEL-WIDE BUDGETS (the fix4 §2.3 MASRAM gates): the recomputed
//     SMEM window-bytes sum and the per-thread FRAG sum — the FRAG sum of
//     register-block footprints, min(raw window, MaxFragPerThread) per
//     site under fix6 §3 — accumulated over ALL LNTs' promoted sites,
//     stay within MaxCache / MaxFragPerThread — the SAME sums the decide
//     feasibility and the lower's budget resolution gate.
//   - FORLOOP STEP: every live ForLoop's dimvar step is a positive const
//     (a zero/negative stride is an infinite/empty loop — the IR-verifier
//     version of this invariant lands with T4; tiling-verify enforces it
//     now).
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
	op := analyzeOverlap(g, f, se, v.Params)

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

	// COVERAGE (non-candidates, R4): every re-derived non-candidate load
	// has its explicit record entry, and the entry is NOT promoted.
	for id, want := range op.NonCandidates {
		sd, ok := rec.Sites[id]
		if !ok {
			return fmt.Errorf("tiling-verify: %s load node %d has no record entry (non-candidates are declared, never silently dropped)",
				want.Reason, id)
		}
		if sd.Promoted {
			return fmt.Errorf("tiling-verify: load node %d is declared non-candidate (%s) but its record entry is promoted",
				id, want.Reason)
		}
	}

	// FEASIBILITY, per LNT — the fix4 §2.7 re-check set, re-derived with
	// the SAME constraint helpers the search prunes with (constraints.go:
	// single source — the products here are the completed-vector MALoC
	// bounds).
	for _, root := range f.Roots() {
		st := initialPartState()
		for _, it := range treeItersOf(root) {
			biv, err := bivOf(it)
			if err != nil {
				return fmt.Errorf("tiling-verify: %w", err)
			}
			d := rec.Iters[biv]
			if d.LoC < 1 || d.TileSize < 2 {
				return fmt.Errorf("tiling-verify: iter node %d carries LoC %d / TileSize %d (LoC must be >= 1 and TileSize >= 2 — the compiler never generates a non-tiling schedule, the 2026-10-24 ruling)",
					it.BasePtr().ID(), d.LoC, d.TileSize)
			}
			// The fix5 R1 effective-extent discipline: the iter's const
			// extent when known, else the NominalExtent stand-in — exactly
			// what digitListOf enumerated the adaptive digit from, so the
			// re-check can never disagree with the search on a const dim.
			effExtent, ok := constExtentOf(it)
			if !ok {
				effExtent = v.Params.NominalExtent
			}
			if !locPresetOK(d.SoC, d.LoC, d.TileSize, effExtent, v.Params) {
				return fmt.Errorf("tiling-verify: iter node %d carries LoC %d outside its SoC %s preset list (fix4 2.7)",
					it.BasePtr().ID(), d.LoC, d.SoC)
			}
			switch d.SoC {
			case SoCSequential, SoCTA, SoCWARP:
				st = st.place(d.SoC, d.LoC)
			default:
				return fmt.Errorf("tiling-verify: iter node %d carries unknown SoC %d",
					it.BasePtr().ID(), int(d.SoC))
			}
		}
		if !st.cumapOK() {
			return fmt.Errorf("tiling-verify: LNT rooted at iter node %d has no CTA assignment",
				root.Iter.BasePtr().ID())
		}
		if st.ctaLoC > v.Params.MaxCTA {
			return fmt.Errorf("tiling-verify: LNT rooted at iter node %d carries a CTA factorization product %d over MaxCTA %d",
				root.Iter.BasePtr().ID(), st.ctaLoC, v.Params.MaxCTA)
		}
		if st.warpK > v.Params.MaxWarpsPerCTA {
			return fmt.Errorf("tiling-verify: LNT rooted at iter node %d carries a warp factorization product %d over MaxWarpsPerCTA %d",
				root.Iter.BasePtr().ID(), st.warpK, v.Params.MaxWarpsPerCTA)
		}
	}

	// SITE checks over the re-derived candidate sites, in precedence
	// order: (1) presence; (2) the KERNEL-WIDE budget recomputation (R3
	// SMEM scope, F6 FRAG budget) over the judge's window bytes; (3) the
	// structural well-formedness of promoted entries; (4) the judge
	// cross-check — the record must agree with the promotion judge re-run
	// over the record's own schedule (promotion flag, class, reuse, and
	// bytes alike; a divergence is corruption).
	type siteCheck struct {
		lf *lntFacts
		sf SiteFacts
		sd SiteDecision
		j  siteJudgment
	}
	var checks []siteCheck
	smem, frag := 0, 0
	for i := range op.Trees {
		lf := &op.Trees[i]
		dec := make(map[ir.Node]IterDecision, len(lf.Levels))
		for _, it := range treeItersOf(lf.Tree) {
			biv, err := bivOf(it)
			if err != nil {
				return fmt.Errorf("tiling-verify: %w", err)
			}
			dec[it] = rec.Iters[biv]
		}
		// The judge's per-LNT launch context, re-derived from the RECORD's
		// own decisions (fix4 §1.2): blockDim = (Π warp K) × 32 — the
		// coupled inside walk's stride and the reuse divisor. The judge
		// cross-check therefore runs the same model the decide pass did.
		ctxSchedule := blockDimCtx{blockDim: blockDimOf(dec)}
		for _, sf := range lf.CS {
			id := sf.Load.BasePtr().ID()
			sd, ok := rec.Sites[id]
			if !ok {
				return fmt.Errorf("tiling-verify: candidate site node %d has no record entry", id)
			}
			j := judgeSite(sf, dec, ctxSchedule, v.Params)
			checks = append(checks, siteCheck{lf: lf, sf: sf, sd: sd, j: j})
			if sd.Promoted {
				// The kernel-wide sums account the RECORD's footprints —
				// the decide-side priced blocks the lower's budget
				// resolution will actually consume — gated against THIS
				// pass's thresholds. (Re-pricing from the judge under this
				// pass's MaxFragPerThread would instead measure a
				// hypothetical re-decide: under a deliberately tight
				// budget the record's 32 B block would re-price to 31 B
				// and the gate could never fire — the gate must ask
				// "does the RECORD fit this budget", fix6 §3.)
				switch sd.Class {
				case ClassSMEM:
					smem += sd.Bytes
				case ClassFRAG:
					frag += sd.Bytes
				}
			}
		}
	}
	if smem > v.Params.MaxCache {
		return fmt.Errorf("tiling-verify: recomputed kernel-wide SMEM footprint %d > MaxCache %d",
			smem, v.Params.MaxCache)
	}
	if frag > v.Params.MaxFragPerThread {
		return fmt.Errorf("tiling-verify: recomputed kernel-wide FRAG footprint %d > MaxFragPerThread %d",
			frag, v.Params.MaxFragPerThread)
	}
	for _, c := range checks {
		id := c.sf.Load.BasePtr().ID()
		if !c.sd.Promoted {
			continue
		}
		if c.sd.Tie != nil {
			if _, ok := f.NodeOf(c.sd.Tie); !ok || c.sd.Tie.BasePtr().Erased() {
				return fmt.Errorf("tiling-verify: promoted site node %d tie node %d is not a live iter",
					id, c.sd.Tie.BasePtr().ID())
			}
			if rootTreeOf(f, c.sd.Tie) != c.lf.Tree {
				return fmt.Errorf("tiling-verify: promoted site node %d tie node %d belongs to another LNT",
					id, c.sd.Tie.BasePtr().ID())
			}
		}
		if c.sd.Class != ClassSMEM && c.sd.Class != ClassFRAG {
			return fmt.Errorf("tiling-verify: promoted site node %d carries class %s",
				id, c.sd.Class)
		}
		if c.sd.Promoted != c.j.promoted || c.sd.Class != c.j.class ||
			c.sd.Reuse != c.j.reuse || c.sd.Bytes != c.j.bytes {
			return fmt.Errorf("tiling-verify: site node %d record (%v,%s,reuse %d,bytes %d) disagrees with the judge over its own schedule (%v,%s,reuse %d,bytes %d)",
				id, c.sd.Promoted, c.sd.Class, c.sd.Reuse, c.sd.Bytes, c.j.promoted, c.j.class, c.j.reuse, c.j.bytes)
		}
	}

	// FORLOOP dimvar steps: positive consts (the IR-verifier version of
	// this invariant lands with T4; tiling-verify enforces it now).
	if err := checkForLoopSteps(g); err != nil {
		return err
	}
	// F7 stamp (261021-o61 T3): the record passed the full consistency
	// sweep — tile-lower consumes this mark and refuses unstamped records
	// fail-closed (a corrupted record must not reach the rewrite).
	rec.Verified = true
	return nil
}

// checkForLoopSteps asserts every live ForLoop's dimvar carries a POSITIVE
// CONST step: ops = [dim, init, step] on the KindDimVar (node_dimvar.go).
// A zero stride is an infinite loop, a negative or symbolic stride is not
// a strided round enumeration at all.
func checkForLoopSteps(g *ir.Graph) error {
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindForLoop {
			continue
		}
		ops := ir.OperandsOf(n)
		if len(ops) == 0 || ops[0] == nil || ops[0].BasePtr() == nil || ops[0].BasePtr().Erased() {
			return fmt.Errorf("tiling-verify: for-loop node %d carries no induction-variable operand",
				n.BasePtr().ID())
		}
		dvOps := ir.DataDeps(ops[0])
		if len(dvOps) < 3 || dvOps[2] == nil {
			return fmt.Errorf("tiling-verify: for-loop node %d dimvar node %d carries no step operand",
				n.BasePtr().ID(), ops[0].BasePtr().ID())
		}
		if step, ok := ir.ConstValueOf(dvOps[2]); !ok || step <= 0 {
			return fmt.Errorf("tiling-verify: for-loop node %d dimvar step is not a positive const (a zero/negative stride is an infinite/empty loop)",
				n.BasePtr().ID())
		}
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
