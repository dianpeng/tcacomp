// Package pipeline is the pass ORCHESTRATOR (fix2.md R-F/D-g, the
// 2026-10-25 author ruling): it composites every analysis, optimization, and
// lower pass into one exported interface and owns the wiring. External users
// do not care about pass composition as long as the code is efficient and
// performant — they call Run (or the internal/compiler facade built on it);
// nothing else assembles a pass sequence.
//
// The composition is THREE EXPLICIT WINDOWS over the registry's factories:
//
//	HIR   the region rules over the high-level sea — early cleanup, the
//	      simplify format passes, typing, CSE, and memory optimization.
//	LOWER the tiling decide/lower pair, scheduling, and the final lowering.
//	TAIL  the post-final-lower cleanup the lowers now delegate to (fix2:
//	      final-lower translates verbs only; if the lower produces garbage,
//	      the pipeline wires the cleanup after it).
//
// Every window is an explicit pass-ID list resolved through
// opt.Lookup(id).Factory() — NEVER opt.Registered()/opt.BuildPipeline(): the
// registry is process-global and tests register fake passes, so the default
// composition must never inherit whatever else happens to be registered.
// The registry stays the pass CATALOG (fix2 D-g); opt.BuildPipeline keeps its
// role for the existing registry-level tests.
package pipeline

import (
	"fmt"
	"math"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/dce"

	// The composition root: blank-importing opt/all registers every pass
	// factory this package resolves its windows through.
	_ "tcacomp/internal/opt/all"
)

// hirWindow is the HIR window: the region rules over the high-level sea, in
// pipeline order (fix2 D-g). Priorities cited per the registry scale
// (internal/opt/registry.go).
var hirWindow = []string{
	"dce",             // 100: early cleanup before anything observes the graph
	"dce-verify",      // 101: the post-sweep invariant check
	"simplify-lower",  // 102: IterCluster format → Iter format
	"simplify-inline", // 103: lambda parameter inlining
	"simplify-verify", // 104: the post-simplify invariant check
	"typeinfer",       // 105: every node typed after Simplification
	"gvn",             // 110: global CSE
	"gvn-verify",      // 111: the post-dedup invariant check
	"memopt",          // 112: store-to-load forwarding + dead-store elimination
	"memopt-verify",   // 113: the post-memopt invariant/fixpoint check
}

// lowerWindow is the LOWER window: the tiling decide/lower pair, the
// scheduler, and the final lowering (fix2 D-g).
var lowerWindow = []string{
	"tiling",        // 114: overlap analysis + search + decision record
	"tiling-verify", // 115: the post-decide record consistency check
	"tile-lower",    // 116: the annotated tiled nests + cache insertion
	"schedule",      // 900: sea-of-nodes → AST-style CFG
	"final-lower",   // 950: reducers/maps/runtime objects over the AST CFG
}

// The TAIL window (fix2 D-e — ruled Option A): the post-final-lower cleanup
// the lowers now delegate to. Wired AFTER final-lower, in this order:
//
//	memopt  a SECOND instance from the registered factory (R-E composite
//	        reuse — never a new registry ID): its region rules (a)-(c) no-op
//	        on the LIR window (C1's chain-totality gate) while rule (d) —
//	        fill liveness — is reachable in-pipeline for the first time.
//	dce     the AST-item-roots instance (D-d): collects every non-item sea
//	        node the lowers left dead, and its item sweep collects the items
//	        of memopt's erasures.
//
// memopt FIRST so the dce instance sees the final residue.
var tailMemoptID = "memopt"

// resolve turns one window's ID list into fresh pass instances through the
// registry factories, fail-closed on an unknown ID (a missing pass is a
// wiring bug, never a silent skip).
func resolve(window []string) ([]opt.Pass, error) {
	passes := make([]opt.Pass, 0, len(window))
	for _, id := range window {
		spec, ok := opt.Lookup(id)
		if !ok {
			return nil, fmt.Errorf("pipeline: pass ID %q is not registered (is the pass package blank-imported via opt/all?)", id)
		}
		passes = append(passes, spec.Factory())
	}
	return passes, nil
}

// HIRPasses returns fresh instances of the HIR window's passes — the exact
// composition Run wires, exposed for tests that drive a partial pipeline.
func HIRPasses() []opt.Pass {
	passes, err := resolve(hirWindow)
	if err != nil {
		// The window lists are compile-time constants over opt/all's
		// registrations; a miss is a programming error, not a runtime input.
		panic(err)
	}
	return passes
}

// LOWERPasses returns fresh instances of the LOWER window's passes — the
// exact composition Run wires, exposed for tests that drive a partial
// pipeline.
func LOWERPasses() []opt.Pass {
	passes, err := resolve(lowerWindow)
	if err != nil {
		panic(err)
	}
	return passes
}

// TailPasses returns the exact tail instances Run wires — the pipeline-owned
// const dedup (fix3 D-o), then the memopt factory instance (R-E: reuse the
// registered pass compositively; never a new ID), then the AST-item-roots
// dce instance (its item sweep collects the dedup's erased duplicates) —
// exposed for tests that pin the tail composition in isolation.
func TailPasses() []opt.Pass {
	spec, ok := opt.Lookup(tailMemoptID)
	if !ok {
		panic(fmt.Errorf("pipeline: pass ID %q is not registered (is the pass package blank-imported via opt/all?)", tailMemoptID))
	}
	return []opt.Pass{constDedupPass{}, spec.Factory(), dce.NewASTRoots()}
}

// constDedupPass is the MINIMAL pipeline-owned tail const dedup (fix3 D-o,
// the fix2 D-e measurement's ruled escape clause): GVN (110) predates the
// passes that mint most const atoms (tile-lower/schedule/final-lower), and
// nothing else re-dedups them — without this step the codegen-ready IR
// carries duplicate literal nodes per (type, value) and the goldens drift.
// NOT a registry ID (R-E: a pipeline-owned composite step, like the tail
// dce instance) and NOT a lower's business (R-A/R-B: canonicalizeConsts
// was exiled from final-lower). Erases run BEFORE the tail's dce instance
// so its item sweep collects the duplicates' AST items.
type constDedupPass struct{}

// ID identifies the composite step in manager instrumentation.
func (constDedupPass) ID() string { return "const-dedup" }

// Requires: no ordering preconditions (it is a fixpoint-free leaf sweep).
func (constDedupPass) Requires() []string { return nil }

// Produces: builder-output stage — consts are legal at every stage, and
// the pass erases nodes only.
func (constDedupPass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — every analysis is invalidated.
func (constDedupPass) Preserved() []string { return nil }

// Run dedups literal constants: at most one live const node per
// (type, value) pair survives (the lowest-ID def of the pair; allocation
// order). Consts are immutable leaves, so sharing is unconditionally
// sound; the interned NumericType pointer keys exactly. The declared
// payload type (ConstTypeOf) is the canonical key — the typeinfer cell is
// nil for most post-typeinfer consts, exactly the population this step
// exists for.
func (constDedupPass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return fmt.Errorf("pipeline: const-dedup: nil graph")
	}
	type constKey struct {
		ty   *ir.NumericType
		bits uint64
	}
	canon := make(map[constKey]ir.Node)
	mctx := &ir.MutContext{Pass: "const-dedup"}
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindConst {
			continue
		}
		v, ok := ir.ConstValueOf(n)
		if !ok {
			continue
		}
		ty := ir.ConstTypeOf(n)
		if ty == nil {
			ty = ir.TypeOf(n)
		}
		nt, ok := ty.(*ir.NumericType)
		if !ok || nt == nil {
			continue // nothing safe to key on
		}
		k := constKey{nt, math.Float64bits(v)}
		c, seen := canon[k]
		if !seen {
			canon[k] = n
			continue
		}
		if err := mctx.ReplaceAllUsesWith(n, c); err != nil {
			return fmt.Errorf("pipeline: const-dedup: canonicalizing const #%d onto #%d: %w",
				n.BasePtr().ID(), c.BasePtr().ID(), err)
		}
		if err := mctx.Erase(n); err != nil {
			return fmt.Errorf("pipeline: const-dedup: erasing duplicate const #%d: %w", n.BasePtr().ID(), err)
		}
	}
	return nil
}

// Run executes the full pass composition over g: ONE opt.Manager over the
// concatenated windows, so declarative preconditions (Pass.Requires) are
// enforced against every pass that ran earlier in the same pipeline run —
// the tail memopt's Requires ["gvn"] is satisfied by the HIR window's gvn,
// and two same-ID instances in one run are fine. The graph must be
// DEMATERIALIZED (a Level-1 sea): the windows' level gates refuse
// materialized/CFG-carrying graphs, and the LOWER window attaches the AST
// CFG the backend consumes.
func Run(g *ir.Graph) error {
	hir, err := resolve(hirWindow)
	if err != nil {
		return err
	}
	lower, err := resolve(lowerWindow)
	if err != nil {
		return err
	}
	tail := TailPasses()
	passes := make([]opt.Pass, 0, len(hir)+len(lower)+len(tail))
	passes = append(passes, hir...)
	passes = append(passes, lower...)
	passes = append(passes, tail...)
	return opt.NewManager(passes).Run(g)
}
