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

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"

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

// Run executes the full pass composition over g: ONE opt.Manager over the
// concatenated windows, so declarative preconditions (Pass.Requires) are
// enforced against every pass that ran earlier in the same pipeline run.
// The graph must be DEMATERIALIZED (a Level-1 sea): the windows' level gates
// refuse materialized/CFG-carrying graphs, and the LOWER window re-attaches
// the AST CFG the backend consumes.
func Run(g *ir.Graph) error {
	hir, err := resolve(hirWindow)
	if err != nil {
		return err
	}
	lower, err := resolve(lowerWindow)
	if err != nil {
		return err
	}
	passes := make([]opt.Pass, 0, len(hir)+len(lower))
	passes = append(passes, hir...)
	passes = append(passes, lower...)
	return opt.NewManager(passes).Run(g)
}
