// Package dce implements the pipeline's mark-sweep dead-code-elimination
// pass (quick-261018-l36 requirements 1-5, 8).
package dce

import (
	"errors"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the pass in the global registry: Priority 100 — early
// cleanup, before anything else observes the graph.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "dce",
		Priority: 100,
		Factory:  func() opt.Pass { return Pass{} },
	})
}

// Pass is the Level-1 (sea-of-nodes) mark-sweep DCE. It is stateless: every
// Run re-derives liveness from the graph, so re-runs are idempotent.
type Pass struct{}

// ID returns the pipeline name "dce".
func (Pass) ID() string { return "dce" }

// Requires: DCE has no ordering preconditions.
func (Pass) Requires() []string { return nil }

// Produces: builder-output stage — the pass erases nodes only and introduces
// no new kinds, so the constructed legality window re-checks the graph.
func (Pass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — every analysis is invalidated.
func (Pass) Preserved() []string { return nil }

// Run erases every node unreachable from the graph's Level-1 two-root set:
// the end node (its return-value operands plus every mem-param's final
// ImpactBy chain head) and the params. The pass mirrors the build-time
// sweep's (internal/ir buildingDCESea) conservative discipline, built from
// the exported ir face and the node-held marks instead:
//
//   - LEVEL GATE first (the Scheduler's refusal pattern, opposite direction):
//     a nil graph is an error, and so is a materialized graph or one with a
//     CFG — DCE runs on the Level-1 sea. The refusal happens before any
//     mutation, so a refused graph is untouched, never partially swept.
//
//   - MARK: ir.Walk with a MarkAlive visitor. Walk's Level-1 roots are
//     exactly the two-root set discipline (all params — graph-owned
//     signature nodes the def-use checker audits, so the sweep must never
//     erase them — plus the end node), and its child relation is operand
//     slots + payload uses (ImpactBy effect inputs included) + lambda
//     bodies, so liveness propagates through BOTH data dependence and effect
//     dependence with zero duplicated traversal logic. This is the same
//     reachable-set the build-time sweep computes; Walk is the exported
//     substrate for it.
//
//   - SWEEP: iterate the AllocatedNodes snapshot in allocation order over a
//     fixpoint bounded by len(nodes)+1 passes; skip nodes already erased,
//     nodes marked alive, and nodes with live uses (a live user, or a dead
//     user not yet erased); erase the rest through MutContext{Pass: "dce"}.
//     Erasing a dead user kills its operand/payload records, freeing its
//     defs for the next pass; a pass that erases nothing ends the loop.
//     Conservative by construction: cyclic dead clusters keep mutual
//     live-looking uses and survive; live-used nodes are never force-erased.
//
// Marks are transient (D-15 exception, ir/mark.go): ResetMarks before the
// walk and deferred after the sweep, so no state leaks across passes.
func (Pass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("dce: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("dce: graph already materialized (dce runs on the Level-1 sea)")
	}
	g.ResetMarks()
	defer g.ResetMarks()

	// MARK from the two-root set (see Run comment: Walk's Level-1 roots ARE
	// the two-root set, and its child relation covers ImpactBy).
	ir.Walk(g, ir.VisitorFunc(func(n ir.Node) ir.VisitResult {
		n.BasePtr().MarkAlive()
		return ir.Descend
	}))

	// SWEEP the allocation registry to a fixpoint.
	nodes := g.AllocatedNodes()
	mctx := &ir.MutContext{Pass: "dce"}
	erased := make(map[ir.Node]bool)
	for passIdx := 0; passIdx <= len(nodes); passIdx++ {
		progress := false
		for _, n := range nodes {
			if n == nil || erased[n] || n.BasePtr().Alive() {
				continue
			}
			if len(ir.UsesOf(n)) > 0 {
				continue // still referenced: a live user, or one not yet erased
			}
			if err := mctx.Erase(n); err != nil {
				return err
			}
			erased[n] = true
			progress = true
		}
		if !progress {
			break
		}
	}
	return nil
}
