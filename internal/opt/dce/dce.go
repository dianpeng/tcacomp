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
//
// The zero Pass{} is the registry's priority-100 instance (the two-root-set
// sweep). NewASTRoots() constructs the TAIL instance (fix2 D-d): over an
// AST-attached (scheduled) graph it roots on ALL AST ITEMS plus the standard
// two-root set — the post-schedule truth of liveness — so effectful runtime
// calls (Init/Reduce/Combine have no SSA users but ARE items) survive while
// every non-item sea node is collected. After the sea sweep it performs the
// item maintenance: AST items whose nodes died (this run's or an earlier
// tail pass's erasures) are removed from every block.
type Pass struct {
	// astRoots selects the AST-item-roots mode (fix2 D-d): the mode exists
	// only post-schedule — Run fails closed without an attached AST CFG.
	astRoots bool
}

// NewASTRoots returns the AST-item-roots DCE instance (fix2 D-d): the tail
// composition's collector over a scheduled graph.
func NewASTRoots() Pass { return Pass{astRoots: true} }

// ID returns the pipeline name "dce" (both modes share the identity: the
// registry catalog knows one dce, the pipeline wires it twice — R-E).
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
// ImpactBy chain head) and the params — plus, in the AST-roots mode, every
// AST item node (fix2 D-d). The pass mirrors the build-time sweep's
// (internal/ir buildingDCESea) conservative discipline, built from the
// exported ir face and the node-held marks instead:
//
//   - LEVEL GATE first (the Scheduler's refusal pattern, opposite direction):
//     a nil graph is an error, and so is a materialized graph or one with a
//     CFG — DCE runs on the Level-1 sea (an AST-ATTACHED graph is neither
//     materialized nor CFG-carrying; the tail instance passes the gate). In
//     the AST-roots mode a graph WITHOUT an attached AST CFG is a named
//     error: the mode exists only post-schedule. The refusal happens before
//     any mutation, so a refused graph is untouched, never partially swept.
//
//   - MARK: ir.Walk with a MarkAlive visitor. Walk's Level-1 roots are
//     exactly the two-root set discipline (all params — graph-owned
//     signature nodes the def-use checker audits, so the sweep must never
//     erase them — plus the end node), and its child relation is operand
//     slots + payload uses (ImpactBy effect inputs included) + lambda
//     bodies, so liveness propagates through BOTH data dependence and effect
//     dependence with zero duplicated traversal logic. In the AST-roots mode
//     ir.WalkFrom marks every AST item node additionally (collected
//     recursively through all blocks — Groups, Loop blocks, and the
//     Prototype): the post-schedule truth of liveness, which preserves
//     effectful runtime calls the sea's use-based reachability cannot see.
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
//   - ITEM MAINTENANCE (AST-roots mode only, after the sea sweep): every
//     block's items whose Node is tombstoned are removed — the tail dce owns
//     the erased-item sweep for the WHOLE tail (this run's and any earlier
//     tail pass's erasures, e.g. the tail memopt's fills — fix2 D-e).
//
// Marks are transient (D-15 exception, ir/mark.go): ResetMarks before the
// walk and deferred after the sweep, so no state leaks across passes.
func (p Pass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("dce: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("dce: graph already materialized (dce runs on the Level-1 sea)")
	}
	ast := g.AST()
	if p.astRoots && ast == nil {
		return errors.New("dce: AST-roots mode requires an attached AST CFG (schedule must run first)")
	}
	g.ResetMarks()
	defer g.ResetMarks()

	// MARK from the two-root set (see Run comment: Walk's Level-1 roots ARE
	// the two-root set, and its child relation covers ImpactBy).
	mark := ir.VisitorFunc(func(n ir.Node) ir.VisitResult {
		n.BasePtr().MarkAlive()
		return ir.Descend
	})
	ir.Walk(g, mark)

	// AST-roots mode (fix2 D-d): additionally mark every AST item node PLUS
	// every Loop block's header node — collected recursively through the
	// Prototype and every block of the Body tree (the post-schedule truth of
	// liveness). The loop headers are structural: no item references a
	// ForLoop through an operand edge (the "nest" attrs point the other
	// way), yet the emitter reads each block's Loop() operand face (soc/
	// dosoc/loc annotations), so a root policy without them would erase the
	// scheduled nest the AST still materializes.
	if p.astRoots {
		ir.WalkFrom(astRootNodes(ast), mark)
	}

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

	// ITEM MAINTENANCE (fix2 D-e): no AST item anywhere may reference an
	// erased node — the tail dce owns the sweep for every tail erasure.
	if p.astRoots {
		var sweep func(blk *ir.ASTBlock)
		sweep = func(blk *ir.ASTBlock) {
			blk.RemoveItemsIf(func(it ir.ASTItem) bool {
				return it.Node != nil && it.Node.BasePtr().Erased()
			})
			for _, it := range blk.Items() {
				if it.Block != nil {
					sweep(it.Block)
				}
			}
		}
		sweep(ast.Body)
		sweep(ast.Prototype)
	}
	return nil
}

// astRootNodes collects the AST's root nodes over the whole kernel — every
// AST item's node PLUS every Loop block's header node (Loop(), whose operand
// closure reaches the IV, the bound wrapper, and the soc/dosoc/loc/nest
// annotations) — in deterministic walk order (Body tree depth-first, then
// the Prototype's items).
func astRootNodes(k *ir.ASTKernel) []ir.Node {
	var roots []ir.Node
	var collect func(blk *ir.ASTBlock)
	collect = func(blk *ir.ASTBlock) {
		if loop := blk.Loop(); loop != nil {
			roots = append(roots, loop)
		}
		for _, it := range blk.Items() {
			if it.Node != nil {
				roots = append(roots, it.Node)
			}
			if it.Block != nil {
				collect(it.Block)
			}
		}
	}
	if k != nil {
		if k.Body != nil {
			collect(k.Body)
		}
		if k.Prototype != nil {
			collect(k.Prototype)
		}
	}
	return roots
}
