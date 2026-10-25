// Package typeinfer implements the Type Inference pass (quick-261019-tu3,
// doc/optana/type_inf.md): after Simplification, every node of the graph —
// HIR and LIR alike — carries a derived type on the IR's computed-type cell
// (Base.ty, written through MutContext.SetType), so later passes (GVN,
// lowering, codegen) and diagnostics read types structurally through
// ir.TypeOf. The pass is re-runnable (the spec: "run multiple times during
// optimization, e.g. after lowering") and idempotent: SetType neither
// writes nor re-stamps when the cell already carries an Equal type, so a
// second run leaves every TypeOf AND every LastTouchedBy identical.
//
// Algorithm — one memoized post-order pass, NO fixed-point loop. This is
// the deliberate answer to the spec's "check whether one pass suffices":
// the IR is SSA with unique defs, no phis, and no back edges (doc/ir.md §5;
// forloop bodies are forward operand slots), so the data-dependency graph
// is a DAG and every node's type is a pure function of its operands' types.
// A worklist/fixed-point would only be needed for cyclic flow, which this
// IR forbids — and an operand cycle is corruption the pass fails closed on
// (the scheduler fails closed on the same class).
//
// Compute-then-stamp discipline (T-ti-02): the entire memo is computed
// before any SetType fires, so a refused graph is never half-stamped.
package typeinfer

import (
	"errors"
	"fmt"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the pass: Priority 105 — after the Simplification
// suite (102-104), before gvn (110), so GVN and every later consumer reads
// types off the cell. The registry scale in internal/opt/registry.go
// documents the anchors.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "typeinfer",
		Priority: 105,
		Requires: []string{"simplify-lower"},
		Factory:  func() opt.Pass { return Pass{} },
	})
}

// Pass types every non-erased node of a post-Simplification graph. The
// deliberately-untyped kinds (allocations, As, Lambda, and a 0-or-multi-
// value Return) stay nil; every other kind carries a type or the pass
// fails closed naming the node.
type Pass struct{}

// ID returns the pipeline name "typeinfer".
func (Pass) ID() string { return "typeinfer" }

// Requires: the spec pins "IterCluster lowered before TI" — the manager
// re-checks this at run time regardless of pipeline construction.
func (Pass) Requires() []string { return []string{"simplify-lower"} }

// Produces: builder-output stage — the pass introduces no node kinds and
// rewrites nothing, so the constructed legality window re-checks the graph.
func (Pass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — consistent with the invalidate-all default.
func (Pass) Preserved() []string { return nil }

// untypedKinds is the closed untyped whitelist: allocations carry no dtype
// anywhere in their operands (memnew/memnewsmem/fragnew), As is an
// output-order clause rather than a value, Lambda is a function
// expression for which no function Type exists yet, and the runtime OBJECT
// CONSTRUCTION kinds carry no scalar Type anywhere (P-4/LR-8,
// quick-261022-l1o: KernelInfo / the two mappings / the three reducers are
// opaque handles — the reducer's element type T is checked against the
// lambda's parameter-definition types at Reduce lowering, never derived
// from the construction node).
var untypedKinds = map[ir.Kind]bool{
	ir.KindMemNew:                   true, // every storage class (fix3 D-h)
	ir.KindAs:                       true,
	ir.KindLambda:                   true,
	ir.KindRuntimeKernelInfo:        true,
	ir.KindRuntimeGridMapping:       true,
	ir.KindRuntimeThreadMapping:     true,
	ir.KindRuntimeSequentialReducer: true,
	ir.KindRuntimeWarpReducer:       true,
	ir.KindRuntimeGlobalReducer:     true,
}

// Run performs the typing over the Level-1 sea. Gates first (nil and
// materialized/CFG-carrying graphs are deterministic errors evaluated
// before anything is read or written); then compute the full memo
// (mutating nothing), stamp through one MutContext in allocation order,
// and sweep the postcondition. Any rule error aborts with the graph
// untouched.
func (Pass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("typeinfer: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("typeinfer: graph already materialized (typeinfer runs on the Level-1 sea)")
	}

	nodes := g.AllocatedNodes() // live nodes only, allocation order

	// Compute phase: memoized iterative post-order from every live node.
	memo := make(map[*ir.Base]ir.Type, len(nodes))
	for _, root := range nodes {
		if _, done := memo[root.BasePtr()]; done {
			continue
		}
		if err := computePostOrder(root, memo); err != nil {
			return err // compute-then-stamp: nothing was written
		}
	}

	// Stamp phase: one choke-point context, allocation order; the untyped
	// kinds' cells are already nil, so a nil memo entry skips the write.
	mctx := &ir.MutContext{Pass: "typeinfer"}
	for _, n := range nodes {
		ty := memo[n.BasePtr()]
		if ty == nil {
			continue
		}
		if err := mctx.SetType(n, ty); err != nil {
			return fmt.Errorf("typeinfer: stamp node %d (%s): %w", n.BasePtr().ID(), n.KindOf(), err)
		}
	}

	// Postcondition sweep (in-pass; ~10 lines, no transformation to
	// re-check): every live node typed outside the whitelist.
	for _, n := range nodes {
		if ir.TypeOf(n) == nil && !untypedAllowed(n) {
			return fmt.Errorf("typeinfer: node %d (%s) carries no computed type (pass bug)",
				n.BasePtr().ID(), n.KindOf())
		}
	}
	return nil
}

// untypedAllowed reports whether n may legitimately stay untyped: a
// whitelist kind, a Return with 0 or >1 value operands (none/multi-
// valued; a tuple Type is the documented future extension point), or an
// Init/Reduce RuntimeCallMethod (statements, not values — P-4,
// quick-261022-l1o; only the Combine call produces the reduced value).
func untypedAllowed(n ir.Node) bool {
	if untypedKinds[n.KindOf()] {
		return true
	}
	if n.KindOf() == ir.KindRuntimeCallMethod && ir.RuntimeMethodOf(n) != "Combine" {
		return true
	}
	return n.KindOf() == ir.KindReturn && len(ir.OperandsOf(n)) != 1
}

// computePostOrder types root's operand DAG bottom-up into memo using an
// explicit (node, child-index) frame stack — recursion depth must never be
// graph-size-dependent (T-ti-01), mirroring bodySubtree's shape in
// simplify/verify.go. Only Input-list edges are followed: ImpactBy edges
// carry effect order, never type inputs, and lambda bodies ride the lambda
// node's operand slots, so OperandsOf reaches them. An on-stack revisit is
// a data-dependency cycle — corruption, fail closed naming the node.
func computePostOrder(root ir.Node, memo map[*ir.Base]ir.Type) error {
	type frame struct {
		n    ir.Node
		ops  []ir.Node
		next int
	}
	onStack := map[*ir.Base]bool{root.BasePtr(): true}
	stack := []frame{{n: root, ops: ir.OperandsOf(root)}}

	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		if f.next < len(f.ops) {
			op := f.ops[f.next]
			f.next++
			if op == nil || ir.IsErased(op) {
				continue // unwired slot / tombstoned def: the verifier owns both
			}
			if _, done := memo[op.BasePtr()]; done {
				continue
			}
			if onStack[op.BasePtr()] {
				return fmt.Errorf("typeinfer: node %d (%s): data-dependency cycle",
					op.BasePtr().ID(), op.KindOf())
			}
			onStack[op.BasePtr()] = true
			stack = append(stack, frame{n: op, ops: ir.OperandsOf(op)})
			continue
		}
		ty, err := typeOfNode(f.n, memo)
		if err != nil {
			return err
		}
		memo[f.n.BasePtr()] = ty // nil entries included: untyped kinds memoize uniformly
		delete(onStack, f.n.BasePtr())
		stack = stack[:len(stack)-1]
	}
	return nil
}
