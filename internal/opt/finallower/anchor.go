package finallower

import (
	"fmt"

	"tcacomp/internal/ir"
)

// anchor.go is final-lower's FINAL gate (fix3.md D-p, carried from fix2
// D-f): the REDUCE ANCHOR. Every live Reduce(value) call's value-operand
// transitive OPERAND closure (data-dependency edges ONLY — liveness never
// crosses memory-chain edges, which would drag in unrelated accesses) must
// reach at least ONE memory access whose subscript closure references a
// DimVar that is the IV of a block ENCLOSING the call's SchedBlock — the
// reduce demonstrably accumulates data whose location varies with a loop
// the call sits inside. A severed piped chain (the value a constant, the
// init, or a loop-invariant expression — the 2026-10-24 "Combine returns
// the init, every output is 0" wall class) fails closed with a named
// error at the lowering that broke it, never at numerical acceptance.
//
// The check runs over the pass's own products (post-lower graph with
// SchedBlock stamps) and is wired in Run AFTER ir.Verify: chain integrity
// first, semantic anchoredness second.

// anchorReduceCalls validates the anchor over every live Reduce call.
func (lr *lowering) anchorReduceCalls() error {
	for _, n := range lr.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindRuntimeCallMethod {
			continue
		}
		if ir.RuntimeMethodOf(n) != "Reduce" {
			continue
		}
		ops := ir.OperandsOf(n)
		if len(ops) != 2 || ops[1] == nil {
			return fmt.Errorf("final-lower: reduce anchor: Reduce call %d carries %d operands, want exactly the value",
				n.BasePtr().ID(), len(ops))
		}
		if err := lr.anchorOne(n, ops[1]); err != nil {
			return err
		}
	}
	return nil
}

// anchorOne checks one Reduce call: its value's operand closure must reach
// a memory access subscripted by an IV whose loop encloses the call.
func (lr *lowering) anchorOne(call, value ir.Node) error {
	callBlk := call.BasePtr().SchedBlock()
	if callBlk == nil {
		return fmt.Errorf("final-lower: reduce anchor: Reduce call %d carries no SchedBlock linkage (unscheduled?)",
			call.BasePtr().ID())
	}
	// The transitive DATA closure of the value (operand edges only — never
	// the ImpactBy memory-chain edges: liveness is data liveness).
	closure := make(map[ir.Node]bool)
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || x.BasePtr().Erased() || closure[x] {
			return
		}
		closure[x] = true
		for _, op := range ir.OperandsOf(x) {
			visit(op)
		}
	}
	visit(value)

	for acc := range closure {
		if !isMemoryAccess(acc) {
			continue
		}
		if lr.accessWalksEnclosingLoop(acc, callBlk) {
			return nil // anchored: this access's subscript varies inside an enclosing loop
		}
	}
	return fmt.Errorf("final-lower: reduce anchor: Reduce call %d's value closure reaches no memory access subscripted by an enclosing loop's IV — the piped producer chain is severed (the reduce would combine constants; the 2026-10-24 all-zero wall class)",
		call.BasePtr().ID())
}

// isMemoryAccess reports whether n is a MemLoad of any storage class (the
// anchor's memory-access predicate; loads are the walk's data source).
func isMemoryAccess(n ir.Node) bool {
	return n != nil && n.KindOf() == ir.KindMemLoad
}

// accessWalksEnclosingLoop reports whether the access's subscript closure
// references a DimVar whose loop block encloses (is an ancestor-or-self of)
// blk: the access reads memory at a coordinate that varies with a loop the
// position sits inside.
func (lr *lowering) accessWalksEnclosingLoop(access ir.Node, blk *ir.ASTBlock) bool {
	if lr.ast == nil || lr.ast.Body == nil {
		return false
	}
	var dominates func(haystack, needle *ir.ASTBlock) bool
	dominates = func(haystack, needle *ir.ASTBlock) bool {
		if haystack == needle {
			return true
		}
		for _, it := range haystack.Items() {
			if it.Block != nil && dominates(it.Block, needle) {
				return true
			}
		}
		return false
	}
	ops := ir.OperandsOf(access)
	for _, ax := range ops[1:] { // operand 0 is the tensor/buffer; the rest are subscripts
		if ax == nil {
			continue
		}
		for d := range operandClosure(ax) {
			if d.KindOf() != ir.KindDimVar {
				continue
			}
			def := d.BasePtr().SchedBlock()
			if def == nil {
				continue
			}
			if dominates(def, blk) {
				return true
			}
		}
	}
	return false
}
