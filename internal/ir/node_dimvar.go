package ir

// node_dimvar.go holds the LIR induction-variable nodes (KindDimVar/
// KindBoundedDimVar, D-3) and their value relations, specified in
// doc/optana/induction_variable.md:
//
//   - An induction variable is a VALUE OBJECT: an (init, step) tuple. Two
//     IVs are equal iff their tuples are equal — identity is value identity,
//     not name identity.
//   - A bounded induction variable arises whenever an IV binds to a
//     dimension with a learned maximum: it is the tuple ((iv tuple), Max).
//   - Include/Belong relations (A ~> B, B <~ A) hold between BIVs: A
//     includes B iff B's iteration space is a subset of A's — for
//     normalized IVs (init=0, step=1) iff A.max <= B.max. Include is not
//     commutative; Belong is its reverse.
//
// dimvar is value-level material (FamilyAbstract), not loop-nest structure;
// the explicit lowered loop is KindForLoop (node_loop.go).

// dimVarNode: lowered induction variable; ops = [init, step]; the dim
// binding is a payload use of a dim InputParam (name/debug payload only —
// identity is the (init, step) tuple).
type dimVarNode struct {
	*Base
	operandList
	dimUse SSAUse
}

// forEachPayloadUse implements payloadBearer: the dim binding.
func (n *dimVarNode) forEachPayloadUse(fn func(SSAUse)) { fn(n.dimUse) }

// NewDimVar creates a lowered induction variable over dim (a dim
// InputParam), with init and step value operands. The IV's identity is the
// (init, step) tuple — never the name.
func (g *Graph) NewDimVar(origin string, dim Node, init, step Node) Node {
	n := g.allocNode(KindDimVar, origin, 2).(*dimVarNode)
	n.dimUse = wirePayloadUse(n, dim)
	wireOperands(n, init, step)
	return n
}

// operandAt returns the node's operand slot i, nil-safely.
func operandAt(n Node, i int) Node {
	if impl := asImpl(n); impl != nil {
		j := 0
		var got Node
		impl.forEachOperand(func(u SSAUse) {
			if j == i {
				got = u.Def()
			}
			j++
		})
		return got
	}
	return nil
}

// constValueOf returns the constant value of n when n is a constNode.
func constValueOf(n Node) (float64, bool) {
	c, ok := n.(*constNode)
	if !ok {
		return 0, false
	}
	return c.val, true
}

// sameValue compares two value nodes for IV-tuple equality: the same node,
// or two constants of equal value. Anything else (distinct non-constant
// defs) is not equal — conservative, per the value-object spec.
func sameValue(a, b Node) bool {
	if a == nil || b == nil {
		return a == b
	}
	if a == b {
		return true
	}
	av, aok := constValueOf(a)
	bv, bok := constValueOf(b)
	return aok && bok && av == bv
}

// SameIV reports whether a and b are the same induction variable: both
// dimVarNodes whose (init, step) tuples are equal (sameValue semantics).
func SameIV(a, b Node) bool {
	da, ok := a.(*dimVarNode)
	if !ok {
		return false
	}
	db, ok := b.(*dimVarNode)
	if !ok {
		return false
	}
	return sameValue(operandAt(da, 0), operandAt(db, 0)) &&
		sameValue(operandAt(da, 1), operandAt(db, 1))
}

// normalizedIV reports whether n is a dimVarNode with the normalized tuple
// init=0, step=1 — the precondition for the Include relation's max
// comparison (doc/optana/induction_variable.md).
func normalizedIV(n Node) bool {
	d, ok := n.(*dimVarNode)
	if !ok {
		return false
	}
	iv, iok := constValueOf(operandAt(d, 0))
	sv, sok := constValueOf(operandAt(d, 1))
	return iok && sok && iv == 0 && sv == 1
}

// boundedDimVarNode: a dimvar bound to a learned maximum — the tuple
// ((iv tuple), Max). ops = [iv, max]; the dim name renders from the bound
// dimvar (op 0).
type boundedDimVarNode struct {
	*Base
	operandList
}

// NewBoundedDimVar creates a bounded induction variable: iv (a dimVarNode)
// bounded by max. The verifier's shape check enforces the iv operand kind.
func (g *Graph) NewBoundedDimVar(origin string, iv, max Node) Node {
	n := g.allocNode(KindBoundedDimVar, origin, 2).(*boundedDimVarNode)
	wireOperands(n, iv, max)
	return n
}

// SameBIV reports whether a and b are the same bounded induction variable:
// equal dimvars (SameIV) AND equal max values (sameValue).
func SameBIV(a, b Node) bool {
	ba, ok := a.(*boundedDimVarNode)
	if !ok {
		return false
	}
	bb, ok := b.(*boundedDimVarNode)
	if !ok {
		return false
	}
	return SameIV(operandAt(ba, 0), operandAt(bb, 0)) &&
		sameValue(operandAt(ba, 1), operandAt(bb, 1))
}

// Includes reports the BIV include relation A ~> B: B's iteration space is
// a subset of A's. Per the spec this relation is defined for normalized
// IVs (init=0, step=1) only, where it reduces to A.max <= B.max. It is NOT
// commutative. Non-BIV or non-normalized operands yield false.
func Includes(a, b Node) bool {
	ba, ok := a.(*boundedDimVarNode)
	if !ok {
		return false
	}
	bb, ok := b.(*boundedDimVarNode)
	if !ok {
		return false
	}
	if !normalizedIV(operandAt(ba, 0)) || !normalizedIV(operandAt(bb, 0)) {
		return false
	}
	am, aok := constValueOf(operandAt(ba, 1))
	bm, bok := constValueOf(operandAt(bb, 1))
	if !aok || !bok {
		return false
	}
	return am <= bm
}

// BelongsTo reports the reverse include relation B <~ A: A belongs to B
// iff B includes A (doc/optana/induction_variable.md — used by partial loop
// fusion, not by the optimizer itself).
func BelongsTo(a, b Node) bool { return Includes(b, a) }
