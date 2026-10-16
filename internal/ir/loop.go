package ir

// This file holds the lowered explicit loop node (review P1R-04, FamilyLoop)
// — the material a loop nest becomes once the abstract iteration cluster
// is lowered: one loop per bounded induction variable, body as an operand
// subtree.

// forLoopNode: lowered loop nest; ops = [iv, bound, body] — the bounded
// induction variable, its bound, and the body subtree (a storetile, a
// nested forloop, or another effect-carrying stage).
type forLoopNode struct {
	*Base
	operandList
}

// NewForLoop creates an explicit lowered loop over the bounded induction
// variable iv, bounded by bound, with body as the loop body subtree. The
// verifier's shape check enforces the iv operand kind.
func (g *Graph) NewForLoop(origin string, iv, bound, body Node) Node {
	n := g.allocNode(KindForLoop, origin, 3).(*forLoopNode)
	wireOperands(n, iv, bound, body)
	return n
}
