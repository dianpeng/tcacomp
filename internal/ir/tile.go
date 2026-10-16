package ir

// This file holds the lowered cooperative tile-load/store nodes (review
// P1R-04, FamilyTileOp). The tensor operand is an OPERAND SLOT (not a
// payload access like the abstract storeNode): lowering promotes the
// abstract payload access into ordinary SSA data flow, and the axes become
// bounded induction variables.

// loadTileNode: lowered cooperative tile load; ops = [tensor, bivs...] —
// the tensor parameter first, then one bounded dimvar per tile axis, in
// axis order.
type loadTileNode struct {
	*Base
	operandList
}

// NewLoadTile creates a cooperative tile load of tensor (a RoleTensor
// InputParam) at the bounded induction variables bivs (axis order). The
// verifier's shape check enforces the tensor operand kind.
func (g *Graph) NewLoadTile(origin string, tensor Node, bivs ...Node) Node {
	n := g.allocNode(KindLoadTile, origin, 1+len(bivs)).(*loadTileNode)
	ops := make([]Node, 0, 1+len(bivs))
	ops = append(ops, tensor)
	ops = append(ops, bivs...)
	wireOperands(n, ops...)
	return n
}

// storeTileNode: lowered cooperative tile store; ops = [value, tensor,
// bivs...] — the stored value first (the pipe-bound input, mirroring the
// abstract store's operand/payload split), then the tensor parameter, then
// one bounded dimvar per tile axis.
type storeTileNode struct {
	*Base
	operandList
}

// NewStoreTile creates a cooperative tile store of value into tensor (a
// RoleTensor InputParam) at the bounded induction variables bivs (axis
// order). The verifier's shape check enforces the tensor operand kind.
func (g *Graph) NewStoreTile(origin string, value, tensor Node, bivs ...Node) Node {
	n := g.allocNode(KindStoreTile, origin, 2+len(bivs)).(*storeTileNode)
	ops := make([]Node, 0, 2+len(bivs))
	ops = append(ops, value, tensor)
	ops = append(ops, bivs...)
	wireOperands(n, ops...)
	return n
}
