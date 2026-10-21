package ir

// node_intrinsic.go holds the loop-tile-lower launch-map query nodes
// (quick-261021-jay, doc/optana/loop_tile_lower.md §5): IntrinsicGridMap and
// IntrinsicThreadMap — the two LIR intrinsics the tile-lower pass
// materializes when it rewrites a decided loop-nest tree into annotated
// tiled ForLoop nests.
//
//   - Semantics (§5): per-LNT scoped queries — the queried object is the
//     GridConcurrencyMap / ThreadMap the runtime constructs for the
//     enclosing LNT nest (§7bis). field ∈ {"index", "dimension"} selects
//     the query; the result type is u32 (a launch-map coordinate/dim).
//   - Payload (P1): the field selector is a STRING payload riding the
//     callNode.name precedent (mixString hashing, value.go). DoSoC is the
//     SINGLE operand and must be a const u32 — the map-dimension binding
//     the pass derives by its per-LNT numbering (§2).
//   - Family: FamilyAbstract (the TensorDimOf/InductionVariable pure-value
//     precedent) — EffectOf is free, so GVN-CSE of the intrinsics is sound
//     and desirable (§5: repeated (DoSoC, field) queries collapse).
//
// The six §5 companions: node struct + catalog row (this file, catalog.go),
// allocNode case (node.go), verify shape case (verify.go), typeinfer rules
// (internal/opt/typeinfer/rules.go), the legality-window note (catalog row
// doc, P9), and contentHash/contentEqual (value.go).

// intrinsicNode: lowered launch-map query; ops = [DoSoC const u32]; field is
// the string payload selecting the query ("index" | "dimension"). Shared by
// both kinds — the kind discriminates grid vs thread map.
type intrinsicNode struct {
	*Base
	operandList
	field string
}

// NewIntrinsicGridMap creates a GridConcurrencyMap query node: field
// selects the query ("index" | "dimension"), dosoc is the const-u32 map
// dimension binding. The verifier's shape check pins the operand count,
// the DoSoC constness, and the field vocabulary (§5 companion 3).
func (g *Graph) NewIntrinsicGridMap(origin string, field string, dosoc Node) Node {
	n := g.allocNode(KindIntrinsicGridMap, origin, 1).(*intrinsicNode)
	n.field = field
	wireOperands(n, dosoc)
	return n
}

// NewIntrinsicThreadMap creates a ThreadMap query node: field selects the
// query ("index" | "dimension"), dosoc is the const-u32 map dimension
// binding. Same shape contract as NewIntrinsicGridMap.
func (g *Graph) NewIntrinsicThreadMap(origin string, field string, dosoc Node) Node {
	n := g.allocNode(KindIntrinsicThreadMap, origin, 1).(*intrinsicNode)
	n.field = field
	wireOperands(n, dosoc)
	return n
}

// IntrinsicFieldOf returns the intrinsic's field payload ("index" or
// "dimension"), or "" for any other node kind — the exported read face
// (payload read faces precedent, node_expr.go), total so passes outside
// package ir never assert the concrete node type.
func IntrinsicFieldOf(n Node) string {
	if in, ok := n.(*intrinsicNode); ok {
		return in.field
	}
	return ""
}
