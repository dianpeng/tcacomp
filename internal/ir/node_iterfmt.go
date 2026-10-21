package ir

// node_iterfmt.go holds the Input-list restructure's node suite
// (quick-261019-ktm, doc/optana/lower_itercluster.md): the three HIR
// reference carriers (IterClusterProject, KeyedAttribute, As) and the two
// LIR iteration VALUE nodes (TensorDimOf, InductionVariable).
//
// The design rule these nodes embody (D1-D6): every incoming reference of a
// Map/Reduce/IterCluster rides the node's Input (operand) list — never a
// hidden payload pointer — so the sea-of-nodes analyses, the verifier's
// chain checks, Walk, DCE, and GVN all see stage inputs as ordinary
// SSA data-dependency edges. The as-order is a trailing KindAs node, the
// reduce init rides operand 1 as a KindKeyedAttribute, and a Map's i-th
// lambda parameter binds to the i-th value operand, which in the
// IterCluster format is an IterClusterProject into the cluster's input
// list. Simplification (internal/opt/simplify) lowers that format into the
// Iter format: one Iter per as-dim bounded by a TensorDimOf, with
// InductionVariable nodes indexing the rewired loads.
//
// Family classification follows the DimVar precedent: all five kinds are
// FamilyAbstract — pure VALUE/reference nodes under EffectOf and the
// accumulated side-effect analysis, so index math over iteration values
// stays GVN-eligible (identity keying keeps distinct defs apart).

// iterClusterProjectNode: HIR bridge from an IterCluster's input slot to a
// Map's lambda parameter — ops = [cluster]; idx is the structural input-slot
// index payload (participates in the GVN content hash, D4). In the
// IterCluster format, Map Input slot i holds IterClusterProject(cluster, i):
// the positional correspondence Input[i] ↔ lambda param i is the binding
// contract the verifier enforces.
type iterClusterProjectNode struct {
	*Base
	operandList
	idx int
}

// NewIterClusterProject creates a project node: Map parameter i binds to
// cluster input slot idx. The verifier's shape check pins operand 0 to an
// itercluster node.
func (g *Graph) NewIterClusterProject(origin string, cluster Node, idx int) Node {
	n := g.allocNode(KindIterClusterProject, origin, 1).(*iterClusterProjectNode)
	n.idx = idx
	wireOperands(n, cluster)
	return n
}

// IterClusterProjectIndex returns the project's cluster input-slot index —
// the positional binding contract's read face (quick-261019-ktm: lowering
// resolves Map Input slot i to the cluster's input slot idx). Non-project
// nodes report ok == false.
func IterClusterProjectIndex(n Node) (int, bool) {
	p, ok := n.(*iterClusterProjectNode)
	if !ok {
		return 0, false
	}
	return p.idx, true
}

// keyedAttributeNode: HIR named attribute riding an Input list —
// ops = [value]; key is the debug/identity payload ("init" for Reduce's
// initializer, D2). Attributes unwrap to their value operand at lambda
// binding time (D9) and in the printer.
type keyedAttributeNode struct {
	*Base
	operandList
	key string
}

// NewKeyedAttribute creates an attribute node wrapping value under key.
func (g *Graph) NewKeyedAttribute(origin, key string, value Node) Node {
	n := g.allocNode(KindKeyedAttribute, origin, 1).(*keyedAttributeNode)
	n.key = key
	wireOperands(n, value)
	return n
}

// KeyedAttributeOf returns the attribute's key payload and its value
// operand, or ("", nil) for any other node kind — the exported read face
// (payload read faces precedent, node_expr.go), so passes outside package
// ir (e.g. tile-lower reading its for-loop soc/dosoc/loc annotations)
// inspect attributes without asserting the concrete node type.
func KeyedAttributeOf(n Node) (string, Node) {
	if ka, ok := n.(*keyedAttributeNode); ok {
		return ka.key, operandAt(ka, 0)
	}
	return "", nil
}

// asNode: HIR as-order clause — ops = the dim InputParam nodes in
// as-order. Carried as the LAST Input-list element of a Map (and, when it
// arrives, a Reduce); the verifier enforces unique-and-last placement plus
// RoleDim operands (D1). Iter creation order in simplify-lower comes from
// this node's operands.
type asNode struct {
	*Base
	operandList
}

// NewAs creates an as-order node over the dim InputParams dims, in order.
func (g *Graph) NewAs(origin string, dims ...Node) Node {
	return g.NewNode(KindAs, origin, dims...)
}

// tensorDimOfNode: LIR shape upper bound — ops = [tensor, axis-const]. The
// constructor interns the TU32 axis constant internally, so a TensorDimOf is
// self-contained: TensorDimOf(lhs, 1) is the u32 upper bound of lhs's axis
// 1. The verifier pins operand 0 to a RoleTensor InputParam and operand 1
// to a const (D6).
type tensorDimOfNode struct {
	*Base
	operandList
}

// NewTensorDimOf creates a shape-bound node over tensor's axis: the TU32
// axis constant is allocated and wired at operand 1 by the constructor.
func (g *Graph) NewTensorDimOf(origin string, tensor Node, axis int) Node {
	n := g.allocNode(KindTensorDimOf, origin, 2).(*tensorDimOfNode)
	axisConst := g.NewConst(origin, InternScalar(TU32), float64(axis))
	wireOperands(n, tensor, axisConst)
	return n
}

// inductionVariableNode: LIR u32 current-iteration value of an Iter —
// ops = [iter]. The loop itself is the iterNode; this node is the VALUE a
// lowered MemLoad axis slot holds after simplify-lower (the doc's
// InductionVariable(iterA)). Participates in normal expressions and in GVN
// (two IVs over the same iter dedupe by operand identity, D6).
type inductionVariableNode struct {
	*Base
	operandList
}

// NewInductionVariable creates the u32 iteration-value node of iter. The
// verifier's shape check pins operand 0 to an iter node.
func (g *Graph) NewInductionVariable(origin string, iter Node) Node {
	n := g.allocNode(KindInductionVariable, origin, 1).(*inductionVariableNode)
	wireOperands(n, iter)
	return n
}
