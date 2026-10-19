package ir

// node_loop.go holds the loop-structure nodes: the HIR abstract iteration
// suite (KindIter/KindIterCluster) and the LIR explicit loop nest
// (KindForLoop, D-3). The LIR induction-variable VALUE objects (KindDimVar/
// KindBoundedDimVar) are value-level material, not loop-nest structure — they
// live with their relations in node_dimvar.go.

// iterNode: abstract induction variable; ops = [dim, bound?, init, step].
// The dim binding and the bound ride the Input list (Input/ImpactBy-only
// convention, quick-261019-ktm revision) — the bound slot EXISTS only in
// the bounded form (`@x/M`), so the unbounded form (`@x`) is a 3-operand
// node [dim, init, step]. The bound accepts any value node (D5: dim params
// and lowered TensorDimOf bounds alike).
type iterNode struct {
	*Base
	operandList
}

// NewIter creates an abstract induction-variable node over the dim
// InputParam node dim, bounded by bound (nil = the unbounded form @x, a
// 3-operand node), with init and step operands. The verifier's shape check
// pins operand 0 to a dim InputParam.
func (g *Graph) NewIter(origin string, dim, bound Node, init, step Node) Node {
	if bound == nil {
		return g.NewNode(KindIter, origin, dim, init, step)
	}
	return g.NewNode(KindIter, origin, dim, bound, init, step)
}

// iterClusterNode: abstract compound header; ops = [dims..., inputs...] —
// the cluster's dims (RoleDim InputParams) LEAD the input list, the inputs
// follow (MemLoad nodes in the IterCluster format, the wrapped Map in the
// legacy builder format). D3's variadic input list plus the convention move
// of the dims (quick-261019-ktm revision) — every reference rides ordinary
// SSA data edges.
type iterClusterNode struct {
	*Base
	operandList
}

// NewIterCluster creates an abstract compound iteration header over dim
// InputParam nodes with the given input list (D3). The legacy single-body
// call shape NewIterCluster(origin, dims, mapNode) is the one-input case
// and compiles unchanged; the IterCluster format passes its MemLoad inputs
// positionally — IterClusterProject(cluster, i) selects INPUT slot i
// (ClusterInputs skips the leading dims).
func (g *Graph) NewIterCluster(origin string, dims []Node, inputs ...Node) Node {
	ops := make([]Node, 0, len(dims)+len(inputs))
	ops = append(ops, dims...)
	ops = append(ops, inputs...)
	return g.NewNode(KindIterCluster, origin, ops...)
}

// isDimParam reports whether n is a RoleDim InputParam node.
func isDimParam(n Node) bool {
	role, ok := RoleOf(n)
	return ok && role == RoleDim
}

// ClusterDims returns the cluster's dim InputParam operands — the leading
// RoleDim operands of its input list (D14 read face).
func ClusterDims(n Node) []Node {
	var dims []Node
	for _, op := range OperandsOf(n) {
		if op == nil || !isDimParam(op) {
			break
		}
		dims = append(dims, op)
	}
	return dims
}

// ClusterInputs returns the cluster's non-dim operands in input order —
// the inputs IterClusterProject indexes into (D14 read face).
func ClusterInputs(n Node) []Node {
	ops := OperandsOf(n)
	dims := len(ClusterDims(n))
	if dims >= len(ops) {
		return nil
	}
	return append([]Node(nil), ops[dims:]...)
}

// forLoopNode: lowered loop nest; ops = [iv, bound, body] — the bounded
// induction variable, its bound, and the body subtree (an LIR memory store, a
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
