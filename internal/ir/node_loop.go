package ir

// node_loop.go holds the loop-structure nodes: the HIR abstract iteration
// suite (KindIter/KindIterCluster) and the LIR explicit loop nest
// (KindForLoop, D-3). The LIR induction-variable VALUE objects (KindDimVar/
// KindBoundedDimVar) are value-level material, not loop-nest structure — they
// live with their relations in node_dimvar.go.

// iterNode: the loop header; ops = [biv] or [biv, parent]. Operand 0 is
// the BoundedDimVar value object (((dim, init, step), max),
// doc/optana/induction_variable.md) — every iter is bounded by
// construction; the unbounded form is gone. Operand 1 is the parental
// iter, PRESENT ONLY when nested (doc/optana/loop_fusion.md: "Iter node
// will have its parental loop, if any, as its second input").
type iterNode struct {
	*Base
	operandList
}

// NewIter creates a loop header over the bounded induction variable bdv,
// optionally nested in the parental iter (nil = a nest root). The
// verifier's shape check pins operand 0 to a bounded dimvar, operand 1
// to an iter, and the operand count to {1, 2}.
func (g *Graph) NewIter(origin string, bdv, parent Node) Node {
	if parent == nil {
		return g.NewNode(KindIter, origin, bdv)
	}
	return g.NewNode(KindIter, origin, bdv, parent)
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

// forLoopNode: lowered loop nest; ops = [iv, bound, attrs...] — the
// bounded induction variable, its bound, and trailing KeyedAttribute
// annotations (M1, quick-261021-jay: the tile-lower pass's soc/dosoc/loc
// materialization metadata — the catalog arity is variadic and the verifier
// pins every trailing operand to a KeyedAttribute).
//
// Sea-of-nodes design (2026-10-24 author ruling, fix1.md §0): a loop node
// is a control-flow node with NO body. Membership is implicit via IV
// back-references; ordering is via data + side-effect edges only. There is
// never a direct link from a control-flow node to an expression.
type forLoopNode struct {
	*Base
	operandList
}

// NewForLoop creates an explicit lowered loop over the bounded induction
// variable iv, bounded by bound. attrs are optional trailing KeyedAttribute
// annotations (M1): the annotation vocabulary the loop-tile-lower pass emits
// is "soc" (0=seq, 1=cta, 2=thr — the materialization primitive selector),
// "dosoc" (the loop's map dimension), and "loc" (that loop's materialization
// concurrency). The verifier's shape check enforces the iv operand kind and
// the annotation operand kinds.
//
// There is no body operand: loops are control-flow nodes with no body
// (fix1.md §0 ruling). Membership is implicit via IV references; ordering
// rides data + chain edges.
func (g *Graph) NewForLoop(origin string, iv, bound Node, attrs ...Node) Node {
	n := g.allocNode(KindForLoop, origin, 2+len(attrs)).(*forLoopNode)
	ops := make([]Node, 0, 2+len(attrs))
	ops = append(ops, iv, bound)
	ops = append(ops, attrs...)
	wireOperands(n, ops...)
	return n
}
