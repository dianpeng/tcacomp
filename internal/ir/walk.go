package ir

// Visit enumerates every node reachable from g's single basic block in
// deterministic order — basic-block statement order, then operand-slot
// order, breadth-first with a visited set — calling visit exactly once per
// node. Statement def cells (once a pass wires them) and lambda bodies are
// walked as graph edges alongside operand uses, exactly as the verifier
// walks them.
//
// Visit performs no checking; it is the exported enumeration substrate for
// external deterministic walks (pass-manager legality windows, printers).
// The walk is iterative with a visited set, so malformed (even cyclic)
// graphs cannot overflow the stack (T-1-01-02).
func Visit(g *Graph, visit func(Node)) {
	if g == nil || visit == nil {
		return
	}
	seen := make(map[Node]bool)
	var queue []Node
	enqueue := func(n Node) {
		if n == nil || seen[n] {
			return
		}
		seen[n] = true
		queue = append(queue, n)
	}
	for _, s := range g.block.statements {
		enqueue(s.def)
		enqueue(s.rhs)
	}
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		visit(n)
		if impl := asImpl(n); impl != nil {
			impl.forEachOperand(func(u *Use) { enqueue(u.Val) })
		}
		if lb, ok := n.(lambdaBearer); ok {
			lb.forEachLambda(func(lam *Lambda) { enqueue(lam.Body) })
		}
	}
}
