package ir

// This file is the visitor-pattern graph walk (directive 3): fine-grained
// per-node visitation with early bailout and subtree pruning, replacing the
// old ir.Visit(g, func(Node)) BFS.
//
// Walk is an iterative pre-order DFS with an explicit stack and a visited
// set (cyclic-safe, no recursion — T-1.1-02). Roots come from the block's
// statements in statement order (a statement's valid def handle, then its
// rhs); a node's children are its operand defs in operand-slot order (live
// records only), then its lambda bodies. The visitor's VisitResult controls
// the walk: Descend enqueues the children, SkipChildren prunes the subtree,
// Stop terminates the walk immediately. Order is fully deterministic (BB
// statement order + operand order + lambda order only, never map iteration).
//
// D-5 resolution: the canonical PRINTER does NOT walk this way — its
// StmtNodes DFS is an emission-order function (chain stages producer-first
// via pipeInput), and its order is golden-pinned. Walk is the substrate for
// analysis and checking (verify, legality); print/dot keep StmtNodes.

// VisitResult is the per-node control returned by a Visitor.
type VisitResult int

const (
	// Descend continues the walk into the node's children (the default).
	Descend VisitResult = iota
	// SkipChildren visits the node but prunes its reachable descendants.
	SkipChildren
	// Stop terminates the walk immediately; remaining nodes are unvisited.
	Stop
)

// Visitor is the fine-grained node-visitation interface (directive 3).
type Visitor interface {
	Visit(n Node) VisitResult
}

// VisitorFunc adapts a function to the Visitor interface, so closures and
// named functions walk without a wrapper type.
type VisitorFunc func(n Node) VisitResult

// Visit implements Visitor.
func (f VisitorFunc) Visit(n Node) VisitResult { return f(n) }

// Walk visits every node reachable from g's single basic block in
// deterministic pre-order DFS — basic-block statement order roots, then
// operand-slot order, then lambda bodies — calling v.Visit exactly once per
// node. The walk is iterative with a visited set, so malformed (even cyclic)
// graphs cannot overflow the stack (T-1.1-02). A nil graph or nil visitor is
// a no-op.
func Walk(g *Graph, v Visitor) {
	if g == nil || v == nil {
		return
	}
	seen := make(map[Node]bool)
	var stack []Node
	push := func(n Node) {
		if n == nil || seen[n] {
			return
		}
		seen[n] = true
		stack = append(stack, n)
	}
	// Roots in statement order. The stack pops last-pushed first, so push
	// statements and the def/rhs pair in reverse: [stmtN..stmt0], and within
	// a statement rhs before its def (def pops first).
	stmts := g.block.statements
	for i := len(stmts) - 1; i >= 0; i-- {
		s := stmts[i]
		push(s.rhs)
		if !s.def.IsInvalid() {
			push(s.def.SSA().Node())
		}
	}
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		switch v.Visit(n) {
		case Stop:
			return
		case SkipChildren:
			continue
		case Descend:
		}
		var children []Node
		if impl := asImpl(n); impl != nil {
			impl.forEachOperand(func(u SSAUse) {
				if u.rec != nil && !u.rec.killed && u.rec.def != nil {
					children = append(children, u.rec.def.node)
				}
			})
		}
		if lb, ok := n.(lambdaBearer); ok {
			lb.forEachLambda(func(lam *Lambda) { children = append(children, lam.Body) })
		}
		for i := len(children) - 1; i >= 0; i-- {
			push(children[i])
		}
	}
}
