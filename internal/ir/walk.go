package ir

// This file is the visitor-pattern graph walk (directive 3): fine-grained
// per-node visitation with early bailout and subtree pruning, replacing the
// old ir.Visit(g, func(Node)) BFS.
//
// Walk is an iterative pre-order DFS with an explicit stack and a visited
// set (cyclic-safe, no recursion — T-1.1-02). The roots are LEVEL-AWARE
// (two-level IR, quick-261017-vdj): at Level 1 they are the graph's
// parameter nodes (declaration order) plus the effect roots (effect.go — the
// sea has no statement list to scan); at Level 2 (after Materialize) they
// are the params plus the entry block's statements in statement order (a
// statement's valid def handle, then its rhs). A node's children are its
// operand defs in operand-slot order (live records only), then its payload
// uses (parameter references, iteration dims/bounds, lambda params, as-orders,
// store targets, ImpactBy effect inputs), then its lambda bodies.
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

// Walk visits every node reachable from g in deterministic pre-order DFS,
// calling v.Visit exactly once per node. The walk is iterative with a visited
// set, so malformed (even cyclic) graphs cannot overflow the stack
// (T-1.1-02). A nil graph or nil visitor is a no-op.
//
// The roots are level-aware (two-level IR, quick-261017-vdj):
//
//   - Level 1 (before Materialize): the graph's parameter nodes (declaration
//     order), then the effect roots (effect.go — every store, tile store, and
//     return registered at allocation) in registration order. There is no
//     block to scan; the effect roots are what keep side-effecting nodes and
//     their operand trees reachable.
//   - Level 2 (after Materialize): the parameters, then the entry block's
//     statements in statement order (a statement's valid def handle, then its
//     rhs).
//
// In both levels a node's children are its operand defs in operand-slot order
// (live records only), then its payload uses (parameter references, iteration
// dims/bounds, lambda params, as-orders, store targets and ImpactBy effect
// inputs), then its lambda bodies. The visitor's VisitResult controls the
// walk: Descend enqueues the children, SkipChildren prunes the subtree, Stop
// terminates the walk immediately. Order is fully deterministic (param order
// + root order + operand order + payload order + lambda order only, never map
// iteration).
//
// D-5 resolution: the canonical PRINTER does NOT walk this way — its
// StmtNodes DFS is an emission-order function (chain stages producer-first
// via pipeInput), and its order is golden-pinned. Walk is the substrate for
// analysis and checking (verify, legality); print/dot keep StmtNodes.
//
// The stack pops last-pushed first, so roots push in REVERSE visit order:
// effect roots / statements tail-to-head (first pops first), then params
// reversed (params pop first); within a statement rhs pushes before its def
// (def pops first).
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
	// Roots: graph params first (InputParam nodes are defs referenced only
	// through payload uses, so without them the def-use direction of parameter
	// ref lists would go unchecked), in declaration order.
	if g.materialized || g.cfg != nil {
		// Level 2: the entry block's statements, tail-to-head so the head
		// pops first; within a statement, rhs pushes before its def so the
		// def pops first.
		for s := g.cfg.entry.tail; s != nil; s = s.prev {
			push(s.rhs)
			if !s.def.IsInvalid() {
				push(s.def.SSA().Node())
			}
		}
	} else {
		// Level 1: the effect roots (stores, tile stores, returns) in
		// registration order — the sea's only side-effect-anchored roots.
		for i := len(g.effectRoots) - 1; i >= 0; i-- {
			push(g.effectRoots[i])
		}
	}
	for i := len(g.params) - 1; i >= 0; i-- {
		push(g.params[i])
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
		if pb, ok := n.(payloadBearer); ok {
			pb.forEachPayloadUse(func(u SSAUse) {
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
