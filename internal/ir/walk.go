package ir

// This file is the visitor-pattern graph walk (directive 3): fine-grained
// per-node visitation with early bailout and subtree pruning, replacing the
// old ir.Visit(g, func(Node)) BFS.
//
// Walk is an iterative pre-order DFS with an explicit stack and a visited
// set (cyclic-safe, no recursion — T-1.1-02). The roots are LEVEL-AWARE
// (two-level IR, quick-261017-vdj): at Level 1 they are the graph's
// parameter nodes (declaration order) plus the end node (R2 — effect.go);
// at Level 2 (after Materialize) they are the params plus the entry block's
// statements in statement order (a statement's valid def handle, then its
// rhs). A node's children are its operand defs in operand-slot order (live
// records only), then its payload uses (parameter references, iteration
// dims/bounds, lambda params, as-orders, store targets, ImpactBy effect
// inputs), then its lambda bodies.
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
//     order), then the end node (R2 — the graph's single KindReturn; the sea
//     has no block to scan). The end node's operand and ImpactBy paths are
//     what keep the side-effecting nodes and their operand trees reachable
//     (R3: the two-root set is the RoleTensor params + the end node, but the
//     walk conservatively roots at ALL params so def-use checking covers
//     every parameter's ref list).
//   - Level 2 (after Materialize): the parameters, then the entry block's
//     statements in statement order (a statement's valid def handle, then its
//     rhs).
//
// In both levels a node's children are its operand defs in operand-slot
// order (live records only), then its payload uses (the ImpactBy effect
// inputs). The visitor's VisitResult controls the
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
// Walk is the level-aware ROOT-SET wrapper over WalkFrom (fix2 D-d's
// extraction): it builds its root slice in the exact VISIT order the
// original monolith popped — params (declaration order) first, then the
// level-specific roots (Level 1: the end node; Level 2: the entry block's
// statements head→tail, each statement's def before its rhs).
func Walk(g *Graph, v Visitor) {
	if g == nil || v == nil {
		return
	}
	// Roots: graph params first (InputParam nodes are defs referenced only
	// through payload uses, so without them the def-use direction of parameter
	// ref lists would go unchecked), in declaration order.
	roots := make([]Node, 0, len(g.params)+1)
	roots = append(roots, g.params...)
	if g.materialized || g.cfg != nil {
		// Level 2: the entry block's statements head→tail, each statement's
		// def before its rhs (collected tail→head as rhs-before-def pairs,
		// then reversed into the visit order).
		var stmts []Node
		for s := g.cfg.entry.tail; s != nil; s = s.prev {
			stmts = append(stmts, s.rhs)
			if !s.def.IsInvalid() {
				stmts = append(stmts, s.def.SSA().Node())
			}
		}
		for i, j := 0, len(stmts)-1; i < j; i, j = i+1, j-1 {
			stmts[i], stmts[j] = stmts[j], stmts[i]
		}
		roots = append(roots, stmts...)
	} else {
		// Level 1: the end node (R2) — the sea's only effect-anchored root;
		// its operand and ImpactBy paths reach every live store.
		roots = append(roots, g.end)
	}
	WalkFrom(roots, v)
}

// WalkFrom visits every node reachable from the given roots in deterministic
// pre-order DFS, calling v.Visit exactly once per node — the same iterative
// stack/visited-set traversal and the same child relation as Walk (operand
// defs in slot order, then payload uses, live records only), but over an
// ARBITRARY root set (fix2 D-d: the mechanical enabler for DCE's AST-item
// roots). The roots are visited in slice order (roots push in REVERSE so the
// first pops first); a nil visitor or an empty/nil root slice is a no-op.
// Nil roots and roots already seen are skipped, so overlapping root sets are
// safe.
func WalkFrom(roots []Node, v Visitor) {
	if v == nil {
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
	for i := len(roots) - 1; i >= 0; i-- {
		push(roots[i])
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
		for i := len(children) - 1; i >= 0; i-- {
			push(children[i])
		}
	}
}
