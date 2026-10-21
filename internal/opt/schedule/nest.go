// nest.go is the spec's Step 0 — the loop-nest forest analysis over the
// lowered KindForLoop nodes (quick-261021-ta3, doc/optana/schedule.md).
// It mirrors internal/analysis/loopforest's structure (collect → link →
// depth DFS, allocation-order determinism, maps for membership only) but
// differs in the EDGE MODEL on purpose: the iter forest links by reversed
// operand edges (a nested iter carries its parent as an operand), while
// ForLoop nesting is BODY-OPERAND CONTAINMENT (a nested ForLoop is
// reachable from the parent's ops[2] — node_loop.go M1), so the
// reversed-edge trick does not apply here.
package schedule

import "tcacomp/internal/ir"

// forNest is one loop-nest tree node over a live KindForLoop — the schedule
// pass's private forest type (the loopforest analysis package is HIR
// KindIter-only and stays untouched). depth follows the spec: the outermost
// loop is 1, each nesting level adds 1 ("outer most loop is 1").
type forNest struct {
	loop     ir.Node         // the KindForLoop this tree node represents
	parent   *forNest        // nil for roots
	children []*forNest      // loops nested in this one, allocation order
	depth    int             // root = 1, child = parent + 1
	body     map[ir.Node]bool // body-operand subtree containment set (ops[2] walk)
	block    *ir.ASTBlock    // the LBBN Loop block built for this node, stamped during scheduling
}

// forLoopForest derives the loop-nest forest over g's live (non-erased)
// KindForLoop nodes — the doc's Step 0 verbatim:
//
//  1. COLLECT: every non-erased KindForLoop in g.AllocatedNodes()
//     (allocation order) gets a forNest and starts as a root candidate —
//     the loopforest collect pattern.
//  2. LINK by body-operand containment: for each loop y, its body subtree
//     is the transitive operand closure from ops[2] (cycle-guarded). For
//     each loop x in allocation order, its parent is the FIRST loop y
//     (allocation order, y != x) whose body subtree contains x; x appends
//     to y.children and leaves the root set. Containment is reachability
//     in y's body walk: x's own subtree never contains its ancestors by
//     construction, so first-match is deterministic.
//  3. DEPTH via DFS from each root in root order (root = 1, child =
//     parent + 1), cycle-safe as in loopforest.
//
// A forest member that is not reachable from the sea roots (the Walk live
// set) is dead building debris: the caller refuses it fail-closed rather
// than scheduling garbage into the AST CFG.
func forLoopForest(g *ir.Graph) []*forNest {
	trees := make(map[ir.Node]*forNest)
	var order []ir.Node // live (non-erased) for-loops in allocation order
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindForLoop {
			continue
		}
		trees[n] = &forNest{loop: n, depth: 1}
		order = append(order, n)
	}

	for _, y := range order {
		trees[y].body = bodySetOf(y)
	}
	nonRoot := make(map[*forNest]bool)
	for _, x := range order {
		for _, y := range order {
			if x.BasePtr() == y.BasePtr() {
				continue
			}
			if trees[y].body[x] {
				xt, yt := trees[x], trees[y]
				xt.parent = yt
				yt.children = append(yt.children, xt)
				nonRoot[xt] = true
				break // first container in allocation order wins
			}
		}
	}

	roots := make([]*forNest, 0, len(order))
	for _, x := range order {
		if t := trees[x]; !nonRoot[t] {
			roots = append(roots, t)
		}
	}

	var setDepth func(t *forNest, d int)
	setDepth = func(t *forNest, d int) {
		t.depth = d
		for _, c := range t.children {
			setDepth(c, d+1)
		}
	}
	for _, r := range roots {
		setDepth(r, 1)
	}
	return roots
}

// bodySetOf computes the transitive operand-slot closure starting at n's
// ops[2] (the loop body), allocation-visit order, cycle-guarded: the set
// keyed by ir.Node identity answers body-containment questions
// ("is x inside y's loop body") in O(1). Payload (ImpactBy) edges are not
// part of the body-subtree relation — containment is operand structure
// only, exactly as the nest LINK step specifies.
func bodySetOf(n ir.Node) map[ir.Node]bool {
	set := make(map[ir.Node]bool)
	if n == nil {
		return set
	}
	ops := ir.OperandsOf(n)
	if len(ops) < 3 || ops[2] == nil {
		return set
	}
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || x.BasePtr().Erased() || set[x] {
			return
		}
		set[x] = true
		for _, d := range ir.OperandsOf(x) {
			visit(d)
		}
	}
	visit(ops[2])
	return set
}

// allNests lists every forest tree in allocation order of its loop node —
// the uniform iteration face for anchoring, ownership, and metadata (root
// order alone skips non-root trees).
func allNests(roots []*forNest) []*forNest {
	var out []*forNest
	var walk func(t *forNest)
	walk = func(t *forNest) {
		out = append(out, t)
		for _, c := range t.children {
			walk(c)
		}
	}
	for _, r := range roots {
		walk(r)
	}
	return out
}

// maxDepth returns the deepest nest depth under t (t included).
func (t *forNest) maxDepth() int {
	m := t.depth
	for _, c := range t.children {
		if d := c.maxDepth(); d > m {
			m = d
		}
	}
	return m
}
