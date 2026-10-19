// Package loopforest implements the loop-nest forest analysis of
// doc/optana/loop_fusion.md ("Loop Nests/Forrest analysis" — the live
// spec; the older "Loop Fusion" prose above it is superseded by GVN): it
// enumerates ALL loop nests over the Level-1 sea so future consumers —
// loop tiling per doc/optana/loop_tiling.md first among them — can walk
// the nest structure without re-deriving it.
//
// Rules:
//   - MEMBERS: live KindIter nodes only. Erased iters are skipped;
//     IterCluster/ForLoop/DimVar are NOT forest members.
//   - REVERSED EDGES: an IR edge iter(x) -> iter(y) (x carries y as an
//     operand — x is nested in y) becomes the forest edge y parent-of x,
//     and x leaves the root set. The parent is the FIRST live-Iter
//     operand in slot order — the doc's forest assumes each nested iter
//     has ONE parental loop riding its operand list, so first wins,
//     deterministic.
//   - SEPARATE STRUCTURE: the forest is its own tree type, NEVER baked
//     into IR nodes — computed on the fly, one snapshot per Analyze call.
//   - READ-ONLY: per the analysis-package contract, Analyze never mutates
//     the IR (no MutContext; only the AllocatedNodes/DataDeps/KindOf/
//     BasePtr read faces).
package loopforest

import (
	"tcacomp/internal/analysis"
	"tcacomp/internal/ir"
)

// init self-registers the analysis in the global analysis registry
// (quick-261019-jck pattern): importing this package — directly or
// through internal/analysis/all — is the single act that makes it visible
// to analysis.Lookup/Registered/BuildAll.
func init() {
	analysis.Register(analysis.AnalysisSpec{
		Name:    "loop-forest", // stable public id — opt.Cache key convention
		Factory: func(g *ir.Graph) analysis.Analysis { return Analyze(g) },
	})
}

// Tree is one loop-nest tree node: a live KindIter plus its forest links.
type Tree struct {
	Iter     ir.Node // the KindIter this tree node represents
	Parent   *Tree   // nil for roots
	Children []*Tree // iters nested in this one, allocation order
	Depth    int     // root = 0, child = parent + 1
}

// Forest is the loop-nest forest over one graph snapshot (read-only
// analysis). The zero value is the empty forest.
type Forest struct {
	trees map[ir.Node]*Tree // live iter -> its tree node
	roots []*Tree           // allocation order
}

// Name implements analysis.Analysis: the stable analysis name used by
// opt.Cache bookkeeping.
func (Forest) Name() string { return "loop-forest" }

// Roots returns the forest's root trees in allocation order. The returned
// slice is a fresh copy — callers may reorder it freely.
func (f Forest) Roots() []*Tree { return append([]*Tree(nil), f.roots...) }

// NodeOf returns iter's tree node, reporting whether it is a forest
// member. Fail-closed (T-uoq-03): nil, erased, foreign, and non-Iter
// nodes report false.
func (f Forest) NodeOf(iter ir.Node) (*Tree, bool) {
	if iter == nil || iter.BasePtr() == nil || iter.BasePtr().Erased() {
		return nil, false
	}
	t, ok := f.trees[iter]
	return t, ok
}

// Analyze computes the loop-nest forest over g's live iters — the doc's
// algorithm verbatim:
//
//  1. COLLECT: every live KindIter in g.AllocatedNodes() (allocation)
//     order gets a Tree and starts as a root candidate.
//  2. LINK (reversed edges): for each live iter x in allocation order,
//     scan ir.DataDeps(x) in slot order; the FIRST dep y that is a live
//     Iter collected in step 1 becomes x's parent — tree(x).Parent =
//     tree(y), tree(x) appends to tree(y).Children — and x leaves the
//     root set (it is reachable from its parent).
//  3. DEPTH: DFS from each root in Roots() order — root 0, child =
//     parent + 1. Cycle-safe by construction: mutually-referencing iters
//     (impossible under def-before-use allocation, defensive anyway)
//     remove each other from the root set and are never reached by the
//     DFS — they keep Depth 0 and the algorithm still terminates.
//
// A nil graph yields the zero Forest (no roots, NodeOf always false).
func Analyze(g *ir.Graph) Forest {
	if g == nil {
		return Forest{}
	}
	trees := make(map[ir.Node]*Tree)
	var order []ir.Node // live iters in allocation order
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindIter {
			continue
		}
		trees[n] = &Tree{Iter: n}
		order = append(order, n)
	}

	nonRoot := make(map[*Tree]bool)
	for _, x := range order {
		for _, d := range ir.DataDeps(x) {
			parent, ok := trees[d]
			if !ok {
				continue // not a live-iter operand — keep scanning slots
			}
			xt := trees[x]
			xt.Parent = parent
			parent.Children = append(parent.Children, xt)
			nonRoot[xt] = true // x is reachable from its parent: not a root
			break              // single-parent convention: first live-Iter operand wins
		}
	}

	roots := make([]*Tree, 0, len(order))
	for _, x := range order {
		if t := trees[x]; !nonRoot[t] {
			roots = append(roots, t)
		}
	}

	var setDepth func(t *Tree, d int)
	setDepth = func(t *Tree, d int) {
		t.Depth = d
		for _, c := range t.Children {
			setDepth(c, d+1)
		}
	}
	for _, r := range roots {
		setDepth(r, 0)
	}
	return Forest{trees: trees, roots: roots}
}
