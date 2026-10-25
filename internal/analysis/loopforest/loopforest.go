// Package loopforest implements the loop-nest forest analysis of
// doc/optana/loop_fusion.md ("Loop Nests/Forrest analysis" — the live
// spec; the older "Loop Fusion" prose above it is superseded by GVN): it
// enumerates ALL loop nests over the Level-1 sea so consumers — loop
// tiling per doc/optana/loop_tiling.md, the final schedule pass per
// doc/optana/schedule.md — can walk the nest structure without re-deriving
// it.
//
// GENERIC OVER THE LOOP CATEGORY (quick-261022-f5o): the analysis is ONE
// algorithm serving BOTH loop dialects, driven by the ir-owned
// ir.LoopNesting spec (internal/ir/loopspec.go, ruling D-01) — the HIR
// KindIter suite (links by the EXPLICIT parent operand, ParentEdge) and
// the LIR KindForLoop suite (links by IV DATA-DEP REFERENCE DERIVATION,
// fix1.md D-b: loop L_in is nested in L_out iff L_out's IV dimvar is in
// L_in's data-dep operand closure). At any pipeline stage exactly one
// dialect is live (tile-lower retires every iter), so there is no stage
// detection; a hypothetical mixed graph links coherently under the same
// rules.
//
// MEMBERSHIP EDGES ONLY (ruling D-04, the quick-261022-b2m lesson — THE
// REASON THE SPEC INTERFACE EXISTS): linking uses ParentEdge and
// IV-data-dep reference derivation and NEVER the uniform data+effect
// closure. tile-lower's repointChains (lower_rewrite.go P6) re-points
// every post-nest consumer's ImpactBy edge at a FOREIGN nest's root loop
// ("the ForLoop node as effect-def" denotes whole-loop completion), so
// loop nodes are uniform-reachable from other nests' bodies: a
// closure-based link would cross-nest sibling LNTs and destroy the
// loop-nest-tree decomposition. Completion edges ORDER; membership edges
// NEST. Effect edges are NEVER linking input (C3).
//
// DEPTH BASE (ruling D-02): roots are depth 0, children parent+1 — the
// historic iter-forest convention. Every tiling consumer uses depths
// relatively; the schedule pass is the sole absolute consumer and maps
// tree.Depth+1 at its nest.go adapter.
//
// Rules:
//   - MEMBERS: live (non-erased) loop-header nodes only, per the spec's
//     IsLoop. Erased loops are skipped; IterCluster/DimVar are never
//     members.
//   - LINK: a node's parent is its spec ParentEdge target when that target
//     is a collected loop; otherwise the FIRST collected loop (allocation
//     order) whose IV dimvar appears in the candidate child's data-dep
//     operand closure. For LIR forloops: loop L_in is nested in L_out iff
//     L_out's IV dimvar is in L_in's data-dep operand closure (fix1.md D-b).
//     The closures are computed lazily (at most one per loop) and walk
//     OPERAND slots only — membership edges, never the uniform data+effect
//     closure (D-04). Effect edges are NEVER linking input (C3: completion
//     edges order, never nest).
//   - SEPARATE STRUCTURE: the forest is its own tree type, NEVER baked
//     into IR nodes — computed on the fly, one snapshot per Analyze call.
//   - READ-ONLY: per the analysis-package contract, Analyze never mutates
//     the IR (no MutContext; only AllocatedNodes/DataDeps/OperandsOf/
//     KindOf/BasePtr read faces).
package loopforest

import (
	"tcacomp/internal/analysis"
	"tcacomp/internal/ir"
)

// init self-registers the analysis in the global analysis registry
// (quick-261019-jck pattern; quick-261022-f5o D-06: name and result type
// unchanged): importing this package — directly or through
// internal/analysis/all — is the single act that makes it visible to
// analysis.Lookup/Registered/BuildAll.
func init() {
	analysis.Register(analysis.AnalysisSpec{
		Name:    "loop-forest", // stable public id — opt.Cache key convention
		Factory: func(g *ir.Graph) analysis.Analysis { return Analyze(g) },
	})
}

// Tree is one loop-nest tree node: a live loop node plus its forest links.
type Tree struct {
	Iter     ir.Node // the loop node (KindIter or KindForLoop) this tree node represents
	Parent   *Tree   // nil for roots
	Children []*Tree // loops nested in this one, allocation order
	Depth    int     // root = 0, child = parent + 1 (D-02: the 0-based convention)
}

// Forest is the loop-nest forest over one graph snapshot (read-only
// analysis). The zero value is the empty forest.
type Forest struct {
	trees map[ir.Node]*Tree // live loops -> their tree nodes
	roots []*Tree           // allocation order
}

// Name implements analysis.Analysis: the stable analysis name used by
// opt.Cache bookkeeping.
func (Forest) Name() string { return "loop-forest" }

// Roots returns the forest's root trees in allocation order. The returned
// slice is a fresh copy — callers may reorder it freely.
func (f Forest) Roots() []*Tree { return append([]*Tree(nil), f.roots...) }

// NodeOf returns loop's tree node, reporting whether it is a forest
// member. Fail-closed (T-uoq-03): nil, erased, foreign, and non-loop
// nodes report false.
func (f Forest) NodeOf(loop ir.Node) (*Tree, bool) {
	if loop == nil || loop.BasePtr() == nil || loop.BasePtr().Erased() {
		return nil, false
	}
	t, ok := f.trees[loop]
	return t, ok
}

// Analyze computes the loop-nest forest over g's live loops — the generic
// algorithm over ir.LoopNesting (D-02):
//
//  1. COLLECT: every live spec.IsLoop node in g.AllocatedNodes()
//     (allocation) order gets a Tree and starts as a root candidate.
//
//  2. LINK: for each loop x in allocation order — ParentEdge first: when
//     x's explicit parent operand p is a collected loop, p is x's parent.
//     Otherwise IV-data-dep derivation (fix1.md D-b): the FIRST loop y
//     (allocation order, y != x) whose IV dimvar appears in x's data-dep
//     operand closure becomes x's parent. The closures are computed
//     lazily (at most one per loop) and walk OPERAND slots only —
//     membership edges, never the uniform data+effect closure (D-04).
//     Effect edges are NEVER linking input (C3). x leaves the root set on
//     either link.
//
//  3. DEPTH: DFS from each root in Roots() order — root 0, child =
//     parent + 1. Cycle-safe by construction: mutually-parenting loops
//     (impossible under def-before-use allocation, defensive anyway) keep
//     each other out of the root set only when linked; an unlinked cycle
//     is never reached past, so the DFS still terminates.
//
// A nil graph yields the zero Forest (no roots, NodeOf always false).
func Analyze(g *ir.Graph) Forest {
	if g == nil {
		return Forest{}
	}
	spec := ir.LoopNesting
	trees := make(map[ir.Node]*Tree)
	var order []ir.Node // live loops in allocation order
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || !spec.IsLoop(n) {
			continue
		}
		trees[n] = &Tree{Iter: n}
		order = append(order, n)
	}

	// ivCaches: lazily computed data-dep operand closure per loop (the
	// IV-membership sets of the LINK step). Operand slots only, cycle-guarded.
	ivCaches := make(map[ir.Node]map[ir.Node]bool, len(order))
	closureOf := func(loop ir.Node) map[ir.Node]bool {
		if set, ok := ivCaches[loop]; ok {
			return set
		}
		set := operandClosure(loop)
		ivCaches[loop] = set
		return set
	}

	nonRoot := make(map[*Tree]bool)
	for _, x := range order {
		xt := trees[x]
		if p := spec.ParentEdge(x); p != nil {
			if pt, ok := trees[p]; ok && pt != xt {
				xt.Parent = pt
				pt.Children = append(pt.Children, xt)
				nonRoot[xt] = true
				continue
			}
		}
		// IV-data-dep derivation (fix1.md D-b): x nests under the first
		// loop y whose IV dimvar appears in x's data-dep operand closure.
		// Only for KindForLoop (HIR iters link via ParentEdge above).
		if x.KindOf() == ir.KindForLoop {
			xClosure := closureOf(x)
			for _, y := range order {
				if y == x || y.KindOf() != ir.KindForLoop {
					continue
				}
				yOps := ir.OperandsOf(y)
				if len(yOps) == 0 {
					continue
				}
				yIV := yOps[0] // the dimvar
				if yIV != nil && xClosure[yIV] {
					yt := trees[y]
					xt.Parent = yt
					yt.Children = append(yt.Children, xt)
					nonRoot[xt] = true
					break // first container in allocation order wins
				}
			}
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

// operandClosure computes the transitive operand-slot closure from root,
// allocation-visit order, cycle-guarded, skipping nil/erased nodes — the
// IV-membership-edge closure (never DataDeps+ImpactBy; D-04). Used to
// derive whether one forloop's IV dimvar appears in another forloop's
// operand closure (fix1.md D-b).
func operandClosure(root ir.Node) map[ir.Node]bool {
	set := make(map[ir.Node]bool)
	if root == nil {
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
	visit(root)
	return set
}
