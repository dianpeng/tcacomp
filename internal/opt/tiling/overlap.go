// overlap.go is the LOCAL MemLoad overlap analysis of
// doc/optana/loop_tiling.md ("MemLoad Overlap Analysis" + "Cache Tie
// Semantics"). The doc is explicit: this analysis is local to the tiling
// optimization — it is NOT registered in internal/analysis and nothing
// outside the pass consumes it.
//
// Two levels, never fused (the doc's "TWO distinct levels, do NOT fuse
// them"):
//
//   - SET level: bucket co-traversed loads at their innermost REFERENCED
//     iter — the level where the >= 2 admission filter and the rank/MaxCS
//     filters live.
//   - TIE level, per load: the innermost NON-referenced iter (RULE 1 — the
//     cache placement), whole-dimension coverage of everything hoisted
//     past (RULE 2), and tiling demotion of the covered dims (RULE 3).
//
// Documented interpretations (each cited where it fires):
//   - LNT ownership: a load belongs to the tree of the DEEPEST iter in its
//     INS (ties: earlier root, then smaller iter id). A load whose INS
//     intersects two trees is owned by exactly one LNT — the other skips it
//     entirely. This avoids site-key collisions in the record.
//   - Tie canonicalization: a tie computed at the OUTERMOST level is
//     recorded as nil. The placement is physically identical (directly
//     above the outermost loop, RULE 1) and the reuse arithmetic is
//     invariant (all-LNT-iters minus covered), so nil is the canonical
//     form. The clamp below may still land a tie AT the outermost level —
//     that one stays a real node, because its covered set is a strict
//     subset of INS and nil would mean "all of INS".
package tiling

import (
	"sort"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
)

// SiteFacts is one admitted candidate site's overlap product — everything
// the PSched search and the promotion judge need about the load.
type SiteFacts struct {
	// Load is the MemLoad node; Tensor is its operand 0.
	Load   ir.Node
	Tensor ir.Node
	// SetLevel is the SET-level bucket iter (the innermost referenced
	// level — where the co-traversed set lives).
	SetLevel ir.Node
	// Tie is the RULE-1 placement iter; nil = directly above the outermost
	// loop of the LNT.
	Tie ir.Node
	// Covered is the INS membership strictly inner than the tie level
	// (post-clamp): the dimensions the cache loads whole (RULE 2) and
	// tiling demotes to tiles (RULE 3).
	Covered []ir.Node
	// INS is every LNT iter the load references (ValueEqual-deduped,
	// tree-restricted).
	INS []ir.Node
	// Levels is the owning LNT's full iter list, inner -> outer — the
	// reuse-dim derivation walks it.
	Levels []ir.Node
	// AxesIters holds, per subscript position in order, the LNT iters
	// appearing in that axis's chased expression — the promotion judge's
	// per-position view.
	AxesIters [][]ir.Node
}

// lntFacts is one loop-nest tree's overlap product.
type lntFacts struct {
	// Tree is the forest root tree the facts belong to.
	Tree *loopforest.Tree
	// Levels lists the tree's iters inner -> outer (depth descending, node
	// id ascending within a depth) — the doc's SET/TIE scan order.
	Levels []ir.Node
	// CS is the admitted candidate set (doc order: set-size, rank, MaxCS
	// filters applied).
	CS []SiteFacts
}

// findINS returns every Iter node a MemLoad's subscripts reference
// (the doc's subroutine 0): for each axis operand, chase the Input list
// transitively via DataDeps with a visited set (deterministic worklist,
// nil/erased nodes skipped — fail-closed, no panic, T-ve6-01), stopping at
// KindInductionVariable nodes, whose operand 0 (the Iter) joins the result.
// Membership is ValueEqual-deduped across axes, exactly the doc's
// "Equality of Iter is defined by ValueEqual, just like GVN" — post-GVN
// equal iters are the same node, and the predicate keeps the stated
// contract honest.
func findINS(load ir.Node, se sideeffect.SideEffects) []ir.Node {
	ops := ir.OperandsOf(load)
	if len(ops) < 2 {
		return nil
	}
	var ins []ir.Node
	for _, axis := range ops[1:] {
		for _, it := range findAxisIters(axis) {
			if !iterIn(se, ins, it) {
				ins = append(ins, it)
			}
		}
	}
	return ins
}

// findAxisIters chases one subscript expression's Input list down to its
// InductionVariable leaves and returns the iters they point at. A plain
// worklist (index cursor, slot order) keeps discovery deterministic; the
// visited set makes cycles and shared subexpressions harmless.
func findAxisIters(axis ir.Node) []ir.Node {
	var out []ir.Node
	seen := make(map[ir.Node]bool)
	work := []ir.Node{axis}
	for i := 0; i < len(work); i++ {
		n := work[i]
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || seen[n] {
			continue // fail-closed: nil/erased nodes contribute nothing
		}
		seen[n] = true
		if n.KindOf() == ir.KindInductionVariable {
			deps := ir.DataDeps(n)
			if len(deps) > 0 {
				out = append(out, deps[0])
			}
			continue // the chase stops at the induction variable
		}
		work = append(work, ir.DataDeps(n)...)
	}
	return out
}

// iterIn reports whether x is a member of iters under the doc's ValueEqual
// semantics (identity fast path first — the post-GVN common case).
func iterIn(se sideeffect.SideEffects, iters []ir.Node, x ir.Node) bool {
	for _, it := range iters {
		if it == x {
			return true
		}
		if v, ok := it.(ir.Value); ok {
			if xv, ok2 := x.(ir.Value); ok2 && v.ValueEqual(xv, se) {
				return true
			}
		}
	}
	return false
}

// ownedLoad pairs a load with its INS for the per-tree ownership pass.
type ownedLoad struct {
	load ir.Node
	ins  []ir.Node
}

// analyzeOverlap computes every forest tree's overlap facts over g:
//
//   - OWNERSHIP: every live MemLoad with a non-empty INS is assigned to the
//     tree of its deepest INS member (ties: earlier root, then smaller iter
//     id) — loads never reference two trees ambiguously in the record.
//   - LEVELS: each tree's iters ordered inner -> outer.
//   - SET bucketing + admission: loads group at their innermost referenced
//     level; the doc-ordered filters (raw set length >= 2; per-element rank
//     >= CacheDims with unavailable rank failing closed; cumulative MaxCS
//     cap) reduce them to CS.
//   - TIE placement: per admitted load, the operational tie rule, the
//     covered set, and the CacheDims clamp.
//
// Trees are returned in Roots() order; a tree owning no load still gets an
// entry (its iters still need schedule decisions).
func analyzeOverlap(g *ir.Graph, f loopforest.Forest, se sideeffect.SideEffects, p Params) []lntFacts {
	roots := f.Roots()
	rootOrder := make(map[*loopforest.Tree]int, len(roots))
	member := make(map[ir.Node]*loopforest.Tree) // iter -> its ROOT tree
	facts := make([]lntFacts, 0, len(roots))
	for i, r := range roots {
		rootOrder[r] = i
		lf := lntFacts{Tree: r, Levels: levelsInnerToOuter(r)}
		facts = append(facts, lf)
		for _, it := range treeItersOf(r) {
			member[it] = r
		}
	}
	treeFacts := make(map[*loopforest.Tree]*lntFacts, len(facts))
	for i := range facts {
		treeFacts[facts[i].Tree] = &facts[i]
	}

	// Ownership pass: one allocation-order scan, loads appended to their
	// tree's list so per-tree order stays allocation order.
	owned := make(map[*loopforest.Tree][]ownedLoad)
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemLoad {
			continue
		}
		ins := findINS(n, se)
		if len(ins) == 0 {
			continue // no iter reference at all: no LNT owns it
		}
		var owner *lntFacts
		bestDepth, bestRoot, bestID := -1, 0, uint64(0)
		for _, it := range ins {
			root, ok := member[it]
			if !ok {
				continue
			}
			node, _ := f.NodeOf(it)
			ro := rootOrder[root]
			id := it.BasePtr().ID()
			if node.Depth > bestDepth ||
				(node.Depth == bestDepth && (ro < bestRoot || (ro == bestRoot && id < bestID))) {
				owner = treeFacts[root]
				bestDepth, bestRoot, bestID = node.Depth, ro, id
			}
		}
		if owner == nil {
			continue // INS holds no forest member (e.g. an erased iter): skip
		}
		owned[owner.Tree] = append(owned[owner.Tree], ownedLoad{load: n, ins: ins})
	}

	for i := range facts {
		facts[i].admit(f, se, p, owned[facts[i].Tree], member)
	}
	return facts
}

// admit performs the SET bucketing, the admission filters, and the TIE
// placement for one tree's owned loads, filling CS.
func (lf *lntFacts) admit(f loopforest.Forest, se sideeffect.SideEffects, p Params, loads []ownedLoad, member map[ir.Node]*loopforest.Tree) {
	if len(loads) == 0 {
		return
	}
	// SET bucketing: each load's bucket is the FIRST inner->outer level in
	// its INS (the innermost REFERENCED level).
	levels := lf.Levels
	groups := make(map[ir.Node][]ownedLoad)
	for _, ol := range loads {
		for _, level := range levels {
			if iterIn(se, ol.ins, level) {
				groups[level] = append(groups[level], ol)
				break
			}
		}
	}

	memberOf := func(it ir.Node) bool {
		return member[it] == lf.Tree
	}

	full := false
	for _, level := range levels { // deterministic: inner -> outer
		grp := groups[level]
		if len(grp) < 2 {
			continue // filter (a): singleton set — no co-traversal, no reuse
		}
		for _, ol := range grp {
			if full {
				break // filter (c): MaxCS reached — stop admitting entirely
			}
			if tensorRankOf(ol.load) < p.CacheDims {
				continue // filter (b): rank below CacheDims (unavailable rank fails closed)
			}
			lf.CS = append(lf.CS, newSiteFacts(f, se, p, ol, level, levels, memberOf))
			if len(lf.CS) >= p.MaxCS {
				full = true
			}
		}
		if full {
			break
		}
	}
}

// newSiteFacts derives one admitted load's tie placement: the operational
// tie rule (first inner->outer level NOT in INS), the canonicalization of
// an outermost tie to nil, the covered set (INS strictly inner than the
// tie; all tree-restricted INS for nil), and the CacheDims clamp. levels is
// the tree's inner->outer level order (the caller's lf.Levels).
func newSiteFacts(f loopforest.Forest, se sideeffect.SideEffects, p Params, ol ownedLoad, level ir.Node, levels []ir.Node, memberOf func(ir.Node) bool) SiteFacts {
	sf := SiteFacts{Load: ol.load, SetLevel: level, Levels: append([]ir.Node(nil), levels...)}
	ops := ir.OperandsOf(ol.load)
	if len(ops) > 0 {
		sf.Tensor = ops[0]
	}
	for _, it := range ol.ins {
		if memberOf(it) {
			sf.INS = append(sf.INS, it)
		}
	}
	for _, axis := range ops[1:] {
		var axisIters []ir.Node
		for _, it := range findAxisIters(axis) {
			if memberOf(it) && !iterIn(se, axisIters, it) {
				axisIters = append(axisIters, it)
			}
		}
		sf.AxesIters = append(sf.AxesIters, axisIters)
	}

	depthOf := func(it ir.Node) int {
		node, ok := f.NodeOf(it)
		if !ok {
			return -1
		}
		return node.Depth
	}

	// Operational tie rule: scan inner -> outer, stop at the first level
	// NOT referenced by the load.
	var tie ir.Node
	for _, l := range levels {
		if !iterIn(se, ol.ins, l) {
			tie = l
			break
		}
	}
	// Canonicalization: a tie AT the outermost level places the cache
	// directly above the outermost loop — identical to nil.
	if len(levels) > 0 && tie == levels[len(levels)-1] {
		tie = nil
	}
	coveredAt := func(t ir.Node) []ir.Node {
		if t == nil {
			return append([]ir.Node(nil), sf.INS...)
		}
		var out []ir.Node
		for _, it := range sf.INS {
			if depthOf(it) > depthOf(t) {
				out = append(out, it)
			}
		}
		return out
	}
	sf.Tie = tie
	sf.Covered = coveredAt(sf.Tie)

	// CLAMP: while covered exceeds CacheDims dims, move the tie inward one
	// level — the outermost covered iter leaves covered (it becomes a reuse
	// dim). This composes RULE 2 with the doc's "ONLY the last Cache.dims
	// dimensions may be promoted" and only fires for rank > CacheDims
	// loads. A tie landed by the clamp at the outermost level stays a real
	// node (nil would mean "all of INS").
	for len(sf.Covered) > p.CacheDims {
		var next ir.Node
		if sf.Tie == nil {
			next = levels[len(levels)-1]
		} else {
			for i := 1; i < len(levels); i++ {
				if levels[i] == sf.Tie {
					next = levels[i-1]
					break
				}
			}
			if next == nil {
				break // tie already innermost: cannot move further inward
			}
		}
		if next == nil {
			break
		}
		sf.Tie = next
		sf.Covered = coveredAt(sf.Tie)
	}
	return sf
}

// treeItersOf returns every iter of the tree rooted at root (DFS preorder).
func treeItersOf(root *loopforest.Tree) []ir.Node {
	var out []ir.Node
	var walk func(t *loopforest.Tree)
	walk = func(t *loopforest.Tree) {
		out = append(out, t.Iter)
		for _, c := range t.Children {
			walk(c)
		}
	}
	if root != nil {
		walk(root)
	}
	return out
}

// levelsInnerToOuter orders a tree's iters inner -> outer: depth
// descending, node id ascending within a depth (the deterministic scan
// order the SET/TIE rules consume).
func levelsInnerToOuter(root *loopforest.Tree) []ir.Node {
	if root == nil {
		return nil
	}
	iters := treeItersOf(root)
	depths := make(map[ir.Node]int, len(iters))
	var walk func(t *loopforest.Tree)
	walk = func(t *loopforest.Tree) {
		depths[t.Iter] = t.Depth
		for _, c := range t.Children {
			walk(c)
		}
	}
	walk(root)
	out := append([]ir.Node(nil), iters...)
	sort.SliceStable(out, func(i, j int) bool {
		if depths[out[i]] != depths[out[j]] {
			return depths[out[i]] > depths[out[j]]
		}
		return out[i].BasePtr().ID() < out[j].BasePtr().ID()
	})
	return out
}

// tensorRankOf returns the load's tensor rank, or -1 when unavailable
// (fail-closed — the admission filter drops such elements). The rank is
// read from the type cell first (the typeinfer pass's product, present in
// the real pipeline), falling back to the tensor InputParam's declared
// payload type so construction-shaped graphs carry their rank too.
func tensorRankOf(load ir.Node) int {
	ops := ir.OperandsOf(load)
	if len(ops) == 0 || ops[0] == nil {
		return -1
	}
	typ := ir.TypeOf(ops[0])
	if typ == nil {
		if declared, ok := ir.InputParamTypeOf(ops[0]); ok {
			typ = declared
		}
	}
	nt, ok := typ.(*ir.NumericType)
	if !ok {
		return -1
	}
	return nt.Rank()
}

// elemTypeOf returns the load's tensor element type (same dual read as
// tensorRankOf), reporting false when unavailable.
func elemTypeOf(load ir.Node) (ir.ElemType, bool) {
	ops := ir.OperandsOf(load)
	if len(ops) == 0 || ops[0] == nil {
		return 0, false
	}
	typ := ir.TypeOf(ops[0])
	if typ == nil {
		if declared, ok := ir.InputParamTypeOf(ops[0]); ok {
			typ = declared
		}
	}
	nt, ok := typ.(*ir.NumericType)
	if !ok {
		return 0, false
	}
	return nt.Elem(), true
}

// elemBytes returns an element type's byte width: bool/i8/u8 -> 1;
// f16/bf16/i16/u16 -> 2; f32/i32/u32 -> 4. An unknown future element type
// budgets as 4 (f32-class) — conservative for the SMEM gate, never zero.
func elemBytes(e ir.ElemType) int {
	switch e {
	case ir.TBool, ir.TI8, ir.TU8:
		return 1
	case ir.TF16, ir.TBF16, ir.TI16, ir.TU16:
		return 2
	case ir.TF32, ir.TI32, ir.TU32:
		return 4
	default:
		return 4
	}
}
