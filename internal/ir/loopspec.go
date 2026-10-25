package ir

// loopspec.go is the FamilyLoop CATEGORY interface (quick-261022-f5o,
// ruling D-01): the loop nest forest analysis is ONE generic algorithm
// (internal/analysis/loopforest) serving both loop dialects — the HIR
// KindIter suite (builder/simplify output, consumed by tiling) and the LIR
// KindForLoop suite (tile-lower output, consumed by the schedule pass) —
// and the only thing the algorithm does not know is each dialect's
// OPERAND LAYOUT. That knowledge is ir's alone (catalog.go's FamilyLoop
// rows), so it lives here behind two layout facts:
//
//   - IsLoop: FamilyLoop's loop-header membership. KindIter and KindForLoop
//     only — KindIterCluster is a compound header the simplify pass lowers
//     into iters (never a forest member), and KindDimVar/KindBoundedDimVar
//     are value objects, not loops.
//   - ParentEdge: the EXPLICIT nesting operand. An iter carries its
//     parental iter as an operand (node_loop.go D3: "Iter node will have
//     its parental loop, if any, as its second input"); a forloop has no
//     parent operand — its nesting is derived from IV data-dep references
//     (fix1.md D-b ruling: loop L_in is nested in L_out iff L_out's IV
//     dimvar is in L_in's data-dep operand closure).
//
// WHY AN INTERFACE AT ALL (ruling D-04, the quick-261022-b2m lesson): the
// generic LFA links loops by these MEMBERSHIP edges only — ParentEdge and
// IV-data-dep derivation — NEVER by the uniform data+effect closure.
// tile-lower's repointChains (lower_rewrite.go P6) re-points every
// post-nest consumer's ImpactBy edge at a FOREIGN nest's root loop ("the
// ForLoop node as effect-def" denotes whole-loop completion), so loop
// nodes are uniform-reachable from other nests' bodies: a closure-based
// link would cross-nest sibling LNTs and destroy the loop-nest-tree
// decomposition. Keeping the linking facts explicit — and effect edges
// absent from them — is what makes the shared algorithm safe for both
// dialects at every pipeline stage. At any stage exactly one dialect is
// live (tile-lower retires every iter), so the single spec below needs no
// stage detection; a hypothetical mixed graph still links coherently
// (iters via ParentEdge, forloops via IV-data-dep derivation).
//
// 2026-10-24 (fix1.md §0, D-b ruling): BodyRoot was deleted. ForLoop has
// no body operand (sea-of-nodes: loops are control-flow nodes with no
// direct link to expressions). LIR nesting is derived from IV data-dep
// references in loopforest, not from a body operand.

// LoopSpec is the FamilyLoop category behavior: the operand-layout facts
// one generic loop-nest forest analysis consumes (see the file comment for
// the membership-edges-only contract). Implementations must be pure read
// faces — no allocation, no mutation, nil-safe.
type LoopSpec interface {
	// IsLoop reports whether n is a loop-header node (KindIter or
	// KindForLoop). KindIterCluster, KindDimVar, and every non-FamilyLoop
	// kind answer false.
	IsLoop(n Node) bool
	// ParentEdge returns n's EXPLICIT parent-loop operand — for a KindIter,
	// the first loop-kind operand in DataDeps slot order (the doc's
	// single-parent convention: first wins); nil for a KindForLoop (its
	// nesting is derived from IV data-dep references, fix1.md D-b) and for
	// non-loop nodes. The returned node may be erased or uncollected; the
	// caller resolves membership.
	ParentEdge(n Node) Node
}

// LoopNesting is the single exported LoopSpec covering the whole FamilyLoop
// dialect span (quick-261022-f5o D-01). The generic loopforest analysis is
// driven by this value; nothing stage-detects.
var LoopNesting LoopSpec = loopNesting{}

// loopNesting implements LoopSpec over the catalog's operand layouts.
type loopNesting struct{}

// IsLoop: explicit switch — KindIter and KindForLoop only. IterCluster is
// a compound header (lowered to iters by simplify-lower), DimVar/BoundedDimVar
// are IV value objects; neither is a loop.
func (loopNesting) IsLoop(n Node) bool {
	if n == nil || n.BasePtr() == nil {
		return false
	}
	switch n.KindOf() {
	case KindIter, KindForLoop:
		return true
	}
	return false
}

// ParentEdge: a KindIter's operand list is [biv] or [biv, parent-iter] —
// scan the data-dep slots in order and return the FIRST loop-kind dep
// (exactly today's first-live-Iter-operand linking rule). Everything else,
// including KindForLoop (nesting derived from IV data-dep references,
// fix1.md D-b), answers nil.
func (loopNesting) ParentEdge(n Node) Node {
	if n == nil || n.BasePtr() == nil || n.KindOf() != KindIter {
		return nil
	}
	for _, d := range DataDeps(n) {
		if LoopNesting.IsLoop(d) {
			return d
		}
	}
	return nil
}
