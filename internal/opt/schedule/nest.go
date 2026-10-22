// nest.go holds ONLY the schedule pass's pass-local derivations on top of
// the SHARED generic loop-nest forest analysis (quick-261022-f5o, D-05):
// the forest itself — collect/link/depth over both loop dialects — lives
// in internal/analysis/loopforest, driven by the ir-owned ir.LoopNesting
// category spec. What stays here is what the analysis must NOT own:
//
//   - the forNest adapter shape the scheduler consumes (loop/parent/
//     children/depth + the per-nest .body/.reach sets + the stamped block),
//   - bodySetOf — the OPERAND-only body closure feeding capOp (the HI-02
//     effectful floor),
//   - reachSetOf — the UNIFORM closure (data + impact defs) feeding capUni
//     and ownership (quick-261022-b2m CR-01: tile-lower makes SMEM fills
//     loop members through effect chains, so containment must see them).
//
// See the loopforest package header for the two rulings this file rides
// on: membership-edges-only linking (D-04 — repointChains completion edges
// order, never nest) and the 0-based depth convention (D-02) — this
// adapter maps tree.Depth+1 to the scheduler's 1-based convention ("the
// outermost loop is 1", doc/optana/schedule.md) at this single point.
package schedule

import (
	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/ir"
)

// forNest is one loop-nest tree node over a live KindForLoop — the
// schedule pass's private mirror of a loopforest.Tree. depth follows the
// spec: the outermost loop is 1, each nesting level adds 1 ("outer most
// loop is 1") — the adapter's +1 mapping of the analysis's 0-based depth.
type forNest struct {
	loop     ir.Node          // the KindForLoop this tree node represents
	parent   *forNest         // nil for roots
	children []*forNest       // loops nested in this one, allocation order
	depth    int              // root = 1, child = parent + 1 (spec convention)
	body     map[ir.Node]bool // body-operand subtree containment set (ops[2] walk over OperandsOf) — the HI-02 effectful floor (capOp)
	reach    map[ir.Node]bool // UNIFORM containment set (ops[2] walk over DataDeps+ImpactBy defs, quick-261022-b2m CR-01) — placement cap and ownership
	block    *ir.ASTBlock     // the LBBN Loop block built for this node, stamped during scheduling
}

// forLoopForest derives the schedule pass's nest list over g's live
// KindForLoop nodes by MIRRORING the shared generic analysis
// (loopforest.Analyze over ir.LoopNesting — the same analysis tiling
// consumes; quick-261022-f5o D-05). The mirror keeps the forNest field
// set the scheduler reads (scheduler.go needs zero changes): loop from
// Tree.Iter, parent/children from the tree links (allocation order
// preserved by the mirror walk), depth = Tree.Depth + 1 (the single
// adaptation point of the D-02 depth-base ruling), and the pass-local
// .body/.reach closures computed here per nest. The KindIter gate in the
// scheduler's Run guarantees no HIR iters reach this forest; the analysis
// would link them coherently regardless.
//
// A forest member that is not reachable from the sea roots (the Walk live
// set) is dead building debris: the caller refuses it fail-closed rather
// than scheduling garbage into the AST CFG.
func forLoopForest(g *ir.Graph) []*forNest {
	f := loopforest.Analyze(g)
	var mirror func(t *loopforest.Tree, parent *forNest) *forNest
	mirror = func(t *loopforest.Tree, parent *forNest) *forNest {
		fn := &forNest{
			loop:   t.Iter,
			parent: parent,
			depth:  t.Depth + 1, // D-02: analysis 0-based → scheduler 1-based
			body:   bodySetOf(t.Iter),
			reach:  reachSetOf(t.Iter),
		}
		for _, c := range t.Children { // allocation order, preserved
			fn.children = append(fn.children, mirror(c, fn))
		}
		return fn
	}
	roots := make([]*forNest, 0, len(f.Roots()))
	for _, r := range f.Roots() { // allocation order
		roots = append(roots, mirror(r, nil))
	}
	return roots
}

// bodySetOf computes the transitive operand-slot closure starting at n's
// ops[2] (the loop body), allocation-visit order, cycle-guarded: the set
// keyed by ir.Node identity answers body-containment questions
// ("is x inside y's loop body") in O(1). Payload (ImpactBy) edges are not
// part of the body-subtree relation — this is the OPERAND-only set the
// HI-02 effectful floor consumes (capOp). Forest LINKING uses the same
// relation, but that now lives in the shared analysis (its
// membership-edges discipline, D-04).
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

// reachSetOf computes the UNIFORM containment closure starting at n's ops[2]
// (quick-261022-b2m, CR-01): identical to bodySetOf (cycle-guarded, starts
// at the body operand, skips nil/erased) except the walk expands ir.DataDeps
// AND ir.ImpactBy payload defs — the EXACT edge relation of tiling's
// lowerRun.reachableFrom (lower_rewrite.go). USER PRINCIPLE, locked for this
// task: the effect chain (ImpactBy) is treated IDENTICALLY to use-def/input
// edges throughout the scheduling model. Tile-lower's cache insertion
// (lower_cache.go PHASE 4, F8) places SMEM fills OUTSIDE the body-operand
// subtree on purpose — the fill becomes a loop member through the
// pass-built effect chains and re-executes per fill-key epoch — so an
// operand-only containment set gives every fill cap 0 and the old clamp
// hoisted multi-round fills into the once-only kernel prologue (stale SMEM
// windows, wrong numerics). The uniform set makes the fill's cap the depth
// of the deepest loop whose body reaches it through data OR impact edges:
// it is an impact-predecessor of the in-body buffer access.
//
// NOTE the asymmetry with forest LINKING (D-04): placement containment is
// UNIFORM (this set — a fill positioned by an effect chain belongs inside
// its host loop), but nest STRUCTURE is membership-edges-only (a
// completion edge to a foreign root must never nest two LNTs). The
// distinction is edge-direction semantics: an impact-PREDECESSOR of an
// in-body node is a same-iteration member; an impact edge AT a loop ROOT
// is "after the whole loop".
func reachSetOf(n ir.Node) map[ir.Node]bool {
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
		for _, d := range ir.DataDeps(x) {
			visit(d)
		}
		for _, u := range ir.ImpactBy(x) {
			visit(u.Def())
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
