// nest.go holds ONLY the schedule pass's pass-local derivations on top of
// the SHARED generic loop-nest forest analysis (quick-261022-f5o, D-05):
// the forest itself — collect/link/depth over both loop dialects — lives
// in internal/analysis/loopforest, driven by the ir-owned ir.LoopNesting
// category spec. What stays here is what the analysis must NOT own:
//
//   - the forNest adapter shape the scheduler consumes (loop/parent/
//     children/depth + the per-nest .members set + the stamped block),
//   - memberSetOf — the ONE membership relation (fix1.md §0/D-b/D-g):
//     the non-loop nodes whose data-dep operand closure references the
//     loop's IV dimvar ("a use of L's IV is membership in L"), with
//     ForLoop nodes OPAQUE to the walk. It feeds BOTH containment caps
//     (capOp, the HI-02 effectful floor's reference; capUni, the CR-01
//     placement bound/ownership reference) — the old body-operand and
//     uniform-effect-chain derivations are deleted with the body model
//     (C4): the fill's epoch membership rides its coordinates' IV
//     references (fix1.md §1.4), so the old effect-chain inclusion is
//     subsumed by the IV relation.
//   - completionCutOf — the C3 completion cut: a node whose data/effect
//     closure TOUCHES a KindForLoop node ("def = ForLoop node → after
//     whole loop") is ordered after that loop's whole nest, so its
//     placement cap is cut below the loop's depth. This is D-c's
//     incoherence form resolved: a node that references L's IV (a member
//     by the relation above) yet is completion-ordered after L is AFTER
//     L, never inside it — the end node and every repointed post-nest
//     store place through this cut.
//   - reduceCompletionCutOf — the fold-completion cut (§5.1): consumers
//     of a KindReduce's whole-domain value order after every walk the
//     fold reads through, so the piped-projection store family lands in
//     the reduced nest's epilogue instead of inheriting the walk's
//     membership depth through the fold (the rejected Option D).
//
// See the loopforest package header for the two rulings this file rides
// on: membership-edges-only linking (D-04 — repointChains completion edges
// order, never nest) and the 0-based depth convention (D-02) — this
// adapter maps tree.Depth+1 to the scheduler's 1-based convention ("the
// outermost loop is 1", doc/optana/schedule.md) at this single point.
package schedule

import (
	"math"

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
	members  map[ir.Node]bool // IV-membership set (memberSetOf): both containment caps and ownership claims derive from it
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
// .members closure computed here per nest. The KindIter gate in the
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
			loop:    t.Iter,
			parent:  parent,
			depth:   t.Depth + 1, // D-02: analysis 0-based → scheduler 1-based
			members: memberSetOf(g, t.Iter),
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

// memberSetOf computes the ONE membership relation (fix1.md §0/Step 5):
// every allocated NON-LOOP node whose data-dep operand closure references
// the loop's IV dimvar (ops[0]) — the same relation loopforest links
// nesting with (D-b) and tiling's P6 inside-set re-derives (D-g).
// Sea-of-nodes membership is implicit: "a node is inside a loop's
// iteration because it uses an InductionVariable that references the
// loop". Notes:
//
//   - KindForLoop nodes are excluded as candidates AND the closure walk
//     treats them as opaque (referencesDataDep): a reference TO a loop
//     node is completion ORDER (C3 — "def = ForLoop node → after whole
//     loop"), never a path to its IV. Without the opacity the end node's
//     operand (the loop itself) would transitively "reference" the IV and
//     conflate whole-loop completion order with iteration membership.
//   - Member sets are chain-nested on real nests: an inner op references
//     the inner IV, whose index/dimvar operands close over the enclosing
//     IVs (D-b's probe-verified round-base fact), so cap(n) — the max
//     member depth — is the deepest loop n belongs to.
//   - Erased nodes are skipped; the per-node closure test is
//     cycle-guarded. O(N·E) per nest is documented as fine at v0 kernel
//     scale (hundreds of nodes).
func memberSetOf(g *ir.Graph, loop ir.Node) map[ir.Node]bool {
	set := make(map[ir.Node]bool)
	if g == nil || loop == nil {
		return set
	}
	ops := ir.OperandsOf(loop)
	if len(ops) == 0 || ops[0] == nil {
		return set
	}
	iv := ops[0]
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() == ir.KindForLoop {
			continue
		}
		if referencesDataDep(n, iv) {
			set[n] = true
		}
	}
	return set
}

// referencesDataDep reports whether target lies in n's transitive
// data-dep operand closure (cycle-guarded identity test over DataDeps).
// ForLoop nodes are OPAQUE: the walk matches the loop itself but never
// descends into its operands — completion-order references must not
// reach through a loop to its IV (C3; see memberSetOf).
func referencesDataDep(n, target ir.Node) bool {
	if n == nil || target == nil {
		return false
	}
	tid := target.BasePtr()
	seen := make(map[ir.Node]bool)
	var visit func(x ir.Node) bool
	visit = func(x ir.Node) bool {
		if x == nil || x.BasePtr() == nil || seen[x] {
			return false
		}
		seen[x] = true
		if x.KindOf() == ir.KindForLoop {
			return x.BasePtr() == tid
		}
		if x.BasePtr() == tid {
			return true
		}
		for _, d := range ir.DataDeps(x) {
			if visit(d) {
				return true
			}
		}
		return false
	}
	return visit(n)
}

// completionCutOf returns n's C3 COMPLETION CUT (fix1.md C3/D-c): one
// less than the depth of the SHALLOWEST loop whose KindForLoop node lies
// in n's transitive data/effect closure — "def = ForLoop node → after
// whole loop": a completion edge to L orders n after L's whole nest, so
// no placement may put n in or under L. The walk treats loops as opaque
// (the touch is the def itself, never a path through a loop to its IV),
// making this exactly D-c's incoherence form — a node that references L's
// IV (a member by memberSetOf) yet is completion-ordered after L —
// resolved as AFTER, never inside. depthOf maps loop nodes to nest
// depths; loops absent from it (dead debris) contribute nothing.
// math.MaxInt means n touches no loop (no cut).
func completionCutOf(n ir.Node, depthOf map[ir.Node]int) int {
	cut := math.MaxInt
	seen := make(map[ir.Node]bool)
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || seen[x] {
			return
		}
		seen[x] = true
		if x.KindOf() == ir.KindForLoop {
			if d, ok := depthOf[x]; ok && d-1 < cut {
				cut = d - 1
			}
			return // opaque: the touch orders, it is never a path
		}
		for _, d := range ir.DataDeps(x) {
			visit(d)
		}
		for _, d := range ir.EffectDeps(x) {
			visit(d)
		}
	}
	visit(n)
	return cut
}

// reduceCompletionCutOf returns n's FOLD-COMPLETION CUT (fix1.md §5.1's
// "KindReduce inside the tile-inside walk, piped map and store at the
// reduced nest's epilogue"): a KindReduce's value is a WHOLE-DOMAIN FOLD
// — it has no per-iteration value, so every node whose data/effect
// closure CONSUMES one is completion-ordered AFTER every walk the fold's
// feeding chain reads through. The cut is one less than the SHALLOWEST
// nest the reduce itself belongs to (reduceFloor maps each KindReduce to
// that depth, derived in computeCaps from the very member sets the
// membership relation produced). The REDUCE node itself is exempt — its
// own closure contains no reduce — so it stays a full member placed at
// its feeding walk (SN == cap), which is what §5.1's first clause pins;
// only its consumers drop past every touched nest into the epilogue.
// Every such consumer carries the folded value among its deps, so its
// propagated SN already equals the walk depth — the cut can only move it
// to the epilogue class, never reshuffle in-loop placement. Without this
// cut the consumers inherit the walk's full membership depth through the
// fold (SN == cap) and land INSIDE the reduced walk — fix1.md's rejected
// Option D ("closure-extended store depth ... violating R-A. Recorded so
// it is never retried"). math.MaxInt means n consumes no reduce.
//
// The walk mirrors completionCutOf's shape (ForLoop-opaque, data+effect),
// descending THROUGH reduce nodes: a reduce-after-reduce chain contributes
// every fold in the chain. Folds are recognized on the DESENT edges only —
// the entry node itself is exempt (a reduce is not its own consumer).
func reduceCompletionCutOf(n ir.Node, reduceFloor map[ir.Node]int) int {
	cut := math.MaxInt
	seen := make(map[ir.Node]bool)
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || seen[x] {
			return
		}
		seen[x] = true
		if x.KindOf() == ir.KindForLoop {
			return // opaque, as in every ordering walk
		}
		for _, d := range ir.DataDeps(x) {
			if d != nil && d.KindOf() == ir.KindReduce {
				if fl, ok := reduceFloor[d]; ok && fl-1 < cut {
					cut = fl - 1
				}
			}
			visit(d)
		}
		for _, d := range ir.EffectDeps(x) {
			if d != nil && d.KindOf() == ir.KindReduce {
				if fl, ok := reduceFloor[d]; ok && fl-1 < cut {
					cut = fl - 1
				}
			}
			visit(d)
		}
	}
	visit(n)
	return cut
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
