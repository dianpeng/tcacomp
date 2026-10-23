package schedule

// phases.go is the scheduling ALGORITHM decomposed into one method per
// design-doc step (quick-261022-fun, user-directed readability refactor):
// the pass's Run (scheduler.go) composes these phases in order —
//
//	gate → markLive → deriveForest (spec Step 0)
//	     → anchorSchedule (spec 3.1 + the SN -1 prototype anchors)
//	     → computeCaps → propagate (spec 3.2) → applyEffectfulFloor
//	     → classify → assignOwnership → resolveEpilogueHomes
//	     → buildKernel (spec Step 1) → newKernel (spec Step 2)
//
// — with scheduleRun's member data as the shared state between them (the
// old Run monolith rebuilt the same locals by hand through 400 lines).
// Every method keeps its ruling citations from the monolith; behavior is
// byte-identical (the scheduler/pipeline/net suites pin it unmodified).

import (
	"fmt"
	"sort"

	"tcacomp/internal/ir"
)

// scheduleRun carries ONE Run's shared state: the graph, the live-set
// snapshot, the derived forest, the SN anchoring/classification tables,
// and the output under construction. Methods are the doc's phases, in
// Run's composition order; determinism rides the allocation-order slices
// (nodes/roots/nests), never map iteration into output.
type scheduleRun struct {
	g          *ir.Graph
	nodes      []ir.Node        // g.AllocatedNodes() snapshot (allocation order)
	live       map[ir.Node]bool // walk-reachable live set (marks are Run-scoped)
	allocIndex map[ir.Node]int  // node → allocation index (topo tiebreak)

	// Step 0: the loop-nest forest (adapter over loopforest.Analyze).
	roots []*forNest // forest roots, allocation order
	nests []*forNest // every nest, allocation order of its loop node

	// SN state: anchors are FIXED during propagation (spec 3.1's "already
	// scheduled" control nodes + the SN -1 placeholders).
	anchored  map[ir.Node]bool // loop headers + prototype placeholders
	prototype *ir.ASTBlock     // the SN -1 signature block (quick-261022-e2u)
	capOp     map[ir.Node]int  // operand-containment cap (the HI-02 floor)
	capUni    map[ir.Node]int  // uniform-containment cap (placement bound)

	// classification: where each in-loop node's cluster is.
	epi   map[ir.Node]bool     // epilogue class: SN > capUni (after a nest)
	owner map[ir.Node]*forNest // in-loop class: the owning LNT root
	home  map[ir.Node]*forNest // epilogue class: the homed LNT root

	// output under construction.
	kernel  *ir.ASTBlock     // the Body group: [pro, lbbn, epi] clusters
	lnts    []*ir.ASTLNT     // per-LNT metadata, cluster order
	claimed map[ir.Node]bool // placed nodes (single placement bookkeeping)
}

// newScheduleRun allocates the run over g with its node snapshot; the
// live set and everything derived land in later phases.
func newScheduleRun(g *ir.Graph) *scheduleRun {
	nodes := g.AllocatedNodes()
	r := &scheduleRun{
		g:          g,
		nodes:      nodes,
		live:       make(map[ir.Node]bool, len(nodes)),
		allocIndex: make(map[ir.Node]int, len(nodes)),
		anchored:   make(map[ir.Node]bool, len(nodes)),
		claimed:    make(map[ir.Node]bool, len(nodes)),
	}
	for i, n := range nodes {
		r.allocIndex[n] = i
	}
	return r
}

// markLive builds the LIVE SET: everything reachable from the sea roots
// via ir.Walk (the same traversal checkLegality uses). Erased and
// unreachable building debris is never scheduled. The graph's transient
// marks are Run-scoped: Run resets them before and after (defer).
func (r *scheduleRun) markLive() {
	ir.Walk(r.g, ir.VisitorFunc(func(n ir.Node) ir.VisitResult {
		n.BasePtr().MarkAlive()
		return ir.Descend
	}))
	for _, n := range r.nodes {
		r.live[n] = n.BasePtr().Alive()
	}
}

// deriveForest is SPEC STEP 0: the loop-nest forest (the shared generic
// analysis via nest.go's adapter), then the pass's two fail-closed shape
// gates — a non-erased ForLoop unreachable from the sea roots is dead
// debris (refuse rather than schedule garbage), and a branched nest would
// let byDepth conflate siblings (tile-lower only builds nest CHAINS, its
// F3 contract — HI-01, quick-261022-b2m). Nests end in allocation order.
func (r *scheduleRun) deriveForest() error {
	r.roots = forLoopForest(r.g)
	r.nests = allNests(r.roots)
	for _, t := range r.nests {
		if !r.live[t.loop] {
			return fmt.Errorf("schedule: for-loop node %d is not reachable from the sea roots",
				t.loop.BasePtr().ID())
		}
	}
	for _, t := range r.nests {
		if len(t.children) > 1 {
			return fmt.Errorf("schedule: branched loop nest under for-loop %d (depth %d, %d children) — nests must be chains (tile-lower F3 contract)",
				t.loop.BasePtr().ID(), t.depth, len(t.children))
		}
	}
	sort.SliceStable(r.nests, func(i, j int) bool {
		return r.allocIndex[r.nests[i].loop] < r.allocIndex[r.nests[j].loop]
	})
	return nil
}

// anchorSchedule is SPEC 3.1 (SN initialization): every anchor is FIXED
// during propagation — the loop headers at their nest depth ("already
// scheduled" control nodes; a shared header takes the MAX of its anchors,
// order-independent) and the placeholder nodes at SN -1 (the signature
// scope, quick-261022-e2u D-1). Every other live node starts at SN 0.
func (r *scheduleRun) anchorSchedule() {
	r.prototype = ir.NewASTPrototype()
	for _, t := range r.nests {
		r.anchor(t.loop, t.depth)
		if ops := ir.OperandsOf(t.loop); len(ops) >= 2 {
			r.anchor(ops[0], t.depth) // iv
			r.anchor(ops[1], t.depth) // bound
		}
	}
	for _, n := range r.nodes { // allocation order: deterministic item order
		if !r.live[n] {
			continue
		}
		switch n.KindOf() {
		case ir.KindInputParam:
			// P-9 (quick-261022-l1o): lambda formals are NOT kernel-signature
			// placeholders — the reduce-inline exemption leaves a combiner's
			// acc/now InputParams alive as declared formals, and they are
			// extracted-function material (the review E2 ripple-5 model), not
			// kernel parameters. Un-anchored they stay ordinary SN-0 nodes
			// (pure, no operands) landing in the kernel Group like the lambda
			// body — never items of the Prototype block.
			if role, ok := ir.RoleOf(n); ok && role == ir.RoleLambda {
				continue
			}
			r.anchor(n, ir.PrototypeSN)
			r.prototype.AppendItem(ir.ASTItem{Node: n})
			n.BasePtr().SetSched(ir.PrototypeSN, r.prototype)
		case ir.KindMemNew:
			// Placeholders to known states, NOT statements (e2u D-1): a
			// MemNew's extent dep at SN 0 must not lift it out of the
			// signature scope; a -1 dep contributes nothing to the max.
			// MemNewSMem/FragNew are in-kernel declarations (D-4) and stay
			// scheduled; extent consts are values and stay scheduled.
			r.anchor(n, ir.PrototypeSN)
			r.prototype.AppendItem(ir.ASTItem{Node: n})
			n.BasePtr().SetSched(ir.PrototypeSN, r.prototype)
		}
	}
}

// anchor marks n FIXED at depth (max-wins for shared nodes): anchored
// nodes never update during propagation and are excluded from rank
// collection and ownership.
func (r *scheduleRun) anchor(n ir.Node, depth int) {
	if n == nil || !r.live[n] {
		return
	}
	if !r.anchored[n] || n.BasePtr().SchedSN() < depth {
		n.BasePtr().SetSched(depth, nil)
	}
	r.anchored[n] = true
}

// computeCaps derives the TWO CONTAINMENT CAPS (quick-261022-b2m): capOp
// is the deepest loop whose BODY-OPERAND subtree contains n (the HI-02
// floor's reference); capUni is the deepest loop whose body reaches n
// through the UNIFORM closure (data + impact-predecessor edges — CR-01's
// fix: tile-lower makes fills loop members via effect chains). capUni >=
// capOp by construction; nests are chains, so every contained node is
// contained at EVERY depth 1..capUni within its root.
func (r *scheduleRun) computeCaps() {
	r.capOp = make(map[ir.Node]int, len(r.nodes))
	r.capUni = make(map[ir.Node]int, len(r.nodes))
	for _, t := range r.nests {
		for n := range t.body {
			if t.depth > r.capOp[n] {
				r.capOp[n] = t.depth
			}
		}
		for n := range t.reach {
			if t.depth > r.capUni[n] {
				r.capUni[n] = t.depth
			}
		}
	}
}

// propagate is SPEC 3.2 run to fixpoint: SN(n) = max over DataDeps and
// EffectDeps (the uniform relation is the design intent — the effect
// chain is treated identically to input edges). Monotone-up and bounded
// by the deepest nest depth, so it terminates; anchored nodes never
// update.
func (r *scheduleRun) propagate() {
	for changed := true; changed; {
		changed = false
		for _, n := range r.nodes {
			if !r.live[n] || r.anchored[n] {
				continue
			}
			sn := n.BasePtr().SchedSN()
			for _, d := range ir.DataDeps(n) {
				if r.live[d] && d.BasePtr().SchedSN() > sn {
					sn = d.BasePtr().SchedSN()
				}
			}
			for _, d := range ir.EffectDeps(n) {
				if r.live[d] && d.BasePtr().SchedSN() > sn {
					sn = d.BasePtr().SchedSN()
				}
			}
			if sn > n.BasePtr().SchedSN() {
				n.BasePtr().SetSched(sn, nil)
				changed = true
			}
		}
	}
}

// applyEffectfulFloor is the HI-02 FLOOR (quick-261022-b2m): a live
// memory-ACCESS node (effectful, non-allocation) never hoists above its
// operand-containing loop — floor SN at capOp, then RE-propagate so
// floors flow to consumers (a kernel-level consumer of a floored in-loop
// load becomes an epilogue node, not a prologue reader). Allocations are
// exempt (isAllocationKind): hoisting a shared declaration is
// semantics-preserving (the buffer pin: SN 0).
func (r *scheduleRun) applyEffectfulFloor() {
	for _, n := range r.nodes {
		if !r.live[n] || r.anchored[n] || n.KindOf() == ir.KindForLoop {
			continue
		}
		if ir.EffectOf(n).Has() && !isAllocationKind(n.KindOf()) {
			if c := r.capOp[n]; c > n.BasePtr().SchedSN() {
				n.BasePtr().SetSched(c, nil)
			}
		}
	}
	r.propagate()
}

// classify is the TWO-CASE PLACEMENT decision (no clamp,
// quick-261022-b2m): a live non-anchored non-ForLoop node is IN-LOOP
// class when SN <= capUni (places AT its SN — SN 0 is the first prologue,
// LICM intact) and EPILOGUE class when SN > capUni (the deps that
// produced the uncontainable SN live in a nest this node is not contained
// in, so it executes AFTER that nest — the end node, reduction shapes,
// post-nest stores).
func (r *scheduleRun) classify() {
	r.epi = make(map[ir.Node]bool, len(r.nodes))
	for _, n := range r.nodes {
		if !r.live[n] || r.anchored[n] || n.KindOf() == ir.KindForLoop {
			continue
		}
		if n.BasePtr().SchedSN() > r.capUni[n] {
			r.epi[n] = true
		}
	}
}

// assignOwnership is UNIFORM OWNERSHIP: an in-loop node at final SN r
// belongs to the LNT of the FIRST loop (allocation order) at depth r
// whose UNIFORM reach contains it — first-claim wins, single placement
// (the old body-only claim starved effect-chained members like the fills,
// CR-01). Map order stays irrelevant: first claim per node wins.
func (r *scheduleRun) assignOwnership() {
	r.owner = make(map[ir.Node]*forNest, len(r.nodes))
	for _, t := range r.nests { // allocation order of the loops
		rank := t.depth
		for n := range t.reach {
			if r.live[n] && n.KindOf() != ir.KindForLoop && !r.anchored[n] &&
				r.owner[n] == nil && n.BasePtr().SchedSN() == rank {
				r.owner[n] = nestRoot(t)
			}
		}
	}
}

// resolveEpilogueHomes assigns each epilogue-class node its cluster: the
// epilogue Group of the LAST root (cluster order) among the homes of its
// live PosetDeps with SN >= 1 — a ForLoop dep's home is its tree's root,
// an in-loop dep's home is its owner, an epilogue dep's home is its own
// home. A bounded deterministic fixpoint (each pass resolves one layer of
// epilogue consumer chains; len(nodes) passes is the hard terminator,
// until-stable is the real exit). A node with no resolvable home fails
// closed — defensive and unreachable: SN > capUni implies some live dep
// has SN >= 1.
func (r *scheduleRun) resolveEpilogueHomes() error {
	rootIndexOf := make(map[*forNest]int, len(r.roots))
	for i, root := range r.roots {
		rootIndexOf[root] = i
	}
	nestOf := make(map[ir.Node]*forNest, len(r.nests))
	for _, t := range r.nests {
		nestOf[t.loop] = t
	}
	r.home = make(map[ir.Node]*forNest, len(r.epi))
	for pass := 0; pass <= len(r.nodes); pass++ {
		changed := false
		for _, n := range r.nodes { // allocation order: deterministic
			if !r.epi[n] || r.home[n] != nil {
				continue
			}
			if best := r.bestEpilogueHome(n, rootIndexOf, nestOf); best != nil {
				r.home[n] = best
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	for _, n := range r.nodes {
		if r.epi[n] && r.home[n] == nil {
			return fmt.Errorf("schedule: node %d has no resolvable epilogue home (live deps carry no in-loop root)",
				n.BasePtr().ID())
		}
	}
	return nil
}

// bestEpilogueHome returns n's deepest-cluster dep home, or nil while any
// contributing epilogue dep is still unresolved (the caller retries next
// fixpoint pass).
func (r *scheduleRun) bestEpilogueHome(n ir.Node, rootIndexOf map[*forNest]int, nestOf map[ir.Node]*forNest) *forNest {
	var best *forNest
	for _, d := range ir.PosetDeps(n) {
		if d == nil || !r.live[d] || d.BasePtr().SchedSN() < 1 {
			continue
		}
		var dh *forNest
		switch {
		case d.KindOf() == ir.KindForLoop:
			if t, ok := nestOf[d]; ok {
				dh = nestRoot(t)
			}
		case r.epi[d]:
			dh = r.home[d] // nil on early passes: try again next pass
		default:
			dh = r.owner[d]
		}
		if dh == nil {
			continue
		}
		if best == nil || rootIndexOf[dh] > rootIndexOf[best] {
			best = dh
		}
	}
	return best
}

// buildKernel is SPEC STEP 1 (per-LNT scheduling) plus the cluster
// concatenation into the kernel Body group: the no-loop degenerate body,
// or one LEXICAL Group cluster per forest root in allocation order
// (quick-261023-lex: [head items, embedded LBBN, after-nest items]).
func (r *scheduleRun) buildKernel() error {
	r.kernel = ir.NewASTGroup()
	if len(r.roots) == 0 {
		return r.scheduleFlatBody()
	}
	for _, root := range r.roots { // forest-root allocation order
		if err := r.scheduleCluster(root); err != nil {
			return err
		}
	}
	return nil
}

// scheduleFlatBody is the no-loop degenerate shape (spec Step 2): the
// kernel body is a single Group holding every live non-placeholder node
// (all at rank 0), in dependency order. The prototype-anchored
// InputParam/MemNew placeholders are excluded (anchored set) — they live
// in the Prototype block (e2u D-1).
func (r *scheduleRun) scheduleFlatBody() error {
	var all []ir.Node
	for _, n := range r.nodes {
		if r.live[n] && n.KindOf() != ir.KindForLoop && !r.anchored[n] {
			all = append(all, n)
		}
	}
	ordered, err := topoRank(0, all, r.allocIndex)
	if err != nil {
		return err
	}
	for _, n := range ordered {
		r.kernel.AppendItem(ir.ASTItem{Node: n})
		n.BasePtr().SetSched(0, r.kernel)
		r.claimed[n] = true
	}
	return nil
}

// scheduleCluster builds ONE LNT's LEXICAL Group cluster (quick-261023-lex,
// the author ruling): the LNT record FIRST (ruling 7 — the Group's Nest()
// target), the LBBN (spec step 1.1), per-rank placement for the in-loop
// class (spec step 1.4) into the Group's HEAD region, the LBBN root as a
// statement INSIDE the Group — the nest is embedded, so the head region's
// declarations (runtime objects, hoisted invariants) lexically scope over
// it — and the after-nest (epilogue) class appended after the nest inside
// the SAME Group. The old [prologue Group, root, epilogue Group] sibling
// triple is deleted: a sibling prologue Group closes its C++ scope before
// the nest opens, stranding every declaration codegen places there; and a
// separate tail Group is redundant — after-nest placement is just an item
// position after the nest in the lexical Group (all old semantics — SN
// depth requirements, effect ordering, the end-node landing — carry over
// positionally).
func (r *scheduleRun) scheduleCluster(root *forNest) error {
	rootBlock, metas, err := buildLBBN(root)
	if err != nil {
		return err
	}
	lnt := &ir.ASTLNT{Root: root.loop, Loops: metas}
	lex := ir.NewASTGroupInNest(lnt)
	byDepth := blocksByDepth(root)
	for rank := 0; rank <= root.maxDepth(); rank++ {
		if err := r.placeRank(rank, lex, byDepth[rank], root); err != nil {
			return err
		}
	}
	lex.AppendItem(ir.ASTItem{Block: rootBlock}) // the nest: embedded, the head scopes over it
	if err := r.placeEpilogue(root, lex); err != nil {
		return err
	}
	r.kernel.AppendItem(ir.ASTItem{Block: lex})
	r.lnts = append(r.lnts, lnt)
	return nil
}

// blocksByDepth maps each nest depth to its Loop block (the chain's
// rank→block table the per-rank placement targets).
func blocksByDepth(root *forNest) map[int]*ir.ASTBlock {
	byDepth := make(map[int]*ir.ASTBlock, root.maxDepth())
	var stamp func(t *forNest)
	stamp = func(t *forNest) {
		byDepth[t.depth] = t.block
		for _, c := range t.children {
			stamp(c)
		}
	}
	stamp(root)
	return byDepth
}

// placeRank is SPEC STEP 1.4 for one rank of one cluster: collect this
// root's unclaimed in-loop nodes at the rank (rank 0 is prologue-shared —
// LICM's landing zone; deeper ranks are owner-checked), topologically
// order them, and stamp the persistent SN + block linkage (D-03).
func (r *scheduleRun) placeRank(rank int, prologue, target *ir.ASTBlock, root *forNest) error {
	if target == nil {
		target = prologue
	}
	var collect []ir.Node
	for _, n := range r.nodes { // allocation order: deterministic collection
		if !r.live[n] || r.claimed[n] || n.KindOf() == ir.KindForLoop {
			continue
		}
		if r.epi[n] {
			continue // epilogue class: placed after the rank loop
		}
		if n.BasePtr().SchedSN() != rank {
			continue
		}
		if rank == 0 || r.owner[n] == root {
			collect = append(collect, n)
		}
	}
	ordered, err := topoRank(rank, collect, r.allocIndex)
	if err != nil {
		return err
	}
	// Insert BEFORE the block's first sub-block item (quick-261023-lex):
	// buildLBBN pre-builds the nesting, so the rank-(r+1) loop block is
	// already an item of target — appending rank-r statements would place
	// them AFTER the very loop that consumes them (the old block-granular
	// order net masked this; the item-position net exposed it). Rank-r
	// statements at their required depth execute before the deeper loops —
	// always legal, and the only correct order when a deeper loop's
	// bounds/axes consume them.
	ins := len(target.Items())
	for i, it := range target.Items() {
		if it.Block != nil {
			ins = i
			break
		}
	}
	for _, n := range ordered {
		target.InsertItemAt(ins, ir.ASTItem{Node: n})
		ins++
		n.BasePtr().SetSched(rank, target) // persistent SN + linkage (D-03)
		r.claimed[n] = true
	}
	return nil
}

// placeEpilogue collects this root's homed after-nest consumers,
// allocation order, topoRank'ed for intra-block order (the rank label is
// DIAGNOSTIC only — placement keeps each node's own propagated SN, ruling
// 6: SN is the depth requirement, the epilogue Group the landing zone).
func (r *scheduleRun) placeEpilogue(root *forNest, epilogue *ir.ASTBlock) error {
	var epiCollect []ir.Node
	for _, n := range r.nodes { // allocation order: deterministic
		if r.epi[n] && !r.claimed[n] && r.home[n] == root {
			epiCollect = append(epiCollect, n)
		}
	}
	epiOrdered, err := topoRank(root.maxDepth()+1, epiCollect, r.allocIndex)
	if err != nil {
		return err
	}
	for _, n := range epiOrdered {
		epilogue.AppendItem(ir.ASTItem{Node: n})
		n.BasePtr().SetSched(n.BasePtr().SchedSN(), epilogue) // keep the propagated SN (ruling 6)
		r.claimed[n] = true
	}
	return nil
}

// newKernel assembles the SPEC STEP 2 kernel: the Prototype block, the
// Body clusters, and the metadata views — Params pass through; Gmem lists
// every live GMEM allocation in allocation order with its extent (op[0]);
// the LNT SoC metadata was assembled per cluster above. Do NOT set
// materialized, do NOT touch g.cfg (D-01).
func (r *scheduleRun) newKernel() *ir.ASTKernel {
	var gmem []ir.ASTGmemAlloc
	for _, n := range r.nodes {
		if r.live[n] && n.KindOf() == ir.KindMemNew {
			var extent ir.Node
			if ops := ir.OperandsOf(n); len(ops) > 0 {
				extent = ops[0]
			}
			gmem = append(gmem, ir.ASTGmemAlloc{New: n, Extent: extent})
		}
	}
	return &ir.ASTKernel{
		Prototype: r.prototype,
		Body:      r.kernel,
		Params:    r.g.Params(),
		Gmem:      gmem,
		LNTs:      r.lnts,
	}
}

// nestRoot walks to the forest root of t's tree (the LNT identity the
// claim pass compares against, in forest-root allocation order).
func nestRoot(t *forNest) *forNest {
	for t.parent != nil {
		t = t.parent
	}
	return t
}

// isAllocationKind reports the HI-02 floor EXEMPTION (quick-261022-b2m
// ruling 5): MemNew/MemNewSMem/FragNew are effectful under the fresh-object
// rule but are shared DECLARATIONS consumed at multiple depths — the SMEM
// window buffer is an operand of the shallow per-round fill AND of the deep
// in-body accesses. Flooring an allocation at its deepest accessor would
// place it after the fill and fire the cross-block ordering net; hoisting a
// declaration is semantics-preserving (the pin: the buffer lands at SN 0,
// first prologue). Only memory-ACCESS kinds (loads/stores) take the floor.
func isAllocationKind(k ir.Kind) bool {
	switch k {
	case ir.KindMemNew, ir.KindMemNewSMem, ir.KindFragNew:
		return true
	default:
		return false
	}
}

// buildLBBN is the spec's step 1.1 for one nest: recursively create the
// Loop BB for every nested loop (mapping each LIR loop node to its Loop
// BB — the LBBN), nest child blocks as Block items of their parent, and
// stamp each loop node's persistent linkage at its own Loop block. The
// prologue/epilogue Groups around the LBBN root are created by the caller.
// Returns the root block plus the outer→inner loop metadata.
func buildLBBN(t *forNest) (*ir.ASTBlock, []ir.ASTLoopMeta, error) {
	blk, err := ir.NewASTLoop(t.loop)
	if err != nil {
		return nil, nil, err
	}
	t.block = blk
	t.loop.BasePtr().SetSched(t.depth, blk) // the loop node lives in its own Loop block
	// The anchored header (iv/bound, ops[0]/ops[1]) belongs to the Loop
	// block too — the spec's header part of the loop structure. A header
	// shared by nested loops anchors at the MAX depth (max-wins anchoring),
	// so it lands at the deepest sharing loop: only the loop whose depth
	// equals the header's anchored SN stamps it.
	if ops := ir.OperandsOf(t.loop); len(ops) >= 2 {
		for _, h := range ops[:2] {
			if h != nil && h.BasePtr().SchedSN() == t.depth && h.BasePtr().SchedBlock() == nil {
				h.BasePtr().SetSched(t.depth, blk)
			}
		}
	}
	metas := []ir.ASTLoopMeta{{
		Loop:  t.loop,
		Soc:   socOf(blk.Kind()),
		Dosoc: blk.DoSoC(),
		Loc:   blk.LoC(),
	}}
	for _, c := range t.children {
		childBlk, childMetas, cerr := buildLBBN(c)
		if cerr != nil {
			return nil, nil, cerr
		}
		blk.AppendItem(ir.ASTItem{Block: childBlk})
		metas = append(metas, childMetas...)
	}
	return blk, metas, nil
}

// socOf maps a Loop block kind back to its soc encoding (0=seq, 1=cta,
// 2=thr) for the per-LNT metadata (spec 4.3[2]). Groups do not occur here.
func socOf(k ir.BlockKind) int {
	switch k {
	case ir.BlockSeqLoop:
		return 0
	case ir.BlockCTALoop:
		return 1
	case ir.BlockThrLoop:
		return 2
	default:
		return -1
	}
}
