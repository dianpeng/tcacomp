package schedule

// scheduler.go is the optimization-pass scheduler (quick-261018-bl0 task 3,
// R5): Materialize's default schedule is allocation order (D-7, the
// deliberately minimal ir-tier default); THIS is the pass-level scheduler
// the pipeline runs when it wants dependency order plus LICM. It is an
// ordinary Pass (PASS-01/IR-07): ID "schedule", no requirements, produces
// StageConstructed, preserves nothing.
//
// Run is three steps, all statement-level and built from the exported ir
// read face only (StmtNodes/OperandsOf/PosetDeps/LambdaBodies/DataDeps/
// EffectOf/Base.Statement) — the schedule package never reaches into ir
// internals:
//
//  1. stable topological sort of the statement list (deps via the
//     statement-ownership poset; ties keep the pre-existing list order);
//  2. LICM over lowered for-loops, innermost-first, hoisting invariant
//     pure computations before the outermost enclosing loop's statement;
//  3. a second stable topo sort to fold the hoisted statements into
//     dependency order.
//
// A dependency cycle (including a self-loop) fails closed with an error and
// leaves the block untouched.

import (
	"errors"
	"fmt"
	"sort"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the scheduler in the global pass registry
// (quick-261018-l36 requirement 9 point 4): importing this package — directly
// or through internal/opt/all — is what makes "schedule" available to
// BuildPipeline.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "schedule",
		Priority: 900,
		Factory:  func() opt.Pass { return Scheduler{} },
	})
}

// Scheduler is the statement scheduler + LICM pass. It is stateless: every
// Run re-derives the schedule from the graph, so re-runs are idempotent.
type Scheduler struct{}

// ID returns the pipeline name "schedule".
func (Scheduler) ID() string { return "schedule" }

// Requires: the scheduler has no ordering preconditions.
func (Scheduler) Requires() []string { return nil }

// Produces: builder-output stage (it reorders statements; it introduces no
// new node kinds), so the constructed legality window re-checks the graph.
func (Scheduler) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — every analysis is invalidated.
func (Scheduler) Preserved() []string { return nil }

// Run executes topo sort → LICM → topo sort. A non-materialized graph is
// refused with a deterministic message, never a partial schedule.
func (Scheduler) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil || !g.Materialized() {
		return errors.New("schedule: graph not materialized")
	}
	if err := topoSchedule(g); err != nil {
		return err
	}
	if err := licm(g); err != nil {
		return err
	}
	return topoSchedule(g)
}

// topoSchedule re-establishes dependency order over the block's statements
// with a stable Kahn sort: a statement's predecessors are the statements
// owning the defs its tree depends on (data slots then ImpactBy payload,
// ir.PosetDeps). Among ready statements the smallest list position wins, so
// statements with no ordering constraint keep their pre-existing relative
// order — on an already-ordered list the sort is the identity. A cycle
// remainder fails closed and leaves the list untouched.
func topoSchedule(g *ir.Graph) error {
	bb := g.Block()
	stmts := blockStatements(bb)
	if len(stmts) == 0 {
		return nil
	}

	owner := statementOwners(stmts)
	preds, succs := statementEdges(stmts, owner)

	pos := make(map[*ir.Statement]int, len(stmts))
	for i, s := range stmts {
		pos[s] = i
	}
	indeg := make(map[*ir.Statement]int, len(stmts))
	ready := make([]*ir.Statement, 0, len(stmts))
	for _, s := range stmts {
		indeg[s] = len(preds[s])
		if indeg[s] == 0 {
			ready = append(ready, s)
		}
	}

	var ordered []*ir.Statement
	for len(ready) > 0 {
		// Stable pick: the ready statement with the smallest list position.
		best := 0
		for i := 1; i < len(ready); i++ {
			if pos[ready[i]] < pos[ready[best]] {
				best = i
			}
		}
		s := ready[best]
		ready = append(ready[:best], ready[best+1:]...)
		ordered = append(ordered, s)
		for _, o := range succs[s] {
			indeg[o]--
			if indeg[o] == 0 {
				ready = append(ready, o)
			}
		}
	}

	if len(ordered) != len(stmts) {
		var stuck []uint64
		emitted := make(map[*ir.Statement]bool, len(ordered))
		for _, s := range ordered {
			emitted[s] = true
		}
		for _, s := range stmts {
			if !emitted[s] {
				stuck = append(stuck, s.ID())
			}
		}
		sort.Slice(stuck, func(i, j int) bool { return stuck[i] < stuck[j] })
		return fmt.Errorf("schedule: dependency cycle among statements (scheduled %d of %d, stuck: %v)",
			len(ordered), len(stmts), stuck)
	}

	// Apply: collect + relink. Statement IDs are stable (D-04) — only list
	// order changes.
	for _, s := range stmts {
		bb.Remove(s)
	}
	for _, s := range ordered {
		bb.Append(s)
	}
	return nil
}

// blockStatements collects the block's statements in list order.
func blockStatements(bb *ir.BasicBlock) []*ir.Statement {
	var out []*ir.Statement
	if bb == nil {
		return out
	}
	for s := bb.First(); s != nil; s = s.Next() {
		out = append(out, s)
	}
	return out
}

// statementOwners maps every node appearing in any statement tree to its
// owning statement. A node that roots a statement (Base.Statement != nil) is
// owned by exactly that statement — this is what makes pass-inserted
// statements (LICM hoists) win over the trees that still reference the
// hoisted node. Any other node is owned by the first statement in block
// order whose tree contains it.
func statementOwners(stmts []*ir.Statement) map[*ir.Base]*ir.Statement {
	owner := make(map[*ir.Base]*ir.Statement)
	for _, s := range stmts {
		for _, n := range ir.StmtNodes(s) {
			b := n.BasePtr()
			if root := b.Statement(); root != nil {
				owner[b] = root
				continue
			}
			if _, ok := owner[b]; !ok {
				owner[b] = s
			}
		}
	}
	return owner
}

// statementEdges builds the predecessor/successor maps of the statement
// dependency graph: for every owned node n of statement S, each live poset
// def of n (data slots, then ImpactBy payload) owned by some statement O
// contributes the edge O → S. A dep owned by S itself is an internal tree
// edge and contributes nothing — EXCEPT a genuine self-dependency (d == n,
// a node depending on itself), which is kept as a self-edge: it makes the
// statement unschedulable, which the cycle remainder reports. InputParam
// deps contribute nothing: kernel/lambda params are graph-owned leaves
// produced by no statement (available from kernel entry), and since store
// target accesses ride operand slots (Input/ImpactBy-only convention,
// quick-261019-ktm revision) a shared tensor/dim param would otherwise
// manufacture false cross-statement edges against the effect chains.
func statementEdges(stmts []*ir.Statement, owner map[*ir.Base]*ir.Statement) (map[*ir.Statement][]*ir.Statement, map[*ir.Statement][]*ir.Statement) {
	predSet := make(map[*ir.Statement]map[*ir.Statement]bool, len(stmts))
	succs := make(map[*ir.Statement][]*ir.Statement, len(stmts))
	for _, s := range stmts {
		for _, n := range ir.StmtNodes(s) {
			for _, d := range ir.PosetDeps(n) {
				if d == nil {
					continue
				}
				if d.BasePtr().KindOf() == ir.KindInputParam {
					continue // graph-owned leaf: produced by no statement, order-free
				}
				o := owner[d.BasePtr()]
				if o == nil {
					continue // unowned (inline leaf): no statement edge
				}
				if o == s && d.BasePtr() != n.BasePtr() {
					continue // internal tree edge: the dep lives inside this very statement
				}
				if predSet[s] == nil {
					predSet[s] = make(map[*ir.Statement]bool)
				}
				if predSet[s][o] {
					continue
				}
				predSet[s][o] = true
				succs[o] = append(succs[o], s)
			}
		}
	}
	preds := make(map[*ir.Statement][]*ir.Statement, len(stmts))
	for s, set := range predSet {
		for o := range set {
			preds[s] = append(preds[s], o)
		}
	}
	return preds, succs
}

// licmHoistableKinds is the whitelist of kinds that may move out of a loop
// body (pure scalar expressions and parameter leaves). Everything else —
// memory family, loops, IV objects, stages — is pinned in place.
var licmHoistableKinds = map[ir.Kind]bool{
	ir.KindConst:      true,
	ir.KindInputParam: true,
	ir.KindParam:      true,
	ir.KindBinOp:      true,
	ir.KindUnaryOp:    true,
	ir.KindSelect:     true,
	ir.KindCast:       true,
	ir.KindCall:       true,
}

// licm runs loop-invariant code motion over the lowered for-loops. For each
// statement (block order), loops are processed innermost-first and
// candidates are considered in StmtNodes order. A candidate hoists only
// when it is invariant with respect to EVERY enclosing loop (else it stays
// for a future pass). Hoisting splices a new statement — via the same
// NewStatement factory Materialize uses — immediately before the EARLIEST
// statement whose tree already contains the candidate: consumers are met
// before their producers in StmtNodes order, so this insertion rule lands
// every producer left of its consumers and keeps the ownership map acyclic
// for the closing topo sort.
func licm(g *ir.Graph) error {
	bb := g.Block()
	for _, s := range blockStatements(bb) {
		nodes := ir.StmtNodes(s)
		loops := make([]ir.Node, 0, 2)
		for _, n := range nodes {
			if n.KindOf() == ir.KindForLoop {
				loops = append(loops, n)
			}
		}
		if len(loops) == 0 {
			continue
		}
		an := newLoopAnalysis(nodes, loops)
		for _, n := range nodes {
			if !licmCandidate(n, an) {
				continue
			}
			hoisted := g.NewStatement(n) // appends at the tail; repositioned below
			bb.Remove(hoisted)
			bb.InsertBefore(hoisted, firstStatementContaining(bb, n))
		}
	}
	return nil
}

// firstStatementContaining returns the earliest statement in block order
// whose tree contains n. The statement being scheduled always qualifies
// (its tree is where the candidate was found), so the result is never nil
// for a candidate taken from a statement tree.
func firstStatementContaining(bb *ir.BasicBlock, n ir.Node) *ir.Statement {
	for s := bb.First(); s != nil; s = s.Next() {
		for _, t := range ir.StmtNodes(s) {
			if t.BasePtr() == n.BasePtr() {
				return s
			}
		}
	}
	return nil
}

// loopAnalysis precomputes, for one statement, the per-loop body subtrees,
// varying sets, and parent maps the candidate checks consume.
type loopAnalysis struct {
	loops []*loopInfo
	pin   pinnedMemo
}

type loopInfo struct {
	loop    ir.Node
	body    map[*ir.Base]bool    // body subtree (operand + lambda closure)
	parent  map[*ir.Base]ir.Node // child -> parent within the body subtree
	varying map[*ir.Base]bool    // body nodes that transitively depend on iv/bound
	ancVary map[*ir.Base]bool    // ancestors (within the body) of varying nodes
}

// newLoopAnalysis builds the per-loop analysis for the loops found in one
// statement's node list. Loops are ordered innermost-first (a loop whose
// body contains another loop is the outer one).
func newLoopAnalysis(nodes []ir.Node, loops []ir.Node) *loopAnalysis {
	an := &loopAnalysis{pin: newPinnedMemo()}
	infos := make([]*loopInfo, 0, len(loops))
	for _, l := range loops {
		infos = append(infos, newLoopInfo(l))
	}
	// Innermost-first: sort by containment depth (outer loops contain more
	// body nodes).
	sort.SliceStable(infos, func(i, j int) bool { return len(infos[i].body) < len(infos[j].body) })
	an.loops = infos
	return an
}

// newLoopInfo computes the body subtree, parent map, and varying set of one
// loop. The varying set is the set of body nodes whose data-dependency
// closure reaches the loop's iv (operand 0) or bound (operand 1); ancestors
// of varying nodes (within the body) are recorded separately.
func newLoopInfo(loop ir.Node) *loopInfo {
	info := &loopInfo{
		loop:    loop,
		body:    make(map[*ir.Base]bool),
		parent:  make(map[*ir.Base]ir.Node),
		varying: make(map[*ir.Base]bool),
		ancVary: make(map[*ir.Base]bool),
	}
	ops := ir.OperandsOf(loop)
	if len(ops) < 3 {
		return info
	}
	iv, bound, body := ops[0], ops[1], ops[2]

	// Body subtree with parent edges (operand slots + lambda bodies).
	var visit func(n, par ir.Node)
	visit = func(n, par ir.Node) {
		if n == nil {
			return
		}
		b := n.BasePtr()
		if info.body[b] {
			return
		}
		info.body[b] = true
		if par != nil {
			info.parent[b] = par
		}
		for _, d := range ir.OperandsOf(n) {
			visit(d, n)
		}
		for _, lb := range ir.LambdaBodies(n) {
			visit(lb, n)
		}
	}
	visit(body, nil)

	// Varying: body nodes whose DataDeps closure hits iv or bound.
	memo := make(map[*ir.Base]bool)
	var reaches func(n ir.Node) bool
	reaches = func(n ir.Node) bool {
		if n == nil {
			return false
		}
		if sameNodePass(n, iv) || sameNodePass(n, bound) {
			return true
		}
		b := n.BasePtr()
		if v, ok := memo[b]; ok {
			return v
		}
		memo[b] = false // DAG in well-formed graphs; the guard also bounds corruption
		for _, d := range ir.DataDeps(n) {
			if reaches(d) {
				memo[b] = true
				break
			}
		}
		return memo[b]
	}
	for _, n := range bodyNodes(info) {
		if reaches(n) {
			info.varying[n.BasePtr()] = true
		}
	}

	// Ancestors of varying nodes, within the body.
	for vb := range info.varying {
		for p := info.parent[vb]; p != nil; p = info.parent[p.BasePtr()] {
			info.ancVary[p.BasePtr()] = true
		}
	}
	return info
}

// bodyNodes lists the body subtree nodes in deterministic (visit) order.
func bodyNodes(info *loopInfo) []ir.Node {
	seen := make(map[*ir.Base]bool, len(info.body))
	out := make([]ir.Node, 0, len(info.body))
	var walk func(n ir.Node)
	walk = func(n ir.Node) {
		if n == nil || seen[n.BasePtr()] {
			return
		}
		seen[n.BasePtr()] = true
		out = append(out, n)
		for _, d := range ir.OperandsOf(n) {
			walk(d)
		}
		for _, lb := range ir.LambdaBodies(n) {
			walk(lb)
		}
	}
	ops := ir.OperandsOf(info.loop)
	if len(ops) >= 3 {
		walk(ops[2])
	}
	return out
}

// enclosingLoops lists the loops whose body contains n, innermost-first.
func (an *loopAnalysis) enclosingLoops(n ir.Node) []*loopInfo {
	var out []*loopInfo
	for _, li := range an.loops {
		if li.body[n.BasePtr()] {
			out = append(out, li)
		}
	}
	return out
}

// licmCandidate reports whether n may hoist out of its enclosing loops: it
// is a whitelisted, non-leaf, effect-free node; nothing in its data-dependency
// closure is pinned (memory/loops/IV objects/effectful nodes); it is not
// already a statement root (idempotence); and it is invariant with respect
// to every enclosing loop — not varying, and not an ancestor of a varying
// node.
func licmCandidate(n ir.Node, an *loopAnalysis) bool {
	if n == nil {
		return false
	}
	if n.BasePtr().Statement() != nil {
		return false // already scheduled (root of some statement) — idempotence
	}
	if !licmHoistableKinds[n.KindOf()] {
		return false
	}
	if len(ir.OperandsOf(n)) == 0 {
		return false // leaf: nothing to compute, hoisting would only pollute the schedule
	}
	if ir.EffectOf(n).Has() {
		return false // belt & braces: whitelisted kinds are all free
	}
	if an.pin.pinned(n) {
		return false // operand closure touches memory/loops/IVs/effectful nodes
	}
	enclosing := an.enclosingLoops(n)
	if len(enclosing) == 0 {
		return false
	}
	for _, li := range enclosing {
		if li.varying[n.BasePtr()] || li.ancVary[n.BasePtr()] {
			return false
		}
	}
	return true
}

// pinnedMemo memoizes the "pinned" (never-hoistable) predicate over the
// data-dependency closure: a node is pinned when it is not whitelisted, is
// effectful, or has a pinned dependency. Cycles (corrupted graphs) are
// treated as pinned — the topological sort reports them first anyway.
type pinnedMemo struct {
	state map[*ir.Base]int // 0 unknown, 1 in progress, 2 pinned, 3 free
}

func newPinnedMemo() pinnedMemo {
	return pinnedMemo{state: make(map[*ir.Base]int)}
}

func (m pinnedMemo) pinned(n ir.Node) bool {
	return m.visit(n)
}

func (m pinnedMemo) visit(n ir.Node) bool {
	if n == nil {
		return false
	}
	b := n.BasePtr()
	switch m.state[b] {
	case 1:
		return true // in progress: cycle — treat as pinned (fail safe)
	case 2:
		return true
	case 3:
		return false
	}
	m.state[b] = 1
	p := !licmHoistableKinds[n.KindOf()] || ir.EffectOf(n).Has()
	if !p {
		for _, d := range ir.DataDeps(n) {
			if m.visit(d) {
				p = true
				break
			}
		}
	}
	if p {
		m.state[b] = 2
	} else {
		m.state[b] = 3
	}
	return p
}

// sameNodePass reports whether a and b are the same IR node (shared Base).
func sameNodePass(a, b ir.Node) bool {
	return a != nil && b != nil && a.BasePtr() == b.BasePtr()
}
