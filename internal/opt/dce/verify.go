// verify.go is the post-DCE verification pass (follow-up to quick-261018-l36):
// it re-derives the two-root-set reachable set with the node-held marks and
// asserts the sweep's structural invariants, so a corrupted sweep fails loudly
// at the next pipeline step instead of surfacing as bogus codegen.
package dce

import (
	"errors"
	"fmt"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the verification pass: Priority 101 — immediately after
// dce (100), before anything else observes the swept graph. The registry scale
// in internal/opt/registry.go documents the anchors.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "dce-verify",
		Priority: 101,
		Requires: []string{"dce"},
		Factory:  func() opt.Pass { return VerifyPass{} },
	})
}

// VerifyPass asserts the post-sweep invariants over the Level-1 sea. It
// mutates nothing — the only writes are the transient marks, reset on entry
// and on exit. The three checks:
//
//   - DANGLING: no live use record points at an erased (tombstoned) node.
//     Erase kills the records it can see and refuses live-used nodes, so a
//     record surviving in a tombstone's ref list is bookkeeping corruption.
//
//   - REACHABLE-ERASED: nothing reachable from the two-root set is erased —
//     the sweep may only remove nodes no root path reaches.
//
//   - ALIVE→DEAD: no edge from an alive (reachable) node to a dead one. Read
//     from the DEF side (every live use of a remaining dead node must have a
//     dead user), which enumerates the SSA ref lists — a different storage
//     walk than the node-side traversal marking uses, so the two views are
//     cross-checked. In a well-formed graph all three checks are invariants
//     by construction (marking descends exactly the edges the ref lists
//     record); they exist to catch drift between the traversal relation and
//     the record bookkeeping — e.g. a future node family whose bearer
//     interfaces Walk and the ref lists disagree about.
//
// The checks that CAN fire only fire on corruption; a green run proves
// nothing was corrupted, not that the pass is correct.
type VerifyPass struct{}

// ID returns the pipeline name "dce-verify".
func (VerifyPass) ID() string { return "dce-verify" }

// Requires: the sweep must have run earlier in the pipeline — verifying an
// unswept graph asserts nothing about DCE.
func (VerifyPass) Requires() []string { return []string{"dce"} }

// Produces: builder-output stage — the pass introduces no node kinds and no
// mutations, so the constructed legality window re-checks the graph.
func (VerifyPass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — every analysis is invalidated. The pass writes
// (transient marks) even though it transforms nothing; staying consistent
// with the invalidate-all default keeps the contract simple.
func (VerifyPass) Preserved() []string { return nil }

// Run asserts the invariants above. Gates mirror the DCE pass: a nil graph
// and a materialized/CFG-carrying graph are deterministic errors evaluated
// before any mark is written, so a refused graph is untouched.
func (VerifyPass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("dce-verify: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("dce-verify: graph already materialized (the post-DCE invariants are Level-1 sea invariants)")
	}

	// DANGLING first, before the marks exist: it is a statement about ref
	// lists alone and names the sharpest failure (a use of a swept node).
	for _, e := range g.Tombstones() {
		for _, u := range ir.UsesOf(e) {
			return fmt.Errorf("dce-verify: dangling edge: node %d (%s) still uses erased node %d (%s)",
				idOf(u.User()), kindOf(u.User()), e.BasePtr().ID(), e.KindOf())
		}
	}

	// Re-derive the reachable set from the two-root set (same substrate the
	// sweep marks with), then check the sweep's output against it.
	g.ResetMarks()
	defer g.ResetMarks()
	ir.Walk(g, ir.VisitorFunc(func(n ir.Node) ir.VisitResult {
		n.BasePtr().MarkAlive()
		return ir.Descend
	}))

	// REACHABLE-ERASED: Walk reached a tombstone — some root path crosses a
	// swept node, i.e. the sweep removed observable code.
	for _, e := range g.Tombstones() {
		if e.BasePtr().Alive() {
			return fmt.Errorf("dce-verify: erased node %d (%s) is reachable from the two-root set",
				e.BasePtr().ID(), e.KindOf())
		}
	}

	// ALIVE→DEAD: every live use of a remaining-but-unmarked node must have
	// an unmarked user. Remaining dead nodes are conservative survivors
	// (cyclic dead clusters); an ALIVE user of one would be an edge the
	// marking traversal missed.
	for _, d := range g.AllocatedNodes() {
		if d.BasePtr().Alive() {
			continue
		}
		for _, u := range ir.UsesOf(d) {
			if user := u.User(); user != nil && user.BasePtr().Alive() {
				return fmt.Errorf("dce-verify: edge from alive node %d (%s) to dead node %d (%s)",
					user.BasePtr().ID(), user.KindOf(), d.BasePtr().ID(), d.KindOf())
			}
		}
	}
	return nil
}

// idOf nil-safely reads a node's ID for diagnostics (0 = nil node).
func idOf(n ir.Node) uint64 {
	if n == nil {
		return 0
	}
	return n.BasePtr().ID()
}

// kindOf nil-safely reads a node's kind for diagnostics (empty = nil node).
func kindOf(n ir.Node) ir.Kind {
	if n == nil {
		return ir.KindInvalid
	}
	return n.KindOf()
}
