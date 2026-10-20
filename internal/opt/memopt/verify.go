// verify.go is the post-memopt verification pass (quick-261020-ejf): it
// re-derives the eligible regions with the same shared derivation code
// (deriveRegions — mirroring gvn's shared lambdaBodyRoots pattern, the
// derivation is re-RUN, not shared state) and asserts the fixpoint and
// integrity invariants a clean memopt leaves behind, so a corrupted or
// incomplete optimization fails loudly at the next pipeline step instead
// of surfacing as bogus codegen.
package memopt

import (
	"errors"
	"fmt"

	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the verification pass: Priority 113 — immediately
// after memopt (112), before anything else observes the optimized graph.
// The registry scale in internal/opt/registry.go documents the anchors.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "memopt-verify",
		Priority: 113,
		Requires: []string{"memopt"},
		Factory:  func() opt.Pass { return VerifyPass{} },
	})
}

// VerifyPass asserts the post-memopt invariants over the Level-1 sea. It
// mutates nothing. The three checks, all over ELIGIBLE (C1-passing)
// regions only — a legally-unoptimized region is the guard's job to skip,
// never this pass's job to flag:
//
//   - INTEGRITY: no live GMEM access holds a killed or dangling use record
//     (an ImpactBy edge resolving at nothing, or at an erased node), and
//     no use of a live access has a nil/erased user (INV-A's memory face).
//
//   - FIXPOINT (INV-F): no forwardable load remains (a load whose
//     first-reached store behind it holds the same proven path), no
//     eliminable store pair remains (a proven-equal pair with only stores
//     between them), and no zero-use store remains (a store no access of
//     the region and not the Return observes).
//
//   - SCOPED USERS (INV-B): every effect user of every access is an access
//     of the same region or the Return — the chain never leaks into
//     foreign nodes.
//
// The checks that CAN fire only fire on a pass that stopped early or on
// bookkeeping corruption; a green run proves the fixpoint held, not that
// the pass was optimal.
type VerifyPass struct{}

// ID returns the pipeline name "memopt-verify".
func (VerifyPass) ID() string { return "memopt-verify" }

// Requires: the optimization must have run earlier in the pipeline —
// verifying an unoptimized graph asserts nothing about memopt (and in
// fact flags its forwardable loads).
func (VerifyPass) Requires() []string { return []string{"memopt"} }

// Produces: builder-output stage — the pass introduces no node kinds and
// no mutations, so the constructed legality window re-checks the graph.
func (VerifyPass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — every analysis is invalidated, consistent with the
// invalidate-all default.
func (VerifyPass) Preserved() []string { return nil }

// Run asserts the invariants above. Gates mirror the memopt pass: a nil
// graph and a materialized/CFG-carrying graph are deterministic errors
// evaluated before anything is read, so a refused graph is untouched.
func (VerifyPass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("memopt-verify: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("memopt-verify: graph already materialized (the post-memopt invariants are Level-1 sea invariants)")
	}
	se := sideeffect.Analyze(g)
	regions := deriveRegions(g)

	// 1. INTEGRITY — post-forward/eliminate/sweep SSA bookkeeping, checked
	// on every live GMEM access regardless of eligibility (a killed record
	// lingering in a skipped region's bearer is corruption too).
	for _, n := range g.AllocatedNodes() {
		if !isGMEMAccess(n.KindOf()) {
			continue
		}
		for _, e := range ir.ImpactBy(n) {
			if e.IsKilled() {
				return fmt.Errorf("memopt-verify: node %d (%s) holds a killed ImpactBy use record",
					n.BasePtr().ID(), n.KindOf())
			}
			if d := e.Def(); d == nil || d.BasePtr().Erased() {
				return fmt.Errorf("memopt-verify: dangling edge: node %d (%s) keeps an ImpactBy edge at an erased/nil def",
					n.BasePtr().ID(), n.KindOf())
			}
		}
		for i, op := range ir.OperandsOf(n) {
			if op != nil && op.BasePtr().Erased() {
				return fmt.Errorf("memopt-verify: dangling edge: node %d (%s) operand %d points at erased node %d (%s)",
					n.BasePtr().ID(), n.KindOf(), i, op.BasePtr().ID(), op.KindOf())
			}
		}
		for _, u := range ir.UsesOf(n) {
			if user := u.User(); user == nil || user.BasePtr().Erased() {
				return fmt.Errorf("memopt-verify: dangling edge: a use of node %d (%s) has a nil/erased user",
					n.BasePtr().ID(), n.KindOf())
			}
		}
	}

	// 2 + 3. FIXPOINT and SCOPED USERS, over the eligible regions.
	end := g.End()
	var endBase *ir.Base
	if end != nil {
		endBase = end.BasePtr()
	}
	for _, r := range regions {
		inSet := make(map[*ir.Base]bool, len(r.accesses))
		for _, a := range r.accesses {
			inSet[a.BasePtr()] = true
		}
		for _, a := range r.accesses {
			switch a.KindOf() {
			case ir.KindMemLoad:
				if b := firstStoreBehind(a); b != nil && sameProvenPath(se, a, b) {
					return fmt.Errorf("memopt-verify: load node %d (%s) is forwardable against store node %d (%s) — memopt did not reach its fixpoint",
						a.BasePtr().ID(), a.KindOf(), b.BasePtr().ID(), b.KindOf())
				}
			case ir.KindMemStore:
				if b, ok := shadowingStoreBehind(se, a); ok {
					return fmt.Errorf("memopt-verify: store node %d (%s) shadows store node %d (%s) — memopt did not reach its fixpoint",
						a.BasePtr().ID(), a.KindOf(), b.BasePtr().ID(), b.KindOf())
				}
				if !hasEffectSuccessor(r, end, a) {
					return fmt.Errorf("memopt-verify: store node %d (%s) has no effect successor and is dead — memopt did not reach its fixpoint",
						a.BasePtr().ID(), a.KindOf())
				}
			}
			// SCOPED USERS: every effect user of an access is an access of
			// the same region or the Return.
			for _, u := range ir.UsesOf(a) {
				user := u.User()
				if user == nil || !holdsImpactEdgeAt(user, a) {
					continue // a data use (e.g. a returned store value): INV-D territory
				}
				if user.BasePtr() != endBase && !inSet[user.BasePtr()] {
					return fmt.Errorf("memopt-verify: effect user of node %d (%s) is foreign: node %d (%s)",
						a.BasePtr().ID(), a.KindOf(), user.BasePtr().ID(), user.KindOf())
				}
			}
		}
	}
	return nil
}

// shadowingStoreBehind reports the eliminable predecessor of store s, if
// one remains: walking s's ImpactBy backwards, the first load stops the
// walk (it may read the position), and any store reached with the same
// proven path (only stores in between, C2b) should have been erased.
func shadowingStoreBehind(se sideeffect.SideEffects, origin ir.Node) (ir.Node, bool) {
	cur := origin // the walk cursor; origin stays the comparison anchor
	for {
		edges := ir.ImpactBy(cur)
		if len(edges) != 1 {
			return nil, false
		}
		s0 := edges[0].Def()
		if s0 == nil || s0.BasePtr().Erased() {
			return nil, false
		}
		if s0.KindOf() == ir.KindMemLoad {
			return nil, false
		}
		if sameProvenPath(se, origin, s0) {
			return s0, true
		}
		cur = s0 // walk past the unequal-position store
	}
}
