// Package gvn implements the pipeline's global value numbering / common
// subexpression elimination pass (quick-261018-nga) plus its dedicated
// invariant verifier (verify.go).
package gvn

import (
	"errors"
	"fmt"

	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the pass in the global registry: Priority 110 — global
// CSE, between early cleanup (dce/dce-verify at 100/101) and late scheduling
// (schedule at 900). The registry scale in internal/opt/registry.go
// documents the anchors.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "gvn",
		Priority: 110,
		Factory:  func() opt.Pass { return Pass{} },
	})
}

// Pass is the Level-1 (sea-of-nodes) GVN/CSE. Because the sea has no
// basic blocks and no CFG, value numbering needs exactly one global hash
// table: every pure computation whose kind, content, and SSA operand
// identities match an earlier representative collapses onto it. The pass
// is stateless — every Run re-derives everything from the graph, so
// re-runs are idempotent.
//
// The value contract lives in the IR (internal/ir/value.go): ValueHash /
// ValueEqual with IDENTITY keying (amendment D — operands key by SSA node
// identity, never structural recursion) and the one-of-a-kind sentinel
// ([3].3 — side-effect nodes never participate). The accumulated
// side-effect analysis (internal/analysis/sideeffect, amendment F)
// supplies the gate.
//
// Iter nodes participate through that contract with NESTED AWARENESS
// (doc/optana/loop_fusion.md): two iters are one value iff their kind,
// operand count, and every operand-slot def identity match — dim, bound
// (in the 4-operand bounded form), init, step, and any parent-Iter
// operands appended after the standard slots for nested loops. Equal
// loop headers therefore fuse, and whole nests fuse through the
// parent-operand RAUW cascade (a loop nest is GVNed only when all its
// nodes are equal); iters fed by effectful operands stay excluded via
// the order-sensitive taint (sideeffect).
//
// Determinism: nodes are processed in AllocatedNodes() order — the
// construction registry is def-before-use topological, so a node's
// operands have already resolved to their FINAL representatives by the
// time the node is hashed, and the LOWEST-ID duplicate survives. The
// RAUW-during-processing cascade composes like classic value numbering:
// once a duplicate def merges, its users' operand records point at the
// representative, so user-level duplicates become identity-equal and merge
// in the same run. (Graphs whose allocation order is not def-before-use —
// only constructible by passes, not by builders — may need a second run
// to fully converge; sound either way, since the pass only ever removes
// PROVEN duplicates.)
type Pass struct{}

// ID returns the pipeline name "gvn".
func (Pass) ID() string { return "gvn" }

// Requires: GVN has no ordering preconditions — it is usable standalone;
// the registry priority supplies the default position.
func (Pass) Requires() []string { return nil }

// Produces: builder-output stage — the pass erases and rewires only, it
// introduces no new kinds, so the constructed legality window re-checks
// the graph.
func (Pass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — every analysis is invalidated. (The side-effect
// analysis is recomputed fresh on entry and never published to the cache:
// a pre-run analysis would be stale after the first Erase, and a post-run
// put would be wiped by Invalidate(nil) anyway.)
func (Pass) Preserved() []string { return nil }

// Run performs global value numbering over the Level-1 sea:
//
//   - LEVEL GATE first (the DCE refusal pattern): a nil graph and a
//     materialized/CFG-carrying graph are deterministic errors evaluated
//     before any mutation, so a refused graph is untouched.
//
//   - The side-effect analysis and the lambda-body-root set are computed
//     up front. Body roots are the ONE non-SSAUse edge (Lambda.Body is a
//     raw pointer — RAUW cannot rewire it), so a body root is one-of-a-kind
//     for GVN: deduping it would strand a dangling lam.Body on the erased
//     twin's bearer. Body INTERIORS still dedup normally — their users are
//     record-holders inside the body. Upgrade path when lambda-body CSE
//     ever matters: a MutContext body-rewire choke point.
//
//   - One pass over the allocation registry: skip erased nodes and body
//     roots; hash; on a bucket hit confirmed by ValueEqual, hard-assert
//     neither side is effectful (below), RAUW the duplicate onto the
//     representative, verify zero uses remain, Erase it — CSE is performed
//     along the way ([4.6]). With no match, the node becomes the bucket's
//     representative ([4.5]: the representative node IS the recorded SSA
//     def — references move via the SSAUse ref records, never by name).
func (Pass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("gvn: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("gvn: graph already materialized (gvn runs on the Level-1 sea)")
	}

	se := sideeffect.Analyze(g)
	bodyRoots := lambdaBodyRoots(g)
	table := make(map[uint64][]ir.Node)
	mctx := &ir.MutContext{Pass: "gvn"}

	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr().Erased() || bodyRoots[n] {
			continue
		}
		v, ok := n.(ir.Value)
		if !ok {
			return fmt.Errorf("gvn: node %d (%s) is not an ir.Value (foreign node — fail loudly)",
				n.BasePtr().ID(), n.KindOf())
		}
		h := v.ValueHash(se)
		if h == ir.ValueHashOneOfAKind {
			continue // [2.5]: side-effect and invalid nodes do not participate
		}
		replaced := false
		for _, e := range table[h] {
			ev, ok := e.(ir.Value)
			if !ok {
				continue // unreachable: only ir.Values enter the table
			}
			if !ev.ValueEqual(v, se) {
				continue
			}
			// ITEM-7 HARD ASSERT (kept from the original spec): a
			// side-effect node must never be replaced. Unreachable
			// unless the sentinel gate and the analysis disagree — which
			// is exactly the drift worth failing the pipeline on.
			if se.HasSideEffect(n) || se.HasSideEffect(e) {
				return fmt.Errorf("gvn: side-effect node %d (%s) would be replaced by %d (%s)",
					n.BasePtr().ID(), n.KindOf(), e.BasePtr().ID(), e.KindOf())
			}
			if err := mctx.ReplaceAllUsesWith(n, e); err != nil {
				return fmt.Errorf("gvn: RAUW node %d onto %d: %w", n.BasePtr().ID(), e.BasePtr().ID(), err)
			}
			if uses := ir.UsesOf(n); len(uses) > 0 {
				return fmt.Errorf("gvn: post-RAUW bookkeeping corruption: node %d (%s) keeps %d use(s)",
					n.BasePtr().ID(), n.KindOf(), len(uses))
			}
			if err := mctx.Erase(n); err != nil {
				return fmt.Errorf("gvn: erase duplicate node %d (%s): %w", n.BasePtr().ID(), n.KindOf(), err)
			}
			replaced = true
			break
		}
		if !replaced {
			table[h] = append(table[h], n)
		}
	}
	return nil
}

// lambdaBodyRoots collects every live node that is the Body of some
// allocated map/reduce lambda — the one-of-a-kind set for GVN (design
// decision 6). Computed fresh by each pass so erased bearers (from an
// earlier gvn run) free their body roots for later participation.
func lambdaBodyRoots(g *ir.Graph) map[ir.Node]bool {
	roots := make(map[ir.Node]bool)
	if g == nil {
		return roots
	}
	for _, n := range g.AllocatedNodes() {
		for _, b := range ir.LambdaBodies(n) {
			if b != nil {
				roots[b] = true
			}
		}
	}
	return roots
}
