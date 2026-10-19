// verify.go is the post-GVN verification pass (quick-261018-nga [4.7]):
// it recomputes the pass's inputs from scratch and asserts the invariants
// a clean GVN leaves behind, so a corrupted dedup fails loudly at the next
// pipeline step instead of surfacing as bogus codegen.
package gvn

import (
	"errors"
	"fmt"

	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the verification pass: Priority 111 — immediately
// after gvn (110), before anything else observes the deduped graph.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "gvn-verify",
		Priority: 111,
		Requires: []string{"gvn"},
		Factory:  func() opt.Pass { return VerifyPass{} },
	})
}

// VerifyPass asserts the post-GVN invariants over the Level-1 sea. It
// mutates nothing and shares no state with the gvn pass — the analysis and
// the body-root set are recomputed FRESH, so the verifier's independence
// is structural, not conventional. The three checks (first failure
// returns a diagnostic naming node IDs and kinds):
//
//   - DANGLING: every live use of every allocated node resolves to a
//     non-erased def with a non-erased, non-nil user — the post-RAUW/Erase
//     SSA bookkeeping integrity ([4.7]'s "any use record dangles").
//
//   - ANALYSIS-CLOSURE ([2.4] as amended by C): the accumulated analysis is
//     a fixpoint over the ACTUAL graph — every node it calls pure has no
//     base verdict and is not an order-sensitive kind with an effectful
//     child; every node it calls effectful has a base verdict or is an
//     order-sensitive kind with an effectful child. The base/order-
//     sensitive tables below are an INDEPENDENT MIRROR of
//     internal/analysis/sideeffect/sideeffect.go's — drift between the two
//     is exactly what this check exists to catch.
//
//   - UNIQUE-REPRESENTATIVE ([4.7] post-hoc): over the participating set
//     (allocated, valid kind, not a body root, pure) no two distinct nodes
//     are ValueEqual. Full pairwise — O(N²), documented as fine at v0
//     kernel scale (hundreds of nodes) — deliberately stronger than
//     bucket-limited checking, because it also catches hash/equality
//     contract drift (equal values that stopped hashing identically would
//     survive gvn as silent duplicates; this check names them).
//
// The checks that CAN fire only fire on corruption or contract drift; a
// green run proves the invariants held, not that the pass is optimal.
type VerifyPass struct{}

// ID returns the pipeline name "gvn-verify".
func (VerifyPass) ID() string { return "gvn-verify" }

// Requires: the dedup must have run earlier in the pipeline — verifying an
// undeduped graph asserts nothing about GVN (and in fact flags its
// duplicates).
func (VerifyPass) Requires() []string { return []string{"gvn"} }

// Produces: builder-output stage — the pass introduces no node kinds and
// no mutations, so the constructed legality window re-checks the graph.
func (VerifyPass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — every analysis is invalidated, consistent with the
// invalidate-all default.
func (VerifyPass) Preserved() []string { return nil }

// Run asserts the invariants above. Gates mirror the gvn pass: a nil graph
// and a materialized/CFG-carrying graph are deterministic errors evaluated
// before anything is read, so a refused graph is untouched.
func (VerifyPass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("gvn-verify: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("gvn-verify: graph already materialized (the post-GVN invariants are Level-1 sea invariants)")
	}

	se := sideeffect.Analyze(g)

	// 1. DANGLING — post-RAUW/Erase SSA integrity.
	for _, n := range g.AllocatedNodes() {
		for _, u := range ir.UsesOf(n) {
			def := u.Def()
			if def == nil || def.BasePtr().Erased() {
				return fmt.Errorf("gvn-verify: dangling edge: node %d (%s) keeps a use resolving to an erased/nil def",
					n.BasePtr().ID(), n.KindOf())
			}
			user := u.User()
			if user == nil || user.BasePtr().Erased() {
				return fmt.Errorf("gvn-verify: dangling edge: a use of node %d (%s) has a nil/erased user",
					n.BasePtr().ID(), n.KindOf())
			}
		}
	}

	// 2. ANALYSIS-CLOSURE — the accumulated verdicts are a fixpoint of
	// base ∪ order-sensitive propagation over the actual child edges.
	for _, n := range g.AllocatedNodes() {
		propagated := orderSensitiveMirror(n.KindOf()) && hasEffectfulChildMirror(n, se)
		base := baseVerdictMirror(n)
		if se.HasSideEffect(n) != (base || propagated) {
			return fmt.Errorf("gvn-verify: side-effect analysis is not a fixpoint at node %d (%s): accumulated=%v base=%v propagated=%v",
				n.BasePtr().ID(), n.KindOf(), se.HasSideEffect(n), base, propagated)
		}
	}

	// 3. UNIQUE-REPRESENTATIVE — no two participating nodes are one value.
	bodyRoots := lambdaBodyRoots(g)
	var participating []ir.Node
	for _, n := range g.AllocatedNodes() {
		if bodyRoots[n] || !validKindMirror(n.KindOf()) || se.HasSideEffect(n) {
			continue
		}
		participating = append(participating, n)
	}
	for i := 0; i < len(participating); i++ {
		vi, ok := participating[i].(ir.Value)
		if !ok {
			continue
		}
		for j := i + 1; j < len(participating); j++ {
			vj, ok := participating[j].(ir.Value)
			if !ok {
				continue
			}
			if vi.ValueEqual(vj, se) {
				return fmt.Errorf("gvn-verify: duplicate representatives remain: node %d (%s) and node %d (%s) are ValueEqual",
					participating[i].BasePtr().ID(), participating[i].KindOf(),
					participating[j].BasePtr().ID(), participating[j].KindOf())
			}
		}
	}
	return nil
}

// baseVerdictMirror is the INDEPENDENT mirror of the analysis's self
// verdict: ImpactBy bearers, the whole memory family (amendment A — loads
// included, nil ImpactBy included), and implicit control flow ([2.3]).
// Kept local so the closure check derives its expectation without
// consulting the code under test.
func baseVerdictMirror(n ir.Node) bool {
	if n == nil {
		return true
	}
	if len(ir.ImpactBy(n)) > 0 {
		return true // [2.1]
	}
	k := n.KindOf()
	if k.Family() == ir.FamilyMemory {
		return true // [2.2] as amended by A
	}
	switch k {
	case ir.KindIter, ir.KindIterCluster, ir.KindForLoop, ir.KindReturn:
		return true // [2.3]
	}
	return false
}

// orderSensitiveMirror is the independent mirror of the taint-propagation
// kind set (amendment C): FamilyMemory ∪ FamilyLoop ∪ {Return} ∪ the
// future-collectives hook. Pure kinds never appear here.
func orderSensitiveMirror(k ir.Kind) bool {
	if f := k.Family(); f == ir.FamilyMemory || f == ir.FamilyLoop || f == ir.FamilyCollective {
		return true
	}
	return k == ir.KindReturn
}

// hasEffectfulChildMirror reports whether any child of n (operand defs plus
// lambda body roots — the same child relation the analysis walks) is
// effectful per the accumulated se.
func hasEffectfulChildMirror(n ir.Node, se sideeffect.SideEffects) bool {
	for _, d := range ir.DataDeps(n) {
		if se.HasSideEffect(d) {
			return true
		}
	}
	for _, b := range ir.LambdaBodies(n) {
		if b != nil && se.HasSideEffect(b) {
			return true
		}
	}
	return false
}

// validKindMirror is the fail-closed kind range (sentinels excluded).
func validKindMirror(k ir.Kind) bool { return k > ir.KindInvalid && k < ir.KindEnd }
