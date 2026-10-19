// verify.go is the third Simplification sub-pass: the post-simplify
// invariant check (quick-261019-ktm), mirroring dce-verify's role. It
// mutates nothing and re-asserts that the lowering and inlining actually
// completed, so a corrupted Simplification fails loudly at the next
// pipeline step instead of surfacing as bogus codegen.
package simplify

import (
	"errors"
	"fmt"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the verification pass: Priority 104 — immediately
// after simplify-inline (103), before gvn (110).
func init() {
	opt.Register(opt.PassSpec{
		ID:       "simplify-verify",
		Priority: 104,
		Requires: []string{"simplify-inline"},
		Factory:  func() opt.Pass { return VerifyPass{} },
	})
}

// VerifyPass asserts the post-Simplification invariants over the Level-1
// sea:
//
//   - NO LIVE PROJECTS: no live IterClusterProject remains — the IterCluster
//     format is fully lowered (clusters themselves may legitimately remain
//     in legacy builder graphs, which Simplification skips by design).
//
//   - FULLY INLINED MAP/REDUCE BODIES: no lambda body contains a paramRef
//     resolving to a parameter still declared in that stage's own record
//     (surviving params with non-paramRef uses are fine — ClearLambdaParams
//     keeps them deliberately).
//
//   - PLACEMENT INVARIANTS: the Task-1 verifier checks (As
//     unique-and-last, reduce's KeyedAttribute at operand 1, map
//     input/param count) hold on the live stages — delegated to ir.Verify,
//     whose violation set is otherwise exactly the chain invariants.
//
// The checks that CAN fire only fire on corruption; a green run proves the
// Simplification output shape, not pass correctness.
type VerifyPass struct{}

// ID returns the pipeline name "simplify-verify".
func (VerifyPass) ID() string { return "simplify-verify" }

// Requires: inlining must have run earlier in the pipeline — verifying an
// unsimplified graph asserts nothing about Simplification.
func (VerifyPass) Requires() []string { return []string{"simplify-inline"} }

// Produces: builder-output stage — the pass introduces no node kinds and no
// mutations, so the constructed legality window re-checks the graph.
func (VerifyPass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — consistent with the invalidate-all default.
func (VerifyPass) Preserved() []string { return nil }

// Run asserts the invariants above. Gates mirror the sibling passes: a nil
// graph and a materialized/CFG-carrying graph are deterministic errors
// evaluated before anything is read.
func (VerifyPass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("simplify-verify: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("simplify-verify: graph already materialized (the post-Simplification invariants are Level-1 sea invariants)")
	}

	// NO LIVE PROJECTS.
	for _, n := range g.AllocatedNodes() {
		if n.KindOf() == ir.KindIterClusterProject {
			return fmt.Errorf("simplify-verify: live iterclusterproject node %d remains (the IterCluster format must be fully lowered)",
				n.BasePtr().ID())
		}
	}

	// FULLY INLINED BODIES: per live map/reduce, no body-reachable paramRef
	// resolves to a param declared in that stage's own record.
	for _, stage := range g.AllocatedNodes() {
		switch stage.KindOf() {
		case ir.KindMap, ir.KindReduce:
		default:
			continue
		}
		declared := make(map[*ir.Base]bool)
		for _, group := range ir.LambdaParams(stage) {
			for _, u := range group {
				if p := u.Def(); p != nil {
					declared[p.BasePtr()] = true
				}
			}
		}
		if len(declared) == 0 {
			continue
		}
		for _, body := range ir.LambdaBodies(stage) {
			if body == nil {
				continue
			}
			for _, n := range bodySubtree(body) {
				if n.KindOf() != ir.KindParam {
					continue
				}
				// Resolve the ref's target from the param side (the ref's
				// payload use is not exported): some declared param holds a
				// live use whose user IS this ref.
				if refTargetDeclared(n, declared) {
					return fmt.Errorf("simplify-verify: node %d (%s) lambda body still references declared param through param ref node %d",
						stage.BasePtr().ID(), stage.KindOf(), n.BasePtr().ID())
				}
			}
		}
	}

	// PLACEMENT INVARIANTS (and the general chain invariants) on live
	// stages: ir.Verify's violation set covers the Task-1 shape checks.
	if vs := ir.Verify(g); len(vs) > 0 {
		return fmt.Errorf("simplify-verify: chain invariants violated: %s", vs[0].String())
	}
	return nil
}

// bodySubtree collects the nodes reachable from body over operand slots and
// lambda bodies (the paramRef nodes ride operand slots of body
// expressions), mirroring hasEffectNode's DFS shape with the exported read
// face.
func bodySubtree(body ir.Node) []ir.Node {
	seen := make(map[*ir.Base]bool)
	var out []ir.Node
	stack := []ir.Node{body}
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if n == nil || seen[n.BasePtr()] {
			continue
		}
		seen[n.BasePtr()] = true
		out = append(out, n)
		for _, op := range ir.OperandsOf(n) {
			stack = append(stack, op)
		}
		for _, b := range ir.LambdaBodies(n) {
			stack = append(stack, b)
		}
	}
	return out
}

// refTargetDeclared reports whether the paramRef n resolves to a param in
// declared. The ref's payload use is not exported, so resolve from the
// param side: some declared param holds a live use whose user IS n.
func refTargetDeclared(n ir.Node, declared map[*ir.Base]bool) bool {
	for p := range declared {
		for _, u := range ir.UsesOf(p) {
			if user := u.User(); user != nil && user.BasePtr() == n.BasePtr() {
				return true
			}
		}
	}
	return false
}
