// inline.go is the second Simplification sub-pass: lambda-parameter
// inlining (quick-261019-ktm, doc/optana/lower_itercluster.md). After
// simplify-lower, a Map/Reduce's Input list holds plain SSA values; the
// lambda bodies still reach their inputs through paramRef nodes. This pass
// binds each lambda parameter POSITIONALLY to the i-th Input-list value
// binding (D9) and replaces every reference, so every later analysis and
// optimization sees plain SSA data flow with no hidden references.
package simplify

import (
	"errors"
	"fmt"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the inlining pass: Priority 103 — immediately after
// simplify-lower (102), before simplify-verify (104) and gvn (110).
func init() {
	opt.Register(opt.PassSpec{
		ID:       "simplify-inline",
		Priority: 103,
		Requires: []string{"simplify-lower"},
		Factory:  func() opt.Pass { return InlinePass{} },
	})
}

// InlinePass inlines every map/reduce lambda parameter. The binding rule
// (D9, uniform and positional): lambda param i binds to the i-th
// Input-list VALUE binding, where
//
//   - a KeyedAttribute element contributes its wrapped value operand
//     (Reduce: [piped, attr(init)] binds [acc, now] as acc↔piped,
//     now↔init-value);
//   - the trailing As node binds nothing;
//   - every other operand contributes itself.
//
// For each binding, every paramRef of the parameter is replaced
// (ReplaceAllUsesWith onto the binding) and erased; a lambda body ROOT that
// is itself a paramRef rewires through the SetLambdaBody choke point (the
// one raw-pointer edge RAUW cannot reach). Dead parameter declarations are
// then stripped via ClearLambdaParams — parameters still referenced by
// live uses stay declared — and use-free lambda InputParams are erased.
//
// Deterministic throughout (allocation order); one MutContext.
type InlinePass struct{}

// ID returns the pipeline name "simplify-inline".
func (InlinePass) ID() string { return "simplify-inline" }

// Requires: lowering must have run earlier in the pipeline — inlining a
// project-bearing Input list would bind params to transient bridges.
func (InlinePass) Requires() []string { return []string{"simplify-lower"} }

// Produces: builder-output stage — the pass rewires and erases only, so the
// constructed legality window re-checks the graph.
func (InlinePass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — every analysis is invalidated.
func (InlinePass) Preserved() []string { return nil }

// Run performs the inlining over the Level-1 sea. Gates first (nil and
// materialized/CFG-carrying graphs are deterministic errors evaluated
// before any mutation); then, for every live map/reduce in allocation
// order, the binding/replacement/stripping sequence of the pass comment.
func (InlinePass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("simplify-inline: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("simplify-inline: graph already materialized (simplify runs on the Level-1 sea)")
	}

	mctx := &ir.MutContext{Pass: "simplify-inline"}

	for _, stage := range g.AllocatedNodes() {
		if stage.BasePtr().Erased() {
			continue
		}
		switch stage.KindOf() {
		case ir.KindMap, ir.KindReduce:
		default:
			continue
		}

		// Value bindings per D9: Input-list operands in order, a trailing
		// As dropped, KeyedAttribute elements unwrapped to their value.
		bindings := valueBindings(stage)

		// The stage's parameter declarations, captured before any edit.
		paramGroups := ir.LambdaParams(stage)
		var stageParams []ir.Node
		for _, group := range paramGroups {
			stageParams = append(stageParams, group...)
		}

		for _, group := range paramGroups {
			for i, param := range group {
				if param == nil || param.BasePtr().Erased() {
					continue
				}
				if i >= len(bindings) || bindings[i] == nil {
					continue // unbound position: the verifier owns the mismatch
				}
				if err := inlineParam(mctx, stage, param, bindings[i]); err != nil {
					return err
				}
			}
		}

		// Strip the dead parameter DECLARATIONS — the declarations ride the
		// stage's trailing lambda node's operand slots (D15), and
		// ClearLambdaParams is the choke point that drops them. A param
		// still holding live uses beyond that declaration slot (an
		// acc-style survivor) stays declared.
		var lamBase *ir.Base
		if lam := ir.LambdaOf(stage); lam != nil {
			lamBase = lam.BasePtr()
		}
		keep := func(p ir.Node) bool {
			for _, u := range ir.UsesOf(p) {
				if user := u.User(); user != nil && user.BasePtr() != lamBase {
					return true // a live use beyond the declaration slot itself
				}
			}
			return false
		}
		if err := mctx.ClearLambdaParams(stage, keep); err != nil {
			return fmt.Errorf("simplify-inline: strip params of node %d (%s): %w",
				stage.BasePtr().ID(), stage.KindOf(), err)
		}

		// Erase the lambda InputParams that are now use-free (RoleLambda
		// params are never graph-owned; Erase's live-use refusal is the
		// safety net).
		for _, p := range stageParams {
			if p == nil || p.BasePtr().Erased() {
				continue
			}
			if len(ir.UsesOf(p)) > 0 {
				continue
			}
			if err := mctx.Erase(p); err != nil {
				return fmt.Errorf("simplify-inline: erase lambda param %d: %w", p.BasePtr().ID(), err)
			}
		}
	}
	return nil
}

// valueBindings computes a stage's Input-list VALUE bindings per D9:
// operands in slot order, the trailing As dropped, KeyedAttribute elements
// unwrapped to their wrapped value operand.
func valueBindings(stage ir.Node) []ir.Node {
	ops := ir.OperandsOf(stage)
	// The trailing lambda node (always, D15) and As node (when present,
	// D1) are layout markers, not value bindings.
	if len(ops) > 0 && ops[len(ops)-1] != nil && ops[len(ops)-1].KindOf() == ir.KindLambda {
		ops = ops[:len(ops)-1]
	}
	if len(ops) > 0 && ops[len(ops)-1] != nil && ops[len(ops)-1].KindOf() == ir.KindAs {
		ops = ops[:len(ops)-1]
	}
	bindings := make([]ir.Node, len(ops))
	for i, op := range ops {
		if op == nil {
			continue
		}
		if op.KindOf() == ir.KindKeyedAttribute {
			inner := ir.OperandsOf(op)
			if len(inner) > 0 {
				bindings[i] = inner[0]
			}
			continue
		}
		bindings[i] = op
	}
	return bindings
}

// inlineParam replaces every paramRef of param with binding: RAUW moves the
// ref's users onto the binding, a body-root paramRef additionally rewires
// the lambda body itself (the one edge RAUW cannot reach), and the ref is
// erased once use-free.
func inlineParam(mctx *ir.MutContext, stage, param, binding ir.Node) error {
	for _, u := range ir.UsesOf(param) { // snapshot: survives its own rewiring
		ref := u.User()
		if ref == nil || ref.BasePtr().Erased() || ref.KindOf() != ir.KindParam {
			continue // the declaration record itself, or a foreign user
		}
		if err := mctx.ReplaceAllUsesWith(ref, binding); err != nil {
			return fmt.Errorf("simplify-inline: replace param ref %d of node %d: %w",
				ref.BasePtr().ID(), stage.BasePtr().ID(), err)
		}
		// A lambda body ROOT that is the ref itself: rewire the body to the
		// binding before erasing, or the Body pointer would dangle at a
		// tombstone (RAUW cannot reach it — the raw-pointer edge).
		for _, body := range ir.LambdaBodies(stage) {
			if body != nil && body.BasePtr() == ref.BasePtr() {
				if err := mctx.SetLambdaBody(stage, binding); err != nil {
					return fmt.Errorf("simplify-inline: rewire body root of node %d: %w",
						stage.BasePtr().ID(), err)
				}
				break
			}
		}
		if err := mctx.Erase(ref); err != nil {
			return fmt.Errorf("simplify-inline: erase param ref %d of node %d: %w",
				ref.BasePtr().ID(), stage.BasePtr().ID(), err)
		}
	}
	return nil
}
