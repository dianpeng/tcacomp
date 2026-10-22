// Package sideeffect implements the ACCUMULATED side-effect analysis of
// the sea of nodes (quick-261018-nga): the gatekeeper GVN consults
// through ir.SideEffectAnalysis before value-numbering a node. It lives
// in its OWN sub-module folder under internal/analysis (quick-261019-jck),
// still implementing the analysis.Analysis contract internal/opt's cache
// builds on; its import edges are internal/analysis/sideeffect →
// {internal/analysis, internal/ir} (no cycle: ir never imports analysis,
// and the top-level analysis package never imports its sub-modules — the
// query face ir.SideEffectAnalysis is defined in ir precisely so this
// file can implement it).
//
// Two effect models coexist by design:
//
//   - ir.EffectOf (internal/ir/effect.go) is the NODE-LOCAL structural
//     verdict — which kinds carry an observable effect of their own (per
//     amendment A: every FamilyMemory node plus Return). It is
//     self-scoped: no lattice meet, no operand accumulation.
//
//   - THIS analysis is the ACCUMULATED GVN gate: effectful(n) = base(n)
//     OR (orderSensitive(n) AND some child is effectful), computed to a
//     fixpoint over the graph. The taint propagates ONLY through
//     order-sensitive kinds (amendment C): FamilyMemory, FamilyLoop
//     (Iter/IterCluster/ForLoop), Return, and — when they land — future
//     collective kinds. PURE kinds (the ScalarExpr family, Map, Reduce,
//     Const, Param, InputParam, DimVar/BoundedDimVar) never propagate, so
//     address and index math over iter values stays GVN-eligible.
//
// The propagation clause is LIVE: since doc/optana/loop_fusion.md
// reclassified the bare Iter as side-effect free ("The Iter node is
// itself side effect free"), KindIter is order-sensitive (FamilyLoop)
// but NOT base-effectful — an Iter with an accumulated-effectful operand
// (e.g. a MemLoad init) taints effectful and stays out of GVN (the
// loop-fusion soundness gate), while a bare Iter is pure and
// GVN-eligible, which is what lets gvn fuse identical loop headers. A
// future collective kind joins the order-sensitive set the same way,
// WITHOUT touching the base verdicts; the negative tests pin that pure
// kinds with effectful operands and bodies stay pure. gvn-verify
// independently re-derives this fixpoint and checks the analysis against
// it.
package sideeffect

import (
	"tcacomp/internal/analysis"
	"tcacomp/internal/ir"
)

// init self-registers the analysis in the global analysis registry
// (quick-261019-jck): importing this package — directly or through
// internal/analysis/all — is the single act that makes it visible to
// analysis.Lookup/Registered/BuildAll.
func init() {
	analysis.Register(analysis.AnalysisSpec{
		Name:    "side-effect", // MUST stay "side-effect": opt.Cache key + SideEffects.Name()
		Factory: func(g *ir.Graph) analysis.Analysis { return Analyze(g) },
	})
}

// SideEffects is the accumulated side-effect analysis over one graph
// snapshot: a memoized verdict per live node. It implements BOTH
// analysis.Analysis (Name — opt.Cache compatibility) and
// ir.SideEffectAnalysis (HasSideEffect — the GVN query face). Queries are
// fail-closed (T-nga-03): unknown, erased, or nil nodes report effectful —
// an unanalyzable node is never a GVN candidate.
type SideEffects struct {
	effectful map[ir.Node]bool
}

// Name implements analysis.Analysis: the stable analysis name used by
// opt.Cache bookkeeping.
func (SideEffects) Name() string { return "side-effect" }

// HasSideEffect reports the accumulated effect verdict of n. Conservative
// by contract: nil nodes, erased nodes, and nodes the analysis never saw
// (foreign graphs, post-run mutations) report true.
func (s SideEffects) HasSideEffect(n ir.Node) bool {
	if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
		return true
	}
	if s.effectful == nil {
		return true // Analyze(nil) or a zero value: all-conservative
	}
	if v, ok := s.effectful[n]; ok {
		return v
	}
	return true // unknown node: fail closed
}

// Analyze computes the accumulated side-effect analysis over g's live
// nodes (g.AllocatedNodes()). Read-only per the analysis-package contract:
// it never mutates the IR. A nil graph yields an all-conservative result.
// The computation is a memoized three-state DFS over children =
// ir.DataDeps(n) ∪ ir.LambdaBodies(n) — O(N+E), deterministic (no map
// iteration influences any verdict).
func Analyze(g *ir.Graph) SideEffects {
	if g == nil {
		return SideEffects{}
	}
	s := &state{verdict: make(map[ir.Node]int)} // 0 unknown, 1 in progress, 2 pure, 3 effectful
	nodes := g.AllocatedNodes()
	for _, n := range nodes {
		s.effectfulOf(n)
	}
	out := make(map[ir.Node]bool, len(nodes))
	for _, n := range nodes {
		out[n] = s.verdict[n] == 3
	}
	return SideEffects{effectful: out}
}

// state carries the memoized DFS: verdict's in-progress marker makes the
// walk cycle-safe (re-entry reads effectful — conservative, mirroring the
// scheduler's pinnedMemo pattern).
type state struct {
	verdict map[ir.Node]int
}

// effectfulOf computes the accumulated verdict of n:
//
//	effectful(n) = baseVerdict(n) OR (orderSensitive(n) AND ∃ effectful child)
//
// with children = ir.DataDeps(n) ∪ ir.LambdaBodies(n). Non-node inputs
// (nil, foreign) read effectful — fail closed.
func (s *state) effectfulOf(n ir.Node) bool {
	if n == nil || n.BasePtr() == nil {
		return true
	}
	switch s.verdict[n] {
	case 1:
		return true // in progress: cycle — treat as effectful (fail safe)
	case 2:
		return false
	case 3:
		return true
	}
	s.verdict[n] = 1

	effectful := baseVerdict(n) || (orderSensitive(n.KindOf()) && s.hasEffectfulChild(n))

	if effectful {
		s.verdict[n] = 3
	} else {
		s.verdict[n] = 2
	}
	return effectful
}

// hasEffectfulChild reports whether any child of n (operand defs plus
// lambda body roots) is accumulated-effectful.
func (s *state) hasEffectfulChild(n ir.Node) bool {
	for _, d := range ir.DataDeps(n) {
		if s.effectfulOf(d) {
			return true
		}
	}
	for _, b := range ir.LambdaBodies(n) {
		if b == nil {
			continue
		}
		if s.effectfulOf(b) {
			return true
		}
	}
	return false
}

// baseVerdict is the SELF verdict (no accumulation): the memory family and
// implicit control flow are effectful on their own account, and any node
// carrying ImpactBy effect inputs is effectful regardless of kind ([2.1] —
// the future-bearer hook: all current bearers are memory/return nodes,
// already covered by amendment A, but the hook keeps the analysis correct
// if a new kind starts carrying effect inputs).
func baseVerdict(n ir.Node) bool {
	if n == nil {
		return true
	}
	if len(ir.ImpactBy(n)) > 0 {
		return true // [2.1]
	}
	k := n.KindOf()
	if k.Family() == ir.FamilyMemory {
		return true // [2.2] as amended by A: loads included, nil ImpactBy included
	}
	// The bare Iter is NOT here: it is side-effect free per
	// doc/optana/loop_fusion.md [1] ("The Iter node is itself side effect
	// free") — that reclassification is what lets gvn fuse identical loop
	// headers. IterCluster/ForLoop keep the conservative closure ([2.3]).
	switch k {
	case ir.KindIterCluster, ir.KindForLoop, ir.KindReturn:
		return true // [2.3]: the conservative closure of "Iter node"
	case ir.KindRuntimeCallMethod:
		// P-2 (quick-261022-l1o), the lockstep mirror of ir.EffectOf's arm:
		// a method call on a stateful reducer object is an ORDERED side
		// effect — never pure, never GVN-CSE'd. Every other Runtime/
		// Collective kind (the constructions, RuntimeGetAttribute) stays
		// free: the objects are immutable and no pass runs after
		// final-lower, so value-identical constructions can never be
		// CSE-merged anyway.
		return true
	}
	return false
}

// orderSensitive classifies the kinds through which the accumulated taint
// may PROPAGATE (amendment C): FamilyMemory ∪ FamilyLoop ∪ {Return} ∪
// FamilyCollective (its first members landed with final lowering,
// quick-261022-l1o: the three RuntimeXXXReducer constructions — conservative
// taint propagation through a construction whose operands somehow turned
// effectful; the constructions themselves are pure, P-2). Pure kinds are
// absent BY SPEC: a pure
// consumer of an effectful value stays pure (identity keying, amendment
// D, keeps its GVN participation sound).
func orderSensitive(k ir.Kind) bool {
	if f := k.Family(); f == ir.FamilyMemory || f == ir.FamilyLoop || f == ir.FamilyCollective {
		return true
	}
	return k == ir.KindReturn
}
