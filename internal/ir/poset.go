package ir

// poset.go is the dependency-poset read face of the sea of nodes (R1/R2):
// the primitive edge enumerations the Level-1 analysis (Walk/Verify/DCE) and
// the pass.Scheduler's statement-level topo sort + LICM (quick-261018-bl0
// task 3) build on. Sea edges point use→def: a node's dependencies are the
// defs its live SSAUse records resolve to. DataDeps covers the operand
// slots; EffectDeps covers the ImpactBy payload (the explicit per-tensor
// ordering edges, effect.go); PosetDeps is the flat "must come after" union,
// data first then effect. Payload uses that are NOT ordering edges (store
// targets, map as-orders, lambda params) are deliberately excluded: they
// name parameters — graph roots — and contribute nothing to a def-poset.
//
// All enumerations skip killed and unwired records (the handle model's
// checkable corruption classes stay checkable, never traversable) and are
// deterministic (slot order, then payload order).

// DataDeps returns the live operand-slot defs of n in slot order — n's data
// "must come after" set (R1). A nil or foreign node yields nil.
func DataDeps(n Node) []Node {
	var deps []Node
	if impl := asImpl(n); impl != nil {
		impl.forEachOperand(func(u SSAUse) {
			if u.rec != nil && !u.rec.killed && u.rec.def != nil {
				deps = append(deps, u.rec.def.node)
			}
		})
	}
	return deps
}

// EffectDeps returns the live ImpactBy payload defs of n — the per-tensor
// ordering edges of the sea of nodes (effect.go). Non-effectful nodes carry
// none; a nil node yields nil.
func EffectDeps(n Node) []Node {
	var deps []Node
	if eb, ok := n.(effectBearer); ok {
		for _, u := range eb.impactBy() {
			if u.rec != nil && !u.rec.killed && u.rec.def != nil {
				deps = append(deps, u.rec.def.node)
			}
		}
	}
	return deps
}

// PosetDeps returns the flat dependency union of n: data edges (slot order)
// followed by effect edges — the single "must come after" list a topological
// ordering consumes.
func PosetDeps(n Node) []Node {
	return append(DataDeps(n), EffectDeps(n)...)
}

// SeaRoots returns the graph's Level-1 root set (R3's two-root set): the
// RoleTensor params (graph param declaration order — dims and scalars are
// not observable roots) followed by the end node (R2). Stores are never
// roots: they stay live through the end node's operand and ImpactBy paths,
// and a store with no such path from the roots is unreachable building
// debris. A nil graph yields nil.
func (g *Graph) SeaRoots() []Node {
	if g == nil {
		return nil
	}
	var roots []Node
	for _, p := range g.params {
		if role, ok := RoleOf(p); ok && role == RoleTensor {
			roots = append(roots, p)
		}
	}
	if g.end != nil {
		roots = append(roots, g.end)
	}
	return roots
}
