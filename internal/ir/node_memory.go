package ir

// node_memory.go holds the HIR GMEM family (D-3: MemLoad/MemStore/MemNew —
// FamilyMemory's HIR tier). The structural tensor access a store/load names
// is a payload access of InputParam uses at this level (review P1R-01: every
// reference is an SSAUse, never a name); lowering promotes the pieces into
// ordinary SSA data flow (the LIR memory family in node_lir_memory.go).
//
// Effect chaining mirrors the tinycudaopto skeleton's memory.go: the
// per-tensor GMEM effect chains include READS as well as writes — a load
// chains after the current chain head exactly like a store does, then becomes
// the new head, so two accesses to one tensor form an explicit
// second-ImpactBy-first edge while distinct tensors stay isolated (the
// restrict-alias rule: kernel tensors never alias).

// access is a structural tensor access: lhs[@m][@n]. Every component is an
// SSAUse — the tensor InputParam and one dim InputParam per axis — never a
// name (review P1R-01).
type access struct {
	tensorUse SSAUse
	axes      []SSAUse
}

// Tensor returns the access's tensor parameter node, or nil when unwired.
func (a access) Tensor() Node { return a.tensorUse.Def() }

// Axes returns the access's axis parameter nodes in subscript order; unwired
// slots yield nil entries.
func (a access) Axes() []Node {
	out := make([]Node, len(a.axes))
	for i, u := range a.axes {
		out[i] = u.Def()
	}
	return out
}

// memLoadNode: HIR GMEM load; ops = [tensor, axes...] — the tensor InputParam
// first, then one dim InputParam per axis in subscript order. effectIns is
// the ImpactBy payload — the per-tensor ordering edge wired by the env-aware
// construction path (effect.go), chaining reads into the same effect chains
// as writes (skeleton memory.go).
type memLoadNode struct {
	*Base
	operandList
	effectIns []SSAUse
}

// forEachPayloadUse implements payloadBearer: the ImpactBy effect inputs. The
// tensor/axes ride the ordinary operand slots, so the verifier's use-def/
// def-use invariants and Walk cover them in both chain directions with zero
// new checkers (T-quick-01), and Level-1 DCE follows the ordering edges.
func (n *memLoadNode) forEachPayloadUse(fn func(SSAUse)) {
	for _, u := range n.effectIns {
		fn(u)
	}
}

// impactBy implements effectBearer: a copy of the load's ordering edges.
func (n *memLoadNode) impactBy() []SSAUse { return append([]SSAUse(nil), n.effectIns...) }

// NewMemLoad creates a HIR GMEM load of tensor (a RoleTensor InputParam) at
// the dim InputParams axes (subscript order), un-chained. The verifier's
// shape check enforces the tensor operand kind.
func (g *Graph) NewMemLoad(origin string, tensor Node, axes []Node) Node {
	ops := make([]Node, 0, 1+len(axes))
	ops = append(ops, tensor)
	ops = append(ops, axes...)
	return g.NewNode(KindMemLoad, origin, ops...)
}

// NewMemLoadChained creates a load exactly like NewMemLoad and additionally
// wires it into the construction environment's per-tensor GMEM effect chain:
// when tensor has a known chain head, the load's ImpactBy payload points at
// it; the load then becomes the new chain head, so a later store/load to the
// same tensor chains after this read (skeleton memory.go: reads participate
// in the chains). A nil env (or a tensor with no prior access) leaves the
// load un-chained — identical to NewMemLoad.
func (g *Graph) NewMemLoadChained(origin string, env *Env, tensor Node, axes []Node) Node {
	n := g.NewMemLoad(origin, tensor, axes).(*memLoadNode)
	if head := env.ChainHead(tensor); head != nil {
		n.effectIns = append(n.effectIns, wirePayloadUse(n, head))
	}
	env.NoteStore(tensor, n)
	return n
}

// memStoreNode: HIR GMEM store — the HIR stage store; ops = [value]; target
// is a payload access of InputParam uses; effectIns is the ImpactBy payload —
// the per-tensor ordering edge wired by the env-aware construction path
// (effect.go). The store returns the stored value (doc §2.5).
type memStoreNode struct {
	*Base
	operandList
	target    access
	effectIns []SSAUse
}

// forEachPayloadUse implements payloadBearer: target tensor, then axes, then
// the ImpactBy effect inputs — so the verifier's use-def/def-use invariants
// and Walk treat ordering edges as first-class graph edges in both chain
// directions with zero new checkers (T-quick-01).
func (n *memStoreNode) forEachPayloadUse(fn func(SSAUse)) {
	fn(n.target.tensorUse)
	for _, u := range n.target.axes {
		fn(u)
	}
	for _, u := range n.effectIns {
		fn(u)
	}
}

// impactBy implements effectBearer: a copy of the store's ordering edges.
func (n *memStoreNode) impactBy() []SSAUse { return append([]SSAUse(nil), n.effectIns...) }

// NewMemStore creates a HIR GMEM structural store node writing value through
// the target access: tensor is the tensor InputParam node and axes its dim
// InputParam nodes (nil/empty for a whole-tensor store).
func (g *Graph) NewMemStore(origin string, tensor Node, axes []Node, value Node) Node {
	n := g.NewNode(KindMemStore, origin, value).(*memStoreNode)
	n.target.tensorUse = wirePayloadUse(n, tensor)
	if len(axes) > 0 {
		n.target.axes = make([]SSAUse, len(axes))
		for i, ax := range axes {
			n.target.axes[i] = wirePayloadUse(n, ax)
		}
	}
	return n
}

// memNewNode: HIR GMEM allocation; ops = [extent]. The created global extent
// is a Value (the memory's initial def in the skeleton's model).
type memNewNode struct {
	*Base
	operandList
}

// NewMemNew creates a HIR GMEM allocation node of the given extent value.
func (g *Graph) NewMemNew(origin string, extent Node) Node {
	return g.NewNode(KindMemNew, origin, extent)
}
