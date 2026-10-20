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

// memLoadNode: HIR GMEM load; ops = [tensor, axes...] — the tensor InputParam
// first, then one dim InputParam per axis in subscript order. effectIns is
// the ImpactBy list — the per-tensor ordering edge wired by the env-aware
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

// removeImpactUse implements effectBearer: drop one ordering-edge handle by
// record identity (C3, quick-261020-ejf). The hook only removes — the
// chained constructor keeps its direct append wiring.
func (n *memLoadNode) removeImpactUse(u SSAUse) bool { return dropImpactHandle(&n.effectIns, u) }

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

// memStoreNode: HIR GMEM store — the HIR stage store; ops =
// [value, tensor, axes...] — the stored value, then the target access's
// tensor InputParam and one dim InputParam per axis in subscript order
// (Input/ImpactBy-only convention, quick-261019-ktm revision: the target
// components are data dependencies and ride the Input list); effectIns is
// the ImpactBy list — the per-tensor ordering edge wired by the env-aware
// construction path (effect.go). The store returns the stored value
// (doc §2.5).
type memStoreNode struct {
	*Base
	operandList
	effectIns []SSAUse
}

// forEachPayloadUse implements payloadBearer: the ImpactBy effect inputs
// (the store's ONLY payload — target access components are operands now).
func (n *memStoreNode) forEachPayloadUse(fn func(SSAUse)) {
	for _, u := range n.effectIns {
		fn(u)
	}
}

// impactBy implements effectBearer: a copy of the store's ordering edges.
func (n *memStoreNode) impactBy() []SSAUse { return append([]SSAUse(nil), n.effectIns...) }

// removeImpactUse implements effectBearer: drop one ordering-edge handle by
// record identity (C3, quick-261020-ejf). The hook only removes — the
// chained constructor keeps its direct append wiring.
func (n *memStoreNode) removeImpactUse(u SSAUse) bool { return dropImpactHandle(&n.effectIns, u) }

// NewMemStore creates a HIR GMEM structural store node writing value through
// the target access: tensor is the tensor InputParam node and axes its dim
// InputParam nodes (nil/empty for a whole-tensor store). The verifier's
// shape check pins operand 1 to a RoleTensor InputParam.
func (g *Graph) NewMemStore(origin string, tensor Node, axes []Node, value Node) Node {
	ops := make([]Node, 0, 1+1+len(axes))
	ops = append(ops, value, tensor)
	ops = append(ops, axes...)
	return g.NewNode(KindMemStore, origin, ops...)
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
