package ir

// node_lir_memory.go holds the LIR memory family (D-3): the lowered SMEM
// tier (MemLoadSMem/MemStoreSMem/MemNewSMem) and the lowered fragment tier
// (FragLoad/FragStore/FragNew) — the material cache lowering binds to once
// the abstract GMEM access is promoted into explicit per-buffer data flow
// (mirroring tinycudaopto's memory_class.go concept: every LIR memory node
// carries the storage class its constructor bound it to).
//
// Operand shapes (the tensor/smem/frag buffer is an OPERAND SLOT, not a
// payload access — lowering promotes the HIR payload access into ordinary
// SSA data flow):
//
//	memLoadSMem:  ops = [src smem buffer, bivs...]
//	memStoreSMem: ops = [value, dst smem buffer, bivs...]
//	memNewSMem:   ops = [extent]
//	fragLoad:     ops = [src frag buffer, bivs...]
//	fragStore:    ops = [value, dst frag buffer, bivs...]
//	fragNew:      ops = [extent]
//
// The biv (bounded induction variable) axes are the lowered form of the HIR
// access axes. Store nodes return the stored value, matching the HIR
// memStore contract (doc §2.5).

// StorageClass is the storage tier an LIR memory node is bound to
// (X-4, mirroring tinycudaopto memory_class.go). Constructors set it at
// allocation; it is immutable node metadata queried via StorageClassOf.
type StorageClass int

const (
	StorageGMEM StorageClass = iota // global memory (the HIR tier; LIR nodes never carry it)
	StorageSMem                     // shared memory
	StorageFrag                     // register fragment
)

// String returns the canonical lowercase name of the storage class.
func (c StorageClass) String() string {
	switch c {
	case StorageGMEM:
		return "gmem"
	case StorageSMem:
		return "smem"
	case StorageFrag:
		return "frag"
	default:
		return "unknown"
	}
}

// StorageClassOf returns the storage class of an LIR memory node and whether
// n is one. GMEM loads/stores/news are HIR nodes and report false — the
// class metadata exists only on the lowered tier.
func StorageClassOf(n Node) (StorageClass, bool) {
	switch t := n.(type) {
	case *memLoadSMemNode:
		return t.class, true
	case *memStoreSMemNode:
		return t.class, true
	case *memNewSMemNode:
		return t.class, true
	case *fragLoadNode:
		return t.class, true
	case *fragStoreNode:
		return t.class, true
	case *fragNewNode:
		return t.class, true
	}
	return StorageGMEM, false
}

// memLoadSMemNode: lowered SMEM load; ops = [src, bivs...].
type memLoadSMemNode struct {
	*Base
	operandList
	class StorageClass
}

// NewMemLoadSMem creates an SMEM load from the smem buffer src at the
// bounded induction variables bivs (axis order). The verifier's shape check
// enforces the src operand kind (a MemNewSMem node).
func (g *Graph) NewMemLoadSMem(origin string, src Node, bivs ...Node) Node {
	n := g.allocNode(KindMemLoadSMem, origin, 1+len(bivs)).(*memLoadSMemNode)
	n.class = StorageSMem
	ops := make([]Node, 0, 1+len(bivs))
	ops = append(ops, src)
	ops = append(ops, bivs...)
	wireOperands(n, ops...)
	return n
}

// memStoreSMemNode: lowered SMEM store; ops = [value, dst, bivs...].
type memStoreSMemNode struct {
	*Base
	operandList
	class StorageClass
}

// NewMemStoreSMem creates an SMEM store of value into the smem buffer dst at
// the bounded induction variables bivs (axis order). The verifier's shape
// check enforces the dst operand kind (a MemNewSMem node).
func (g *Graph) NewMemStoreSMem(origin string, value, dst Node, bivs ...Node) Node {
	n := g.allocNode(KindMemStoreSMem, origin, 2+len(bivs)).(*memStoreSMemNode)
	n.class = StorageSMem
	ops := make([]Node, 0, 2+len(bivs))
	ops = append(ops, value, dst)
	ops = append(ops, bivs...)
	wireOperands(n, ops...)
	return n
}

// memNewSMemNode: lowered SMEM allocation; ops = [extent].
type memNewSMemNode struct {
	*Base
	operandList
	class StorageClass
}

// NewMemNewSMem creates an SMEM allocation node of the given extent value.
func (g *Graph) NewMemNewSMem(origin string, extent Node) Node {
	n := g.allocNode(KindMemNewSMem, origin, 1).(*memNewSMemNode)
	n.class = StorageSMem
	wireOperands(n, extent)
	return n
}

// fragLoadNode: lowered fragment load; ops = [src, bivs...].
type fragLoadNode struct {
	*Base
	operandList
	class StorageClass
}

// NewFragLoad creates a fragment load from the frag buffer src at the
// bounded induction variables bivs (axis order). The verifier's shape check
// enforces the src operand kind (a FragNew node).
func (g *Graph) NewFragLoad(origin string, src Node, bivs ...Node) Node {
	n := g.allocNode(KindFragLoad, origin, 1+len(bivs)).(*fragLoadNode)
	n.class = StorageFrag
	ops := make([]Node, 0, 1+len(bivs))
	ops = append(ops, src)
	ops = append(ops, bivs...)
	wireOperands(n, ops...)
	return n
}

// fragStoreNode: lowered fragment store; ops = [value, dst, bivs...].
type fragStoreNode struct {
	*Base
	operandList
	class StorageClass
}

// NewFragStore creates a fragment store of value into the frag buffer dst at
// the bounded induction variables bivs (axis order). The verifier's shape
// check enforces the dst operand kind (a FragNew node).
func (g *Graph) NewFragStore(origin string, value, dst Node, bivs ...Node) Node {
	n := g.allocNode(KindFragStore, origin, 2+len(bivs)).(*fragStoreNode)
	n.class = StorageFrag
	ops := make([]Node, 0, 2+len(bivs))
	ops = append(ops, value, dst)
	ops = append(ops, bivs...)
	wireOperands(n, ops...)
	return n
}

// fragNewNode: lowered fragment allocation; ops = [extent].
type fragNewNode struct {
	*Base
	operandList
	class StorageClass
}

// NewFragNew creates a fragment allocation node of the given extent value.
func (g *Graph) NewFragNew(origin string, extent Node) Node {
	n := g.allocNode(KindFragNew, origin, 1).(*fragNewNode)
	n.class = StorageFrag
	wireOperands(n, extent)
	return n
}
