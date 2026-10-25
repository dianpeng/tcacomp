package ir

// node_lir_memory.go holds the TIERED memory-verb constructors (fix3.md
// R-G/D-h, quick-261025-of7): the memory IR is exactly MemLoad/MemStore/
// MemNew parameterized by a storage class — this file's constructors mint
// the SMEM and FRAG tiers of the three verbs over the SAME kinds and the
// SAME structs as node_memory.go's GMEM constructors (the v3-era six
// lowered kinds are gone; a cache is a COMPOSITION of the verbs, never a
// fused node).
//
// The constructor NAMES keep their historical SMem/Frag spellings — they
// name the CLASS the node is stamped with, never a kind:
//
//	memLoadSMem tier:   ops = [src smem buffer, coords...]
//	memStoreSMem tier:  ops = [value, dst smem buffer, coords...]
//	memNewSMem tier:    ops = [extent]
//	frag tier:          the same three shapes, class FRAG
//
// The coordinates are the lowered access axes (windowed buffer indices).
// Store nodes return the stored value, matching the HIR memStore contract
// (doc §2.5).

// StorageClass is the storage tier a memory-verb node is bound to
// (X-4, mirroring tinycudaopto memory_class.go; fix3 R-G: the parameter the
// three verbs carry). Constructors set it at allocation; it is immutable
// node metadata queried via StorageClassOf. The zero value is StorageGMEM —
// the node_memory.go GMEM constructors allocate zero-valued structs.
type StorageClass int

const (
	StorageGMEM StorageClass = iota // global memory (the HIR tier)
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

// StorageClassOf returns the storage class of a memory-verb node and
// whether n is one of MemLoad/MemStore/MemNew at all. Every memory node
// reports its class; non-memory nodes report (StorageGMEM, false).
func StorageClassOf(n Node) (StorageClass, bool) {
	switch t := n.(type) {
	case *memLoadNode:
		return t.class, true
	case *memStoreNode:
		return t.class, true
	case *memNewNode:
		return t.class, true
	}
	return StorageGMEM, false
}

// NewMemLoadSMem creates an SMEM-class MemLoad from the smem buffer src at
// the coordinates coords (axis order). The verifier's shape check enforces
// the src operand class (an SMEM-class MemNew node).
func (g *Graph) NewMemLoadSMem(origin string, src Node, coords ...Node) Node {
	ops := make([]Node, 0, 1+len(coords))
	ops = append(ops, src)
	ops = append(ops, coords...)
	n := g.NewNode(KindMemLoad, origin, ops...).(*memLoadNode)
	n.class = StorageSMem
	return n
}

// NewMemLoadSMemChained creates an SMEM-class MemLoad exactly like
// NewMemLoadSMem and additionally wires it into the construction
// environment's per-buffer effect chain: when src (the buffer node) has a
// known chain head, the load's ImpactBy payload points at it; the load then
// becomes the new head (skeleton memory.go chaining, applied to the SMEM
// tier by quick-261021-jay's tile-lower pass). A nil env (or a buffer with
// no prior access) leaves the load un-chained — identical to
// NewMemLoadSMem.
func (g *Graph) NewMemLoadSMemChained(origin string, env *Env, src Node, coords ...Node) Node {
	n := g.NewMemLoadSMem(origin, src, coords...).(*memLoadNode)
	if head := env.ChainHead(src); head != nil {
		n.effectIns = append(n.effectIns, wirePayloadUse(n, head))
	}
	env.NoteStore(src, n)
	return n
}

// NewMemStoreSMem creates an SMEM-class MemStore of value into the smem
// buffer dst at the coordinates coords (axis order). The verifier's shape
// check enforces the dst operand class (an SMEM-class MemNew node).
func (g *Graph) NewMemStoreSMem(origin string, value, dst Node, coords ...Node) Node {
	ops := make([]Node, 0, 2+len(coords))
	ops = append(ops, value, dst)
	ops = append(ops, coords...)
	n := g.NewNode(KindMemStore, origin, ops...).(*memStoreNode)
	n.class = StorageSMem
	return n
}

// NewMemNewSMem creates an SMEM-class MemNew allocation node of the given
// extent value.
func (g *Graph) NewMemNewSMem(origin string, extent Node) Node {
	n := g.NewNode(KindMemNew, origin, extent).(*memNewNode)
	n.class = StorageSMem
	return n
}

// NewFragLoad creates a FRAG-class MemLoad from the frag buffer src at the
// coordinates coords (axis order). The verifier's shape check enforces the
// src operand class (a FRAG-class MemNew node).
func (g *Graph) NewFragLoad(origin string, src Node, coords ...Node) Node {
	ops := make([]Node, 0, 1+len(coords))
	ops = append(ops, src)
	ops = append(ops, coords...)
	n := g.NewNode(KindMemLoad, origin, ops...).(*memLoadNode)
	n.class = StorageFrag
	return n
}

// NewFragLoadChained creates a FRAG-class MemLoad exactly like NewFragLoad
// and additionally wires it into the construction environment's per-buffer
// effect chain (the NewMemLoadSMemChained contract, FRAG tier).
func (g *Graph) NewFragLoadChained(origin string, env *Env, src Node, coords ...Node) Node {
	n := g.NewFragLoad(origin, src, coords...).(*memLoadNode)
	if head := env.ChainHead(src); head != nil {
		n.effectIns = append(n.effectIns, wirePayloadUse(n, head))
	}
	env.NoteStore(src, n)
	return n
}

// NewFragStore creates a FRAG-class MemStore of value into the frag buffer
// dst at the coordinates coords (axis order). The verifier's shape check
// enforces the dst operand class (a FRAG-class MemNew node).
func (g *Graph) NewFragStore(origin string, value, dst Node, coords ...Node) Node {
	ops := make([]Node, 0, 2+len(coords))
	ops = append(ops, value, dst)
	ops = append(ops, coords...)
	n := g.NewNode(KindMemStore, origin, ops...).(*memStoreNode)
	n.class = StorageFrag
	return n
}

// NewFragNew creates a FRAG-class MemNew allocation node of the given
// extent value.
func (g *Graph) NewFragNew(origin string, extent Node) Node {
	n := g.NewNode(KindMemNew, origin, extent).(*memNewNode)
	n.class = StorageFrag
	return n
}
