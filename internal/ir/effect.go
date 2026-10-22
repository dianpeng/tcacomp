package ir

import "slices"

// This file is the side-effect model of the two-level IR (quick-261017-vdj,
// tinycudaopto skeleton effect.go/memory.go adapted tcacomp-local — reference
// only, NOT an import).
//
// Between construction and Materialize the graph is a pure SEA OF NODES: there
// is no BasicBlock, no Statement list, and therefore no implicit "program
// order" to serialize memory operations. The total order of side effects is
// carried EXPLICITLY instead:
//
//   - The graph has ONE end node (R2, node_end.go): the first KindReturn,
//     carrying the kernel's returned values AND — via its ImpactBy payload —
//     the final chain state of every mem-param. Level-1 liveness roots at the
//     two-root set (R3): the RoleTensor params plus the end node. Effectful
//     nodes are NOT roots: a store no end path reaches is unreachable
//     building debris, and SMEM/FRAG stores are not observable at all.
//
//   - A memory access additionally records its per-tensor ordering edge:
//     effectIns, the ImpactBy payload — SSAUse records pointing at the
//     previous chain head for the same tensor. Chains include READS as well
//     as writes (skeleton memory.go): NewMemLoadChained chains a load exactly
//     like NewMemStoreChained chains a store. Because effectIns ride the
//     ordinary payloadBearer enumeration, the verifier's existing use-def/
//     def-use invariants and Walk cover them in BOTH chain directions with
//     zero new checkers (T-quick-01), and Level-1 DCE follows them so an
//     overwritten chain keeps its predecessors alive.
//
// The per-tensor chains are maintained by a construction-time environment
// object (Env). RESTRICT-ALIAS RULE (skeleton memory.go): the model works
// — concurrently tracking isolated side-effect chains per symbol — solely
// because kernel tensors NEVER alias; two distinct tensor parameters never
// name the same memory. Chain identity is the tensor's InputParam NODE
// pointer (review P1R-01: identity is never a name).
//
// Effect verdicts (skeleton effect.go shape, local naming): for now there is
// no fine-grained categorization, just free vs. full (X-3). As amended by
// quick-261018-nga decision A, EVERY FamilyMemory node is node-local
// EFFECTFUL — loads included, nil ImpactBy included — under the single-slot
// memory model: each memory region is one slot, tensor subscripts are too
// coarse to disprove aliasing between a load and a write, so all operations
// on one memory node must serialize. Allocations (MemNew/MemNewSMem/FragNew)
// are effectful under the fresh-object rule as well: merging distinct
// allocations is unsound — each allocation denotes a distinct memory object
// (LLVM never CSEs allocas). The verdict stays SELF-SCOPED ([4].3): no
// lattice meet, no operand accumulation — the accumulated GVN gatekeeper
// lives in internal/analysis. DCE dead-code eligibility is governed by
// two-root-set reachability (R3), NOT by this verdict: unreferenced memory
// ops (an SMEM store off every root path included) remain sweepable.

// Effect verdict constants (Effect.Kind values).
const (
	// EffectFree marks a node with no observable side effect.
	EffectFree = iota
	// EffectFull marks a node with an observable side effect (GMEM store,
	// kernel return).
	EffectFull
)

// Effect is the side-effect verdict of a node: free vs. full. It is a small
// value-type verdict mirroring the skeleton's Effect; there is deliberately
// no per-effect categorization yet.
type Effect struct {
	kind int
}

// NewEffectFree builds the side-effect-free verdict.
func NewEffectFree() Effect { return Effect{kind: EffectFree} }

// NewEffectFull builds the fully-side-effectful verdict.
func NewEffectFull() Effect { return Effect{kind: EffectFull} }

// Kind returns the verdict constant (EffectFree or EffectFull).
func (e Effect) Kind() int { return e.kind }

// Has reports whether the node has any side effect.
func (e Effect) Has() bool { return e.kind != EffectFree }

// String renders the canonical verdict name for diagnostics.
func (e Effect) String() string {
	if e.Has() {
		return "full"
	}
	return "free"
}

// EffectOf returns the side-effect verdict of n (skeleton Value.Effect,
// delivered as a function to keep the Node interface at exactly two methods —
// directive 4's interface-width contract). As amended by quick-261018-nga
// decision A, EVERY FamilyMemory node (GMEM MemLoad/MemStore/MemNew, the
// lowered SMEM and FRAG families — loads and allocations included, nil
// ImpactBy included) and the kernel return are fully effectful: the
// single-slot memory model serializes all operations on one memory region,
// and the fresh-object rule keeps distinct allocations unmergeable. Pure
// kinds — the scalar-expression family, map/reduce stages, the value-level
// dimvar objects — are free. P-2 (quick-261022-l1o) adds the one
// final-lower arm: KindRuntimeCallMethod is FULL (an ordered method call on
// a stateful reducer object — never pure, never GVN-CSE'd); every other
// Runtime/Collective kind (the constructions, RuntimeGetAttribute) stays
// free — the objects are immutable and no pass runs after final-lower, so
// value-identical reducer constructions can never be CSE-merged anyway
// (mirrored in lockstep by internal/analysis/sideeffect's baseVerdict).
// The verdict is node-local and self-scoped
// ([4].3); DCE liveness is root-reachability based (R3), so a memory op no
// root path reaches is still dead-code-eligible.
//
// Consumers (grep-audited at quick-261018-nga, refreshed at
// quick-261022-b2m after the schedule rewrite deleted the statement-level
// scheduler and its licmHoistableKinds whitelist): the schedule pass now
// gates effectful hoisting itself — a memory-ACCESS kind with invariant
// operands floors at its operand-containing loop (HI-02), while the
// allocation kinds (MemNew/MemNewSMem/FragNew) are exempt: hoisting a
// shared declaration is semantics-preserving. Neither DCE, Materialize's
// isEffectKind, nor the legality windows consult EffectOf (that part is
// unchanged).
func EffectOf(n Node) Effect {
	if n == nil {
		return NewEffectFree()
	}
	k := n.KindOf()
	if k.Family() == FamilyMemory || k == KindReturn || k == KindRuntimeCallMethod {
		return NewEffectFull()
	}
	return NewEffectFree()
}

// ImpactBy returns the effect-input uses of n — the SSAUse records forming
// its per-tensor ordering edges — or nil when n carries none. It is the
// exported read face of the memory family's effectIns payload; like the
// skeleton's Value.ImpactBy, a nil result means "no ordering constraint of
// its own" (which includes every non-memory node and every chain-head
// access).
func ImpactBy(n Node) []SSAUse {
	if eb, ok := n.(effectBearer); ok {
		return eb.impactBy()
	}
	return nil
}

// effectBearer is implemented by nodes carrying ImpactBy effect inputs; the
// payloadBearer enumeration exposes the same records to the verifier and
// Walk, this accessor exposes them to scheduling and tests, and the removal
// hook lets the MutContext.RemoveImpactBy choke point drop one record by
// identity (quick-261020-ejf condition C3) — the effect-chain twin of
// operandList.removeSlot.
type effectBearer interface {
	impactBy() []SSAUse
	// removeImpactUse drops the effectIns handle whose record identity
	// matches u.rec, reporting whether one was held. Only the HANDLE COPY
	// is dropped (CR-01/CR-02): the record's life cycle belongs to the
	// choke point that calls this hook.
	removeImpactUse(u SSAUse) bool
}

// dropImpactHandle is the shared body of every bearer's removeImpactUse: a
// record-identity scan over the bearer's effectIns slice, mirroring
// operandList.removeSlot's slices.Delete drop of a handle copy.
func dropImpactHandle(ins *[]SSAUse, u SSAUse) bool {
	if u.rec == nil {
		return false
	}
	for i, s := range *ins {
		if s.rec == u.rec { // record-identity comparison
			*ins = slices.Delete(*ins, i, i+1)
			return true
		}
	}
	return false
}

// Env is the construction-time side-effect environment (skeleton memory.go's
// "global state, ie environment"): one isolated effect chain per kernel
// tensor. Each chain is keyed by the tensor's RoleTensor InputParam NODE —
// never by name (review P1R-01) — and holds the chain's current head: the
// most recent memory access (store OR load, see node_memory.go) against that
// tensor. An access created through the env-aware path (NewMemStoreChained/
// NewMemLoadChained) wires ImpactBy = the current head, then becomes the new
// head, so two accesses to one tensor form an explicit
// second-ImpactBy-first edge while accesses to distinct tensors stay
// isolated.
//
// The restrict-alias rule is the precondition that makes name-free chaining
// sound: kernel tensors never alias (skeleton memory.go rationale). It is a
// documented construction precondition — enforcement arrives with the
// frontend, exactly like the skeleton.
//
// An Env is graph-agnostic (chains are keyed by *Base identity), so one Env
// can serve one builder's whole construction; a nil *Env is accepted by the
// chained constructors and simply leaves the access un-chained.
type Env struct {
	heads map[*Base]Node // tensor InputParam Base -> current chain-head memory node
}

// NewEnv creates an empty construction environment: no known chains.
func NewEnv() *Env { return &Env{heads: make(map[*Base]Node)} }

// NoteStore records store as the new chain head for tensor (the tensor's
// RoleTensor InputParam node). It is the Env half of the env-aware memory
// path; the chained constructors call it — direct use is for white-box
// tests. Despite the historical name it accepts any memory access (loads
// chain too, skeleton memory.go).
func (e *Env) NoteStore(tensor Node, store Node) {
	if e == nil || tensor == nil || store == nil {
		return
	}
	e.heads[tensor.BasePtr()] = store
}

// ChainHead returns the current chain head for tensor, or nil when no access
// against that tensor has been noted.
func (e *Env) ChainHead(tensor Node) Node {
	if e == nil || tensor == nil {
		return nil
	}
	return e.heads[tensor.BasePtr()]
}

// FinalStates returns the current chain head of every tensor in tensors, in
// argument order (nil for an untouched tensor). It is the Env read face the
// end-node wiring uses: NewReturnChained consumes exactly these states as
// its ImpactBy payload (R2).
func (e *Env) FinalStates(tensors []Node) []Node {
	states := make([]Node, len(tensors))
	for i, t := range tensors {
		states[i] = e.ChainHead(t)
	}
	return states
}

// NewMemStoreChained creates a store exactly like NewMemStore and additionally
// wires it into the construction environment's per-tensor effect chain: when
// tensor has a known chain head, the new store's ImpactBy payload points at
// it (one SSAUse per chain head, via the ordinary payload wiring), giving the
// sea-of-nodes its explicit total order for that tensor; the new store then
// becomes the chain head. A nil env (or a tensor with no prior access) leaves
// the store un-chained — identical to NewMemStore. The store stays live
// through the end node's ImpactBy path once the builder wires it (R2/R3), so
// Level-1 Walk/Verify/DCE see it without any statement structure.
func (g *Graph) NewMemStoreChained(origin string, env *Env, tensor Node, axes []Node, value Node) Node {
	n := g.NewMemStore(origin, tensor, axes, value).(*memStoreNode)
	if head := env.ChainHead(tensor); head != nil {
		n.effectIns = append(n.effectIns, wirePayloadUse(n, head))
	}
	env.NoteStore(tensor, n)
	return n
}
