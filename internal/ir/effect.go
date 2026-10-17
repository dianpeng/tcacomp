package ir

// This file is the side-effect model of the two-level IR (quick-261017-vdj,
// tinycudaopto skeleton effect.go/memory.go adapted tcacomp-local — reference
// only, NOT an import).
//
// Between construction and Materialize the graph is a pure SEA OF NODES: there
// is no BasicBlock, no Statement list, and therefore no implicit "program
// order" to serialize memory operations. The total order of side effects is
// carried EXPLICITLY instead:
//
//   - Every effectful node (store, tile store, kernel return) is registered in
//     Graph.effectRoots at allocation, so Level-1 Walk/Verify/DCE have
//     non-statement roots (allocation order — deterministic, never map
//     iteration).
//
//   - A store additionally records its per-tensor ordering edge: effectIns,
//     the ImpactBy payload — SSAUse records pointing at the previous chain
//     head for the same tensor. Because effectIns ride the ordinary
//     payloadBearer enumeration (storeNode.forEachPayloadUse appends them
//     after the target access), the verifier's existing use-def/def-use
//     invariants and Walk cover them in BOTH chain directions with zero new
//     checkers (T-quick-01), and Level-1 DCE follows them so an overwritten
//     chain keeps its predecessors alive.
//
// The per-tensor chains are maintained by a construction-time environment
// object (Env). RESTRICT-ALIAS RULE (skeleton memory.go): the model works
// — concurrently tracking isolated side-effect chains per symbol — solely
// because kernel tensors NEVER alias; two distinct tensor parameters never
// name the same memory. Chain identity is the tensor's InputParam NODE
// pointer (review P1R-01: identity is never a name).
//
// Effect verdicts (skeleton effect.go shape, local naming): for now there is
// no fine-grained categorization, just free vs. full. A store is fully
// effectful; a kernel return is effectful-via-operand — it reads the stored
// value through its operand SSAUse, so it carries NO chain of its own and its
// ImpactBy is empty.

// Effect verdict constants (Effect.Kind values).
const (
	// EffectFree marks a node with no observable side effect.
	EffectFree = iota
	// EffectFull marks a node with an observable side effect (store, tile
	// store, kernel return).
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
// directive 4's interface-width contract). Stores, tile stores, and kernel
// returns are fully effectful; every other kind is free. Effect-free subtrees
// are dead-code-eligible at Level 1 when unreachable from the roots.
func EffectOf(n Node) Effect {
	if n == nil {
		return NewEffectFree()
	}
	switch n.KindOf() {
	case KindStore, KindStoreTile, KindReturn:
		return NewEffectFull()
	}
	return NewEffectFree()
}

// ImpactBy returns the effect-input uses of n — the SSAUse records forming
// its per-tensor ordering edges — or nil when n carries none. It is the
// exported read face of the storeNode.effectIns payload; like the skeleton's
// Value.ImpactBy, a nil result means "no ordering constraint of its own"
// (which includes every non-store node and every chain-head store).
func ImpactBy(n Node) []SSAUse {
	if eb, ok := n.(effectBearer); ok {
		return eb.impactBy()
	}
	return nil
}

// effectBearer is implemented by nodes carrying ImpactBy effect inputs; the
// payloadBearer enumeration exposes the same records to the verifier and
// Walk, this accessor exposes them to scheduling and tests.
type effectBearer interface {
	impactBy() []SSAUse
}

// Env is the construction-time side-effect environment (skeleton memory.go's
// "global state, ie environment"): one isolated effect chain per kernel
// tensor. Each chain is keyed by the tensor's RoleTensor InputParam NODE —
// never by name (review P1R-01) — and holds the chain's current head: the
// most recent store node against that tensor. A store created through the
// env-aware path (NewStoreChained) wires ImpactBy = the current head, then
// becomes the new head, so two stores to one tensor form an explicit
// second-ImpactBy-first edge while stores to distinct tensors stay isolated.
//
// The restrict-alias rule is the precondition that makes name-free chaining
// sound: kernel tensors never alias (skeleton memory.go rationale). It is a
// documented construction precondition — enforcement arrives with the
// frontend, exactly like the skeleton.
//
// An Env is graph-agnostic (chains are keyed by *Base identity), so one Env
// can serve one builder's whole construction; a nil *Env is accepted by
// NewStoreChained and simply leaves the store un-chained.
type Env struct {
	heads map[*Base]Node // tensor InputParam Base -> current chain-head store node
}

// NewEnv creates an empty construction environment: no known chains.
func NewEnv() *Env { return &Env{heads: make(map[*Base]Node)} }

// NoteStore records store as the new chain head for tensor (the tensor's
// RoleTensor InputParam node). It is the Env half of the env-aware store
// path; NewStoreChained calls it — direct use is for white-box tests.
func (e *Env) NoteStore(tensor Node, store Node) {
	if e == nil || tensor == nil || store == nil {
		return
	}
	e.heads[tensor.BasePtr()] = store
}

// ChainHead returns the current chain head for tensor, or nil when no store
// against that tensor has been noted.
func (e *Env) ChainHead(tensor Node) Node {
	if e == nil || tensor == nil {
		return nil
	}
	return e.heads[tensor.BasePtr()]
}

// NewStoreChained creates a store exactly like NewStore and additionally
// wires it into the construction environment's per-tensor effect chain: when
// tensor has a known chain head, the new store's ImpactBy payload points at
// it (one SSAUse per chain head, via the ordinary payload wiring), giving the
// sea-of-nodes its explicit total order for that tensor; the new store then
// becomes the chain head. A nil env (or a tensor with no prior store) leaves
// the store un-chained — identical to NewStore. The store is registered as an
// effect root either way (allocation-time, kind-based), so Level-1
// Walk/Verify/DCE see it without any statement structure.
func (g *Graph) NewStoreChained(origin string, env *Env, tensor Node, axes []Node, value Node) Node {
	n := g.NewStore(origin, tensor, axes, value).(*storeNode)
	if head := env.ChainHead(tensor); head != nil {
		n.effectIns = append(n.effectIns, wirePayloadUse(n, head))
	}
	env.NoteStore(tensor, n)
	return n
}
