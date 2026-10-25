package ir

import "math"

// value.go is the GVN value-numbering surface of the IR
// (quick-261018-nga): ValueHash and ValueEqual, promoted from *Base to
// every concrete node, plus the SideEffectAnalysis query face those methods
// consume. Together they implement the identity-keyed value contract the
// internal/opt/gvn pass hashes and compares with:
//
//   - IDENTITY KEYING (amendment D): every operand and payload target is
//     keyed by SSA node identity — the def node's ID in the hash, def-node
//     identity in equality — NEVER by recursive structural equality over
//     operands. Two one-of-a-kind defs with identical fields (two stores,
//     two allocations) therefore never merge their consumers, while
//     duplicates over the SAME defs do. This is also what makes pure nodes
//     with side-effectful operands safely GVN-eligible: the operand's
//     effect belongs to the operand's evaluation, not to the pure
//     consumer's identity.
//
//   - ONE-OF-A-KIND SENTINEL ([3].3): a node the side-effect analysis marks
//     effectful, a node with an invalid kind, or a nil analysis hashes to
//     ValueHashOneOfAKind and compares false to everything — including
//     ITSELF. Side-effect nodes never participate in value numbering.
//
//   - LAMBDA NODES (quick-261019-ktm revision): a Map/Reduce's lambda is a
//     KindLambda NODE carried as a trailing operand; its params and body
//     are ITS operands. Nothing recurses: the lambda compares by identity
//     like any other operand (the pre-revision recursive body equality is
//     gone with the hidden Lambda record).
//
// CONTRACT (relied upon by gvn-verify): ValueEqual(a, b) == true MUST imply
// ValueHash(a) == ValueHash(b). The hash is best-effort — collisions are
// legal and resolved by ValueEqual — but a hash/equality split (equal
// values hashing differently) would silently break bucket lookup. Hash
// mixing is plain FNV-style arithmetic, not a security hash (T-nga-01;
// cf. the T-1-01-04 note in types.go).

// ValueHashOneOfAKind is the NOT-INTERESTED sentinel of the value-numbering
// contract ([3].3): a node hashing to it is one-of-a-kind — side-effectful,
// invalid, or unanalyzable — and never enters a GVN bucket.
const ValueHashOneOfAKind uint64 = ^uint64(0)

// SideEffectAnalysis is the query face the ACCUMULATED side-effect analysis
// (internal/analysis, quick-261018-nga amendment F) implements for the IR.
// It is defined here, in package ir, so internal/analysis can implement it
// without an import cycle: analysis imports ir, ir never imports analysis.
//
// Semantics: HasSideEffect(n) reports whether n must be excluded from value
// numbering — either it carries an observable effect of its own or an
// order-sensitive taint reached it through the accumulated closure.
// Implementations must be conservative: unknown, erased, or nil nodes
// report true (fail-closed, T-nga-03).
type SideEffectAnalysis interface {
	HasSideEffect(n Node) bool
}

// FNV-1a mixing constants (inline, stdlib-free arithmetic; deterministic
// and allocation-free, unlike hash/fnv streams — the same note as types.go:
// dedup key only, never collision-resistant).
const (
	valueHashOffset uint64 = 14695981039346656037
	valueHashPrime  uint64 = 1099511628211
	// valueHashPerturb is the golden-ratio constant used to perturb a pure
	// node's hash away from the sentinel ([3].1 step 4).
	valueHashPerturb uint64 = 0x9e3779b97f4a7c15
)

// mixUint64 folds one 64-bit word into the running hash (one FNV-1a step).
func mixUint64(h, v uint64) uint64 { return (h ^ v) * valueHashPrime }

// mixUint32 folds a 32-bit word (opcodes, kinds are uint32 underneath).
func mixUint32(h uint64, v uint32) uint64 {
	return mixUint64(h, uint64(v)<<32|uint64(^v)) // both the value and its complement: 0 mixes non-trivially
}

// mixString folds a string bytewise in FNV-1a order. The string payload
// kinds reach this (callNode.name, keyedAttributeNode.key, the launch-map
// query field — quick-261021-jay — and the runtime method/attribute
// payloads, quick-261022-l1o).
func mixString(h uint64, s string) uint64 {
	for i := 0; i < len(s); i++ {
		h = (h ^ uint64(s[i])) * valueHashPrime
	}
	return mixUint64(h, uint64(len(s)))
}

// ValueHash returns the node's value-numbering hash under the accumulated
// side-effect analysis se. The promoted *Base method serves every concrete
// node kind from one implementation keyed on the concrete node reachable as
// b.ssa.node (set in allocNode).
//
// Gates (each returns the sentinel): se == nil (fail-closed: no analysis ⇒
// no GVN), se.HasSideEffect(node) ([2].5), invalid kind (KindInvalid or
// >= KindEnd — the pass manager's fail-closed stance mirrored here).
//
// Atoms (KindInputParam) mix kind + node ID: one node per parameter, never
// structurally pooled. Composites mix: kind; per-kind content fields
// (const type + Float64bits value; opcodes; cast type; call name; project
// idx; attribute key; LIR storage class); the operand slot count and each
// slot's def node ID in canonical slot order (0 for unwired); each payload
// use's target node ID in payload order. Operands are NEVER recursed into
// structurally (amendment D) — since the Input/ImpactBy-only convention
// there is no recursion anywhere in the contract (lambda bodies are
// operands of the KindLambda node, compared by identity like everything
// else).
func (b *Base) ValueHash(se SideEffectAnalysis) uint64 {
	node := hashTarget(b)
	if node == nil || se == nil || !validValueKind(b.kind) || se.HasSideEffect(node) {
		return ValueHashOneOfAKind
	}
	if b.kind == KindInputParam {
		// Atom: identity alone. Distinct params carry distinct IDs, so the
		// former "InputParam atoms" rule survives as the ID mix.
		return mixUint64(mixUint64(valueHashOffset, uint64(b.kind)), b.id)
	}

	h := valueHashOffset
	h = mixUint32(h, uint32(b.kind))
	h = mixUint64(h, contentHash(node))

	// Operand slots: count first, then per-slot def IDs (0 = unwired).
	ops := operandDefsOf(node)
	h = mixUint64(h, uint64(len(ops)))
	for _, d := range ops {
		h = mixUint64(h, nodeIDOrZero(d))
	}

	// Payload uses (the ImpactBy effect inputs of the memory/end nodes):
	// target IDs in payload order. Effectful nodes hash to the sentinel
	// before ever reaching here, so this is completeness plumbing.
	if pb, ok := node.(payloadBearer); ok {
		var targets []uint64
		pb.forEachPayloadUse(func(u SSAUse) { targets = append(targets, nodeIDOrZero(u.Def())) })
		h = mixUint64(h, uint64(len(targets)))
		for _, id := range targets {
			h = mixUint64(h, id)
		}
	}

	if h == ValueHashOneOfAKind {
		h ^= valueHashPerturb // no pure node may collide with the sentinel
	}
	return h
}

func (b *Base) ValueEqual(other Value, se SideEffectAnalysis) bool {
	a, o := b, other
	if o == nil || o.BasePtr() == nil || se == nil || a == nil {
		return false
	}
	if !validValueKind(a.kind) || !validValueKind(o.BasePtr().kind) {
		return false
	}
	na, nb := hashTarget(a), hashTarget(o.BasePtr())
	if se.HasSideEffect(na) || se.HasSideEffect(nb) {
		return false // [3.3]: side-effect nodes are never equal, even to themselves
	}
	if a == o.BasePtr() {
		return true // the same pure node with a valid kind
	}
	if a.kind != o.BasePtr().kind {
		return false
	}
	if a.kind == KindInputParam {
		return false // distinct atoms (identity was checked above)
	}

	if !contentEqual(na, nb) {
		return false
	}

	// Operand slots: same count, pairwise identity (unwired matches
	// unwired; the verifier flags unwired slots separately, WR-01).
	opsA, opsB := operandDefsOf(na), operandDefsOf(nb)
	if len(opsA) != len(opsB) {
		return false
	}
	for i := range opsA {
		if !defIdentity(opsA[i], opsB[i]) {
			return false
		}
	}

	// Payload uses: same count, pairwise target identity.
	if pbA, okA := na.(payloadBearer); okA {
		pbB, okB := nb.(payloadBearer)
		if !okB {
			return false // same kind ⇒ same bearer-ness; a mismatch is corruption
		}
		var ta, tb []Node
		pbA.forEachPayloadUse(func(u SSAUse) { ta = append(ta, u.Def()) })
		pbB.forEachPayloadUse(func(u SSAUse) { tb = append(tb, u.Def()) })
		if len(ta) != len(tb) {
			return false
		}
		for i := range ta {
			if !defIdentity(ta[i], tb[i]) {
				return false
			}
		}
	}
	return true
}

// validValueKind reports whether k is a real node kind (fail-closed on the
// sentinels — mirrors the pass manager's stance on unknown kinds).
func validValueKind(k Kind) bool { return k > KindInvalid && k < KindEnd }

// hashTarget returns the concrete node a Base cell belongs to (b.ssa.node,
// set in allocNode), or nil for a detached Base (zero SSAVar) — the
// fail-closed "unhashable" case.
func hashTarget(b *Base) Node {
	if b == nil || b.ssa == nil {
		return nil
	}
	return b.ssa.node
}

// nodeIDOrZero returns n's per-graph ID, or 0 for nil/unowned nodes (IDs
// start at 1, so 0 is an unambiguous "no def" marker).
func nodeIDOrZero(n Node) uint64 {
	if n == nil || n.BasePtr() == nil {
		return 0
	}
	return n.BasePtr().id
}

// operandDefsOf collects n's operand-slot defs in slot order, nil for
// unwired slots. It is OperandsOf's in-package twin over the concrete node.
func operandDefsOf(n Node) []Node {
	var out []Node
	if impl := asImpl(n); impl != nil {
		impl.forEachOperand(func(u SSAUse) { out = append(out, u.Def()) })
	}
	return out
}

// defIdentity compares two def slots for node identity: both unwired, or
// both wired to the same node (the representative, post-RAUW — Def()
// resolves through the shared ref record).
func defIdentity(a, b Node) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return a.BasePtr() == b.BasePtr()
}

// contentHash mixes the per-kind content fields beyond kind + operands +
// payloads (the same fields contentEqual confirms). Nodes without content
// fields contribute a fixed constant.
func contentHash(n Node) uint64 {
	switch t := n.(type) {
	case *constNode:
		h := typeHash(t.typ)
		return mixUint64(h, math.Float64bits(t.val))
	case *binOpNode:
		return mixUint32(valueHashOffset, uint32(t.op))
	case *unaryOpNode:
		return mixUint32(valueHashOffset, uint32(t.op))
	case *castNode:
		return typeHash(t.typ)
	case *callNode:
		return mixString(valueHashOffset, t.name)
	case *iterClusterProjectNode:
		// D4/D13: the input-slot index is structural content — projects
		// with distinct idx never collide into one bucket entry.
		return mixUint64(valueHashOffset, uint64(t.idx))
	case *keyedAttributeNode:
		// D2/D13: the attribute key is identity content ("init" vs any
		// other key distinguishes attributes wrapping the same value).
		return mixString(valueHashOffset, t.key)
	case *kernelMetaNode:
		// quick-261023-cgq: the name payloads are identity content (the
		// callNode.name precedent). The graph holds at most one, so this
		// arm exists for the content contract's totality, not for CSE.
		return mixString(mixString(valueHashOffset, t.kernelName), t.launcherName)
	case *runtimeQueryNode:
		// §5 companion 6 (quick-261021-jay): the field selector is
		// identity content (the callNode.name precedent) — the DoSoC is
		// an operand and identity-keyed automatically.
		return mixString(valueHashOffset, t.field)
	case *runtimeCallNode:
		// LR-4 companion 6 (quick-261022-l1o): the method payload is
		// identity content. NOTE: EffectOf marks every RuntimeCallMethod
		// full, so these nodes never actually enter a GVN bucket (the
		// one-of-a-kind sentinel); the arm keeps the content contract
		// total regardless.
		return mixString(valueHashOffset, t.method)
	case *runtimeGetAttributeNode:
		// LR-4 companion 6: the attribute payload is identity content —
		// two getattr reads over the same object with distinct attributes
		// never merge.
		return mixString(valueHashOffset, t.attribute)
	case *memLoadNode, *memStoreNode, *memNewNode:
		// fix3 D-h: the storage class is identity content (a same-kind node
		// of another class never merges — the tiered constructors' whole
		// distinction). NOTE: EffectOf marks every memory node full, so
		// these nodes never actually enter a GVN bucket (the one-of-a-kind
		// sentinel); the arms keep the content contract total regardless.
		sc, _ := StorageClassOf(t)
		return mixUint32(valueHashOffset, uint32(sc))
	default:
		return valueHashOffset
	}
}

// contentEqual confirms the per-kind content fields: the exact fields
// contentHash mixes, compared with the equality semantics the printer and
// the type system define (Type.Equal for types — never panics;
// Float64bits for const values — keeps ±0.0 distinct; opcode/name/class
// equality). Only called between same-kind nodes.
func contentEqual(a, b Node) bool {
	switch ta := a.(type) {
	case *constNode:
		tb, ok := b.(*constNode)
		return ok && typesEqual(ta.typ, tb.typ) && math.Float64bits(ta.val) == math.Float64bits(tb.val)
	case *binOpNode:
		tb, ok := b.(*binOpNode)
		return ok && ta.op == tb.op
	case *unaryOpNode:
		tb, ok := b.(*unaryOpNode)
		return ok && ta.op == tb.op
	case *castNode:
		tb, ok := b.(*castNode)
		return ok && typesEqual(ta.typ, tb.typ)
	case *callNode:
		tb, ok := b.(*callNode)
		return ok && ta.name == tb.name
	case *iterClusterProjectNode:
		tb, ok := b.(*iterClusterProjectNode)
		return ok && ta.idx == tb.idx
	case *kernelMetaNode:
		tb, ok := b.(*kernelMetaNode)
		return ok && ta.kernelName == tb.kernelName && ta.launcherName == tb.launcherName
	case *keyedAttributeNode:
		tb, ok := b.(*keyedAttributeNode)
		return ok && ta.key == tb.key
	case *runtimeQueryNode:
		tb, ok := b.(*runtimeQueryNode)
		return ok && ta.field == tb.field
	case *runtimeCallNode:
		tb, ok := b.(*runtimeCallNode)
		return ok && ta.method == tb.method
	case *runtimeGetAttributeNode:
		tb, ok := b.(*runtimeGetAttributeNode)
		return ok && ta.attribute == tb.attribute
	case *memLoadNode:
		tb, ok := b.(*memLoadNode)
		return ok && ta.class == tb.class
	case *memStoreNode:
		tb, ok := b.(*memStoreNode)
		return ok && ta.class == tb.class
	case *memNewNode:
		tb, ok := b.(*memNewNode)
		return ok && ta.class == tb.class
	default:
		return true // no content fields beyond operands/payloads/lambdas
	}
}

// typeHash hashes a Type by content: interned *NumericType by elem + dims
// (mirroring typeContentHash in types.go), any other implementation by a
// fixed constant (future components must add theirs here; ValueEqual still
// guards via Type.Equal, so a missing case degrades to a bucket collision,
// never a false merge). Nil contributes the offset basis.
func typeHash(t Type) uint64 {
	nt, ok := t.(*NumericType)
	if !ok {
		return valueHashOffset
	}
	h := valueHashOffset
	h = mixUint32(h, uint32(nt.elem))
	dims := nt.Dims()
	h = mixUint64(h, uint64(len(dims)))
	for _, d := range dims {
		h = mixString(h, d.Name)
		h = mixUint64(h, d.Extent)
	}
	return h
}

// typesEqual compares two Type references nil-safely: both nil, or
// content-equal via Type.Equal (which never panics — see types.go).
func typesEqual(a, b Type) bool {
	if a == nil || b == nil {
		return a == b
	}
	return a.Equal(b)
}
