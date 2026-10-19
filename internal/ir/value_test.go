package ir

import (
	"math"
	"testing"
)

// value_test.go pins the GVN value-numbering contract (quick-261018-nga):
// the identity-keyed ValueHash/ValueEqual methods promoted from *Base, the
// ValueHashOneOfAKind sentinel gates, and the hash/equality implication
// (equal ⇒ same hash). Amendments D (identity keying), E (lambda machinery
// over Map AND Reduce), and G ([3].3 literal: side-effect nodes are never
// equal, even to themselves) are pinned by name.

// pureAll is a SideEffectAnalysis stub marking nothing effectful.
type pureAll struct{}

func (pureAll) HasSideEffect(Node) bool { return false }

// effectfulAll is a SideEffectAnalysis stub marking everything effectful.
type effectfulAll struct{}

func (effectfulAll) HasSideEffect(Node) bool { return true }

// effectfulIters marks only iter nodes effectful — the shape the accumulated
// analysis (internal/analysis) produces for a binop over an iter value.
type effectfulIters struct{}

func (effectfulIters) HasSideEffect(n Node) bool {
	return n != nil && n.KindOf() == KindIter
}

// valueFixture builds the shared pure-expression graph: one dim, one scalar
// param, one tensor param, and a small const/binop tree.
func valueFixture() (g *Graph, m, sc, out, c1, c2, sum Node) {
	g = NewGraph("value")
	f32 := InternScalar(TF32)
	m = g.NewInputParam("ctor", RoleDim, "m", nil)
	sc = g.NewInputParam("ctor", RoleScalar, "alpha", f32)
	out = g.NewInputParam("ctor", RoleTensor, "out", InternTensor(TF32, []DimRef{{Name: "m"}}))
	c1 = g.NewConst("ctor", f32, 1)
	c2 = g.NewConst("ctor", f32, 2)
	sum = g.NewBinOp("ctor", BinOpAdd, c1, c2)
	return g, m, sc, out, c1, c2, sum
}

// asValue type-asserts a node to Value — every in-repo node satisfies it via
// *Base promotion; a failed assertion is a test bug, not a pass concern.
func asValue(t *testing.T, n Node) Value {
	t.Helper()
	v, ok := n.(Value)
	if !ok {
		t.Fatalf("node %d (%s) does not satisfy Value", n.BasePtr().ID(), n.KindOf())
	}
	return v
}

// TestValueSentinelGates pins [3].3 and the fail-closed defaults: a pure node
// under a pure-stub analysis hashes to a non-sentinel value; the SAME node
// under an effectful-stub analysis returns the sentinel and compares false —
// even against ITSELF.
func TestValueSentinelGates(t *testing.T) {
	_, _, _, _, _, _, sum := valueFixture()
	v := asValue(t, sum)

	if h := v.ValueHash(pureAll{}); h == ValueHashOneOfAKind {
		t.Fatal("pure binop under a pure analysis must hash to a non-sentinel value")
	}
	if h1, h2 := v.ValueHash(pureAll{}), v.ValueHash(pureAll{}); h1 != h2 {
		t.Fatalf("hash must be deterministic across calls: %d vs %d", h1, h2)
	}
	if got := v.ValueHash(effectfulAll{}); got != ValueHashOneOfAKind {
		t.Fatalf("effectful-stub hash = %#x, want the sentinel %#x", got, ValueHashOneOfAKind)
	}
	if v.ValueEqual(v, effectfulAll{}) {
		t.Fatal("[3].3 literal: a side-effect node must not be equal even to itself")
	}
	if v.ValueEqual(nil, pureAll{}) {
		t.Fatal("nil other must compare false")
	}
	if v.ValueEqual(v, nil) {
		t.Fatal("nil analysis must compare false (fail-closed: no analysis means no GVN)")
	}
	if v.ValueHash(nil) != ValueHashOneOfAKind {
		t.Fatal("nil analysis must hash to the sentinel (fail-closed)")
	}
}

// TestValueIdentityKeying pins amendment D: operands are keyed by SSA node
// identity, never by structural recursion. Two binops over the SAME operand
// nodes are one value; two binops over DISTINCT-but-equal consts are two
// values — unequal AND differently hashed.
func TestValueIdentityKeying(t *testing.T) {
	g, _, _, _, c1, c2, _ := valueFixture()
	f32 := InternScalar(TF32)

	b1 := asValue(t, g.NewBinOp("ctor", BinOpAdd, c1, c2))
	b2 := asValue(t, g.NewBinOp("ctor", BinOpAdd, c1, c2))
	if !b1.ValueEqual(b2, pureAll{}) {
		t.Fatal("binops over the same operand nodes must be ValueEqual")
	}
	if b1.ValueHash(pureAll{}) != b2.ValueHash(pureAll{}) {
		t.Fatal("equal values must hash identically (equal ⇒ same hash)")
	}

	// Distinct-but-equal consts: same type, same value, different nodes.
	d1 := g.NewConst("ctor", f32, 1)
	d2 := g.NewConst("ctor", f32, 2)
	l1 := asValue(t, g.NewBinOp("ctor", BinOpAdd, c1, c2))
	l2 := asValue(t, g.NewBinOp("ctor", BinOpAdd, d1, d2))
	if l1.ValueEqual(l2, pureAll{}) {
		t.Fatal("binops over DISTINCT operand defs must never merge (identity keying)")
	}
	if l1.ValueHash(pureAll{}) == l2.ValueHash(pureAll{}) {
		t.Fatal("distinct-operand lookalikes must hash differently")
	}

	// Same opcode with swapped operands is a different value (slot order is
	// canonical, not commutative-aware).
	s1 := asValue(t, g.NewBinOp("ctor", BinOpSub, c1, c2))
	s2 := asValue(t, g.NewBinOp("ctor", BinOpSub, c2, c1))
	if s1.ValueEqual(s2, pureAll{}) {
		t.Fatal("swapped operands must compare unequal (slot order is canonical)")
	}

	// Differing opcodes over identical operands never merge.
	o1 := asValue(t, g.NewBinOp("ctor", BinOpAdd, c1, c2))
	o2 := asValue(t, g.NewBinOp("ctor", BinOpMul, c1, c2))
	if o1.ValueEqual(o2, pureAll{}) {
		t.Fatal("different opcodes must compare unequal")
	}
	if o1.ValueHash(pureAll{}) == o2.ValueHash(pureAll{}) {
		t.Fatal("different opcodes must hash differently")
	}
}

// TestValueLeafPooling pins atom and near-atom pooling: equal consts dedup as
// values; paramRefs pool per referenced param; distinct InputParams are
// one-of-a-kind atoms (never equal, differently hashed); differing const
// values / types compare unequal (Float64bits keeps ±0.0 distinct too).
func TestValueLeafPooling(t *testing.T) {
	g, _, sc, _, _, _, _ := valueFixture()
	f32 := InternScalar(TF32)

	k1 := asValue(t, g.NewConst("ctor", f32, 2))
	k2 := asValue(t, g.NewConst("ctor", f32, 2))
	if !k1.ValueEqual(k2, pureAll{}) || k1.ValueHash(pureAll{}) != k2.ValueHash(pureAll{}) {
		t.Fatal("two NewConst(f32, 2) must be one value")
	}

	p1 := asValue(t, g.NewParamRef("ctor", sc))
	p2 := asValue(t, g.NewParamRef("ctor", sc))
	if !p1.ValueEqual(p2, pureAll{}) {
		t.Fatal("two paramRefs to the SAME InputParam must be one value")
	}
	sc2 := g.NewInputParam("ctor", RoleScalar, "beta", f32)
	p3 := asValue(t, g.NewParamRef("ctor", sc2))
	if p1.ValueEqual(p3, pureAll{}) {
		t.Fatal("paramRefs to DIFFERENT params must compare unequal")
	}

	if asValue(t, sc).ValueEqual(asValue(t, sc2), pureAll{}) {
		t.Fatal("two distinct InputParams are one-of-a-kind atoms — never equal")
	}
	if asValue(t, sc).ValueHash(pureAll{}) == asValue(t, sc2).ValueHash(pureAll{}) {
		t.Fatal("distinct InputParams must hash differently (kind + node ID)")
	}

	neg := asValue(t, g.NewConst("ctor", f32, math.Copysign(0, -1)))
	pos := asValue(t, g.NewConst("ctor", f32, 0))
	if neg.ValueEqual(pos, pureAll{}) {
		t.Fatal("±0.0 must compare unequal (Float64bits semantics)")
	}
	if neg.ValueHash(pureAll{}) == pos.ValueHash(pureAll{}) {
		t.Fatal("±0.0 must hash differently (Float64bits semantics)")
	}
	i32 := asValue(t, g.NewConst("ctor", InternScalar(TI32), 2))
	if k1.ValueEqual(i32, pureAll{}) {
		t.Fatal("consts of different types must compare unequal")
	}
}

// TestValuePureOverEffectfulOperand pins amendment C's GVN-eligibility
// consequence: a PURE binop whose operand is an effectful node (an iter)
// still hashes non-sentinel and equals an identically-keyed binop — pure
// nodes with side-effectful operands participate in GVN.
func TestValuePureOverEffectfulOperand(t *testing.T) {
	g, m, _, _, c1, _, _ := valueFixture()

	i1 := g.NewIter("ctor", m, nil, c1, c1)
	i2 := g.NewIter("ctor", m, nil, c1, c1)
	// Both iters are one-of-a-kind (distinct defs) — the analysis decides
	// effectfulness; identity keying keeps their consumers distinct.
	b1 := asValue(t, g.NewBinOp("ctor", BinOpMul, i1, c1))
	b2 := asValue(t, g.NewBinOp("ctor", BinOpMul, i1, c1))
	se := effectfulIters{}
	if !se.HasSideEffect(i1) {
		t.Fatal("fixture setup: the stub must mark the iter effectful")
	}
	if h := b1.ValueHash(se); h == ValueHashOneOfAKind {
		t.Fatal("a pure binop over an effectful operand must stay GVN-eligible")
	}
	if !b1.ValueEqual(b2, se) {
		t.Fatal("pure binops keyed on the same iter def must be one value")
	}
	b3 := asValue(t, g.NewBinOp("ctor", BinOpMul, i2, c1))
	if b1.ValueEqual(b3, se) {
		t.Fatal("binops over DISTINCT iter defs must never merge (identity keying)")
	}
}

// TestValueInvalidKindsFailClosed pins the fail-closed stance on unknown
// kinds: KindInvalid and Kind >= KindEnd hash to the sentinel and compare
// false — including self-comparison. A pure composite equals itself.
func TestValueInvalidKindsFailClosed(t *testing.T) {
	g, _, _, _, _, _, sum := valueFixture()

	bad1 := asValue(t, g.NewNode(KindInvalid, "t"))
	bad2 := asValue(t, g.NewNode(KindEnd, "t"))
	for name, v := range map[string]Value{"KindInvalid": bad1, "KindEnd": bad2} {
		if got := v.ValueHash(pureAll{}); got != ValueHashOneOfAKind {
			t.Fatalf("%s must hash to the sentinel, got %#x", name, got)
		}
		if v.ValueEqual(v, pureAll{}) {
			t.Fatalf("%s must compare false even to itself", name)
		}
	}

	if v := asValue(t, sum); !v.ValueEqual(v, pureAll{}) {
		t.Fatal("a pure composite must equal itself")
	}
}

// TestValueLambdaMachinery pins amendment E: Map AND Reduce compare their
// lambda payloads through the shared machinery — params by identity, bodies
// by recursive identity-keyed equality. Maps/reduces sharing param atoms and
// body nodes are one value; distinct combiner-param atoms force unequal; the
// body recursion terminates (fail-safe false) under cycles.
//
// quick-261019-ktm D13 (recorded tension): NewMap/NewReduce allocate a fresh
// As/KeyedAttribute wrapper node per call, and operands compare by IDENTITY
// (amendment D) — so two constructor calls with identical shapes are NOT one
// value anymore. The lambda machinery is therefore pinned over as-less maps
// and an attribute-sharing reduce (white-box), and the wrapper-identity
// consequence is pinned explicitly below.
func TestValueLambdaMachinery(t *testing.T) {
	g, m, _, _, c1, _, _ := valueFixture()
	f32 := InternScalar(TF32)

	// Shared atoms: one lambda param, one body over it. Two separate Lambda
	// records over the SAME atoms: the (as-less) maps are one value.
	lp := g.NewLambdaInputParam("ctor", "l", f32)
	body := g.NewBinOp("ctor", BinOpAdd, lp, c1)
	piped := g.NewConst("ctor", f32, 5)
	m1 := asValue(t, g.NewMap("ctor", []Node{piped}, NewLambda([]Node{lp}, body), nil))
	m2 := asValue(t, g.NewMap("ctor", []Node{piped}, NewLambda([]Node{lp}, body), nil))
	if !m1.ValueEqual(m2, pureAll{}) {
		t.Fatal("maps sharing lambda-param atoms and body node must be one value")
	}
	if m1.ValueHash(pureAll{}) != m2.ValueHash(pureAll{}) {
		t.Fatal("equal maps must hash identically")
	}

	// Distinct lambda-param atoms: conservatively unequal (E).
	lp2 := g.NewLambdaInputParam("ctor", "l2", f32)
	body2 := g.NewBinOp("ctor", BinOpAdd, lp2, c1)
	m3 := asValue(t, g.NewMap("ctor", []Node{piped}, NewLambda([]Node{lp2}, body2), nil))
	if m1.ValueEqual(m3, pureAll{}) {
		t.Fatal("maps with distinct lambda-param atoms must compare unequal (conservative)")
	}

	// Reduce participates through the same machinery (E): a reduce sharing
	// the attr wrapper node AND the combiner atoms + body is one value. The
	// twin is built white-box (raw NewNode over the first reduce's attr +
	// a second Lambda record over the same atoms) because two NewReduce
	// calls allocate distinct attr wrappers — the D13 consequence pinned
	// below.
	acc := g.NewLambdaInputParam("ctor", "acc", f32)
	rbody := g.NewBinOp("ctor", BinOpMul, acc, c1)
	init := g.NewConst("ctor", f32, 0)
	r1 := asValue(t, g.NewReduce("ctor", init, piped, NewLambda([]Node{acc}, rbody)))
	r2n := g.NewNode(KindReduce, "ctor", piped, OperandsOf(r1)[1]).(*reduceNode)
	r2n.combiner = NewLambda([]Node{acc}, rbody)
	r2 := asValue(t, r2n)
	if !r1.ValueEqual(r2, pureAll{}) {
		t.Fatal("reduces sharing the attr wrapper, combiner atoms, and body node must be one value (E)")
	}
	if r1.ValueHash(pureAll{}) != r2.ValueHash(pureAll{}) {
		t.Fatal("equal reduces must hash identically")
	}
	acc2 := g.NewLambdaInputParam("ctor", "acc2", f32)
	rbody2 := g.NewBinOp("ctor", BinOpMul, acc2, c1)
	r3 := asValue(t, g.NewReduce("ctor", init, piped, NewLambda([]Node{acc2}, rbody2)))
	if r1.ValueEqual(r3, pureAll{}) {
		t.Fatal("reduces with distinct combiner-param atoms must compare unequal")
	}
	if r1.ValueHash(pureAll{}) == r3.ValueHash(pureAll{}) {
		t.Fatal("reduces with distinct combiner atoms must hash differently")
	}

	// D13 (recorded): identical constructor calls allocate DISTINCT wrapper
	// nodes — the as-clause As and the init KeyedAttribute — and identity
	// keying (amendment D) keeps the stages apart. Stage-level GVN dedup
	// becomes As/attr-node-aware future work.
	m4 := asValue(t, g.NewMap("ctor", []Node{piped}, NewLambda([]Node{lp}, body), []Node{m}))
	m5 := asValue(t, g.NewMap("ctor", []Node{piped}, NewLambda([]Node{lp}, body), []Node{m}))
	if m4.ValueEqual(m5, pureAll{}) {
		t.Fatal("maps with distinct As wrapper nodes must compare unequal (D13: wrapper identity keys the Input list)")
	}
	r4 := asValue(t, g.NewReduce("ctor", init, piped, NewLambda([]Node{acc}, rbody)))
	r5 := asValue(t, g.NewReduce("ctor", init, piped, NewLambda([]Node{acc}, rbody)))
	if r4.ValueEqual(r5, pureAll{}) {
		t.Fatal("reduces with distinct KeyedAttribute wrappers must compare unequal (D13)")
	}
}

// TestValueLambdaBodyCycleGuard pins the recursion guards: a lambda-body
// cycle closed through raw Body pointers (only representable in-package)
// makes both the hash and the equality terminate, failing safe (hash mixes
// a fixed contribution; equality returns false).
func TestValueLambdaBodyCycleGuard(t *testing.T) {
	g, m, _, _, _, _, _ := valueFixture()
	f32 := InternScalar(TF32)

	lp := g.NewLambdaInputParam("ctor", "l", f32)
	x := g.NewMap("ctor", nil, NewLambda([]Node{lp}, nil), []Node{m}).(*mapNode)
	y := g.NewMap("ctor", nil, NewLambda([]Node{lp}, nil), []Node{m}).(*mapNode)
	x.lam.Body = y // raw body cycle: x's body is y, y's body is x
	y.lam.Body = x

	// Both calls must return (termination) and fail safe.
	vx, vy := asValue(t, x), asValue(t, y)
	vx.ValueHash(pureAll{}) //nolint:staticcheck // termination probe
	if vx.ValueHash(pureAll{}) != vx.ValueHash(pureAll{}) {
		t.Fatal("hash must stay deterministic under a body cycle")
	}
	if vx.ValueEqual(vy, pureAll{}) {
		t.Fatal("cyclic bodies must compare false (fail-safe cycle guard)")
	}
	if !vx.ValueEqual(vx, pureAll{}) {
		t.Fatal("self-comparison of a cyclic-body map must still succeed (pair (x,x) recurses symmetrically)")
	}
}

// TestValueOperandCycleTerminates pins that operand cycles closed via
// MutContext.AddUse (the sanctioned test path for manufactured cycles)
// cannot loop the comparison: operands compare by identity, never by
// recursion.
func TestValueOperandCycleTerminates(t *testing.T) {
	g, _, _, _, _, _, _ := valueFixture()

	a := g.NewNode(KindBinOp, "t")
	b := g.NewNode(KindBinOp, "t")
	mctx := &MutContext{Pass: "t"}
	mctx.AddUse(a, b)
	mctx.AddUse(b, a)

	va, vb := asValue(t, a), asValue(t, b)
	if va.ValueEqual(vb, pureAll{}) {
		t.Fatal("mutually-cycling distinct binops must compare unequal (identity keying)")
	}
	if !va.ValueEqual(va, pureAll{}) {
		t.Fatal("a cyclic node must still equal itself")
	}
}

// TestValueContentFields pins the per-kind content contributions: cast types,
// call names, unary opcodes, and select/cast identity — the fields the hash
// mixes and the equality confirms beyond kind + operands.
func TestValueContentFields(t *testing.T) {
	g, _, _, _, c1, _, _ := valueFixture()

	u1 := asValue(t, g.NewUnaryOp("ctor", UnaryOpNeg, c1))
	u2 := asValue(t, g.NewUnaryOp("ctor", UnaryOpNeg, c1))
	u3 := asValue(t, g.NewUnaryOp("ctor", UnaryOpPos, c1))
	if !u1.ValueEqual(u2, pureAll{}) || u1.ValueEqual(u3, pureAll{}) {
		t.Fatal("unary equality must follow the opcode")
	}
	if u1.ValueHash(pureAll{}) == u3.ValueHash(pureAll{}) {
		t.Fatal("different unary opcodes must hash differently")
	}

	k1 := asValue(t, g.NewCast("ctor", InternScalar(TF16), c1))
	k2 := asValue(t, g.NewCast("ctor", InternScalar(TF16), c1))
	k3 := asValue(t, g.NewCast("ctor", InternScalar(TF32), c1))
	if !k1.ValueEqual(k2, pureAll{}) || k1.ValueEqual(k3, pureAll{}) {
		t.Fatal("cast equality must follow the target type")
	}
	if k1.ValueHash(pureAll{}) == k3.ValueHash(pureAll{}) {
		t.Fatal("different cast targets must hash differently")
	}

	n1 := asValue(t, g.NewCall("ctor", "exp", c1))
	n2 := asValue(t, g.NewCall("ctor", "exp", c1))
	n3 := asValue(t, g.NewCall("ctor", "log", c1))
	if !n1.ValueEqual(n2, pureAll{}) || n1.ValueEqual(n3, pureAll{}) {
		t.Fatal("call equality must follow the builtin name")
	}
	if n1.ValueHash(pureAll{}) == n3.ValueHash(pureAll{}) {
		t.Fatal("different builtin names must hash differently")
	}

	s1 := asValue(t, g.NewSelect("ctor", c1, c1, c1))
	s2 := asValue(t, g.NewSelect("ctor", c1, c1, c1))
	if !s1.ValueEqual(s2, pureAll{}) {
		t.Fatal("selects over the same operand nodes must be one value")
	}

	// float64 bit pattern of the sentinel perturbation: a pure node whose
	// content would collide with the sentinel is perturbed, never confused
	// with one-of-a-kind. Deterministic across calls is pinned above.
	_ = math.Float64bits // keep math imported for the ±0.0 test above
}
