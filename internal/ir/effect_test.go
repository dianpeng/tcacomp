package ir

import "testing"

// twoTensorGraph builds the shared fixture for the effect-chain tests: two
// rank-1 tensors and one dim, plus a typed constant as a store value.
func twoTensorGraph(t *testing.T) (g *Graph, m, out1, out2, val Node) {
	t.Helper()
	g = NewGraph("effect")
	f32 := InternScalar(TF32)
	m = g.NewInputParam("ctor", RoleDim, "m", nil)
	out1 = g.NewInputParam("ctor", RoleTensor, "out1", InternTensor(TF32, []DimRef{{Name: "m"}}))
	out2 = g.NewInputParam("ctor", RoleTensor, "out2", InternTensor(TF32, []DimRef{{Name: "m"}}))
	val = g.NewConst("ctor", f32, 1.0)
	return g, m, out1, out2, val
}

// TestEffectVerdicts pins the free/full split: stores and returns are
// effectful, everything else (including the value feeding a store) is free.
func TestEffectVerdicts(t *testing.T) {
	g, m, out1, _, val := twoTensorGraph(t)
	if e := EffectOf(val); e.Has() {
		t.Fatalf("const must be effect-free, got %s", e)
	}
	st := g.NewMemStore("ctor", out1, []Node{m}, val)
	if e := EffectOf(st); !e.Has() {
		t.Fatal("store must be fully effectful")
	}
	if e := EffectOf(g.NewReturn("ctor", st)); !e.Has() {
		t.Fatal("kernel return must be effectful (it observes memory through its operand)")
	}
	if got := NewEffectFree().String(); got != "free" {
		t.Fatalf("free verdict renders %q, want \"free\"", got)
	}
	if got := NewEffectFull().String(); got != "full" {
		t.Fatalf("full verdict renders %q, want \"full\"", got)
	}
	if NewEffectFree().Kind() != EffectFree || NewEffectFull().Kind() != EffectFull {
		t.Fatal("verdict Kind() must return the constants")
	}
}

// TestUnchainedStoreHasNoImpactBy pins the chain-head invariant: a store
// built without an env (or as the first store of its chain) carries NO
// ordering edges — an ImpactBy-nil node can still be effectful (it IS the
// memory node, per the skeleton's invariant).
func TestUnchainedStoreHasNoImpactBy(t *testing.T) {
	g, m, out1, _, val := twoTensorGraph(t)
	st := g.NewMemStore("ctor", out1, []Node{m}, val)
	if got := ImpactBy(st); len(got) != 0 {
		t.Fatalf("un-chained store has %d ImpactBy edges, want 0", len(got))
	}
	if got := ImpactBy(val); got != nil {
		t.Fatalf("const must carry no ImpactBy, got %v", got)
	}
}

// TestSameTensorStoresChain pins the core sea-of-nodes ordering property:
// two stores to ONE tensor through the env form an explicit
// second-ImpactBy-first edge; the first store stays chain-head-free.
func TestSameTensorStoresChain(t *testing.T) {
	g, m, out1, _, val := twoTensorGraph(t)
	env := NewEnv()

	st1 := g.NewMemStoreChained("ctor", env, out1, []Node{m}, val)
	st2 := g.NewMemStoreChained("ctor", env, out1, []Node{m}, val)

	if got := ImpactBy(st1); len(got) != 0 {
		t.Fatalf("first store must not be ordered after anything, got %d edges", len(got))
	}
	edges := ImpactBy(st2)
	if len(edges) != 1 {
		t.Fatalf("second store must carry exactly 1 ImpactBy edge, got %d", len(edges))
	}
	if edges[0].Def() != st1 {
		t.Fatalf("second store's ImpactBy must resolve to the first store, got %v", edges[0].Def())
	}
	// The edge is an ordinary payload record: it appears in the FIRST
	// store's def-use direction too (Walk/Verify follow both directions).
	uses := UsesOf(st1)
	if len(uses) != 1 || uses[0] != edges[0] {
		t.Fatalf("first store's ref list must hold exactly the ImpactBy record, got %v", uses)
	}
	// The env's chain head advanced to the newest store.
	if h := env.ChainHead(out1); h != st2 {
		t.Fatalf("env chain head = %v, want the second store", h)
	}
}

// TestCrossTensorStoresStayIsolated pins the restrict-alias benefit: stores
// to DISTINCT tensors never order against each other — independent chains.
func TestCrossTensorStoresStayIsolated(t *testing.T) {
	g, m, out1, out2, val := twoTensorGraph(t)
	env := NewEnv()

	g.NewMemStoreChained("ctor", env, out1, []Node{m}, val)
	stB := g.NewMemStoreChained("ctor", env, out2, []Node{m}, val)

	if got := ImpactBy(stB); len(got) != 0 {
		t.Fatalf("store to a fresh tensor must not be ordered after the other tensor's chain, got %d edges", len(got))
	}
}

// TestEffectRootsRegistered pins Level-1 rooting: every effectful node
// (allocation order, no map iteration) is registered at construction so
// Walk/Verify/DCE have non-statement roots; effect-free nodes are not.
func TestEffectRootsRegistered(t *testing.T) {
	g, m, out1, _, val := twoTensorGraph(t)
	bin := g.NewBinOp("ctor", BinOpAdd, val, val)
	st := g.NewMemStore("ctor", out1, []Node{m}, bin)
	ret := g.NewReturn("ctor", st)

	roots := g.effectRoots
	if len(roots) != 2 || roots[0] != st || roots[1] != ret {
		t.Fatalf("effect roots = %v, want [store return] in allocation order", roots)
	}
	if containsNode(g.nodes, bin) != true {
		t.Fatal("the node registry must hold every allocated node (binop)")
	}
	if len(g.nodes) != 7 { // dim m, tensors out1/out2, val, bin, store, return
		t.Fatalf("node registry holds %d nodes, want 7 (3 params, const, binop, store, return)", len(g.nodes))
	}
}

func containsNode(ns []Node, want Node) bool {
	for _, n := range ns {
		if n == want {
			return true
		}
	}
	return false
}

// TestImpactByPayloadEnumeration pins the payload contract the verifier and
// Walk rely on (T-quick-01): the ImpactBy records ride the store's ordinary
// payloadBearer enumeration AFTER the target access, so both chain directions
// see them as first-class edges. (End-to-end verifier coverage lands with
// the Level-1 Walk roots in Task 2 of this refactor.)
func TestImpactByPayloadEnumeration(t *testing.T) {
	g, m, out1, _, val := twoTensorGraph(t)
	env := NewEnv()
	g.NewMemStoreChained("ctor", env, out1, []Node{m}, val)
	st2 := g.NewMemStoreChained("ctor", env, out1, []Node{m}, val).(*memStoreNode)

	// Payload order: tensor, axes, then effect inputs.
	var payload []Node
	st2.forEachPayloadUse(func(u SSAUse) { payload = append(payload, u.Def()) })
	want := []Node{out1, m, g.effectRoots[0]} // tensor, axis, first store
	if len(payload) != 3 {
		t.Fatalf("store payload enumeration = %d uses, want 3 (tensor, axis, ImpactBy)", len(payload))
	}
	for i := range want {
		if payload[i] != want[i] {
			t.Fatalf("payload use %d = %v, want %v", i, payload[i], want[i])
		}
	}

	// The ImpactBy record is owned by the second store in BOTH directions:
	// it resolves from the user side (ImpactBy) and from the def side
	// (UsesOf of the first store).
	edges := ImpactBy(st2)
	uses := UsesOf(edges[0].Def())
	if len(uses) != 1 || uses[0] != edges[0] {
		t.Fatalf("chain-head def must own exactly the ImpactBy record, got %v", uses)
	}
}

// TestNilEnvStoreChained pins the nil-env path: NewMemStoreChained with a nil
// environment degrades to exactly NewStore (no edges, still an effect root).
func TestNilEnvStoreChained(t *testing.T) {
	g, m, out1, _, val := twoTensorGraph(t)
	var env *Env
	st := g.NewMemStoreChained("ctor", env, out1, []Node{m}, val)
	if got := ImpactBy(st); len(got) != 0 {
		t.Fatalf("nil-env chained store must carry no ImpactBy edges, got %d", len(got))
	}
	if len(g.effectRoots) != 1 || g.effectRoots[0] != st {
		t.Fatal("nil-env chained store must still register as an effect root")
	}
}
