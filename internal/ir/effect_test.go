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

// TestEndNodeRegistered pins R2's end-node contract: the FIRST return
// created on a graph becomes its end node (Graph.End), and the graph's node
// registry holds every allocated node in allocation order.
func TestEndNodeRegistered(t *testing.T) {
	g, m, out1, _, val := twoTensorGraph(t)
	bin := g.NewBinOp("ctor", BinOpAdd, val, val)
	st := g.NewMemStore("ctor", out1, []Node{m}, bin)
	ret := g.NewReturn("ctor", st)

	if g.End() != ret {
		t.Fatalf("Graph.End() = %v, want the first return node (the graph's end node)", g.End())
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
// see them as first-class edges.
func TestImpactByPayloadEnumeration(t *testing.T) {
	g, m, out1, _, val := twoTensorGraph(t)
	env := NewEnv()
	st1 := g.NewMemStoreChained("ctor", env, out1, []Node{m}, val)
	st2 := g.NewMemStoreChained("ctor", env, out1, []Node{m}, val).(*memStoreNode)

	// Payload order: tensor, axes, then effect inputs.
	var payload []Node
	st2.forEachPayloadUse(func(u SSAUse) { payload = append(payload, u.Def()) })
	want := []Node{out1, m, st1} // tensor, axis, first store
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

// TestNilEnvStoreChained pins the nil-env path: NewMemStoreChained with a
// nil environment degrades to exactly NewMemStore (no ordering edges, still
// fully effectful).
func TestNilEnvStoreChained(t *testing.T) {
	g, m, out1, _, val := twoTensorGraph(t)
	var env *Env
	st := g.NewMemStoreChained("ctor", env, out1, []Node{m}, val)
	if got := ImpactBy(st); len(got) != 0 {
		t.Fatalf("nil-env chained store must carry no ImpactBy edges, got %d", len(got))
	}
	if e := EffectOf(st); !e.Has() {
		t.Fatal("nil-env chained store must still be fully effectful")
	}
}

// TestReturnChainedPinsR2 pins the end node's chain wiring (R2): the end
// node's ImpactBy payload carries the FINAL chain state of every tensor
// passed to NewReturnChained — in tensor-argument order, heads only —
// while the returned values stay ordinary operand slots.
func TestReturnChainedPinsR2(t *testing.T) {
	g, m, out1, out2, val := twoTensorGraph(t)
	env := NewEnv()
	g.NewMemStoreChained("ctor", env, out1, []Node{m}, val)        // out1's chain: head advances below
	st2 := g.NewMemStoreChained("ctor", env, out2, []Node{m}, val) // out2's chain head
	st3 := g.NewMemStoreChained("ctor", env, out1, []Node{m}, val) // out1's final chain head

	ret := g.NewReturnChained("ctor", env, []Node{out1, out2}, st3).(*returnNode)
	if g.End() != ret {
		t.Fatal("NewReturnChained must create the graph's end node")
	}

	// Returned values: ordinary operands.
	if got := ret.ops[0].Def(); got != st3 {
		t.Fatalf("end node's returned value = %v, want st3", got)
	}

	// Final chain states: ImpactBy payload, one edge per tensor, heads only.
	imp := ImpactBy(ret)
	if len(imp) != 2 {
		t.Fatalf("end node must carry 2 final-chain-state edges, got %d", len(imp))
	}
	if imp[0].Def() != st3 || imp[1].Def() != st2 {
		t.Fatalf("end node ImpactBy = [%v %v], want [st3(st1's chain head) st2]", imp[0].Def(), imp[1].Def())
	}

	// The edges are ordinary payload records: Walk/Verify see them, so the
	// stores stay live through the end node even without a value edge.
	if got := len(UsesOf(st2)); got != 1 {
		t.Fatalf("st2 (a returned value AND a chain head) keeps its edge set: %d uses", got)
	}
	if vs := Verify(g); len(vs) != 0 {
		t.Fatalf("R2-chained graph must verify clean: %v", vs)
	}
}

// TestEnvFinalStates pins the Env read face the builder uses to wire the
// end node: FinalStates reports each tensor's current chain head in
// argument order, nil for untouched tensors.
func TestEnvFinalStates(t *testing.T) {
	g, m, out1, out2, val := twoTensorGraph(t)
	env := NewEnv()
	st1 := g.NewMemStoreChained("ctor", env, out1, []Node{m}, val)

	states := env.FinalStates([]Node{out1, out2})
	if len(states) != 2 || states[0] != st1 || states[1] != nil {
		t.Fatalf("FinalStates = %v, want [st1 nil] (out2 untouched)", states)
	}
}
