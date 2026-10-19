package ir

import "testing"

// TestCatalogListsEveryKind pins the catalog invariants (R4e): the listing
// covers exactly the real kinds (len == int(KindEnd)-1), the HIR/LIR tier
// split is exactly the enum's tier boundary, and tier queries never lie.
func TestCatalogListsEveryKind(t *testing.T) {
	all := AllKinds()
	if len(all) != int(KindEnd)-1 {
		t.Fatalf("catalog must list exactly int(KindEnd)-1 = %d kinds, got %d", int(KindEnd)-1, len(all))
	}
	for i, k := range all {
		if uint32(k) != uint32(i+1) {
			t.Fatalf("catalog row %d describes kind %d (%s): rows must be in enum order", i, k, k)
		}
		if k == KindInvalid || k == KindEnd {
			t.Fatalf("sentinel kind %s must not appear in the catalog", k)
		}
	}

	hir, lir := HIRKinds(), LIRKinds()
	if len(hir)+len(lir) != len(all) {
		t.Fatal("HIRKinds + LIRKinds must partition AllKinds")
	}
	for _, k := range hir {
		if k.Tier() != TierHIR {
			t.Errorf("%s listed as HIR but reports tier %v", k, k.Tier())
		}
	}
	for _, k := range lir {
		if k.Tier() != TierLIR {
			t.Errorf("%s listed as LIR but reports tier %v", k, k.Tier())
		}
	}
	// The tier boundary IS the enum boundary: the last HIR kind is
	// KindLambda (KindReturn closes the original HIR suite; the Input-list
	// restructure additions — IterClusterProject/KeyedAttribute/As/Lambda,
	// quick-261019-ktm — extend the HIR tier after it), and the LIR tier
	// begins at KindForLoop.
	if len(hir) == 0 || hir[len(hir)-1] != KindLambda {
		t.Error("the HIR tier must end at KindLambda (enum boundary before KindForLoop is wrong)")
	}
	if len(lir) == 0 || lir[0] != KindForLoop {
		t.Error("the LIR tier must begin at KindForLoop")
	}
	for k := KindMap; k < KindForLoop; k++ {
		if k.Tier() != TierHIR {
			t.Errorf("kind %s before the LIR boundary must report TierHIR", k)
		}
	}
}

// TestCatalogArityPinsVariadicKinds pins the arity metadata every
// constructor relies on: fixed-arity kinds reject surplus operands through
// the verifier's fail-closed surplus check, variadic kinds wire them all.
func TestCatalogArityPinsVariadicKinds(t *testing.T) {
	g := NewGraph("arity")
	c := g.NewConst("ctor", InternScalar(TF32), 0)
	for _, k := range AllKinds() {
		n := g.NewNode(k, "ctor", c, c, c, c)
		arity := fixedArity(k)
		if arity >= 0 && n.BasePtr().surplus == 0 {
			t.Errorf("fixed-arity kind %s must record surplus operands (arity %d)", k, arity)
		}
		if arity < 0 && n.BasePtr().surplus != 0 {
			t.Errorf("variadic kind %s must wire every operand, surplus %d", k, n.BasePtr().surplus)
		}
	}
}

// TestCatalogFamilyCoverage pins the family taxonomy for the v3 suite.
func TestCatalogFamilyCoverage(t *testing.T) {
	want := map[Kind]Family{
		KindMap:           FamilyAbstract,
		KindReduce:        FamilyAbstract,
		KindMemLoad:       FamilyMemory,
		KindMemStore:      FamilyMemory,
		KindMemNew:        FamilyMemory,
		KindIter:          FamilyLoop,
		KindIterCluster:   FamilyLoop,
		KindConst:         FamilyScalarExpr,
		KindParam:         FamilyScalarExpr,
		KindBinOp:         FamilyScalarExpr,
		KindUnaryOp:       FamilyScalarExpr,
		KindSelect:        FamilyScalarExpr,
		KindCast:          FamilyScalarExpr,
		KindCall:          FamilyScalarExpr,
		KindInputParam:    FamilyAbstract,
		KindReturn:        FamilyAbstract,
		KindForLoop:       FamilyLoop,
		KindDimVar:        FamilyAbstract,
		KindBoundedDimVar: FamilyAbstract,
		KindMemLoadSMem:   FamilyMemory,
		KindMemStoreSMem:  FamilyMemory,
		KindMemNewSMem:    FamilyMemory,
		KindFragLoad:      FamilyMemory,
		KindFragStore:     FamilyMemory,
		KindFragNew:       FamilyMemory,
	}
	for k, f := range want {
		if got := k.Family(); got != f {
			t.Errorf("Kind(%s).Family() = %v, want %v", k, got, f)
		}
	}
}
