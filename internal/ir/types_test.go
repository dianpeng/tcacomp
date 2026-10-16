package ir

import "testing"

// stubType is a second, test-only Type implementation: it proves that
// NumericType.Equal compares across components without panicking and without
// special-casing any particular implementation.
type stubType struct{}

func (stubType) Equal(other Type) bool { return false }

// TestTypeInterningIdentity pins directive-1's interning contract under the
// Type interface: interface == IS identity (the dynamic type *NumericType is
// a comparable pointer and interns are deduped), so every pointer-equality
// site keeps working unchanged on the interface.
func TestTypeInterningIdentity(t *testing.T) {
	dims := []DimRef{{Name: "m"}, {Name: "k"}}
	a := InternTensor(TF32, dims)
	b := InternTensor(TF32, []DimRef{{Name: "m"}, {Name: "k"}})
	if a != b {
		t.Fatal("interned tensors with equal content must be interface-equal (==)")
	}
	// The interner must copy the caller's slice: mutating it afterwards must
	// not corrupt the interned type.
	dims[0] = DimRef{Name: "z"}
	if a != InternTensor(TF32, []DimRef{{Name: "m"}, {Name: "k"}}) {
		t.Fatal("interned type must not alias the caller's dims slice")
	}
	if InternScalar(TF32) == a {
		t.Fatal("scalar and tensor interns must be distinct")
	}
	if InternTensor(TF32, nil) != InternScalar(TF32) {
		t.Fatal("nil dims must intern as the scalar")
	}
	if InternTensor(TI32, []DimRef{{Name: "m"}}) == InternTensor(TF32, []DimRef{{Name: "m"}}) {
		t.Fatal("differing element types must intern differently")
	}
}

// TestNumericTypeEqual pins Equal as the cross-component comparison point:
// content compare numeric-vs-numeric, false for nil and for any other Type
// implementation — with no panic and no type special-casing.
func TestNumericTypeEqual(t *testing.T) {
	a := InternTensor(TF32, []DimRef{{Name: "m"}})
	if !a.Equal(InternTensor(TF32, []DimRef{{Name: "m"}})) {
		t.Fatal("content-equal numerics must be Equal")
	}
	if a.Equal(InternTensor(TF32, []DimRef{{Name: "k"}})) {
		t.Fatal("differing dims must not be Equal")
	}
	if a.Equal(InternScalar(TF32)) {
		t.Fatal("tensor vs scalar must not be Equal")
	}
	var nilType Type
	if a.Equal(nilType) {
		t.Fatal("nil Type must not be Equal")
	}
	if a.Equal(stubType{}) {
		t.Fatal("a different Type implementation must not be Equal (and must not panic)")
	}
}

// TestNumericTypeAccessors pins the type-assertion access pattern: consumers
// needing element/dim details assert *NumericType; Dims returns a copy.
func TestNumericTypeAccessors(t *testing.T) {
	s := InternScalar(TI32)
	ns, ok := s.(*NumericType)
	if !ok {
		t.Fatal("InternScalar must return a *NumericType")
	}
	if ns.Elem() != TI32 || !ns.IsScalar() || ns.Rank() != 0 {
		t.Fatalf("scalar accessors wrong: elem=%v scalar=%v rank=%d", ns.Elem(), ns.IsScalar(), ns.Rank())
	}

	tens := InternTensor(TF32, []DimRef{{Name: "m"}, {Extent: 4}})
	nt, ok := tens.(*NumericType)
	if !ok {
		t.Fatal("InternTensor must return a *NumericType")
	}
	if nt.IsScalar() || nt.Rank() != 2 {
		t.Fatalf("tensor accessors wrong: scalar=%v rank=%d", nt.IsScalar(), nt.Rank())
	}
	if got := nt.Dims(); len(got) != 2 || got[0].Name != "m" || got[1].Extent != 4 {
		t.Fatalf("Dims = %v, want [{@m} {4}]", got)
	}
	d := nt.Dims()
	d[0] = DimRef{Name: "corrupted"}
	if nt.Dims()[0].Name != "m" {
		t.Fatal("Dims must return a copy; mutating it must not affect the type")
	}
}
