package ir

import "testing"

// builtins_test.go pins the ir-level pure-builtin signature registry
// (quick-261019-tu3 Task 1): the single source of truth moved from
// pkg/expr.go's pureBuiltins table, consulted by pkg.Call and the
// typeinfer pass alike. Every entry is (f32, ...f32) -> f32 per
// syntax.md §6.4.

// TestBuiltinTable pins the exact name/arity set (the table previously in
// pkg/expr.go) and the uniform f32 signature.
func TestBuiltinTable(t *testing.T) {
	want := map[string]int{
		"expf": 1, "logf": 1, "sqrtf": 1, "rsqrtf": 1,
		"fabsf": 1, "tanhf": 1, "sinf": 1, "cosf": 1,
		"fminf": 2, "fmaxf": 2,
		"fmaf": 3,
	}
	got := map[string]int{}
	for name := range want {
		sig, ok := Builtin(name)
		if !ok {
			t.Fatalf("Builtin(%q) must be registered", name)
		}
		if sig.Arity != want[name] {
			t.Fatalf("Builtin(%q).Arity = %d, want %d", name, sig.Arity, want[name])
		}
		if sig.Arg != TF32 || sig.Result != TF32 {
			t.Fatalf("Builtin(%q) = (arg %s, result %s), want (f32, f32)", name, sig.Arg, sig.Result)
		}
		got[name] = sig.Arity
	}
	if len(got) != len(want) {
		t.Fatalf("registry holds %d entries, want exactly %d", len(got), len(want))
	}
}

// TestBuiltinSpot pins the doc's named examples and the miss case.
func TestBuiltinSpot(t *testing.T) {
	sig, ok := Builtin("expf")
	if !ok || sig.Arity != 1 || sig.Arg != TF32 || sig.Result != TF32 {
		t.Fatalf("Builtin(expf) = %+v, %v; want arity 1, f32 -> f32", sig, ok)
	}
	if sig, ok := Builtin("fmaf"); !ok || sig.Arity != 3 {
		t.Fatalf("Builtin(fmaf) = %+v, %v; want arity 3", sig, ok)
	}
	if _, ok := Builtin("nosuch"); ok {
		t.Fatal("Builtin(nosuch) must report ok == false")
	}
}
