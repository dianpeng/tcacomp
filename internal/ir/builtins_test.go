package ir

import "testing"

// builtins_test.go pins the ir-level pure-builtin signature registry
// (quick-261019-tu3 Task 1): the single source of truth moved from
// pkg/expr.go's pureBuiltins table, consulted by pkg.Call and the
// typeinfer pass alike. Every entry is (f32, ...f32) -> f32 per
// syntax-design.md §4.6 rule 20, EXCEPT the u32 min/max pair quick-261021-jay added for
// the tile-lower pass's materialized bounds (doc/optana/loop_tile_lower.md
// §3's integer-bounds recipe).

// TestBuiltinTable pins the exact name/arity set (the table previously in
// pkg/expr.go) and the uniform f32 signature of the f32 family.
func TestBuiltinTable(t *testing.T) {
	want := map[string]int{
		"expf": 1, "logf": 1, "sqrtf": 1, "rsqrtf": 1,
		"fabsf": 1, "tanhf": 1, "sinf": 1, "cosf": 1,
		"fminf": 2, "fmaxf": 2,
		"fmaf": 3,
	}
	for name, arity := range want {
		sig, ok := Builtin(name)
		if !ok {
			t.Fatalf("Builtin(%q) must be registered", name)
		}
		if sig.Arity != arity {
			t.Fatalf("Builtin(%q).Arity = %d, want %d", name, sig.Arity, arity)
		}
		if sig.Arg != TF32 || sig.Result != TF32 {
			t.Fatalf("Builtin(%q) = (arg %s, result %s), want (f32, f32)", name, sig.Arg, sig.Result)
		}
	}
}

// TestBuiltinU32MinMax pins the integer-bounds pair (quick-261021-jay):
// "umin"/"umax" are (u32, u32) -> u32 — the tile-lower pass emits umin for
// every materialized tile_max = umin(T, extent − offset).
func TestBuiltinU32MinMax(t *testing.T) {
	for _, name := range []string{"umin", "umax"} {
		sig, ok := Builtin(name)
		if !ok {
			t.Fatalf("Builtin(%q) must be registered", name)
		}
		if sig.Arity != 2 || sig.Arg != TU32 || sig.Result != TU32 {
			t.Fatalf("Builtin(%q) = %+v, want arity 2, u32 -> u32", name, sig)
		}
	}
}

// TestBuiltinRegistrySize pins the total entry count: 11 f32-family entries
// plus the two u32 bounds builtins — a registry drift tripwire.
func TestBuiltinRegistrySize(t *testing.T) {
	count := 0
	for name := range map[string]struct{}{
		"expf": {}, "logf": {}, "sqrtf": {}, "rsqrtf": {},
		"fabsf": {}, "tanhf": {}, "sinf": {}, "cosf": {},
		"fminf": {}, "fmaxf": {}, "fmaf": {},
		"umin": {}, "umax": {},
	} {
		if _, ok := Builtin(name); ok {
			count++
		}
	}
	if count != 13 {
		t.Fatalf("registry must hold exactly 13 entries, got %d", count)
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
