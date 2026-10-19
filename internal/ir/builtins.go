package ir

// builtins.go is the pure scalar-builtin signature registry — the single
// source of truth shared by the builder DSL (pkg.Call) and the type
// inference pass (internal/opt/typeinfer), moved here from pkg/expr.go's
// local pureBuiltins table (quick-261019-tu3 Task 1) so the pass layer never
// duplicates the table. Every entry is (f32, ...f32) -> f32 with the exact
// arity syntax.md §6.4 pins; anything else — memory access, synchronization,
// atomics, thread-index builtins — is not a scalar-lambda call and is not
// registered. The table is a plain map, never iterated, so it cannot leak
// map-order nondeterminism; it is package-level and populated only here —
// no runtime mutation path exists (a compile-time constant table).

// BuiltinSignature is one registered pure scalar builtin's contract.
type BuiltinSignature struct {
	Arity       int      // exact argument count
	Arg, Result ElemType // argument and result element types (f32 today)
}

// builtins is the registry proper (syntax.md §6.4).
var builtins = map[string]BuiltinSignature{
	"expf": {1, TF32, TF32}, "logf": {1, TF32, TF32},
	"sqrtf": {1, TF32, TF32}, "rsqrtf": {1, TF32, TF32},
	"fabsf": {1, TF32, TF32}, "tanhf": {1, TF32, TF32},
	"sinf": {1, TF32, TF32}, "cosf": {1, TF32, TF32},
	"fminf": {2, TF32, TF32}, "fmaxf": {2, TF32, TF32},
	"fmaf": {3, TF32, TF32},
}

// Builtin returns the registered signature for name, and whether one exists.
func Builtin(name string) (BuiltinSignature, bool) {
	sig, ok := builtins[name]
	return sig, ok
}
