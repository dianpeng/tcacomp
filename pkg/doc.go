// Package pkg is the public TCA graph-builder API.
//
// Design note (Phase 01.1): ir.Type is now an interface with NumericType as
// the concrete numeric component. pkg consumes types through one unexported
// helper — numericOf(t ir.Type) *ir.NumericType — used at every elem/dims
// access site instead of scattered inline assertions. Interning is preserved
// (shared *NumericType pointers behind the interface), so the builder's
// identity comparisons (x.typ != y.typ) remain plain interface == and need
// no Equal calls on the hot path.
package pkg
