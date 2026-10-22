package pkg

import (
	"fmt"

	"tcacomp/internal/ir"
)

// This file is the expr-builder DSL (D-09): typed scalar-expression handles
// composed bottom-up, mirroring doc/syntax.md §6.2 operators 1:1 (same names,
// same operand typing rules) and re-rendering through the canonical printer.
//
// Pinned receiver shapes: binary/unary operators and literals are methods on
// *Builder because they consult the builder's graph and type context;
// Select, Cast, and Call are package-level functions because they are
// self-contained (they adopt the builder shared by their operands).
//
// Operators never panic and never return errors: a type violation yields a
// poisoned Expr carrying the error, which the next chain rung (Map/Reduce/
// Store) or Build surfaces (D-10: errors, never panics — library-code rule).

// Expr is a typed scalar-expression handle (D-09). It carries its builder,
// its IR node, and its interned type, so the Go type system plus the recorded
// type catch misuse at the call site instead of in a later pass.
//
// A scalar-param handle (ScalarParam) is LAZY: it carries the resolved
// InputParam node and materializes its paramRef node on first use.
// Declaration order must never consume graph IDs ahead of the
// statement-ID reservation in Stmt — the doc-pinned oracle print expects
// `statement #1` even though `scale` is declared before Stmt. Each use
// materializes one paramRef (SSA-duplicated references; dedup is a later
// pass's job); the use-site order is deterministic, so prints stay
// byte-stable.
type Expr struct {
	b      *Builder
	node   ir.Node // eager node (operators, literals, lambda params)
	scalar ir.Node // lazy: resolved RoleScalar InputParam node; nil for eager exprs
	typ    ir.Type
	err    error // first deferred type error; surfaced by the chain rungs
}

// materialize returns e's IR node, creating the paramRef for a lazy scalar
// handle. Poisoned exprs return nil; operators return early on err.
func (e Expr) materialize() ir.Node {
	if e.node != nil {
		return e.node
	}
	if e.scalar == nil || e.b == nil {
		return nil
	}
	return e.b.g.NewParamRef(origin, e.scalar)
}

// ok reports whether e is error-free.
func (e Expr) ok() bool { return e.err == nil }

// bad returns a poisoned Expr carrying err, bound to b.
func (b *Builder) bad(err error) Expr { return Expr{b: b, err: err} }

// firstErr returns the first deferred error among exprs, if any.
func firstErr(exprs ...Expr) error {
	for _, e := range exprs {
		if e.err != nil {
			return e.err
		}
	}
	return nil
}

// numericOf asserts the numeric type component behind an ir.Type (directive 1
// assertion pattern). A nil Type or any future non-numeric component yields
// nil; every elem/dim access in this package routes through this one helper
// so no inline assertions scatter across the rungs.
func numericOf(t ir.Type) *ir.NumericType {
	nt, _ := t.(*ir.NumericType)
	return nt
}

// typeName renders an interned type for diagnostics.
func typeName(t ir.Type) string {
	if t == nil {
		return "<nil>"
	}
	nt := numericOf(t)
	if nt == nil {
		return "?"
	}
	if nt.IsScalar() {
		return nt.Elem().String()
	}
	return "tensor<" + nt.Elem().String() + ">"
}

func isIntElem(e ir.ElemType) bool {
	switch e {
	case ir.TI8, ir.TI16, ir.TI32, ir.TU8, ir.TU16, ir.TU32:
		return true
	}
	return false
}

func isFloatElem(e ir.ElemType) bool {
	switch e {
	case ir.TF32, ir.TF16, ir.TBF16:
		return true
	}
	return false
}

// Operand type classes per syntax.md §6.2. All Exprs are scalar by
// construction; the scalar check is defensive against future handle leaks.
func isIntType(t ir.Type) bool {
	nt := numericOf(t)
	return nt != nil && nt.IsScalar() && isIntElem(nt.Elem())
}

func isFloatType(t ir.Type) bool {
	nt := numericOf(t)
	return nt != nil && nt.IsScalar() && isFloatElem(nt.Elem())
}

func isArithType(t ir.Type) bool { return isIntType(t) || isFloatType(t) }

func isArithElem(e ir.ElemType) bool { return isIntElem(e) || isFloatElem(e) }

func isBoolType(t ir.Type) bool {
	nt := numericOf(t)
	return nt != nil && nt.IsScalar() && nt.Elem() == ir.TBool
}

// Operand class names for diagnostics.
const (
	classArith = "arithmetic"
	classInt   = "integer"
	classBool  = "bool"
	classCmp   = "comparable"
)

// typedBinOp is the shared bottom for every binary operator: operand errors
// propagate, operands must belong to this builder and share one type, the
// type must satisfy the operator's class, and comparisons yield bool.
func (b *Builder) typedBinOp(op ir.BinOpCode, x, y Expr, class string) Expr {
	if err := firstErr(x, y); err != nil {
		return b.bad(err)
	}
	if x.b != b || y.b != b {
		return b.bad(fmt.Errorf("%s: operands come from a different builder", op))
	}
	if x.typ != y.typ {
		return b.bad(fmt.Errorf("%s: operand types differ (%s vs %s); implicit promotions are not supported (syntax.md §6.2)",
			op, typeName(x.typ), typeName(y.typ)))
	}
	var result ir.Type
	switch class {
	case classArith:
		if !isArithType(x.typ) {
			return b.bad(fmt.Errorf("%s: operands must have the same arithmetic type, got %s", op, typeName(x.typ)))
		}
		result = x.typ
	case classInt:
		if !isIntType(x.typ) {
			return b.bad(fmt.Errorf("%s: operands must have the same integer type, got %s", op, typeName(x.typ)))
		}
		result = x.typ
	case classBool:
		if !isBoolType(x.typ) {
			return b.bad(fmt.Errorf("%s: operands must be bool, got %s", op, typeName(x.typ)))
		}
		result = x.typ
	case classCmp:
		// Comparisons require compatible same-type operands (checked above);
		// they are defined for every supported scalar type and return bool.
		result = ir.InternScalar(ir.TBool)
	default:
		return b.bad(fmt.Errorf("%s: unknown operand class %q", op, class))
	}
	return Expr{b: b, node: b.g.NewBinOp(origin, op, x.materialize(), y.materialize()), typ: result}
}

// Binary arithmetic, same-type operands, result of that type (§6.2).
func (b *Builder) Add(x, y Expr) Expr { return b.typedBinOp(ir.BinOpAdd, x, y, classArith) }
func (b *Builder) Sub(x, y Expr) Expr { return b.typedBinOp(ir.BinOpSub, x, y, classArith) }
func (b *Builder) Mul(x, y Expr) Expr { return b.typedBinOp(ir.BinOpMul, x, y, classArith) }
func (b *Builder) Div(x, y Expr) Expr { return b.typedBinOp(ir.BinOpDiv, x, y, classArith) }

// Mod requires integer operands (§6.2: `%` is in the integer group).
func (b *Builder) Mod(x, y Expr) Expr { return b.typedBinOp(ir.BinOpMod, x, y, classInt) }

// Comparisons require same-type operands and return bool (§6.2).
func (b *Builder) Eq(x, y Expr) Expr { return b.typedBinOp(ir.BinOpEq, x, y, classCmp) }
func (b *Builder) Ne(x, y Expr) Expr { return b.typedBinOp(ir.BinOpNE, x, y, classCmp) }
func (b *Builder) Lt(x, y Expr) Expr { return b.typedBinOp(ir.BinOpLT, x, y, classCmp) }
func (b *Builder) Le(x, y Expr) Expr { return b.typedBinOp(ir.BinOpLE, x, y, classCmp) }
func (b *Builder) Gt(x, y Expr) Expr { return b.typedBinOp(ir.BinOpGT, x, y, classCmp) }
func (b *Builder) Ge(x, y Expr) Expr { return b.typedBinOp(ir.BinOpGE, x, y, classCmp) }

// And/Or require Boolean operands (§6.2).
func (b *Builder) And(x, y Expr) Expr { return b.typedBinOp(ir.BinOpLAnd, x, y, classBool) }
func (b *Builder) Or(x, y Expr) Expr  { return b.typedBinOp(ir.BinOpLOr, x, y, classBool) }

// Integer bitwise operators and shifts, same integer type (§6.2).
func (b *Builder) BitAnd(x, y Expr) Expr { return b.typedBinOp(ir.BinOpAnd, x, y, classInt) }
func (b *Builder) BitOr(x, y Expr) Expr  { return b.typedBinOp(ir.BinOpOr, x, y, classInt) }
func (b *Builder) BitXor(x, y Expr) Expr { return b.typedBinOp(ir.BinOpXor, x, y, classInt) }
func (b *Builder) Shl(x, y Expr) Expr    { return b.typedBinOp(ir.BinOpShL, x, y, classInt) }
func (b *Builder) Shr(x, y Expr) Expr    { return b.typedBinOp(ir.BinOpShR, x, y, classInt) }

// typedUnary is the shared bottom for the unary operators.
func (b *Builder) typedUnary(op ir.UnaryOpCode, x Expr, class string) Expr {
	if err := firstErr(x); err != nil {
		return b.bad(err)
	}
	if x.b != b {
		return b.bad(fmt.Errorf("%s: operand comes from a different builder", op))
	}
	switch class {
	case classArith:
		if !isArithType(x.typ) {
			return b.bad(fmt.Errorf("%s: operand must have an arithmetic type, got %s", op, typeName(x.typ)))
		}
	case classInt:
		if !isIntType(x.typ) {
			return b.bad(fmt.Errorf("%s: operand must have an integer type, got %s", op, typeName(x.typ)))
		}
	case classBool:
		if !isBoolType(x.typ) {
			return b.bad(fmt.Errorf("%s: operand must be bool, got %s", op, typeName(x.typ)))
		}
	default:
		return b.bad(fmt.Errorf("%s: unknown operand class %q", op, class))
	}
	return Expr{b: b, node: b.g.NewUnaryOp(origin, op, x.materialize()), typ: x.typ}
}

// Neg/Pos are arithmetic; Not is Boolean; BitsNot is integer (§6.2/§7.6).
func (b *Builder) Neg(x Expr) Expr     { return b.typedUnary(ir.UnaryOpNeg, x, classArith) }
func (b *Builder) Pos(x Expr) Expr     { return b.typedUnary(ir.UnaryOpPos, x, classArith) }
func (b *Builder) Not(x Expr) Expr     { return b.typedUnary(ir.UnaryOpLNot, x, classBool) }
func (b *Builder) BitsNot(x Expr) Expr { return b.typedUnary(ir.UnaryOpNot, x, classInt) }

// Scalar literals (§6.3): decimal integers are i32, trailing-u integers are
// u32, floating literals are f32, true/false are bool. Unary minus is a
// separate operator (Neg), never part of a literal.
func (b *Builder) Float32(v float32) Expr {
	t := ir.InternScalar(ir.TF32)
	return Expr{b: b, node: b.g.NewConst(origin, t, float64(v)), typ: t}
}

func (b *Builder) Int32(v int32) Expr {
	t := ir.InternScalar(ir.TI32)
	return Expr{b: b, node: b.g.NewConst(origin, t, float64(v)), typ: t}
}

func (b *Builder) UInt32(v uint32) Expr {
	t := ir.InternScalar(ir.TU32)
	return Expr{b: b, node: b.g.NewConst(origin, t, float64(v)), typ: t}
}

func (b *Builder) Bool(v bool) Expr {
	t := ir.InternScalar(ir.TBool)
	f := float64(0)
	if v {
		f = 1
	}
	return Expr{b: b, node: b.g.NewConst(origin, t, f), typ: t}
}

// Select is the lazy conditional cond ? a : b (§6.2): the condition must be
// Boolean and the branches identically typed. It is a package-level function
// because it is self-contained — it adopts the builder shared by its
// operands. The lazy select preserves C++-style conditional evaluation: an
// unselected branch is not semantically evaluated (§6.2).
func Select(cond, a, b Expr) Expr {
	if err := firstErr(cond, a, b); err != nil {
		return cond.b.bad(err)
	}
	if cond.b != a.b || a.b != b.b {
		return a.b.bad(fmt.Errorf("select: operands come from different builders"))
	}
	if !isBoolType(cond.typ) {
		return a.b.bad(fmt.Errorf("select: condition must be bool, got %s", typeName(cond.typ)))
	}
	if a.typ != b.typ {
		return a.b.bad(fmt.Errorf("select: branches must have identical types, got %s vs %s",
			typeName(a.typ), typeName(b.typ)))
	}
	return Expr{b: a.b, node: a.b.g.NewSelect(origin, cond.materialize(), a.materialize(), b.materialize()), typ: a.typ}
}

// Cast is the explicit conversion to a named scalar type (§6.2: explicit
// only, registry-checked pairs). The v0 cast registry lives here in the
// builder DSL until the shared frontend/IR registry lands with the parser;
// it admits arithmetic-to-arithmetic pairs only — no bool involvement, no
// redundant identity casts.
func Cast(e Expr, t ir.ElemType) Expr {
	if err := firstErr(e); err != nil {
		return e.b.bad(err)
	}
	target := ir.InternScalar(t)
	if !isArithType(e.typ) || !isArithElem(t) {
		return e.b.bad(fmt.Errorf("cast: pair %s -> %s is not in the v0 cast registry (arithmetic-to-arithmetic only)",
			typeName(e.typ), t))
	}
	if nt := numericOf(e.typ); nt != nil && nt.Elem() == t {
		return e.b.bad(fmt.Errorf("cast: redundant cast to %s", t))
	}
	return Expr{b: e.b, node: e.b.g.NewCast(origin, target, e.materialize()), typ: target}
}

// Call resolves a registered pure builtin against its pinned signature
// (§6.4): the name must be registered (the signature table lives in
// internal/ir's Builtin registry — the single source of truth shared with
// the type-inference pass) and every argument must be f32. It is a
// package-level function because it is self-contained — it adopts the
// builder shared by its arguments.
func Call(name string, args ...Expr) (Expr, error) {
	sig, ok := ir.Builtin(name)
	if !ok {
		return Expr{}, fmt.Errorf("function %s is not a registered pure scalar builtin", name)
	}
	if len(args) != sig.Arity {
		return Expr{}, fmt.Errorf("builtin %s expects %d argument(s) of type f32, got %d", name, sig.Arity, len(args))
	}
	b := args[0].b
	f32 := ir.InternScalar(ir.TF32)
	nodes := make([]ir.Node, len(args))
	for i, a := range args {
		if a.err != nil {
			return Expr{}, a.err
		}
		if a.b != b {
			return Expr{}, fmt.Errorf("call %s: argument %d comes from a different builder", name, i)
		}
		if a.typ != f32 {
			return Expr{}, fmt.Errorf("call %s: argument %d must be f32, got %s", name, i, typeName(a.typ))
		}
		nodes[i] = a.materialize()
	}
	return Expr{b: b, node: b.g.NewCall(origin, name, nodes...), typ: f32}, nil
}
