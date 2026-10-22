// Package cxx is the CUDA C++ backend (doc/cg.md): it consumes the
// post-final-lower graph plus its attached ASTKernel and emits one .cu
// translation unit (device kernel + extern "C" host launcher) and a
// CMakeLists.txt linking the tcacomp runtime library. The emitter is a pure
// consumer — it never mutates the graph — and its output is deterministic:
// every ordering comes from AST item order, param declaration order, or node
// allocation order; no map is ever iterated unsorted.
package cxx

import (
	"fmt"
	"strconv"
	"strings"

	"tcacomp/internal/ir"
)

// elemType maps a scalar ElemType to its C++ spelling (doc/cg.md §5). f16/
// bf16 map to the CUDA half types; their compute-type policy is a runtime
// library decision deferred to that task.
func elemType(e ir.ElemType) (string, error) {
	switch e {
	case ir.TBool:
		return "bool", nil
	case ir.TF32:
		return "float", nil
	case ir.TF16:
		return "__half", nil
	case ir.TBF16:
		return "__nv_bfloat16", nil
	case ir.TI8:
		return "int8_t", nil
	case ir.TI16:
		return "int16_t", nil
	case ir.TI32:
		return "int32_t", nil
	case ir.TU8:
		return "uint8_t", nil
	case ir.TU16:
		return "uint16_t", nil
	case ir.TU32:
		return "uint32_t", nil
	default:
		return "", fmt.Errorf("cxx: unmapped element type %v", e)
	}
}

// scalarType renders a scalar (rank-0) NumericType; tensor types are an
// error at scalar positions (fail closed, never silently demoted).
func scalarType(t ir.Type) (string, error) {
	nt, ok := t.(*ir.NumericType)
	if !ok || nt == nil {
		return "", fmt.Errorf("cxx: expected a numeric type, got %T", t)
	}
	if nt.Rank() != 0 {
		return "", fmt.Errorf("cxx: expected a scalar type, got a rank-%d tensor", nt.Rank())
	}
	return elemType(nt.Elem())
}

// scalarTypeOfParam renders an InputParam's scalar payload type (nil for
// dim params by design — an error here).
func scalarTypeOfParam(p ir.Node) (string, error) {
	t, _ := ir.InputParamTypeOf(p)
	if t == nil {
		return "", fmt.Errorf("cxx: param %s carries no type", ir.InputParamNameOf(p))
	}
	return scalarType(t)
}

// constLiteral renders a typed constant as a C++ literal with its type's
// suffix (doc/cg.md §6). Only the literal-formattable v0 set is accepted;
// half-precision constants have no C++ literal form and fail closed until
// the runtime library pins a conversion idiom.
func constLiteral(n ir.Node) (string, error) {
	t := ir.ConstTypeOf(n)
	if t == nil {
		return "", fmt.Errorf("cxx: const %d carries no declared type", n.BasePtr().ID())
	}
	nt, ok := t.(*ir.NumericType)
	if !ok || nt == nil || nt.Rank() != 0 {
		return "", fmt.Errorf("cxx: const %d is not a typed scalar", n.BasePtr().ID())
	}
	v, ok := ir.ConstValueOf(n)
	if !ok {
		return "", fmt.Errorf("cxx: const %d carries no value", n.BasePtr().ID())
	}
	switch nt.Elem() {
	case ir.TF32:
		lit := strconv.FormatFloat(v, 'g', -1, 32)
		if !strings.ContainsAny(lit, ".eE") {
			lit += ".0" // "0f" is not a C++ literal; "0.0f" is
		}
		return lit + "f", nil
	case ir.TU32:
		return strconv.FormatUint(uint64(v), 10) + "u", nil
	case ir.TI32:
		return strconv.FormatInt(int64(v), 10), nil
	case ir.TBool:
		if v != 0 {
			return "true", nil
		}
		return "false", nil
	default:
		return "", fmt.Errorf("cxx: const %d of type %s has no literal form in v0", n.BasePtr().ID(), nt.Elem())
	}
}

// binOpToken is the C++ operator for a BinOpCode — the identical source
// spelling (syntax.md §7.6 mirrors C++; doc/cg.md §5).
func binOpToken(op ir.BinOpCode) (string, error) {
	switch op {
	case ir.BinOpLOr:
		return "||", nil
	case ir.BinOpLAnd:
		return "&&", nil
	case ir.BinOpOr:
		return "|", nil
	case ir.BinOpXor:
		return "^", nil
	case ir.BinOpAnd:
		return "&", nil
	case ir.BinOpEq:
		return "==", nil
	case ir.BinOpNE:
		return "!=", nil
	case ir.BinOpLT:
		return "<", nil
	case ir.BinOpLE:
		return "<=", nil
	case ir.BinOpGT:
		return ">", nil
	case ir.BinOpGE:
		return ">=", nil
	case ir.BinOpShL:
		return "<<", nil
	case ir.BinOpShR:
		return ">>", nil
	case ir.BinOpAdd:
		return "+", nil
	case ir.BinOpSub:
		return "-", nil
	case ir.BinOpMul:
		return "*", nil
	case ir.BinOpDiv:
		return "/", nil
	case ir.BinOpMod:
		return "%", nil
	default:
		return "", fmt.Errorf("cxx: unmapped binary opcode %d", int(op))
	}
}

// unOpToken is the C++ operator for a UnaryOpCode.
func unOpToken(op ir.UnaryOpCode) (string, error) {
	switch op {
	case ir.UnaryOpPos:
		return "+", nil
	case ir.UnaryOpNeg:
		return "-", nil
	case ir.UnaryOpLNot:
		return "!", nil
	case ir.UnaryOpNot:
		return "~", nil
	default:
		return "", fmt.Errorf("cxx: unmapped unary opcode %d", int(op))
	}
}

// callTarget renders a KindCall builtin name (doc/cg.md §5): the u32 pair is
// runtime-owned (tca::rt), the f32 math set rides the CUDA math library
// verbatim.
func callTarget(name string) (string, error) {
	switch name {
	case "umin", "umax":
		return "tca::rt::" + name, nil
	case "expf", "logf", "sqrtf", "rsqrtf", "fabsf", "tanhf", "sinf", "cosf",
		"fminf", "fmaxf", "fmaf":
		return name, nil
	default:
		return "", fmt.Errorf("cxx: unregistered builtin call %q", name)
	}
}
