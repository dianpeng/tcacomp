package typeinfer

import (
	"fmt"

	"tcacomp/internal/ir"
)

// rules.go is the per-kind typing rule switch over the full 31-kind
// catalog, faithful to doc/optana/type_inf.md's rules (1)-(5) and extended
// to the implemented suite (bdimvar joins the dimvar class; the LIR memory
// family, keyedattribute/as/lambda covered; itercluster is the legacy
// transparent wrapper; iterclusterproject fails on sight). The switch is
// CLOSED: an unknown kind errors (T-ti-03 — the catalog is closed, no
// silent nil-typing of unrecognized kinds).
//
// Element-view convention (the spec's "Expr node CAN ONLY work on scalar
// type" meets this IR's shape): a rank-0 numeric operand contributes
// itself, and a TENSOR operand contributes its element type. In expression
// position a tensor denotes its per-point element — the same convention the
// builder embodies (a paramRef to a tensor param IS the structural access,
// print.go) and simplify-inline produces (post-inline lambda bodies route
// whole-tensor stage values where the body means their elements, e.g.
// matmul's `l * r` over lhs/rhs refs and `acc + now` over the piped stage).
// Without the element view, every post-inline builder graph would fail the
// strict rank-0 check at its own lambda bodies.

// u32 is the interned u32 scalar — the loop/IV value type (spec (2)).
func u32() ir.Type { return ir.InternScalar(ir.TU32) }

// nodeErr builds the fail-closed, node-naming error (simplify's style).
func nodeErr(n ir.Node, format string, args ...any) error {
	return fmt.Errorf("typeinfer: node %d (%s): %s", n.BasePtr().ID(), n.KindOf(), fmt.Sprintf(format, args...))
}

// numericOf asserts the numeric component of t (nil for anything else).
func numericOf(t ir.Type) *ir.NumericType {
	nt, _ := t.(*ir.NumericType)
	return nt
}

// operandType returns memo's type for n's operand slot i, failing closed
// on an unwired slot or an operand that carries no computed type (killed or
// untyped defs must not be silently typed).
func operandType(n ir.Node, ops []ir.Node, i int, memo map[*ir.Base]ir.Type) (ir.Type, error) {
	if i >= len(ops) || ops[i] == nil {
		return nil, nodeErr(n, "operand slot %d is unwired", i)
	}
	ty, ok := memo[ops[i].BasePtr()]
	if !ok || ty == nil {
		return nil, nodeErr(n, "operand slot %d (node %d, %s) carries no computed type",
			i, ops[i].BasePtr().ID(), ops[i].KindOf())
	}
	return ty, nil
}

// scalarView returns the scalar type an expression operand contributes
// (see the file comment): a rank-0 numeric as-is, a tensor's element type.
// Nil or non-numeric views error — expr nodes have no other operands.
func scalarView(n ir.Node, slot int, t ir.Type) (ir.Type, error) {
	nt := numericOf(t)
	if nt == nil {
		return nil, nodeErr(n, "operand slot %d has no numeric type (expr nodes require scalar operands)", slot)
	}
	if nt.IsScalar() {
		return t, nil
	}
	return ir.InternScalar(nt.Elem()), nil
}

// exprOperand composes operandType and scalarView for operand slot i.
func exprOperand(n ir.Node, ops []ir.Node, i int, memo map[*ir.Base]ir.Type) (ir.Type, error) {
	t, err := operandType(n, ops, i, memo)
	if err != nil {
		return nil, err
	}
	return scalarView(n, i, t)
}

// Element classes per syntax.md §6.2 (mirroring pkg/expr.go's tables).
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

func isArithElem(e ir.ElemType) bool { return isIntElem(e) || isFloatElem(e) }

// typeOfNode computes n's type from memo — guaranteed populated for every
// live operand of n by the post-order traversal. (nil, nil) marks the
// deliberately-untyped kinds; every other failure is a node-naming error.
func typeOfNode(n ir.Node, memo map[*ir.Base]ir.Type) (ir.Type, error) {
	ops := ir.OperandsOf(n)
	switch n.KindOf() {

	// --- scalar expressions (spec (1)) ---

	case ir.KindConst:
		t := ir.ConstTypeOf(n)
		if nt := numericOf(t); t == nil || nt == nil || !nt.IsScalar() {
			return nil, nodeErr(n, "const carries no scalar payload type")
		}
		return t, nil

	case ir.KindParam:
		// The reference types as its target InputParam's rule result.
		return operandType(n, ops, 0, memo)

	case ir.KindInputParam:
		role, _ := ir.RoleOf(n)
		if role == ir.RoleDim {
			// tca.md §4.1: a runtime dimension is a runtime VALUE — the
			// payload stays nil by design; u32 is the COMPUTED type.
			return u32(), nil
		}
		t, _ := ir.InputParamTypeOf(n)
		if t == nil {
			return nil, nodeErr(n, "input parameter carries no user type (the parameter is typed by the user, otherwise this is an error)")
		}
		return t, nil

	case ir.KindBinOp:
		return binOpType(n, ops, memo)

	case ir.KindUnaryOp:
		return unaryOpType(n, ops, memo)

	case ir.KindSelect:
		cond, err := exprOperand(n, ops, 0, memo)
		if err != nil {
			return nil, err
		}
		if nt := numericOf(cond); nt == nil || nt.Elem() != ir.TBool {
			return nil, nodeErr(n, "condition must be bool, got %v", cond)
		}
		a, err := exprOperand(n, ops, 1, memo)
		if err != nil {
			return nil, err
		}
		b, err := exprOperand(n, ops, 2, memo)
		if err != nil {
			return nil, err
		}
		if !a.Equal(b) {
			return nil, nodeErr(n, "branches have differing types (%v vs %v)", a, b)
		}
		return a, nil

	case ir.KindCast:
		t := ir.CastTypeOf(n)
		if nt := numericOf(t); t == nil || nt == nil || !nt.IsScalar() {
			return nil, nodeErr(n, "cast carries no scalar payload type")
		}
		if _, err := exprOperand(n, ops, 0, memo); err != nil {
			return nil, err
		}
		return t, nil

	case ir.KindCall:
		name := ir.CallNameOf(n)
		sig, ok := ir.Builtin(name)
		if !ok {
			return nil, nodeErr(n, "call to %q: not a registered pure scalar builtin", name)
		}
		if len(ops) != sig.Arity {
			return nil, nodeErr(n, "builtin %s expects %d argument(s) of type %s, got %d",
				name, sig.Arity, sig.Arg, len(ops))
		}
		want := ir.InternScalar(sig.Arg)
		for i := range ops {
			arg, err := exprOperand(n, ops, i, memo)
			if err != nil {
				return nil, err
			}
			if !arg.Equal(want) {
				return nil, nodeErr(n, "builtin %s argument %d must be %s", name, i, sig.Arg)
			}
		}
		return ir.InternScalar(sig.Result), nil

	// --- loop / IV value kinds (spec (2)) ---

	case ir.KindIter, ir.KindForLoop, ir.KindDimVar, ir.KindBoundedDimVar,
		ir.KindInductionVariable, ir.KindTensorDimOf:
		return u32(), nil

	// --- iteration clusters (spec (2): lowered before TI) ---

	case ir.KindIterCluster:
		inputs := ir.ClusterInputs(n)
		if len(inputs) != 1 || inputs[0] == nil {
			return nil, nodeErr(n, "cluster carries %d inputs: the IterCluster format must be lowered before typeinfer", len(inputs))
		}
		ty, ok := memo[inputs[0].BasePtr()]
		if !ok || ty == nil {
			return nil, nodeErr(n, "cluster input node %d carries no computed type", inputs[0].BasePtr().ID())
		}
		return ty, nil // transparent for piping (print.go's wrapper semantics)

	case ir.KindIterClusterProject:
		return nil, nodeErr(n, "live iterclusterproject: the IterCluster format must be lowered before typeinfer")

	// --- lambda stages (spec (3)) ---

	case ir.KindMap:
		return mapType(n, memo)

	case ir.KindReduce:
		return reduceType(n, ops, memo)

	// --- memory (spec (4): HIR and LIR alike) ---

	case ir.KindMemLoad:
		return memLoadType(n, ops, memo)

	case ir.KindMemStore, ir.KindMemStoreSMem, ir.KindFragStore:
		// The store returns the stored value's type (doc §2.5). No
		// cross-check against the target: post-lowering axes are IVs, so
		// the stored shape is not statically derivable.
		return operandType(n, ops, 0, memo)

	case ir.KindMemLoadSMem, ir.KindFragLoad:
		// Same partial/full deref shape as memload — but only once the
		// buffer carries a tensor type. Today's buffers (memnewsmem/
		// fragnew) are untyped allocations, so the rule fails closed
		// ("same, regardless whether HIR or LIR": the LIR rule exists and
		// the future cache-lowering pass satisfies it by typing buffers).
		if len(ops) == 0 || ops[0] == nil {
			return nil, nodeErr(n, "operand slot 0 is unwired")
		}
		buf := ops[0]
		ty, ok := memo[buf.BasePtr()]
		nt := numericOf(ty)
		if !ok || ty == nil || nt == nil {
			return nil, nodeErr(n, "buffer node %d (%s) carries no dtype; cache lowering must bind buffer types first",
				buf.BasePtr().ID(), buf.KindOf())
		}
		return derefType(n, ops, nt, memo)

	case ir.KindMemNew, ir.KindMemNewSMem, ir.KindFragNew:
		return nil, nil // untyped allocations (the closed untyped set)

	// --- structural wrappers ---

	case ir.KindAs:
		return nil, nil // an output-order clause, not a value

	case ir.KindLambda:
		return nil, nil // a function expression; no function Type exists yet

	case ir.KindKeyedAttribute:
		// Unwraps to its value at binding/print time (D9).
		return operandType(n, ops, 0, memo)

	case ir.KindReturn:
		if len(ops) == 1 {
			ty, err := operandType(n, ops, 0, memo)
			if err != nil {
				return nil, err
			}
			return ty, nil
		}
		return nil, nil // 0 or multi-valued: the tuple-Type future extension

	default:
		return nil, nodeErr(n, "unknown node kind (the catalog is closed)")
	}
}

// binOpType types a binop per syntax.md §6.2's classes, at element view.
func binOpType(n ir.Node, ops []ir.Node, memo map[*ir.Base]ir.Type) (ir.Type, error) {
	op, ok := ir.BinOpCodeOf(n)
	if !ok {
		return nil, nodeErr(n, "binop carries no opcode")
	}
	x, err := exprOperand(n, ops, 0, memo)
	if err != nil {
		return nil, err
	}
	y, err := exprOperand(n, ops, 1, memo)
	if err != nil {
		return nil, err
	}
	xe, ye := numericOf(x).Elem(), numericOf(y).Elem()
	classErr := func(want string) error {
		return nodeErr(n, "%s: operand types differ or miss the %s class (%s vs %s)", op, want, x, y)
	}
	switch op {
	case ir.BinOpAdd, ir.BinOpSub, ir.BinOpMul, ir.BinOpDiv:
		if x.Equal(y) && isArithElem(xe) {
			return x, nil
		}
		return nil, classErr("arithmetic")
	case ir.BinOpMod, ir.BinOpShL, ir.BinOpShR, ir.BinOpOr, ir.BinOpXor, ir.BinOpAnd:
		if x.Equal(y) && isIntElem(xe) {
			return x, nil
		}
		return nil, classErr("integer")
	case ir.BinOpEq, ir.BinOpNE, ir.BinOpLT, ir.BinOpLE, ir.BinOpGT, ir.BinOpGE:
		if !x.Equal(y) {
			return nil, classErr("same-type comparable")
		}
		return ir.InternScalar(ir.TBool), nil
	case ir.BinOpLOr, ir.BinOpLAnd:
		if xe == ir.TBool && ye == ir.TBool {
			return ir.InternScalar(ir.TBool), nil
		}
		return nil, classErr("bool")
	default:
		return nil, nodeErr(n, "unknown binop opcode %v", op)
	}
}

// unaryOpType types a unaryop per syntax.md §6.2's classes.
func unaryOpType(n ir.Node, ops []ir.Node, memo map[*ir.Base]ir.Type) (ir.Type, error) {
	op, ok := ir.UnaryOpCodeOf(n)
	if !ok {
		return nil, nodeErr(n, "unaryop carries no opcode")
	}
	x, err := exprOperand(n, ops, 0, memo)
	if err != nil {
		return nil, err
	}
	e := numericOf(x).Elem()
	switch op {
	case ir.UnaryOpPos, ir.UnaryOpNeg:
		if isArithElem(e) {
			return x, nil
		}
		return nil, nodeErr(n, "%s: operand must have an arithmetic type, got %v", op, x)
	case ir.UnaryOpLNot:
		if e == ir.TBool {
			return x, nil
		}
		return nil, nodeErr(n, "%s: operand must be bool, got %v", op, x)
	case ir.UnaryOpNot:
		if isIntElem(e) {
			return x, nil
		}
		return nil, nodeErr(n, "%s: operand must have an integer type, got %v", op, x)
	default:
		return nil, nodeErr(n, "unknown unaryop opcode %v", op)
	}
}

// valueBindings computes a stage's Input-list VALUE bindings per D9 (the
// same rule simplify-inline binds by): operands in slot order, the trailing
// As and lambda nodes dropped, KeyedAttribute elements unwrapped to their
// wrapped value operand.
func valueBindings(n ir.Node) []ir.Node {
	ops := ir.OperandsOf(n)
	if len(ops) > 0 && ops[len(ops)-1] != nil && ops[len(ops)-1].KindOf() == ir.KindLambda {
		ops = ops[:len(ops)-1]
	}
	if len(ops) > 0 && ops[len(ops)-1] != nil && ops[len(ops)-1].KindOf() == ir.KindAs {
		ops = ops[:len(ops)-1]
	}
	bindings := make([]ir.Node, len(ops))
	for i, op := range ops {
		if op == nil {
			continue
		}
		if op.KindOf() == ir.KindKeyedAttribute {
			if inner := ir.OperandsOf(op); len(inner) > 0 {
				bindings[i] = inner[0]
			}
			continue
		}
		bindings[i] = op
	}
	return bindings
}

// mapType types a Map: the result is tensor<bodyElem, As-dims> — rank =
// len(As), dims symbolic by dim-param name (the same DimRef source the
// builder uses, so chained types intern equal; a static extent is not
// recoverable from a dim InputParam node — documented v0 simplification).
// The spec's meet is the reconcile: lambda params paired positionally with
// the value bindings must agree at element type.
func mapType(n ir.Node, memo map[*ir.Base]ir.Type) (ir.Type, error) {
	bodies := ir.LambdaBodies(n)
	if len(bodies) != 1 || bodies[0] == nil {
		return nil, nodeErr(n, "map carries no lambda body")
	}
	bodyTy := memo[bodies[0].BasePtr()]
	bt := numericOf(bodyTy)
	if bodyTy == nil || bt == nil || !bt.IsScalar() {
		return nil, nodeErr(n, "lambda body type %v is not scalar (the map produces one element per point)", bodyTy)
	}
	as := ir.AsOf(n)
	if as == nil {
		return nil, nodeErr(n, "map carries no as-order (tca.md §2.7: As is mandatory)")
	}
	asOps := ir.OperandsOf(as)
	dims := make([]ir.DimRef, 0, len(asOps))
	for i, d := range asOps {
		if d == nil {
			return nil, nodeErr(n, "as operand %d is unwired", i)
		}
		dims = append(dims, ir.DimRef{Name: ir.InputParamNameOf(d)})
	}

	// Reconcile: params vs value bindings, element-level agreement.
	bindings := valueBindings(n)
	var params []ir.Node
	for _, group := range ir.LambdaParams(n) {
		params = append(params, group...)
	}
	for i, p := range params {
		if p == nil {
			return nil, nodeErr(n, "lambda param %d is unwired", i)
		}
		if i >= len(bindings) || bindings[i] == nil {
			return nil, nodeErr(n, "lambda param %d has no value binding (%d bindings)", i, len(bindings))
		}
		pt, ok := memo[p.BasePtr()]
		pnt := numericOf(pt)
		if !ok || pt == nil || pnt == nil {
			return nil, nodeErr(n, "lambda param %d carries no computed type", i)
		}
		bty, ok := memo[bindings[i].BasePtr()]
		bnt := numericOf(bty)
		if !ok || bty == nil || bnt == nil {
			return nil, nodeErr(n, "value binding %d (node %d, %s) carries no numeric type",
				i, bindings[i].BasePtr().ID(), bindings[i].KindOf())
		}
		if pnt.Elem() != bnt.Elem() {
			return nil, nodeErr(n, "lambda param %d has element type %s but its binding carries %s",
				i, pnt.Elem(), bnt.Elem())
		}
	}
	return ir.InternTensor(bt.Elem(), dims), nil
}

// reduceType types a Reduce: the derived type is the piped value's type
// with the last dimension dropped (rank-1 input reduces to the scalar
// element). Homogeneity (tca.md §4.2): the init and every declared combiner
// param must equal the piped element's scalar type. An optional As is
// validated against the derived shape axis by axis.
func reduceType(n ir.Node, ops []ir.Node, memo map[*ir.Base]ir.Type) (ir.Type, error) {
	piped, err := operandType(n, ops, 0, memo)
	if err != nil {
		return nil, err
	}
	pt := numericOf(piped)
	if pt == nil {
		return nil, nodeErr(n, "piped value type %v is not numeric", piped)
	}
	if pt.Rank() < 1 {
		return nil, nodeErr(n, "piped value is scalar: nothing to reduce")
	}
	want := ir.InternScalar(pt.Elem())

	if init := ir.ReduceInitOf(n); init != nil {
		it, ok := memo[init.BasePtr()]
		if !ok || it == nil || !viewEqual(it, want) {
			return nil, nodeErr(n, "init type violates the homogeneous reducer contract (tca.md §4.2): want %s", want)
		}
	}
	pi := 0
	for _, group := range ir.LambdaParams(n) {
		for _, p := range group {
			if p == nil {
				continue
			}
			pty, ok := memo[p.BasePtr()]
			if !ok || pty == nil || !viewEqual(pty, want) {
				return nil, nodeErr(n, "combiner param %d violates the homogeneous reducer contract (tca.md §4.2): want %s", pi, want)
			}
			pi++
		}
	}

	derived := ir.InternTensor(pt.Elem(), pt.Dims()[:pt.Rank()-1])
	if as := ir.AsOf(n); as != nil {
		asOps := ir.OperandsOf(as)
		if len(asOps) != pt.Rank()-1 {
			return nil, nodeErr(n, "as lists %d axes, the reduced type has rank %d", len(asOps), pt.Rank()-1)
		}
		dd := numericOf(derived).Dims()
		for i, d := range asOps {
			name := ""
			if d != nil {
				name = ir.InputParamNameOf(d)
			}
			if dd[i].Name == "" || dd[i].Name != name {
				return nil, nodeErr(n, "as axis %d (@%s) does not match the derived dim", i, name)
			}
		}
	}
	return derived, nil
}

// viewEqual compares t against the scalar want at element view.
func viewEqual(t ir.Type, want ir.Type) bool {
	nt := numericOf(t)
	if nt == nil {
		return false
	}
	if !nt.IsScalar() {
		return ir.InternScalar(nt.Elem()).Equal(want)
	}
	return t.Equal(want)
}

// memLoadType types an HIR GMEM load: operand 0 must be a RoleTensor
// InputParam carrying a numeric payload; the axes decide partial vs full
// deref (derefType, shared with the LIR buffer loads).
func memLoadType(n ir.Node, ops []ir.Node, memo map[*ir.Base]ir.Type) (ir.Type, error) {
	if len(ops) == 0 || ops[0] == nil {
		return nil, nodeErr(n, "operand slot 0 is unwired")
	}
	t := ops[0]
	if role, ok := ir.RoleOf(t); !ok || role != ir.RoleTensor {
		return nil, nodeErr(n, "operand 0 (node %d, %s) is not a tensor parameter", t.BasePtr().ID(), t.KindOf())
	}
	payload, ok := ir.InputParamTypeOf(t)
	nt := numericOf(payload)
	if !ok || payload == nil || nt == nil {
		return nil, nodeErr(n, "tensor parameter node %d carries no type", t.BasePtr().ID())
	}
	return derefType(n, ops, nt, memo)
}

// derefType applies the spec's partial/full deref rule over a rank-r
// numeric tensor/buffer type with k = len(ops)-1 axis operands: k == r is
// the full deref (the element type — the doc's [M,K,N] + 3 indexes case),
// k < r keeps dims[k:] ([M,K,N] + 1 index -> [K,N]), k > r is
// over-dereferenced. Every axis operand must be scalar u32 — post-lowering
// they are inductionvariables, and a RoleDim InputParam axis is u32 by the
// inputparam rule: uniform, no special case.
func derefType(n ir.Node, ops []ir.Node, nt *ir.NumericType, memo map[*ir.Base]ir.Type) (ir.Type, error) {
	r := nt.Rank()
	k := len(ops) - 1
	if k > r {
		return nil, nodeErr(n, "over-dereferenced: %d axes for rank %d", k, r)
	}
	for i := 1; i < len(ops); i++ {
		if ops[i] == nil {
			return nil, nodeErr(n, "axis operand %d is unwired", i-1)
		}
		axTy, ok := memo[ops[i].BasePtr()]
		ax := numericOf(axTy)
		if !ok || axTy == nil || !ax.IsScalar() || ax.Elem() != ir.TU32 {
			return nil, nodeErr(n, "axis operand %d is not scalar u32", i-1)
		}
	}
	if k == r {
		return ir.InternScalar(nt.Elem()), nil
	}
	return ir.InternTensor(nt.Elem(), nt.Dims()[k:]), nil
}
