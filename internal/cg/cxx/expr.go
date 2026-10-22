package cxx

import (
	"fmt"

	"tcacomp/internal/ir"
)

// emitExpr renders n as a C++ expression (doc/cg.md §6). Materialized nodes
// (loads, Combine results, hoisted items) reference their local; consts
// materialize as locals at first use (final-lower already deduped them, so
// one local per distinct constant); everything else renders inline — pure
// value trees nest directly, dimvars read their owning loop's iterator.
func (e *emitter) emitExpr(n ir.Node) (string, error) {
	if n == nil {
		return "", fmt.Errorf("cxx: nil expression operand")
	}
	if name, ok := e.values[n]; ok {
		return e.slottedRead(n, name)
	}
	switch n.KindOf() {
	case ir.KindConst:
		return e.materializeConst(n)
	case ir.KindParam:
		return e.paramRef(n)
	case ir.KindDimVar:
		it, ok := e.iters[n]
		if !ok {
			return "", fmt.Errorf("cxx: dimvar %d has no enclosing loop in scope", n.BasePtr().ID())
		}
		return it + ".Index()", nil
	case ir.KindBoundedDimVar:
		return "", fmt.Errorf("cxx: bdimvar %d is loop-header material, never a body expression", n.BasePtr().ID())
	case ir.KindBinOp:
		return e.emitBinOp(n)
	case ir.KindUnaryOp:
		return e.emitUnaryOp(n)
	case ir.KindSelect:
		ops := ir.OperandsOf(n)
		if len(ops) != 3 {
			return "", fmt.Errorf("cxx: select %d carries %d operands, want 3", n.BasePtr().ID(), len(ops))
		}
		c, err := e.emitExpr(ops[0])
		if err != nil {
			return "", err
		}
		a, err := e.emitExpr(ops[1])
		if err != nil {
			return "", err
		}
		b, err := e.emitExpr(ops[2])
		if err != nil {
			return "", err
		}
		return "(" + c + " ? " + a + " : " + b + ")", nil
	case ir.KindCast:
		ts, err := scalarType(ir.CastTypeOf(n))
		if err != nil {
			return "", fmt.Errorf("cxx: cast %d: %w", n.BasePtr().ID(), err)
		}
		x, err := e.emitExpr(ir.OperandsOf(n)[0])
		if err != nil {
			return "", err
		}
		return "static_cast<" + ts + ">(" + x + ")", nil
	case ir.KindCall:
		return e.emitCall(n)
	case ir.KindTensorDimOf:
		return e.emitTensorDimOf(n)
	case ir.KindRuntimeGetAttribute:
		return e.emitGetAttr(n)
	default:
		return "", fmt.Errorf("cxx: node %d (%s) is not emittable as an expression — it must be materialized at its statement first",
			n.BasePtr().ID(), n.KindOf())
	}
}

// slottedRead adapts a materialized local to its use site: a slotted
// Combine (D7) is consumed per-slot inside the re-iteration as
// local.Get(slot ivs...), the slot iterators resolved from the dim scope in
// the Reduce call's slot-arg order. Every other local reads bare.
func (e *emitter) slottedRead(n ir.Node, name string) (string, error) {
	dims, ok := e.slottedVal[n]
	if !ok {
		return name, nil
	}
	var args []string
	for _, d := range dims {
		it, ok := e.dimLookup(d)
		if !ok {
			return "", fmt.Errorf("cxx: slotted Combine %d used outside the re-iteration of dim %q", n.BasePtr().ID(), ir.InputParamNameOf(d))
		}
		args = append(args, it+".Index()")
	}
	return name + ".Get(" + join(args) + ")", nil
}

// dimLookup resolves a dim InputParam to the innermost in-scope iterator
// over it (the scope stack is searched top-down, so shadowing re-iteration
// dimvars win).
func (e *emitter) dimLookup(dim ir.Node) (string, bool) {
	for i := len(e.dimScope) - 1; i >= 0; i-- {
		if it, ok := e.dimScope[i][dim]; ok {
			return it, true
		}
	}
	return "", false
}

// materializeConst declares a deduped constant's local at first use
// (doc/cg.md §6: tca_cN, one per distinct constant).
func (e *emitter) materializeConst(n ir.Node) (string, error) {
	lit, err := constLiteral(n)
	if err != nil {
		return "", err
	}
	name := fmt.Sprintf("tca_c%d", e.constN)
	e.constN++
	e.line("auto " + name + " = " + lit + ";")
	e.values[n] = name
	return name, nil
}

// paramRef renders a KindParam: a kernel scalar argument by name, a lambda
// parameter by its function-scope local.
func (e *emitter) paramRef(n ir.Node) (string, error) {
	ops := ir.OperandsOf(n)
	if len(ops) != 1 || ops[0] == nil {
		return "", fmt.Errorf("cxx: param %d carries no InputParam operand", n.BasePtr().ID())
	}
	p := ops[0]
	if role, _ := ir.RoleOf(p); role == ir.RoleLambda {
		name, ok := e.lambdaParams[p]
		if !ok {
			return "", fmt.Errorf("cxx: lambda param %s referenced outside its lambda", ir.InputParamNameOf(p))
		}
		return name, nil
	}
	return ir.InputParamNameOf(p), nil
}

func (e *emitter) emitBinOp(n ir.Node) (string, error) {
	op, ok := ir.BinOpCodeOf(n)
	if !ok {
		return "", fmt.Errorf("cxx: node %d is not a binop", n.BasePtr().ID())
	}
	tok, err := binOpToken(op)
	if err != nil {
		return "", err
	}
	ops := ir.OperandsOf(n)
	if len(ops) != 2 {
		return "", fmt.Errorf("cxx: binop %d carries %d operands, want 2", n.BasePtr().ID(), len(ops))
	}
	a, err := e.emitExpr(ops[0])
	if err != nil {
		return "", err
	}
	b, err := e.emitExpr(ops[1])
	if err != nil {
		return "", err
	}
	return "(" + a + " " + tok + " " + b + ")", nil
}

func (e *emitter) emitUnaryOp(n ir.Node) (string, error) {
	ops := ir.OperandsOf(n)
	if len(ops) != 1 {
		return "", fmt.Errorf("cxx: unaryop %d carries %d operands, want 1", n.BasePtr().ID(), len(ops))
	}
	code, ok := ir.UnaryOpCodeOf(n)
	if !ok {
		return "", fmt.Errorf("cxx: node %d is not a unaryop", n.BasePtr().ID())
	}
	tok, err := unOpToken(code)
	if err != nil {
		return "", err
	}
	x, err := e.emitExpr(ops[0])
	if err != nil {
		return "", err
	}
	return "(" + tok + x + ")", nil
}

func (e *emitter) emitCall(n ir.Node) (string, error) {
	name := ir.CallNameOf(n)
	target, err := callTarget(name)
	if err != nil {
		return "", err
	}
	ops := ir.OperandsOf(n)
	var args []string
	for _, a := range ops {
		s, err := e.emitExpr(a)
		if err != nil {
			return "", err
		}
		args = append(args, s)
	}
	return target + "(" + join(args) + ")", nil
}

// emitTensorDimOf renders the shape upper bound: a constant DimRef extent
// inline, a symbolic dim as its kernel argument (the u32 dim value).
func (e *emitter) emitTensorDimOf(n ir.Node) (string, error) {
	ops := ir.OperandsOf(n)
	if len(ops) != 2 || ops[0] == nil {
		return "", fmt.Errorf("cxx: tensordimof %d malformed", n.BasePtr().ID())
	}
	_, dims, err := e.tensorTypeOf(ops[0])
	if err != nil {
		return "", err
	}
	axis, ok := ir.ConstValueOf(ops[1])
	if !ok || axis < 0 || axis >= float64(len(dims)) {
		return "", fmt.Errorf("cxx: tensordimof %d axis is not a const within rank", n.BasePtr().ID())
	}
	d := dims[int(axis)]
	if d.Extent > 0 {
		return fmt.Sprintf("%du", d.Extent), nil
	}
	p := e.dimParam(d.Name)
	if p == nil {
		return "", fmt.Errorf("cxx: tensordimof %d dim %q has no kernel dim parameter", n.BasePtr().ID(), d.Name)
	}
	return ir.InputParamNameOf(p), nil
}

// emitGetAttr renders the pure accessor call: KernelInfo totals and mapping
// index/dimension reads (doc/cg.md §6). GVN already collapsed repeats.
func (e *emitter) emitGetAttr(n ir.Node) (string, error) {
	ops := ir.OperandsOf(n)
	if len(ops) == 0 || ops[0] == nil {
		return "", fmt.Errorf("cxx: getattr %d carries no object", n.BasePtr().ID())
	}
	obj, ok := e.objName[ops[0]]
	if !ok {
		return "", fmt.Errorf("cxx: getattr %d references an unconstructed runtime object %d",
			n.BasePtr().ID(), ops[0].BasePtr().ID())
	}
	attr := ir.RuntimeAttributeOf(n)
	switch attr {
	case "total_cta", "total_thread_per_cta":
		if len(ops) != 1 {
			return "", fmt.Errorf("cxx: getattr %s carries extra operands", attr)
		}
		return obj + "." + attr + "()", nil
	case "index", "dimension":
		if len(ops) != 2 || ops[1] == nil {
			return "", fmt.Errorf("cxx: getattr %s needs exactly one dim operand", attr)
		}
		d, ok := ir.ConstValueOf(ops[1])
		if !ok {
			return "", fmt.Errorf("cxx: getattr %s dim operand is not a constant", attr)
		}
		return fmt.Sprintf("%s.%s(%du)", obj, attr, uint(d)), nil
	default:
		return "", fmt.Errorf("cxx: unknown attribute %q", attr)
	}
}

// join is strings.Join for the hot path with no separator surprises.
func join(parts []string) string {
	out := ""
	for i, p := range parts {
		if i > 0 {
			out += ", "
		}
		out += p
	}
	return out
}
