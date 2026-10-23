package cxx

import (
	"fmt"
	"strconv"
	"strings"

	"tcacomp/internal/ir"
)

// stmtRuntimeObject emits a runtime-object construction declaration
// (doc/cg.md §6): the dedicated construction kinds render as tca::rt
// declarations at their scheduled positions — KernelInfo first in the
// kernel body, the per-LNT mapping pair at the LNT prologue, reducers in
// the owning LNT's prologue (final-lower's placements, consumed verbatim).
func (e *emitter) stmtRuntimeObject(n ir.Node) error {
	switch n.KindOf() {
	case ir.KindRuntimeKernelInfo:
		name := e.local("kinfo")
		e.objName[n] = name
		e.line("tca::rt::KernelInfo " + name + ";")
		return nil

	case ir.KindRuntimeGridMapping, ir.KindRuntimeThreadMapping:
		ops := ir.OperandsOf(n)
		if len(ops) < 2 || ops[0] == nil {
			return fmt.Errorf("cxx: mapping %d malformed", n.BasePtr().ID())
		}
		ki, ok := e.objName[ops[0]]
		if !ok {
			return fmt.Errorf("cxx: mapping %d references an unconstructed KernelInfo", n.BasePtr().ID())
		}
		// Portions are TI32 consts (−1 = the remainder tail) rendered as
		// plain integers; ndim is implied by the count (verifier-pinned).
		var ports []string
		for _, p := range ops[2:] {
			v, ok := ir.ConstValueOf(p)
			if !ok {
				return fmt.Errorf("cxx: mapping %d portion is not a constant", n.BasePtr().ID())
			}
			ports = append(ports, strconv.FormatInt(int64(v), 10))
		}
		cls, base := "GridMapping", "grid_map"
		if n.KindOf() == ir.KindRuntimeThreadMapping {
			cls, base = "ThreadMapping", "thread_map"
		}
		local := e.local(fmt.Sprintf("%s%d", base, e.mapN))
		e.mapN++
		if cls == "GridMapping" {
			e.c("CTA-space mapping (portions %s — the launch grammar this nest remaps)", strings.Join(ports, ", "))
		} else {
			e.c("per-CTA thread-space mapping (portions %s)", strings.Join(ports, ", "))
		}
		e.objName[n] = local
		if cls == "GridMapping" {
			e.curGrid = local
		} else {
			e.curThread = local
		}
		e.line(fmt.Sprintf("tca::rt::%s %s(%s, %s);", cls, local, ki, strings.Join(ports, ", ")))
		return nil

	default: // the three reducer specializations
		ops := ir.OperandsOf(n)
		if len(ops) != 2 || ops[0] == nil || ops[1] == nil {
			return fmt.Errorf("cxx: reducer %d malformed (want init, lambda)", n.BasePtr().ID())
		}
		ts, err := e.initType(ops[0])
		if err != nil {
			return err
		}
		init, err := e.emitExpr(ops[0])
		if err != nil {
			return err
		}
		if err := e.registerLambda(ops[1]); err != nil {
			return err
		}
		cls := "SeqReducer"
		switch n.KindOf() {
		case ir.KindRuntimeWarpReducer:
			cls = "WarpReducer"
		case ir.KindRuntimeGlobalReducer:
			cls = "GlobalReducer"
		}
		slots := e.slotCount[n] // the slot tile extent; 0 = slotless (scalar Combine)
		local := e.local(fmt.Sprintf("reducer%d", e.redN))
		e.redN++
		scope := "sequential"
		switch cls {
		case "WarpReducer":
			scope = "warp"
		case "GlobalReducer":
			scope = "global (CTA)"
		}
		e.c("%s reduction — the combiner is runtime-owned; Init/Combine bracket ALL tiles of the reduced dim", scope)
		e.objName[n] = local
		e.line(fmt.Sprintf("tca::rt::%s<%s, %du> %s(%s, %s);", cls, ts, uint(slots), local, init, e.lambdaName(ops[1])))
		return nil
	}
}

// initType resolves a reducer construction's element type T from its init
// operand: a const's declared type, a scalar param's type (the two shapes
// final-lower lowers); the combiner is homogeneous against it (checked at
// Reduce lowering).
func (e *emitter) initType(init ir.Node) (string, error) {
	if t := ir.ConstTypeOf(init); t != nil {
		return scalarType(t)
	}
	if init.KindOf() == ir.KindParam {
		ops := ir.OperandsOf(init)
		if len(ops) == 1 && ops[0] != nil {
			if ts, err := scalarTypeOfParam(ops[0]); err == nil {
				return ts, nil
			}
		}
	}
	return "", fmt.Errorf("cxx: reducer init %d carries no resolvable element type", init.BasePtr().ID())
}

// stmtCallMethod emits the runtime method calls (final_lower.md's lowered
// shape): Init argument-free before loop X, Reduce per step inside the home
// loop (slot IVs outer→inner, then the piped value — LR-13), Combine after
// loop X. The (prev, cur) combiner application is runtime-internal — the
// compiler emits no lambda-call node (final_lower.md 2.3).
func (e *emitter) stmtCallMethod(n ir.Node) error {
	ops := ir.OperandsOf(n)
	if len(ops) == 0 || ops[0] == nil {
		return fmt.Errorf("cxx: callmethod %d carries no object", n.BasePtr().ID())
	}
	obj, ok := e.objName[ops[0]]
	if !ok {
		return fmt.Errorf("cxx: callmethod %d references an unconstructed object %d",
			n.BasePtr().ID(), ops[0].BasePtr().ID())
	}
	args := ops[1:]
	switch ir.RuntimeMethodOf(n) {
	case "Init":
		if len(args) != 0 {
			return fmt.Errorf("cxx: reducer Init %d must be argument-free", n.BasePtr().ID())
		}
		e.c("reduction bracket opens (argument-free; re-initializes from the construction-time init)")
		e.line(obj + ".Init();")
		return nil

	case "Reduce":
		if len(args) == 0 {
			return fmt.Errorf("cxx: reducer Reduce %d carries no value", n.BasePtr().ID())
		}
		// Leading args are the accumulator-tile slot dimvars (LR-13);
		// record them for the D7 slot reads at the Combine use sites.
		var slots []ir.Node
		for _, a := range args[:len(args)-1] {
			d := e.bivDim(a)
			if d == nil {
				return fmt.Errorf("cxx: reducer Reduce %d slot arg is not a dimvar over a dim", n.BasePtr().ID())
			}
			slots = append(slots, d)
		}
		if prev, ok := e.slots[ops[0]]; ok && len(prev) != len(slots) {
			return fmt.Errorf("cxx: reducer %s slot arity changed between Reduce calls (%d → %d)",
				obj, len(prev), len(slots))
		}
		e.slots[ops[0]] = slots
		var exprs []string
		for _, a := range args {
			s, err := e.emitExpr(a)
			if err != nil {
				return err
			}
			exprs = append(exprs, s)
		}
		e.line(obj + ".Reduce(" + join(exprs) + ");")
		return nil

	case "Combine":
		if len(args) != 0 {
			return fmt.Errorf("cxx: reducer Combine %d must be argument-free", n.BasePtr().ID())
		}
		// D7: a slotted Combine materializes ONCE; uses inside the
		// re-iteration read slots via Get(slot ivs...).
		if slots, ok := e.slots[ops[0]]; ok && len(slots) > 0 {
			e.slottedVal[n] = slots
		}
		e.c("one Combine — every participant receives the final value")
		name := e.allocVal()
		e.line("auto " + name + " = " + obj + ".Combine();")
		e.values[n] = name
		return nil

	default:
		return fmt.Errorf("cxx: unknown runtime method %q", ir.RuntimeMethodOf(n))
	}
}

// registerLambda records a surviving combiner lambda for the file-scope
// section, in first-reference order (deterministic; duplicates skipped).
func (e *emitter) registerLambda(lam ir.Node) error {
	if lam == nil || lam.KindOf() != ir.KindLambda {
		return fmt.Errorf("cxx: reducer combiner is not a lambda node")
	}
	for _, l := range e.lambdas {
		if l == lam {
			return nil
		}
	}
	e.lambdas = append(e.lambdas, lam)
	return nil
}

// lambdaName is the lambda's file-scope function symbol (quick-261023-cgq,
// Google style): <Kernel>Combine<i>, indexed by emission order — never the
// unstable node ID.
func (e *emitter) lambdaName(lam ir.Node) string {
	i := 0
	for _, l := range e.lambdas {
		if l == lam {
			break
		}
		i++
	}
	return fmt.Sprintf("%sCombine%d", e.kernelSym, i)
}

// lambdaSection hoists every surviving combiner lambda to file scope as a
// stateless __device__ static inline template function (final_lower.md's
// lambda contract: the body references only its own parameters and
// constants — hoisting is trivially sound; passed to the reducer as its
// Lambda construction argument).
func (e *emitter) lambdaSection() (string, error) {
	if len(e.lambdas) == 0 {
		return "", nil
	}
	var b strings.Builder
	for _, lam := range e.lambdas {
		ops := ir.OperandsOf(lam)
		if len(ops) < 1 {
			return "", fmt.Errorf("cxx: lambda %d carries no body", lam.BasePtr().ID())
		}
		params, body := ops[:len(ops)-1], ops[len(ops)-1]

		// Function-scoped value environment: consts and locals inside the
		// lambda body never share the kernel body's symbols.
		savedValues, savedValN, savedConstN := e.values, e.valN, e.constN
		savedParams := e.lambdaParams
		savedDeep, savedOut := e.deep, e.out
		e.values = map[ir.Node]string{}
		e.valN, e.constN = 0, 0
		e.lambdaParams = map[ir.Node]string{}
		e.out, e.deep = &b, 1

		var decls []string
		for i, p := range params {
			name := fmt.Sprintf("p%d", i)
			e.lambdaParams[p] = name
			decls = append(decls, "T "+name)
		}
		fmt.Fprintf(&b, "template <typename T>\n")
		fmt.Fprintf(&b, "__device__ static __forceinline__ T %s(%s) {\n", e.lambdaName(lam), strings.Join(decls, ", "))
		expr, err := e.emitExpr(body)
		if err != nil {
			return "", fmt.Errorf("cxx: lambda %d body: %w", lam.BasePtr().ID(), err)
		}
		fmt.Fprintf(&b, "  return %s;\n", expr)
		fmt.Fprintf(&b, "}\n\n")

		e.values, e.valN, e.constN = savedValues, savedValN, savedConstN
		e.lambdaParams = savedParams
		e.out, e.deep = savedOut, savedDeep
	}
	return b.String(), nil
}
