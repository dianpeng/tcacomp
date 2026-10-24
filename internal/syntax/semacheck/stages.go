package semacheck

import (
	"tcacomp/internal/ir"
	"tcacomp/internal/syntax/ast"
	"tcacomp/internal/syntax/lexer"
)

// Stage resolution (design §5.1 steps 2-4 + §4 rules 11-14). Each resolver
// returns the stage's output element type and live axes (reduce keeps the
// element type; only the axes shrink).

// resolveRootMap resolves the first chain stage (§5.1 step 2): bindings feed
// the lambda param types, the body type becomes the element type T, and an
// omitted as infers first-binding-appearance order.
func (c *checker) resolveRootMap(st *ast.MapStage) (ir.ElemType, []axis, bool) {
	if len(st.Bindings) == 0 {
		c.errorf(st.Pos(), "a let's chain must begin with a root map holding ≥ 1 tensor binding (rule 26)")
		return ir.TF32, nil, false
	}
	ok := true
	elems := make([]ir.ElemType, len(st.Bindings))
	for i, b := range st.Bindings {
		e, good := c.bindOperand(b)
		elems[i] = e
		ok = ok && good
	}
	if st.Lambda == nil {
		c.errorf(st.Pos(), "map stage requires a lambda")
		return ir.TF32, nil, false
	}
	lambda := st.Lambda
	if len(lambda.Params) != len(st.Bindings) {
		c.errorf(lambda.Pos(), "root map lambda has %d params, want %d (one per binding, rule 13)", len(lambda.Params), len(st.Bindings))
		ok = false
	}
	scope, capOK := c.lambdaScope(lambda, elems, ir.TF32, false)
	ok = ok && capOK
	T := c.typeExpr(lambda.Body, scope)
	c.warnUnusedCaptures(lambda, scope)
	if lambda.RetType == nil {
		lambda.RetType = ast.NewScalarType(lambda.Pos(), ast.Elem(T.String()))
	} else if want, good := elemOf(lambda.RetType.Elem); !good || want != T {
		c.errorf(lambda.RetType.Pos(), "map lambda return annotation %s does not match inferred %s (rule 8)", lambda.RetType.Elem, elemName(T))
		ok = false
	}
	axes, asOK := c.resolveRootAs(st)
	ok = ok && asOK
	return T, axes, ok
}

// resolvePipedMap resolves a map stage after a pipe (§5.1 step 4): param 0 is
// the piped value (type T from the previous stage), the remaining params ride
// the bindings, and as is the live axes in order (an explicit clause must
// match exactly).
func (c *checker) resolvePipedMap(st *ast.MapStage, pipeElem ir.ElemType, live []axis) (ir.ElemType, []axis, bool) {
	ok := true
	elems := make([]ir.ElemType, len(st.Bindings))
	for i, b := range st.Bindings {
		e, good := c.bindOperand(b)
		elems[i] = e
		ok = ok && good
	}
	if st.Lambda == nil {
		c.errorf(st.Pos(), "map stage requires a lambda")
		return ir.TF32, nil, false
	}
	lambda := st.Lambda
	if len(lambda.Params) != 1+len(st.Bindings) {
		c.errorf(lambda.Pos(), "piped map lambda has %d params, want %d (the pipe value plus one per binding, rule 13)",
			len(lambda.Params), 1+len(st.Bindings))
		ok = false
	}
	scope, capOK := c.lambdaScope(lambda, elems, pipeElem, true)
	ok = ok && capOK
	T := c.typeExpr(lambda.Body, scope)
	c.warnUnusedCaptures(lambda, scope)
	if lambda.RetType == nil {
		lambda.RetType = ast.NewScalarType(lambda.Pos(), ast.Elem(T.String()))
	} else if want, good := elemOf(lambda.RetType.Elem); !good || want != T {
		c.errorf(lambda.RetType.Pos(), "map lambda return annotation %s does not match inferred %s (rule 8)", lambda.RetType.Elem, elemName(T))
		ok = false
	}
	axes, asOK := c.resolveAsExact(st.Pos(), &st.As, live, axisNames(live))
	ok = ok && asOK
	return T, axes, ok
}

// resolveReduce resolves a reduce stage (§5.1 step 3): the homogeneous type T
// is the pipe element type, both on-params are typed T, the init must be a
// constant/kernel-scalar expression of type T, and the live axes shrink by
// the last axis (rules 12/13; ruling R4/rule 22 for init).
func (c *checker) resolveReduce(st *ast.ReduceStage, pipeElem ir.ElemType, live []axis) ([]axis, bool) {
	ok := true
	if len(st.OnParams) != 2 {
		c.errorf(st.Pos(), "reduce requires exactly two on-params (accumulator, element), got %d (rule 13)", len(st.OnParams))
		return nil, false
	}
	acc, elem := st.OnParams[0], st.OnParams[1]
	if acc.Pipe {
		c.errorf(acc.Pos(), "reduce accumulator must be named; %s may only fill the element param (design §7.3)", acc.Name)
		ok = false
	}
	if ret, good := elemOf(st.RetType.Elem); !good || ret != pipeElem {
		c.errorf(st.RetType.Pos(), "reduce return type %s does not match the pipe element type %s (homogeneous reducer, rule 8)", st.RetType.Elem, elemName(pipeElem))
		ok = false
	}
	c.checkParamSpec(acc, pipeElem)
	c.checkParamSpec(elem, pipeElem)
	// Init: a constant expression over literals and kernel scalar params of
	// type T (rule 22). An omitted init is the deferred scan-seed feature
	// (§8.1); pkg's Reduce rung requires one.
	if st.Init == nil {
		c.errorf(st.Pos(), "reduce requires init = ... in v0 (omitted init is the deferred scan-seed feature, §8.1)")
		ok = false
	} else {
		initScope := newExprScope()
		for n, e := range c.scalarParams {
			initScope.params[n] = e
		}
		initT := c.typeExpr(st.Init, initScope)
		if initT != pipeElem {
			c.errorf(st.Init.Pos(), "reduce init has type %s, want %s (the homogeneous reducer type)", elemName(initT), elemName(pipeElem))
			ok = false
		}
	}
	if len(live) < 2 {
		c.errorf(st.Pos(), "reduce requires ≥ 1 surviving axis; reducing (%s) would leave a scalar (fully-reduced results are not v0, §8.4)", axisNames(live))
		return nil, false
	}
	axes, asOK := c.resolveReduceAs(st.Pos(), &st.As, live)
	ok = ok && asOK
	// Body scope: acc + element param (the pipe slot of a reduce body),
	// both typed T. Reduce stages have no capture clause (grammar §2.6), so
	// kernel scalar params are invisible here (rule 15).
	scope := newExprScope()
	scope.params[acc.Name] = pipeElem
	ename := elem.Name
	if elem.Pipe {
		ename = "%0"
		scope.hasPipe, scope.pipeElem = true, pipeElem
	}
	scope.params[ename] = pipeElem
	bodyT := c.typeExpr(st.Body, scope)
	if bodyT != pipeElem {
		c.errorf(st.Body.Pos(), "reduce body has type %s, want %s (homogeneous reducer)", elemName(bodyT), elemName(pipeElem))
		ok = false
	}
	return axes, ok
}

// lambdaScope builds the expression scope for a map lambda: captures resolve
// to kernel scalar params (rule 14), param 0 (piped) types from the pipe
// element type, the remaining params from the binding element types, and
// omitted param annotations are filled (rule 8).
func (c *checker) lambdaScope(lambda *ast.Lambda, bindingElems []ir.ElemType, pipeElem ir.ElemType, piped bool) (*exprScope, bool) {
	ok := true
	scope := newExprScope()
	for _, name := range lambda.Captures {
		e, found := c.scalarParams[name]
		if !found {
			if c.taken(name) {
				c.errorf(lambda.Pos(), "capture %q is not a kernel scalar param (no tensor/dim/dimvar capture, rule 14)", name)
			} else {
				c.errorf(lambda.Pos(), "capture references unknown name %q (rule 14)", name)
			}
			ok = false
			continue
		}
		scope.captures[name] = e
	}
	offset := 0
	if piped {
		offset = 1
		if len(lambda.Params) >= 1 {
			p0 := lambda.Params[0]
			c.checkParamSpec(p0, pipeElem)
			if p0.Pipe {
				scope.hasPipe, scope.pipeElem = true, pipeElem
				scope.params["%0"] = pipeElem
			} else {
				scope.params[p0.Name] = pipeElem
			}
		}
	}
	for i := offset; i < len(lambda.Params); i++ {
		bi := i - offset
		if bi >= len(bindingElems) {
			break // arity error already reported
		}
		p := lambda.Params[i]
		c.checkParamSpec(p, bindingElems[bi])
		scope.params[p.Name] = bindingElems[bi]
	}
	return scope, ok
}

// warnUnusedCaptures emits one warning per capture never referenced by the
// lambda body (rule 14: unused captures are allowed, mirroring C++).
// Captures that failed to resolve are not tracked and produce no warning —
// their error already names them.
func (c *checker) warnUnusedCaptures(lambda *ast.Lambda, scope *exprScope) {
	for _, name := range lambda.Captures {
		if _, resolved := scope.captures[name]; !resolved {
			continue
		}
		if !scope.usedCaptures[name] {
			c.warnf(lambda.Pos(), "capture %q is never used", name)
		}
	}
}

// checkParamSpec fills an omitted param annotation or verifies a written one
// against the inferred type (rule 8).
func (c *checker) checkParamSpec(p *ast.ParamSpec, want ir.ElemType) {
	if p.Type == nil {
		p.Type = ast.NewScalarType(p.Pos(), ast.Elem(want.String()))
		return
	}
	if got, ok := elemOf(p.Type.Elem); !ok || got != want {
		c.errorf(p.Type.Pos(), "param annotation %s does not match inferred %s (rule 8)", p.Type.Elem, elemName(want))
	}
}

// resolveRootAs resolves a root map's as clause (rule 11): an explicit list
// must be a permutation of the bound dimvars; an omitted clause infers
// first-binding-appearance order (§5.1 step 2) and fills the cell.
func (c *checker) resolveRootAs(st *ast.MapStage) ([]axis, bool) {
	var candidates []axis
	seen := map[string]bool{}
	for _, b := range st.Bindings {
		for _, dv := range b.Dims {
			if seen[dv] {
				continue
			}
			seen[dv] = true
			if a, ok := c.axisOf(dv); ok {
				candidates = append(candidates, a)
			}
		}
	}
	if st.As == nil {
		st.As = axisNames(candidates)
		return candidates, true
	}
	if len(st.As) != len(candidates) {
		c.errorf(st.Pos(), "as (%v) lists %d axes, the stage binds %d (rule 11)", st.As, len(st.As), len(candidates))
	}
	byName := map[string]axis{}
	for _, a := range candidates {
		byName[a.name] = a
	}
	out := make([]axis, 0, len(st.As))
	ok := true
	seen = map[string]bool{}
	for _, name := range st.As {
		if seen[name] {
			c.errorf(st.Pos(), "as lists %q twice (rule 11)", name)
			ok = false
			continue
		}
		seen[name] = true
		a, good := byName[name]
		if !good {
			c.errorf(st.Pos(), "as references %q, which is not one of the bound dimvars (rule 11)", name)
			ok = false
			continue
		}
		out = append(out, a)
	}
	return out, ok
}

// resolveAsExact resolves an as clause that must equal the expected axes in
// order (piped maps and reduces, rule 11), filling the cell when omitted.
func (c *checker) resolveAsExact(pos lexer.Pos, cell *[]string, live []axis, expected []string) ([]axis, bool) {
	byName := map[string]axis{}
	for _, a := range live {
		byName[a.name] = a
	}
	if *cell == nil {
		*cell = expected
		out := make([]axis, len(expected))
		for i, n := range expected {
			out[i] = byName[n]
		}
		return out, true
	}
	as := *cell
	if len(as) != len(expected) {
		c.errorf(pos, "as (%v) does not match the live axes (%v) in order (rule 11)", as, expected)
		return nil, false
	}
	out := make([]axis, 0, len(as))
	ok := true
	for i, name := range as {
		a, good := byName[name]
		if !good {
			c.errorf(pos, "as references %q, which is not a live axis (rule 11)", name)
			ok = false
			continue
		}
		if name != expected[i] {
			c.errorf(pos, "as (%v) must equal the live axes (%v) exactly, in order (rule 11)", as, expected)
			ok = false
		}
		out = append(out, a)
	}
	return out, ok
}

// resolveReduceAs resolves a reduce's as clause (rule 11): an explicit list
// must be the live axes minus exactly one axis — the reduced axis — in the
// live axes' relative order (Appendix A: live (a,b,c) with as (a,c) reduces
// b, the contraction). An omitted clause drops the last live axis (§5.1
// step 3, mirroring pkg's buildReduce) and fills the cell.
func (c *checker) resolveReduceAs(pos lexer.Pos, cell *[]string, live []axis) ([]axis, bool) {
	byName := map[string]axis{}
	for _, a := range live {
		byName[a.name] = a
	}
	if *cell == nil {
		filled := axisNames(live[:len(live)-1])
		*cell = filled
		out := make([]axis, len(live)-1)
		copy(out, live[:len(live)-1])
		return out, true
	}
	as := *cell
	if len(as) != len(live)-1 {
		c.errorf(pos, "reduce as (%v) must list the live axes (%v) minus exactly one (the reduced axis, rule 11)", as, axisNames(live))
		return nil, false
	}
	// Every listed name must be a live axis, in live-relative order.
	out := make([]axis, 0, len(as))
	ok := true
	seen := map[string]bool{}
	for _, name := range as {
		if seen[name] {
			c.errorf(pos, "reduce as lists %q twice (rule 11)", name)
			ok = false
			continue
		}
		seen[name] = true
		a, good := byName[name]
		if !good {
			c.errorf(pos, "reduce as references %q, which is not a live axis (rule 11)", name)
			ok = false
			continue
		}
		out = append(out, a)
	}
	oi := 0
	for _, a := range live {
		if oi < len(out) && out[oi].name == a.name {
			oi++
		}
	}
	if oi != len(out) {
		c.errorf(pos, "reduce as (%v) must preserve the live axes' order (%v, rule 11)", as, axisNames(live))
		ok = false
	}
	return out, ok
}

func axisNames(axes []axis) []string {
	names := make([]string, len(axes))
	for i, a := range axes {
		names[i] = a.name
	}
	return names
}
