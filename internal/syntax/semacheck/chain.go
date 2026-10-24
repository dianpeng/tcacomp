package semacheck

import (
	"tcacomp/internal/ir"
	"tcacomp/internal/syntax/ast"
)

// Stage-chain resolution (design §5.1): a single forward pass per chain
// threading the live-axis list and element type through the stages, applying
// the §4 binding/axis/lambda rules. The omitted annotation cells (param
// types, lambda return, as lists, the let's type) are filled in place.

// axis is one live axis of a stage: a dimvar name and the shape dim it is
// bound to (its extent source, rule 26).
type axis struct {
	name string
	src  ast.ShapeDim
}

// checkBody scans the kernel body statements in order (rules 23-26) and
// resolves each let's chain (§5.1).
func (c *checker) checkBody() {
	var returns int
	for i, s := range c.k.Body {
		switch s := s.(type) {
		case *ast.DimVarDecl:
			c.registerDimvars(s)
		case *ast.LetDecl:
			c.checkLet(s)
		case *ast.ReturnStmt:
			returns++
			if i != len(c.k.Body)-1 {
				c.errorf(s.Pos(), "return must be the last statement (rule 25)")
			}
			c.checkReturn(s)
		}
	}
	if returns == 0 {
		c.errorf(c.k.Pos(), "kernel %s must end with exactly one return (rule 25)", c.k.Name)
	}
}

// registerDimvars registers one dimvar declaration (rule 5).
func (c *checker) registerDimvars(d *ast.DimVarDecl) {
	for _, name := range d.Names {
		if c.taken(name) {
			c.errorf(d.Pos(), "duplicate name %q (one namespace, rule 5)", name)
			continue
		}
		c.dimvars[name] = &dimvarInfo{pos: d.Pos()}
	}
}

// checkLet resolves one let's chain and registers the let (rules 23, 24, 26).
func (c *checker) checkLet(l *ast.LetDecl) {
	if c.taken(l.Name) {
		c.errorf(l.Pos(), "duplicate name %q (a second let of the same name is a duplicate, not a reassignment, rules 5/23)", l.Name)
		return
	}
	if l.Chain == nil || len(l.Chain.Stages) == 0 {
		c.errorf(l.Pos(), "let %s requires a stage chain", l.Name)
		return
	}
	elem, axes, ok := c.resolveChain(l.Chain)
	if !ok {
		// Fail fast at this let's granularity: register a poison let so
		// later statements (the return check) do not cascade with
		// "unknown let" noise (design §4: one diagnostic per violation).
		c.lets[l.Name] = &letInfo{pos: l.Pos(), elem: ir.TF32}
		return
	}
	out := make([]ast.ShapeDim, len(axes))
	for i, a := range axes {
		out[i] = a.src
	}
	// Fill or verify the written annotation (rule 8/24).
	if l.Type == nil {
		shape := make([]ast.ShapeDim, len(out))
		copy(shape, out)
		l.Type = ast.NewTensorType(l.Pos(), ast.ScalarType{Elem: ast.Elem(elem.String())}, shape)
	} else {
		tt, ok := l.Type.(*ast.TensorType)
		if !ok {
			c.errorf(l.Type.Pos(), "let %s annotation must be a tensor type; a chain always produces a tensor", l.Name)
			return
		}
		c.checkShapeCompat(tt, elem, out, "let "+l.Name)
	}
	c.lets[l.Name] = &letInfo{pos: l.Pos(), elem: elem, shape: out}
}

// checkShapeCompat verifies an annotation's element type and shape against
// the inferred values (rules 8/24/26): element equality plus per-dim extent
// compatibility — two symbolic dims must carry the same name; a symbolic and
// a static dim never match; two static dims must carry the same extent.
func (c *checker) checkShapeCompat(tt *ast.TensorType, inferred ir.ElemType, axes []ast.ShapeDim, what string) {
	annElem, ok := elemOf(tt.Elem.Elem)
	if !ok {
		c.errorf(tt.Pos(), "%s: unknown element type %q", what, tt.Elem.Elem)
		return
	}
	if annElem != inferred {
		c.errorf(tt.Pos(), "%s: annotated element %s does not match inferred %s (rule 8)", what, elemName(annElem), elemName(inferred))
	}
	if tt.Shape == nil {
		c.errorf(tt.Pos(), "%s: tensor shape is required on a let annotation", what)
		return
	}
	if len(tt.Shape) != len(axes) {
		c.errorf(tt.Pos(), "%s: annotated rank %d does not match the chain's live axes %d (rule 26)", what, len(tt.Shape), len(axes))
		return
	}
	for i, d := range tt.Shape {
		want := axes[i]
		annDyn := d.Dynamic || d.Name != ""
		wantDyn := want.Dynamic || want.Name != ""
		switch {
		case annDyn && wantDyn:
			if d.Name != want.Name {
				c.errorf(d.Pos(), "%s: dim %d is bound to shape dim %q, annotated %q (rule 26)", what, i, want.Name, d.Name)
			}
		case annDyn != wantDyn:
			c.errorf(d.Pos(), "%s: dim %d mixes a symbolic dim with a static extent (rule 26)", what, i)
		case d.Extent != want.Extent:
			c.errorf(d.Pos(), "%s: dim %d has extent %d, the chain's axis has %d (rule 26)", what, i, d.Extent, want.Extent)
		}
	}
}

// checkReturn validates the return statement (rule 25): the name is a
// let-bound tensor whose type equals the kernel return type.
func (c *checker) checkReturn(r *ast.ReturnStmt) {
	info, ok := c.lets[r.Name]
	if !ok {
		if c.taken(r.Name) {
			c.errorf(r.Pos(), "return names %q, which is not a let-bound tensor (rule 25)", r.Name)
		} else {
			c.errorf(r.Pos(), "return names unknown let %q (rule 25)", r.Name)
		}
		return
	}
	if info.shape == nil {
		return // poisoned let: the let's own error already names the problem
	}
	switch ret := c.k.Ret.(type) {
	case *ast.TensorType:
		annElem, ok := elemOf(ret.Elem.Elem)
		if !ok {
			return
		}
		if annElem != info.elem {
			c.errorf(r.Pos(), "return type mismatch: %s is tensor<%s>, kernel returns tensor<%s> (rule 8/25)",
				r.Name, elemName(info.elem), elemName(annElem))
			return
		}
		c.checkShapeCompat(ret, info.elem, info.shape, "return of "+r.Name)
	case *ast.ScalarType:
		c.errorf(r.Pos(), "return type mismatch: %s is a tensor, kernel returns scalar %s (rule 25)", r.Name, ret.Elem)
	}
}

// resolveChain runs the §5.1 forward pass over one chain, returning the final
// element type and live axes. ok is false when any stage failed (the caller
// skips dependent checks but diagnostics are already recorded).
func (c *checker) resolveChain(chain *ast.StageChain) (ir.ElemType, []axis, bool) {
	var elem ir.ElemType
	var live []axis
	for i, st := range chain.Stages {
		switch st := st.(type) {
		case *ast.MapStage:
			var out []axis
			var ok bool
			if i == 0 {
				elem, out, ok = c.resolveRootMap(st)
			} else {
				elem, out, ok = c.resolvePipedMap(st, elem, live)
			}
			if !ok {
				return ir.TF32, nil, false
			}
			live = out
		case *ast.ReduceStage:
			out, ok := c.resolveReduce(st, elem, live)
			if !ok {
				return ir.TF32, nil, false
			}
			live = out
		default:
			c.errorf(st.Pos(), "unknown stage node %T", st)
			return ir.TF32, nil, false
		}
	}
	return elem, live, true
}

// bindOperand validates one stage binding (rules 9/10): the tensor is a
// declared tensor (parameter or let), the subscript count equals the rank,
// every subscript is a declared dimvar, and each dimvar's structural binding
// is consistent. It returns the operand's element type.
func (c *checker) bindOperand(b *ast.Binding) (ir.ElemType, bool) {
	var tt *ast.TensorType
	var elem ir.ElemType
	if p, ok := c.tensorParams[b.Tensor]; ok {
		tt = p.Type.(*ast.TensorType)
		e, _ := elemOf(tt.Elem.Elem)
		elem = e
	} else if info, ok := c.lets[b.Tensor]; ok {
		elem = info.elem
		tt = ast.NewTensorType(info.pos, ast.ScalarType{Elem: ast.Elem(elem.String())}, info.shape)
	} else {
		if c.taken(b.Tensor) {
			c.errorf(b.Pos(), "binding %q is not a tensor (rule 9)", b.Tensor)
		} else {
			c.errorf(b.Pos(), "binding references unknown tensor %q (rule 9)", b.Tensor)
		}
		return ir.TF32, false
	}
	if len(b.Dims) != len(tt.Shape) {
		c.errorf(b.Pos(), "binding %s[%v]: subscript count %d does not match tensor rank %d (structural indexing, rule 9)",
			b.Tensor, b.Dims, len(b.Dims), len(tt.Shape))
		return elem, false
	}
	for i, dv := range b.Dims {
		info, ok := c.dimvars[dv]
		if !ok {
			if _, isDim := c.dims[dv]; isDim {
				c.errorf(b.Pos(), "binding %s uses shape dim %q where a dimvar is required (rule 10)", b.Tensor, dv)
			} else {
				c.errorf(b.Pos(), "binding %s uses undeclared dimvar %q (rule 10)", b.Tensor, dv)
			}
			continue
		}
		src := tt.Shape[i]
		if info.set {
			prevDyn := info.src.Dynamic || info.src.Name != ""
			srcDyn := src.Dynamic || src.Name != ""
			if prevDyn != srcDyn || (prevDyn && info.src.Name != src.Name) || (!prevDyn && info.src.Extent != src.Extent) {
				c.errorf(b.Pos(), "dimvar %q is bound both to shape dim %q and %q (inconsistent structural binding)", dv, shapeDimName(info.src), shapeDimName(src))
			}
			continue
		}
		info.src, info.set = src, true
	}
	return elem, true
}

// shapeDimName renders a shape dim for diagnostics.
func shapeDimName(d ast.ShapeDim) string {
	if d.Dynamic || d.Name != "" {
		return d.Name
	}
	return fmtUint(d.Extent)
}

func fmtUint(v uint64) string {
	// Small local renderer: strconv would be fine, but this keeps the import
	// surface minimal and extents are diagnostic-only.
	const digits = "0123456789"
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = digits[v%10]
		v /= 10
	}
	return string(buf[i:])
}

// axisOf resolves a dimvar name to its live axis (rule 11 helpers).
func (c *checker) axisOf(name string) (axis, bool) {
	info, ok := c.dimvars[name]
	if !ok || !info.set {
		return axis{}, false
	}
	return axis{name: name, src: info.src}, true
}
