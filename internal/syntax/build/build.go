// Package build is the per-kernel lowering driver (doc/syntax-design.md §6
// Task 4): it walks one semachecked kernel AST and drives the pkg builder
// rungs row-by-row, per the design's mapping table. The driver performs NO
// semantic validation of its own — every well-formedness rule was already
// checked by semacheck, whose filled cells (inferred as clauses, let types,
// param roles) this package consumes; the pkg rungs re-check the structural
// invariants (axis coverage, permutation shapes, homogeneous reducers) at
// construction time as the backstop (design R5).
package build

import (
	"fmt"

	"tcacomp/internal/ir"
	"tcacomp/internal/syntax/ast"
	"tcacomp/pkg"
)

// origin stamps the few nodes the driver itself constructs (the KernelMeta
// naming annotation); every other node comes out of the pkg rungs ("builder").
const origin = "frontend"

// Lower drives k's stage chains through the pkg builder and returns the
// verified graph (pkg.Build's BuildingDCE + Materialize + VerifyEach gate).
// k must have passed semacheck — the driver consumes its filled cells and
// errors out clearly when a required cell is missing.
func Lower(k *ast.Kernel) (*ir.Graph, error) {
	d := &driver{
		dims:       map[string]pkg.Dim{},
		tensors:    map[string]pkg.Tensor{},
		tensorDims: map[string][]pkg.Dim{},
		scalars:    map[string]pkg.Expr{},
	}
	return d.lower(k)
}

// driver carries one kernel's lowering state: the builder under construction
// plus the driver-side symbol tables (the builder's own tables are keyed the
// same way; the driver keeps parallel maps because pkg handles do not expose
// their dim lists).
type driver struct {
	b          *pkg.Builder
	dims       map[string]pkg.Dim    // declared shape-dim handles, keyed by name
	tensors    map[string]pkg.Tensor // params, locals, and the output tensor
	tensorDims map[string][]pkg.Dim  // per declared tensor, its shape dims in order
	scalars    map[string]pkg.Expr   // scalar param handles (captures)
	dimvars    map[string]pkg.Dim    // per-chain dimvar -> shape-dim handle table
	err        error                 // first expression-lowering error (deferred)
}

// lower lowers one kernel: dims and params first (the declaration order is
// the launcher parameter order), then each let's chain in body order.
func (d *driver) lower(k *ast.Kernel) (*ir.Graph, error) {
	// Body order facts semacheck guarantees: exactly one return statement,
	// last, naming a let. Collect lets in order.
	var lets []*ast.LetDecl
	returnName := ""
	for _, st := range k.Body {
		switch st := st.(type) {
		case *ast.DimVarDecl:
			// Dimvars declare logical axes only — no IR role. The dims
			// themselves come from the tensor shapes (R1 implicit dims).
		case *ast.LetDecl:
			lets = append(lets, st)
		case *ast.ReturnStmt:
			returnName = st.Name
		}
	}
	var outLet *ast.LetDecl
	for _, l := range lets {
		if l.Name == returnName {
			outLet = l
		}
	}
	if outLet == nil {
		return nil, fmt.Errorf("frontend: kernel %s: return names no let", k.Name)
	}

	d.b = pkg.NewBuilder(k.Name)

	// Dims: every shape dim of every tensor type, in first-appearance order
	// (params in signature order, then lets in body order). A dynamic dim is
	// keyed by its declared name; a static extent uses the canonical e<extent>
	// spelling (matching pkg.Local's canonicalization).
	seen := map[string]bool{}
	declare := func(tt *ast.TensorType) error {
		for _, sd := range tt.Shape {
			name, extent := dimKey(sd)
			if seen[name] {
				continue
			}
			seen[name] = true
			d.dims[name] = d.b.Dim(name, extent)
		}
		return nil
	}
	for _, p := range k.Params {
		if tt, ok := p.Type.(*ast.TensorType); ok {
			_ = declare(tt)
		}
	}
	for _, l := range lets {
		tt, ok := l.Type.(*ast.TensorType)
		if !ok || tt == nil {
			return nil, fmt.Errorf("frontend: kernel %s: let %s has no inferred tensor type (run semacheck first)", k.Name, l.Name)
		}
		_ = declare(tt)
	}

	// Tensor params in signature order.
	for _, p := range k.Params {
		tt, ok := p.Type.(*ast.TensorType)
		if !ok {
			continue
		}
		elem, ok := elemOf(tt.Elem.Elem)
		if !ok {
			return nil, fmt.Errorf("frontend: kernel %s: param %s has unmappable element type %q", k.Name, p.Name, tt.Elem.Elem)
		}
		handles := make([]pkg.Dim, len(tt.Shape))
		for i, sd := range tt.Shape {
			handles[i] = d.dim(dimKeyName(sd))
		}
		t := d.b.Tensor(p.Name, elem, handles...)
		d.tensors[p.Name] = t
		d.tensorDims[p.Name] = handles
	}

	// The output tensor, declared before the scalars so the launcher order is
	// (tensor params..., output, scalars...): the returned let's type was
	// inferred by semacheck against the kernel's return type.
	{
		tt := outLet.Type.(*ast.TensorType)
		elem, ok := elemOf(tt.Elem.Elem)
		if !ok {
			return nil, fmt.Errorf("frontend: kernel %s: output has unmappable element type %q", k.Name, tt.Elem.Elem)
		}
		handles := make([]pkg.Dim, len(tt.Shape))
		for i, sd := range tt.Shape {
			handles[i] = d.dim(dimKeyName(sd))
		}
		t := d.b.Tensor(outLet.Name, elem, handles...)
		d.tensors[outLet.Name] = t
		d.tensorDims[outLet.Name] = handles
	}

	// Scalar params (launch-uniform captures).
	for _, p := range k.Params {
		if st, ok := p.Type.(*ast.ScalarType); ok {
			elem, ok := elemOf(st.Elem)
			if !ok {
				return nil, fmt.Errorf("frontend: kernel %s: scalar %s has unmappable element type %q", k.Name, p.Name, st.Elem)
			}
			d.scalars[p.Name] = d.b.ScalarParam(p.Name, elem)
		}
	}

	// Chains in body order. The returned let stores into the pre-declared
	// output tensor; every other let is a MemNew-backed local.
	for _, l := range lets {
		var target pkg.Tensor
		if l == outLet {
			target = d.tensors[l.Name]
		} else {
			tt := l.Type.(*ast.TensorType)
			elem, ok := elemOf(tt.Elem.Elem)
			if !ok {
				return nil, fmt.Errorf("frontend: kernel %s: let %s has unmappable element type %q", k.Name, l.Name, tt.Elem.Elem)
			}
			refs := make([]ir.DimRef, len(tt.Shape))
			handles := make([]pkg.Dim, len(tt.Shape))
			for i, sd := range tt.Shape {
				name, extent := dimKey(sd)
				if sd.Dynamic {
					refs[i] = ir.DimRef{Name: name}
				} else {
					refs[i] = ir.DimRef{Extent: uint64(extent)}
				}
				handles[i] = d.dim(name)
			}
			target = d.b.Local(l.Name, ir.InternTensor(elem, refs))
			d.tensors[l.Name] = target
			d.tensorDims[l.Name] = handles
		}
		if err := d.lowerChain(k.Name, l, target); err != nil {
			return nil, err
		}
	}

	g, err := d.b.Build()
	if err != nil {
		return nil, fmt.Errorf("frontend: kernel %s: %w", k.Name, err)
	}

	// @name{cxx="..."} overrides the emitted C++ symbol names.
	for _, ann := range k.Anns {
		if ann.Name != "name" {
			continue
		}
		for _, a := range ann.Args {
			if a.Key == "cxx" && a.Value.Kind == ast.ConstString {
				g.NewKernelMeta(origin, a.Value.Text, "")
			}
		}
	}
	return g, nil
}

// dimKey normalizes one shape dim to its (name, extent) declaration key:
// dynamic dims key by name with extent 0; static extents key by the
// canonical e<extent> spelling (the same spelling pkg.Local expects for
// unnamed static dims) with the extent carried for the Dim declaration.
func dimKey(sd ast.ShapeDim) (string, int) {
	if sd.Dynamic {
		return sd.Name, 0
	}
	return fmt.Sprintf("e%d", sd.Extent), int(sd.Extent)
}

// dimKeyName returns the declaration-table name for one shape dim.
func dimKeyName(sd ast.ShapeDim) string {
	name, _ := dimKey(sd)
	return name
}

// dim fetches one declared dim handle by name. Names come from shapes the
// driver itself declared, so a miss is an internal error, not user input;
// the zero handle's failed checkDim surfaces it as a rung error.
func (d *driver) dim(name string) pkg.Dim {
	return d.dims[name]
}

// lowerChain drives one let's stage chain through the pkg rungs. The chain
// shape boundaries the rungs enforce (root map -> at least one reduce ->
// at most one piped map -> store) are checked up front with kernel-shaped
// messages so a semacheck regression surfaces here, not as a rung error.
func (d *driver) lowerChain(kernel string, l *ast.LetDecl, target pkg.Tensor) error {
	stages := l.Chain.Stages
	root, ok := stages[0].(*ast.MapStage)
	if !ok {
		return fmt.Errorf("frontend: kernel %s: let %s: the chain must open with a map stage", kernel, l.Name)
	}
	if root.As == nil {
		return fmt.Errorf("frontend: kernel %s: let %s: the root map's as clause was not inferred (run semacheck first)", kernel, l.Name)
	}

	// Per-chain dimvar table: each root binding positions dimvars on shape
	// dims. A dimvar binding two distinct dims is the rule-10 violation
	// semacheck already rejects; recheck cheaply for a clear driver message.
	d.dimvars = map[string]pkg.Dim{}
	for _, b := range root.Bindings {
		dims, ok := d.tensorDims[b.Tensor]
		if !ok {
			return fmt.Errorf("frontend: kernel %s: let %s: binding names unknown tensor %q", kernel, l.Name, b.Tensor)
		}
		if len(b.Dims) != len(dims) {
			return fmt.Errorf("frontend: kernel %s: let %s: binding %s[%v] has %d subscripts, want %d (run semacheck first)",
				kernel, l.Name, b.Tensor, b.Dims, len(b.Dims), len(dims))
		}
		for i, dv := range b.Dims {
			if prev, exists := d.dimvars[dv]; exists && prev.Name() != dims[i].Name() {
				return fmt.Errorf("frontend: kernel %s: let %s: dimvar @%s binds distinct shape dims @%s and @%s",
					kernel, l.Name, dv, prev.Name(), dims[i].Name())
			}
			d.dimvars[dv] = dims[i]
		}
	}

	// Iter-order relocation: pkg's Reduce always drops the LAST axis of the
	// root-as live order, so the root as clause must place the to-be-reduced
	// axes last, in reverse reduce order; the surviving axes keep the source
	// root-as order. The iteration cluster uses the same order (the rung only
	// requires as to be a permutation of iters). The source's root as (a,b,c)
	// with a reduce as (a,c) therefore lowers with @b parked last.
	axisOrder, err := relocate(kernel, l.Name, stages)
	if err != nil {
		return err
	}
	iters := make([]pkg.Dim, len(axisOrder))
	for i, dv := range axisOrder {
		h, ok := d.dimvars[dv]
		if !ok {
			return fmt.Errorf("frontend: kernel %s: let %s: dimvar @%s is not bound by the root map", kernel, l.Name, dv)
		}
		iters[i] = h
	}

	// Root map.
	ops := d.operands(kernel, l.Name, root.Bindings)
	fn := func(args ...pkg.Expr) pkg.Expr {
		scope := d.lambdaScope(root.Lambda, args)
		return d.lowerExpr(root.Lambda.Body, scope)
	}
	rm := d.b.Stmt().Iter(iters...)
	ra, err := rm.Map(ops, fn)
	if err != nil {
		return fmt.Errorf("frontend: kernel %s: let %s: map: %w", kernel, l.Name, err)
	}
	if d.err != nil {
		return fmt.Errorf("frontend: kernel %s: let %s: %w", kernel, l.Name, d.err)
	}
	// The root map's as clause lowers in the relocated axis order (see
	// relocate): the rung pins the pipe's live order from these dims.
	mid := ra.As(d.asHandles(axisOrder)...)

	// Remaining stages.
	var cur any = mid
	piped := false
	for _, st := range stages[1:] {
		switch st := st.(type) {
		case *ast.ReduceStage:
			if st.As == nil {
				return fmt.Errorf("frontend: kernel %s: let %s: a reduce's as clause was not inferred (run semacheck first)", kernel, l.Name)
			}
			init := pkg.Expr{}
			if st.Init != nil {
				init = d.lowerExpr(st.Init, d.scalarScope())
				if d.err != nil {
					return fmt.Errorf("frontend: kernel %s: let %s: %w", kernel, l.Name, d.err)
				}
			}
			fn := func(acc, v pkg.Expr) pkg.Expr {
				scope := d.scalarScope()
				if len(st.OnParams) > 0 {
					scope[specName(st.OnParams[0])] = acc
				}
				if len(st.OnParams) > 1 {
					scope[specName(st.OnParams[1])] = v
				}
				return d.lowerExpr(st.Body, scope)
			}
			switch c := cur.(type) {
			case *pkg.MidChain:
				cur, err = c.Reduce(init, fn)
			case *pkg.AfterReduce:
				cur, err = c.Reduce(init, fn)
			default:
				return fmt.Errorf("frontend: kernel %s: let %s: a reduction cannot follow a piped map (pkg v0 rung shape)", kernel, l.Name)
			}
			if err != nil {
				return fmt.Errorf("frontend: kernel %s: let %s: reduce: %w", kernel, l.Name, err)
			}
			if d.err != nil {
				return fmt.Errorf("frontend: kernel %s: let %s: %w", kernel, l.Name, d.err)
			}
		case *ast.MapStage:
			if piped {
				return fmt.Errorf("frontend: kernel %s: let %s: v0 pkg rungs admit at most one piped map per chain", kernel, l.Name)
			}
			if st.As == nil {
				return fmt.Errorf("frontend: kernel %s: let %s: the piped map's as clause was not inferred (run semacheck first)", kernel, l.Name)
			}
			ar, ok := cur.(*pkg.AfterReduce)
			if !ok {
				return fmt.Errorf("frontend: kernel %s: let %s: a piped map requires a preceding reduction (pkg v0 rung shape)", kernel, l.Name)
			}
			ops := d.operands(kernel, l.Name, st.Bindings)
			fn := func(args ...pkg.Expr) pkg.Expr {
				scope := d.lambdaScope(st.Lambda, args)
				return d.lowerExpr(st.Lambda.Body, scope)
			}
			am, err := ar.Map(ops, fn)
			if err != nil {
				return fmt.Errorf("frontend: kernel %s: let %s: piped map: %w", kernel, l.Name, err)
			}
			if d.err != nil {
				return fmt.Errorf("frontend: kernel %s: let %s: %w", kernel, l.Name, d.err)
			}
			cur = am.As(d.asHandles(st.As)...)
			piped = true
		default:
			return fmt.Errorf("frontend: kernel %s: let %s: unknown stage %T", kernel, l.Name, st)
		}
	}

	// Store. The store axes are the target's own shape dims in order — the
	// structural rule the rungs re-check against the live set.
	axes := d.tensorDims[target.Name()]
	switch c := cur.(type) {
	case *pkg.AfterReduce:
		if _, err := c.Store(target, axes...); err != nil {
			return fmt.Errorf("frontend: kernel %s: let %s: store: %w", kernel, l.Name, err)
		}
	case *pkg.Tail:
		if _, err := c.Store(target, axes...); err != nil {
			return fmt.Errorf("frontend: kernel %s: let %s: store: %w", kernel, l.Name, err)
		}
	default:
		return fmt.Errorf("frontend: kernel %s: let %s: v0 pkg rungs require at least one reduction before the store", kernel, l.Name)
	}
	return nil
}

// relocate computes the axis order one chain lowers with — the order of both
// the iteration cluster and the root map's as clause: the root as dimvars
// that no reduce removes (in root as order), then the reduced dimvars in
// reverse reduce order. pkg's Reduce always drops the last axis of the live
// (root-as) order, so parking each chain reduce's removed axis last — first
// reduce's axis last of all — makes each rung Reduce drop exactly the axis
// the chain's reduce as removes.
func relocate(kernel, let string, stages []ast.Stage) ([]string, error) {
	root := stages[0].(*ast.MapStage)
	reduced := map[string]bool{}
	var reducedOrder []string
	live := append([]string(nil), root.As...)
	for _, st := range stages[1:] {
		r, ok := st.(*ast.ReduceStage)
		if !ok {
			continue // piped maps do not change the live set
		}
		if r.As == nil {
			return nil, fmt.Errorf("frontend: kernel %s: let %s: a reduce's as clause was not inferred (run semacheck first)", kernel, let)
		}
		var dropped []string
		kept := map[string]bool{}
		for _, dv := range r.As {
			kept[dv] = true
		}
		for _, dv := range live {
			if !kept[dv] {
				dropped = append(dropped, dv)
			}
		}
		if len(dropped) != 1 {
			return nil, fmt.Errorf("frontend: kernel %s: let %s: reduce drops %d axes %v, want exactly 1 (run semacheck first)", kernel, let, len(dropped), dropped)
		}
		reduced[dropped[0]] = true
		reducedOrder = append(reducedOrder, dropped[0])
		live = append([]string(nil), r.As...)
	}
	var survivors []string
	for _, dv := range root.As {
		if !reduced[dv] {
			survivors = append(survivors, dv)
		}
	}
	// Reverse reduce order: the axis the FIRST reduce removes must be LAST in
	// the iteration cluster, so it is the last live axis at that Reduce.
	for i, j := 0, len(reducedOrder)-1; i < j; i, j = i+1, j-1 {
		reducedOrder[i], reducedOrder[j] = reducedOrder[j], reducedOrder[i]
	}
	return append(survivors, reducedOrder...), nil
}

// operands builds the root/piped map operand accesses: each binding's tensor
// at its own shape dims in order (the dimvar identity is what binds binding
// position i to shape dim i — structural indexing, rule 9).
func (d *driver) operands(kernel, let string, bindings []*ast.Binding) []pkg.Operand {
	ops := make([]pkg.Operand, len(bindings))
	for i, b := range bindings {
		t := d.tensors[b.Tensor]
		ops[i] = pkg.At(t, d.tensorDims[b.Tensor]...)
	}
	return ops
}

// asHandles maps one inferred as clause's dimvars to their per-chain handles.
func (d *driver) asHandles(as []string) []pkg.Dim {
	handles := make([]pkg.Dim, len(as))
	for i, dv := range as {
		handles[i] = d.dimvars[dv]
	}
	return handles
}

// specName returns the scope key of one param spec: the pipe placeholder
// %0 for pipe specs, the declared name otherwise.
func specName(ps *ast.ParamSpec) string {
	if ps.Pipe {
		return "%0"
	}
	return ps.Name
}

// lambdaScope binds one map lambda's params positionally to the rung's
// lambda arguments (position, not spelling, identifies each slot) and its
// captures to the scalar param handles.
func (d *driver) lambdaScope(l *ast.Lambda, args []pkg.Expr) map[string]pkg.Expr {
	scope := d.scalarScope()
	for i, ps := range l.Params {
		if i < len(args) {
			scope[specName(ps)] = args[i]
		}
	}
	return scope
}

// scalarScope is the base expression scope: the kernel's scalar params.
func (d *driver) scalarScope() map[string]pkg.Expr {
	scope := make(map[string]pkg.Expr, len(d.scalars))
	for name, e := range d.scalars {
		scope[name] = e
	}
	return scope
}

// lowerExpr lowers one resolved typed expr into the builder's expression
// DSL. Errors are recorded in d.err (first wins) and a zero Expr returned,
// so the enclosing rung's own error surfaces the failure.
func (d *driver) lowerExpr(e ast.Expr, scope map[string]pkg.Expr) pkg.Expr {
	if d.err != nil {
		return pkg.Expr{}
	}
	switch e := e.(type) {
	case *ast.BasicLit:
		switch v := e.Val.(type) {
		case int32:
			return d.b.Int32(v)
		case uint32:
			return d.b.UInt32(v)
		case float32:
			return d.b.Float32(v)
		case bool:
			return d.b.Bool(v)
		default:
			d.err = fmt.Errorf("literal %q has unmappable value type %T", e.Value, e.Val)
			return pkg.Expr{}
		}
	case *ast.NameRef:
		if e2, ok := scope[e.Name]; ok {
			return e2
		}
		d.err = fmt.Errorf("unknown name %q in expression (run semacheck first)", e.Name)
		return pkg.Expr{}
	case *ast.PipeRef:
		if e2, ok := scope["%0"]; ok {
			return e2
		}
		d.err = fmt.Errorf("pipe reference outside a reduction (run semacheck first)")
		return pkg.Expr{}
	case *ast.BinaryExpr:
		l, r := d.lowerExpr(e.L, scope), d.lowerExpr(e.R, scope)
		if d.err != nil {
			return pkg.Expr{}
		}
		return d.binOp(string(e.Op), l, r)
	case *ast.UnaryExpr:
		x := d.lowerExpr(e.X, scope)
		if d.err != nil {
			return pkg.Expr{}
		}
		return d.unOp(string(e.Op), x)
	case *ast.SelectExpr:
		c, tt, fe := d.lowerExpr(e.Cond, scope), d.lowerExpr(e.Then, scope), d.lowerExpr(e.Else, scope)
		if d.err != nil {
			return pkg.Expr{}
		}
		return pkg.Select(c, tt, fe)
	case *ast.CallExpr:
		args := make([]pkg.Expr, len(e.Args))
		for i, a := range e.Args {
			args[i] = d.lowerExpr(a, scope)
		}
		if d.err != nil {
			return pkg.Expr{}
		}
		e2, err := pkg.Call(e.Name, args...)
		if err != nil {
			d.err = err
			return pkg.Expr{}
		}
		return e2
	case *ast.CastExpr:
		x := d.lowerExpr(e.X, scope)
		if d.err != nil {
			return pkg.Expr{}
		}
		elem, ok := elemOf(e.Target)
		if !ok {
			d.err = fmt.Errorf("cast target %q has no ir element type", e.Target)
			return pkg.Expr{}
		}
		return pkg.Cast(x, elem)
	default:
		d.err = fmt.Errorf("unsupported expression node %T", e)
		return pkg.Expr{}
	}
}

// binOp maps one binary operator spelling onto the builder method table.
func (d *driver) binOp(op string, l, r pkg.Expr) pkg.Expr {
	var e pkg.Expr
	switch op {
	case "||":
		e = d.b.Or(l, r)
	case "&&":
		e = d.b.And(l, r)
	case "|":
		e = d.b.BitOr(l, r)
	case "^":
		e = d.b.BitXor(l, r)
	case "&":
		e = d.b.BitAnd(l, r)
	case "==":
		e = d.b.Eq(l, r)
	case "!=":
		e = d.b.Ne(l, r)
	case "<":
		e = d.b.Lt(l, r)
	case "<=":
		e = d.b.Le(l, r)
	case ">":
		e = d.b.Gt(l, r)
	case ">=":
		e = d.b.Ge(l, r)
	case "<<":
		e = d.b.Shl(l, r)
	case ">>":
		e = d.b.Shr(l, r)
	case "+":
		e = d.b.Add(l, r)
	case "-":
		e = d.b.Sub(l, r)
	case "*":
		e = d.b.Mul(l, r)
	case "/":
		e = d.b.Div(l, r)
	case "%":
		e = d.b.Mod(l, r)
	default:
		d.err = fmt.Errorf("unsupported binary operator %q", op)
		return pkg.Expr{}
	}
	return e
}

// unOp maps one unary operator spelling onto the builder method table.
func (d *driver) unOp(op string, x pkg.Expr) pkg.Expr {
	switch op {
	case "-":
		return d.b.Neg(x)
	case "+":
		return d.b.Pos(x)
	case "!":
		return d.b.Not(x)
	case "~":
		return d.b.BitsNot(x)
	default:
		d.err = fmt.Errorf("unsupported unary operator %q", op)
		return pkg.Expr{}
	}
}

// elemOf maps one frontend element-type spelling onto the ir element type,
// 1:1 with semacheck's table (types.go layering: the AST carries spellings,
// ir carries types).
func elemOf(e ast.Elem) (ir.ElemType, bool) {
	switch e {
	case ast.EBool:
		return ir.TBool, true
	case ast.EF32:
		return ir.TF32, true
	case ast.EF16:
		return ir.TF16, true
	case ast.EBF16:
		return ir.TBF16, true
	case ast.EI8:
		return ir.TI8, true
	case ast.EI16:
		return ir.TI16, true
	case ast.EI32:
		return ir.TI32, true
	case ast.EU8:
		return ir.TU8, true
	case ast.EU16:
		return ir.TU16, true
	case ast.EU32:
		return ir.TU32, true
	}
	return ir.TBool, false
}
