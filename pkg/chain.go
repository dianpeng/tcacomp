package pkg

import (
	"fmt"
	"strings"

	"tcacomp/internal/ir"
)

// This file is the fluent pipe-chain (D-07 + RESEARCH Pattern 8): one Go
// type per legal rung, so illegal pipe orders — Iter after Reduce, anything
// after Store — are compile-time errors, not runtime failures. The rungs
// mirror doc/tca.md's `|`-separated stage notation 1:1, and Phase 5's
// parser lowers near-mechanically onto them.
//
// Rung shapes (the pinned oracle chain compiles against these exactly):
//
//	Stmt().Iter(dims...) -> RootMap
//	RootMap.Map(ops, fn) -> RootAs            (root map; fn arity == len(ops))
//	RootAs.As(dims...) -> MidChain            (root as: permutation of iter dims)
//	MidChain.Reduce(init, fn) -> AfterReduce
//	AfterReduce.Reduce(init, fn) -> AfterReduce   (chained reductions, §9.5)
//	AfterReduce.Map(ops, fn) -> AfterMap          (piped map; fn arity == 1+len(ops))
//	AfterReduce.Store(t, axes...) -> Stored       (store directly after reduce, §9.8)
//	AfterMap.As(dims...) -> Tail                  (piped as: exactly the live axes in order)
//	Tail.Store(t, axes...) -> Stored              (*Stored is terminal)
//
// Map lambdas are Go funcs over the expr builders (D-09); there are no
// string-parsed lambdas. Reduce combinators take exactly (acc, v).

// chain is the state threaded through one statement's rungs.
type chain struct {
	b     *Builder
	id    uint64   // reserved statement ID (ir.ReserveStatementID)
	retID uint64   // reserved return-statement ID, consumed by a new output
	stage string   // rung reached, for diagnostics
	iters []Dim    // root iteration-cluster dims (nil once the root map is built)
	live  []Dim    // axes live at the pipe, in the current as-order
	pipe  ir.Node // the pipe's stage node
	elem  ir.Type // the pipe's scalar element type
	err   error    // first deferred validation error (rung or declaration)
}

// deferErr records the first validation error; rungs consult it before
// doing anything else, so the earliest violation wins.
func (c *chain) deferErr(err error) {
	if err != nil && c.err == nil {
		c.err = err
	}
}

// pending returns the first error recorded anywhere on the builder or chain.
func (c *chain) pending() error {
	if c.b.declErr != nil {
		return c.b.declErr
	}
	return c.err
}

// liveIndex returns the position of the named axis among the live axes, or -1.
func (c *chain) liveIndex(name string) int {
	for i, d := range c.live {
		if d.Name() == name {
			return i
		}
	}
	return -1
}

// dimNodes resolves the dims' InputParam nodes in order — the on-the-fly
// resolution is a field read on the handle (the symbol table lookup happened
// at declaration).
func dimNodes(dims []Dim) []ir.Node {
	ns := make([]ir.Node, len(dims))
	for i, d := range dims {
		ns[i] = d.node
	}
	return ns
}

// Lambda-parameter name tables: a lone parameter is v, two are l, r; longer
// chains continue through the alphabet. The piped value (when present)
// leads the piped sequence.
var (
	rootMapNames  = []string{"l", "r", "s", "t", "u", "w", "x", "y", "z"}
	pipedMapNames = []string{"v", "w", "x", "y", "z"}
)

// mapParamName returns the doc-style name for parameter i of a map lambda
// with n parameters, or an error when the arity exceeds the v0 name table.
func mapParamName(piped bool, i, n int) (string, error) {
	if n == 1 {
		return "v", nil
	}
	names := rootMapNames
	if piped {
		names = pipedMapNames
	}
	if n > len(names) {
		return "", fmt.Errorf("map: %d operands exceed the v0 lambda-name table (%d)", n, len(names))
	}
	return names[i], nil
}

// mapFn is a lambda whose arity is pinned to the operand count the map will
// pass it.
type mapFn struct {
	call func(args []Expr) Expr
}

// arityErr builds the wrong-arity diagnostic; the piped value counts first.
func arityErr(got, want int) error {
	return fmt.Errorf("map: lambda arity mismatch: lambda takes %d argument(s), the map passes %d (piped value first, then operands)", got, want)
}

// coerceMapFn validates a map lambda's shape and pins its arity (D-10:
// wrong arity errors at the Map call). The oracle passes fixed-arity Go
// funcs (func(l, r Expr) Expr), which Go's type system will not convert to
// func(...Expr) Expr — the parameter is therefore any and the dispatch is a
// type switch over the supported shapes. A variadic lambda accepts any
// arity by construction and is invoked with exactly the expected arguments.
func coerceMapFn(fn any, want int) (mapFn, error) {
	switch f := fn.(type) {
	case func(...Expr) Expr:
		return mapFn{call: func(a []Expr) Expr { return f(a...) }}, nil
	case func(Expr) Expr:
		if want != 1 {
			return mapFn{}, arityErr(1, want)
		}
		return mapFn{call: func(a []Expr) Expr { return f(a[0]) }}, nil
	case func(Expr, Expr) Expr:
		if want != 2 {
			return mapFn{}, arityErr(2, want)
		}
		return mapFn{call: func(a []Expr) Expr { return f(a[0], a[1]) }}, nil
	case func(Expr, Expr, Expr) Expr:
		if want != 3 {
			return mapFn{}, arityErr(3, want)
		}
		return mapFn{call: func(a []Expr) Expr { return f(a[0], a[1], a[2]) }}, nil
	case func(Expr, Expr, Expr, Expr) Expr:
		if want != 4 {
			return mapFn{}, arityErr(4, want)
		}
		return mapFn{call: func(a []Expr) Expr { return f(a[0], a[1], a[2], a[3]) }}, nil
	default:
		return mapFn{}, fmt.Errorf("map: lambda must be func(...pkg.Expr) pkg.Expr or a fixed-arity func of 1-4 pkg.Expr parameters, got %T", fn)
	}
}

// checkOperand validates one structural tensor access and returns its
// paramRef node plus the tensor's scalar element type.
func (c *chain) checkOperand(op Operand) (ir.Node, ir.Type, error) {
	t := op.t
	if t.b != c.b || !c.b.tensorDeclared(t.name) {
		return nil, nil, fmt.Errorf("map: tensor %q is not declared in this builder", t.name)
	}
	if len(op.axes) != len(t.dims) {
		return nil, nil, fmt.Errorf("map: operand %s has %d subscripts for rank %d (structural indexing, syntax.md §4.1)",
			t.name, len(op.axes), len(t.dims))
	}
	for i, ax := range op.axes {
		if err := c.b.checkDim(ax, "map operand "+t.name); err != nil {
			return nil, nil, err
		}
		if ax.Name() != t.dims[i].Name() {
			return nil, nil, fmt.Errorf("map: operand %s subscript %d binds @%s, but dim %d of %s is @%s (structural indexing binds subscripts positionally)",
				t.name, i, ax.Name(), i, t.name, t.dims[i].Name())
		}
		if c.liveIndex(ax.Name()) < 0 {
			return nil, nil, fmt.Errorf("map: operand %s binds @%s, which is not a live axis at this point of the chain",
				t.name, ax.Name())
		}
	}
	return c.b.g.NewParamRef(origin, t.node), ir.InternScalar(t.elem), nil
}

// buildMapLambda validates the operands and the lambda, then invokes it with
// fresh typed lambda parameters. piped selects the piped-map convention:
// the incoming pipe is lambda argument 0 and the operands follow (§5.2).
// The map NODE is not constructed here — its `as` clause only exists at the
// following As rung; RootAs/AfterMap hold the pieces until then.
func (c *chain) buildMapLambda(ops []Operand, fn any, piped bool) (opNodes []ir.Node, params []ir.Node, body Expr, err error) {
	want := len(ops)
	if piped {
		want++
	}
	mf, err := coerceMapFn(fn, want)
	if err != nil {
		return nil, nil, Expr{}, err
	}
	elems := make([]ir.Type, 0, want)
	if piped {
		elems = append(elems, c.elem)
	}
	for _, op := range ops {
		node, elem, err := c.checkOperand(op)
		if err != nil {
			return nil, nil, Expr{}, err
		}
		opNodes = append(opNodes, node)
		elems = append(elems, elem)
	}
	params = make([]ir.Node, want)
	args := make([]Expr, want)
	for i := range params {
		name, err := mapParamName(piped, i, want)
		if err != nil {
			return nil, nil, Expr{}, err
		}
		params[i] = c.b.g.NewLambdaInputParam(origin, name, elems[i])
		args[i] = Expr{b: c.b, node: c.b.g.NewParamRef(origin, params[i]), typ: elems[i]}
	}
	body = mf.call(args)
	if body.err != nil {
		return nil, nil, Expr{}, body.err
	}
	if body.b != c.b {
		return nil, nil, Expr{}, fmt.Errorf("map: lambda body comes from a different builder")
	}
	return opNodes, params, body, nil
}

// buildReduce validates the homogeneous-reducer contract (tca.md §4.2: the
// initializer, the piped element type, both combiner parameter types, and
// the combiner result are one and the same scalar type), constructs the
// reduce stage over the current pipe, and drops the reduced-away live axis.
func (c *chain) buildReduce(init Expr, fn func(acc, v Expr) Expr) (ir.Node, ir.Type, error) {
	if fn == nil {
		return nil, nil, fmt.Errorf("reduce: nil combiner")
	}
	if err := firstErr(init); err != nil {
		return nil, nil, err
	}
	if init.b != c.b {
		return nil, nil, fmt.Errorf("reduce: init comes from a different builder")
	}
	if len(c.live) == 0 {
		return nil, nil, fmt.Errorf("reduce: no live axis left to reduce")
	}
	T := c.elem
	if init.typ != T {
		return nil, nil, fmt.Errorf("reduce: homogeneous reducer violated (tca.md §4.2): init has type %s, the piped value has element type %s",
			typeName(init.typ), typeName(T))
	}
	acc := c.b.g.NewLambdaInputParam(origin, "acc", T)
	now := c.b.g.NewLambdaInputParam(origin, "now", T)
	body := fn(Expr{b: c.b, node: c.b.g.NewParamRef(origin, acc), typ: T},
		Expr{b: c.b, node: c.b.g.NewParamRef(origin, now), typ: T})
	if body.err != nil {
		return nil, nil, body.err
	}
	if body.b != c.b {
		return nil, nil, fmt.Errorf("reduce: combiner body comes from a different builder")
	}
	if body.typ != T {
		return nil, nil, fmt.Errorf("reduce: homogeneous reducer violated (tca.md §4.2): combiner returns %s, want %s",
			typeName(body.typ), typeName(T))
	}
	node := c.b.g.NewReduce(origin, init.materialize(), c.pipe, ir.NewLambda([]ir.Node{acc, now}, body.materialize()))
	c.live = append([]Dim(nil), c.live[:len(c.live)-1]...)
	return node, T, nil
}

// checkAsPermutation validates a root map's as clause (tca.md §2.7 rule 2):
// a permutation of the iteration cluster's dims — each live axis exactly
// once, no more, no fewer.
func (c *chain) checkAsPermutation(dims []Dim, what string) error {
	if len(dims) != len(c.iters) {
		return fmt.Errorf("%s: as lists %d axes, the iteration cluster has %d", what, len(dims), len(c.iters))
	}
	seen := make(map[string]bool, len(dims))
	var dups []string
	for _, d := range dims {
		if err := c.b.checkDim(d, what+" as"); err != nil {
			return err
		}
		if seen[d.Name()] {
			dups = append(dups, "@"+d.Name())
		}
		seen[d.Name()] = true
	}
	var missing []string
	for _, it := range c.iters {
		if !seen[it.Name()] {
			missing = append(missing, "@"+it.Name())
		}
	}
	if len(dups) > 0 || len(missing) > 0 {
		return fmt.Errorf("%s: as must list every live axis exactly once; duplicated: %s; missing: %s",
			what, strings.Join(dups, ", "), strings.Join(missing, ", "))
	}
	return nil
}

// checkAsExact validates a piped map's as clause (syntax.md §5.2, tca.md
// §2.7 rule 5): exactly the producer's as with the reduced axes removed, in
// order — no permutation, no new axes, no resurrected reductions.
func (c *chain) checkAsExact(dims []Dim, what string) error {
	if len(dims) != len(c.live) {
		return fmt.Errorf("%s: as lists %d axes, %d are live at this point of the chain (producer's as minus reduced axes, in order)",
			what, len(dims), len(c.live))
	}
	for i, d := range dims {
		if err := c.b.checkDim(d, what+" as"); err != nil {
			return err
		}
		if d.Name() != c.live[i].Name() {
			return fmt.Errorf("%s: as position %d binds @%s, the incoming axis order requires @%s (fused maps cannot reorder axes in v0)",
				what, i, d.Name(), c.live[i].Name())
		}
	}
	return nil
}

// store validates a structural store and appends the statement. Every store
// axis must be live (the store executes at iteration points), and index i
// binds the tensor's dim i.
func (c *chain) store(t Tensor, axes ...Dim) (*Stored, error) {
	if t.b != c.b || !c.b.tensorDeclared(t.name) {
		return nil, fmt.Errorf("store: tensor %q is not declared in this builder", t.name)
	}
	if len(axes) != len(t.dims) {
		return nil, fmt.Errorf("store: %s has rank %d, got %d index axes", t.Name(), len(t.dims), len(axes))
	}
	for i, ax := range axes {
		if err := c.b.checkDim(ax, "store "+t.Name()); err != nil {
			return nil, err
		}
		if ax.Name() != t.dims[i].Name() {
			return nil, fmt.Errorf("store: index %d of %s binds @%s, but dim %d of %s is @%s",
				i, t.Name(), ax.Name(), i, t.Name(), t.dims[i].Name())
		}
		if c.liveIndex(ax.Name()) < 0 {
			return nil, fmt.Errorf("store: @%s is not a live axis at this point of the chain", ax.Name())
		}
	}
	node := c.b.g.NewStore(origin, t.node, dimNodes(axes), c.pipe)
	c.b.g.AddStatementAs(c.id, node)
	c.b.chain = nil
	c.b.noteOutput(t, node, c.retID)
	return &Stored{c: c}, nil
}

// StmtIter is the first chain rung: a fresh statement whose iteration
// cluster is fixed by Iter.
type StmtIter struct {
	c *chain
}

// Iter pins the root iteration cluster: the ordered dims the root map's as
// clause must permute. Dims are recorded in as-construction order.
func (s *StmtIter) Iter(dims ...Dim) *RootMap {
	c := s.c
	c.stage = "Iter"
	if len(dims) == 0 {
		c.deferErr(fmt.Errorf("iter: an iteration cluster needs at least one dim"))
	}
	for _, d := range dims {
		if err := c.b.checkDim(d, "iter"); err != nil {
			c.deferErr(err)
			continue
		}
		if c.liveIndex(d.Name()) >= 0 {
			c.deferErr(fmt.Errorf("iter: dim @%s is listed twice", d.Name()))
			continue
		}
		c.iters = append(c.iters, d)
		c.live = append(c.live, d)
	}
	return &RootMap{c: c}
}

// RootMap is the rung after Iter: only Map is possible.
type RootMap struct {
	c *chain
}

// Map constructs the root map's lambda over the operand accesses. The lambda
// arity must equal the operand count; each operand supplies one scalar
// lambda argument in source order (syntax.md §5.1). The map stage node is
// built at the following As rung, where the mandatory as clause arrives.
func (rm *RootMap) Map(ops []Operand, fn any) (*RootAs, error) {
	c := rm.c
	c.stage = "Map"
	if err := c.pending(); err != nil {
		return nil, err
	}
	if len(c.iters) == 0 {
		return nil, fmt.Errorf("map: the iteration cluster is empty")
	}
	opNodes, params, body, err := c.buildMapLambda(ops, fn, false)
	if err != nil {
		return nil, err
	}
	return &RootAs{c: c, opNodes: opNodes, params: params, body: body}, nil
}

// RootAs is the rung after a root map's lambda: only As is possible.
type RootAs struct {
	c       *chain
	opNodes []ir.Node
	params  []ir.Node
	body    Expr
}

// As pins the root map's mandatory as clause — a permutation of the
// iteration cluster's dims (tca.md §2.7 rules 1-2) — and wraps the map in
// its iteration cluster. The pipe now carries the mapped value with the
// as-order as its live axes.
func (ra *RootAs) As(dims ...Dim) *MidChain {
	c := ra.c
	c.stage = "As"
	if err := c.pending(); err == nil {
		c.deferErr(c.checkAsPermutation(dims, "root map"))
	}
	mapNode := c.b.g.NewMap(origin, ra.opNodes, ir.NewLambda(ra.params, ra.body.materialize()), dimNodes(dims))
	ic := c.b.g.NewIterCluster(origin, dimNodes(c.iters), mapNode)
	c.pipe = ic
	c.elem = ra.body.typ
	c.live = append([]Dim(nil), dims...)
	c.iters = nil
	return &MidChain{c: c}
}

// MidChain is the rung after a root map's as clause: only Reduce is
// possible (a root map must flow through a reduction before it can be
// stored or further mapped in v0).
type MidChain struct {
	c *chain
}

// Reduce appends a reduction over the last live axis (the last of as,
// tca.md §2.7 rule 3). The reducer is homogeneous (tca.md §4.2): init, pipe
// element, both combiner params, and the combiner result share one scalar
// type. The pipe now carries the reduced value with the reduced axis removed
// from the live set.
func (mc *MidChain) Reduce(init Expr, fn func(acc, v Expr) Expr) (*AfterReduce, error) {
	c := mc.c
	c.stage = "Reduce"
	if err := c.pending(); err != nil {
		return nil, err
	}
	node, elem, err := c.buildReduce(init, fn)
	if err != nil {
		return nil, err
	}
	c.pipe = node
	c.elem = elem
	return &AfterReduce{c: c}, nil
}

// AfterReduce is the rung after a reduction: Reduce (chained reductions,
// §9.5), Map (piped map), or Store (direct store after reduce, §9.8) — and
// nothing else; in particular there is no Iter.
type AfterReduce struct {
	c *chain
}

// Reduce appends a chained reduction over the new last live axis.
func (ar *AfterReduce) Reduce(init Expr, fn func(acc, v Expr) Expr) (*AfterReduce, error) {
	c := ar.c
	c.stage = "Reduce"
	if err := c.pending(); err != nil {
		return nil, err
	}
	node, elem, err := c.buildReduce(init, fn)
	if err != nil {
		return nil, err
	}
	c.pipe = node
	c.elem = elem
	return &AfterReduce{c: c}, nil
}

// Map constructs a piped map: the previous stage's value is lambda argument
// 0 and the explicit operands follow it (syntax.md §5.2). The lambda arity
// must be 1 + len(ops). The piped map's as clause arrives at the following
// As rung.
func (ar *AfterReduce) Map(ops []Operand, fn any) (*AfterMap, error) {
	c := ar.c
	c.stage = "Map"
	if err := c.pending(); err != nil {
		return nil, err
	}
	opNodes, params, body, err := c.buildMapLambda(ops, fn, true)
	if err != nil {
		return nil, err
	}
	return &AfterMap{c: c, opNodes: opNodes, params: params, body: body}, nil
}

// Store appends a structural store of the reduced value and terminates the
// chain.
func (ar *AfterReduce) Store(t Tensor, axes ...Dim) (*Stored, error) {
	ar.c.stage = "Store"
	if err := ar.c.pending(); err != nil {
		return nil, err
	}
	return ar.c.store(t, axes...)
}

// AfterMap is the rung after a piped map's lambda: only As is possible.
type AfterMap struct {
	c       *chain
	opNodes []ir.Node
	params  []ir.Node
	body    Expr
}

// As pins the piped map's mandatory as clause: exactly the producer's axis
// order minus the reduced axes (syntax.md §5.2, tca.md §2.7 rule 5). The
// piped map renders as an empty iteration cluster, iter() -> map(...)
// (doc/tca.md §5.1).
func (am *AfterMap) As(dims ...Dim) *Tail {
	c := am.c
	c.stage = "As"
	if err := c.pending(); err == nil {
		c.deferErr(c.checkAsExact(dims, "piped map"))
	}
	ops := make([]ir.Node, 0, len(am.opNodes)+1)
	ops = append(ops, c.pipe)
	ops = append(ops, am.opNodes...)
	mapNode := c.b.g.NewMap(origin, ops, ir.NewLambda(am.params, am.body.materialize()), dimNodes(dims))
	ic := c.b.g.NewIterCluster(origin, nil, mapNode)
	c.pipe = ic
	c.elem = am.body.typ
	c.live = append([]Dim(nil), dims...)
	return &Tail{c: c}
}

// Tail is the rung after a piped map's as clause: only Store is possible.
type Tail struct {
	c *chain
}

// Store appends a structural store of the mapped value and terminates the
// chain.
func (t *Tail) Store(tensor Tensor, axes ...Dim) (*Stored, error) {
	t.c.stage = "Store"
	if err := t.c.pending(); err != nil {
		return nil, err
	}
	return t.c.store(tensor, axes...)
}

// Stored is the terminal rung: the chain is complete, the statement has been
// appended, and the stored tensor is registered as a kernel output binding
// whose return statement Build emits automatically.
type Stored struct {
	c *chain
}
