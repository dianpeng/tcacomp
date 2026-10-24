package pkg

import (
	"fmt"
	"io"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// This file is the public builder root (FE-01, D-08): typed first-class
// handles for dims, tensors, and scalar expressions, constructed straight
// into the single-tier IR through the exported ir constructors (no MutContext
// — construction exemption, RESEARCH Pattern 2) with origin "builder" so
// Loc.PassOrigin / LastTouchedBy are meaningful from birth (D-17).
//
// The builder's declaration maps are the on-the-fly symbol resolution the
// Phase-1 review mandates: a handle carries its RESOLVED InputParam node, so
// a freshly built graph contains zero name-keyed references — every use site
// materializes its own SSAUse record against the node (review P1R-01/03).
//
// Eager checks (D-10): declaration-time violations (duplicate names, negative
// extents) and chain-rung violations (arity, axis coverage, homogeneous
// reducers, structural subscripts) return errors at the call site. The
// pinned handle-creating signatures (Dim/Tensor/ScalarParam/Iter/As) cannot
// return errors, so those violations are recorded and surfaced by the next
// error-returning rung or by Build — the library never panics.

// origin stamps every node the builder constructs.
const origin = "builder"

// Builder is the public TCA graph-builder root (D-08). Handles produced by a
// builder carry it, so passing a handle from another builder is caught by
// the eager checks instead of corrupting the graph.
type Builder struct {
	g       *ir.Graph
	env     *ir.Env           // construction-time side-effect environment: one ImpactBy chain per tensor (effect.go)
	dims    map[string]Dim    // declaration-time symbol table: dim name -> handle
	tensors map[string]Tensor // declaration-time symbol table: tensor name -> handle
	scalars map[string]ir.ElemType
	outputs []output // kernel output bindings, in first-store order
	chain   *chain   // statement chain under construction, nil when none
	final   bool     // Build has run; no new statements allowed
	declErr error    // first declaration-time violation (recorded, deferred)
}

// output is one declared kernel output binding: the tensor name (debug
// payload; the graph's identity for the tensor is the store's target use),
// the tensor's resolved InputParam node (the end node's ImpactBy wiring
// names the chain per tensor, R2), the store node feeding its return, and
// the reserved end-node statement ID.
type output struct {
	name   string
	tensor ir.Node
	store  ir.Node
	retID  uint64
}

// Dim is a first-class handle for a declared shape dimension (D-08). Extent
// 0 means dynamic/symbolic (the canonical print shows a bare @name); a
// positive extent is a compile-time constant. The handle carries the
// resolved InputParam node — the dim's SSA def — plus its declared name for
// diagnostics (names are debug payload only, review P1R-01).
type Dim struct {
	b      *Builder
	node   ir.Node // resolved InputParam node (role RoleDim); the dim's SSADef
	name   string  // declared name, debug payload
	extent int
}

// Name returns the dimension's declared name.
func (d Dim) Name() string { return d.name }

// Extent returns the declared extent; 0 means dynamic.
func (d Dim) Extent() int { return d.extent }

// Tensor is a first-class handle for a declared tensor parameter (D-08):
// the kernel I/O symbol plus its ordered shape dims. The handle carries the
// resolved InputParam node (the tensor's SSA def) and its declared name for
// diagnostics.
type Tensor struct {
	b    *Builder
	node ir.Node // resolved InputParam node (role RoleTensor)
	name string
	dims []Dim
	elem ir.ElemType
}

// Name returns the tensor's declared name.
func (t Tensor) Name() string { return t.name }

// Elem returns the tensor's element type.
func (t Tensor) Elem() ir.ElemType { return t.elem }

// Rank returns the tensor's number of shape dimensions.
func (t Tensor) Rank() int { return len(t.dims) }

// Operand is one structural tensor access with its axis binding baked in
// (D-08): At(lhs, m, n) is the map operand lhs[@m][@n].
type Operand struct {
	t    Tensor
	axes []Dim
}

// Tensor returns the operand's tensor.
func (o Operand) Tensor() Tensor { return o.t }

// Axes returns a copy of the operand's bound axes in subscript order.
func (o Operand) Axes() []Dim { return append([]Dim(nil), o.axes...) }

// At binds axes to a tensor as one structural access (D-08). The binding is
// validated eagerly by the consuming rung: subscript count must equal the
// tensor rank and subscript i must bind the tensor's dim i (structural
// indexing, syntax-design.md §4.4 rule 9).
func At(t Tensor, axes ...Dim) Operand {
	return Operand{t: t, axes: append([]Dim(nil), axes...)}
}

// NewBuilder creates an empty kernel builder with the given graph name. The
// graph starts as a Level-1 sea of nodes (no CFG, no statements); Build
// schedules it into the single-block CFG at the end.
func NewBuilder(name string) *Builder {
	return &Builder{
		g:       ir.NewGraph(name),
		env:     ir.NewEnv(),
		dims:    make(map[string]Dim),
		tensors: make(map[string]Tensor),
		scalars: make(map[string]ir.ElemType),
	}
}

// noteDeclErr records the first declaration-time violation; rungs and Build
// surface it before doing anything else.
func (b *Builder) noteDeclErr(err error) {
	if err != nil && b.declErr == nil {
		b.declErr = err
	}
}

func (b *Builder) dimDeclared(name string) bool {
	_, ok := b.dims[name]
	return ok
}

func (b *Builder) tensorDeclared(name string) bool {
	_, ok := b.tensors[name]
	return ok
}

// checkDim validates a dim handle: it must belong to this builder and be
// declared.
func (b *Builder) checkDim(d Dim, what string) error {
	if d.b != b || !b.dimDeclared(d.name) {
		return fmt.Errorf("%s: dim @%q is not declared in this builder", what, d.name)
	}
	return nil
}

// Dim declares a shape dimension. Extent 0 declares a dynamic/symbolic
// runtime dim; a positive extent is a compile-time constant. Duplicate names
// and negative extents are recorded and surfaced by the next error-returning
// call (the pinned signature carries no error return).
func (b *Builder) Dim(name string, extent int) Dim {
	if b.final {
		b.noteDeclErr(fmt.Errorf("dim %s: builder already finalized by Build", name))
	}
	if extent < 0 {
		b.noteDeclErr(fmt.Errorf("dim %s: extent must be >= 0 (0 = dynamic), got %d", name, extent))
	}
	if _, taken := b.dims[name]; taken {
		b.noteDeclErr(fmt.Errorf("dim %s: already declared", name))
	}
	d := Dim{b: b, node: b.g.NewInputParam(origin, ir.RoleDim, name, nil), name: name, extent: extent}
	b.dims[name] = d
	return d
}

// Tensor declares a tensor parameter with an ordered shape. Dynamic dims
// (extent 0) contribute a symbolic dim ref to the tensor's type; static dims
// contribute their constant extent. Shape violations are caught by the eager
// checks on the handles, not here.
func (b *Builder) Tensor(name string, elem ir.ElemType, dims ...Dim) Tensor {
	if b.final {
		b.noteDeclErr(fmt.Errorf("tensor %s: builder already finalized by Build", name))
	}
	if _, taken := b.tensors[name]; taken {
		b.noteDeclErr(fmt.Errorf("tensor %s: already declared", name))
	}
	refs := make([]ir.DimRef, len(dims))
	tdims := make([]Dim, len(dims))
	for i, d := range dims {
		if err := b.checkDim(d, "tensor "+name); err != nil {
			b.noteDeclErr(err)
		}
		ref := ir.DimRef{Name: d.Name()}
		if d.extent > 0 {
			ref.Name = ""
			ref.Extent = uint64(d.extent)
		}
		refs[i] = ref
		tdims[i] = d
	}
	t := Tensor{b: b, node: b.g.NewInputParam(origin, ir.RoleTensor, name, ir.InternTensor(elem, refs)), name: name, dims: tdims, elem: elem}
	b.tensors[name] = t
	return t
}

// ScalarParam declares an explicit scalar dependency and returns its typed
// expression handle. The [scale] capture of syntax-design.md §4.5 rule 14 becomes an SSA
// operand of the expression graph through this reference. The handle is
// lazy: its paramRef node materializes on first use (see Expr.materialize)
// so scalar declarations never consume graph IDs ahead of the
// statement-ID reservation in Stmt.
func (b *Builder) ScalarParam(name string, t ir.ElemType) Expr {
	if b.final {
		b.noteDeclErr(fmt.Errorf("scalar %s: builder already finalized by Build", name))
	}
	if _, taken := b.scalars[name]; taken {
		b.noteDeclErr(fmt.Errorf("scalar %s: already declared", name))
	}
	typ := ir.InternScalar(t)
	b.scalars[name] = t
	return Expr{b: b, scalar: b.g.NewInputParam(origin, ir.RoleScalar, name, typ), typ: typ}
}

// Stmt starts a new statement chain and returns its first typestate rung.
// Two IDs are reserved now — the statement's and the end node's — before any
// of the chain's nodes exist, so the canonical print shows dense statement
// numbers (see ir.ReserveStatementID): the doc-pinned oracle expects
// `statement #8` for the chain and `statement #9` for the end node (IDs 1-7
// are the kernel params, 10+ the chain's nodes — the reservation lands ahead
// of them). The IDs are pinned on the store/end nodes (ir.PinStatementID)
// and consumed by Materialize at Build time. The end-node ID is consumed by
// the FIRST chain that stores a new output tensor (Build wires one end node
// per graph, R2); chains storing further tensors keep their own statement
// IDs but leave later end-node reservations unused (IDs may skip). Starting
// a new chain abandons any unfinished one; Build rejects a builder left
// with an unfinished chain.
func (b *Builder) Stmt() *StmtIter {
	if b.final {
		b.noteDeclErr(fmt.Errorf("stmt: builder already finalized by Build"))
	}
	id := b.g.ReserveStatementID()
	retID := b.g.ReserveStatementID()
	b.chain = &chain{b: b, id: id, retID: retID}
	return &StmtIter{c: b.chain}
}

// Build finalizes the graph and returns it, verified exclusively through
// the normal pass-manager path (D-10). The two-level flow
// (quick-261017-vdj), with the one-end-node model (quick-261018-bl0 task 2,
// R2): a SINGLE kernel end node is emitted over all declared output
// bindings — the stores in first-store order as its returned values, and
// every output tensor's final chain state via the builder env
// (ir.NewReturnChained) — pinned with the FIRST output binding's reserved
// end-node ID (later bindings' reserved IDs stay unused; IDs may skip). The
// building-phase DCE then runs over the Level-1 sea (review P1R-08: dead
// side-effect-free chains are marked and wiped before any pass sees the
// graph — debris from abandoned chains included), Materialize schedules the
// sea into the single-block CFG (one statement per live effectful node
// reachable from the sea roots, in allocation order), and then a
// opt.Manager with an EMPTY pass list carrying the VerifyEach
// instrumentation runs the graph.
// VerifyEach's AfterPipeline hook runs the graph verifier even with zero
// passes and Manager.Run surfaces its error, so Build returns verification
// failures as Build errors — never a half-built graph, and never by calling
// the verifier directly from pkg (that would bypass the single authoritative
// path).
func (b *Builder) Build() (*ir.Graph, error) {
	if err := b.declErr; err != nil {
		return nil, err
	}
	if b.chain != nil {
		return nil, fmt.Errorf("build: statement chain (reserved statement id %d) was never stored", b.chain.id)
	}
	b.final = true
	if len(b.outputs) > 0 {
		tensors := make([]ir.Node, 0, len(b.outputs))
		stores := make([]ir.Node, 0, len(b.outputs))
		for _, o := range b.outputs {
			tensors = append(tensors, o.tensor)
			stores = append(stores, o.store)
		}
		ret := b.g.NewReturnChained(origin, b.env, tensors, stores...)
		b.g.PinStatementID(ret, b.outputs[0].retID)
	}
	if err := ir.BuildingDCE(b.g); err != nil {
		return nil, err
	}
	if err := b.g.Materialize(); err != nil {
		return nil, err
	}
	pm := opt.NewManager(nil)
	pm.AddInstrumentation(opt.NewVerifyEach(io.Discard))
	if err := pm.Run(b.g); err != nil {
		return nil, err
	}
	return b.g, nil
}

// noteOutput records a store to t as an output binding, consuming the
// chain's reserved end-node statement ID. Later stores to the same tensor
// update the binding the end node reads (last store wins — the chain head
// the ImpactBy wiring names) without consuming another ID.
func (b *Builder) noteOutput(t Tensor, store ir.Node, retID uint64) {
	for i := range b.outputs {
		if b.outputs[i].name == t.name {
			b.outputs[i].store = store
			return
		}
	}
	b.outputs = append(b.outputs, output{name: t.name, tensor: t.node, store: store, retID: retID})
}
