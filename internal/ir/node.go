package ir

import "slices"

// Span is a source location range. It is the zero value throughout Phase 1:
// there is no frontend yet, so nodes carry no source spans (IR-06).
type Span struct {
	File            string
	Line, Col       int
	EndLine, EndCol int
}

// Loc is the minimal location record every node carries (IR-06): the
// frontend span (zero in Phase 1) and the pass that originated the node.
type Loc struct {
	Span       Span
	PassOrigin string
}

// Base is the shared cell embedded in every IR node (D-15). It holds the
// node's stable ID, kind, location, last-touched-by stamp, and the node's
// SSAVar — the sole pointee of the SSA handle layer (directive 2): the uses
// that point AT this node as their def live in Base.ssa's ref list.
//
// Base exposes no public mutators (D-15): the only sanctioned mutation path
// is the MutContext choke-point API in mutate.go (IR-04).
type Base struct {
	id            uint64
	kind          Kind
	loc           Loc
	lastTouchedBy string
	ssa           *SSAVar    // this node's SSA cell; allocated at construction (1:1, ID == node ID)
	erased        bool       // tombstone set by Erase; IDs are never reused (IR-05)
	surplus       int        // operands supplied beyond the fixed-arity slot count at construction; the verifier flags > 0 (fail-closed, WR-01)
	graph         *Graph     // owning graph, set at construction; lets Erase unlink from the block
	stmt          *Statement // the statement whose RHS root this node is, nil for non-root nodes; lets Erase unlink in O(1)
	stmtID        uint64     // builder-pinned statement ID for Materialize (PinStatementID); 0 = Materialize draws from the counter
}

// BasePtr returns the shared cell, implementing Node.
func (b *Base) BasePtr() *Base { return b }

// KindOf returns the node's kind, implementing Node.
func (b *Base) KindOf() Kind { return b.kind }

// ID returns the node's stable per-graph integer ID (IR-05).
func (b *Base) ID() uint64 { return b.id }

// LastTouchedBy returns the name of the last pass that mutated this node
// through a choke-point mutation (D-17); empty for never-mutated nodes.
func (b *Base) LastTouchedBy() string { return b.lastTouchedBy }

// Loc returns the node's location record (IR-06).
func (b *Base) Loc() Loc { return b.loc }

// Node is the root IR node interface (D-15, go/ast shape): every concrete
// node family satisfies it through its embedded *Base.
type Node interface {
	BasePtr() *Base
	KindOf() Kind
}

// ScalarExprNode is the marker interface for scalar-expression node kinds
// (const, param, binop, unaryop, select, cast, call). Passes type-switch
// over families rather than enumerating kinds.
type ScalarExprNode interface {
	Node
	scalarExprNode()
}

// Value is the RHS-capable node interface (directive 2): every statement root
// that produces a referenceable value implements it. A none Value is a Value
// but just nothing — the return root overrides IsNone to report exactly that,
// so it can sit in a statement RHS slot while binding no SSA name. Every
// concrete node embeds *Base, which promotes IsNone() == false; only
// returnNode overrides it. Node stays the universal root (D-15).
type Value interface {
	Node
	IsNone() bool
}

// IsNone reports whether the node is the none Value. *Base promotes this to
// every node kind; only returnNode overrides it (true).
func (b *Base) IsNone() bool { return false }

// ParamRole discriminates InputParam nodes. Kernel roles (RoleDim,
// RoleTensor, RoleScalar) are owned by the Graph's param list; RoleLambda
// marks lambda-bound parameters, which are graph-external by design (the
// review's IR modeling: lambda parameters are InputParams, inlined at
// lowering).
type ParamRole int

const (
	RoleDim    ParamRole = iota // shape dimension (runtime shape symbol; untyped)
	RoleTensor                  // tensor parameter (interned tensor type)
	RoleScalar                  // scalar parameter (interned scalar type)
	RoleLambda                  // lambda-bound parameter (map/reduce lambda argument)
)

// Graph is one kernel/function's IR: a name, the kernel input-parameter
// nodes, and — once Materialize runs (Level 2) — a single-block CFG
// (IR-02). Between construction and Materialize the graph is a pure sea of
// nodes: cfg is nil, Block()/CFG() return nil (T-quick-02 nil-safety), and
// ordering of side effects rides the explicit ImpactBy effect chains (see
// effect.go). Node and statement IDs come only from NewID: per-graph
// monotonic, starting at 1, never reused even across Erase (IR-05). Every
// reference anywhere in the graph is an SSAUse; every definition is an
// SSADef — names survive only as InputParam debug payload.
type Graph struct {
	name         string
	params       []Node // InputParam nodes (kernel roles only), in declaration order
	cfg          *CFG   // Level 2 only: nil from NewGraph until Materialize
	nextID       uint64 // next ID to hand out; monotonic
	nodes        []Node // every node ever allocated, in allocation order (never shrinks; erased nodes stay tombstoned) — the sea's sweep registry
	effectRoots  []Node // effectful nodes (stores, tile stores, returns), in allocation order — Level-1 Walk/Verify/DCE roots
	materialized bool   // Materialize has run; cfg and the statement list exist
}

// NewID returns the graph's next monotonic ID. This is the ONLY source of
// node and statement IDs; IDs are never reused (IR-05).
func (g *Graph) NewID() uint64 {
	id := g.nextID
	g.nextID++
	return id
}

// Name returns the graph's name.
func (g *Graph) Name() string { return g.name }

// Params returns a copy of the graph's kernel input-parameter nodes in
// declaration order. There is deliberately NO name→param lookup: names are
// debug payload, and resolution happens at construction time (review P1R-01).
func (g *Graph) Params() []Node { return append([]Node(nil), g.params...) }

// Block returns the CFG's entry basic block — the graph's single block in v0.
// It is nil until Materialize runs (Level 1 is a statement-free sea of nodes,
// T-quick-02 nil-safety): construction and passes must not consult the block.
func (g *Graph) Block() *BasicBlock {
	if g == nil || g.cfg == nil {
		return nil
	}
	return g.cfg.entry
}

// CFG returns the graph's control-flow graph (exactly one block in v0), or nil
// before Materialize.
func (g *Graph) CFG() *CFG { return g.cfg }

// Materialized reports whether Materialize has run: the graph is Level 2 with
// a CFG and a statement list. Codegen-adjacent surfaces (print, DOT) require
// it; passes operate on the sea and never need it.
func (g *Graph) Materialized() bool { return g != nil && g.materialized }

// nodeImpl is the internal interface every concrete node type satisfies:
// the public Node surface plus operand-slot access for chain maintenance.
// All mutation and verification flows through these hooks, never through
// concrete node fields. Operand slots hold SSAUse value-type handles; a slot
// is wired iff its handle's record is non-nil with a non-nil target.
type nodeImpl interface {
	Node
	forEachOperand(fn func(SSAUse)) // visit each operand slot in canonical order
	addSlot() SSAUse                // append one operand slot and return its (zero) handle
	setSlot(i int, u SSAUse)        // write slot i (i < slot count); construction wiring only
	removeSlot(u SSAUse) bool       // drop u's slot from the operand storage (record identity)
}

// operandList is embedded by every concrete node type to provide the
// operand-slot half of nodeImpl uniformly. The slice stores SSAUse handles BY
// VALUE; each handle points at a heap ref record shared with the def's SSAVar
// ref list, so slice growth/removal moves handle copies only — the records
// (and the chain state they carry) never move and never dangle (CR-01/CR-02).
// A zero handle (rec == nil) is an unwired slot.
type operandList struct {
	ops []SSAUse // one SSAUse per operand, in canonical operand order
}

func (l *operandList) forEachOperand(fn func(SSAUse)) {
	for _, u := range l.ops {
		fn(u)
	}
}

func (l *operandList) addSlot() SSAUse {
	l.ops = append(l.ops, SSAUse{}) // a fresh slot is a zero (unwired) handle
	return l.ops[len(l.ops)-1]
}

func (l *operandList) setSlot(i int, u SSAUse) {
	l.ops[i] = u
}

func (l *operandList) removeSlot(u SSAUse) bool {
	if u.rec == nil {
		return false
	}
	for i, s := range l.ops {
		if s.rec == u.rec { // record-identity comparison
			l.ops = slices.Delete(l.ops, i, i+1) // drops a handle copy; the record stays valid
			return true
		}
	}
	return false
}
