package ir

import "slices"

// Span is a source location range. It is the zero value throughout Phase 1:
// there is no frontend yet, so nodes carry no source spans (IR-06).
type Span struct {
	File           string
	Line, Col      int
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
	ssa           *SSAVar // this node's SSA cell; allocated at construction (1:1, ID == node ID)
	erased        bool    // tombstone set by Erase; IDs are never reused (IR-05)
	surplus       int     // operands supplied beyond the fixed-arity slot count at construction; the verifier flags > 0 (fail-closed, WR-01)
	graph         *Graph  // owning graph, set at construction; lets Erase unlink from the block
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

// paramRole discriminates kernel I/O symbols.
type paramRole int

const (
	roleDim paramRole = iota
	roleTensor
	roleScalar
)

// Param is a kernel I/O symbol: a dim, tensor, or scalar parameter. Params
// are immutable after creation and are owned by the Graph's param list.
type Param struct {
	name string
	typ  Type // nil for dim params (dims are not typed values in v0)
	role paramRole
}

// Name returns the parameter's name.
func (p Param) Name() string { return p.name }

// Type returns the parameter's interned type; nil for dim parameters.
func (p Param) Type() Type { return p.typ }

// Statement is one ordered entry in the kernel's single basic block (IR-02):
// an immutable SSA binding of a value-producing RHS. The ID is stable across
// passes so pass-to-pass diffs can track a statement (D-04). The def entry
// point stays Node-typed (not Value-typed): every exported constructor
// returns Node, which never statically satisfies Value (Node + IsNone), so a
// Value-typed AddStatement would force casts at six out-of-package call sites
// and in irx_test — the NoneValue classification lives solely in Value().
type Statement struct {
	id    uint64
	graph *Graph  // owning graph, set at construction; lets the printer detect statement-chain boundaries
	def   SSADef  // reserved SSA-name handle; zero value = unwired until a pass wires it
	rhs   Node    // the statement's root node (NOT Value — see the contract note above)
}

// ID returns the statement's stable per-graph integer ID.
func (s *Statement) ID() uint64 { return s.id }

// RHS returns the statement's value-producing right-hand side.
func (s *Statement) RHS() Node { return s.rhs }

// Value returns the statement's RHS viewed as a Value — the single
// classification accessor for NoneValue semantics. A nil rhs yields a nil
// Value; otherwise the rhs asserts to Value, which is total for in-repo nodes
// (every concrete node embeds *Base, promoting IsNone), so there is no panic
// path (T-1.1-03).
func (s *Statement) Value() Value {
	if s.rhs == nil {
		return nil
	}
	return s.rhs.(Value)
}

// Def returns the statement's SSA-name handle; IsInvalid reports whether it
// is still unwired.
func (s *Statement) Def() SSADef { return s.def }

// Block is the kernel's single basic block (IR-02): an ordered statement
// list. v0 has no control flow, so a Graph holds exactly one Block.
type Block struct {
	statements []*Statement
}

// Statements returns a copy of the block's statements in insertion order.
func (b *Block) Statements() []*Statement { return append([]*Statement(nil), b.statements...) }

// Graph is one kernel/function's IR: a name, the kernel I/O symbols, and the
// single basic block (IR-02). Node and statement IDs come only from NewID:
// per-graph monotonic, starting at 1, never reused even across Erase (IR-05).
type Graph struct {
	name   string
	params []Param
	block  *Block
	nextID uint64 // next ID to hand out; monotonic
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

// Params returns a copy of the graph's kernel I/O symbols in declaration order.
func (g *Graph) Params() []Param { return append([]Param(nil), g.params...) }

// Block returns the graph's single basic block.
func (g *Graph) Block() *Block { return g.block }

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
