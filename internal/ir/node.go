package ir

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

// Base is the shared SSAVar cell embedded in every IR node (design.md:
// "SSADef and SSAUse shared an underlying pointer"). It holds the node's
// stable ID, kind, location, last-touched-by stamp, and the head of the
// node's intrusive use-list — the uses that point AT this node as their def.
//
// Base exposes no public mutators (D-15): the only sanctioned mutation path
// is the MutContext choke-point API in mutate.go (IR-04).
type Base struct {
	id            uint64
	kind          Kind
	loc           Loc
	lastTouchedBy string
	firstUse      *Use // head of this def's intrusive use-list (D-16)
	erased        bool // tombstone set by Erase; IDs are never reused (IR-05)
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

// Value aliases Node: in the single-tier IR an SSA value IS an IR node —
// the def-producing node is the SSAVar cell. Type alias, not a wrapper, so
// Value == Node by type identity (waves 2-3 may rely on this).
type Value = Node

// SSADef aliases Node for the same reason: a statement's def is a node.
type SSADef = Node

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
	typ  *Type // nil for dim params (dims are not typed values in v0)
	role paramRole
}

// Name returns the parameter's name.
func (p Param) Name() string { return p.name }

// Type returns the parameter's interned type; nil for dim parameters.
func (p Param) Type() *Type { return p.typ }

// Statement is one ordered entry in the kernel's single basic block (IR-02):
// an immutable SSA binding of a value-producing RHS. The ID is stable across
// passes so pass-to-pass diffs can track a statement (D-04).
type Statement struct {
	id  uint64
	def SSADef // reserved SSA-name cell; nil until a pass wires it
	rhs Value
}

// ID returns the statement's stable per-graph integer ID.
func (s *Statement) ID() uint64 { return s.id }

// RHS returns the statement's value-producing right-hand side.
func (s *Statement) RHS() Value { return s.rhs }

// Def returns the statement's SSA-name cell, or nil if unwired.
func (s *Statement) Def() SSADef { return s.def }

// Block is the kernel's single basic block (IR-02): an ordered statement
// list. v0 has no control flow, so a Graph holds exactly one Block.
type Block struct {
	statements []*Statement
}

// Statements returns the block's statements in insertion order.
func (b *Block) Statements() []*Statement { return b.statements }

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
// concrete node fields.
type nodeImpl interface {
	Node
	forEachOperand(fn func(*Use)) // visit each operand slot in canonical order
	addSlot() *Use                // append one operand slot and return it
}

// operandList is embedded by every concrete node type to provide the
// operand-slot half of nodeImpl uniformly.
type operandList struct {
	ops []Use // one Use per operand, in canonical operand order
}

func (l *operandList) forEachOperand(fn func(*Use)) {
	for i := range l.ops {
		fn(&l.ops[i])
	}
}

func (l *operandList) addSlot() *Use {
	l.ops = append(l.ops, Use{})
	return &l.ops[len(l.ops)-1]
}
