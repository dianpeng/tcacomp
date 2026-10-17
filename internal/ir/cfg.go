package ir

// This file is the control-flow shape (review P1R-02/P1R-06): CFG →
// BasicBlock → Statement. v0 has exactly one basic block per kernel and no
// control flow (IR-02), so a CFG holds one entry block; the struct exists so
// multi-block extension later is purely additive.
//
// Statements live in an INTRUSIVE doubly-linked list owned by the block:
// removing a statement given its pointer is O(1) — no position search, no
// slice reindex (the review's explicit demand). Iteration is
//
//	for s := bb.First(); s != nil; s = s.Next() { ... }
//
// Statement identity is stable across passes (D-04): the ID comes from
// the graph's single monotonic counter at Materialize time (or from a
// builder-reserved ID, see ReserveStatementID).

// CFG is the kernel's control-flow graph. v0: exactly one basic block.
type CFG struct {
	entry *BasicBlock
}

// Entry returns the CFG's entry block — the only block in v0.
func (c *CFG) Entry() *BasicBlock { return c.entry }

// BasicBlock is one ordered statement list (IR-02). v0 kernels have exactly
// one block and no control flow; the intrusive list is the block's storage.
type BasicBlock struct {
	graph *Graph     // owning graph, set at construction
	head  *Statement // first statement, nil when empty
	tail  *Statement // last statement, nil when empty
}

// Graph returns the block's owning graph.
func (bb *BasicBlock) Graph() *Graph { return bb.graph }

// First returns the block's first statement, or nil when empty.
func (bb *BasicBlock) First() *Statement { return bb.head }

// Last returns the block's last statement, or nil when empty.
func (bb *BasicBlock) Last() *Statement { return bb.tail }

// Append links s at the list tail in O(1). s must not already be linked.
func (bb *BasicBlock) Append(s *Statement) {
	if s == nil || s.prev != nil || s.next != nil || bb.head == s {
		return
	}
	s.graph = bb.graph
	s.prev = bb.tail
	if bb.tail != nil {
		bb.tail.next = s
	} else {
		bb.head = s
	}
	bb.tail = s
}

// InsertBefore links s immediately before at in O(1). Both must be non-nil
// and s must not already be linked.
func (bb *BasicBlock) InsertBefore(s, at *Statement) {
	if s == nil || at == nil || s.prev != nil || s.next != nil || bb.head == s {
		return
	}
	s.graph = bb.graph
	s.next = at
	s.prev = at.prev
	if at.prev != nil {
		at.prev.next = s
	} else {
		bb.head = s
	}
	at.prev = s
}

// InsertAfter links s immediately after at in O(1). Both must be non-nil and
// s must not already be linked.
func (bb *BasicBlock) InsertAfter(s, at *Statement) {
	if s == nil || at == nil || s.prev != nil || s.next != nil || bb.head == s {
		return
	}
	s.graph = bb.graph
	s.prev = at
	s.next = at.next
	if at.next != nil {
		at.next.prev = s
	} else {
		bb.tail = s
	}
	at.next = s
}

// Remove unlinks s from the list in O(1) via its intrusive prev/next links —
// no position search, no slice reindex, no touch of any other statement's
// identity. It returns false when s is nil or not linked in this block.
func (bb *BasicBlock) Remove(s *Statement) bool {
	if s == nil || (s.prev == nil && s.next == nil && bb.head != s) {
		return false
	}
	if s.prev != nil {
		s.prev.next = s.next
	} else {
		bb.head = s.next
	}
	if s.next != nil {
		s.next.prev = s.prev
	} else {
		bb.tail = s.prev
	}
	s.prev, s.next = nil, nil
	if s.rhs != nil {
		s.rhs.BasePtr().stmt = nil
	}
	return true
}

// StmtKind classifies a statement per the review's building-phase model: a
// statement cluster either binds a new local (assign: has LHS SSADef + RHS)
// or is a purely side-effecting operation (store to observable memory,
// kernel return — RHS carries the effect and binds no new local name).
type StmtKind int

const (
	// StmtAssign binds a new local: the statement's def is wired to the
	// value its RHS produces.
	StmtAssign StmtKind = iota
	// StmtEffect is a pure side-effect operation: store to observable
	// memory, or a kernel return. It still occupies the block list; its
	// def stays unwired (it binds no new local).
	StmtEffect
)

// Statement is one ordered entry in its basic block's intrusive list (IR-02):
// an SSA binding of a value-producing RHS (StmtAssign) or a pure side-effect
// operation (StmtEffect). The ID is stable across passes so pass-to-pass
// diffs can track a statement (D-04). The def entry point stays Node-typed
// (not Value-typed): every exported constructor returns Node, which never
// statically satisfies Value (Node + IsNone), so a Value-typed statement
// entry point would force casts at out-of-package call sites — the NoneValue
// classification lives solely in Value(). Statements are created only by
// Materialize (schedule.go).
type Statement struct {
	id    uint64
	graph *Graph     // owning graph, set at link time; lets the printer detect chain boundaries
	prev  *Statement // intrusive list links: previous statement in the block (nil at head)
	next  *Statement // intrusive list links: next statement in the block (nil at tail)
	kind  StmtKind   // assign (binds a local) vs pure side-effect
	def   SSADef     // the statement's SSA-name handle; wired for StmtAssign, zero for StmtEffect
	rhs   Node       // the statement's root node (NOT Value — see the contract note above)
}

// ID returns the statement's stable per-graph integer ID.
func (s *Statement) ID() uint64 { return s.id }

// Kind reports whether the statement binds a new local (StmtAssign) or is a
// pure side-effect operation (StmtEffect).
func (s *Statement) Kind() StmtKind { return s.kind }

// Next returns the next statement in the block, or nil at the list tail.
func (s *Statement) Next() *Statement { return s.next }

// Prev returns the previous statement in the block, or nil at the list head.
func (s *Statement) Prev() *Statement { return s.prev }

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
// is unwired (always true for StmtEffect statements).
func (s *Statement) Def() SSADef { return s.def }

// classifyStmt assigns the statement kind from the RHS root: a store is the
// v0 side-effecting stage (StmtEffect), a none-Value root (kernel return)
// binds no local either, and everything else is an assign statement.
func classifyStmt(rhs Node) StmtKind {
	if rhs == nil {
		return StmtAssign
	}
	switch rhs.KindOf() {
	case KindStore, KindReturn:
		return StmtEffect
	}
	return StmtAssign
}
