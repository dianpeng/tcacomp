package ir

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// This file is the canonical deterministic IR printer (TOOL-04, D-01..D-04).
// Two structurally identical graphs print byte-identically (D-02): emission
// order comes only from basic-block statement order, chain stage order, and
// operand slot order — never map iteration (RESEARCH Pitfall 1). SSA
// numbering is recomputed from scratch on every print (Pitfall 3). The
// minimal default print carries no locations, times, or paths (D-03,
// T-1-03-01), which is what makes byte-exact goldens feasible (D-13).
//
// Exported helpers (ComputeNumbering, StmtNodes, OperandsOf, Label) share
// the printer's ordering and rendering with internal/visualization's DOT
// emitter so .dot output mirrors the text numbering (D-05).

// PrintOptions controls the canonical printer.
type PrintOptions struct {
	// Verbose appends `// pass: <origin>` annotations naming the pass that
	// originated each stage (D-03 escape hatch, for pass debugging only).
	// Phase 1 goldens pin the minimal default.
	Verbose bool
}

// Print renders g in the canonical doc-faithful notation of doc/tca.md
// §5.1 (D-01): `statement #N:` headers with stable construction-time IDs
// (D-04), chain stages joined by ` |`, 2-space continuation indent, one
// blank line between statements, single trailing newline.
func Print(g *Graph) string { return PrintOpts(g, PrintOptions{}) }

// PrintOpts is Print with explicit options.
func PrintOpts(g *Graph, o PrintOptions) string {
	if g == nil {
		return ""
	}
	r := &renderer{num: ComputeNumbering(g), opts: o}
	var b strings.Builder
	for i, s := range g.block.statements {
		if i > 0 {
			b.WriteString("\n")
		}
		fmt.Fprintf(&b, "statement #%d:\n", s.id)
		r.printChain(&b, s)
	}
	return b.String()
}

// FormatConst is the single owner of the canonical FLOAT-family constant
// spelling (Pitfall 9): strconv.FormatFloat shortest-round-trip at float32
// precision (locale-independent), with `.0` appended when the result carries
// no radix point or exponent so integers-valued floats print as `0.0`,
// never `0` or `0f` — IR text is algebra notation, not frontend source. The
// committed goldens tripwire any shortest-round-trip drift across Go versions.
// Integer- and bool-typed constants render through constText instead, at full
// precision (WR-02).
func FormatConst(v float64) string {
	s := strconv.FormatFloat(v, 'g', -1, 32)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

// constText renders one const node's value with type-faithful spelling. The
// node already carries its interned type; integer-typed constants print at
// full integer precision — float32's 24-bit mantissa would otherwise collapse
// distinct i32/u32 values (e.g. 2147483647 vs 2147483600) to identical text
// (WR-02) — bools print as true/false, and the float family keeps
// FormatConst's shortest-round-trip f32 spelling. A nil or non-numeric type
// falls back to the float spelling so the printer stays deterministic on
// malformed nodes (the verifier owns their rejection).
func constText(t *constNode) string {
	nt, ok := t.typ.(*NumericType)
	if !ok {
		return FormatConst(t.val)
	}
	switch nt.Elem() {
	case TI8, TI16, TI32:
		return strconv.FormatInt(int64(t.val), 10)
	case TU8, TU16, TU32:
		return strconv.FormatUint(uint64(t.val), 10)
	case TBool:
		if t.val != 0 {
			return "true"
		}
		return "false"
	default: // f16/bf16/f32 and any future float type
		return FormatConst(t.val)
	}
}

// Numbering is the SSA definition-order numbering of a graph: value-producing
// nodes numbered 0,1,2,... in basic-block statement order, each statement's
// tree walked in the same order the printer emits it (chain stages first,
// then operand subtrees depth-first, lambda bodies last). Numbering is a
// pure function of the graph, recomputed per print, never cached (Pitfall 3).
type Numbering struct {
	byBase map[*Base]int
}

// Num returns the SSA number assigned to def and whether def was numbered.
func (t *Numbering) Num(def Node) (int, bool) {
	if t == nil || def == nil {
		return 0, false
	}
	n, ok := t.byBase[def.BasePtr()]
	return n, ok
}

func (t *Numbering) has(n Node) bool {
	_, ok := t.byBase[n.BasePtr()]
	return ok
}

// ComputeNumbering computes the deterministic SSA numbering of g.
func ComputeNumbering(g *Graph) *Numbering {
	t := &Numbering{byBase: make(map[*Base]int)}
	if g == nil {
		return t
	}
	next := 0
	for _, s := range g.block.statements {
		for _, n := range StmtNodes(s) {
			if !t.has(n) {
				t.byBase[n.BasePtr()] = next
				next++
			}
		}
	}
	return t
}

// OperandsOf returns n's operand defs in canonical slot order; unwired slots
// yield nil entries. It is the exported face of the operand-slot walk shared
// by the printer, the numbering pass, and DOT edge emission (D-05).
func OperandsOf(n Node) []Node {
	var out []Node
	if impl := asImpl(n); impl != nil {
		impl.forEachOperand(func(u *Use) { out = append(out, u.Val) })
	}
	return out
}

// LambdaBodies returns the body roots of n's lambda payloads (map/reduce
// stages) — the IR's single non-Use graph edge kind, walked as an edge by
// the verifier and rendered as a def-to-user edge by DOT emission (D-05).
func LambdaBodies(n Node) []Node {
	var out []Node
	if lb, ok := n.(lambdaBearer); ok {
		lb.forEachLambda(func(lam *Lambda) { out = append(out, lam.Body) })
	}
	return out
}

// StmtNodes returns stmt's value nodes in canonical print order: chain
// stages producer-first, then per stage the operand subtrees depth-first
// (operand slots in order, lambda bodies included), every node exactly once.
// The order is deterministic; no map iteration (Pitfall 1).
func StmtNodes(stmt *Statement) []Node {
	if stmt == nil {
		return nil
	}
	chain := chainOf(stmt)
	out := make([]Node, 0, len(chain))
	seen := make(map[*Base]bool, len(chain))
	for _, st := range chain {
		seen[st.BasePtr()] = true
		out = append(out, st)
	}
	var visit func(n Node)
	visit = func(n Node) {
		if n == nil || seen[n.BasePtr()] {
			return
		}
		seen[n.BasePtr()] = true
		out = append(out, n)
		for _, op := range OperandsOf(n) {
			visit(op)
		}
		if lb, ok := n.(lambdaBearer); ok {
			lb.forEachLambda(func(lam *Lambda) { visit(lam.Body) })
		}
	}
	for _, st := range chain {
		for _, op := range OperandsOf(st) {
			visit(op) // the pipe operand is already seen and skips itself
		}
		if lb, ok := st.(lambdaBearer); ok {
			lb.forEachLambda(func(lam *Lambda) { visit(lam.Body) })
		}
	}
	return out
}

// Label returns the canonical one-node text label for n under num — the same
// rendering the text printer uses, so .dot labels mirror the text output
// (D-05). Stage nodes label with their stage text; other nodes with their
// value text.
func Label(n Node, num *Numbering) string {
	r := &renderer{num: num}
	return r.stageText(n, nil)
}

// renderer carries the per-print shared state: the numbering (for `%N`
// references to anonymous defs) and the verbosity option.
type renderer struct {
	num  *Numbering
	opts PrintOptions
}

// printChain renders one statement's stage chain: `  stage0 |\n  stage1...`,
// each stage on its own line, ` |` marking all but the last.
func (r *renderer) printChain(b *strings.Builder, s *Statement) {
	chain := chainOf(s)
	for i, st := range chain {
		b.WriteString("  ")
		var prev Node
		if i > 0 {
			prev = chain[i-1]
		}
		b.WriteString(r.stageText(st, prev))
		last := i == len(chain)-1
		if !last {
			b.WriteString(" |")
		}
		if r.opts.Verbose {
			fmt.Fprintf(b, "  // pass: %s", st.BasePtr().loc.PassOrigin)
		}
		if !last {
			b.WriteString("\n")
		}
	}
	b.WriteString("\n")
}

// isStage reports whether n can occupy a chained stage position.
func isStage(n Node) bool {
	if n == nil {
		return false
	}
	switch n.KindOf() {
	case KindIterCluster, KindMap, KindReduce, KindStore, KindReturn:
		return true
	}
	return false
}

// pipeInput returns the node feeding n's chained input — store value,
// return value, map operand 0 (piped input first, per construct.go), reduce
// operand 1 (piped), iteration-cluster body — or nil when the input is not
// itself a stage, meaning n starts (or breaks) the chain.
func pipeInput(n Node) Node {
	var next Node
	switch n.KindOf() {
	case KindStore, KindReturn, KindMap, KindIterCluster:
		next = opAt(n, 0)
	case KindReduce:
		next = opAt(n, 1)
	}
	if isStage(next) {
		return next
	}
	return nil
}

// chainOf collects stmt's stage chain in producer-to-consumer order. The
// walk follows pipe inputs backward from the statement's RHS root; it stops
// at a node that is some other statement's RHS (a statement boundary — that
// stage prints as its own statement), at a non-stage input, or on a cycle.
func chainOf(s *Statement) []Node {
	if s == nil || s.rhs == nil {
		return nil
	}
	rhsOf := make(map[*Base]bool, len(s.graph.block.statements))
	for _, o := range s.graph.block.statements {
		if o.rhs != nil {
			rhsOf[o.rhs.BasePtr()] = true
		}
	}
	var chain []Node
	seen := make(map[*Base]bool)
	for n := s.rhs; n != nil && isStage(n) && !seen[n.BasePtr()]; {
		seen[n.BasePtr()] = true
		if n != s.rhs && rhsOf[n.BasePtr()] {
			break // another statement's root: chain boundary
		}
		// An iteration cluster renders its body stage inline (`iter(...) ->
		// map(...)` is one printed stage), so the body is not a chain
		// element of its own; the walk continues from the body's pipe input.
		if len(chain) > 0 {
			if ic, ok := chain[len(chain)-1].(*iterClusterNode); ok && sameNode(opAt(ic, 0), n) {
				n = pipeInput(n)
				continue
			}
		}
		chain = append(chain, n)
		n = pipeInput(n)
	}
	slices.Reverse(chain)
	return chain
}

// opAt returns the def at operand slot i, or nil when out of range/unwired.
func opAt(n Node, i int) Node {
	ops := OperandsOf(n)
	if i < 0 || i >= len(ops) {
		return nil
	}
	return ops[i]
}

// stageText renders one chain stage. prev is the preceding stage in the
// chain (nil for the first); when the stage's pipe-bound input is prev, it
// renders as the stage-local `%0` (doc §2.6 pipe semantics — `%0` is
// stage-local, which is why every chained input prints `%0`).
func (r *renderer) stageText(n Node, prev Node) string {
	switch t := n.(type) {
	case *iterClusterNode:
		var b strings.Builder
		b.WriteString("iter(")
		for i, d := range t.dims {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString("@" + d.Name())
		}
		b.WriteString(") -> ")
		if body := opAt(t, 0); body != nil {
			// The iteration cluster is transparent for piping: a value
			// piped THROUGH an iter() wrapper is still the previous stage
			// (doc §2.6), so the body sees prev — this is what renders
			// `iter() -> map(%0, ...)` for a fused piped map.
			b.WriteString(r.stageText(body, prev))
		}
		return b.String()
	case *mapNode:
		var b strings.Builder
		b.WriteString("map(")
		for i, op := range OperandsOf(t) {
			if i > 0 {
				b.WriteString(", ")
			}
			if i == 0 && prev != nil && sameNode(op, prev) {
				b.WriteString("%0")
			} else {
				b.WriteString(r.value(op))
			}
		}
		b.WriteString(", ")
		b.WriteString(r.lambda(t.lam))
		b.WriteString(")")
		b.WriteString(asClause(t.as))
		return b.String()
	case *reduceNode:
		var b strings.Builder
		b.WriteString("reduce(init=")
		b.WriteString(r.value(opAt(t, 0)))
		b.WriteString(", ")
		if piped := opAt(t, 1); prev != nil && sameNode(piped, prev) {
			b.WriteString("%0")
		} else {
			b.WriteString(r.value(piped))
		}
		b.WriteString(", ")
		b.WriteString(r.lambda(t.combiner))
		b.WriteString(")")
		return b.String()
	case *storeNode:
		var b strings.Builder
		b.WriteString("store(")
		b.WriteString(accessText(t.target))
		b.WriteString(", ")
		if val := opAt(t, 0); prev != nil && sameNode(val, prev) {
			b.WriteString("%0")
		} else {
			b.WriteString(r.value(val))
		}
		b.WriteString(")")
		return b.String()
	case *returnNode:
		return "return " + r.value(opAt(t, 0))
	default:
		return r.value(n)
	}
}

// value renders a non-stage-position operand: scalar expressions with
// canonical §7.6 precedence parens, kernel params bare (`scale`), tensor
// accesses via their axis structure, stored bindings as `%name`, and any
// other anonymous SSA def as `%N` in definition order.
func (r *renderer) value(n Node) string {
	if n == nil {
		return "<nil>" // unwired operand; deterministic marker, verifier owns rejection
	}
	switch t := n.(type) {
	case *constNode:
		return constText(t)
	case *paramRefNode:
		// A reference to a tensor param IS the structural access: with no
		// separate access node kind in the v0 suite, the axes render from
		// the tensor type's dims (lhs[@m][@n]); scalar params and lambda
		// params render bare (scale, v). The builder enforces that the
		// bound axes match these dims positionally, so the printed form is
		// faithful.
		if pt, ok := t.param.Type().(*NumericType); ok && !pt.IsScalar() {
			var b strings.Builder
			b.WriteString(t.param.Name())
			for _, d := range pt.Dims() {
				b.WriteString("[@" + d.Name + "]")
			}
			return b.String()
		}
		return t.param.Name() // lambda params and captured kernel scalars both render bare
	case *binOpNode:
		return r.binExpr(t)
	case *unaryOpNode:
		return r.unaryExpr(t)
	case *selectNode:
		return r.value(opAt(t, 0)) + " ? " + r.value(opAt(t, 1)) + " : " + r.value(opAt(t, 2))
	case *castNode:
		return "cast<" + typeText(t.typ) + ">(" + r.value(opAt(t, 0)) + ")"
	case *callNode:
		var b strings.Builder
		b.WriteString(t.name)
		b.WriteString("(")
		for i, a := range OperandsOf(t) {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(r.value(a))
		}
		b.WriteString(")")
		return b.String()
	case *iterNode:
		s := "@" + t.dim.Name()
		if b := t.bound.Name(); b != "" {
			s += "/" + b // bounded induction variable, doc §2.3
		}
		return s
	case *storeNode:
		return "%" + t.target.Tensor.Name() // store returns the stored value (doc §2.5)
	default:
		if num, ok := r.num.Num(n); ok {
			return fmt.Sprintf("%%%d", num)
		}
		return fmt.Sprintf("%%<%d>", n.BasePtr().id) // not reachable from the block; deterministic
	}
}

// lambda renders `lambda(p1, p2): <body>`.
func (r *renderer) lambda(l Lambda) string {
	var b strings.Builder
	b.WriteString("lambda(")
	for i, p := range l.Params {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(p.Name())
	}
	b.WriteString("): ")
	b.WriteString(r.value(l.Body))
	return b.String()
}

// asClause renders the mandatory output-order clause: ` as (@m, @k)`.
func asClause(as []Param) string {
	var b strings.Builder
	b.WriteString(" as (")
	for i, p := range as {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString("@" + p.Name())
	}
	b.WriteString(")")
	return b.String()
}

// accessText renders a structural store access: `%output[@m][@k]`.
func accessText(a Access) string {
	var b strings.Builder
	b.WriteString("%" + a.Tensor.Name())
	for _, ax := range a.Axes {
		b.WriteString("[@" + ax.Name() + "]")
	}
	return b.String()
}

// typeText renders an interned type for cast annotations. Casts are scalar
// in v0; tensor spellings are best-effort for diagnostics. A nil or
// non-numeric type renders as `?` (deterministic; the verifier owns
// rejection) — the printer's numeric paths are never touched by future
// non-numeric Type components.
func typeText(t Type) string {
	nt, ok := t.(*NumericType)
	if !ok {
		return "?"
	}
	if nt.IsScalar() {
		return nt.Elem().String()
	}
	var b strings.Builder
	fmt.Fprintf(&b, "tensor<dtype=%s, shape=[", nt.Elem())
	for i, d := range nt.Dims() {
		if i > 0 {
			b.WriteString(", ")
		}
		if d.Extent != 0 {
			fmt.Fprintf(&b, "%d", d.Extent)
		} else {
			b.WriteString(d.Name)
		}
	}
	b.WriteString("]>")
	return b.String()
}

// binPrec maps a binary opcode to its doc/syntax.md §7.6 precedence level,
// loosest (conditional handled separately at 0) to tightest.
func binPrec(op BinOpCode) int {
	switch op {
	case BinOpLOr:
		return 1
	case BinOpLAnd:
		return 2
	case BinOpOr:
		return 3
	case BinOpXor:
		return 4
	case BinOpAnd:
		return 5
	case BinOpEq, BinOpNE:
		return 6
	case BinOpLT, BinOpLE, BinOpGT, BinOpGE:
		return 7
	case BinOpShL, BinOpShR:
		return 8
	case BinOpAdd, BinOpSub:
		return 9
	case BinOpMul, BinOpDiv, BinOpMod:
		return 10
	}
	return 0
}

// exprPrec is the precedence of a rendered value expression, used to decide
// parens. Atoms bind tightest; the conditional sits below every binary
// level, mirroring the §7.6 table.
func exprPrec(n Node) int {
	switch t := n.(type) {
	case *binOpNode:
		return binPrec(t.op)
	case *unaryOpNode:
		return 11
	case *selectNode:
		return 0
	}
	return 12
}

// binExpr renders `a + b` with canonical parens: a child parenthesized when
// it binds looser than the parent, and also when it binds equal on the right
// (binary levels are left-associative per §7.6, so `a - (b - c)` keeps its
// parens).
func (r *renderer) binExpr(t *binOpNode) string {
	p := binPrec(t.op)
	l := opAt(t, 0)
	rhs := opAt(t, 1)
	ls := r.value(l)
	if exprPrec(l) < p {
		ls = "(" + ls + ")"
	}
	rs := r.value(rhs)
	if exprPrec(rhs) <= p {
		rs = "(" + rs + ")"
	}
	return ls + " " + t.op.String() + " " + rs
}

// unaryExpr renders `-x`; the operand parenthesizes unless it binds tighter
// than unary (per §7.6 only casts/primaries do).
func (r *renderer) unaryExpr(t *unaryOpNode) string {
	x := opAt(t, 0)
	xs := r.value(x)
	if exprPrec(x) <= 11 {
		xs = "(" + xs + ")"
	}
	return t.op.String() + xs
}

// sameNode reports whether a and b are the same IR node (by shared Base).
func sameNode(a, b Node) bool {
	return a != nil && b != nil && a.BasePtr() == b.BasePtr()
}
