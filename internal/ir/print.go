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

// PrintOpts is Print with explicit options. The printer renders the Level-2
// scheduled graph: before Materialize there is no statement list to print,
// so an unmaterialized graph yields a deterministic one-line marker instead
// of partial content (codegen-adjacent surfaces schedule first).
func PrintOpts(g *Graph, o PrintOptions) string {
	if g == nil {
		return ""
	}
	if !g.materialized {
		return "ir: graph not materialized\n"
	}
	r := &renderer{num: ComputeNumbering(g), opts: o}
	var b strings.Builder
	first := true
	for s := g.cfg.entry.head; s != nil; s = s.next {
		if !first {
			b.WriteString("\n")
		}
		first = false
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

// ComputeNumbering computes the deterministic SSA numbering of g. It is a
// Level-2 operation (numbering walks the materialized statement list); a nil
// or unmaterialized graph yields an empty numbering.
func ComputeNumbering(g *Graph) *Numbering {
	t := &Numbering{byBase: make(map[*Base]int)}
	if g == nil || g.cfg == nil {
		return t
	}
	next := 0
	for s := g.cfg.entry.head; s != nil; s = s.next {
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
// (zero handles) yield nil entries. It is the exported face of the
// operand-slot walk shared by the printer, the numbering pass, and DOT edge
// emission (D-05).
func OperandsOf(n Node) []Node {
	var out []Node
	if impl := asImpl(n); impl != nil {
		impl.forEachOperand(func(u SSAUse) { out = append(out, u.Def()) })
	}
	return out
}

// LambdaBodies returns the body roots of map/reduce stages — DERIVED since
// the Input/ImpactBy-only convention (quick-261019-ktm revision): the body
// is the trailing operand of the stage's trailing KindLambda node. The
// verifier, DOT emission (D-05), the GVN body-root set, and the side-effect
// analysis all consume this face; the storage behind it is ordinary
// operand slots.
func LambdaBodies(n Node) []Node {
	lam := LambdaOf(n)
	if lam == nil {
		return nil
	}
	if body := lambdaNodeBody(lam); body != nil {
		return []Node{body}
	}
	return nil
}

// LambdaParams returns a map/reduce stage's lambda parameter atoms in
// binding order — DERIVED since the Input/ImpactBy-only convention: the
// leading operands of the stage's trailing KindLambda node (the read face
// internal/opt/simplify's inline pass consumes). Non-lambda-carrying nodes
// yield nil.
func LambdaParams(n Node) [][]Node {
	lam := LambdaOf(n)
	if lam == nil {
		return nil
	}
	return [][]Node{lambdaNodeParams(lam)}
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
	// Operand slots carry every data edge since the Input/ImpactBy-only
	// convention (lambda nodes and their params/bodies included), so the
	// visit is a plain operand DFS.
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
	}
	for _, st := range chain {
		for _, op := range OperandsOf(st) {
			visit(op) // the pipe operand is already seen and skips itself
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
// references to anonymous defs), the verbosity option, and the statement
// currently being rendered (a node that roots a DIFFERENT statement renders
// as `%N` — the statement-boundary rule for pass-inserted value statements).
type renderer struct {
	num  *Numbering
	opts PrintOptions
	cur  *Statement
}

// printChain renders one statement's stage chain: `  stage0 |\n  stage1...`,
// each stage on its own line, ` |` marking all but the last.
func (r *renderer) printChain(b *strings.Builder, s *Statement) {
	r.cur = s
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
// Extensibility note (Phase 01.1 audit): kinds absent from this list are
// NOT stages — a future stage kind must be added here explicitly, and the
// golden bytes will catch any emission drift immediately.
func isStage(n Node) bool {
	if n == nil {
		return false
	}
	switch n.KindOf() {
	case KindIterCluster, KindMap, KindReduce, KindMemStore, KindMemStoreSMem,
		KindFragStore, KindReturn, KindForLoop:
		return true
	}
	return false
}

// pipeInput returns the node feeding n's chained input — GMEM/SMEM/FRAG
// store value, return value, map/reduce operand 0 (piped input first, the
// Input-list layout of quick-261019-ktm: reduce operand 0 is the piped
// value, operand 1 the init attribute), iteration-cluster input 0 — or nil
// when the input is not itself a stage, meaning n starts (or breaks) the
// chain. A map whose operand 0 is an As/KeyedAttribute (zero value inputs)
// has no pipe: nil, defensively, so a malformed stage never chains.
func pipeInput(n Node) Node {
	var next Node
	switch n.KindOf() {
	case KindMemStore:
		next = opAt(n, 0) // the stored value (tensor/axes follow)
	case KindReturn:
		next = opAt(n, 0)
	case KindIterCluster:
		// D14: dims LEAD the cluster's input list; the pipe candidate is
		// the first non-dim INPUT (the legacy wrapped map).
		if inputs := ClusterInputs(n); len(inputs) > 0 {
			next = inputs[0]
		}
	case KindMap:
		next = opAt(n, 0)
		if next != nil {
			switch next.KindOf() {
			case KindAs, KindKeyedAttribute, KindLambda:
				return nil // no value operands: nothing pipes into the map
			}
		}
	case KindReduce:
		next = opAt(n, 0) // D12: piped first, init attribute second, lambda last
	case KindMemStoreSMem, KindFragStore:
		next = opAt(n, 0) // the stored value
	case KindForLoop:
		next = opAt(n, 2) // the body
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
// A non-stage RHS (a value-rooted statement, created by a pass after
// scheduling — e.g. the pass.Scheduler's LICM hoists via NewStatement) is a
// degenerate one-element chain: it prints as a single value line.
func chainOf(s *Statement) []Node {
	if s == nil || s.rhs == nil {
		return nil
	}
	if !isStage(s.rhs) {
		return []Node{s.rhs}
	}
	rhsOf := make(map[*Base]bool)
	for o := s.graph.cfg.entry.head; o != nil; o = o.next {
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
		// element of its own; the walk continues from the body's pipe
		// input. D14: the body is the cluster's first non-dim INPUT.
		if len(chain) > 0 {
			if ic, ok := chain[len(chain)-1].(*iterClusterNode); ok {
				if inputs := ClusterInputs(ic); len(inputs) > 0 && sameNode(inputs[0], n) {
					n = pipeInput(n)
					continue
				}
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

// paramNameOf resolves a use to its InputParam's debug name (the ImpactBy
// renderers' form). A zero/killed use or a def that is not an InputParam
// renders a deterministic placeholder — the verifier owns rejecting such
// graphs (review P1R-01: names are debug payload only, never resolution
// state).
func paramNameOf(u SSAUse) string {
	return paramNameOfNode(u.Def())
}

// paramNameOfNode resolves a node to its InputParam debug name (the
// operand-based twin of paramNameOf — the D14/D15 renderers read operands).
func paramNameOfNode(n Node) string {
	ip, ok := n.(*inputParamNode)
	if !ok {
		return "?"
	}
	return ip.name
}

// inputParamType returns the interned type carried by the InputParam n
// resolves to, or nil for any other node kind.
func inputParamType(n Node) (Type, bool) {
	ip, ok := n.(*inputParamNode)
	if !ok {
		return nil, false
	}
	return ip.typ, true
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
		for i, d := range ClusterDims(t) {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString("@" + paramNameOfNode(d))
		}
		b.WriteString(") -> ")
		inputs := ClusterInputs(t)
		switch len(inputs) {
		case 0:
			// bare header (dims only)
		case 1:
			if body := inputs[0]; body != nil {
				// The iteration cluster is transparent for piping: a value
				// piped THROUGH an iter() wrapper is still the previous stage
				// (doc §2.6), so the body sees prev — this is what renders
				// `iter() -> map(%0, ...)` for a fused piped map. The legacy
				// single-input cluster (the wrapped Map) keeps today's bytes.
				b.WriteString(r.stageText(body, prev))
			}
		default:
			// The IterCluster format's input list (MemLoads pending
			// simplify-lower): each input renders joined with ", ".
			for i, in := range inputs {
				if i > 0 {
					b.WriteString(", ")
				}
				b.WriteString(r.stageText(in, nil))
			}
		}
		return b.String()
	case *mapNode:
		var b strings.Builder
		b.WriteString("map(")
		valueOps := OperandsOf(t)
		// The trailing lambda node (always) and As node (when present) are
		// layout markers, not value operands.
		if len(valueOps) > 0 && valueOps[len(valueOps)-1] != nil &&
			valueOps[len(valueOps)-1].KindOf() == KindLambda {
			valueOps = valueOps[:len(valueOps)-1]
		}
		if len(valueOps) > 0 && valueOps[len(valueOps)-1] != nil &&
			valueOps[len(valueOps)-1].KindOf() == KindAs {
			valueOps = valueOps[:len(valueOps)-1] // the trailing As renders as the as-clause
		}
		for i, op := range valueOps {
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
		b.WriteString(r.lambdaText(lambdaNodeOf(t)))
		b.WriteString(")")
		b.WriteString(r.asClause(AsOf(t)))
		return b.String()
	case *reduceNode:
		var b strings.Builder
		b.WriteString("reduce(init=")
		b.WriteString(r.value(ReduceInitOf(t)))
		b.WriteString(", ")
		if piped := opAt(t, 0); prev != nil && sameNode(piped, prev) {
			b.WriteString("%0")
		} else {
			b.WriteString(r.value(piped))
		}
		b.WriteString(", ")
		b.WriteString(r.lambdaText(lambdaNodeOf(t)))
		b.WriteString(")")
		return b.String()
	case *memStoreNode:
		var b strings.Builder
		b.WriteString("memstore(")
		b.WriteString(r.accessText(t))
		b.WriteString(", ")
		if val := opAt(t, 0); prev != nil && sameNode(val, prev) {
			b.WriteString("%0")
		} else {
			b.WriteString(r.value(val))
		}
		b.WriteString(")")
		return b.String()
	case *returnNode:
		var b strings.Builder
		b.WriteString("return ")
		for i, v := range OperandsOf(t) {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(r.value(v))
		}
		return b.String()
	case *memLoadNode:
		return "memload(" + r.memAccess(t, 1) + ")"
	case *memNewNode:
		return "memnew(" + r.value(opAt(t, 0)) + ")"
	case *memLoadSMemNode:
		return "memloadsmem(" + r.bufferAccess(t, 1) + ")"
	case *memStoreSMemNode:
		var b strings.Builder
		b.WriteString("memstoresmem(")
		b.WriteString(r.bufferAccess(t, 2))
		b.WriteString(", ")
		if val := opAt(t, 0); prev != nil && sameNode(val, prev) {
			b.WriteString("%0")
		} else {
			b.WriteString(r.value(val))
		}
		b.WriteString(")")
		return b.String()
	case *memNewSMemNode:
		return "memnewsmem(" + r.value(opAt(t, 0)) + ")"
	case *fragLoadNode:
		return "fragload(" + r.bufferAccess(t, 1) + ")"
	case *fragStoreNode:
		var b strings.Builder
		b.WriteString("fragstore(")
		b.WriteString(r.bufferAccess(t, 2))
		b.WriteString(", ")
		if val := opAt(t, 0); prev != nil && sameNode(val, prev) {
			b.WriteString("%0")
		} else {
			b.WriteString(r.value(val))
		}
		b.WriteString(")")
		return b.String()
	case *fragNewNode:
		return "fragnew(" + r.value(opAt(t, 0)) + ")"
	case *forLoopNode:
		var b strings.Builder
		b.WriteString("for(")
		b.WriteString(r.value(opAt(t, 0)))
		b.WriteString(" in ")
		b.WriteString(r.value(opAt(t, 1)))
		b.WriteString(") ")
		b.WriteString(r.value(opAt(t, 2)))
		// M1 trailing annotations (quick-261021-jay): the tile-lower
		// pass's soc/dosoc/loc KeyedAttribute inputs render after the
		// body, each through the keyedattribute value rendering.
		for i := 3; i < len(OperandsOf(t)); i++ {
			b.WriteString(" ")
			b.WriteString(r.value(operandAt(t, i)))
		}
		return b.String()
	default:
		return r.value(n)
	}
}

// memAccess renders an HIR GMEM access's `tensor[@m][@n]` form: the tensor
// operand's debug name followed by one `[@dim]` per axis operand (the axes
// start at operand `axesFrom` — 1 for memload, whose operand 0 is the
// tensor).
func (r *renderer) memAccess(n Node, axesFrom int) string {
	var b strings.Builder
	if t := operandAt(n, 0); t != nil {
		if ip, ok := t.(*inputParamNode); ok {
			b.WriteString(ip.name)
		} else {
			b.WriteString(r.value(t))
		}
	} else {
		b.WriteString("<nil>")
	}
	for i := axesFrom; i < len(OperandsOf(n)); i++ {
		b.WriteString("[")
		if ax := operandAt(n, i); ax != nil {
			if ip, ok := ax.(*inputParamNode); ok {
				b.WriteString("@" + ip.name)
			} else {
				b.WriteString(r.value(ax))
			}
		} else {
			b.WriteString("<nil>")
		}
		b.WriteString("]")
	}
	return b.String()
}

// bufferAccess renders an LIR memory access's `buffer[biv][biv]...` form:
// the buffer operand's value rendering followed by one `[biv]` per biv
// operand (the bivs start at operand `bivsFrom` — 1 for loads, 2 for
// stores, whose leading operands are the stored value and the buffer). A
// KeyedAttribute operand in the trailing region is an ANNOTATION, not a
// coordinate (LO-01, 261021-o61 T5): the tile-lower fill stores carry their
// loadtile loader-selection variant there, and it renders as the attribute
// suffix `[key=value]` — never through the keyedattribute value spelling
// that would read as a coordinate subscript.
func (r *renderer) bufferAccess(n Node, bivsFrom int) string {
	var b strings.Builder
	if buf := operandAt(n, bivsFrom-1); buf != nil {
		b.WriteString(r.value(buf))
	} else {
		b.WriteString("<nil>")
	}
	for i := bivsFrom; i < len(OperandsOf(n)); i++ {
		op := operandAt(n, i)
		b.WriteString("[")
		if op != nil && op.KindOf() == KindKeyedAttribute {
			key, val := KeyedAttributeOf(op)
			b.WriteString(key + "=" + r.value(val))
		} else {
			b.WriteString(r.value(op))
		}
		b.WriteString("]")
	}
	return b.String()
}

// value renders a non-stage-position operand: scalar expressions with
// canonical §7.6 precedence parens, kernel params bare (`scale`), tensor
// accesses via their axis structure, stored bindings as `%name`, and any
// other anonymous SSA def as `%N` in definition order.
func (r *renderer) value(n Node) string {
	if n == nil {
		return "<nil>" // unwired operand; deterministic marker, verifier owns rejection
	}
	// Statement-boundary rule: a node that roots a DIFFERENT assign
	// statement (created by a pass after scheduling — LICM hoists) renders
	// as its SSA number from every other statement's tree, exactly like a
	// chained stage boundary. Restricted to StmtAssign roots so effect
	// statements keep their established spellings (e.g. `return %output`,
	// where the store root keeps rendering through its own access rule) and
	// to the text printer (cur != nil) so DOT labels are untouched.
	if r.rendersAsRef(n) {
		if num, ok := r.num.Num(n); ok {
			return fmt.Sprintf("%%%d", num)
		}
		return fmt.Sprintf("%%<%d>", n.BasePtr().id)
	}
	switch t := n.(type) {
	case *constNode:
		return constText(t)
	case *paramRefNode:
		// A reference to a tensor param IS the structural access: with no
		// separate access node kind in the v0 suite, the axes render from
		// the tensor type's dims (lhs[@m][@n]); scalar params and lambda
		// params render bare (scale, v). The name comes from the operand
		// def node's debug payload — never from resolution state (D14: the
		// reference rides operand 0).
		target := opAt(t, 0)
		if typ, ok := inputParamType(target); ok {
			if pt, ok := typ.(*NumericType); ok && !pt.IsScalar() {
				if ip, ok := target.(*inputParamNode); ok {
					var b strings.Builder
					b.WriteString(ip.name)
					for _, d := range pt.Dims() {
						b.WriteString("[@" + d.Name + "]")
					}
					return b.String()
				}
			}
		}
		return paramNameOfNode(target) // lambda params and captured kernel scalars both render bare
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
		// ops = [biv] (nest root) or [biv, parent] (nested): the loop
		// header IS the bounded IV value object — render through it. The
		// parent edge is nest structure (loopforest's face), not inline
		// syntax: a parented iter renders exactly like its root.
		if biv := opAt(t, 0); biv != nil {
			if bv, ok := biv.(*boundedDimVarNode); ok {
				if mx := opAt(bv, 1); mx != nil {
					if ip, ok := mx.(*inputParamNode); ok {
						if dv, ok := opAt(bv, 0).(*dimVarNode); ok {
							return "@" + paramNameOfNode(opAt(dv, 0)) + "/" + ip.name
						}
					}
				}
			}
			return r.value(biv) // "@x/max" via the boundedDimVarNode case
		}
		return "@?"
	case *tensorDimOfNode:
		// `tensordimof(lhs, 1)` — the tensor's debug name (its value
		// rendering when not a param) and the axis const's typed spelling.
		var b strings.Builder
		b.WriteString("tensordimof(")
		if tp := operandAt(t, 0); tp != nil {
			if ip, ok := tp.(*inputParamNode); ok {
				b.WriteString(ip.name)
			} else {
				b.WriteString(r.value(tp))
			}
		} else {
			b.WriteString("<nil>")
		}
		b.WriteString(", ")
		if ax := operandAt(t, 1); ax != nil {
			if c, ok := ax.(*constNode); ok {
				b.WriteString(constText(c))
			} else {
				b.WriteString(r.value(ax))
			}
		} else {
			b.WriteString("<nil>")
		}
		b.WriteString(")")
		return b.String()
	case *inductionVariableNode:
		return "inductionvariable(" + r.value(opAt(t, 0)) + ")"
	case *runtimeQueryNode:
		// The launch-map queries render like calls with the field as the
		// leading selector: runtimegridquery(index, %N)
		// (quick-261021-jay; renamed LR-6, quick-261022-l1o).
		var b strings.Builder
		if t.KindOf() == KindRuntimeGridQuery {
			b.WriteString("runtimegridquery(")
		} else {
			b.WriteString("runtimethreadquery(")
		}
		b.WriteString(t.field)
		if d := opAt(t, 0); d != nil {
			b.WriteString(", ")
			b.WriteString(r.value(d))
		}
		b.WriteString(")")
		return b.String()
	case *runtimeKernelInfoNode:
		return "runtimekernelinfo()"
	case *runtimeMappingNode:
		// runtimegridmapping(%ki, 1, -1): the object operand followed by the
		// ndim and portion constants at their typed spellings.
		var b strings.Builder
		b.WriteString(t.KindOf().String() + "(")
		first := true
		for _, op := range OperandsOf(t) {
			if !first {
				b.WriteString(", ")
			}
			first = false
			b.WriteString(r.value(op))
		}
		b.WriteString(")")
		return b.String()
	case *runtimeReducerNode:
		// runtimewarpreducer(%init, %lambda): the construction carries ALL
		// state (LR-3).
		var b strings.Builder
		b.WriteString(t.KindOf().String() + "(")
		for i, op := range OperandsOf(t) {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(r.value(op))
		}
		b.WriteString(")")
		return b.String()
	case *runtimeCallNode:
		// runtimecallmethod(%reducer, "Init", ...): the method payload as a
		// quoted selector after the object, then the args as values (LR-4).
		var b strings.Builder
		b.WriteString("runtimecallmethod(")
		for i, op := range OperandsOf(t) {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(r.value(op))
		}
		b.WriteString(", " + strconv.Quote(t.method) + ")")
		return b.String()
	case *runtimeGetAttributeNode:
		// runtimegetattribute(%mapping, "index", 1): the attribute payload
		// as a quoted selector after the object, then the args (LR-4).
		var b strings.Builder
		b.WriteString("runtimegetattribute(")
		for i, op := range OperandsOf(t) {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(r.value(op))
		}
		b.WriteString(", " + strconv.Quote(t.attribute) + ")")
		return b.String()
	case *keyedAttributeNode:
		return "keyedattribute(" + t.key + "=" + r.value(opAt(t, 0)) + ")"
	case *dimVarNode:
		// Lowered IV: `@x(init=0.0, step=1.0)` — identity is the tuple,
		// the name is debug payload (doc/optana/induction_variable.md).
		// D14: ops = [dim, init, step].
		return "@" + paramNameOfNode(opAt(t, 0)) +
			"(init=" + r.value(opAt(t, 1)) + ", step=" + r.value(opAt(t, 2)) + ")"
	case *boundedDimVarNode:
		// Bounded IV: `@x/M` — the dimvar's name with its max bound.
		if iv := opAt(t, 0); iv != nil {
			if dv, ok := iv.(*dimVarNode); ok {
				return "@" + paramNameOfNode(opAt(dv, 0)) + "/" + r.value(opAt(t, 1))
			}
		}
		return "@" + r.value(opAt(t, 0)) + "/" + r.value(opAt(t, 1))
	case *memStoreNode:
		// The store returns the stored value (doc §2.5) — the bare tensor
		// name in value position (the full access rendering belongs to the
		// stage printer; the tensor rides slot 1, D14).
		return "%" + paramNameOfNode(opAt(t, 1))
	default:
		if num, ok := r.num.Num(n); ok {
			return fmt.Sprintf("%%%d", num)
		}
		return fmt.Sprintf("%%<%d>", n.BasePtr().id) // not reachable from the block; deterministic
	}
}

// lambdaText renders a lambda node's `lambda(p1, p2): <body>` — the
// params from the node's leading operands, the body from its trailing
// operand (D15).
func (r *renderer) lambdaText(lam Node) string {
	if lam == nil {
		return "lambda(): <nil>"
	}
	ops := OperandsOf(lam)
	var body Node
	params := ops
	if len(ops) > 0 {
		params = ops[:len(ops)-1]
		body = ops[len(ops)-1]
	}
	var b strings.Builder
	b.WriteString("lambda(")
	for i, p := range params {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(paramNameOfNode(p))
	}
	b.WriteString("): ")
	b.WriteString(r.value(body))
	return b.String()
}

// asClause renders the mandatory output-order clause: ` as (@m, @k)`. The
// dims come from the trailing As node's operand list (D1) — a nil as
// renders nothing (the zero-dim/legacy form).
func (r *renderer) asClause(as Node) string {
	if as == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString(" as (")
	for i, d := range OperandsOf(as) {
		if i > 0 {
			b.WriteString(", ")
		}
		if ip, ok := d.(*inputParamNode); ok {
			b.WriteString("@" + ip.name)
		} else {
			b.WriteString("@" + r.value(d)) // non-param dim: deterministic; verifier owns rejection
		}
	}
	b.WriteString(")")
	return b.String()
}

// accessText renders a structural store access: `%output[@m][@k]` — the
// tensor operand (slot 1) and axis operands (slots 2...) of the store
// (D14: the target access rides the Input list). A dim-param axis renders
// `@name`; any other axis (a lowered index expression — the tile-lower
// pass's rank/offset arithmetic) renders through the value renderer, the
// same rule memAccess applies (quick-261021-o61 T5: the old bare `[@?]`
// fallback silently dropped every lowered coordinate from the print).
func (r *renderer) accessText(n Node) string {
	var b strings.Builder
	if tp := opAt(n, 1); tp != nil {
		b.WriteString("%" + paramNameOfNode(tp))
	} else {
		b.WriteString("%?")
	}
	for i := 2; i < len(OperandsOf(n)); i++ {
		b.WriteString("[")
		if ax := opAt(n, i); ax != nil {
			if ip, ok := ax.(*inputParamNode); ok {
				b.WriteString("@" + ip.name)
			} else {
				b.WriteString(r.value(ax))
			}
		} else {
			b.WriteString("@?")
		}
		b.WriteString("]")
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

// rendersAsRef reports whether n renders through the statement-boundary
// rule (as `%N`) from the statement currently being printed.
func (r *renderer) rendersAsRef(n Node) bool {
	if r.cur == nil {
		return false
	}
	os := n.BasePtr().stmt
	return os != nil && os != r.cur && os.kind == StmtAssign &&
		os.rhs != nil && os.rhs.BasePtr() == n.BasePtr()
}

// operandPrec returns the effective binding precedence of n as an operand:
// a node that renders through the statement-boundary rule (`%N`) is an atom
// and binds tightest; otherwise the structural expression precedence. Using
// the RENDERED precedence keeps hoisted operands parenthesis-free (`%0 * 3.0`,
// never `(%0) * 3.0`).
func (r *renderer) operandPrec(n Node) int {
	if r.rendersAsRef(n) {
		return 12
	}
	return exprPrec(n)
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
	if r.operandPrec(l) < p {
		ls = "(" + ls + ")"
	}
	rs := r.value(rhs)
	if r.operandPrec(rhs) <= p {
		rs = "(" + rs + ")"
	}
	return ls + " " + t.op.String() + " " + rs
}

// unaryExpr renders `-x`; the operand parenthesizes unless it binds tighter
// than unary (per §7.6 only casts/primaries do).
func (r *renderer) unaryExpr(t *unaryOpNode) string {
	x := opAt(t, 0)
	xs := r.value(x)
	if r.operandPrec(x) <= 11 {
		xs = "(" + xs + ")"
	}
	return t.op.String() + xs
}

// sameNode reports whether a and b are the same IR node (by shared Base).
func sameNode(a, b Node) bool {
	return a != nil && b != nil && a.BasePtr() == b.BasePtr()
}
