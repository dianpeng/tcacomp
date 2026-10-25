// astdot is a dev-only tool: it builds the canonical matmul kernel through
// the exported IR construction surface, runs the full default pipeline
// (dce → simplify → typeinfer → gvn → memopt → tiling → tile-lower →
// schedule → final-lower), and emits the final kernel for visualization.
//
// Two views (the compiler itself never renders DOT; this tool is a viewer):
//
//   - default  — the SSA graph: every live IR node rendered EXACTLY ONCE
//     with def→use edges (operand order), clustered by its scheduled AST
//     block, so sharing is visible and nothing is duplicated. This is the
//     correct answer to "what does the lowered IR look like" — labels are
//     short (no inlined subtrees); the structure lives in the edges.
//   - -items   — the AST/CFG placement view: one box per AST block ITEM
//     (the L2 completeness invariant lands every scheduled node as an item,
//     including bound-expression atoms and loop metadata), labels via the
//     canonical text printer. Faithful but flat — kept for pass debugging.
//
// A k-reduction GEMM (Map + Reduce + store over [m,n]) currently trips the
// known-open tile-lower fill-splice defect family (schedule-pass review
// rulings, quick-261021-o61 follow-up) on the default tiling, so this tool
// builds the probe shape the full default pipeline runs clean: the
// canonical matmul probe (internal/opt/schedule pipeline_test.go's
// matmulGraphK — out[m,n,k] = lhs[m,k]·rhs[k,n], defaults promote lhs into
// one SMEM window fill per run).
package main

import (
	"fmt"
	"io"
	"os"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	_ "tcacomp/internal/opt/all"
)

// matmul builds the canonical matmul probe shape (internal/opt/schedule
// pipeline_test.go's matmulGraphK): u32 extents M/N/K over the M→N→K iter
// chain, InductionVariable-wired lhs[m,k] and rhs[k,n] loads, a fused
// product, and a chained store of lhs·rhs over [m,n,k].
func matmul(name string, m, n, kExtent float64) *ir.Graph {
	g := ir.NewGraph(name)
	u32 := ir.InternScalar(ir.TU32)
	dims := []ir.Node{
		g.NewInputParam("ctor", ir.RoleDim, "m", nil),
		g.NewInputParam("ctor", ir.RoleDim, "n", nil),
		g.NewInputParam("ctor", ir.RoleDim, "k", nil),
	}
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}, {Name: "n"}, {Name: "k"}}))
	lhs := g.NewInputParam("ctor", ir.RoleTensor, "lhs",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}, {Name: "k"}}))
	rhs := g.NewInputParam("ctor", ir.RoleTensor, "rhs",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "k"}, {Name: "n"}}))

	var parent ir.Node
	iters := make([]ir.Node, 3)
	for i, extent := range []float64{m, n, kExtent} {
		dv := g.NewDimVar("ctor", dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, extent))
		parent = g.NewIter("ctor", biv, parent)
		iters[i] = parent
	}
	iv := func(i int) ir.Node { return g.NewInductionVariable("ctor", iters[i]) }

	l := g.NewMemLoad("ctor", lhs, []ir.Node{iv(0), iv(2)}) // lhs[m,k]
	r := g.NewMemLoad("ctor", rhs, []ir.Node{iv(2), iv(1)}) // rhs[k,n]
	prod := g.NewBinOp("ctor", ir.BinOpMul, l, r)

	env := ir.NewEnv()
	st := g.NewMemStoreChained("ctor", env, out, []ir.Node{iv(0), iv(1), iv(2)}, prod)
	g.NewReturnChained("ctor", env, []ir.Node{out}, st)
	return g
}

func main() {
	itemsView := len(os.Args) > 1 && os.Args[1] == "-items"
	out := "matmul_ast.dot"
	arg := 1
	if itemsView {
		arg = 2
	}
	if len(os.Args) > arg {
		out = os.Args[arg]
	}
	g := matmul("matmul", 1024, 1024, 512)
	if err := opt.NewManager(opt.BuildPipeline()).Run(g); err != nil {
		fmt.Fprintf(os.Stderr, "pipeline: %v\n", err)
		os.Exit(1)
	}
	k := g.AST()
	if k == nil {
		fmt.Fprintln(os.Stderr, "no AST attached")
		os.Exit(1)
	}
	f, err := os.Create(out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	if itemsView {
		err = emitItemsDot(f, g, k)
	} else {
		err = emitSSADot(f, g, k)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	stats(g)
	fmt.Println("wrote", out)
}

// stats reports the sharing evidence: live nodes, operand edges, and the
// multi-use distribution — the SSA property the DOT edges visualize.
func stats(g *ir.Graph) {
	live, edges, multiUse := 0, 0, 0
	maxUses, maxID := 0, uint64(0)
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		live++
		ops := ir.OperandsOf(n)
		for _, op := range ops {
			if op != nil && op.BasePtr() != nil && !op.BasePtr().Erased() {
				edges++
			}
		}
		if u := len(ir.UsesOf(n)); u > 1 {
			multiUse++
			if u > maxUses {
				maxUses, maxID = u, n.BasePtr().ID()
			}
		}
	}
	fmt.Printf("live nodes=%d operand-edges=%d multi-use nodes=%d (max %d uses, node #%d)\n",
		live, edges, multiUse, maxUses, maxID)
}

// ---------------------------------------------------------------------------
// Default view: the SSA graph — every live node once, def→use edges,
// clustered by scheduled AST block.
// ---------------------------------------------------------------------------

func emitSSADot(w io.Writer, g *ir.Graph, k *ir.ASTKernel) error {
	// Live nodes grouped by their scheduled block; nil-block nodes go last.
	byBlock := make(map[*ir.ASTBlock][]ir.Node)
	var unplaced []ir.Node
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		if blk := n.BasePtr().SchedBlock(); blk != nil {
			byBlock[blk] = append(byBlock[blk], n)
		} else {
			unplaced = append(unplaced, n)
		}
	}
	e := &ssaEmitter{w: w}
	e.printf("digraph %s {\n  compound=true;\n  rankdir=TB;\n  node [shape=box, fontname=monospace];\n\n", g.Name())
	e.block(k.Prototype, byBlock, "Prototype (kernel signature params)")
	e.printf("\n")
	e.block(k.Body, byBlock, "kernel Body")
	if len(unplaced) > 0 {
		e.printf("\n  // nodes with no SchedBlock (metadata not landed by the scheduler)\n")
		e.cluster++
		e.printf("  subgraph cluster_%d {\n    label=%s;\n    color=gray;\n", e.cluster, quote("unplaced (no SchedBlock)"))
		for _, n := range unplaced {
			e.printf("    n%d [label=%s];\n", n.BasePtr().ID(), quote(shortLabel(n)))
		}
		e.printf("  }\n")
	}
	// def→use edges, operand order (deterministic).
	e.printf("\n  // SSA def→use edges\n")
	seen := make(map[ir.Node]bool)
	var emitEdges func(n ir.Node)
	emitEdges = func(n ir.Node) {
		if n == nil || seen[n] {
			return
		}
		seen[n] = true
		for _, op := range ir.OperandsOf(n) {
			if op == nil || op.BasePtr() == nil || op.BasePtr().Erased() {
				continue
			}
			e.printf("  n%d -> n%d;\n", op.BasePtr().ID(), n.BasePtr().ID())
			emitEdges(op)
		}
	}
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
			continue
		}
		emitEdges(n)
	}
	e.printf("}\n")
	return e.err
}

type ssaEmitter struct {
	w       io.Writer
	err     error
	cluster int
}

func (e *ssaEmitter) printf(format string, args ...any) {
	if e.err != nil {
		return
	}
	_, e.err = fmt.Fprintf(e.w, format, args...)
}

func (e *ssaEmitter) block(blk *ir.ASTBlock, byBlock map[*ir.ASTBlock][]ir.Node, title string) {
	if blk == nil || e.err != nil {
		return
	}
	label := title
	if loop := blk.Loop(); loop != nil {
		ops := ir.OperandsOf(loop)
		iv, wrapper := ops[0], ops[1]
		maxExpr := iv
		if wops := ir.OperandsOf(wrapper); len(wops) > 1 {
			maxExpr = wops[1]
		}
		label = fmt.Sprintf("%s #%d  iv=%s  bound=%s  SN=%d", blk.Kind(),
			loop.BasePtr().ID(), shortLabel(iv), shortLabel(maxExpr), loop.BasePtr().SchedSN())
	} else if blk.Nest() != nil {
		label = "Group (LNT cluster prologue/epilogue)"
	}
	e.cluster++
	cid := e.cluster
	e.printf("  subgraph cluster_%d {\n    label=%s;\n    color=blue;\n", cid, quote(label))
	for _, n := range byBlock[blk] {
		e.printf("    n%d [label=%s];\n", n.BasePtr().ID(), quote(shortLabel(n)))
	}
	e.printf("  }\n")
	for _, it := range blk.Items() {
		if it.Block != nil {
			e.block(it.Block, byBlock, it.Block.Kind().String())
		}
	}
}

// shortLabel renders ONE LINE per node — never an inlined subtree: shared
// operands are pointed to by edges, not re-printed (the exact discipline
// the sea-of-nodes SSA form is built on).
func shortLabel(n ir.Node) string {
	if n == nil || n.BasePtr() == nil {
		return "?"
	}
	id := n.BasePtr().ID()
	k := n.KindOf()
	switch k {
	case ir.KindConst:
		if v, ok := ir.ConstValueOf(n); ok {
			return fmt.Sprintf("[%d] %g", id, v)
		}
	case ir.KindInputParam:
		if name := ir.InputParamNameOf(n); name != "" {
			return fmt.Sprintf("[%d] %s", id, name)
		}
	case ir.KindDimVar:
		if ops := ir.OperandsOf(n); len(ops) > 0 {
			return fmt.Sprintf("[%d] @%s", id, ir.InputParamNameOf(ops[0]))
		}
	case ir.KindRuntimeCallMethod:
		return fmt.Sprintf("[%d] call.%s #%d", id, ir.RuntimeMethodOf(n), ir.OperandsOf(n)[0].BasePtr().ID())
	case ir.KindRuntimeGetAttribute:
		return fmt.Sprintf("[%d] getattr(%s)", id, ir.RuntimeAttributeOf(n))
	case ir.KindKeyedAttribute:
		if name, v := ir.KeyedAttributeOf(n); name != "" {
			return fmt.Sprintf("[%d] attr.%s=%s", id, name, shortLabel(v))
		}
	case ir.KindMemLoad:
		return fmt.Sprintf("[%d] %s", id, k)
	case ir.KindMemStore:
		return fmt.Sprintf("[%d] %s", id, k)
	case ir.KindBinOp:
		if op, ok := ir.BinOpCodeOf(n); ok {
			return fmt.Sprintf("[%d] binop(%v)", id, op)
		}
	case ir.KindCall:
		if name := ir.CallNameOf(n); name != "" {
			return fmt.Sprintf("[%d] %s", id, name)
		}
	}
	return fmt.Sprintf("[%d] %s", id, k)
}

// ---------------------------------------------------------------------------
// -items view: the AST/CFG placement view — one box per AST block item,
// canonical text-printer labels (operands inlined). Faithful to the L2
// completeness invariant; flat by nature. For pass debugging, not reading.
// ---------------------------------------------------------------------------

func emitItemsDot(w io.Writer, g *ir.Graph, k *ir.ASTKernel) error {
	num := ir.ComputeNumbering(g)
	e := &itemsEmitter{w: w, num: num}
	fmt.Fprintf(w, "digraph %s {\n  compound=true;\n  rankdir=TB;\n  node [shape=box, fontname=monospace];\n\n", g.Name())
	e.block(k.Prototype, "Prototype (kernel signature params)")
	fmt.Fprintln(w)
	e.block(k.Body, "kernel Body")
	fmt.Fprintln(w, "}")
	return e.err
}

type itemsEmitter struct {
	w       io.Writer
	num     *ir.Numbering
	cluster int
	node    int
	err     error
}

func (e *itemsEmitter) printf(format string, args ...any) {
	if e.err != nil {
		return
	}
	_, e.err = fmt.Fprintf(e.w, format, args...)
}

func (e *itemsEmitter) block(blk *ir.ASTBlock, title string) {
	if blk == nil || e.err != nil {
		return
	}
	cid := e.cluster
	e.cluster++
	label := title
	if loop := blk.Loop(); loop != nil {
		ops := ir.OperandsOf(loop)
		iv, wrapper := ops[0], ops[1]
		maxExpr := iv
		if wops := ir.OperandsOf(wrapper); len(wops) > 1 {
			maxExpr = wops[1]
		}
		label = fmt.Sprintf("%s  loop #%d  iv=%s  bound=%s  SN=%d",
			blk.Kind(), loop.BasePtr().ID(),
			ir.Label(iv, e.num), ir.Label(maxExpr, e.num), loop.BasePtr().SchedSN())
	} else if blk.Nest() != nil && title == "Group" {
		label = "Group (LNT cluster prologue/epilogue)"
	}
	e.printf("  subgraph cluster_%d {\n    label=%s;\n    color=blue;\n", cid, quote(label))
	prev := -1
	for _, it := range blk.Items() {
		if it.Node != nil {
			id := e.node
			e.node++
			e.printf("    i%d [label=%s];\n", id, quote(itemLabel(it.Node, e.num)))
			if prev >= 0 {
				e.printf("    i%d -> i%d [style=dotted, arrowhead=open];\n", prev, id)
			}
			prev = id
			continue
		}
		if it.Block != nil {
			before := e.node
			e.block(it.Block, it.Block.Kind().String())
			if e.err != nil {
				return
			}
			if prev >= 0 && e.node > before {
				e.printf("    i%d -> i%d [style=dotted, arrowhead=open];\n", prev, before)
			}
			prev = -1
		}
	}
	e.printf("  }\n")
}

// itemLabel renders an AST item node's canonical label, spelling kernel
// signature params by name (the bare "%<id>" placeholder is opaque).
func itemLabel(n ir.Node, num *ir.Numbering) string {
	if n.KindOf() == ir.KindInputParam {
		if name := ir.InputParamNameOf(n); name != "" {
			role := "param"
			switch r, _ := ir.RoleOf(n); r {
			case ir.RoleDim:
				role = "dim"
			case ir.RoleTensor:
				role = "tensor"
			case ir.RoleScalar:
				role = "scalar"
			}
			return fmt.Sprintf("%s (%s)", name, role)
		}
	}
	return ir.Label(n, num)
}

func quote(s string) string {
	out := make([]byte, 0, len(s)+2)
	out = append(out, '"')
	for _, r := range s {
		switch r {
		case '"', '\\':
			out = append(out, '\\', byte(r))
		case '\n':
			out = append(out, '\\', 'n')
		default:
			out = append(out, string(r)...)
		}
	}
	out = append(out, '"')
	return string(out)
}
