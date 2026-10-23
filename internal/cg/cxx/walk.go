package cxx

import (
	"fmt"
	"strings"

	"tcacomp/internal/cg"
	"tcacomp/internal/ir"
)

// signature renders the kernel's parameter list from the Prototype's
// declaration-order params (doc/cg.md §6): dims as uint32_t by value,
// scalars by their type, tensors as pointers — const-qualified when no
// MemStore writes them. GMEM temporaries were rejected at the gates.
func (e *emitter) signature() (string, error) {
	var parts []string
	for _, p := range e.k.Params {
		role, ok := ir.RoleOf(p)
		if !ok {
			return "", fmt.Errorf("cxx: prototype param %d is not an InputParam", p.BasePtr().ID())
		}
		name := ir.InputParamNameOf(p)
		switch role {
		case ir.RoleDim:
			parts = append(parts, "uint32_t "+name)
		case ir.RoleScalar:
			ts, err := scalarTypeOfParam(p)
			if err != nil {
				return "", fmt.Errorf("cxx: scalar param %s: %w", name, err)
			}
			parts = append(parts, ts+" "+name)
		case ir.RoleTensor:
			ts, dims, err := e.tensorTypeOf(p)
			if err != nil {
				return "", err
			}
			_ = dims
			// __restrict__: the restrict-alias rule is structural in the IR
			// (kernel tensors never alias) — claim the free perf hint.
			if e.stored[p] {
				parts = append(parts, ts+"* __restrict__ "+name)
			} else {
				parts = append(parts, "const "+ts+"* __restrict__ "+name)
			}
		default:
			return "", fmt.Errorf("cxx: param %s carries role %d — not a kernel signature role", name, int(role))
		}
	}
	return strings.Join(parts, ", "), nil
}

// tensorTypeOf returns a tensor param's element spelling and its DimRefs.
func (e *emitter) tensorTypeOf(p ir.Node) (string, []ir.DimRef, error) {
	t, _ := ir.InputParamTypeOf(p)
	nt, ok := t.(*ir.NumericType)
	if !ok || nt == nil || nt.Rank() == 0 {
		return "", nil, fmt.Errorf("cxx: tensor param %s carries no tensor type", ir.InputParamNameOf(p))
	}
	ts, err := elemType(nt.Elem())
	if err != nil {
		return "", nil, err
	}
	return ts, nt.Dims(), nil
}

// kernelTop renders the kernel-top declarations (rulings D3/D4/D5/D8): the
// TensorView shims over every tensor param, the one dynamic __shared__ pool
// with a typed SmemView per buffer at its compiler-assigned offset, and the
// per-thread Frag tiles. Generated code never touches raw memory (D6).
func (e *emitter) kernelTop() (string, error) {
	var b strings.Builder
	write := func(s string) { b.WriteString("  " + s + "\n") }

	// Tensor views: one per tensor param, constructed from the pointer and
	// the runtime dim values (constant extents inline).
	for _, p := range e.k.Params {
		role, _ := ir.RoleOf(p)
		if role != ir.RoleTensor {
			continue
		}
		ts, dims, err := e.tensorTypeOf(p)
		if err != nil {
			return "", err
		}
		args, err := e.dimArgs(dims)
		if err != nil {
			return "", err
		}
		name := ir.InputParamNameOf(p)
		qual := ""
		if !e.stored[p] {
			qual = "const "
		}
		view := e.local(name + "_v")
		e.tensorView[p] = view
		write(fmt.Sprintf("tca::rt::TensorView<%s%s, %du> %s(%s, %s);", qual, ts, uint(len(dims)), view, name, strings.Join(args, ", ")))
	}

	// The SMEM pool (D5: dynamic — the 64 KB budget exceeds the 48 KB
	// static ceiling) and its disjoint v0 offset views (D8).
	if len(e.buffers) > 0 {
		if e.comments >= cg.CommentSections {
			write(fmt.Sprintf("// dynamic SMEM pool (%d B): one pool, disjoint v0 offsets — the future RA pass overlaps them", e.smemCap))
		}
		write("extern __shared__ __align__(16) char tca_smem[];")
		for _, buf := range e.buffers {
			view, err := e.smemViewDecl(buf, write)
			if err != nil {
				return "", err
			}
			_ = view
		}
	}

	// FRAG tiles: per-thread compile-time-shaped register wrappers.
	for _, f := range e.frags {
		ts, dims, err := e.bufferTypeOf(f)
		if err != nil {
			return "", err
		}
		local := fmt.Sprintf("tca_f%d", len(e.fragLocal))
		e.fragLocal[f] = local
		var ds []string
		for _, d := range dims {
			ds = append(ds, fmt.Sprintf("%du", d.Extent))
		}
		write(fmt.Sprintf("tca::rt::Frag<%s, %du, %s> %s;", ts, uint(len(dims)), strings.Join(ds, ", "), local))
	}
	if b.Len() > 0 {
		b.WriteString("\n")
	}
	return b.String(), nil
}

// smemViewDecl renders one buffer's typed view over the pool at its
// compiler-assigned offset and registers the view name.
func (e *emitter) smemViewDecl(buf ir.Node, write func(string)) (string, error) {
	ts, dims, err := e.bufferTypeOf(buf)
	if err != nil {
		return "", err
	}
	base := fmt.Sprintf("smem%d", len(e.smemView))
	if tn, ok := e.bufTensor[buf]; ok && tn != "" {
		base = identSanitize(tn) + "_tile"
	}
	local := e.local(base)
	e.smemView[buf] = local
	var exts []string
	for _, d := range dims {
		exts = append(exts, fmt.Sprintf("%du", d.Extent))
	}
	write(fmt.Sprintf("tca::rt::SmemView<%s, %du> %s(reinterpret_cast<%s*>(tca_smem + %du), %s);",
		ts, uint(len(dims)), local, ts, uint(e.smemOff[buf]), strings.Join(exts, ", ")))
	return local, nil
}

// bufferTypeOf returns a MemNewSMem/FragNew's element spelling and its
// window dims (compile-time extents required — a symbolic window extent
// cannot size a register tile or a pool view).
func (e *emitter) bufferTypeOf(n ir.Node) (string, []ir.DimRef, error) {
	nt, ok := ir.TypeOf(n).(*ir.NumericType)
	if !ok || nt == nil || nt.Rank() == 0 {
		return "", nil, fmt.Errorf("cxx: buffer %d carries no tensor type (tile-lower must bind it)", n.BasePtr().ID())
	}
	ts, err := elemType(nt.Elem())
	if err != nil {
		return "", nil, err
	}
	for _, d := range nt.Dims() {
		if d.Extent == 0 {
			return "", nil, fmt.Errorf("cxx: buffer %d dim %q has a symbolic extent — v0 requires compile-time windows", n.BasePtr().ID(), d.Name)
		}
	}
	return ts, nt.Dims(), nil
}

// dimArgs renders a tensor's dim arguments: the constant extent inline, the
// symbolic dim as its kernel argument name.
func (e *emitter) dimArgs(dims []ir.DimRef) ([]string, error) {
	var out []string
	for _, d := range dims {
		if d.Extent > 0 {
			out = append(out, fmt.Sprintf("%du", d.Extent))
			continue
		}
		p := e.dimParam(d.Name)
		if p == nil {
			return nil, fmt.Errorf("cxx: dim %q has no kernel dim parameter", d.Name)
		}
		out = append(out, ir.InputParamNameOf(p))
	}
	return out, nil
}

// dimParam resolves a dim name to its InputParam (names are debug payload;
// the signature order is the resolution order — deterministic).
func (e *emitter) dimParam(name string) ir.Node {
	for _, p := range e.k.Params {
		if role, _ := ir.RoleOf(p); role == ir.RoleDim && ir.InputParamNameOf(p) == name {
			return p
		}
	}
	return nil
}

// walkBody emits the kernel Body's items at FUNCTION scope — no wrapping
// braces for the Body Group itself (quick-261023-cgq): its only direct
// items are the KernelInfo declaration and the per-LNT lexical Groups,
// which carry their own scopes; a redundant outer scope pair is noise.
func (e *emitter) walkBody() error {
	for _, it := range e.k.Body.Items() {
		if it.Block != nil {
			if err := e.walkBlock(it.Block); err != nil {
				return err
			}
			continue
		}
		if it.Node == nil {
			continue
		}
		if err := e.statement(it.Node); err != nil {
			return err
		}
	}
	return nil
}

// walkBlock emits one AST block. Groups render as plain C++ scopes; Loop
// blocks render as runtime iterator for-statements (loops.go); every item
// statement renders per the emission table (doc/cg.md §6).
func (e *emitter) walkBlock(blk *ir.ASTBlock) error {
	if blk == nil {
		return nil
	}
	if blk.Kind() == ir.BlockGroup {
		if blk.Nest() != nil {
			e.lntHeader(blk.Nest())
		}
		e.line("{")
		e.deep++
		if err := e.walkItems(blk); err != nil {
			return err
		}
		e.deep--
		e.line("}")
		return nil
	}
	return e.walkLoop(blk)
}

// walkItems emits a block's items in placement order.
func (e *emitter) walkItems(blk *ir.ASTBlock) error {
	for _, it := range blk.Items() {
		if it.Block != nil {
			if err := e.walkBlock(it.Block); err != nil {
				return err
			}
			continue
		}
		if it.Node == nil {
			continue
		}
		if err := e.statement(it.Node); err != nil {
			return err
		}
	}
	return nil
}

// lntHeader emits the per-LNT summary comment (quick-261023-cgq): every
// loop of the nest with its dim, materialization class, tile, and LoC —
// the tiling decision a reviewer needs, synthesized from the loop
// annotations the IR already carries.
func (e *emitter) lntHeader(lnt *ir.ASTLNT) {
	if e.comments < cg.CommentSections {
		return
	}
	idx := 0
	for i, l := range e.k.LNTs {
		if l == lnt {
			idx = i
			break
		}
	}
	var parts []string
	for _, m := range lnt.Loops {
		dim := "dim"
		if ops := ir.OperandsOf(m.Loop); len(ops) >= 1 && ops[0] != nil {
			if ivOps := ir.OperandsOf(ops[0]); len(ivOps) == 3 && ivOps[0] != nil {
				dim = identSanitize(ir.InputParamNameOf(ivOps[0]))
			}
		}
		if t, ok := e.loopTile(m.Loop); ok {
			parts = append(parts, fmt.Sprintf("%s=%s(T=%d,LoC=%d)", dim, socName(m.Soc), t, m.Loc))
		} else {
			parts = append(parts, fmt.Sprintf("%s=%s(LoC=%d)", dim, socName(m.Soc), m.Loc))
		}
	}
	e.c("LNT #%d: %s", idx, strings.Join(parts, " "))
}
