package cxx

import (
	"fmt"

	"tcacomp/internal/ir"
)

// statement emits one AST item node per the emission table (doc/cg.md §6).
// The dispatch is total over the codegen whitelist: anything else fails
// closed.
func (e *emitter) statement(n ir.Node) error {
	// Lambda-owned items (the combiner's params and body, anchored by the
	// scheduler) emit file-scope instead — never inside the kernel body.
	if e.lambdaOwned[n] {
		return nil
	}
	switch n.KindOf() {
	// Hoisted declarations — nothing at their scheduled position.
	case ir.KindInputParam, ir.KindMemNewSMem, ir.KindFragNew:
		return nil
	// The end node is metadata (the launcher owns outputs); emits nothing.
	case ir.KindReturn:
		return nil
	// Annotation carriers ride their host nodes' operand lists; the
	// scheduler may still anchor them as items — metadata only.
	case ir.KindKeyedAttribute:
		return nil
	// Loops materialize as Block items, never as statement items.
	case ir.KindForLoop:
		return fmt.Errorf("cxx: forloop %d appears as a statement item — the schedule must place it as a Loop block", n.BasePtr().ID())

	// GMEM through the TensorView shim (D3/D6).
	case ir.KindMemLoad:
		return e.stmtMemLoad(n)
	case ir.KindMemStore:
		return e.stmtMemStore(n)

	// SMEM/FRAG buffers.
	case ir.KindMemLoadSMem, ir.KindFragLoad:
		return e.stmtBufLoad(n)
	case ir.KindMemStoreSMem, ir.KindFragStore:
		if isFillStore(n) {
			return e.stmtFill(n)
		}
		return e.stmtBufStore(n)

	// Runtime objects (final-lower product).
	case ir.KindRuntimeKernelInfo, ir.KindRuntimeGridMapping, ir.KindRuntimeThreadMapping,
		ir.KindRuntimeSequentialReducer, ir.KindRuntimeWarpReducer, ir.KindRuntimeGlobalReducer:
		return e.stmtRuntimeObject(n)
	case ir.KindRuntimeCallMethod:
		return e.stmtCallMethod(n)

	// Hoisted pure values (SN-LICM products): materialize as locals.
	case ir.KindConst, ir.KindParam, ir.KindBinOp, ir.KindUnaryOp, ir.KindSelect,
		ir.KindCast, ir.KindCall, ir.KindTensorDimOf, ir.KindRuntimeGetAttribute, ir.KindDimVar:
		return e.ensureLocal(n)

	default:
		return fmt.Errorf("cxx: statement item %d carries kind %s — outside the codegen whitelist", n.BasePtr().ID(), n.KindOf())
	}
}

// ensureLocal guarantees a materialized local exists for the (pure) item
// node, declaring it at its item position on first materialization. Two
// quality guards (quick-261023-cgq): an item the emitted statements never
// reference emits NOTHING (B1 — dead locals are warnings), and an item
// whose expression text duplicates an earlier one REUSES that local (B2 —
// post-GVN passes mint duplicate attribute reads the compiler-side CSE
// never saw).
func (e *emitter) ensureLocal(n ir.Node) error {
	if _, ok := e.values[n]; ok {
		return nil // referenced earlier — already declared
	}
	if !e.pureRef[n] {
		return nil // never referenced by any emitted statement or loop header
	}
	text, err := e.emitExpr(n)
	if err != nil {
		return err
	}
	if _, ok := e.values[n]; ok {
		return nil // the const path declared itself
	}
	if first, ok := e.cseText[text]; ok {
		e.values[n] = first // identical pure expression — reuse the first local
		return nil
	}
	name := e.allocVal()
	e.cseText[text] = name
	e.line("auto " + name + " = " + text + ";" + e.provenance(n))
	e.values[n] = name
	return nil
}

// allocVal hands out the next value local (v0, v1, ... — collision-checked).
func (e *emitter) allocVal() string {
	name := e.local(fmt.Sprintf("v%d", e.valN))
	e.valN++
	return name
}

// stmtMemLoad: a GMEM read through the tensor's TensorView — the shim owns
// bounds and layout (D3/D6). Fill-source loads emit nothing: they exist to
// carry the loader's source coordinates (see prescan).
func (e *emitter) stmtMemLoad(n ir.Node) error {
	if e.fillSrc[n] {
		return nil
	}
	coords, _, err := e.accessAxes(n, 1)
	if err != nil {
		return err
	}
	view, err := e.viewOf(ir.OperandsOf(n)[0])
	if err != nil {
		return err
	}
	name := e.allocVal()
	e.line("auto " + name + " = " + view + ".Get(" + join(coords) + ");")
	e.values[n] = name
	return nil
}

// stmtMemStore: the epilogue store — one owner per address (structural
// race-freedom), so a plain Set through the view.
func (e *emitter) stmtMemStore(n ir.Node) error {
	ops := ir.OperandsOf(n)
	if len(ops) < 2 {
		return fmt.Errorf("cxx: memstore %d malformed", n.BasePtr().ID())
	}
	val, err := e.emitExpr(ops[0])
	if err != nil {
		return err
	}
	view, err := e.viewOf(ops[1])
	if err != nil {
		return err
	}
	coords, _, err := e.accessAxes(n, 2)
	if err != nil {
		return err
	}
	e.line(view + ".Set(" + val + ", " + join(coords) + ");")
	return nil
}

// accessAxes emits the axis expressions of a memory access starting at
// operand from, returning their C++ texts and the raw operand nodes.
func (e *emitter) accessAxes(n ir.Node, from int) ([]string, []ir.Node, error) {
	ops := ir.OperandsOf(n)
	var texts []string
	for _, a := range ops[from:] {
		s, err := e.emitExpr(a)
		if err != nil {
			return nil, nil, err
		}
		texts = append(texts, s)
	}
	return texts, ops[from:], nil
}

// viewOf resolves a tensor InputParam operand to its TensorView local.
func (e *emitter) viewOf(tensor ir.Node) (string, error) {
	if tensor == nil {
		return "", fmt.Errorf("cxx: nil tensor operand")
	}
	if v, ok := e.tensorView[tensor]; ok {
		return v, nil
	}
	return "", fmt.Errorf("cxx: tensor %d has no view (not a kernel tensor param?)", tensor.BasePtr().ID())
}

// stmtBufLoad: a buffer read (MemLoadSMem/FragLoad) at its in-tile
// coordinates — the lowered index forms (R2 rank-in-index for WARP dims,
// plain inside dimvars for SEQ dims) are ordinary expressions by now.
func (e *emitter) stmtBufLoad(n ir.Node) error {
	ops := ir.OperandsOf(n)
	if len(ops) == 0 || ops[0] == nil {
		return fmt.Errorf("cxx: buffer load %d carries no source buffer", n.BasePtr().ID())
	}
	view, err := e.bufferView(ops[0])
	if err != nil {
		return err
	}
	coords, _, err := e.accessAxes(n, 1)
	if err != nil {
		return err
	}
	name := e.allocVal()
	e.line("auto " + name + " = " + view + ".Get(" + join(coords) + ");")
	e.values[n] = name
	return nil
}

// stmtBufStore: a non-fill buffer store (no v0 pass emits one; the kind
// stays total) — a plain Set through the buffer's view.
func (e *emitter) stmtBufStore(n ir.Node) error {
	ops := ir.OperandsOf(n)
	if len(ops) < 2 || ops[1] == nil {
		return fmt.Errorf("cxx: buffer store %d malformed", n.BasePtr().ID())
	}
	val, err := e.emitExpr(ops[0])
	if err != nil {
		return err
	}
	view, err := e.bufferView(ops[1])
	if err != nil {
		return err
	}
	coords, _, err := e.accessAxes(n, 2)
	if err != nil {
		return err
	}
	e.line(view + ".Set(" + val + ", " + join(coords) + ");")
	return nil
}

// bufferView resolves a buffer operand (MemNewSMem/FragNew) to its view
// local.
func (e *emitter) bufferView(buf ir.Node) (string, error) {
	if v, ok := e.smemView[buf]; ok {
		return v, nil
	}
	if v, ok := e.fragLocal[buf]; ok {
		return v, nil
	}
	return "", fmt.Errorf("cxx: buffer %d has no view (not a live allocation?)", buf.BasePtr().ID())
}
