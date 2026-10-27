package cxx

import (
	"fmt"

	"tcacomp/internal/ir"
)

// stmtFill emits the fill composition as its cooperative loader call
// (doc/cg.md §6 M-3, fix3 D-m/D-n per R-H/R-J): the codegen-side fusion of
// MemNew + tile-shaped MemLoad(GMEM) + MemStore(class, window) into ONE
// call under the coordination-mapping signature
//
//	tca::rt::LoadTileCta/LoadTileThread(mapping, dstView, srcWindowView)
//
// — the ThreadMapping comes from the emitter's tracked enclosing-LNT
// mapping (curThread): the runtime composes mapping->work and the compiler
// NEVER inlines thread enumeration. The source arrives as a WINDOWED view
// (tca::rt::TensorWindow) whose coordinate list is the merged tensor-dim
// tuple the composition carries (the tile load's axes + the store's
// trailing round bases); the loader receives no raw bases. Loader
// selection derives from the fill's storage class — SMEM window class →
// LoadTileCta (cooperative, owns ALL synchronization and internal bounds
// clamping), FRAG class → LoadTileThread (per-thread, no barrier). The
// compiler never emits barriers (runtime_objects.md §7). A fill site whose
// scope lacks a mapping is a named R-D stop — never an emitter patch.
//
// Source-coordinate convention (pinned from tile-lower's fill construction,
// lower_cache.go): the fill's value operand is a MemLoad whose axes are the
// NON-covered tensor positions (uniform-substituted for the SMEM class),
// and the trailing bivs are the covered dims' round bases. The merged
// coordinate tuple is built in TENSOR DIM ORDER:
//
//   - len(axes) == rank: the FRAG all-axes form — axes verbatim;
//   - len(axes) + len(bivs) == rank: every covered position was skipped —
//     covered positions (named by the buffer's window dims / the bivs' dim
//     params) take their biv, the rest take their axis, in order;
//   - anything else: the FRAG mixed-with-skips form — fail closed (the v0
//     search policy does not produce it).
func (e *emitter) stmtFill(n ir.Node) error {
	ops := ir.OperandsOf(n)
	if len(ops) < 2 {
		return fmt.Errorf("cxx: fill %d malformed (want value, dst[, bivs...])", n.BasePtr().ID())
	}
	src, dst, bivs := ops[0], ops[1], ops[2:]
	if src == nil || dst == nil || src.KindOf() != ir.KindMemLoad {
		return fmt.Errorf("cxx: fill %d value operand is not a MemLoad", n.BasePtr().ID())
	}
	// Loader selection derived from the storage class (R-H/R-J): the same
	// tier rule the retired loader attribute recorded.
	sc, _ := ir.StorageClassOf(n)
	v := 2 // FRAG class: per-thread
	if sc == ir.StorageSMem {
		v = 1 // SMEM window class: cooperative
	}
	// The coordination mapping (R-J): the enclosing LNT's ThreadMapping. A
	// fill site with no mapping in scope is an uncovered shape (R-D) — a
	// named error, never patched around.
	if e.curThread == "" {
		return fmt.Errorf("cxx: fill %d sits outside every LNT cluster (no ThreadMapping in scope) — the mapping-driven loader call needs the enclosing LNT's mapping (R-D: uncovered shape, needs a ruling)",
			n.BasePtr().ID())
	}

	srcOps := ir.OperandsOf(src)
	if len(srcOps) == 0 || srcOps[0] == nil {
		return fmt.Errorf("cxx: fill source load %d malformed", src.BasePtr().ID())
	}
	tensor := srcOps[0]
	axes := srcOps[1:]
	_, dims, err := e.tensorTypeOf(tensor)
	if err != nil {
		return err
	}
	rank := len(dims)

	var coordNodes []ir.Node
	switch {
	case len(axes) == rank:
		coordNodes = axes
	case len(axes)+len(bivs) == rank:
		covered := map[string]bool{}
		for _, b := range bivs {
			d := e.bivDim(b)
			if d == nil {
				return fmt.Errorf("cxx: fill %d round base is not a dimvar", n.BasePtr().ID())
			}
			covered[ir.InputParamNameOf(d)] = true
		}
		ai, bi := 0, 0
		for _, ref := range dims {
			if covered[ref.Name] {
				coordNodes = append(coordNodes, bivs[bi])
				bi++
			} else {
				if ai >= len(axes) {
					return fmt.Errorf("cxx: fill %d coordinate reconstruction ran out of axes", n.BasePtr().ID())
				}
				coordNodes = append(coordNodes, axes[ai])
				ai++
			}
		}
	default:
		return fmt.Errorf("cxx: fill %d coordinate form unsupported in v0 (%d axes + %d bases over rank %d)",
			n.BasePtr().ID(), len(axes), len(bivs), rank)
	}

	view, err := e.viewOf(tensor)
	if err != nil {
		return err
	}
	bufView, err := e.bufferView(dst)
	if err != nil {
		return err
	}
	var coords []string
	for _, c := range coordNodes {
		s, err := e.emitExpr(c)
		if err != nil {
			return err
		}
		coords = append(coords, s)
	}
	loader := "tca::rt::LoadTileThread"
	if v == 1 {
		loader = "tca::rt::LoadTileCta"
		e.c("cooperative tile fill of %s (LoadTileCta) — the runtime owns the copy, the barriers, and tail clamping", ir.InputParamNameOf(tensor))
	} else {
		e.c("per-thread tile fill of %s (LoadTileThread) — no synchronization", ir.InputParamNameOf(tensor))
	}
	// The coordination-mapping call (fix3 D-m/D-n, R-H/R-J): the mapping in,
	// the destination view, the source as a WINDOWED view — the offset
	// implicit in the window, the thread enumeration the runtime's.
	e.line(loader + "(" + e.curThread + ", " + bufView + ", tca::rt::TensorWindow(" + view +
		", " + join(coords) + "));")
	return nil
}

// bivDim returns a between-IV dimvar's dim InputParam operand.
func (e *emitter) bivDim(biv ir.Node) ir.Node {
	if biv == nil || biv.KindOf() != ir.KindDimVar {
		return nil
	}
	ops := ir.OperandsOf(biv)
	if len(ops) != 3 || ops[0] == nil {
		return nil
	}
	if role, ok := ir.RoleOf(ops[0]); !ok || role != ir.RoleDim {
		return nil
	}
	return ops[0]
}
