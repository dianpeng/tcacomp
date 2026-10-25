package cxx

import (
	"fmt"

	"tcacomp/internal/ir"
)

// stmtFill emits a fill store as its cooperative loader call (doc/cg.md §6
// M-3, fix3 D-h): the loader is DERIVED from the fill's storage class per
// R-H/R-J — SMEM window class → tca::rt::LoadTileCta (cooperative, owns
// ALL synchronization and internal bounds clamping), FRAG class →
// tca::rt::LoadTileThread (per-thread, no barrier); the retired v0-era
// attribute is gone. The compiler never emits barriers
// (runtime_objects.md §7).
//
// Source-coordinate convention (pinned from tile-lower's fill construction,
// lower_cache.go): the fill's value operand is a MemLoad whose axes are the
// NON-covered tensor positions (uniform-substituted for the SMEM class),
// and the trailing bivs are the covered dims' round bases. The merged
// coordinate tuple is reconstructed in TENSOR DIM ORDER:
//
//   - len(axes) == rank: the FRAG all-axes form — axes verbatim;
//   - len(axes) + len(bivs) == rank: every covered position was skipped —
//     covered positions (named by the buffer's window dims / the bivs' dim
//     params) take their biv, the rest take their axis, in order;
//   - anything else: the FRAG mixed-with-skips form — fail closed (the v0
//     search policy does not produce it).
func (e *emitter) stmtFill(n ir.Node) error {
	ops := ir.OperandsOf(n)
	if len(ops) < 3 {
		return fmt.Errorf("cxx: fill %d malformed (want value, dst, bivs...)", n.BasePtr().ID())
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
	e.line(loader + "(" + bufView + ", " + view + ", " + join(coords) + ");")
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
