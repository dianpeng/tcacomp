package cxx

import (
	"fmt"
	"strconv"

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
//
// Const-dim reconciliation (D-fix6-03, quick-261031-c8n): a covered dim may
// be COMPILE-TIME CONST, and the frontend builds its tensor DimRef as
// {Extent: e} with an EMPTY Name (build.go dimKey) — the name-keyed covered
// set can never match it. A DimRef{Extent>0} position is therefore
// reconciled against the covered bivs' STATIC DIM DECLARATIONS: a static
// dim's canonical InputParam spelling is "e<extent>" (build.go dimKey =
// pkg.Builder's single declaration source), which IS its declared extent —
// the ONLY extent surviving to codegen (RoleDim params carry no type, and
// tile-lower's round bases are fresh dimvars whose original BIV bounds were
// RAUW'd away). Claiming is by biv IDENTITY, never positional:
//   - a DimRef{Extent>0} position claims the UNCLAIMED covered biv whose
//     declared extent equals the ref's; ZERO matches means the position is
//     NOT covered and takes its axis (a CTA-assigned const dim is a legal
//     axis — never an error); MORE THAN ONE is a named fail-closed
//     ambiguity (the repeated-extent D-fix6-01 class — never guessed);
//   - a DimRef{Name} position claims the covered biv whose dim param
//     carries that name (the pre-fix behavior, made identity-explicit so a
//     const-covered dim preceding a dynamic-covered dim in tensor order —
//     row_sum/softmax's [R, Ku, C] — cannot mis-assign the positional
//     cursor); an already-claimed name match is a named fail-closed
//     double-claim;
//   - a covered biv left UNCLAIMED after the walk (its dim's declared
//     extent matched no position — a non-canonical static dim name) is a
//     named fail-closed error, never a silent drop.
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
		// The covered set keyed two ways: dim-param NAME (dynamic dims)
		// and, for the const positions the name can never match,
		// DECLARED EXTENT via the canonical "e<extent>" static-dim
		// spelling (see the header convention block).
		byName := make(map[string]ir.Node, len(bivs))
		var statics []struct {
			b      ir.Node
			extent uint64
		}
		for _, b := range bivs {
			d := e.bivDim(b)
			if d == nil {
				return fmt.Errorf("cxx: fill %d round base is not a dimvar", n.BasePtr().ID())
			}
			name := ir.InputParamNameOf(d)
			if _, dup := byName[name]; dup {
				return fmt.Errorf("cxx: fill %d covered set declares dim @%s twice (repeated-extent ambiguity — the D-fix6-01 class: fail closed, never guessed)", n.BasePtr().ID(), name)
			}
			byName[name] = b
			if ext, ok := staticDimExtent(name); ok {
				statics = append(statics, struct {
					b      ir.Node
					extent uint64
				}{b, ext})
			}
		}
		claimed := make(map[ir.Node]bool, len(bivs))
		claim := func(b ir.Node) { claimed[b] = true; coordNodes = append(coordNodes, b) }
		ai := 0
		axis := func() error {
			if ai >= len(axes) {
				return fmt.Errorf("cxx: fill %d coordinate reconstruction ran out of axes", n.BasePtr().ID())
			}
			coordNodes = append(coordNodes, axes[ai])
			ai++
			return nil
		}
		for pos, ref := range dims {
			if ref.Extent > 0 {
				// A const position: claim the covered biv whose declared
				// extent equals it; zero matches → the position is not
				// covered and rides its axis (e.g. a CTA-assigned const
				// dim) — exactly the pre-fix axis path for this case.
				var hit ir.Node
				for _, s := range statics {
					if s.extent != ref.Extent || claimed[s.b] {
						continue
					}
					if hit != nil {
						return fmt.Errorf("cxx: fill %d const position %d (extent %d) matches multiple covered bivs (repeated-extent ambiguity — the D-fix6-01 class: fail closed, never guessed)", n.BasePtr().ID(), pos, ref.Extent)
					}
					hit = s.b
				}
				if hit != nil {
					claim(hit)
					continue
				}
				if err := axis(); err != nil {
					return err
				}
				continue
			}
			if b, ok := byName[ref.Name]; ok {
				if claimed[b] {
					return fmt.Errorf("cxx: fill %d position %d (dim @%s) double-claims an already-matched covered biv (repeated-extent ambiguity — the D-fix6-01 class: fail closed, never guessed)", n.BasePtr().ID(), pos, ref.Name)
				}
				claim(b)
				continue
			}
			if err := axis(); err != nil {
				return err
			}
		}
		for _, b := range bivs {
			if !claimed[b] {
				return fmt.Errorf("cxx: fill %d covered dim @%s matched no tensor position — a static dim's declared-extent spelling is the canonical \"e<extent>\" (build.go dimKey); a non-canonical name cannot be reconciled (fail closed, never guessed)",
					n.BasePtr().ID(), ir.InputParamNameOf(e.bivDim(b)))
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

// staticDimExtent parses a dim InputParam's canonical static spelling
// "e<extent>" (build.go dimKey / pkg.Builder's single declaration source)
// into its declared compile-time extent. A dynamic dim's name never matches
// (identifiers like "elem" fail the digit run; extent 0 is dynamic by
// definition and is rejected), so a false return means "no declared static
// extent" — the caller's name-keyed dynamic path owns the position.
func staticDimExtent(name string) (uint64, bool) {
	if len(name) < 2 || name[0] != 'e' {
		return 0, false
	}
	for _, c := range name[1:] {
		if c < '0' || c > '9' {
			return 0, false
		}
	}
	ext, err := strconv.ParseUint(name[1:], 10, 64)
	if err != nil || ext == 0 {
		return 0, false
	}
	return ext, true
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
