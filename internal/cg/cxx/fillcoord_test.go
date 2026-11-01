package cxx

import (
	"regexp"
	"testing"

	"tcacomp/internal/golden"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/finallower"
	"tcacomp/internal/opt/schedule"
	"tcacomp/internal/opt/tiling"
	"tcacomp/internal/opt/typeinfer"
)

// ---------------------------------------------------------------------------
// D-fix6-03 RED pins (quick-261031-c8n, design/fix/fix6.md §4)
// ---------------------------------------------------------------------------
//
// The frontend builds a compile-time-CONST tensor dim as ir.DimRef{Extent: e}
// with an EMPTY Name (internal/syntax/build/build.go dimKey — the static dim
// table keys by the canonical "e<extent>" spelling), while stmtFill's
// covered-position walk keys the covered set by the bivs' dim-param NAMES
// (internal/cg/cxx/memory.go). A fill whose covered set includes a const dim
// therefore never matches that position: the DimRef{Extent} position falls
// into the axis path, consumes the single axis slot, and emission fails with
// `cxx: fill <n> coordinate reconstruction ran out of axes`. fix6 §3's
// evidence: the all-dynamic collapse emits cleanly, the mixed [R, C, Ku]
// collapse (selected by the v62 honest-FRAG pricing on the six const-shaped
// suite kernels) dies at exactly this gate.
//
// The fixtures below are frontend-faithful by construction: the const dim's
// tensor DimRef is ir.DimRef{Extent: 4} (empty Name) and its dim InputParam
// is named "e4" — precisely what build.go dimKey produces for shape [R, C,
// 4u]. The crafted record mirrors the collapse shape (R=CTA, C=SEQ, e4=SEQ)
// so the promoted site's fill covers [C, e4] with ONE axis (R) and TWO bivs.

// fillCoordGraph builds the rank-3 mixed dynamic/static reduce chain: a
// chain of iters over dims R (dynamic), C (dynamic), e4 (static extent 4),
// TWO co-traversed loads x[R, C, e4] and z[R, C, e4] whose DimRefs are
// built per mode (mixed: frontend product with DimRef{Extent:4} at the
// const position; dynamic: every position Name-carrying; static: every
// position Extent-carrying), a reduce(init=0, +) over the e4 dim of x+z,
// and an out[R, C] store. The crafted record assigns R=CTA / C=SEQ /
// e4=SEQ and promotes both loads (FRAG class — R is a non-covered CTA INS
// iter, the C1b tier rule), so each promoted site's fill covers [C, e4]
// with one axis (R) and two bivs — the exact shape whose coordinate
// reconstruction pre-fix fails with "ran out of axes".
//
// The two loads exist because the overlap admission filters a SINGLETON
// set (filter (a): no co-traversal, no reuse) — the co-traversed pair is
// the minimal admitted shape, mirroring the suite's real two-load kernels
// (e.g. scale_copy's x/z).
func fillCoordGraph(t *testing.T, name string, mode string) *ir.Graph {
	t.Helper()
	g := ir.NewGraph(name)
	u32 := ir.InternScalar(ir.TU32)

	dims := []ir.Node{
		g.NewInputParam("ctor", ir.RoleDim, "R", nil),
		g.NewInputParam("ctor", ir.RoleDim, "C", nil),
		g.NewInputParam("ctor", ir.RoleDim, "e4", nil),
	}
	// The tensor DimRefs per mode — the mixed mode's const position is
	// ir.DimRef{Extent:4} with an EMPTY Name, exactly build.go:177.
	var xRefs []ir.DimRef
	switch mode {
	case "mixed":
		xRefs = []ir.DimRef{{Name: "R"}, {Name: "C"}, {Extent: 4}}
	case "dynamic":
		xRefs = []ir.DimRef{{Name: "R"}, {Name: "C"}, {Name: "e4"}}
	case "static":
		xRefs = []ir.DimRef{{Extent: 512}, {Extent: 4}, {Extent: 1024}}
	default:
		t.Fatalf("unknown DimRef mode %q", mode)
	}
	x := g.NewInputParam("ctor", ir.RoleTensor, "x", ir.InternTensor(ir.TF32, xRefs))
	z := g.NewInputParam("ctor", ir.RoleTensor, "z", ir.InternTensor(ir.TF32, xRefs))
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "R"}, {Name: "C"}}))

	extents := []float64{64, 32, 4}
	if mode == "static" {
		extents = []float64{512, 4, 1024}
	}
	var parent ir.Node
	iters := make([]ir.Node, 3)
	for i := range extents {
		dv := g.NewDimVar("ctor", dims[i], g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, extents[i]))
		parent = g.NewIter("ctor", biv, parent)
		iters[i] = parent
	}
	iv := func(i int) ir.Node { return g.NewInductionVariable("ctor", iters[i]) }

	env := ir.NewEnv()
	xl := g.NewMemLoadChained("ctor", env, x, []ir.Node{iv(0), iv(1), iv(2)})
	zl := g.NewMemLoadChained("ctor", env, z, []ir.Node{iv(0), iv(1), iv(2)})
	sum := g.NewBinOp("ctor", ir.BinOpAdd, xl, zl)
	mp := g.NewMap("ctor", []ir.Node{sum}, ir.NewLambda(nil, sum), append([]ir.Node(nil), dims...))
	acc := g.NewLambdaInputParam("ctor", "acc", ir.InternScalar(ir.TF32))
	now := g.NewLambdaInputParam("ctor", "now", ir.InternScalar(ir.TF32))
	comb := g.NewBinOp("ctor", ir.BinOpAdd, g.NewParamRef("ctor", acc), g.NewParamRef("ctor", now))
	red := g.NewReduce("ctor", g.NewConst("ctor", ir.InternScalar(ir.TF32), 0), mp,
		ir.NewLambda([]ir.Node{acc, now}, comb))
	store := g.NewMemStoreChained("ctor", env, out, []ir.Node{iv(0), iv(1)}, red)
	g.NewReturnChained("ctor", env, []ir.Node{out}, store)

	if err := (typeinfer.Pass{}).Run(nil, g); err != nil {
		t.Fatalf("fixture typeinfer: %v", err)
	}
	// The collapse-shape record: R=CTA (the single axis), C=SEQ / e4=SEQ
	// (covered), both loads promoted FRAG — each fill covers [C, e4] with
	// one axis (R) and two bivs. The tie is the CacheDims clamp product:
	// all of INS is referenced, so the nil tie's covered set (all INS) is
	// clamped to the outermost level (R) — a real node, per the
	// canonicalization rule.
	cache := opt.NewCache()
	rec := &tiling.Record{Iters: map[ir.Node]tiling.IterDecision{}, Sites: map[uint64]tiling.SiteDecision{}, Verified: true}
	for i, it := range iters {
		biv := ir.DataDeps(it)[0]
		if biv == nil {
			t.Fatalf("iter %d carries no BIV", i)
		}
		rec.Iters[biv] = []tiling.IterDecision{
			{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
			{SoC: tiling.SoCSequential, LoC: 1, TileSize: 2},
			{SoC: tiling.SoCSequential, LoC: 1, TileSize: 2},
		}[i]
	}
	rec.Sites[xl.BasePtr().ID()] = tiling.SiteDecision{Promoted: true, Tie: iters[0], Class: tiling.ClassFRAG}
	rec.Sites[zl.BasePtr().ID()] = tiling.SiteDecision{Promoted: true, Tie: iters[0], Class: tiling.ClassFRAG}
	cache.Put(rec)
	if err := (tiling.LowerPass{Params: tiling.DefaultParams()}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, g); err != nil {
		t.Fatalf("fixture tile-lower: %v", err)
	}
	if err := (schedule.Scheduler{}).Run(nil, g); err != nil {
		t.Fatalf("fixture schedule: %v", err)
	}
	if err := (finallower.Pass{}).Run(&opt.Context{Pass: "final-lower", Cache: cache}, g); err != nil {
		t.Fatalf("fixture final-lower: %v", err)
	}
	return g
}

// fillCoordOrder pins the load-bearing fact: the emitted fill's TensorWindow
// coordinate list is in TENSOR DIM ORDER — the R position (the CTA axis)
// first, then the C round base, then the e4 round base. Targeted (hunk-local)
// regex, not a whole-file match, so the pin survives unrelated emission
// churn; the mixed subtest additionally byte-goldens the whole emission.
var fillCoordOrder = regexp.MustCompile(
	`tca::rt::TensorWindow\((?:x|z)_v, [^,()]*, C_it\d+\.Index\(\), e4_it\d+\.Index\(\)\)`)

// TestFillCoordReconstruction pins D-fix6-03: a fill whose covered dims
// include a compile-time-const dim reconstructs its coordinates in tensor
// dim order instead of failing with "coordinate reconstruction ran out of
// axes".
func TestFillCoordReconstruction(t *testing.T) {
	// The RED gate: pre-fix this subtest FAILS at Emit with `cxx: fill <n>
	// coordinate reconstruction ran out of axes` — the DimRef{Extent}
	// position's empty Name never matches the name-keyed covered set, so it
	// consumes the single axis slot. Post-fix it must emit with the
	// coordinate list in tensor dim order (R axis, C biv, e4 biv).
	// RE-DERIVED under fix7 §4 slice 1b (quick-261101-mt9): this fixture is
	// the qualifying const-innermost shape (one-tier SEQ reduce, between
	// chain [thread-specialized R trip, uniform SEQ C element walk], FRAG
	// fills at X level, element chunk = C's clamp target 2 > 1), so its
	// reducer bracket amortizes — the golden regen carries the SAME
	// load-bearing D-fix6-03 fact (the fill's tensor-dim-order coordinates,
	// asserted by fillCoordOrder below, UNWEAKENED) plus the amortized
	// nest (the chunk-wide acc block, init/store walks, inlined combiner).
	// Red evidence: red-drift2.txt in the task dir; the amortized shape is
	// byte-pinned by testdata/bracket_amort.cu.
	t.Run("mixed dynamic/static", func(t *testing.T) {
		g := fillCoordGraph(t, "fillcoord", "mixed")
		res, err := Emit(g, emitOptions())
		if err != nil {
			t.Fatalf("Emit over the mixed dynamic/static fill (D-fix6-03 RED gate — pre-fix: coordinate reconstruction ran out of axes): %v", err)
		}
		cu := string(res.Files["fillcoord.cu"])
		if !fillCoordOrder.MatchString(cu) {
			t.Fatalf("the fill's TensorWindow coordinate list must be in tensor dim order (R axis, C biv, e4 biv); no match of %s in:\n%s", fillCoordOrder, cu)
		}
		golden.Assert(t, "testdata/fillcoord_mixed.cu", res.Files["fillcoord.cu"])
	})

	// The dynamic-only control: the identical fixture with all-Name DimRefs
	// (the reduce3Graph shape). It emits with the same coordinate order PRE-
	// and POST-fix — proving the fix changed only the const reconciliation,
	// not the dynamic path. No golden: this subtest must stay green at the
	// RED commit too.
	t.Run("dynamic only control", func(t *testing.T) {
		g := fillCoordGraph(t, "fillcoord", "dynamic")
		res, err := Emit(g, emitOptions())
		if err != nil {
			t.Fatalf("Emit over the dynamic-only control: %v", err)
		}
		cu := string(res.Files["fillcoord.cu"])
		if !fillCoordOrder.MatchString(cu) {
			t.Fatalf("the dynamic-only control's coordinate order must match the mixed fixture's; no match of %s in:\n%s", fillCoordOrder, cu)
		}
	})

	// The fully-static all-distinct-extents probe (the [512u,4u,1024u]
	// row_sum class from D-fix6-03's original finding): reconciliation by
	// extent is unambiguous here, so the walk must get PAST stmtFill.
	// Whether emission then completes end-to-end is PROBED and recorded for
	// fix6.md §4 — it is explicitly NOT a gate of this task.
	t.Run("fully static all distinct probe", func(t *testing.T) {
		g := fillCoordGraph(t, "fillcoord", "static")
		res, err := Emit(g, emitOptions())
		if err != nil {
			t.Logf("PROBE fully-static all-distinct: emission rejected at: %v", err)
			return
		}
		cu := string(res.Files["fillcoord.cu"])
		staticOrder := regexp.MustCompile(
			`tca::rt::TensorWindow\((?:x|z)_v, [^,()]*, e4_it\d+\.Index\(\), e1024_it\d+\.Index\(\)\)`)
		if !staticOrder.MatchString(cu) {
			t.Logf("PROBE fully-static all-distinct: emission COMPLETED but the coordinate order regex %s did not match — recorded for fix6.md §4", staticOrder)
			return
		}
		t.Logf("PROBE fully-static all-distinct: emission COMPLETED with tensor-dim-order coordinates")
	})
}
