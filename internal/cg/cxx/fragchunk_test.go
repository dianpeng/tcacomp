package cxx

// ---------------------------------------------------------------------------
// fix7 §3 slice 1a RED pins (quick-261101-kel, design/fix/fix7.md §3.5)
// ---------------------------------------------------------------------------
//
// fix7 §2 pathology 1: a promoted FRAG site whose raw per-thread window
// exceeds its priced block footprint is declared at the RAW window
// (Frag<float, 2u, 64u, 64u> = 16 KB/thread, dynamically indexed -> nvcc
// spills the whole array to local memory/DRAM), while the v62 pricing
// (judge.go siteWindowBytes) already pays only min(raw, MaxFragPerThread)
// per site. Slice 1a makes the allocation, the round structure, and the
// inside clamps honor that contract: the buffer's InternTensor dims BECOME
// the chunk (walk.go prints the type's dims verbatim — the buffer type is
// the ONE source of truth), the tile-between rounds step by the chunk, and
// the inside walks clamp to the chunk via the existing umin chain.
//
// The fixtures below follow the fillcoord_test.go idiom exactly:
// frontend-faithful DimRefs, TWO co-traversed loads (the overlap admission
// filters a singleton set at decide time), a crafted VERIFIED record, the
// real pipeline (tiling.Lower -> schedule -> finallower) to Emit, targeted
// regexes PLUS a byte golden under testdata/.
//
// Exactly ONE of the two co-traversed loads is stamped promoted — the same
// shape the real search stamps when a site's raw window prices at the full
// budget: the F6 eviction keeps one site at min(raw, MaxFragPerThread) so
// the kernel-wide sum fits (fix7 §2 quotes scale_copy's real emission with
// a single tca_f0 decl), and the lower's resolveBudget backstop
// (lower_cache.go) would otherwise halve the SHARED covered tiles until
// the priced sum fits, defeating the fixture. The unpromoted z stays a
// plain GMEM load — the emission control.
//
// Pin 1 re-derivation (the chunk policy, T-kel-01: policy + arithmetic
// recorded HERE, never a silently-updated golden): covered [C, K] both
// dynamic SEQ at T=64, float32 (4 B), MaxFragPerThread = 256. Raw window =
// 64*64*4 = 16,384 B; the priced budget = siteWindowBytes =
// min(16384, 256) = 256 B. Chunk policy: while sizeof(chunk) > budget,
// halve the OUTERMOST covered dim whose chunk > 1 (floor 1):
// [64,64] -> [32,64] -> ... -> [1,64] = 1*64*4 = 256 B exactly. Expected
// decl: tca::rt::Frag<float, 2u, 1u, 64u>; C's between loop steps 1u (the
// rounds ARE the chunk loops), K's between keeps 64u (its chunk == T), C's
// inside-walk clamp target is 1u, and both fills' TensorWindow anchors
// ride the chunk-stepped between iterators (C_itM.Index(), K_itK.Index()).

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

// fragChunkGraph builds the rank-3 budget-clamp fixture: dims R (dynamic,
// the CTA axis), C (dynamic SEQ), and a third dim that is dynamic "K" in
// mode "dynamic" or the frontend const e4 in mode "constextent"
// (ir.DimRef{Extent:4} with an EMPTY Name, the dim InputParam named "e4" —
// exactly build.go dimKey's spelling). Two co-traversed loads x/z over
// [R, C, third], a reduce(init=0, +) over the innermost dim of x+z, and an
// out[R, C] store. The crafted record assigns R=CTA / C=SEQ(tileC) /
// third=SEQ(tileK) and promotes both loads FRAG tied at the R iter (the
// C1b tier rule: R is a non-covered CTA INS iter), so each promoted site's
// fill covers [C, third] with one axis (R) and two bivs — the exact shape
// whose window fix7 §3 streams through a budget-clamped chunk.
func fragChunkGraph(t *testing.T, name, mode string, tileC, tileK, maxFrag int) *ir.Graph {
	t.Helper()
	g := ir.NewGraph(name)
	u32 := ir.InternScalar(ir.TU32)

	thirdName := "K"
	var thirdRef ir.DimRef
	extents := []float64{64, 128, 128}
	switch mode {
	case "dynamic":
		thirdRef = ir.DimRef{Name: "K"}
	case "constextent":
		// The frontend const dim: canonical "e<extent>" name spelling,
		// DimRef{Extent} position with an EMPTY Name (build.go dimKey).
		thirdName = "e4"
		thirdRef = ir.DimRef{Extent: 4}
		extents = []float64{64, 128, 4}
	default:
		t.Fatalf("unknown fragchunk mode %q", mode)
	}

	dims := []ir.Node{
		g.NewInputParam("ctor", ir.RoleDim, "R", nil),
		g.NewInputParam("ctor", ir.RoleDim, "C", nil),
		g.NewInputParam("ctor", ir.RoleDim, thirdName, nil),
	}
	xRefs := []ir.DimRef{{Name: "R"}, {Name: "C"}, thirdRef}
	x := g.NewInputParam("ctor", ir.RoleTensor, "x", ir.InternTensor(ir.TF32, xRefs))
	z := g.NewInputParam("ctor", ir.RoleTensor, "z", ir.InternTensor(ir.TF32, xRefs))
	out := g.NewInputParam("ctor", ir.RoleTensor, "out",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "R"}, {Name: "C"}}))

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
	// The crafted record mirrors the collapse shape: R=CTA (the single
	// axis), C / third SEQ (covered), ONE load promoted FRAG tied at
	// iters[0] (the F6-eviction shape — see the file header; the fill
	// covers [C, third] with one axis and two bivs). z stays a GMEM load.
	cache := opt.NewCache()
	rec := &tiling.Record{Iters: map[ir.Node]tiling.IterDecision{}, Sites: map[uint64]tiling.SiteDecision{}, Verified: true}
	for i, it := range iters {
		biv := ir.DataDeps(it)[0]
		if biv == nil {
			t.Fatalf("iter %d carries no BIV", i)
		}
		rec.Iters[biv] = []tiling.IterDecision{
			{SoC: tiling.SoCTA, LoC: 100, TileSize: 32},
			{SoC: tiling.SoCSequential, LoC: 1, TileSize: tileC},
			{SoC: tiling.SoCSequential, LoC: 1, TileSize: tileK},
		}[i]
	}
	rec.Sites[xl.BasePtr().ID()] = tiling.SiteDecision{Promoted: true, Tie: iters[0], Class: tiling.ClassFRAG}
	cache.Put(rec)
	params := tiling.DefaultParams()
	params.MaxFragPerThread = maxFrag
	if err := (tiling.LowerPass{Params: params}).Run(&opt.Context{Pass: "tile-lower", Cache: cache}, g); err != nil {
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

var (
	// fragChunkDecl pins the chunked allocation (pin 1a): dims = the chunk
	// [1, 64] (256 B), never the raw 64x64 window (16 KB/thread).
	fragChunkDecl = regexp.MustCompile(`tca::rt::Frag<float, 2u, 1u, 64u> tca_f\d+;`)
	// fragChunkRawDecl is the pre-1a pathology: the raw-window decl must be
	// gone from every emitter.
	fragChunkRawDecl = regexp.MustCompile(`tca::rt::Frag<float, 2u, 64u, 64u>`)
	// fragChunkCStep pins the stream structure (pin 1b): the clamped dim's
	// tile-between loop advances by the CHUNK (step 1u against the hoisted
	// extent const), NOT by the nominal T=64. (The inside walks also step
	// 1u, but against a computed bound — the cN-hoisted-extent form is
	// unique to a between loop.)
	fragChunkCStep = regexp.MustCompile(`SeqIterator\(0u, 1u, c\d+\)`)
	// fragChunkKStep pins the innermost covered dim's between loop keeping
	// its full-tile step (its chunk == T == 64).
	fragChunkKStep = regexp.MustCompile(`SeqIterator\(0u, 64u, c\d+\)`)
	// fragChunkCClamp pins the inside-walk clamp target (pin 1b): a hoisted
	// 1u const exists feeding the C dim's umin clamp chain (pre-1a the
	// clamp const is 64u; the inside walks' step literals are inline, never
	// hoisted, so a hoisted "= 1u;" const is the clamp target exactly).
	fragChunkCClamp = regexp.MustCompile(`(?m)auto c\d+ = 1u;$`)
	// fragChunkAnchors pins the per-round window anchor (pin 1c): the
	// fill's TensorWindow coordinates ride the between iterators of the two
	// covered dims — the C anchor is the chunk-stepped iterator (the CStep
	// pin proves its loop steps 1u).
	fragChunkAnchors = regexp.MustCompile(
		`tca::rt::TensorWindow\(x_v, [^,()]*, C_it\d+\.Index\(\), K_it\d+\.Index\(\)\)`)

	// Pin 2 regexes (const true extent — see the subtest comment).
	fragChunkConstDecl    = regexp.MustCompile(`tca::rt::Frag<float, 2u, 16u, 4u> tca_f\d+;`)
	fragChunkConstRawDecl = regexp.MustCompile(`tca::rt::Frag<float, 2u, \d+u, 256u>`)
	fragChunkConstCStep   = regexp.MustCompile(`SeqIterator\(0u, 16u, c\d+\)`)
	fragChunkConstE4Step  = regexp.MustCompile(`SeqIterator\(0u, 4u, c\d+\)`)
	fragChunkConstAnchors = regexp.MustCompile(
		`tca::rt::TensorWindow\(x_v, [^,()]*, C_it\d+\.Index\(\), e4_it\d+\.Index\(\)\)`)
)

// TestFragChunkStream pins fix7 §3 slice 1a pin 1: a promoted FRAG site
// whose raw per-thread window exceeds its priced block footprint emits the
// streaming chunk — a Frag declaration sized to the budget-clamped chunk
// ([1, 64] = 256 B), chunk-stepped tile-between rounds, chunk-clamped
// inside walks, and per-round anchors riding the chunk-stepped iterators.
//
// RED gate: pre-1a the decl prints the raw window (Frag<float, 2u, 64u,
// 64u>), the C between loop steps 64u, no hoisted 1u clamp const exists,
// and the golden file is absent — the subtest fails at every assertion.
func TestFragChunkStream(t *testing.T) {
	t.Run("dynamic budget clamp", func(t *testing.T) {
		g := fragChunkGraph(t, "fragchunk", "dynamic", 64, 64, 256)
		res, err := Emit(g, emitOptions())
		if err != nil {
			t.Fatalf("Emit over the dynamic budget-clamp fixture: %v", err)
		}
		cu := string(res.Files["fragchunk.cu"])
		if fragChunkRawDecl.MatchString(cu) {
			t.Fatalf("fix7 §3 1a: the raw-window Frag decl must be gone (the chunked decl replaces it); found %s in:\n%s", fragChunkRawDecl, cu)
		}
		if !fragChunkDecl.MatchString(cu) {
			t.Fatalf("fix7 §3 1a pin 1a: the decl must be sized to the chunk [1, 64] (256 B), re-derivation in the file header; no match of %s in:\n%s", fragChunkDecl, cu)
		}
		if !fragChunkCStep.MatchString(cu) {
			t.Fatalf("fix7 §3 1a pin 1b: the clamped dim's tile-between loop must advance by the chunk (SeqIterator step 1u); no match of %s in:\n%s", fragChunkCStep, cu)
		}
		if !fragChunkKStep.MatchString(cu) {
			t.Fatalf("fix7 §3 1a pin 1b control: the innermost covered dim's between loop keeps its full-tile step (chunk == T == 64); no match of %s in:\n%s", fragChunkKStep, cu)
		}
		if !fragChunkCClamp.MatchString(cu) {
			t.Fatalf("fix7 §3 1a pin 1b: the inside-walk clamp target must be the chunk (a hoisted 1u const feeding the umin chain); no match of %s in:\n%s", fragChunkCClamp, cu)
		}
		if n := len(fragChunkAnchors.FindAllString(cu, -1)); n != 1 {
			t.Fatalf("fix7 §3 1a pin 1c: the fill's TensorWindow anchor must ride the covered dims' between iterators (x's window); got %d matches of %s in:\n%s", n, fragChunkAnchors, cu)
		}
		golden.Assert(t, "testdata/fragchunk_stream.cu", res.Files["fragchunk.cu"])
	})

	// Pin 2 (fix7 §3 slice 1a): a covered COMPILE-TIME-CONST dim sizes its
	// chunk by its TRUE extent — the suite's exact pathology-2 shape: e4
	// covered at the search's NOMINAL tile T=256 while its true const
	// extent is 4. Re-derivation (policy + arithmetic, T-kel-01): raw =
	// 64*256*4 = 65,536 B, priced budget = min(65536, 256) = 256 B. The
	// e4 chunk base reduces to the TRUE extent 4 (NEVER the nominal 256;
	// the const source precedence is pinned in
	// lower_fragchunk.go staticIterExtent — the BIV max operand's const
	// payload, then the dim param's canonical "e<extent>" spelling; NEVER
	// ir.TypeOf, the nil-computed-cell gotcha). Then the halving loop:
	// [64,4] = 1,024 B > 256 -> C 64->32 (512 B) -> 16 (256 B = budget).
	// Expected: decl Frag<float, 2u, 16u, 4u>, C between step 16u, e4
	// between step 4u, no 256 anywhere in the decl. RED at the pin's
	// commit: the dynamic path chunks e4 by its nominal base (the halving
	// loop crushes C to 1, then e4 256->...->8), emitting
	// Frag<float, 2u, 1u, 8u>-class — pathology 2 alive.
	// RE-DERIVED under fix7 §4 slice 1b (quick-261101-mt9): this fixture is
	// the qualifying const-innermost shape (one-tier SEQ reduce, between
	// chain [thread-specialized R trip, uniform SEQ C element walk], FRAG
	// fills at X level, element chunk 16 > 1), so its reducer bracket
	// amortizes: the golden regen carries the SAME chunk facts (decl
	// [16, 4], C between step 16u, e4 between step 4u, the per-round
	// anchors) plus the amortized nest — the acc block Frag<float, 1u, 16u>
	// tca_f1, the init walk (c3 = the construction's own 0.0f init), the
	// element walk moved inside X, the inlined combiner, the store walk.
	// The 1a chunk assertions below are UNWEAKENED; red evidence:
	// red-drift3.txt in the task dir; the amortized shape is byte-pinned
	// by testdata/bracket_amort.cu.
	t.Run("const true extent", func(t *testing.T) {
		g := fragChunkGraph(t, "fragchunkconst", "constextent", 64, 256, 256)
		res, err := Emit(g, emitOptions())
		if err != nil {
			t.Fatalf("Emit over the const-extent fixture (pathology 2 RED gate): %v", err)
		}
		cu := string(res.Files["fragchunkconst.cu"])
		if fragChunkConstRawDecl.MatchString(cu) {
			t.Fatalf("fix7 §3 1a pin 2: the decl must size the const dim by its TRUE extent 4, never the nominal T=256; found %s in:\n%s", fragChunkConstRawDecl, cu)
		}
		if !fragChunkConstDecl.MatchString(cu) {
			t.Fatalf("fix7 §3 1a pin 2: the chunk must be [16, 4] (256 B), re-derivation in the subtest comment; no match of %s in:\n%s", fragChunkConstDecl, cu)
		}
		if !fragChunkConstCStep.MatchString(cu) {
			t.Fatalf("fix7 §3 1a pin 2: C's between loop must advance by the chunk (SeqIterator step 16u); no match of %s in:\n%s", fragChunkConstCStep, cu)
		}
		if !fragChunkConstE4Step.MatchString(cu) {
			t.Fatalf("fix7 §3 1a pin 2: the const dim's between loop must advance by its TRUE extent (SeqIterator step 4u, not 256u); no match of %s in:\n%s", fragChunkConstE4Step, cu)
		}
		if n := len(fragChunkConstAnchors.FindAllString(cu, -1)); n != 1 {
			t.Fatalf("fix7 §3 1a pin 2: the fill's TensorWindow anchor must ride the covered dims' between iterators (C chunk-stepped, e4 true-extent-stepped); got %d matches of %s in:\n%s", n, fragChunkConstAnchors, cu)
		}
		golden.Assert(t, "testdata/fragchunk_const_extent.cu", res.Files["fragchunkconst.cu"])
	})
}
