package cxx

// ---------------------------------------------------------------------------
// fix7 §4 slice 1b RED pin (quick-261101-mt9, design/fix/fix7.md §4)
// ---------------------------------------------------------------------------
//
// Bracket amortization: with the window streamed in chunks (slice 1a), the
// reducer bracket's per-element ceremony (Init + Combine per output element)
// and the fill riding inside that bracket are the dominant residue on the
// const-innermost reduce family. Slice 1b inverts the nest: ONE bracket per
// chunk — an explicit chunk-wide Frag accumulator block is initialized by an
// init walk (the reduce's OWN init value) before the chunk's reduced-dim
// walk X, the per-element walk E moves INSIDE X (the fill rides at X
// level — the §3.4 fill-placement gap closes by construction), the combiner
// body inlines per element as an acc read-modify-write Frag store, and a
// store walk drains the acc block to the output after X. NO SeqReducer
// object, no Init/Reduce/Combine calls — ceremony count, never semantics
// (per-element accumulation stays k-ascending: X rounds ascending, home
// ascending — bit-identical numerics).
//
// The fixture reuses fragChunkGraph's constextent mode EXACTLY (the
// qualifying flat-bracket shape: R=CTA / C=SEQ / e4=SEQ-const, one promoted
// FRAG site — the F6-eviction shape, R_it2 ThrIterator directly above the
// element walk E=C_it3). Qualification (Q0-Q5, design/fix/fix7.md §4 of the
// 261101-mt9 plan): one-tier SeqReducer (Q1), flat-bracket degenerate
// (Q2 — this branch), FRAG-class fills only (Q0), no fill window
// coordinate referencing E's IV (Q3), exactly one uniform SEQ element walk
// above X with a thread-specialized loop above it (Q4), consumers passing
// the moved-consumer gates (Q5).
//
// Re-derivation arithmetic (T-kel-01 / T-mt9-01: policy + numbers recorded
// HERE, never a silently-updated golden): the priced chunk is [C=16, e4=4]
// (Frag<float, 2u, 16u, 4u> = 256 B, pinned by 1a's pin 2). Per (R trip,
// C chunk round) the emitted defect refilled the window once per
// (element, k-round): 16 elements x 4 rounds = 64 fills, each loading 16x4
// floats and consuming one element's worth. Amortized: the fill executes
// once per X round = 4 fills per (R trip, C chunk round) — a 16x ceremony
// cut on the fill path, plus the Init/Combine pair per element (64 pairs)
// replaced by one init walk (16 acc Sets) + one store walk (16 out Sets).
// Expected decls: the acc block tca::rt::Frag<float, 1u, 16u> tca_f1
// (element chunk 16 — E's clamped bound v8 = umin(c7=16u, remaining)); the
// window tca_f0 keeps 1a's [16, 4] decl.
//
// RED gate: pre-1b the emission carries the per-element bracket
// (reducer0.Init() per C_it3 step, SeqReducer construction, Combine after
// X, NO chunk-wide acc decl) — every assertion below fails, and the golden
// is absent. RED evidence: red-task1.txt in the task dir.

import (
	"regexp"
	"strings"
	"testing"

	"tcacomp/internal/golden"
)

var (
	// amortAccDecl pins the chunk-wide accumulator block: one element per
	// output element of the chunk (E's clamp target 16u).
	amortAccDecl = regexp.MustCompile(`tca::rt::Frag<float, 1u, 16u> tca_f1;`)
	// amortXLoop is the reduced-dim rounds walk X (the e4 step-1u between
	// loop over the true const extent 4u).
	amortXLoop = regexp.MustCompile(`for \(auto e4_it\d+ = tca::rt::SeqIterator\(0u, 4u, c2\)`)
	// amortElemWalk is the moved per-element walk E (C, step 1u against the
	// clamped chunk bound v8).
	amortElemWalk = regexp.MustCompile(`for \(auto C_it\d+ = tca::rt::SeqIterator\(0u, 1u, v8\)`)
	// amortHomeWalk is the home walk (e4 inside, step 1u against the umin
	// chain bound v12) — nested inside E after the reshape.
	amortHomeWalk = regexp.MustCompile(`for \(auto e4_it\d+ = tca::rt::SeqIterator\(0u, 1u, v12\)`)
	// amortFillSite is the ONE per-thread fill riding at X level.
	amortFillSite = regexp.MustCompile(`tca::rt::LoadTileThread\(thread_map1, tca_f0, tca::rt::TensorWindow\(x_v, v9, C_it1\.Index\(\), e4_it\d+\.Index\(\)\)\);`)
	// amortAccUpdate pins the inlined combiner body: an acc read-modify-write
	// over (acc.Get(elementIV), piped) — the suite combiner textually renders
	// as tca_f1.Set((tca_f1.Get(c) + v), c).
	amortAccUpdate = regexp.MustCompile(`tca_f1\.Set\(\(tca_f1\.Get\(C_it\d+\.Index\(\)\) \+ v\d+\), C_it\d+\.Index\(\)\);`)
	// amortInitWalk pins the init walk: the construction's OWN init value
	// (c3 = 0.0f, never a hardcoded zero) stored per acc slot.
	amortInitWalk = regexp.MustCompile(`tca_f1\.Set\(c3, C_it\d+\.Index\(\)\);`)
	// amortStoreWalk pins the store walk: the Combine-value consumer drained
	// from the acc block at the store-walk IV, coordinates rebuilt onto it
	// (the trip coordinate renders as the hoisted per-trip local vN).
	amortStoreWalk = regexp.MustCompile(`out_v\.Set\(tca_f1\.Get\(C_it\d+\.Index\(\)\), v\d+, \(C_it\d+\.Index\(\) \+ C_it\d+\.Index\(\)\)\);`)
)

// TestBracketAmortEmission pins fix7 §4 slice 1b: the qualified flat-bracket
// reduce emits the amortized shape — chunk-wide Frag accumulator, zero
// reducer ceremony, one fill per (R trip, C chunk round, X round), the
// element walk nested inside X, the combiner inlined per element, and the
// init/store walks cloning E's header.
//
// RED equivalence: pre-1b the same fixture emits the per-element bracket
// (see testdata/fragchunk_const_extent.cu — reducer0.Init() per C_it3
// step, Combine after X, no tca_f1 decl); every assertion below fails and
// the golden is absent. RED evidence: red-task1.txt.
//
// RE-DERIVED under fix7 §5 slice 1c (quick-261101-oqg): the golden's fill
// dst decl Frag<float, 2u, 16u, 4u> gains the alignas(16) spelling — rank 2
// (RowMajorShape<2> = 16 B, data_ at the 16B-clean offset 16), trailing 4
// (16 B rows): the 1c decl rule qualifies it, the loader's vec16 dst-side
// check holds by construction. The 1b acc block Frag<float, 1u, 16u> is NOT
// a fill dst and stays plain; every other line is byte-identical to the 1b
// golden. The 1b assertions below are UNWEAKENED.
func TestBracketAmortEmission(t *testing.T) {
	g := fragChunkGraph(t, "bracketamort", "constextent", 64, 256, 256)
	res, err := Emit(g, emitOptions())
	if err != nil {
		t.Fatalf("Emit over the amortization fixture: %v", err)
	}
	cu := string(res.Files["bracketamort.cu"])

	// Pin (b): zero reducer ceremony — no construction, no Init/Reduce/
	// Combine sites anywhere.
	if strings.Contains(cu, "SeqReducer") || strings.Contains(cu, "reducer0") {
		t.Fatalf("fix7 §4 1b pin (b): the amortized emission carries NO SeqReducer construction and no Init/Reduce/Combine sites; found reducer ceremony in:\n%s", cu)
	}
	// Pin (a): the chunk-wide accumulator decl (element chunk 16).
	if !amortAccDecl.MatchString(cu) {
		t.Fatalf("fix7 §4 1b pin (a): the kernel-top must declare the chunk-wide acc block %s; no match in:\n%s", amortAccDecl, cu)
	}
	// Pin (c): exactly one fill site, riding at X level — textually BEFORE
	// the element walk (pre-1b the fill sits inside X inside E, i.e. after
	// the element walk opens).
	fill := amortFillSite.FindAllString(cu, -1)
	if len(fill) != 1 {
		t.Fatalf("fix7 §4 1b pin (c): exactly one LoadTileThread fill site, got %d (%v) in:\n%s", len(fill), fill, cu)
	}
	xAt := indexOfMatch(t, cu, amortXLoop, "the X loop (e4 rounds)")
	homeAt := indexOfMatch(t, cu, amortHomeWalk, "the home walk")
	accAt := indexOfMatch(t, cu, amortAccUpdate, "the acc update")
	fillAt := strings.Index(cu, fill[0])
	// The init/E/store walks clone E's header — textually identical C
	// step-1u loops; they are identified by OCCURRENCE ORDER: [0] the init
	// walk, [1] the element walk E (inside X), [2] the store walk.
	cWalks := amortElemWalk.FindAllStringIndex(cu, -1)
	if len(cWalks) != 3 {
		t.Fatalf("fix7 §4 1b pin (b): exactly three C step-1u walks (init, element, store — E's header cloned), got %d in:\n%s", len(cWalks), cu)
	}
	initAt, eAt, storeAt := cWalks[0][0], cWalks[1][0], cWalks[2][0]
	// The full nest order: the init walk opens the bracket, then X { fill;
	// E { home { acc update } } }, then the store walk drains the acc.
	if !(initAt < xAt && xAt < fillAt && fillAt < eAt && eAt < homeAt && homeAt < accAt && accAt < storeAt) {
		t.Fatalf("fix7 §4 1b pin (c/d): the nest order must be init-walk, X { fill; E { home { acc update } } }, store-walk — got init@%d x@%d fill@%d e@%d home@%d acc@%d store@%d in:\n%s", initAt, xAt, fillAt, eAt, homeAt, accAt, storeAt, cu)
	}
	// Pin (d): the element walk is nested inside X before home, and the
	// fill's nest path excludes it (asserted by the order pin above).
	if n := len(amortAccUpdate.FindAllString(cu, -1)); n != 1 {
		t.Fatalf("fix7 §4 1b pin (e): exactly one inlined combiner acc update, got %d in:\n%s", n, cu)
	}
	if n := len(amortInitWalk.FindAllString(cu, -1)); n != 1 {
		t.Fatalf("fix7 §4 1b pin (f): exactly one init-walk acc store (the construction's own init value), got %d in:\n%s", n, cu)
	}
	if n := len(amortStoreWalk.FindAllString(cu, -1)); n != 1 {
		t.Fatalf("fix7 §4 1b pin (f): exactly one store-walk drain (consumer coordinates rebuilt onto the store-walk IV), got %d in:\n%s", n, cu)
	}
	golden.Assert(t, "testdata/bracket_amort.cu", res.Files["bracketamort.cu"])
}

// indexOfMatch returns the index of re's first match, failing the test when
// the pattern is absent (a locator, not an assertion).
func indexOfMatch(t *testing.T, cu string, re *regexp.Regexp, what string) int {
	t.Helper()
	m := re.FindStringIndex(cu)
	if m == nil {
		t.Fatalf("fix7 §4 1b: %s not found (pattern %s) in:\n%s", what, re, cu)
	}
	return m[0]
}
