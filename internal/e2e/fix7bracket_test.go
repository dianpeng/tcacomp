// fix7bracket_test.go — quick-261101-mt9 (fix7 §4 slice 1b) pin: the
// amortized flat-bracket emission through the REAL optimizer, closing the
// §3.4 fill-placement gap as one restructure.
//
// On the qualified reduce pair (row_sum, softmax — the const-innermost
// [R,Ku,C] shape) the emitted .cu carries: (1) ZERO reducer ceremony — no
// SeqReducer construction, no Init()/Reduce()/Combine() sites; (2) exactly
// one LoadTileThread fill site per kernel, whose nest path EXCLUDES the
// @C per-element inside walk (the fill rides at the reduced-dim rounds
// level — the §3.4 gap closed structurally, not patched); (3) the X loop's
// trip count equal to the emitted TRUE const reduced extent (4u row_sum
// from TensorView x_v(x, R, 4u, C); 8u softmax); (4) the chunk-wide acc
// Frag block (Frag<float, 1u, 64u> — E's clamp target). On the gap family
// (axpy3, scale_copy) the same fill-nest-path assertion holds suite-wide;
// on layernorm/transpose the emission is UNCHANGED (their Q4-failing shape
// — a second uniform SEQ walk above the element walk — keeps today's
// per-element bracket: the 1-bracket + fill-per-element structure is a
// regression guard).
//
// Sources are embedded VERBATIM from testing/kernels (the fix5t2/fix7stream
// embedding idiom — the pin must not drift with suite edits); rowSumSrc is
// shared with fix7stream_test.go (same package).

package e2e

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// softmaxSrc is testing/kernels/fusion/softmax.tca verbatim.
const softmaxSrc = `// softmax — the softmax reduction core with a mask/weight operand z (the
// attention-mask pattern), over the middle axis: out[i][j] = sum_k
// exp(x[i][k][j]) * z[i][k][j] — the batched masked-softmax normalizer.
//
// FALLBACK VARIANTS (spike walls, recorded in SUMMARY): (1) the full
// softmax is not expressible in the v0 frontend — rank-2 dynamic
// intermediates are rejected by the builder ("a multiplicative GMEM size
// is not constructed"); the final normalize division is a rank-preserving
// map-only chain, rejected by the v0 pkg rungs ("require at least one
// reduction before the store"; a reduce must drop exactly one axis); and
// reading a rank-1 dynamic intermediate (a max-reduce result) from a
// downstream chain trips a typeinfer invariant ("memload operand is not a
// tensor parameter"), while an unused max-reduce let trips final-lower
// ("cannot erase node: 1 use(s) still point at it"); (2) the [R,C] -> [R]
// normalizer — single- or two-input — selects the v0-unreachable
// GlobalReducer in the schedule (every rank-1-output reduction probed
// does; recorded as a compiler defect); the shipped rank-2-output
// two-input form schedules the WARP-reducer class.
// const-K declaration (mixed dynamic/static form per design/fix/fix6.md —
// fully-static is blocked by D-fix6-01/03): the MIDDLE reduced extent is a
// compile-time constant, Ku = 8.
@runtime(cxx_path="runtime/tca_runtime.h");
@name(cxx="Softmax")
kernel softmax(
  x: tensor<dtype=f32, shape=[R, 8u, C]>,
  z: tensor<dtype=f32, shape=[R, 8u, C]>,
) -> tensor<dtype=f32, shape=[R, C]>
{
  dimvar (i, k, j);
  let out = map
    on(x[i][k][j], z[i][k][j])
    (v: f32, w: f32) -> f32:
      expf(v) * w
  | reduce(init = 0.0f)
    on(acc: f32, %0: f32)
    as (i, j) -> f32:
      acc + %0
  ;
  return out;
}
`

var (
	// reducerCeremonyRe matches any surviving reducer ceremony site.
	reducerCeremonyRe = regexp.MustCompile(`SeqReducer|reducer\d+\.(Init|Reduce|Combine)`)
	// accDeclRe matches the chunk-wide accumulator block decl.
	accDeclRe = regexp.MustCompile(`tca::rt::Frag<float, 1u, \d+u> tca_f\d+;`)
	// trueExtentRe captures a hoisted true-const reduced extent local
	// (vN = 4u / vN = 8u — the X loop's bound after 1a's true-extent
	// chunking).
	trueExtentRe = regexp.MustCompile(`(?m)auto (v\d+) = ([48])u;$`)
	// cElemWalkRe matches one of the cloned C element walks (E's header:
	// step 1u against the clamped chunk bound). The init/E/store walks are
	// textually identical — E is identified POSITIONALLY (the one whose
	// scope carries the acc update / the Reduce call).
	cElemWalkRe = regexp.MustCompile(`for \(auto C_it\d+ = tca::rt::SeqIterator\(0u, 1u,`)
	// accUpdateRe matches the inlined combiner acc update.
	accUpdateRe = regexp.MustCompile(`tca_f\d+\.Set\(\(tca_f\d+\.Get\(C_it\d+\.Index\(\)\) \+ v\d+\), C_it\d+\.Index\(\)\);`)
	// reduceCallSiteRe matches a per-element Reduce call (the unqualified
	// shape's bracket body).
	reduceCallSiteRe = regexp.MustCompile(`reducer\d+\.Reduce\(v\d+\);`)
)

// enclosingLoopBefore returns the index of the last cElemWalkRe loop
// header starting before pos — the innermost cloned-C element walk
// enclosing the given position.
func enclosingLoopBefore(cu string, pos int) int {
	last := -1
	for _, m := range cElemWalkRe.FindAllStringIndex(cu, -1) {
		if m[0] < pos {
			last = m[0]
		}
	}
	return last
}

// TestFix7Bracket is fix7 §4 slice 1b's e2e pin: the real optimizer emits
// the amortized bracket on the qualified reduce pair, the fill's nest path
// excludes the per-element walk (the §3.4 gap closed structurally), and
// the Q4-failing pair (layernorm/transpose) keeps the per-element bracket
// UNCHANGED.
//
// RED equivalence: at the pin's commit the reduce pair still emits the
// per-element bracket (reducer0.Init() per C_it3 step — the fix7stream
// pin's emission shows the shape), so the amortization assertions fail;
// the layernorm/transpose regression-guard subtests PASS at RED (they pin
// the current shape — they turn into regression guards at GREEN). RED
// evidence: red-task2.txt in the task dir.
func TestFix7Bracket(t *testing.T) {
	t.Run("row_sum", func(t *testing.T) {
		cu := compileTcaSource(t, "row_sum.tca", rowSumSrc)
		assertAmortizedBracket(t, "row_sum", cu, "4")
	})
	t.Run("softmax", func(t *testing.T) {
		cu := compileTcaSource(t, "softmax.tca", softmaxSrc)
		assertAmortizedBracket(t, "softmax", cu, "8")
	})
	t.Run("layernorm unchanged", func(t *testing.T) {
		cu := compileSuiteKernel(t, "llm", "layernorm.tca")
		assertPerElementBracketUnchanged(t, "layernorm", cu)
	})
	t.Run("transpose unchanged", func(t *testing.T) {
		cu := compileSuiteKernel(t, "tiling", "transpose.tca")
		assertPerElementBracketUnchanged(t, "transpose", cu)
	})
	t.Run("suite fill nest paths", func(t *testing.T) {
		// The §3.4 gap family: on every qualified gap kernel the fill rides
		// at the reduced-dim rounds level — the nest path excludes the @C
		// per-element inside walk. Sources are read from the suite tree
		// (the 1a audit's sweep idiom).
		for _, k := range []struct{ group, file string }{
			{"fusion", "axpy3.tca"},
			{"runtime", "scale_copy.tca"},
			{"runtime", "row_sum.tca"},
			{"fusion", "softmax.tca"},
		} {
			cu := compileSuiteKernel(t, k.group, k.file)
			assertFillOutsideElementWalk(t, k.file, cu)
		}
	})
}

// assertFillOutsideElementWalk pins the §3.4 closure on one suite kernel:
// the fill rides OUTSIDE the element walk E (the innermost cloned-C walk
// enclosing the acc update) and no reducer ceremony survives.
func assertFillOutsideElementWalk(t *testing.T, file string, cu string) {
	t.Helper()
	sites := fillSiteRe.FindAllString(cu, -1)
	if len(sites) == 0 {
		t.Fatalf("fix7 §4 1b (%s): no LoadTileThread fill site found", file)
	}
	acc := accUpdateRe.FindStringIndex(cu)
	if acc == nil {
		t.Fatalf("fix7 §4 1b (%s): the inlined combiner acc update is absent", file)
	}
	eAt := enclosingLoopBefore(cu, acc[0])
	for _, s := range sites {
		path := nestPathOf(t, cu, s)
		t.Logf("§4 %s: fill nest-path=%s", file, path)
		if fillAt := strings.Index(cu, s); fillAt > eAt {
			t.Fatalf("fix7 §4 1b (%s): the fill rides INSIDE the element walk (fill@%d > e-walk@%d) — the §3.4 gap is NOT closed", file, fillAt, eAt)
		}
	}
	if reducerCeremonyRe.MatchString(cu) {
		t.Fatalf("fix7 §4 1b (%s): the qualified emission carries reducer ceremony (the amortized shape must have none)", file)
	}
}

// assertAmortizedBracket pins the amortized emission contract on one
// qualified reduce kernel: zero reducer ceremony, one fill site OUTSIDE
// the element walk E (the walk whose scope carries the acc update — the
// §3.4 gap closed structurally), the X loop tripping the true const
// reduced extent, and the chunk-wide acc block.
func assertAmortizedBracket(t *testing.T, kernel, cu, wantExtent string) {
	t.Helper()
	if reducerCeremonyRe.MatchString(cu) {
		m := reducerCeremonyRe.FindString(cu)
		t.Fatalf("fix7 §4 1b (%s): zero reducer ceremony expected (no SeqReducer construction, no Init/Reduce/Combine sites); found %q", kernel, m)
	}
	if !accDeclRe.MatchString(cu) {
		t.Fatalf("fix7 §4 1b (%s): the chunk-wide acc Frag block decl must be present; no match of %s", kernel, accDeclRe)
	}
	sites := fillSiteRe.FindAllString(cu, -1)
	if len(sites) != 1 {
		t.Fatalf("fix7 §4 1b (%s): exactly one LoadTileThread fill site, got %d", kernel, len(sites))
	}
	path := nestPathOf(t, cu, sites[0])
	t.Logf("§4 %s: fill nest-path=%s", kernel, path)
	// The fill must sit OUTSIDE the element walk E — identified
	// positionally as the innermost cloned-C walk enclosing the acc
	// update. (The init walk clones E's header, so the raw comment path
	// cannot distinguish them — the positional check is the structural
	// assertion.)
	acc := accUpdateRe.FindStringIndex(cu)
	if acc == nil {
		t.Fatalf("fix7 §4 1b (%s): the inlined combiner acc update is absent; the amortized shape is not emitted", kernel)
	}
	eAt := enclosingLoopBefore(cu, acc[0])
	if eAt < 0 {
		t.Fatalf("fix7 §4 1b (%s): no cloned-C element walk encloses the acc update", kernel)
	}
	fillAt := strings.Index(cu, sites[0])
	if fillAt > eAt {
		t.Fatalf("fix7 §4 1b (%s): the fill rides INSIDE the element walk (fill@%d > e-walk@%d) — the §3.4 gap is NOT closed", kernel, fillAt, eAt)
	}
	// The X loop's trip count equals the emitted true const reduced
	// extent: a hoisted `auto vN = <extent>u;` feeding an e4/e8 step-1u
	// between loop over that very local.
	ext := trueExtentRe.FindAllStringSubmatch(cu, -1)
	found := false
	for _, e := range ext {
		if e[2] != wantExtent {
			continue
		}
		if regexp.MustCompile(`SeqIterator\(0u, 1u, ` + e[1] + `\)`).MatchString(cu) {
			found = true
		}
	}
	if !found {
		t.Fatalf("fix7 §4 1b (%s): the X loop must trip the true const reduced extent %su (a hoisted `auto vN = %su;` feeding SeqIterator(0u, 1u, vN)); no match in the emission", kernel, wantExtent, wantExtent)
	}
}

// assertPerElementBracketUnchanged is the regression guard for the
// Q4d-failing pair: layernorm/transpose keep today's per-element bracket —
// one Init site per element walk, the fill riding INSIDE the element walk
// (the walk enclosing the per-element Reduce call).
func assertPerElementBracketUnchanged(t *testing.T, kernel, cu string) {
	t.Helper()
	nInit := len(regexp.MustCompile(`reducer\d+\.Init\(\)`).FindAllString(cu, -1))
	if nInit != 1 {
		t.Fatalf("fix7 §4 1b (%s regression guard): the Q4d-failing shape must keep exactly one per-element bracket (one Init site); got %d", kernel, nInit)
	}
	if !strings.Contains(cu, "SeqReducer") {
		t.Fatalf("fix7 §4 1b (%s regression guard): the SeqReducer construction must survive on the unqualified shape", kernel)
	}
	sites := fillSiteRe.FindAllString(cu, -1)
	if len(sites) == 0 {
		t.Fatalf("fix7 §4 1b (%s regression guard): no fill site found", kernel)
	}
	red := reduceCallSiteRe.FindStringIndex(cu)
	if red == nil {
		t.Fatalf("fix7 §4 1b (%s regression guard): the per-element Reduce call is absent", kernel)
	}
	eAt := enclosingLoopBefore(cu, red[0])
	if eAt < 0 {
		t.Fatalf("fix7 §4 1b (%s regression guard): no C element walk encloses the Reduce call", kernel)
	}
	for _, s := range sites {
		path := nestPathOf(t, cu, s)
		t.Logf("§4 %s: fill nest-path=%s", kernel, path)
		fillAt := strings.Index(cu, s)
		if fillAt < eAt {
			t.Fatalf("fix7 §4 1b (%s regression guard): the unqualified shape must keep the fill inside the per-element walk; the fill escaped it (fill@%d < e-walk@%d)", kernel, fillAt, eAt)
		}
	}
}

// compileSuiteKernel compiles one suite kernel straight from the
// testing/kernels tree (the fix7stream audit sweep's idiom).
func compileSuiteKernel(t *testing.T, group, file string) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", "..", "testing", "kernels", group, file))
	if err != nil {
		t.Fatalf("resolve testing/kernels/%s/%s: %v", group, file, err)
	}
	src, err := os.ReadFile(root)
	if err != nil {
		t.Fatalf("read %s: %v", root, err)
	}
	return compileTcaSource(t, file, string(src))
}
