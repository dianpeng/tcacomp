// fix7stream_test.go — quick-261101-kel (fix7 §3 slice 1a) pin 3 + the
// §3.4 actual-side reuse data.
//
// Pin 3 (design/fix/fix7.md §3.5): scale_copy and row_sum emitted through
// the REAL optimizer (the fix5t2_one_tier_test.go e2e idiom — no
// hand-stamped record) declare the STREAMED chunk, never the raw 256
// KB/thread window (Frag<float, 2u, 256u, 256u> — nvcc spills a
// dynamically-indexed array that size to local memory/DRAM), with the
// fill's TensorWindow coordinate lists in TENSOR DIM ORDER (the c8n
// D-fix6-03 semantics, unchanged by 1a). The subtests also report the
// LoadTileThread fill-site count per nest path — the §3.4 actual-side
// audit data (the priced side is computed from the tiling package's OWN
// pricing functions in internal/opt/tiling/reuse_audit_test.go; the
// composed record lives in the task dir's reuse-audit.txt).
//
// Sources are testing/kernels/runtime/{scale_copy,row_sum}.tca VERBATIM
// (the fix5t2 embedding idiom — the pin must not drift with suite edits;
// the §3.4 audit reads the suite files from disk and covers all eight
// schedule-changed emitters).

package e2e

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"tcacomp/internal/opt"
	"tcacomp/internal/opt/tiling"
	"tcacomp/internal/pipeline"
	"tcacomp/internal/syntax"
	"tcacomp/internal/syntax/lexer"
)

// scaleCopySrc is testing/kernels/runtime/scale_copy.tca verbatim.
const scaleCopySrc = `// scale_copy — elementwise affine map + tiled store coverage.
//
// FALLBACK VARIANTS (spike walls, recorded in SUMMARY): (1) the v0 pkg
// rungs require at least one reduction before ANY store, so a pure
// no-reducer elementwise kernel is not expressible — the affine map lowers
// over a trailing reduced axis; (2) the single-input reduction of that
// shape selects the v0-unreachable GlobalReducer (recorded compiler
// defect), so a second operand z weights the affine term:
// out[i][j] = sum_k (gain * x + bias) * z. The elementwise map and the
// tiled store remain exercised.
// const-K declaration (mixed dynamic/static form per design/fix/fix6.md —
// fully-static is blocked by D-fix6-01/03): the trailing reduced extent is
// a compile-time constant, Ku = 4.
@runtime(cxx_path="runtime/tca_runtime.h");
@name(cxx="ScaleCopy")
kernel scale_copy(
  x: tensor<dtype=f32, shape=[R, C, 4u]>,
  z: tensor<dtype=f32, shape=[R, C, 4u]>,
  gain: f32,
  bias: f32,
) -> tensor<dtype=f32, shape=[R, C]>
{
  dimvar (i, j, k);
  let out = map
    on(x[i][j][k], z[i][j][k])
    [gain, bias] (v: f32, w: f32) -> f32:
      (gain * v + bias) * w
  | reduce(init = 0.0f)
    on(acc: f32, %0: f32)
    as (i, j) -> f32:
      acc + %0
  ;
  return out;
}
`

// rowSumSrc is testing/kernels/runtime/row_sum.tca verbatim.
const rowSumSrc = `// row_sum — reducer coverage: per-(i, j) weighted sum over the middle
// axis, out[i][j] = sum_k x[i][k][j] * z[i][k][j], over x,z[R,K,C] ->
// [R,C] (the batched row-reduction / per-row dot-product pattern).
//
// FALLBACK VARIANTS (spike walls, recorded in SUMMARY): (1) the plain
// single-input [R,C] -> [R] row sum selects the v0-unreachable
// GlobalReducer in the schedule (the runtime asserts "GlobalReducer: not
// enabled in v0" at launch — recorded as a compiler defect), and the
// two-input rank-1-output form selects it as well — every rank-1-output
// reduction probed does; (2) the shipped rank-2-output two-input form
// schedules the WARP-reducer class. The row-reduction coverage is
// unchanged.
// const-K declaration (mixed dynamic/static form per design/fix/fix6.md —
// fully-static is blocked by D-fix6-01/03): the MIDDLE reduced extent is a
// compile-time constant, Ku = 4.
@runtime(cxx_path="runtime/tca_runtime.h");
@name(cxx="RowSum")
kernel row_sum(
  x: tensor<dtype=f32, shape=[R, 4u, C]>,
  z: tensor<dtype=f32, shape=[R, 4u, C]>,
) -> tensor<dtype=f32, shape=[R, C]>
{
  dimvar (i, k, j);
  let out = map
    on(x[i][k][j], z[i][k][j])
    (v: f32, w: f32) -> f32:
      v * w
  | reduce(init = 0.0f)
    on(acc: f32, %0: f32)
    as (i, j) -> f32:
      acc + %0
  ;
  return out;
}
`

// fragDeclRe captures a FRAG declaration's dims for the budget assertion.
var fragDeclRe = regexp.MustCompile(`tca::rt::Frag<float, 2u, (\d+)u, (\d+)u>`)

// fillSiteRe captures one fill site: the buffer local, the source tensor
// view, and the full coordinate list.
var fillSiteRe = regexp.MustCompile(
	`tca::rt::LoadTileThread\([^,]+, (tca_f\d+), tca::rt::TensorWindow\(([xz])_v, ([^;]*)\)\);`)

// TestFix7StreamEmission is fix7 §3 slice 1a pin 3: the real optimizer
// emits the streamed chunk on the suite's reduce-pair emitters.
//
// RED equivalence: pre-1a (base 6b3d2df) both emitters declared the raw
// window — design/fix/fix7.md §2 quotes the actual pre-merge scale_copy
// emission (tca::rt::Frag<float, 2u, 256u, 256u> = 256 KB/thread), and
// red-task1.txt pins the same raw-window decl at the unit level. At the
// Task-2 state the machinery is live, so this pin lands GREEN on arrival
// (the mechanism owner is Tasks 1-2; this pin guards the REAL optimizer
// path end-to-end, which the unit fixtures cannot).
func TestFix7StreamEmission(t *testing.T) {
	t.Run("scale_copy", func(t *testing.T) {
		cu := compileTcaSource(t, "scale_copy.tca", scaleCopySrc)
		assertChunkedStream(t, "scale_copy", cu,
			// tensor dim order [R, C, e4]: the covered window rides the C
			// round base, then the e4 round base (c8n semantics).
			regexp.MustCompile(`tca::rt::TensorWindow\([xz]_v, [^,()]*, C_it\d+\.Index\(\), e4_it\d+\.Index\(\)\)`))
	})
	t.Run("row_sum", func(t *testing.T) {
		cu := compileTcaSource(t, "row_sum.tca", rowSumSrc)
		assertChunkedStream(t, "row_sum", cu,
			// tensor dim order [R, e4, C] (const MIDDLE): the covered
			// window rides the e4 round base, then the C round base.
			regexp.MustCompile(`tca::rt::TensorWindow\([xz]_v, [^,()]*, e4_it\d+\.Index\(\), C_it\d+\.Index\(\)\)`))
	})
}

// assertChunkedStream pins the 1a emission contract on one kernel: no raw
// 256 KB/thread decl anywhere, every FRAG decl sized within the priced
// block footprint (≤ MaxFragPerThread bytes), and the fill's window
// coordinates in tensor dim order. It also reports the LoadTileThread
// fill sites with their nest path (the §3.4 actual-side data).
func assertChunkedStream(t *testing.T, kernel, stringCU string, order *regexp.Regexp) {
	t.Helper()
	if strings.Contains(stringCU, "Frag<float, 2u, 256u, 256u>") {
		t.Fatalf("fix7 §3 1a pin 3 (%s): the raw 256 KB/thread Frag decl must be gone from every suite emitter", kernel)
	}
	decls := fragDeclRe.FindAllStringSubmatch(stringCU, -1)
	if len(decls) == 0 {
		t.Fatalf("fix7 §3 1a pin 3 (%s): expected at least one promoted FRAG site (a chunked decl); the emission changed shape?", kernel)
	}
	for _, d := range decls {
		n, m := atoi(t, d[1]), atoi(t, d[2])
		if bytes := n * m * 4; bytes > 256 {
			t.Fatalf("fix7 §3 1a pin 3 (%s): the chunk [%d, %d] = %d B exceeds the priced block footprint (MaxFragPerThread 256 B); the streaming contract is broken", kernel, n, m, bytes)
		}
	}
	if !order.MatchString(stringCU) {
		t.Fatalf("fix7 §3 1a pin 3 (%s): the fill's TensorWindow coordinate list must be in TENSOR DIM ORDER (c8n D-fix6-03 semantics); no match of %s", kernel, order)
	}
	// §3.4 actual-side data: the fill sites and their nest paths.
	sites := fillSiteRe.FindAllStringSubmatch(stringCU, -1)
	t.Logf("§3.4 %s: %d Frag decl(s) %v, %d LoadTileThread fill site(s)", kernel, len(decls), decls, len(sites))
	for _, s := range sites {
		t.Logf("§3.4 %s: fill site buf=%s tensor=%s_v coords=(%s) nest-path=%s",
			kernel, s[1], s[2], s[3], nestPathOf(t, stringCU, s[0]))
	}
}

// nestPathOf returns the chain of loop comments enclosing the first
// occurrence of the given fill statement — the textual nest path the §3.4
// audit counts refills against.
func nestPathOf(t *testing.T, cu, fillLine string) string {
	t.Helper()
	idx := strings.Index(cu, fillLine)
	if idx < 0 {
		t.Fatalf("fill statement not found: %s", fillLine)
	}
	head := cu[:idx]
	var loops []string
	depth := 0
	type open struct {
		comment string
		depth   int
	}
	var stack []open
	lastComment := ""
	for _, line := range strings.Split(head, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "// @") {
			lastComment = trimmed
		}
		if strings.HasPrefix(trimmed, "for (") {
			stack = append(stack, open{lastComment, depth + 1})
		}
		depth += strings.Count(line, "{") - strings.Count(line, "}")
	}
	for _, o := range stack {
		if o.depth <= depth {
			loops = append(loops, o.comment)
		}
	}
	return strings.Join(loops, " -> ")
}

func atoi(t *testing.T, s string) int {
	t.Helper()
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			t.Fatalf("non-digit in %q", s)
		}
		n = n*10 + int(c-'0')
	}
	return n
}

// pricedSitesFor runs the real HIR window + the tiling decide pair over
// the suite source (the exact composition the user path runs, exposed by
// pipeline.HIRPasses) and quotes the priced model via the tiling
// package's own pricing functions (tiling.PricedSites — the §3.4
// priced-side audit hook; the shared siteWindowBytes/siteReuseCount, not
// a re-derivation).
func pricedSitesFor(t *testing.T, fname, src string) []tiling.PricedSite {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, fname)
	if err := os.WriteFile(p, []byte(src), 0o644); err != nil {
		t.Fatalf("audit: write fixture: %v", err)
	}
	mod, err := syntax.ParseModule(lexer.NewFileSet(), p)
	if err != nil {
		t.Fatalf("audit: frontend parse/semacheck failed: %v", err)
	}
	if len(mod.Kernels) != 1 {
		t.Fatalf("audit: want 1 kernel, got %d", len(mod.Kernels))
	}
	g, err := mod.Kernels[0].Build()
	if err != nil {
		t.Fatalf("audit: kernel build failed: %v", err)
	}
	if err := g.Dematerialize(); err != nil {
		t.Fatalf("audit: dematerialize failed: %v", err)
	}
	params := tiling.DefaultParams()
	passes := append(pipeline.HIRPasses(), tiling.New(params), tiling.VerifyPass{Params: params})
	mgr := opt.NewManager(passes)
	if err := mgr.Run(g); err != nil {
		t.Fatalf("audit: HIR+tiling run failed: %v", err)
	}
	rec, ok := mgr.Cache().Get(tiling.RecordName).(*tiling.Record)
	if !ok {
		t.Fatalf("audit: no tiling record in the manager cache after the decide pass")
	}
	sites, err := tiling.PricedSites(g, rec, params)
	if err != nil {
		t.Fatalf("audit: priced sites: %v", err)
	}
	return sites
}

func formatPricedDims(ds []tiling.PricedDim) string {
	var parts []string
	for _, d := range ds {
		parts = append(parts, d.Name+"="+d.SoC.String()+"(T"+itoa(d.Tile)+")")
	}
	return strings.Join(parts, ", ")
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// TestFix7StreamFillAudit is the §3.4 measurement over ALL
// schedule-changed suite emitters (the six const-shaped kernels +
// layernorm/transpose; the matmul family is bit-identical — the Task-3
// sha256 diff records that). It compiles every suite kernel through the
// real facade and logs each LoadTileThread fill site with its nest path,
// then quotes the priced model (j.bytes / j.rawBytes / j.reuse) from the
// tiling package's own pricing functions; reuse-audit.txt in the task dir
// composes the two sides.
func TestFix7StreamFillAudit(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", "..", "testing", "kernels"))
	if err != nil {
		t.Fatalf("resolve testing/kernels: %v", err)
	}
	changed := map[string]bool{
		"dup_expr": true, "dup_const": true, "axpy3": true, "scale_copy": true,
		"row_sum": true, "softmax": true, "layernorm": true, "transpose": true,
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatalf("read testing/kernels: %v", err)
	}
	for _, group := range entries {
		if !group.IsDir() {
			continue
		}
		kernels, err := os.ReadDir(filepath.Join(root, group.Name()))
		if err != nil {
			t.Fatalf("read %s: %v", group.Name(), err)
		}
		for _, k := range kernels {
			name := strings.TrimSuffix(k.Name(), ".tca")
			if !changed[name] {
				continue
			}
			src, err := os.ReadFile(filepath.Join(root, group.Name(), k.Name()))
			if err != nil {
				t.Fatalf("read %s: %v", k.Name(), err)
			}
			cu := compileTcaSource(t, k.Name(), string(src))
			sites := fillSiteRe.FindAllStringSubmatch(cu, -1)
			t.Logf("§3.4 AUDIT %s/%s: fill sites = %d", group.Name(), name, len(sites))
			for _, s := range sites {
				t.Logf("§3.4 AUDIT %s/%s: buf=%s tensor=%s_v coords=(%s) nest-path=%s",
					group.Name(), name, s[1], s[2], s[3], nestPathOf(t, cu, s[0]))
			}
			for _, ps := range pricedSitesFor(t, k.Name(), string(src)) {
				if ps.Class != tiling.ClassFRAG {
					continue // 1a touches FRAG sites only
				}
				t.Logf("§3.4 PRICED %s/%s: tensor=%s class=%s covered=[%s] tie=%s reuse-dims=[%s] raw=%dB window=%dB reuse=%d",
					group.Name(), name, ps.Tensor, ps.Class, formatPricedDims(ps.Covered), ps.Tie,
					formatPricedDims(ps.ReuseDims), ps.RawBytes, ps.WindowBytes, ps.Reuse)
			}
		}
	}
}
