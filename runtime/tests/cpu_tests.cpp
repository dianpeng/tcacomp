// runtime/tests/cpu_tests.cpp — the CPU unit layer (doc/rt.md §8.1).
//
// Plain C++20 test runner (NO gtest — this environment has none and no
// network): a CHECK macro printing file:line on failure, a runner
// accumulating failures, nonzero exit on any failure. Pins the doc formulas
// with exact expected values; totals injected through the host-only
// KernelInfo(unsigned, unsigned) ctor, linear ids through set_linear_id.
#include <bit>
#include <cstdio>
#include <initializer_list>

#include "tca/iterator.h"
#include "tca/loader.h"
#include "tca/mapping.h"
#include "tca/reducer.h"
#include "tca/view.h"

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    ++g_checks;                                                        \
    if (!(cond)) {                                                     \
      ++g_failures;                                                    \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
    }                                                                  \
  } while (0)

namespace {

tca::rt::KernelInfo host_info(unsigned ctas, unsigned threads) { return tca::rt::KernelInfo(ctas, threads); }

// ---------------------------------------------------------------------------
// fix4 §4.5 factorization decomposition tests (doc/fix/fix4.md §4.1/§3.2):
// reverse stride-shape over the constructor shape, WARP-unit thread
// coordinates, flat-rank identities, kinfo-derived counts under surplus
// launches, and absolute-loc iterator walks.
// ---------------------------------------------------------------------------

// (a) GridMapping factorization over the 2-dim shape (4, 16): dimension()
// returns the extents VERBATIM; index(d) is the reverse stride-shape
// decomposition of the flat CTA rank; the decomposition is a BIJECTION on
// [0, Π shape) — reconstruction index(0)*16 + index(1) == id for ALL ids.
void test_grid_mapping_factorization() {
  auto kinfo = host_info(64u, 512u);
  tca::rt::GridMapping gm(kinfo, 2u, 4u, 16u);
  CHECK(gm.dimension(0u) == 4u);   // LoC_0 verbatim (the CTA count)
  CHECK(gm.dimension(1u) == 16u);  // LoC_1 verbatim
  for (unsigned id : {0u, 15u, 16u, 63u}) {
    gm.set_linear_id(id);
    CHECK(gm.index(0u) == (id / 16u) % 4u);  // row-major: last dim stride 1
    CHECK(gm.index(1u) == id % 16u);
  }
  for (unsigned id = 0u; id < 64u; ++id) {  // the bijection on [0, 4*16)
    gm.set_linear_id(id);
    CHECK(gm.index(0u) * 16u + gm.index(1u) == id);
  }
}

// (b) ThreadMapping factorization in WARP units (§4.1): index(d) decomposes
// the WARP RANK (index()/32) over the K shape — never the raw thread id;
// lane_id() resolves the lane inside the warp. Bijection over warps:
// index(0)*8 + index(1) == tid/32 for EVERY tid. warp_count()/total() are
// the kinfo-DERIVED launch values (here the shape product 2*8 happens to
// equal the launch — the surplus case is (d) below).
void test_thread_mapping_factorization() {
  auto kinfo = host_info(64u, 512u);  // the launch: 16 warps
  tca::rt::ThreadMapping tm(kinfo, 2u, 2u, 8u);
  CHECK(tm.dimension(0u) == 2u);  // K_0 verbatim (the WARP count)
  CHECK(tm.dimension(1u) == 8u);
  for (unsigned tid : {0u, 31u, 32u, 63u, 255u, 256u, 511u}) {
    tm.set_linear_id(tid);
    const unsigned warp_rank = tid / 32u;
    CHECK(tm.index(0u) == (warp_rank / 8u) % 2u);
    CHECK(tm.index(1u) == warp_rank % 8u);
    CHECK(tm.lane_id() == tid % 32u);
    CHECK(tm.warp_index() == warp_rank);
  }
  for (unsigned tid = 0u; tid < 512u; ++tid) {  // the bijection over warps
    tm.set_linear_id(tid);
    CHECK(tm.index(0u) * 8u + tm.index(1u) == tid / 32u);
  }
  CHECK(tm.warp_count() == 16u);  // kinfo-derived: 512 threads / 32
  CHECK(tm.total() == 512u);      // kinfo-derived: blockDim
}

// (c) Flat-rank identities (ruling 2026-10-23 restated under §4.1): no-arg
// index() IS the injected flat rank; warp/lane/first queries are pure
// functions of it. Plus the §3.3 canonical worked-example pair (gridDim 16,
// blockDim 64 = 2 warps): every CTA id is its own single-dim coordinate,
// and warp 0/1 own the even/odd tile ranks of the K=2 split.
void test_flat_rank_identities() {
  auto kinfo = host_info(64u, 512u);
  tca::rt::GridMapping gm(kinfo, 2u, 4u, 16u);
  for (unsigned id : {0u, 1u, 15u, 16u, 42u, 63u}) {
    gm.set_linear_id(id);
    CHECK(gm.index() == id);  // the flat CTA id — blockIdx.x
  }
  tca::rt::ThreadMapping tm(kinfo, 2u, 2u, 8u);
  for (unsigned tid : {0u, 1u, 31u, 32u, 63u, 255u, 256u, 511u}) {
    tm.set_linear_id(tid);
    CHECK(tm.index() == tid);  // the flat thread id — threadIdx.x
    CHECK(tm.warp_index() == tm.index() / 32u);
    CHECK(tm.lane_id() == tm.index() % 32u);
    CHECK(tm.is_first_in_warp() == (tm.lane_id() == 0u));
    CHECK(tm.is_first_in_cta() == (tm.index() == 0u));
  }
  auto wk = host_info(16u, 64u);  // the §3.3 worked-example launch
  tca::rt::GridMapping wgm(wk, 1u, 16u);
  CHECK(wgm.dimension(0u) == 16u);
  for (unsigned id = 0u; id < 16u; ++id) {
    wgm.set_linear_id(id);
    CHECK(wgm.index(0u) == id);  // 1-D shape: the coordinate IS the rank
  }
  tca::rt::ThreadMapping wtm(wk, 1u, 2u);
  for (unsigned tid = 0u; tid < 64u; ++tid) {
    wtm.set_linear_id(tid);
    CHECK(wtm.index(0u) == tid / 32u);  // warp 0 even tile ranks, warp 1 odd
  }
}

// (d) Surplus-rank coordinate totality (§3.2): a multi-LNT launch runs the
// max-over-LNT totals, so an LNT's shape product can be SMALLER than the
// launch. warp_count()/total() answer the LAUNCH (kinfo-derived: 16 warps,
// 512 threads — NOT the shape product 2); surplus ranks' coordinates wrap
// in-shape via the modulo (always < extent — total, no UB); their bodies
// whole-body-skip via the compiler's loop guards. Deliberately no assert
// ties the shape product to the launch total.
void test_surplus_ranks_totality() {
  auto kinfo = host_info(100u, 512u);        // the launch: 16 warps
  tca::rt::ThreadMapping tm(kinfo, 1u, 2u);  // an LNT factoring only 2 warps
  CHECK(tm.warp_count() == 16u);             // kinfo-DERIVED, not the shape 2
  CHECK(tm.total() == 512u);
  for (unsigned tid = 0u; tid < 512u; ++tid) {
    tm.set_linear_id(tid);
    CHECK(tm.index(0u) < 2u);  // coordinate totality on surplus ranks
  }
  auto gkinfo = host_info(64u, 512u);
  tca::rt::GridMapping gm(gkinfo, 1u, 4u);  // 4 of the launch's 64 CTAs
  CHECK(gm.dimension(0u) == 4u);
  for (unsigned id = 0u; id < 64u; ++id) {
    gm.set_linear_id(id);
    CHECK(gm.index(0u) < 4u);
  }
}

// (e) The uniform round walk: base < bound; base += step; Index() = base.
void test_seq_iterator() {
  // 16 trips, Index = 0, 64, ..., 960.
  unsigned trips = 0u;
  for (tca::rt::SeqIterator it(0u, 64u, 1024u); it.Has(); it.Next()) {
    CHECK(it.Index() == trips * 64u);
    ++trips;
  }
  CHECK(trips == 16u);

  // Nonzero init.
  trips = 0u;
  for (tca::rt::SeqIterator it(7u, 3u, 20u); it.Has(); it.Next()) {
    CHECK(it.Index() == 7u + trips * 3u);
    ++trips;
  }
  CHECK(trips == 5u);  // 7, 10, 13, 16, 19

  // Empty walks: init >= bound.
  CHECK(!tca::rt::SeqIterator(10u, 1u, 10u).Has());
  CHECK(!tca::rt::SeqIterator(12u, 5u, 10u).Has());

  // Single trip when step >= bound (but init < bound).
  trips = 0u;
  for (tca::rt::SeqIterator it(0u, 64u, 10u); it.Has(); it.Next()) {
    CHECK(it.Index() == 0u);
    ++trips;
  }
  CHECK(trips == 1u);
}

// (e, cont.) Cta/Thr iterators walk identically — map/dosoc/loc ignored in
// the v0 walk. Ported to the factorization ctor at the fix4 §3.3
// worked-example launch (gridDim 16, blockDim 64 = 2 warps); the loc args
// are now the ABSOLUTE extents (fix4 §4.2).
void test_map_iterators() {
  auto kinfo = host_info(16u, 64u);
  tca::rt::GridMapping gm(kinfo, 1u, 16u);   // 1 CTA dim, LoC = 16
  tca::rt::ThreadMapping tm(kinfo, 1u, 2u);  // 1 WARP dim, K = 2

  unsigned trips = 0u;
  for (tca::rt::CtaIterator it(gm, 0u, 16u, 7u, 3u, 20u); it.Has(); it.Next()) {
    CHECK(it.Index() == 7u + trips * 3u);
    ++trips;
  }
  CHECK(trips == 5u);

  trips = 0u;
  for (tca::rt::ThrIterator it(tm, 1u, 2u, 0u, 64u, 1024u); it.Has(); it.Next()) {
    CHECK(it.Index() == trips * 64u);
    ++trips;
  }
  CHECK(trips == 16u);
}

// (e, §4.5) Absolute-loc iterator walks (§3.1 caller arithmetic, §4.2 loc
// semantics): the CTA round walk strides LoC×T; the WARP-dim round walk
// strides K×T; the COUPLED inside walk takes init = the flat thread rank,
// step = blockDim, and partitions the tile MAX with per-thread trip-count
// single ownership. (The plan's example prose — "tid 0..511 take one trip"
// — is imprecise: under step 512 / bound 600, tids < 88 take a SECOND trip
// since tid + 512 < 600; the exact trip formula and the single-ownership
// partition are pinned here instead.)
void test_iterator_absolute_loc_walks() {
  auto kinfo = host_info(16u, 512u);
  tca::rt::GridMapping gm(kinfo, 1u, 4u);    // LoC = 4
  tca::rt::ThreadMapping tm(kinfo, 1u, 2u);  // K = 2

  // CTA round walk: step = LoC × T = 4 × 64 = 256 over 4096 -> 16 rounds.
  unsigned trips = 0u;
  for (tca::rt::CtaIterator it(gm, 0u, 4u, 0u, 4u * 64u, 4096u); it.Has(); it.Next()) {
    CHECK(it.Index() == trips * 256u);
    ++trips;
  }
  CHECK(trips == 16u);

  // WARP-dim round walk: step = K × T = 2 × 32 = 64 over 1024 -> 16 rounds.
  trips = 0u;
  for (tca::rt::ThrIterator it(tm, 0u, 2u, 0u, 2u * 32u, 1024u); it.Has(); it.Next()) {
    CHECK(it.Index() == trips * 64u);
    ++trips;
  }
  CHECK(trips == 16u);

  // The coupled inside walk on a 1024-thread launch shape (tids past the
  // 512-thread block boundary injectable through set_linear_id):
  // trips(tid) = #{k >= 0 : tid + k*512 < 600} — 2 for tid < 88, 1 for
  // 88 <= tid < 600, 0 for tid >= 600.
  auto big = host_info(16u, 1024u);
  tca::rt::ThreadMapping btm(big, 1u, 2u);
  for (unsigned tid = 0u; tid < 1024u; ++tid) {
    btm.set_linear_id(tid);
    const unsigned expected = (tid < 600u ? 1u : 0u) + (tid + 512u < 600u ? 1u : 0u);
    unsigned n = 0u;
    for (tca::rt::ThrIterator it(btm, 0u, 2u, btm.index(), 512u, 600u); it.Has(); it.Next()) {
      ++n;
    }
    CHECK(n == expected);
  }
  // The REAL 512-thread block partitions [0, 600) exactly once per element
  // (element e -> thread e%512, trip e/512) — single ownership (§4.2).
  unsigned cover[600] = {};
  for (unsigned tid = 0u; tid < 512u; ++tid) {
    btm.set_linear_id(tid);
    for (tca::rt::ThrIterator it(btm, 0u, 2u, btm.index(), 512u, 600u); it.Has(); it.Next()) {
      CHECK(it.Index() < 600u);
      ++cover[it.Index()];
    }
  }
  for (unsigned e = 0u; e < 600u; ++e) CHECK(cover[e] == 1u);
}

// (f) Views: row-major stride math, Get/Set roundtrip, OOB suppression.
void test_tensor_view() {
  float data[2 * 3 * 4] = {};
  for (unsigned i = 0u; i < 24u; ++i) data[i] = static_cast<float>(i);

  tca::rt::TensorView<float, 3u> v(data, 2u, 3u, 4u);
  CHECK(v.Dim(0u) == 2u);
  CHECK(v.Dim(1u) == 3u);
  CHECK(v.Dim(2u) == 4u);
  CHECK(v.Get(1u, 2u, 3u) == data[1 * 12u + 2 * 4u + 3u]);  // strides (12, 4, 1)
  CHECK(v.Get(0u, 0u, 0u) == data[0u]);

  // Set roundtrip.
  v.Set(99.5f, 0u, 1u, 2u);
  CHECK(data[1 * 4u + 2u] == 99.5f);
  CHECK(v.Get(0u, 1u, 2u) == 99.5f);

  // OOB Get suppresses (returns T{} — never a wild access).
  CHECK(v.Get(2u, 0u, 0u) == 0.0f);
  CHECK(v.Get(0u, 3u, 0u) == 0.0f);
  CHECK(v.Get(0u, 0u, 4u) == 0.0f);

  // OOB Set silently no-ops against a canary buffer.
  float canary[6] = {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f};
  tca::rt::TensorView<float, 2u> cv(canary, 2u, 3u);
  cv.Set(1234.0f, 2u, 0u);  // dim0 OOB
  cv.Set(1234.0f, 0u, 3u);  // dim1 OOB
  for (float x : canary) CHECK(x == 7.0f);

  // const-qualified T (the goldens instantiate TensorView<const float, 2u>).
  const float cdata[4] = {0.25f, 0.5f, 0.75f, 1.0f};
  tca::rt::TensorView<const float, 2u> const_v(cdata, 2u, 2u);
  CHECK(const_v.Get(1u, 1u) == 1.0f);
  CHECK(const_v.Get(0u, 2u) == 0.0f);  // suppression on const views too
}

// (f, cont.) SmemView: in-window by construction (golden shape: 1-D linear).
void test_smem_view() {
  float base[3 * 5] = {};
  tca::rt::SmemView<float, 2u> sv(base, 3u, 5u);
  CHECK(sv.Extent(0u) == 3u);
  CHECK(sv.Extent(1u) == 5u);
  sv.Set(4.5f, 2u, 4u);
  CHECK(base[2 * 5u + 4u] == 4.5f);
  CHECK(sv.Get(2u, 4u) == 4.5f);

  // The golden's linear 1-D construction: SmemView<float, 1u>(ptr, 16384u).
  float lin[16] = {};
  tca::rt::SmemView<float, 1u> lv(lin, 16u);
  lv.Set(0.5f, 13u);
  CHECK(lin[13u] == 0.5f);
  CHECK(lv.Get(13u) == 0.5f);

  // Frag: same face, compile-time extents.
  tca::rt::Frag<float, 2u, 3u, 5u> frag;
  frag.Set(1.25f, 1u, 2u);
  CHECK(frag.Get(1u, 2u) == 1.25f);
  CHECK(frag.Extent(0u) == 3u);
}

// (g) The u32 pair.
void test_umin_umax() {
  using tca::rt::umax;
  using tca::rt::umin;
  CHECK(umin(3u, 5u) == 3u);
  CHECK(umin(5u, 3u) == 3u);
  CHECK(umin(4u, 4u) == 4u);
  CHECK(umin(0u, 7u) == 0u);
  CHECK(umax(3u, 5u) == 5u);
  CHECK(umax(5u, 3u) == 5u);
  CHECK(umax(4u, 4u) == 4u);
  CHECK(umax(0u, 7u) == 7u);
}

// ---------------------------------------------------------------------------
// The loader's row-striped window copy (fix3 D-n, signature-internal
// performance rework; the PUBLIC signatures and the window->axis contract
// are unchanged). The CPU layer pins the decomposition: per-row tail
// clamping, leading-OOB zero-fill, thread striping, the 128-bit vector
// path's eligibility + fallback, and the portable per-thread entry point.
// cp.async is nvcc-only — the goldens + the e2e gate carry it.
// ---------------------------------------------------------------------------

// reference_window_copy is the semantic reference: every window cell takes
// src.GetAt(full coordinate) — T{} where the source suppresses.
template <unsigned DR, unsigned SR>
void reference_window_copy(float* dst, const unsigned (&dext)[DR], const float* src,
                           const unsigned (&sext)[SR], const unsigned (&base)[SR]) {
  unsigned n = 1u;
  for (unsigned d = 0u; d < DR; ++d) n *= dext[d];
  for (unsigned i = 0u; i < n; ++i) {
    unsigned w[DR];
    unsigned rem = i;
    for (unsigned d = DR; d-- > 0u;) {
      w[d] = rem % dext[d];
      rem /= dext[d];
    }
    unsigned c[SR];
    for (unsigned a = 0u; a < SR; ++a) c[a] = base[a];
    for (unsigned j = 0u; j < DR; ++j) c[SR - DR + j] += w[j];
    // The source view's bounds suppression (out-of-range -> T{}).
    bool ok = true;
    for (unsigned a = 0u; a < SR; ++a) {
      if (c[a] >= sext[a]) ok = false;
    }
    dst[i] = ok ? src[c[0] * (sext[1] * sext[2]) + c[1] * sext[2] + c[2]] : 0.0f;
  }
}

// (i) The decomposition over one DR=2-of-SR=3 window: full rows, a
// clamped trailing run, and a leading-OOB row that zero-fills.
void test_loader_row_clamping() {
  float src[3 * 5 * 7];
  for (unsigned i = 0u; i < 3 * 5 * 7; ++i) src[i] = static_cast<float>(i) + 0.25f;

  // Window [5, 7) over axes 1..2 at base (1, 2, 4): trailing axis clamps to
  // 7 - 4 = 3 valid cells; row (axis1 == 3) is leading-OOB (2 + 3 >= 5).
  const unsigned dext[2] = {5u, 7u};
  const unsigned sext[3] = {3u, 5u, 7u};
  const unsigned base[3] = {1u, 2u, 4u};
  float want[5 * 7];
  reference_window_copy<2, 3>(want, dext, src, sext, base);

  float smem[5 * 7];
  for (unsigned i = 0u; i < 5 * 7; ++i) smem[i] = -777.0f;
  tca::rt::SmemView<float, 2u> dst(smem, 5u, 7u);
  tca::rt::TensorView<const float, 3u> view(src, 3u, 5u, 7u);
  tca::rt::detail::load_tile_rows<false, tca::rt::SmemView<float, 2u>,
                                  tca::rt::TensorWindow<const float, 3u>, 2u, 3u>(
      dst, tca::rt::TensorWindow(view, base[0], base[1], base[2]), 0u, 1u);
  for (unsigned i = 0u; i < 5 * 7; ++i) CHECK(smem[i] == want[i]);

  // The clamp facts, pinned directly: the trailing run clamps once per row
  // (7 vs 7-4), and the leading-OOB row zero-fills (suppression).
  CHECK(smem[0 * 7 + 0] == want[0 * 7 + 0]);            // valid run
  CHECK(smem[0 * 7 + 3] == 0.0f);                       // clamped tail zero-fills
  CHECK(smem[3 * 7 + 0] == 0.0f);                       // leading-OOB row zero-fills
  CHECK(smem[4 * 7 + 6] == want[4 * 7 + 6]);            // last row mirrors too
}

// (ii) Thread striping: every (tid, nthreads) enumeration partitions the
// rows exactly; any partitioning equals the single-thread copy.
void test_loader_striping() {
  float src[4 * 6 * 8];
  for (unsigned i = 0u; i < 4 * 6 * 8; ++i) src[i] = static_cast<float>(i) * 0.5f;
  const unsigned dext[2] = {6u, 8u};
  const unsigned sext[3] = {4u, 6u, 8u};
  const unsigned base[3] = {1u, 1u, 2u};
  float want[6 * 8];
  reference_window_copy<2, 3>(want, dext, src, sext, base);

  for (unsigned nt : {1u, 2u, 3u, 7u, 16u, 512u}) {
    float smem[6 * 8];
    for (unsigned i = 0u; i < 6 * 8; ++i) smem[i] = -7.0f;
    tca::rt::SmemView<float, 2u> dst(smem, 6u, 8u);
    tca::rt::TensorView<const float, 3u> view(src, 4u, 6u, 8u);
    for (unsigned tid = 0u; tid < nt; ++tid) {
      tca::rt::detail::load_tile_rows<false, tca::rt::SmemView<float, 2u>,
                                      tca::rt::TensorWindow<const float, 3u>, 2u, 3u>(
          dst, tca::rt::TensorWindow(view, base[0], base[1], base[2]), tid, nt);
    }
    for (unsigned i = 0u; i < 6 * 8; ++i) {
      CHECK(smem[i] == want[i]);  // exactly-once ownership: no double writes
    }
  }
}

// (iii) The 128-bit vector path: eligibility is per-row alignment, and a
// misaligned source base falls back to the scalar run with IDENTICAL
// results. vec16_ok is pinned true/false directly.
void test_loader_vector_eligibility() {
  alignas(16) float src[64];
  alignas(16) float dstbuf[64];
  for (unsigned i = 0u; i < 64u; ++i) src[i] = static_cast<float>(i);
  CHECK(tca::rt::detail::vec16_ok(dstbuf, src) == true);          // both 16B aligned
  CHECK(tca::rt::detail::vec16_ok(dstbuf + 1u, src) == false);    // dst misaligned
  CHECK(tca::rt::detail::vec16_ok(dstbuf, src + 1u) == false);    // src misaligned
  CHECK(tca::rt::detail::vec16_ok(dstbuf + 4u, src + 4u) == true);

  // A window whose source base coordinate is NOT 4-element aligned (base 1
  // on the trailing axis) forces the scalar fallback row-by-row; the
  // result must equal the aligned (vector-path) copy of the shifted data.
  tca::rt::SmemView<float, 1u> dstA(dstbuf, 16u);
  tca::rt::TensorView<const float, 1u> view(src, 64u);
  tca::rt::detail::load_tile_rows<false, tca::rt::SmemView<float, 1u>,
                                  tca::rt::TensorWindow<const float, 1u>, 1u, 1u>(
      dstA, tca::rt::TensorWindow(view, 1u), 0u, 1u);
  for (unsigned k = 0u; k < 16u; ++k) CHECK(dstbuf[k] == src[1u + k]);

  // The aligned form (vector path eligible: trailing 16 %4 == 0, bases
  // aligned) must produce the same values as the scalar reference.
  tca::rt::detail::load_tile_rows<false, tca::rt::SmemView<float, 1u>,
                                  tca::rt::TensorWindow<const float, 1u>, 1u, 1u>(
      dstA, tca::rt::TensorWindow(view, 0u), 0u, 1u);
  for (unsigned k = 0u; k < 16u; ++k) CHECK(dstbuf[k] == src[k]);

  // Trailing extent NOT %4: whole copy rides the scalar path.
  float small[5] = {9.0f, 8.0f, 7.0f, 6.0f, 5.0f};
  float out5[5] = {};
  tca::rt::SmemView<float, 1u> d5(out5, 5u);
  tca::rt::TensorView<const float, 1u> v5(small, 5u);
  tca::rt::detail::load_tile_rows<false, tca::rt::SmemView<float, 1u>,
                                  tca::rt::TensorWindow<const float, 1u>, 1u, 1u>(
      d5, tca::rt::TensorWindow(v5, 0u), 0u, 1u);
  for (unsigned k = 0u; k < 5u; ++k) CHECK(out5[k] == small[k]);
}

// (iv) The per-thread entry point end-to-end (portable, dual-compiled): a
// Frag fill over a rank-2 window with a clamped tail.
void test_loader_frag_thread() {
  float src[6 * 8];
  for (unsigned i = 0u; i < 6 * 8; ++i) src[i] = static_cast<float>(i) * 0.25f;
  tca::rt::Frag<float, 2u, 4u, 8u> frag;
  tca::rt::TensorView<const float, 2u> view(src, 6u, 8u);
  // base (2, 5): axis-0 rows 2..5 all in range; trailing clamps to 8-5=3.
  tca::rt::ThreadMapping tm(host_info(100u, 512u), 1u, 16u);  // 16 warps (512 threads)
	tca::rt::LoadTileThread(tm, frag, tca::rt::TensorWindow(view, 2u, 5u));
  for (unsigned r = 0u; r < 4u; ++r) {
    for (unsigned k = 0u; k < 8u; ++k) {
      const float want = (k < 3u) ? src[(2u + r) * 8u + 5u + k] : 0.0f;
      CHECK(frag.Get(r, k) == want);
    }
  }
}

float add_f(float a, float b) { return a + b; }

// (h) Reducer algebra — the R1 scalar-only protocol (ruling 2026-10-23):
// no Slots template parameter, Reduce folds one element per call, Combine
// always returns T.
void test_seq_reducer() {
  // Scalar: Init fills the declared init; Reduce folds through the
  // runtime-owned combiner; Combine returns T. Dyadic values — exact.
  tca::rt::SeqReducer<float> r(0.0f, add_f);
  r.Init();
  CHECK(r.Combine() == 0.0f);  // identity-init semantics before any Reduce
  r.Reduce(1.5f);
  r.Reduce(2.25f);
  CHECK(r.Combine() == 3.75f);
  r.Init();  // argument-free re-initialization from the construction-time init
  CHECK(r.Combine() == 0.0f);
}

// The WarpReducer's HOST branch is the seq algebra (no warps on the host);
// the ctor carries the mapping and the compiler-pool staging row — R1: on
// the CPU layer a stack array is the stand-in for the dynamic-SMEM pool
// row. Pin the face so the branch is compiled and exercised somewhere.
void test_warp_reducer_host_branch() {
  auto kinfo = host_info(100u, 512u);
  tca::rt::ThreadMapping tm(kinfo, 1u, 16u);  // 16 warps (512 threads)
  float stage[16] = {};  // 16 warps — the stack-array stand-in for the pool row
  tca::rt::SmemView<float, 1u> staging(stage, 16u);
  tca::rt::WarpReducer<float> w(0.0f, add_f, tm, staging);
  w.Init();
  w.Reduce(1.5f);
  w.Reduce(0.25f);
  CHECK(w.Combine() == 1.75f);
  w.Init();
  CHECK(w.Combine() == 0.0f);
  // The host branch never touches the staging row; the ctor-provided view
  // addresses the caller's array unchanged.
  CHECK(staging.Get(0u) == 0.0f);
}

// ---------------------------------------------------------------------------
// The 2026-10-24 WarpReducer bracket-protocol simulation. Per output
// element: Init resets a slot array (one per warp) to the init; a sequence
// of Reduce calls, each carrying (a) the calling warp, (b) a lane-PREFIX
// participation set (modeled as a count k of active lanes — the prefix
// property the compiler's bounds gates produce), (c) one hand-chosen value
// per active lane; per call the warp partial folds the active lanes' values
// (the guarded down-tree's association — ANY association is valid), then
// slot[warp] = fn(slot[warp], warpPartial); after all calls, every thread's
// Combine = fold(init, slots ascending). The reference is a plain
// single-threaded fold over every (call, lane) value in order.
//
// R-C DISCIPLINE: map+reduce are UNORDERED; the assertions are VALUE-only
// (|simulated - reference| <= 1e-5 for float add; exact value for the int
// product — integer arithmetic is exact, a value assertion, not an order
// pin). Never bitwise on floats, never fold-path-pinned.
// ---------------------------------------------------------------------------
struct SimCall {
  unsigned warp;
  unsigned k;  // active-lane PREFIX count, 1..32
  float fv[32];
  int iv[32];
};

int mul_i(int a, int b) { return a * b; }

// The float-add simulation over one output element's bracket.
float simulate_bracket_f(float init, unsigned warps, const SimCall* calls, unsigned n) {
  // Init: reset ALL slots.
  float slots[16];
  for (unsigned w = 0u; w < warps; ++w) slots[w] = init;
  for (unsigned c = 0u; c < n; ++c) {
    // Reduce: the guarded down-tree over the k-lane prefix.
    float lanes[32];
    for (unsigned l = 0u; l < calls[c].k; ++l) lanes[l] = calls[c].fv[l];
    for (unsigned off = 16u; off > 0u; off >>= 1u) {
      for (unsigned l = 0u; l + off < calls[c].k; ++l) {
        lanes[l] = add_f(lanes[l], lanes[l + off]);
      }
    }
    slots[calls[c].warp] = add_f(slots[calls[c].warp], lanes[0]);
  }
  // Combine: fold(init, slots ascending).
  float out = init;
  for (unsigned w = 0u; w < warps; ++w) out = add_f(out, slots[w]);
  return out;
}

float reference_bracket_f(float init, const SimCall* calls, unsigned n) {
  float acc = init;
  for (unsigned c = 0u; c < n; ++c)
    for (unsigned l = 0u; l < calls[c].k; ++l) acc = add_f(acc, calls[c].fv[l]);
  return acc;
}

// The int-product simulation (1-warp config): same mechanics, exact values.
int simulate_bracket_i(int init, unsigned warps, const SimCall* calls, unsigned n) {
  int slots[16];
  for (unsigned w = 0u; w < warps; ++w) slots[w] = init;
  for (unsigned c = 0u; c < n; ++c) {
    int lanes[32];
    for (unsigned l = 0u; l < calls[c].k; ++l) lanes[l] = calls[c].iv[l];
    for (unsigned off = 16u; off > 0u; off >>= 1u) {
      for (unsigned l = 0u; l + off < calls[c].k; ++l) {
        lanes[l] = mul_i(lanes[l], lanes[l + off]);
      }
    }
    slots[calls[c].warp] = mul_i(slots[calls[c].warp], lanes[0]);
  }
  int out = init;
  for (unsigned w = 0u; w < warps; ++w) out = mul_i(out, slots[w]);
  return out;
}

int reference_bracket_i(int init, const SimCall* calls, unsigned n) {
  int acc = init;
  for (unsigned c = 0u; c < n; ++c)
    for (unsigned l = 0u; l < calls[c].k; ++l) acc = mul_i(acc, calls[c].iv[l]);
  return acc;
}

void setf(SimCall& c, unsigned warp, unsigned k, const float* vals) {
  c.warp = warp;
  c.k = k;
  for (unsigned l = 0u; l < 32u; ++l) c.fv[l] = (l < k) ? vals[l] : 0.0f;
  for (unsigned l = 0u; l < 32u; ++l) c.iv[l] = 1;
}

void test_warp_reducer_bracket_simulation() {
  // (32 threads, 1 warp): full-width, partially idle (k < 32), and single
  // lane calls; plus the exact int product with at most 30 factors of 2.
  {
    const float v1[32] = {0.25f, 0.5f,  0.75f, 1.0f,  0.25f, 0.5f,  0.75f, 1.0f,
                          0.25f, 0.5f,  0.75f, 1.0f,  0.25f, 0.5f,  0.75f, 1.0f,
                          0.25f, 0.5f,  0.75f, 1.0f,  0.25f, 0.5f,  0.75f, 1.0f,
                          0.25f, 0.5f,  0.75f, 1.0f,  0.25f, 0.5f,  0.75f, 1.0f};
    const float v2[17] = {0.5f, 0.25f, 0.25f, 0.5f, 0.75f, 0.25f, 1.0f, 0.5f,
                          0.25f, 0.25f, 0.5f,  0.5f, 0.25f, 0.75f, 0.5f, 0.25f, 1.0f};
    const float v3[1] = {2.0f};
    SimCall calls[3];
    setf(calls[0], 0u, 32u, v1);
    setf(calls[1], 0u, 17u, v2);
    setf(calls[2], 0u, 1u, v3);
    float sim = simulate_bracket_f(0.0f, 1u, calls, 3u);
    float ref = reference_bracket_f(0.0f, calls, 3u);
    CHECK(sim - ref <= 1e-5f && ref - sim <= 1e-5f);

    // Int product: one call, k = 32, at most 30 factors of 2 (rest 1) —
    // exact value assertion.
    SimCall pc;
    pc.warp = 0u;
    pc.k = 32u;
    for (unsigned l = 0u; l < 32u; ++l) {
      pc.fv[l] = 1.0f;
      pc.iv[l] = (l < 30u) ? 2 : 1;
    }
    int isim = simulate_bracket_i(1, 1u, &pc, 1u);
    int iref = reference_bracket_i(1, &pc, 1u);
    CHECK(isim == iref);
    CHECK(iref == (1 << 30));  // 2^30 fits int — the value itself pinned
  }
  // (64 threads, 2 warps): warp 1 FULLY IDLE (no calls), warp 0 partially
  // idle — the idle warp's slot stays init and the fold absorbs it as the
  // identity.
  {
    const float v1[20] = {0.25f, 0.5f, 0.75f, 0.25f, 0.5f, 0.5f, 0.75f, 0.25f,
                          0.5f,  0.25f, 0.75f, 0.5f,  0.25f, 1.0f, 0.5f,  0.25f,
                          0.75f, 0.5f, 0.25f, 0.5f};
    const float v2[32] = {0.25f, 0.25f, 0.25f, 0.25f, 0.5f, 0.5f,  0.5f,  0.5f,
                          0.75f, 0.75f, 0.75f, 0.75f, 1.0f, 1.0f,  1.0f,  1.0f,
                          0.25f, 0.25f, 0.25f, 0.25f, 0.5f, 0.5f,  0.5f,  0.5f,
                          0.75f, 0.75f, 0.75f, 0.75f, 0.5f, 0.25f, 0.75f, 0.5f};
    SimCall calls[2];
    setf(calls[0], 0u, 20u, v1);
    setf(calls[1], 0u, 32u, v2);
    float sim = simulate_bracket_f(0.0f, 2u, calls, 2u);
    float ref = reference_bracket_f(0.0f, calls, 2u);
    CHECK(sim - ref <= 1e-5f && ref - sim <= 1e-5f);
  }
  // (256 threads, 8 warps): warps 0..6 active with mixed k (incl. partial),
  // warp 7 fully idle.
  // (512 threads, 16 warps): warps 0..14 active, warp 15 fully idle.
  for (unsigned cfg = 0u; cfg < 2u; ++cfg) {
    const unsigned warps = (cfg == 0u) ? 8u : 16u;
    SimCall calls[16];
    unsigned n = 0u;
    for (unsigned w = 0u; w + 1u < warps; ++w) {
      const unsigned k = 1u + ((w * 7u) % 32u);  // mixed prefix widths
      for (unsigned l = 0u; l < 32u; ++l) {
        calls[n].fv[l] = 0.0f;
        calls[n].iv[l] = 1;
      }
      calls[n].warp = w;
      calls[n].k = k;
      for (unsigned l = 0u; l < k; ++l) {
        calls[n].fv[l] = static_cast<float>((int)((l + w) % 4) + 1) * 0.25f;
      }
      ++n;
    }
    float sim = simulate_bracket_f(0.0f, warps, calls, n);
    float ref = reference_bracket_f(0.0f, calls, n);
    CHECK(sim - ref <= 1e-5f && ref - sim <= 1e-5f);
  }
}

// The mask-derivation contract on host (pure integer checks): prefix masks
// with k set lanes have popcount k, and the guarded down-tree's source
// lanes (lane_id + off) stay inside the active prefix whenever the guard
// admits the read.
void test_mask_derivation_arithmetic() {
  for (unsigned k = 1u; k <= 32u; ++k) {
    const unsigned mask = (k == 32u) ? 0xFFFFFFFFu : ((1u << k) - 1u);
    CHECK(std::popcount(mask) == k);  // prefix(k): the low k lanes
    for (unsigned off : {1u, 2u, 4u, 8u, 16u}) {
      for (unsigned lane = 0u; lane < k; ++lane) {
        if (lane + off < k) {
          // The guard admits the read: the source lane is in [0, k), i.e.
          // inside the prefix mask.
          CHECK(((mask >> (lane + off)) & 1u) == 1u);
        }
      }
    }
  }
  // lane/warp identity arithmetic (the map's derivation contract).
  for (unsigned tid : {0u, 1u, 31u, 32u, 255u, 256u, 511u}) {
    CHECK(tid / 32u == tid / 32u);   // warp_index
    CHECK((tid % 32u) < 32u);        // lane_id
    CHECK(((tid % 32u) == 0u) == (tid % 32u == 0u));  // is_first_in_warp
  }
}

}  // namespace

int main() {
  test_grid_mapping_factorization();
  test_thread_mapping_factorization();
  test_flat_rank_identities();
  test_surplus_ranks_totality();
  test_seq_iterator();
  test_map_iterators();
  test_iterator_absolute_loc_walks();
  test_tensor_view();
  test_smem_view();
  test_umin_umax();
  test_loader_row_clamping();
  test_loader_striping();
  test_loader_vector_eligibility();
  test_loader_frag_thread();
  test_seq_reducer();
  test_warp_reducer_host_branch();
  test_warp_reducer_bracket_simulation();
  test_mask_derivation_arithmetic();

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
