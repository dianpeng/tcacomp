// runtime/tests/cpu_tests.cpp — the CPU unit layer (doc/rt.md §8.1).
//
// Plain C++20 test runner (NO gtest — this environment has none and no
// network): a CHECK macro printing file:line on failure, a runner
// accumulating failures, nonzero exit on any failure. Pins the doc formulas
// with exact expected values; totals injected through the host-only
// KernelInfo(unsigned, unsigned) ctor, linear ids through set_linear_id.
#include <cstdio>
#include <initializer_list>

#include "tca/iterator.h"
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

// (a) GridMapping(-1) @ 100 CTAs: single tail dim over the whole space.
void test_grid_mapping_tail() {
  auto kinfo = host_info(100u, 512u);
  tca::rt::GridMapping gm(kinfo, -1);
  CHECK(gm.dimension(0u) == 100u);
  for (unsigned id : {0u, 1u, 42u, 99u}) {
    gm.set_linear_id(id);
    CHECK(gm.index(0u) == id);  // coord_0 = (id / 1) % 100 = id
  }
}

// (b) GridMapping(4, -1) @ 100 CTAs: quantum Q = 100/10 = 10 -> extents (40, 60) in CTAs.
void test_grid_mapping_two_dims() {
  auto kinfo = host_info(100u, 512u);
  tca::rt::GridMapping gm(kinfo, 4, -1);
  CHECK(gm.dimension(0u) == 40u);
  CHECK(gm.dimension(1u) == 60u);
  for (unsigned id : {0u, 1u, 59u, 60u, 61u, 99u}) {
    gm.set_linear_id(id);
    CHECK(gm.index(0u) == (id / 60u) % 40u);  // row-major: last dim stride 1
    CHECK(gm.index(1u) == id % 60u);
  }
}

// (c) ThreadMapping(4, -1) @ 512 threads: 16 warps, Q_warps = 2 -> warps (8, 8),
// exposed in threads (256, 256); index over thread-granular extents.
void test_thread_mapping_two_dims() {
  auto kinfo = host_info(100u, 512u);
  tca::rt::ThreadMapping tm(kinfo, 4, -1);
  CHECK(tm.dimension(0u) == 256u);
  CHECK(tm.dimension(1u) == 256u);
  for (unsigned tid : {0u, 1u, 255u, 256u, 511u}) {
    tm.set_linear_id(tid);
    CHECK(tm.index(0u) == (tid / 256u) % 256u);
    CHECK(tm.index(1u) == tid % 256u);
  }
}

// (d) ThreadMapping(-1) @ 256 threads: minimum config (8 warps = 1 portion).
void test_thread_mapping_min() {
  auto kinfo = host_info(100u, 256u);
  tca::rt::ThreadMapping tm(kinfo, -1);
  CHECK(tm.dimension(0u) == 256u);
  for (unsigned tid : {0u, 1u, 128u, 255u}) {
    tm.set_linear_id(tid);
    CHECK(tm.index(0u) == tid);
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

// (e, cont.) Cta/Thr iterators walk identically — map/dosoc/loc ignored in v0.
void test_map_iterators() {
  auto kinfo = host_info(100u, 512u);
  tca::rt::GridMapping gm(kinfo, -1);
  tca::rt::ThreadMapping tm(kinfo, 4, -1);

  unsigned trips = 0u;
  for (tca::rt::CtaIterator it(gm, 0u, 100u, 7u, 3u, 20u); it.Has(); it.Next()) {
    CHECK(it.Index() == 7u + trips * 3u);
    ++trips;
  }
  CHECK(trips == 5u);

  trips = 0u;
  for (tca::rt::ThrIterator it(tm, 1u, 256u, 0u, 64u, 1024u); it.Has(); it.Next()) {
    CHECK(it.Index() == trips * 64u);
    ++trips;
  }
  CHECK(trips == 16u);
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

float add_f(float a, float b) { return a + b; }

// (h) Reducer algebra over <T, Slots>.
void test_seq_reducer() {
  // Scalar (Slots == 0): Init fills the declared init; Reduce folds through
  // the runtime-owned combiner; Combine returns T. Dyadic values — exact.
  tca::rt::SeqReducer<float, 0u> r0(0.0f, add_f);
  r0.Init();
  CHECK(r0.Combine() == 0.0f);  // identity-init semantics before any Reduce
  r0.Reduce(1.5f);
  r0.Reduce(2.25f);
  CHECK(r0.Combine() == 3.75f);
  r0.Init();  // argument-free re-initialization from the construction-time init
  CHECK(r0.Combine() == 0.0f);

  // Slotted (Slots == 32): Init fills the DECLARED init (2.0f here — proves
  // the fill is the construction-time value, not a hardcoded zero); per-slot
  // folds are independent; Combine returns SlotValue with Get.
  tca::rt::SeqReducer<float, 32u> r32(2.0f, add_f);
  r32.Init();
  CHECK(r32.Combine().Get(0u) == 2.0f);
  CHECK(r32.Combine().Get(31u) == 2.0f);
  r32.Reduce(3u, 0.5f);
  r32.Reduce(3u, 0.25f);
  r32.Reduce(31u, 1.0f);
  auto sv = r32.Combine();
  CHECK(sv.Get(3u) == 2.75f);   // 2.0 + 0.5 + 0.25
  CHECK(sv.Get(31u) == 3.0f);   // 2.0 + 1.0
  CHECK(sv.Get(0u) == 2.0f);    // untouched slot keeps the declared init
}

// The WarpReducer's HOST branch is the seq algebra (no warps on the host);
// pin it so the branch is compiled and pinned somewhere.
void test_warp_reducer_host_branch() {
  tca::rt::WarpReducer<float, 0u> w0(0.0f, add_f);
  w0.Init();
  w0.Reduce(1.5f);
  w0.Reduce(0.25f);
  CHECK(w0.Combine() == 1.75f);

  tca::rt::WarpReducer<float, 32u> w32(0.0f, add_f);
  w32.Init();
  w32.Reduce(7u, 0.5f);
  CHECK(w32.Combine().Get(7u) == 0.5f);
  CHECK(w32.Combine().Get(8u) == 0.0f);
}

}  // namespace

int main() {
  test_grid_mapping_tail();
  test_grid_mapping_two_dims();
  test_thread_mapping_two_dims();
  test_thread_mapping_min();
  test_seq_iterator();
  test_map_iterators();
  test_tensor_view();
  test_smem_view();
  test_umin_umax();
  test_seq_reducer();
  test_warp_reducer_host_branch();

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
