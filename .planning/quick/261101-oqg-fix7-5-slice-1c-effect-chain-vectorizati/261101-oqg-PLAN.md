---
phase: quick-261101-oqg
plan: 01
type: execute
wave: 1
depends_on: [261101-mt9]
files_modified:
  - internal/opt/tiling/lower_fragchunk.go
  - internal/opt/tiling/fragvec_test.go
  - internal/cg/cxx/walk.go
  - internal/cg/cxx/fix7vec_test.go
  - internal/cg/cxx/testdata/fix7vec.cu
  - internal/e2e/fix7vec_test.go
  - runtime/tests/cpu_tests.cpp
  - design/fix/fix7.md
  - .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/red-task1.txt
  - .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/red-task2.txt
  - .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/before-bench.txt
  - .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/after-bench.txt
  - .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/after-sha256-manifest.txt
  - .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/sha256-diff-vs-1b.txt
  - .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/suite-run.txt
  - .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/memcheck.txt
autonomous: true
requirements: [QUICK-261101-OQG]
must_haves:
  truths:
    - "every per-thread FRAG fill destination (LoadTileThread dst) in the emitted suite .cu whose trailing extent makes the row 16B-clean (extent*elemBytes % 16 == 0 — the float suite: trailing %4 == 0, the 1b shapes 4/8/16/64 qualify) carries alignas(16) on its declaration — so vec16_ok's dst-side check holds by construction, not by luck"
    - "the chunk policy (lower_fragchunk.go halving loop) keeps a covered dim's chunk at the largest vector-width multiple the priced budget allows when the dim's TRUE static extent is vector-width-compatible; where the extent or budget does not allow it, behavior is BYTE-IDENTICAL to 1b (no silent clamp, budget gate never violated — v62 pricing untouched)"
    - "the loader's mechanism stays AS-IS: runtime/tca/loader.h and runtime/tca/view.h ZERO diff (the copy_row/vec16_ok/cp_async paths are already pinned in cpu_tests.cpp); the ONLY runtime-dir change is a new pin test in runtime/tests/cpu_tests.cpp"
    - "the matmul family (gemm, mlp, mm_small, mm_mid, mm_splitk, qk_score) + layernorm + transpose .cu are sha256 bit-identical to the 1b manifest — alignas is scoped to per-thread FRAG fill destinations with 16B-clean trailing only, and layernorm/transpose (chunk trailing 1) are excluded by that rule"
    - "GPU suite 14/14 at --jobs 1, canary 0, maxdiff identical to 1b (the vec16 copy path is byte-exact vs the scalar path — same zero-fill suppression semantics), memcheck clean at FULL dims on every schedule-changed kernel"
    - "an honest --jobs 1 --bench-iters 20 before/after table is recorded (§5 gate: incremental, NOT gating — the fills are already amortized to 4 per (trip,chunk); a noise-level result recorded honestly is a PASS, a >5% regression on any kernel is not)"
  artifacts:
    - path: "internal/opt/tiling/fragvec_test.go"
      provides: "the RED unit pin on the chunk-policy bias: a crafted record whose budget forces the trailing covered SEQ dim (true static extent vector-compatible, e.g. 24, with a WARP-covered outer dim absorbing no budget) to reduce past a vector-width boundary — pin the planned chunk snaps to the largest vector-width multiple within budget, never the raw half"
    - path: "internal/cg/cxx/fix7vec_test.go + testdata/fix7vec.cu"
      provides: "the emission pin (unit byte golden): the fill-dst FRAG decl renders alignas(16) iff trailing is 16B-clean; the 1b acc Frag and non-fill FRAGs render UNCHANGED"
    - path: "internal/e2e/fix7vec_test.go"
      provides: "the e2e pin through the real optimizer on the six amortized kernels: emitted .cu decl lines carry alignas(16) with 16B-clean trailing extents; the unchanged-kernel set is byte-identical to the 1b sha256 manifest"
    - path: "runtime/tests/cpu_tests.cpp"
      provides: "the eligibility-chain pin: a Frag declared with the EMITTED spelling (alignas(16)) and the EMITTED dims (4/8/16/64 trailing) reports vec16_ok(Data() + row*trailing, aligned src + row*trailing) == true for EVERY row — the loader's uint4 fast-path eligibility holds row-invariantly on the emitted shape"
    - path: ".planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/after-bench.txt"
      provides: "the honest --jobs 1 benchmark table vs the 1b numbers"
  key_links:
    - from: "internal/cg/cxx/walk.go frag decl printer"
      to: "the per-thread fill classification"
      via: "the emitter computes the set of FRAG buffers that are LoadTileThread dsts (the same FRAG-class v==0 classification memory.go's emitFill uses) BEFORE the decl section runs; only decls in that set AND with trailing extent*elemBytes % 16 == 0 gain the alignas(16) spelling"
      pattern: "alignas\\(16\\) tca::rt::Frag<"
    - from: "internal/opt/tiling/lower_fragchunk.go halving loop"
      to: "the priced budget (siteWindowBytes — the ONE shared source)"
      via: "the snap-down target = largest multiple of vecElems (16/elemBytes) that is <= the halved size AND keeps fragChunkBytes <= budget; the budget comparison itself is untouched"
      pattern: "fragChunkBytes\\(.*\\) > budget"
    - from: "the emitted alignas(16) decl"
      to: "detail::vec16_ok / copy_row's vec16 branch (loader.h, untouched)"
      via: "Data() of a 16B-aligned object + row starts at row*trailing elements with trailing %4 == 0 ⇒ every drow is 16B-aligned; the src side stays the runtime's per-row check (dynamic dims cannot be guaranteed statically — the fallback scalar path is the pinned mechanism's contract)"
      pattern: "vec16_ok\\(drow"
---

<objective>
fix7 §5 slice 1c — effect-chain vectorization: the emission-side shape choice that makes the
loader's EXISTING uint4 fast path actually fire for the streamed FRAG fills, with the
emission-side pins §5 names as new.

Purpose: `detail::load_tile_rows` / `copy_row` already carry a 128-bit vector path
(`vec16`, loader.h:57-95) gated per row on `sizeof(E)==4 && trailing % 4 == 0 &&
vec16_ok(drow, srow)` — but in the emitted code it can never fire: (a) the halving loop in
`lower_fragchunk.go` snaps chunk dims to raw halves (4→2, 12→6, 24→6…), so a budget-forced
chunk can land on a trailing extent that fails `trailing % 4 == 0`; (b) `Frag::data_`
(view.h:228) has no `alignas`, so the dst side of `vec16_ok` is 4-byte-aligned by type and
fails the 16B check on EVERY row, emitted shapes notwithstanding. §5's scope is exactly this:
emission-side shape choice ONLY — pin chunk trailing extents to vector-width multiples where
the true const extents allow, and pin the emission so the eligibility chain holds. The
runtime mechanism, the pricing model (v62), and the schedule pass stay untouched.

Output: the biased chunk policy + the alignas(16) fill-dst decl emission + four pins (unit
chunk policy, unit emission golden, e2e eligibility, CPU eligibility chain) + the §9 GPU
verification record with an honest incremental benchmark table.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@design/fix/fix7.md
@.planning/quick/261101-mt9-fix7-4-slice-1b-bracket-amortization/261101-mt9-SUMMARY.md

<interfaces>
<!-- Key sites the executor needs; extracted from the codebase. No exploration required. -->

From internal/opt/tiling/lower_fragchunk.go (the 1a chunk policy — the ONLY shape-choice site):
- planSiteChunk(id, sf): chunks[i] = effectiveTile (SEQ) or ceilDiv(effectiveTile,32) (C1b WARP),
  in covered-tensor order; staticIterExtent caps const dims to TRUE extent BEFORE the halving
  loop; budget = siteWindowBytes(sf, m, ClassFRAG, lr.params) — the ONE shared pricing source,
  NEVER reimplemented; halving loop: outermost reducible SEQ dim first, floor 1, fail closed.
- staticIterExtent(iter): TRUE compile-time extent — BIV max const payload or the dim param's
  canonical "e<extent>" spelling. NEVER ir.TypeOf (post-typeinfer nil computed cell gotcha);
  use ir.ConstValueOf for const payloads.

From internal/cg/cxx/walk.go (the FRAG decl printer, ~line 120-135):
- The `for _, f := range e.frags` loop prints `tca::rt::Frag<%s, %du, %s> %s;` via
  bufferTypeOf(f) (element spelling + []ir.DimRef with .Extent). This is where alignas(16)
  joins the spelling, conditioned on the fill-dst set + 16B-clean trailing.

From internal/cg/cxx/memory.go (the fill emission, ~line 195-217):
- emitFill chooses `tca::rt::LoadTileThread` (per-thread, FRAG class) vs LoadTileCta (v==1,
  cooperative SMEM); the emitted call is `LoadTileThread(<thread>, <bufView>,
  tca::rt::TensorWindow(<view>, <coords>))`. The fill-dst set for the alignas rule MUST be the
  same FRAG-class per-thread classification (fail closed if it cannot be computed exactly).

From runtime/tca/loader.h (MECHANISM AS-IS — zero diff in this slice):
- copy_row (line ~82): `if constexpr (sizeof(E)==4u) { if (trailing % 4u == 0u && vec16_ok(dst,
  src)) { …vec16 run + scalar tail + zero-fill… } }` — the scalar loop is the retained
  reference; both paths byte-identical results (suppression zero-fill semantics).
- vec16_ok (line ~71): both endpoints 16B-aligned — `(uintptr & 15u) == 0u`.
- The CPU unit layer already pins vec16_ok and the copy semantics (runtime/tests/cpu_tests.cpp
  ~line 459-467: aligned true, +1 misaligned false, +4 aligned true).
- Frag::data_ (runtime/tca/view.h:228): `T data_[kElems > 0u ? kElems : 1u] = {};` — NO alignas;
  class alignment = alignof(T) = 4 for float. Hence the emitted DECLARATION must carry
  `alignas(16)` (valid on a variable declaration; the object then reports 16B alignment).

The 1b baseline (BEFORE for this task's table; --jobs 1 --bench-iters 20, µs):
dup_expr 1,844.5 / dup_const 2,994.3 / axpy3 3,667.5 / scale_copy 3,243.6 / row_sum 5,232.8 /
softmax 1,005.3 / transpose 948.4 / layernorm 8,929.7 / mlp 6,597.4 / mm_small 2,881.3 /
mm_mid 6,464.3 / mm_splitk 6,089.3 / gemm 9,702.2 / qk_score 17,612.3.
Chunk trailing extents pinned by 1b on the six amortized kernels: 4/8/16/64 (all 16B-clean).
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: RED — the three eligibility pins (chunk-policy bias, emission alignas, CPU eligibility chain)</name>
  <files>internal/opt/tiling/fragvec_test.go, internal/cg/cxx/fix7vec_test.go, internal/cg/cxx/testdata/fix7vec.cu, internal/e2e/fix7vec_test.go, runtime/tests/cpu_tests.cpp, .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/red-task1.txt, .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/red-task2.txt</files>
  <behavior>
    - Pin A (unit, tiling, RED): a crafted record (the fragchunk_test.go fillcoord idiom) with covered = [C1b-WARP outer dim, SEQ trailing dim whose staticIterExtent is vector-compatible, e.g. true extent 24] and a priced budget that forces the trailing chunk below 12 elements. Expected planned chunk trailing: the largest multiple of vecElems (16/elemBytes = 4 for float) within budget — 4 for a budget that today's halving would satisfy with 6 (24→12→6). RED today: the policy halves to 6. The pin records its arithmetic in-comment (budget bytes, vecElems, snap target).
    - Pin B (unit emission byte golden + e2e, RED): for the suite's six amortized kernels (row_sum, softmax, scale_copy, axpy3, dup_expr, dup_const), every emitted per-thread FRAG fill-dst declaration with trailing*elemBytes % 16 == 0 renders as `alignas(16) tca::rt::Frag<...>`; the 1b accumulator Frag (not a fill dst) and all other decls render byte-unchanged. RED today: no alignas anywhere. The e2e half (internal/e2e/fix7vec_test.go, the fix7bracket_test.go positional-identification idiom) emits through the real optimizer and asserts the decl lines + that gemm/mlp/mm_small/mm_mid/mm_splitk/qk_score/layernorm/transpose emitted text is byte-identical to the 1b sha256 manifest constants.
    - Pin C (runtime CPU chain pin, expected GREEN immediately — record honestly which pins passed at RED): declaring a Frag with the EMITTED spelling (`alignas(16) tca::rt::Frag<float, R, …trailing…>`) and the EMITTED dims (trailing in {4,8,16,64}) makes vec16_ok(frag.Data() + row*trailing, src + row*trailing) true for EVERY row (0..nrows-1) with a 16B-aligned src — proving the dst-side alignment is row-invariant under trailing %4 == 0, so the loader's per-row check reduces to the src side (runtime-checked, fallback pinned). Also assert the same WITHOUT the alignas spelling vec16_ok(Data()) is false for some row — the emission-side choice is what makes eligibility reachable at all.
  </behavior>
  <action>
    Write the three pins BEFORE any implementation, following the 1b RED discipline: test-only
    commits, RED evidence captured to the task dir. Do NOT weaken or touch any existing pin —
    fragchunk_test.go's 1a assertions (D-fix6-03 included), bracket_amort_test.go, fix7bracket_test.go,
    fillcoord/fragchunk goldens all stay untouched in this task. For Pin B's unit half, craft the
    fill fixture at the cxx level mirroring the 1b acc/fill arrangement (a per-thread LoadTileThread
    into a %4-trailing FRAG) rather than duplicating an entire kernel walk. Name identifiers
    canonically (TestFragChunkVecBias, TestFix7VecFillDecl / fix7vec.cu, TestFix7VecEligibility,
    TEST(Loader, VecEligibilityEmittedShape) or per cpu_tests.cpp's existing naming).
  </action>
  <verify>
    <automated>go test ./internal/opt/tiling/ -run TestFragChunkVecBias -v 2&gt;&amp;1 | tee red-task1.txt; go test ./internal/cg/cxx/ ./internal/e2e/ -run 'Fix7Vec' -v 2&gt;&1 | tee red-task2.txt; go test ./... 2&gt;&1 | tail -5; grep -c 'alignas(16)' red-task2.txt || true</automated>
    RED evidence: TestFragChunkVecBias FAILS (chunk 6, want 4); the Fix7Vec emission/e2e pins FAIL
    (no alignas in emitted decls, manifest mismatch on the six); ALL pre-existing pins stay green;
    Pin C (cpu_tests) PASSES — recorded in red-task2.txt header as an honestly-green chain pin.
    Runtime CPU build: cmake -S runtime -B &lt;tmp&gt; -DTCACOMP_RUNTIME_TESTS=ON &amp;&amp; cmake --build &lt;tmp&gt; &amp;&amp; ctest --test-dir &lt;tmp&gt; --output-on-failure (target name per runtime/tests/CMakeLists.txt).
  </verify>
  <done>Three pins committed test-only (RED evidence files present); pre-existing suite fully green; the RED failure output shows the exact current shape (halved-to-6 chunk, alignas-free decls) — the GREEN target is unambiguous.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: GREEN — the two emission-side changes (chunk snap-to-vector policy + alignas(16) fill-dst decls)</name>
  <files>internal/opt/tiling/lower_fragchunk.go, internal/cg/cxx/walk.go, internal/cg/cxx/fix7vec_test.go, internal/cg/cxx/testdata/fix7vec.cu, internal/e2e/fix7vec_test.go</files>
  <behavior>
    - The halving loop's snap rule: when a reducible SEQ dim's static true extent is a multiple of vecElems (16/elemBytes), each halving step on that dim snaps DOWN to the largest multiple of vecElems that is <= the raw half AND keeps fragChunkBytes <= budget; if no multiple of vecElems satisfies the budget, the dim keeps the raw-halving floor-1 behavior EXACTLY (and the existing fail-closed error fires when nothing can shrink — unchanged). WARP dims never halve (unchanged); the budget source and comparison are untouched (v62 owns pricing); the inter-site agreement check runs after the bias, unchanged. On the current suite the bias must be a NO-OP (trailing extents stay 4/8/16/64) — proven by the sha256 diff in Task 3 showing only decl-line changes on the six.
    - The decl rule: the emitter computes the set of FRAG buffers that are per-thread fill destinations (LoadTileThread dsts — the identical FRAG-class v==0 classification memory.go's emitFill uses; compute it in a pre-pass before the decl section) and prints `alignas(16) ` before `tca::rt::Frag<` iff the buffer is in that set AND trailing extent*elemBytes % 16 == 0. Everything else — SMEM views, the 1b acc Frag, chunk-1 layernorm/transpose fills, all matmul-family decls — prints byte-unchanged. Elem-size generality: vecElems = 16/elemBytes; for non-4-byte elements where trailing*elemBytes % 16 != 0, no alignas (the mechanism gate is float-specific today; do not extend it).
  </behavior>
  <action>
    Implement the two changes against the Task 1 pins. Keep both changes minimal and local: the
    snap is a few lines inside the existing halving loop (a snapTarget helper beside
    fragChunkBytes); the decl change is the fill-dst pre-pass + a conditional prefix in walk.go's
    frag loop. If the fill-dst classification cannot be computed exactly (e.g. a fill path that
    does not flow through the v==0 FRAG classification), FAIL CLOSED: stop, record the blocking
    shape in the summary, and ask — never approximate the set (a wrong alignas on a matmul-family
    decl breaks the bit-identity gate; a missed one silently dead-fires the slice's purpose).
    Regenerate any drifted golden ONLY via the established golden-helper path with the
    re-derivation arithmetic in-comment (the 1b discipline); if fix7vec.cu itself drifts during
    GREEN, regenerate only after re-confirming the RED evidence captured it.
  </action>
  <verify>
    <automated>go build ./... &amp;&amp; go vet ./... &amp;&amp; go test ./... 2&gt;&amp;1 | tail -8; go list -m all | wc -l; git diff --stat runtime/tca/ internal/opt/tiling/judge.go internal/opt/tiling/search.go internal/opt/tiling/record.go internal/opt/schedule/</automated>
    All green; module count 1; git diff shows runtime/tca/loader.h + view.h ZERO diff, judge/search/record/schedule ZERO diff (only lower_fragchunk.go in tiling; walk.go + new pins in cxx/e2e).
  </verify>
  <done>Task 1 pins GREEN unweakened; suite 25/25 packages ok; module count 1; the diff scope gate holds (mechanism/pricing/schedule zero-diff); the emitted six carry alignas(16) on exactly the qualifying fill-dst decls.</done>
</task>

<task type="auto">
  <name>Task 3: §9 GPU verification + honest benchmark table + fix7.md §5 status</name>
  <files>design/fix/fix7.md, .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/before-bench.txt, .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/after-bench.txt, .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/after-sha256-manifest.txt, .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/sha256-diff-vs-1b.txt, .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/suite-run.txt, .planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/memcheck.txt</files>
  <action>
    Run the n63/qwo §9 protocol in exact order (fix7.md §9):
    1. BEFORE benchmarks FIRST: scripts/build.sh is NOT rebuilt yet — capture before-bench.txt
       from the CURRENT (1b) binary... NO — correct order: run `scripts/build.sh` FIRST (the
       gitignored prebuilt shadows source), then capture the BEFORE table from a stash-free 1b
       baseline — the 1b after-bench.txt IS the sanctioned BEFORE (copy it verbatim as
       before-bench.txt, the 1b precedent); do not re-measure a binary you are about to replace.
    2. Hermetic: go build/vet/test exit 0, module count 1, diff-scope gate as Task 2.
    3. Rebuild via scripts/build.sh, then sha256 manifest of all emitted kernels vs the 1b
       manifest: expected DIFFER = exactly the six amortized kernels (decl lines only — verify
       the textual diff is ONLY the alignas spelling; if any chunk dim changed, the Task 2 bias
       was not a no-op: STOP and re-derive, do not proceed); IDENTICAL = the matmul family + mlp
       + layernorm + transpose.
    4. python3.13 -I testing/orchestrator.py --jobs 1 — suite 14/14, canary 0, maxdiff IDENTICAL
       to 1b on every kernel (the vec16/scalar copy paths are byte-exact). Log: suite-run.txt.
    5. Memcheck at FULL dims on every schedule-changed kernel (the six). Log: memcheck.txt.
    6. Benchmarks --jobs 1 --bench-iters 20 AFTER: after-bench.txt. §5's gate is an honest
       INCREMENTAL table, NOT gating — the fills are already amortized (4 per (trip,chunk) on
       row_sum), so the expected effect is small; record the numbers honestly including
       noise-level deltas (mm_small-style machine-state spread precedent). A >5% regression on
       ANY kernel vs the 1b best-known-good is a HARD stop: investigate before shipping.
    7. Update fix7.md §5 with the STATUS block (what landed, the DECISION on the two shape
       choices, pins, the §9 record) mirroring the §4 STATUS style.
  </action>
  <verify>
    <automated>python3.13 -I testing/orchestrator.py --jobs 1 2&gt;&amp;1 | tee suite-run.txt | tail -20; sha256sum testing/generated/**/*.cu 2>/dev/null | sort &gt; after-sha256-manifest.txt; grep -c PASS suite-run.txt</automated>
    Suite 14/14 PASS + canary 0 recorded; memcheck 0 errors on the six; sha256 diff scoped to
    the six (diff text shows only alignas decl-line changes); after-bench.txt has all 14 kernels;
    fix7.md §5 STATUS committed.
  </verify>
  <done>§9 record complete in the task dir (suite-run, memcheck, manifest, sha256-diff, before/after tables); all §9 global gates hold (no kernel >5% slower vs 1b best-known-good; maxdiff identical; matmul family bit-identical); fix7.md §5 STATUS landed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| compiler → generated .cu | the only output surface; the alignas spelling and chunk dims cross into user-compiled code — must stay standard C++11 (alignas on a declaration) and nvcc-12.2/g++-11.4-clean |
| compiler → runtime headers | NONE in this slice — loader.h/view.h are zero-diff by gate (mechanism as-is) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-oqg-01 | Tampering | goldens/pins | mitigate | drifted goldens regenerated ONLY via the golden helper after recorded RED evidence (red-task*.txt), re-derivation arithmetic in-comment — the 1b discipline; no pin weakened |
| T-oqg-02 | Tampering | scope creep into pricing/schedule/runtime | mitigate | git-diff scope gate in every task: judge.go/search.go/record.go, internal/opt/schedule/, runtime/tca/loader.h, runtime/tca/view.h ZERO diff — the bias reads the budget from siteWindowBytes, never reimplements it |
| T-oqg-03 | Tampering | matmul-family bit-identity gate | mitigate | alignas scoped to per-thread FRAG fill dsts with 16B-clean trailing; sha256 manifest vs the 1b hashes is a hard Task-3 gate; fail-closed rule if the classification is not exact |
| T-oqg-04 | DoS (perf) | suite regression | mitigate | honest --jobs 1 before/after table; >5% regression on any kernel = HARD stop and investigate, never ship silently |
| T-oqg-SC | Tampering | package installs | accept | none — stdlib-only module (count 1 enforced); no installs in this slice |
</threat_model>

<verification>
- RED-first: all three pins exist and fail (Pins A/B) or are honestly recorded green (Pin C) BEFORE any implementation; evidence in red-task1.txt / red-task2.txt.
- Hermetic: go build/vet/test all green; `go list -m all | wc -l` == 1; scope gate zero-diff on pricing/schedule/loader/view.
- GPU: suite 14/14 canary 0 at --jobs 1; maxdiff identical to 1b; memcheck clean at FULL dims on the six changed kernels; sha256: six differ (decl lines only), eight bit-identical.
- Benchmark: honest --jobs 1 --bench-iters 20 table; incremental non-gating per §5; no kernel >5% slower vs 1b.
</verification>

<success_criteria>
- The emitted per-thread FRAG fill destinations with 16B-clean trailing extents carry alignas(16), and the chunk policy keeps vector-width multiples where true extents + budget allow — so the loader's existing uint4 fast path is reachable by construction (eligibility chain pinned at unit, e2e, and CPU levels).
- Everything §5 marks as out of scope is untouched: loader.h/view.h mechanism, v62 pricing, schedule pass, matmul-family/layernorm/transpose emitted bytes.
- fix7.md §5 records STATUS, DECISION, pins, and the honest §9 table.
</success_criteria>

<output>
Create `.planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/261101-oqg-SUMMARY.md` when done, following the 1b SUMMARY shape (dependency graph, tech tracking incl. drifted-pin records, task commits, §9 record, before/after table, gates verdict).
</output>
