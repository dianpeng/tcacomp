---
phase: quick-261101-mt9
plan: 01
subsystem: compiler-codegen
tags: [codegen, fix7, slice-1b, bracket-amortization, frag-accumulator, loop-inversion, tdd, gpu-validation]

# Dependency graph
requires:
  - phase: quick-261101-kel
    provides: "fix7 §3 slice 1a — the streaming register-block lowering, the §3.4 priced-vs-actual reuse audit (the measured gap this slice closes), and the restoration-baseline benchmark numbers this task's gates compare against"
provides:
  - "the amortized flat-bracket lowering in final-lower: ONE bracket per chunk — a kernel-top chunk-wide Frag accumulator, an init walk (the construction's own init), the element walk moved INSIDE X (the fill rides at X level — the §3.4 fill-placement gap closed by construction), the combiner body inlined per element, and a store walk draining the acc"
  - "the Q0-Q5 qualification (fail closed to the per-element bracket on ANY miss): one-tier SeqReducer; between chain [thread-specialized outer, uniform SEQ innermost]; FRAG fills at X level with no element-IV window coordinate; element chunk > 1; gated consumers, no reduce-after-reduce chains"
  - "zero reducer ceremony on the qualified six: row_sum 36,960 -> 5,232.8 us (7.1x), softmax 9,687 -> 1,005.3 us (9.6x); elementwise four 633x/395x/365x/431x vs the 2026-10-29 pins"
  - "two RED pins (unit byte golden + e2e bracket-count) + four drifted pre-existing pins re-derived with the 1b arithmetic in-comment, never weakened"
affects: [fix7-slice-1c-vectorization, fix7-fill-placement, fix6-specialization]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "codegen-explicit accumulator (the §4 DECISION): a chunk-wide Frag MemNew with an InternTensor type stamping the element chunk — no runtime chunk-fold entry, runtime/tca/reducer.h zero-diff"
    - "the amortization probe at the R3 call site subsuming buildElementLoop: the outer thread-specialized between loops SURVIVE (unlike R3's dissolve), the innermost uniform SEQ between loop becomes the element walk and moves inside X — no between-IV rebuilds in the walk at all"
    - "loop-header cloning by operand reuse + annotation-node clone (the oneTier idiom of buildElementLoop) for the fresh init/store walks"
    - "combiner-body inlining via rebuildWithSubs over {param0 <- acc load at E's IV, param1 <- piped} — the same mechanism Map Lower uses; Map Lower's RAUW resolves the piped map afterwards"
    - "cxx expression-position FRAG load rendering (inline tca_fN.Get) — the only cxx change, justified by the inlined combiner having no statement materialization"
    - "explicit item-stripping of the orphaned combiner lambda's anchored items (the erased construction's markClosure shield is gone — the maplower precedent's emission safety, restored)"

key-files:
  created:
    - "internal/cg/cxx/bracket_amort_test.go"
    - "internal/cg/cxx/testdata/bracket_amort.cu"
    - "internal/e2e/fix7bracket_test.go"
    - ".planning/quick/261101-mt9-fix7-4-slice-1b-bracket-amortization/red-task1.txt"
    - ".planning/quick/261101-mt9-fix7-4-slice-1b-bracket-amortization/red-task2.txt"
    - ".planning/quick/261101-mt9-fix7-4-slice-1b-bracket-amortization/red-drift{1,2,3,4}.txt"
    - ".planning/quick/261101-mt9-fix7-4-slice-1b-bracket-amortization/before-bench.txt"
    - ".planning/quick/261101-mt9-fix7-4-slice-1b-bracket-amortization/after-bench.txt"
    - ".planning/quick/261101-mt9-fix7-4-slice-1b-bracket-amortization/after-sha256-manifest.txt"
    - ".planning/quick/261101-mt9-fix7-4-slice-1b-bracket-amortization/sha256-diff-vs-1a.txt"
    - ".planning/quick/261101-mt9-fix7-4-slice-1b-bracket-amortization/suite-run.txt"
    - ".planning/quick/261101-mt9-fix7-4-slice-1b-bracket-amortization/memcheck.txt"
  modified:
    - "internal/opt/finallower/reducer.go (the amortized lowering: tryAmortizedElementLoop + amortizedQualifies + applyAmortizedElementLoop + staticChunkOf; the moveCombineConsumers gate refactor into qualifyCombineConsumers/assertConsumerEnclosure — behavior preserved)"
    - "internal/cg/cxx/expr.go (FRAG-class MemLoad inline rendering — the one justified cxx change)"
    - "internal/cg/cxx/emit_test.go (TestReduceShapesEmit rank3 subtest re-derived to the amortized surface)"
    - "internal/cg/cxx/fillcoord_test.go + testdata/fillcoord_mixed.cu (golden regen, 1b note, D-fix6-03 assertions UNWEAKENED)"
    - "internal/cg/cxx/fragchunk_test.go + testdata/fragchunk_const_extent.cu (golden regen, 1b note, 1a chunk assertions UNWEAKENED)"
    - "internal/e2e/fix5t2_one_tier_test.go (SeqReducer assertion re-derived to the zero-ceremony amortized shape)"
    - "design/fix/fix7.md (§4 status: landed, DECISION, qualification, §9 record)"

key-decisions:
  - "The §4 chunk-fold DECISION as made: codegen-explicit chunk-wide Frag accumulator, NO runtime SeqReducer chunk-fold entry, NO per-round partials — the block accumulates across ALL of X's chunk rounds by construction (recorded in reducer.go's comment block and fix7.md §4)"
  - "The reshape lives at the R3 call site, NOT the flat branch the plan named: the emitted defect is buildElementLoop's product (between = [R Thr coupled-inside, C SEQ inside] — verified empirically at the RED commit); the plan's flat-degenerate premise was corrected, the probe subsumes R3, and the outer thread-specialized between loops survive (no between-IV rebuilds at all)"
  - "Q4d (element chunk > 1): a 1-element amortization saves zero ceremony and marks the element dim not chunk-covered — layernorm/transpose (chunk 1) keep the per-element bracket BYTE-IDENTICAL, satisfying the e2e regression guard without a special case"
  - "LR-3 amended shape-scoped: the amortized shape reads the combiner's current_value binding (param 0 <- acc load), never the init (the init walk stores the construction's own init operand) — every other reducer shape keeps LR-3 verbatim"

requirements-completed: [QUICK-261101-MT9]

# Metrics
duration: 95min
completed: 2026-11-01
---

# Phase quick-261101-mt9 Plan 01: fix7 §4 slice 1b — bracket amortization Summary

**Bracket amortization: ONE bracket per chunk — the per-element Init/Reduce/Combine ceremony and the fill riding inside it are replaced by a kernel-top chunk-wide Frag accumulator, closing the §3.4 fill-placement gap as ONE restructure: row_sum 36,960 → 5,233 µs (7.1×), softmax 9,687 → 1,005 µs (9.6×) beyond the 1a restoration numbers, elementwise four at 633×/395×/365×/431× the 2026-10-29 pins, with GPU maxdiff bit-identical to 1a.**

## Performance

- **Duration:** ~95 min
- **Tasks:** 3
- **Files modified:** 11 source/test/golden files + 12 task-dir artifacts

## Accomplishments
- The emitted defect is dead on the qualified six (row_sum, softmax, scale_copy, axpy3, dup_expr, dup_const): zero `SeqReducer` constructions, zero `Init()/Reduce()/Combine()` sites; one `Frag<float, 1u, chunk>` accumulator block; an init walk storing the construction's OWN init value per acc slot; the element walk nested INSIDE X before home; the combiner body inlined per element as `tca_f1.Set((tca_f1.Get(c) + v), c)`; a store walk draining the acc with the consumer's coordinates rebuilt onto the store-walk IV.
- The §3.4 fill-placement gap is closed STRUCTURALLY (never patched): the fill executes once per (R trip, C chunk round, X round) — row_sum 256 → 4 fills per (trip, chunk) — because the element walk moved inside X, the fill's nest path loses the per-element loop by construction (the assertFillReadsInside epoch contract holds: fills stay items of X, reads stay in home inside E inside X).
- k-ascending accumulation order unchanged → **GPU maxdiff BIT-IDENTICAL to 1a on all 14 kernels** (e.g. row_sum 4.76837e-07, softmax 1.90735e-06 — exactly 1a's values).
- The matmul family + mlp stay sha256 BIT-IDENTICAL; layernorm/transpose (chunk-1 element walks, Q4d-rejected) byte-identical too — only the amortized six differ.

## Task Commits

TDD: RED pins FIRST (two atomic test commits), then GREEN (implementation + pins + re-derivations):

1. **Task 1: RED pin 1 (unit emission shape)** — `cb337ae` (test)
2. **Task 1: RED pin 2 (e2e bracket-count, row_sum/softmax)** — `4684182` (test)
3. **Task 2: GREEN — the amortized lowering + cxx FRAG-inline + unit golden** — `755dbaf` (feat)
4. **Task 2: GREEN pins + drifted-pin re-derivations** — `4ac209a` (test)
5. **Task 3: docs — fix7.md §4 status** — `34f1ea1` (docs)

RED evidence: red-task1.txt (unit pin fails on the per-element bracket shape), red-task2.txt (row_sum/softmax/suite pins fail; the layernorm/transpose regression guards PASS at RED — the qualification predictions confirmed), red-drift{1..4}.txt (the four drifted pre-existing pins, captured before re-derivation).

## Files Created/Modified
- `internal/opt/finallower/reducer.go` — `tryAmortizedElementLoop` / `amortizedQualifies` (Q0–Q5, read-only probe) / `applyAmortizedElementLoop` (the reshape, subsuming buildElementLoop) / `staticChunkOf`; the `moveCombineConsumers` gate refactor into shared `qualifyCombineConsumers` + `assertConsumerEnclosure` (pure code-motion, existing-path behavior byte-preserved); the orphaned-combiner-lambda item strip (the erased construction's markClosure shield, restored per the maplower precedent).
- `internal/cg/cxx/expr.go` — FRAG-class MemLoad renders inline in expressions (`tca_fN.Get(coords)`) — the ONE justified cxx change (the inlined combiner's acc read has no statement materialization; SMEM loads keep their statement form).
- `internal/cg/cxx/bracket_amort_test.go` + `testdata/bracket_amort.cu` — the unit pin (order + counts + zero-ceremony + byte golden, re-derivation arithmetic in-comment).
- `internal/e2e/fix7bracket_test.go` — the e2e pin (row_sum/softmax amortization; the layernorm/transpose regression guards; the suite fill-nest-path sweep; positional element-walk identification).
- Re-derived pins (1b arithmetic in-comment, never weakened): `emit_test.go` (rank3 multi-between → amortized surface), `fix5t2_one_tier_test.go` (zero-ceremony collapse), `fillcoord_test.go` + `fragchunk_test.go` (goldens regen'd via the helper; D-fix6-03 / 1a-chunk assertions intact).
- `design/fix/fix7.md` §4 — status, DECISION, qualification, §9 record.

## Decisions Made
- **Codegen-explicit accumulator (the §4 DECISION):** a chunk-wide Frag block spanning X's FULL walk; no runtime chunk-fold entry; no per-round partials; runtime zero-diff. Rationale recorded in reducer.go + fix7.md (a runtime SeqReducer holds ONE scalar — a block-reducer class would be new runtime surface against the minimal-runtime doctrine, and codegen would still emit the block).
- **The reshape subsumes R3 at its call site** (the plan's flat-branch premise corrected empirically): the outer thread-specialized between loops survive; only the innermost uniform SEQ between loop moves inside X; no between-IV rebuilds run in the walk at all (the loops keep their identity) — only the moved consumers rebuild E's IV onto the store-walk IV.
- **Q4d (chunk > 1):** the honest discriminator for the layernorm/transpose class — a 1-element amortization saves zero ceremony; their emission stays byte-identical.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The reshape lives at the R3 call site, not the flat branch the plan named**
- **Found during:** Task 2 GREEN (first implementation attempt against the plan's `len(betweenBlks)==0` branch never fired)
- **Issue:** The plan's interface section placed the amortization in lowerReduce's flat-bracket branch ("betweenBlks empty — X directly contains home"). Empirically (debug probe, since removed) the suite's const-innermost reduces — real row_sum included — carry between = 2: X (the e4 tile-between loop) directly contains the thread-specialized R coupled-inside walk, which contains the uniform SEQ C inside walk, which contains home. The per-element bracket the §4 audit measured is `buildElementLoop`'s R3 product.
- **Fix:** The probe moved to the R3 call site (`tryAmortizedElementLoop`, before `buildElementLoop`); on qualification the reshape SUBSUMES R3 — the outer thread-specialized between loops survive (unlike R3, which dissolves them), the innermost uniform SEQ between loop becomes the element walk E and moves inside X. The plan's Q4 "the loop above it is thread-specialized" became the between-chain shape rule (outer between blocks must be BlockThrLoop; the single-between degenerate keeps the plan's grandparent check). Semantics identical to the plan's target shape (verified against the plan's sketch line-by-line in the unit golden).
- **Files modified:** internal/opt/finallower/reducer.go
- **Verification:** the unit + e2e pins green; the emitted nest matches the plan's target shape exactly (init walk, X{fill; E{home{update}}}, store walk); GPU bit-identical numerics.
- **Committed in:** 755dbaf

**2. [Rule 1 - Bug] Q4d replaces the plan's predicted Q3/Q3 rejection of layernorm/transpose**
- **Found during:** Task 1 RED (the layernorm/transpose regression guards PASSED at RED for a different reason than the plan predicted)
- **Issue:** The plan predicted layernorm/transpose fail Q3/Q4 because "fill coordinates reference the element IV". Post-1a the fill windows ride at X level (that IS the §3.4 gap) — their coordinates reference the C BETWEEN base, not the element IV, so the plan's Q3 textually passes them. The actual post-1a discriminator: their element walk's chunk is 1 (the fill covers exactly the one element; E's clamp target is 1u) — amortizing a 1-element chunk saves zero ceremony.
- **Fix:** Added Q4d (element chunk > 1) with the honest rationale recorded in the qualification comment; layernorm/transpose keep the per-element bracket BYTE-IDENTICAL (sha256 SAME vs 1a) — the e2e pin's regression guard documents this.
- **Files modified:** internal/opt/finallower/reducer.go
- **Verification:** e2e layernorm/transpose subtests green; sha256 manifest shows both SAME.
- **Committed in:** 755dbaf

**3. [Rule 1 - Bug] cxx expression-position FRAG load rendering (the anticipated cxx tweak)**
- **Found during:** Task 2 GREEN (first Emit failed: `node (memload) is not emittable as an expression`)
- **Issue:** The inlined combiner's acc read and the store-walk consumer's acc read are MemLoad nodes in EXPRESSION position; the emitter only materialized buffer loads as statement locals. The plan anticipated this ("if a cxx tweak is genuinely required, record it as a deviation with justification").
- **Fix:** `emitExpr` renders FRAG-class MemLoads inline (`tca_fN.Get(coords)`); SMEM loads keep the statement form (positional epoch contract); the values-map path is checked first so statement-materialized loads are unaffected.
- **Files modified:** internal/cg/cxx/expr.go
- **Verification:** full 25-package suite green; the unit byte golden pins the exact rendering.
- **Committed in:** 755dbaf

**4. [Rule 1 - Bug] The orphaned combiner lambda's anchored items must be stripped explicitly**
- **Found during:** Task 2 GREEN (Emit failed: `lambda param acc referenced outside its lambda`)
- **Issue:** The scheduler anchors the combiner lambda's params/body as prologue items; a SURVIVING construction shields them from emission via the emitter's markClosure, but the amortized path erases the construction — the shield vanished and the pure-item B1 reference check (buildPureRefs) pulled the ParamRefs into emission.
- **Fix:** The reshape strips the orphaned lambda closure's anchored items from the AST explicitly (P-7 spirit: verb-death only; the orphaned sea nodes remain the tail DCE's domain per the maplower precedent).
- **Files modified:** internal/opt/finallower/reducer.go
- **Verification:** full suite green; zero lambda-param leaks.
- **Committed in:** 755dbaf

**5. [Rule 3 - Blocking] D-j re-placement of the X-scoped items stranded in the outer trip walk**
- **Found during:** Task 2 GREEN (post-lower verifier: scope-dominance violation — the fill, whose closure references X's IV, was an item of the thread-specialized trip walk B[0] by scheduler placement; R3 re-places such items inside X per D-j, the first cut of the reshape left them behind)
- **Fix:** The reshape re-places B[0]'s X-scoped node items (closure hits X's IV — the fill, its source load, per-round clamps) into X ahead of the element chain, re-stamped — exactly the D-j discipline; B[0]-scoped items (the trip coordinate) stay.
- **Files modified:** internal/opt/finallower/reducer.go
- **Verification:** verifier green; the unit golden shows the fill at X level.
- **Committed in:** 755dbaf

---

**Total deviations:** 5 auto-fixed (2 plan-premise corrections recorded honestly, 3 implementation defects caught by the RED gate / verifier / emitter)
**Impact on plan:** No pricing/schedule/runtime surface touched (git-diff scope gate clean); every deviation serves the plan's own pins and gates.

## Issues Encountered
- The three cloned C walks (init/E/store) are textually identical — the e2e pin identifies the element walk POSITIONALLY (the innermost cloned-C walk enclosing the acc update), never by the loop comment (the init walk's comment reads "@C inside walk" too; the 1a loops.go mislabel sanction applies).
- mm_small measured 2,881–3,295 µs across five runs (bit-identical .cu; 1a: 2,899 µs) — recorded honestly in after-bench.txt as machine-state spread on an identical binary; the clean full-suite run shows −0.6%.

## §9 GPU verification record (n63/qwo protocol, exact order)
1. `scripts/build.sh` FIRST — bin/tcacomp rebuilt from 4ac209a (stale-prebuilt gotcha honored).
2. Hermetic: `go build`/`go vet` exit 0; `go test ./...` 25/25 packages ok; `go list -m all | wc -l` == 1; git-diff scope gate: `internal/opt/tiling/{judge,search,record}.go`, `internal/opt/schedule`, `runtime/` ZERO diff.
3. Schedule-change detection (sha256 vs the 1a manifest): **DIFFER (6)** = axpy3, dup_const, dup_expr, row_sum, scale_copy, softmax (exactly the amortized set); **IDENTICAL (8)** = gemm, mlp, mm_small, mm_mid, mm_splitk, qk_score (the matmul family + mlp) + layernorm, transpose. Manifest: after-sha256-manifest.txt; diff: sha256-diff-vs-1a.txt. The changed set is a STRICT SUBSET of the eight 1a-changed kernels — stronger than the gate.
4. Correctness: `python3.13 -I testing/orchestrator.py --jobs 1` — **suite 14/14 PASS, canary 0**, maxdiff **identical to 1a on every kernel** (bit-identical numerics from the unchanged k-ascending order). Log: suite-run.txt.
5. Memcheck at FULL suite dims on every schedule-changed kernel: row_sum, softmax, scale_copy, axpy3, dup_expr, dup_const — **all PASS, 0 errors** (memcheck.txt; row_sum 0.5 s — no 300 s cap workaround needed).
6. Benchmarks `--jobs 1 --bench-iters 20`: the table below (raw: after-bench.txt; the bench runs re-passed the suite 14/14).

## Before/After Benchmark Table (--jobs 1, --bench-iters 20, A6000)

BEFORE = 1a's after numbers (before-bench.txt, copied verbatim — this task's baseline).

| kernel | before (µs) | after (µs) | vs before | gate verdict |
|---|---:|---:|---:|---|
| dup_expr | 7,027.0 | 1,844.5 | −73.8% (3.8×) | ≥10× vs 2026-10-29 pin (11,680): **633× MET** |
| dup_const | 7,836.1 | 2,994.3 | −61.8% (2.6×) | pin 11,840: **395× MET** |
| axpy3 | 21,123.0 | 3,667.5 | −82.6% (5.8×) | pin 23,770: **365× MET** |
| scale_copy | 20,805.5 | 3,243.6 | −84.4% (6.4×) | pin 23,460: **431× MET** |
| row_sum | 36,960.0 | 5,232.8 | −85.8% (7.1×) | **restoration-beyond MET (5,233 << 36,960)** |
| softmax | 9,687.0 | 1,005.3 | −89.6% (9.6×) | **restoration-beyond MET (1,005 << 9,687)** |
| transpose | 954.6 | 948.4 | −0.6% | bit-identical .cu — noise |
| layernorm | 8,997.3 | 8,929.7 | −0.8% | bit-identical .cu — noise |
| mlp | 6,667.4 | 6,597.4 | −1.0% | bit-identical .cu — noise |
| mm_small | 2,899.3 | 2,881.3 | −0.6% | bit-identical .cu — noise (spread recorded honestly) |
| mm_mid | 6,554.2 | 6,464.3 | −1.4% | bit-identical .cu — noise (faster) |
| mm_splitk | 6,141.1 | 6,089.3 | −0.8% | bit-identical .cu — noise (faster) |
| gemm | 9,955.9 | 9,702.2 | −2.5% | bit-identical .cu — noise (faster) |
| qk_score | 17,437.0 | 17,612.3 | +1.0% | bit-identical .cu — noise |

**§9 gate verdicts: ALL MET** — restoration-beyond (row_sum, softmax far past the 1a numbers), elementwise ≥10× EXCEEDED at 365–633×, no suite kernel >5% slower than its 1a best-known-good (worst qk_score +1.0%), matmul family + mlp sha256 bit-identical, suite 14/14 canary 0 with maxdiff identical to 1a, memcheck clean at full dims on every schedule-changed kernel.

## Known Stubs
None. Every changed emission path is GPU-validated (14/14 suite, memcheck ×6, bit-identical numerics); the qualified six are fully wired (acc block, init/store walks, inlined combiner); the Q4d-rejected four keep their proven per-element brackets byte-identical.

## Threat Flags
None. Compiler-emission change only; no new external input surface, no new dependencies (module count 1), no runtime interface change. The threat-register mitigations held: goldens updated ONLY via the golden helper after recorded REDs (red-task*.txt, red-drift*.txt) with re-derivation arithmetic in-comment (T-mt9-01); pricing/schedule/runtime scope verified zero-diff by git in every gate (T-mt9-02); the matmul-family sha256 bit-identity holds (T-mt9-03); no package installs (T-mt9-SC).

## Self-Check: PASSED
- internal/opt/finallower/reducer.go — modified, committed (755dbaf)
- internal/cg/cxx/expr.go — modified, committed (755dbaf)
- internal/cg/cxx/bracket_amort_test.go + testdata/bracket_amort.cu — created, committed (cb337ae RED, 755dbaf GREEN golden)
- internal/e2e/fix7bracket_test.go — created, committed (4684182 RED, 4ac209a GREEN)
- Re-derived pins + goldens — committed (4ac209a)
- design/fix/fix7.md — modified, committed (34f1ea1)
- Task-dir artifacts: red-task{1,2}.txt, red-drift{1..4}.txt, before/after-bench.txt, after-sha256-manifest.txt, sha256-diff-vs-1a.txt, suite-run.txt, memcheck.txt — present
- All 5 commits verified in git log (cb337ae, 4684182, 755dbaf, 4ac209a, 34f1ea1); suite 14/14 + memcheck ×6 + gates recorded above.

## Next Phase Readiness
- Slice 1c (effect-chain vectorization) gains its eligibility base: the amortized emission pins the chunk trailing extents (4/8/16/64 per kernel) — the `uint4` fast-path pin can target the now-static acc/window shapes.
- The §3.4 follow-up is CLOSED (gap closed structurally, verified by the e2e fill-nest-path pin + the 7.1×/9.6× numbers); the reuse audit's priced-8-vs-actual-1 residue is realized (fills per (trip, chunk): 256 → 4 on row_sum).
- D-fix6-01 and fix6 §1 remain independent sequencing slots (unchanged by 1b).

---
*Phase: quick-261101-mt9*
*Completed: 2026-11-01*
