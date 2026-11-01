---
phase: quick-261101-oqg
plan: 01
subsystem: compiler-codegen
tags: [codegen, fix7, slice-1c, vectorization, alignas, chunk-policy, tdd, gpu-validation]

# Dependency graph
requires:
  - phase: quick-261101-mt9
    provides: "fix7 §4 slice 1b — bracket amortization; the BEFORE benchmark table, the 1b sha256 manifest this task's gates diff against, and the amortized-six chunk shapes (trailing 4/8/16/64) whose fill destinations are this slice's eligibility surface"
provides:
  - "the emission-side eligibility chain for the loader's EXISTING 128-bit row path: per-thread FRAG fill destinations with by-construction-clean rows (rank even, trailing extent*elemBytes % 16 == 0) declare alignas(16), and budget-forced SEQ halving steps snap to the largest vector-width multiple the TRUE extent and priced budget allow — the vec16 fast path is reachable in emitted code, never by luck"
  - "the snapHalfChunk chunk-policy bias (raw halving byte-identical where extent or budget does not qualify — a verified NO-OP on the whole suite)"
  - "the strip-proof e2e pin family: stripping exactly the alignas decl prefix byte-reproduces the 1b manifest on every 1c-affected kernel — any chunk-dim change fails the pin"
  - "the CPU eligibility-chain pin (emitted spelling + emitted dims ⇒ vec16_ok row-invariantly on the dst side; the plain-spelling counterfactual fails from row 0)"
affects: [fix7-slice-1d-smem-pipeline, fix6-specialization]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "decl-site alignment contract: the runtime's loader gates 16B endpoints per row; the emitter guarantees the dst side statically (alignas(16) on the variable — Frag::data_ carries only alignof(T)) where the object layout permits (RowMajorShape places data_ at 8*Rank bytes), and the runtime keeps the per-row check for the dynamic src side"
    - "prescan classification reuse: the fill-dst set is collected in the SAME prescan walk that collects fillSrc/bufTensor, with the IDENTICAL storage-class branch stmtFill emits from — the decl rule can never disagree with the emitted loader call"
    - "chunk-policy bias as a halving-step refinement: the snap target is computed against the OTHER dims' product (others*m <= budget) — the budget value and comparison stay siteWindowBytes's (v62) own"
    - "strip-proof emission pins: strings.ReplaceAll of the exact decl prefix, then sha256 vs the prior slice's manifest — a byte-level delta contract stronger than diff-by-eye"

key-files:
  created:
    - "internal/opt/tiling/fragvec_test.go (Pin A: TestFragChunkVecBias — snap + both controls)"
    - "internal/cg/cxx/fix7vec_test.go + testdata/fix7vec.cu (Pin B unit: the emission golden, BORN at GREEN)"
    - "internal/e2e/fix7vec_test.go (Pin B e2e: 14-kernel strip-proof sweep)"
    - ".planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/{red-task1,red-task2,before-bench,after-bench,after-sha256-manifest,sha256-diff-vs-1b,suite-run,memcheck}.txt"
  modified:
    - "internal/opt/tiling/lower_fragchunk.go (snapHalfChunk + vecWidthElems; the halving loop routes its SEQ step through the bias; header policy note)"
    - "internal/cg/cxx/walk.go (the alignas decl rule: fill-dst set + rank-even + clean-trailing)"
    - "internal/cg/cxx/emit.go (fragFill pre-pass classification in the existing fill walk)"
    - "runtime/tests/cpu_tests.cpp (Pin C: test_loader_vec_eligibility_emitted_shape + check_emitted_shape_rows<Trailing> — the ONLY runtime-dir change; loader.h/view.h zero-diff)"
    - "internal/cg/cxx/fragchunk_test.go + testdata/{fragchunk_stream,fragchunk_const_extent}.cu (golden regen, 1c re-derivation in-comment, 1a/1b assertions UNWEAKENED)"
    - "internal/cg/cxx/bracket_amort_test.go + testdata/bracket_amort.cu (golden regen, 1c re-derivation in-comment, 1b assertions UNWEAKENED)"
    - "design/fix/fix7.md (§5 STATUS: landed, the two DECISIONS, §9 record)"

key-decisions:
  - "Eligibility by construction, not literal-trailing-only: the plan's literal rule (fill-dst set + clean trailing) fired DEAD alignas bytes on the matmul family's rank-1 fill windows (Frag<float, 1u, 8u> — data_ at object offset 8 can never be 16B-aligned via the declaration) and broke the matmul bit-identity hard gate at GREEN; the rank-even condition (8*Rank % 16 == 0) excludes them BY CONSTRUCTION — the must_have's 'vec16_ok holds by construction, not by luck' is the binding rule (fail-closed refinement, recorded in the GREEN commit and fix7.md §5)"
  - "layernorm/transpose ARE 1c-affected (8 kernels, not 6): the plan's 'chunk trailing 1' premise misread the ELEMENT-dim chunk (1b's bracket-amortization domain) as the fill window's trailing extent; the actual 1b fill dst is Frag<float, 2u, 1u, 64u> (rank 2, 256 B rows) which qualifies identically to row_sum's window — the e2e pin re-derived to the STRICTLY STRONGER strip-proof form (never weakened: strip ⇒ byte-equals the 1b manifest)"
  - "the chunk bias is emission-side ONLY and provably dormant on the suite: snapHalfChunk requires a static TRUE extent multiple of 4 float elements AND a qualifying budget; the sha256 gate shows every 1c delta is alignas-decl-line-only (strip-diff EMPTY) — no chunk dim moved anywhere"
  - "Pin C honestly GREEN at RED (recorded in red-task2.txt's header): it pins emitted-spelling arithmetic (alignas(16) decl + trailing 4/8/16/64 ⇒ row-invariant dst eligibility), with the placement-new counterfactual proving the plain spelling fails from row 0"

requirements-completed: [QUICK-261101-OQG]

# Metrics

task-commits:
  - "d1deb20 test(quick-261101-oqg): fix7 §5 1c RED pins — vec-bias, alignas emission, eligibility chain"
  - "2418fd8 feat(quick-261101-oqg): fix7 §5 1c GREEN — vector-width chunk snap + alignas(16) fill-dst decls"
  - "(this commit) docs(quick-261101-oqg): fix7.md §5 status — effect-chain vectorization landed"

gates:
  hermetic: "go build/vet/test ALL GREEN (25 packages); go list -m all | wc -l == 1; scope gate: runtime/tca/loader.h + view.h, judge.go/search.go/record.go, internal/opt/schedule/ ZERO diff — mechanism, pricing, schedule untouched"
  gpu: "suite 14/14 PASS at --jobs 1 (canary 0 harness-verified per kernel; maxdiff IDENTICAL to 1b on all 14 — the vec16/scalar copy paths are byte-exact); memcheck clean at FULL dims on all eight 1c-affected kernels"
  sha256: "8 DIFFER (axpy3, dup_const, dup_expr, layernorm, row_sum, scale_copy, softmax, transpose) — each exactly 2 changed lines, ALL alignas-decl-only (strip-diff EMPTY); 6 IDENTICAL (gemm, mlp, mm_small, mm_mid, mm_splitk, qk_score — the matmul-family bit-identity gate)"
  bench: "honest --jobs 1 --bench-iters 20, 4 runs recorded (BEFORE = 1b's table verbatim): genuine vec16 gain softmax -16.5% (1005.3 -> 840 us, stable in EVERY run) and dup_const -7.7% on min; every other kernel within +-2% on min; the six bit-identical-binary controls inflate on avg on noisy runs (+4..+12%) proving that inflation is machine state (Chrome GPU process spikes — the 1b mm_small precedent), never this change; NO kernel >5% slower on min (the stable statistic)"
  pins: "RED->GREEN: TestFragChunkVecBias, TestFix7VecFillDecl, TestFix7VecEligibility all green unweakened; three drifted goldens regenerated with re-derivation arithmetic in-comment (never weakened); Pin C honestly green at RED"

deviations:
  - "The plan's Task-1 shape said the golden fix7vec.cu is created at RED; it cannot be — the unit pin fails at the alignas regex BEFORE golden.Assert, so -update never runs; the golden is BORN at GREEN (recorded in red-task2.txt)"
  - "The executor agent died mid-Task-1 to an upstream API quota error (429) after writing the three Go pins and confirming the e2e RED state; the orchestrator session completed Pins A/B evidence, Pin C, GREEN, and the §9 protocol directly in the worktree — same plan, same gates, recorded in the session transcript"
  - "8 affected kernels instead of the planned 6 (see key-decisions); the Task-3 sha256 STOP rule ('any chunk dim changed') never fired — all deltas are decl-line-only"
