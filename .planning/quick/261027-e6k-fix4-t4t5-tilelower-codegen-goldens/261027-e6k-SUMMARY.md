---
phase: quick-261027-e6k
plan: 01
subsystem: opt/tiling (tile-lower), cg/cxx (codegen), ir (runtime vocabulary), runtime/tca (maps)
tags: [fix4, tile-lower, lane-chunks, launch-derivation, factorization, goldens]
requires:
  - "261026-ky1 (fix4 T1+T2: decide + tiling-verify)"
  - "261026-tq8 (fix4 T1.1: one-currency objective)"
  - "261027-b1h (fix4 T3: runtime maps/iterators/launch)"
provides:
  - "fix4 §3.1/§3.2 multi-dim tile-lower templates with lane-chunk WARP walks"
  - "fix4 §3.5 Wave-1 defense (WARP-reduced dims fail closed under ReducedDimWarp=false)"
  - "fix4 §5.1 annotation-derived launch dimensions end-to-end (no driver totals)"
  - "lane_id first-class IR vocabulary (query + getattr + emitter)"
  - "zero-extent ThreadMapping admission (one-warp kernels)"
  - "restored tca_golden_matmul / tca_golden_reduce sm_86 compile gates"
affects:
  - "internal/e2e (Options compile-fix; T6 runs the gated numerics)"
  - "doc/optana/final_lower.md §6"
tech-stack:
  added: []
  patterns:
    - "launch dimensions derived from loop annotations (soc/dosoc/loc products), never driver-supplied"
    - "lane chunks: consecutive per-lane tile segments (non-increasing lengths, zero-trip out of range)"
    - "capability-flag fixtures: ReducedDimWarp=true admits the split-k shape for machinery tests while the default-flag policy is pinned separately"
key-files:
  created: []
  modified:
    - internal/opt/tiling/lower_rewrite.go
    - internal/opt/tiling/lower.go
    - internal/opt/tiling/lower_cache.go
    - internal/opt/tiling/record.go
    - internal/opt/tiling/lower_test.go
    - internal/opt/tiling/testdata/tile_lowered_matmul.ir
    - internal/ir/node_runtime.go
    - internal/ir/verify.go
    - internal/cg/cg.go
    - internal/cg/cxx/emit.go
    - internal/cg/cxx/launch.go
    - internal/cg/cxx/runtime.go
    - internal/cg/cxx/loops.go
    - internal/cg/cxx/expr.go
    - internal/cg/cxx/memory.go
    - internal/cg/cxx/statements.go
    - internal/cg/cxx/emit_test.go
    - internal/cg/cxx/testdata/matmul_store.cu
    - internal/cg/cxx/testdata/reduce_warp.cu
    - internal/opt/finallower/objects.go
    - internal/opt/finallower/finallower_test.go
    - internal/e2e/frontend_test.go
    - internal/e2e/reduce3_test.go
    - cmd/tcacomp/main.go
    - runtime/tca/mapping.h
    - runtime/tests/cpu_tests.cpp
    - runtime/tests/CMakeLists.txt
    - doc/optana/final_lower.md
decisions:
  - "Wave-1 defense implemented as the plan's forward IV-closure walk over WARP-assigned iters; the finallower reducer-machinery fixtures keep WARP-reduced shapes by running tile-lower under ReducedDimWarp=true (the capability flag's purpose), while the cxx goldens pin the default-legal schedule (reduced dim SEQ, SeqReducer, zero-extent ThreadMapping)"
  - "WARP between loc annotation = K_d in warp units (absolute); the CTA-coupled inside carries NO dosoc and loc = launch blockDim (the §3.2 convention codegen consumes)"
  - "launch blockDim = max over LNTs of (Pi warp K) x 32 computed in a tile-lower Run pre-pass (kernel-wide product deleted); codegen re-derives the same value from the AST annotations and cross-checks the reducer staging row against it"
metrics:
  duration: ~4h
  completed: 2026-10-27
  tasks: 3/3
  commits: 3
---

# Phase quick-261027-e6k Plan 01: fix4 T4+T5 (tile-lower + codegen + goldens) Summary

Multi-dim factorization tile-lower (lane-chunk WARP walks, flat coupled walks at the derived launch blockDim, WARP-only ThreadMap slots, Wave-1 split-k defense) plus annotation-derived launch emission (literal dim3s, 4-arg ValidateLaunch, explicit-ndim absolute-shape map ctors, zero-warp admission) with every golden re-pinned and the two stood-down sm_86 golden compile gates restored green.

## What Was Built

### Task 1 — T4 tile-lower templates (RED cb3ebb2 / GREEN 4b9d0af)

- **Multi-dim templates (§3.1)**: the R-SEAM fail-closed guard is deleted; CTA dims nest one between/inside pair each (stride LoC_d×T_d, own GridMap slot, coupled inside = flat-init walk at the LAUNCH blockDim), WARP dims one pair each (stride K_d×T_d, loc = K_d in warp units, own ThreadMap slot, inside = lane chunks). `TestLowerMultiDimTemplates` pins 2-CTA+2-WARP, 2-CTA-only, and 2-WARP-only schedules.
- **Lane chunks**: the WARP inside IV init is `lane × ceil(T_d/32)` over a new ZERO-OPERAND `lane_id` thread query (first-class vocabulary: `NewRuntimeThreadQueryLane`, verifier admission for query+getattr with a closed vocabulary everywhere else, typeinfer rides the kind, final-lower rewrites it to the zero-arg `lane_id` getattr, emitter renders `obj.lane_id()`); step 1; bound `umin((lane+1)·c, F1-tileMax)`.
- **Launch blockDim pre-pass**: max over forest roots of (Π warp K)×32, floor 32 — replaces the kernel-wide `blockDimOf` product in the templates and feeds the provenance `MaxThreadsPerCTA`. `TestLaunchBlockDimMaxOverLNTs` pins max-not-product (128 for K=2/K=4 LNTs).
- **§3.2 dimensionality**: CTA dims take NO ThreadMap slot (coupled inside carries NO dosoc; `checkLoopAnnotations` amended accordingly; provenance `CoupledDoSoC = -1`).
- **Mirror gate deleted** (§3.4 survive-list) with the rationale recorded at the old site; `hasCTADim`/`ctaThreadDim` removed.
- **Wave-1 defense (§3.5)**: any WARP-assigned iter whose IV users' forward data-dep closure reaches a live `KindReduce` fails closed naming fix4 3.5 / ReducedDimWarp; the capability flag stands it down (`TestLowerWarpReducedDimRejected`, incl. the SEQ positive control).
- `resolveBudget` documented as the unreachable §3.3 fail-closed backstop (kept as defense).
- Minimal cxx `loops.go` admission so the intermediate commit stayed green: the dosoc-less soc-2 loop is admitted only for non-const IV inits (the coupled flat walk), passing dosoc 0u + loc=blockDim (v0-ignored per iterator.h).

### Task 2 — T5 codegen + final-lower (3e3edf4)

- **cg.Options**: `MaxCTA`/`MaxThreadsPerCTA` retired; `MaxCTAPolicy` added (§4.3 policy constant, zero → 512).
- **deriveLaunch (§5.1)**: per LNT gridProd = Π loc over Soc==CTA loops, warpProd = Π loc over Soc==Thr loops WITH dosoc ≥ 0 (WARP betweens only); gridDim = max gridProd (fail-closed on loc < 1 or a CTA-less kernel — CuMAP), blockDim = max warpProd × 32 (floor 32). Consumed by the kernel header comment, `__launch_bounds__`, and the launcher's literal `dim3`s.
- **4-arg ValidateLaunch** with the policy constant; `cudaFuncSetAttribute` path unchanged.
- **Mapping emission**: explicit ndim + absolute shapes — `GridMapping grid_mapN(kinfo, 1u, 4u)`, `ThreadMapping thread_mapN(kinfo, 1u, 2u)`, zero-extent `(kinfo, 0u)` — matching `conformance_new_api.cu`. The `stagingCount` cross-check against the derived blockDim is restored (T5 debt paid).
- **Verifier**: mapping shapes are absolute extents ≥ 2; the dead D3 −1-remainder rule is struck.
- **Runtime**: zero-extent ThreadMapping admission (`sizeof...(E) >= 0`, `if constexpr` zero-pack guard, `Decomposition::build` n==0 skip); `test_zero_extent_thread_mapping` in cpu_tests (flat-rank identities at ndim 0; warp_count/total kinfo-derived).
- **Restored gates**: `tca_golden_matmul` / `tca_golden_reduce` back in runtime/tests/CMakeLists.txt (pre-T3 recipe), both compiling at sm_86.
- Two emission defects the restored compile gate surfaced (both fixed): a zero-coordinate view `Set` rendered `Set(v, )` (now `Set(v)` via `setArgs`), and the ZERO-BIV C1b fill fell out of `isFillStore`'s ≥3-operand check and emitted a raw (non-compiling) Set instead of fusing into `LoadTileThread`.
- `cmd/tcacomp` + e2e Options sites compile-fixed; `final_lower.md` §6 reworked (dims+shapes args; worked example rewritten to the multi-dim absolute-shape emission with derived dim3/ValidateLaunch; flat/lane zero-arg query rewrite documented).

## Task 3 — Acceptance Gate Results

| # | Gate | Result |
|---|------|--------|
| 1 | Hermetic `go build ./... && go vet ./... && go test ./...`, twice | GREEN ×2 (25 packages, both runs identical) |
| 2 | `go list -m all \| wc -l == 1` | 1 (stdlib-only) |
| 3 | Golden integrity: git clean over testdata after re-pins; golden tests green WITHOUT -update | CLEAN; tiling + cxx compare-mode green |
| 4 | Runtime cmake (fresh configure+build, TCACOMP_RUNTIME_TESTS=ON): tcacomp_cpu_tests, tca_conformance_new, tca_golden_matmul, tca_golden_reduce | 4/4 targets built at sm_86; cpu_tests 6100 checks / 0 failures (incl. the new ndim=0 case) |
| 5 | §9(2) signature agreement: TestFix33WorkedExample (decide) vs TestMatmulWorkedCheck (lower); golden self-consistency | Both PASS on the §3.3 vector (M=CTA/N=SEQ/K=WARP, both sites promoted, rhs 65536 B at the cap); matmul golden's LNT annotations (CTA LoC 4, warp K 2) match its own `dim3(4u)/dim3(64u)/__launch_bounds__(64)` literals |
| 6a | No 3-arg ValidateLaunch in testdata (`ValidateLaunch(dim3.., dim3.., <digit>` pattern) | 0 matches |
| 6b | Every testdata mapping ctor carries ndim | 4/4 (`grid_map0(kinfo, 1u, 100u)`, `thread_map1(kinfo, 0u)`, `grid_map0(kinfo, 1u, 4u)`, `thread_map1(kinfo, 1u, 2u)`) |
| 6c | `ortion` absent from runtime/ | 0 matches |
| 6d | No "lands in fix4 T4" guard text in internal/opt/tiling/ | 0 matches |
| 7 | Scope fence: `git diff --stat 3179bcc..HEAD` | 32 files, all within the plan's file set plus the documented deviations below; go.mod/go.sum untouched |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Plan/spec tension] Wave-1 defense vs. WARP-reduced test fixtures**
- **Found during:** Task 1 (the defense landed)
- **Issue:** The plan mandates the §3.5 defense AND the re-pinning of TestGoldenReduceWarp / the finallower reducer tests — but every one of those fixtures assigns WARP to the REDUCED dim, which the defense rejects by design. The plan's Task 2 zero-extent rationale ("e.g. K reduced -> CTA/SEQ under ReducedDimWarp=false") confirmed the goldens should be default-legal.
- **Fix:** Split treatment. The cxx golden fixtures (reduceWarp/reduce3/reduceT1) were re-targeted to Wave-1-legal schedules (reduced dim SEQ → SeqReducer brackets + the zero-extent `ThreadMapping(kinfo, 0u)` golden coverage); the WarpReducer ctor face stays pinned at the runtime level by conformance_new_api.cu. The finallower reducer-machinery fixtures (bracket placement, RAUW, staging, element loops — machinery that is flag-independent) keep their WARP-reduced shapes by running tile-lower through a new `runTileLowerScheduleSplitK` helper with `ReducedDimWarp=true` — the capability flag's exact purpose; the default-flag policy is pinned by `tiling.TestLowerWarpReducedDimRejected`.
- **Files:** internal/cg/cxx/emit_test.go, internal/opt/finallower/finallower_test.go
- **Commits:** 4b9d0af

**2. [Rule 1 - Stale pin] matmulRecord / TestMatmulWorkedCheck re-target to the §3.3 signature**
- **Found during:** Task 1 RED authoring
- **Issue:** The crafted record still carried the pre-fix4 vector (M=CTA/N=WARP/K=SEQ); Task 3's gate 5 requires decide/lower agreement on the §3.3 vector.
- **Fix:** matmulRecord now pins the §3.3 signature (M=CTA/4/256, N=SEQ/256, K=WARP/2/32; rhs 65536 B exactly at the MaxCache cap, lhs 4 B C1b per-lane chunk); TestLowerMatmulSiteSurgery / TestMatmulWorkedCheck / TestLowerDefaultShapeDecline / TestBudgetFallback re-derived (the lhs fill now carries zero biv round bases — the chunk origin rides the axes path).
- **Files:** internal/opt/tiling/lower_test.go
- **Commits:** cb3ebb2, 4b9d0af

**3. [Rule 1 - Verifier rule change] −1-portion fixtures in unrelated packages**
- **Found during:** Task 2 (the −1 remainder struck from the mapping shape rule)
- **Issue:** sideeffect/typeinfer/effect/ir test fixtures construct `ThreadMapping(ki, []int{4, -1})` as LEGAL shapes.
- **Fix:** Mechanical re-pin to absolute extents (`[4, 2]`; the map-remainder probe became map-extent1 pinning the ≥2 rule).
- **Files:** internal/analysis/sideeffect/sideeffect_test.go, internal/opt/typeinfer/typeinfer_test.go, internal/ir/effect_test.go, internal/ir/node_runtime_test.go
- **Commits:** 3e3edf4

**4. [Rule 1/2 - Compile gate] Two emission defects surfaced by the restored golden targets**
- **Found during:** Task 2 (cmake gate)
- **Issue:** (a) a view `Set` with an empty coordinate list rendered `Set(v, )`; (b) the zero-biv C1b fill (WARP-covered-only FRAG site) fell out of `isFillStore`'s ≥3-operand check and emitted a raw Set that cannot compile (latent since C1b, hidden by the T3 stand-down; exposed by the §3.3-signature matmul schedule).
- **Fix:** `setArgs` builds the argument list without a trailing separator; `isFillStore`/`stmtFill` admit the (value, buffer) form so the fill fuses into `LoadTileThread(thread_map, frag, TensorWindow(...))`.
- **Files:** internal/cg/cxx/statements.go, internal/cg/cxx/emit.go, internal/cg/cxx/memory.go
- **Commits:** 3e3edf4

## TDD Gate Compliance

Task 1 was `tdd="true"`: RED commit cb3ebb2 (failing pins, build green) → GREEN commit 4b9d0af (implementation + downstream re-pins; full suite green). Task 2's changes are verified by the existing golden/gate suite plus the restored compile gate (no new behavior surface requiring a fresh RED). Note: the RED commit intentionally has failing tests per the TDD flow (repo precedent 261026-tq8); build/vet were green at every commit and the suite is green from GREEN onward.

## Known Stubs

None. No placeholder values, no unwired data paths.

## Handoff (explicitly NOT done here)

- **T6**: numerical e2e + compute-sanitizer (memcheck/racecheck) on the A6000 over the regenerated goldens / frontend fixtures (TCACOMP_E2E-gated tests were compile-fixed only, never run).
- **T7**: pointer-stub headers on loop_tile_lower.md / rt.md / cg.md ("superseded by doc/fix/fix4.md §X").
- **Wave 2 (§3.5)**: split-k enablement behind `ReducedDimWarp=true` — the uniform-call Combine rule, ownership masks, and the dedicated split-k e2e; the flag currently only stands the defense down (the finallower fixtures exercise the machinery, not a numerically validated split-k lowering).
- **Latent gap (observed, out of scope)**: an OUTPUT-side WARP dim inside a reduce LNT (legal for the decide side) would give the R-A element loop a lane-dependent bound — the uniform-bracket law assumes uniform bounds. No current fixture or pipeline shape produces it; worth a decide-side guard or a ruling when reduce LNTs with WARP output dims first appear.

## Self-Check: PASSED

- Commits found: cb3ebb2 (RED), 4b9d0af (T4 GREEN), 3e3edf4 (T5) — all on worktree-agent-a94c497f06dd33221.
- Key artifacts exist: internal/opt/tiling/testdata/tile_lowered_matmul.ir, internal/cg/cxx/testdata/{matmul_store.cu, reduce_warp.cu}, runtime/tests/CMakeLists.txt (golden targets present).
- All Task 3 gates re-run green at summary time (see table).
