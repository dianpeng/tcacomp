---
phase: quick-261101-kel
plan: 01
subsystem: compiler-codegen
tags: [codegen, fix7, slice-1a, streaming-register-block, frag-chunking, tdd, gpu-validation]

# Dependency graph
requires:
  - phase: quick-261031-c8n
    provides: "D-fix6-03 fill coordinate reconstruction + the honest 2026-10-31 post-merge baseline numbers (before-bench.txt source)"
  - phase: quick-261030-v62
    provides: "the honest min(raw, MaxFragPerThread) FRAG pricing (siteWindowBytes — the ONE shared pricing source this plan reads)"
  - phase: quick-261030-uai
    provides: "the const-shaped suite baseline .cu + sha256 manifests (schedule-change detection base)"
provides:
  - "the streaming register-block lowering: a promoted FRAG site's buffer dims ARE the priced chunk (true const extents, budget-clamped via siteWindowBytes), between rounds step by the chunk, inside walks clamp to the chunk"
  - "planFragChunks: the pre-rewrite chunk plan (lower_fragchunk.go) — one chunk per covered iter, fail-closed on cross-site disagreement and on unfit WARP-only windows"
  - "three RED-pinned emission pins (fragchunk_stream.cu, fragchunk_const_extent.cu, fix7stream e2e) + the read-only tiling.PricedSites audit hook"
  - "the §3.4 priced-vs-actual reuse audit (placement gap confirmed and quantified — the slice 1b follow-up)"
  - "ALL §3.6 gates MET: restoration (row_sum 37.0 ms < 459 ms, softmax 9.7 ms < 59.3 ms), improvement (elementwise four 7.0/7.8/21.1/20.8 ms vs 787/804/1351/1397 ms), >=10x elementwise check at 166x/151x/113x/113x (recorded, not a 1a gate)"
affects: [fix7-slice-1b-bracket-amortization, fix7-fill-placement, fix7-slice-1c-vectorization, fix6-specialization]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "pre-rewrite plan phase in tile-lower (planFragChunks) — the chunk must exist before generateDimExprs builds between steps/inside clamps"
    - "one-chunk-per-covered-iter with cross-site agreement fail-closed (one between loop realizes one chunk per dim)"
    - "the buffer's InternTensor type as the ONE source of truth for the cxx Frag decl (walk.go prints type dims verbatim — no cxx change)"
    - "const-extent source precedence: BIV max operand const payload, then the canonical e<extent> dim-name spelling; NEVER ir.TypeOf (nil computed cell)"
    - "TDD RED pins with frontend-faithful fixtures + byte goldens; pin re-derivation comments carry the policy arithmetic (T-kel-01)"

key-files:
  created:
    - "internal/opt/tiling/lower_fragchunk.go"
    - "internal/opt/tiling/audit.go"
    - "internal/cg/cxx/fragchunk_test.go"
    - "internal/cg/cxx/testdata/fragchunk_stream.cu"
    - "internal/cg/cxx/testdata/fragchunk_const_extent.cu"
    - "internal/e2e/fix7stream_test.go"
    - ".planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/reuse-audit.txt"
    - ".planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/before-bench.txt"
    - ".planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/after-bench.txt"
    - ".planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/after-sha256-manifest.txt"
  modified:
    - "internal/opt/tiling/lower.go (chunk-plan call + fragChunk map)"
    - "internal/opt/tiling/lower_cache.go (chunked buffer dims, siteFill.chunks, doctrine comments)"
    - "internal/opt/tiling/lower_rewrite.go (chunk-stepped between IV + chunk-clamped tileMax for SEQ dims)"
    - "design/fix/fix7.md (§3 status note)"

key-decisions:
  - "Fixture promotes exactly ONE site per co-traversed pair (the F6-eviction shape): both-promoted trips the resolveBudget backstop over the shared covered tiles and would defeat the plan's [1,64] arithmetic"
  - "WARP-covered dims never halve (the lane-chunk walk width is fixed within a round) — a WARP-only window that cannot fit fails closed, per the §3.3 scope guard"
  - "The audit's priced side reads the shared pricing functions via the new read-only tiling.PricedSites hook (additive file), because package tiling internals cannot reach the e2e emission (import cycle) and judge/search/record are frozen"

patterns-established:
  - "chunk policy pinned in two places (code comment + pin comment) with the re-derivation arithmetic — never a silently-updated golden"
  - "RED evidence recorded per pin in the task dir (red-taskN.txt) before any GREEN commit"

requirements-completed: [QUICK-261101-KEL]

# Metrics
duration: 43min
completed: 2026-11-01
---

# Phase quick-261101-kel Plan 01: fix7 §3 slice 1a — streaming register-block lowering Summary

**Streaming FRAG lowering: the emitted code now realizes the priced block — chunk-sized declarations (256 B, was 256 KB/thread), chunk-stepped tile-between rounds, chunk-clamped inside walks — restoring row_sum/softmax 12x/27x past the pre-merge lines and putting the elementwise four at 113-166x the 2026-10-29 pins.**

## Performance

- **Duration:** 43 min
- **Started:** 2026-11-01T07:10:40Z
- **Completed:** 2026-11-01T07:53:17Z
- **Tasks:** 3
- **Files modified:** 14 (4 source/test created or modified outside the task dir + 10 task-dir artifacts)

## Accomplishments
- Pathology 1 (raw-window materialization) dead: every suite FRAG decl is now the priced chunk (e.g. `Frag<float, 2u, 16u, 4u>` / `[1u, 64u]` = 256 B); the 256 KB/thread local-memory decl is gone from every emitter; softmax memchecks clean at FULL dims (the c8n reduced-dims protocol is retired by a pass, as §3.6 demanded).
- Pathology 2 (nominal-T sizing) dead by construction: const dims chunk by true extent (e4 -> 4, not 256), read from the BIV max const / the canonical `e<extent>` spelling — never `ir.TypeOf`.
- The §3.6 gates are ALL MET: restoration row_sum 37.0 ms (< 459 ms) and softmax 9.7 ms (< 59.3 ms); improvement dup_expr 7.0 ms / dup_const 7.8 ms / axpy3 21.1 ms / scale_copy 20.8 ms (all strictly better than 787/804/1351/1397 ms); the >=10x elementwise check measures 166x/151x/113x/113x vs the 2026-10-29 pins (recorded honestly — not a 1a gate).
- The §3.4 priced-vs-actual audit is recorded (reuse-audit.txt): the fill-placement gap is real (priced reuse 8 not realized on the const-innermost four kernels) — attributed as the slice 1b / fill-placement follow-up, never patched.

## Task Commits

Each task was committed atomically (TDD: RED pin commit, then GREEN commit):

1. **Task 1: RED pin 1 (unit fill fixture)** - `70566ee` (test)
2. **Task 1: GREEN chunk machinery (dynamic budget clamp)** - `4262fa5` (feat)
3. **Task 2: RED pin 2 (const-extent chunking)** - `27331e3` (test)
4. **Task 2: GREEN const-extent source** - `4a96144` (feat)
5. **Task 3: RED-recorded pin 3 (e2e emission) + §3.4 audit** - `a350e8b` (test)

Task 3's §9 GPU verification (suite, memcheck x8, benches, sha256 manifest) and docs (this SUMMARY, reuse-audit.txt, before/after-bench.txt, fix7.md §3 note) are covered by the executor's docs commit; the orchestrator handles docs artifacts per the task brief.

## Files Created/Modified
- `internal/opt/tiling/lower_fragchunk.go` - planFragChunks: the pre-rewrite streaming-chunk plan (class re-derivation, siteWindowBytes budget read, outermost-SEQ halving policy, fail-closed guards, staticIterExtent const source)
- `internal/opt/tiling/lower.go` - lowerRun.fragChunk + the plan call before any rewrite
- `internal/opt/tiling/lower_rewrite.go` - SEQ-arm between step / tileMax clamp retargeted to the chunk (dimRewrite.chunk; annotations + GATE 3 untouched)
- `internal/opt/tiling/lower_cache.go` - buffer dims realize the chunk (the decl fix via the ONE source of truth); siteFill.chunks; 1a doctrine comments
- `internal/opt/tiling/audit.go` - read-only tiling.PricedSites audit hook (shared pricing functions only)
- `internal/cg/cxx/fragchunk_test.go` + `testdata/fragchunk_{stream,const_extent}.cu` - pins 1+2 (decl/structure/anchors; const true-extent) with re-derivation comments
- `internal/e2e/fix7stream_test.go` - pin 3 (scale_copy + row_sum through the real optimizer) + the suite-wide fill audit
- `design/fix/fix7.md` - §3 status note (landed, gates, audit verdict)
- task dir: red-task{1,2,3}.txt, reuse-audit.txt, before/after-bench.txt, after-sha256-manifest.txt, kel-manifest.sh

## Decisions Made
- One promoted site per co-traversed pair in the pins (the F6-eviction shape the real search stamps) — both-promoted trips the resolveBudget backstop and crushes the shared covered tiles; recorded in the pin comments and red-task1.txt.
- WARP dims are never halved by the chunk policy; an unfit WARP-only window fails closed (named error) rather than desyncing the lane-chunk walk — the §3.3 scope guard.
- The §3.4 priced side reads the shared pricing functions through a new read-only `tiling.PricedSites` hook (an additive file in package tiling) because the import cycle keeps tiling internals away from the e2e emission and judge/search/record are frozen.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] Read-only audit hook `tiling.PricedSites` added**
- **Found during:** Task 3 (§3.4 reuse audit)
- **Issue:** The plan requires deriving the priced reuse "via the tiling package's own APIs in the audit test", but the pricing functions (siteWindowBytes/siteReuseCount) are package-internal while the emission lives in package e2e — the import cycle makes a single test impossible, and re-deriving the math would violate the no-reimplementation rule.
- **Fix:** Added `internal/opt/tiling/audit.go`: an exported, read-only quote of the priced model over the shared functions (additive file; judge.go/search.go/record.go untouched — verified by git diff in every task's gate).
- **Files modified:** internal/opt/tiling/audit.go
- **Verification:** the audit's priced numbers (uniform raw=256 KB, window=256 B, reuse=8) match fix7 §2's recorded model exactly; full suite green.
- **Committed in:** a350e8b (Task 3 commit)

**2. [Rule 1 - Bug in fixture design, caught by RED] Pin fixture promotes exactly one site**
- **Found during:** Task 1 RED (first pin run emitted `Frag<float, 2u, 4u, 4u>` with T=4, not the plan's [1,64] arithmetic)
- **Issue:** The plan's example stamped BOTH co-traversed loads promoted; with per-site priced bytes 256 the kernel-wide sum (512) exceeds MaxFragPerThread, so the resolveBudget backstop halved the shared covered tiles to T=4 — the fixture could never exercise the budget clamp at T=64.
- **Fix:** Promote exactly one load (the F6-eviction shape the real search stamps — the real suite emits a single `tca_f0` per kernel, per fix7 §2); the other stays a GMEM control. Recorded in the pin comments and red-task1.txt.
- **Files modified:** internal/cg/cxx/fragchunk_test.go
- **Verification:** RED shows the raw `Frag<float, 2u, 64u, 64u>` with steps 64u; GREEN shows the chunked [1,64] with step 1u.
- **Committed in:** 70566ee / 4262fa5

**3. [Rule 3 - Blocking] kel-manifest.sh repo-root resolution**
- **Found during:** Task 3 (§9 step c manifest generation)
- **Issue:** The helper script's `cd` landed one directory short; `bin/tcacomp` was not found.
- **Fix:** Corrected the relative depth; manifest generated and diffed.
- **Committed in:** docs commit (task-dir artifact)

---

**Total deviations:** 3 auto-fixed (1 missing critical, 1 fixture bug caught by the RED gate, 1 blocking script fix)
**Impact on plan:** No pricing/schedule/runtime surface touched; the deviations serve the plan's own pins and audit deliverables. No scope creep.

## Issues Encountered
- The loops.go comment classification labels a chunk-1 between loop "@dim inside walk" (step-1u heuristic) — cosmetic-only, explicitly sanctioned by the plan ("leave loops.go alone"); visible in the new goldens.
- The sha256 diff vs the uai baseline conflates v62's collapse flip with 1a's chunking (same 8-kernel differing set either way); the load-bearing matmul-family bit-identity holds (6/6). The c8n-state manifest was not preserved, so 1a's isolated diff set cannot be computed — recorded honestly.

## §9 GPU verification record (n63/qwo protocol, exact order)
1. `scripts/build.sh` FIRST — bin/tcacomp rebuilt from a350e8b (stale-prebuilt gotcha honored).
2. Hermetic: `go build ./...` + `go vet ./...` exit 0; `go test ./...` 25/25 packages ok; `go list -m all | wc -l` == 1; git diff touches only the plan's files (judge/search/record/schedule/runtime untouched — `git diff --name-only -- internal/opt/tiling/{judge,search,record}.go` empty).
3. Schedule-change detection (sha256 manifest diff vs the uai pinned baseline): **DIFFER (8)** = dup_expr, dup_const, axpy3, scale_copy, row_sum, softmax, layernorm, transpose (exactly the expected schedule-changed set); **IDENTICAL (6)** = gemm, mlp, mm_small, mm_mid, mm_splitk, qk_score (the matmul family bit-identical). Manifest: after-sha256-manifest.txt.
4. Correctness: `python3.13 -I testing/orchestrator.py --jobs 1` — **suite 14/14 PASS, canary 0, maxdiff ≤ 5.34e-05** (all within suite.toml tolerances). row_sum's bare correctness run fell 42.6 s -> 0.2 s.
5. Memcheck at FULL suite dims on every schedule-changed kernel: dup_expr, dup_const, axpy3, scale_copy, layernorm, transpose, **softmax (0.6 s — the reduced-dims protocol is retired by a pass)**, **row_sum (1.5 s via the orchestrator — the c8n 300 s manual-cap workaround was not needed)** — all 0 errors.
6. Benchmarks `--jobs 1 --bench-iters 20` (the same run re-passed the suite 14/14): see the table below; raw log in after-bench.txt.

## Before/After Benchmark Table (--jobs 1, --bench-iters 20, A6000)

BEFORE = c8n's 2026-10-31 post-merge baseline (before-bench.txt, copied verbatim). Full raw logs in the task dir.

| kernel | before (µs) | after (µs) | vs before | gate verdict |
|---|---:|---:|---:|---|
| dup_expr | 787,319 | 7,027 | −99.1% (112x) | improvement MET (<< 787 ms); 10x check: 166x |
| dup_const | 803,706 | 7,836 | −99.0% (103x) | improvement MET (<< 804 ms); 10x check: 151x |
| axpy3 | 1,350,980 | 21,123 | −98.4% (64x) | improvement MET (<< 1351 ms); 10x check: 113x |
| scale_copy | 1,396,520 | 20,806 | −98.5% (67x) | improvement MET (<< 1397 ms); 10x check: 113x |
| row_sum | 1,979,530 | 36,960 | −98.1% (54x) | **restoration MET (37.0 ms < 459 ms)** |
| softmax | 263,084 | 9,687 | −96.3% (27x) | **restoration MET (9.7 ms < 59.3 ms)** |
| transpose | 180,375 | 955 | −99.5% (189x) | no-gate kernel; massively faster |
| layernorm | 460,099 | 8,997 | −98.0% (51x) | no-gate kernel; massively faster |
| mlp | 6,402 | 6,667 | +4.1% | bit-identical .cu — noise band, < 5% line |
| mm_small | 2,929 | 2,899 | −1.0% | bit-identical .cu — noise |
| mm_mid | 6,987 | 6,554 | −6.2% | bit-identical .cu — noise (faster) |
| mm_splitk | 6,330 | 6,141 | −3.0% | bit-identical .cu — noise (faster) |
| gemm | 10,067 | 9,956 | −1.1% | bit-identical .cu — noise (faster) |
| qk_score | 18,099 | 17,437 | −3.7% | bit-identical .cu — noise (faster) |

**§3.6 gate verdicts: ALL MET** — restoration (row_sum, softmax), improvement (elementwise four strictly better), global (no kernel >5% slower; matmul family bit-identical). The >=10x elementwise gate (not a 1a gate per §10) is recorded at 113-166x — already exceeded.

## Known Stubs
None. Every changed emission path is wired to the real pricing source and validated by the GPU suite + memcheck; the only recorded residual is the §3.4 placement gap (a measured follow-up, not a stub).

## Threat Flags
None. Compiler-only change; no new external input surface, no new dependencies (module count 1), no runtime interface change. The T-kel-01/02/03 threat-register mitigations held: goldens updated only via the golden helper with re-derivation comments after recorded REDs; pricing files verified untouched by git diff in every task; no new modules.

## Self-Check: PASSED
- internal/opt/tiling/lower_fragchunk.go — created, committed (4262fa5, 4a96144)
- internal/opt/tiling/audit.go — created, committed (a350e8b)
- internal/cg/cxx/fragchunk_test.go + 2 goldens — created, committed (70566ee, 4262fa5, 27331e3, 4a96144)
- internal/e2e/fix7stream_test.go — created, committed (a350e8b)
- design/fix/fix7.md — modified, committed (docs commit)
- reuse-audit.txt / before-bench.txt / after-bench.txt / after-sha256-manifest.txt / red-task{1,2,3}.txt — present in the task dir
- All 5 task commits verified in git log; suite 14/14 + memcheck x8 + gates recorded above.

## Next Phase Readiness
- Slice 1b (bracket amortization) and the fill-placement task have their measured baseline: the §3.4 audit quantifies the remaining 8x placement gap (priced reuse 8 vs actual consumption 1-4 on the const-innermost kernels); the gates are already MET, so 1b is pure upside on row_sum/softmax beyond restoration.
- Slice 1c (vectorization) gains its eligibility check: chunk trailing extents are 4/8/64 depending on the kernel — the `uint4` fast-path pin will need per-kernel shapes.
- D-fix6-01 and fix6 §1 remain independent sequencing slots (unchanged by 1a).

---
*Phase: quick-261101-kel*
*Completed: 2026-11-01*
