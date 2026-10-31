---
phase: quick-261031-c8n
plan: 01
subsystem: codegen (internal/cg/cxx) + tiling merge
tags: [codegen, fix6, d-fix6-03, fill-coordinate, const-dim, gpu-validation, merge]
requires:
  - "v62 honest-FRAG pricing branch worktree-agent-a895cd8fde2fee08d (unmerged, blocked on D-fix6-03)"
  - "uai const-shape suite baseline .cu + sha256 manifests"
provides:
  - "D-fix6-03 fixed: const-covered dims reconcile in stmtFill by declared extent (canonical e<extent> identity), fail-closed on ambiguity"
  - "v62 pricing merged on dev: the six const-shaped suite kernels emit the all-SEQ collapse schedule"
  - "fully-static all-distinct probe verdict (emits end-to-end in the crafted path)"
affects:
  - "fix6 §1 per-shape specialization (fully-static now emits for distinct extents; repeated extents still D-fix6-01-blocked)"
  - "fix6 §3 proper (register-blocking codegen) — the collapse's 256 KB/thread Frag spill remains the performance prerequisite"
tech-stack:
  added: []
  patterns:
    - "identity-based covered-biv claiming (name for dynamic dims, declared extent for const dims) in the stmtFill walk"
    - "TDD RED pin with frontend-faithful DimRefs (DimRef{Extent} empty-Name) via crafted-record FRAG fills"
key-files:
  created:
    - "internal/cg/cxx/fillcoord_test.go"
    - "internal/cg/cxx/testdata/fillcoord_mixed.cu"
    - ".planning/quick/261031-c8n-fix-d-fix6-03-stmtfill-const-dim-coordin/before-bench.txt"
    - ".planning/quick/261031-c8n-fix-d-fix6-03-stmtfill-const-dim-coordin/after-bench.txt"
  modified:
    - "internal/cg/cxx/memory.go"
    - "design/fix/fix6.md"
decisions:
  - "const positions reconcile by the bivs' DECLARED extents via the canonical 'e<extent>' dim-name spelling (build.go dimKey = pkg.Builder's single declaration source) — the only extent surviving to codegen; zero matches = not covered (axis path), >1 = named fail-closed (D-fix6-01 class never guessed)"
  - "claiming is by biv identity for BOTH dynamic (name) and const (extent) positions, not positional cursors — required for row_sum/softmax's [R, Ku, C] where a const-covered dim precedes the dynamic-covered dim in tensor order"
metrics:
  duration: "45m (2026-10-31T00:58Z → 01:43Z, incl. GPU memcheck/bench wall time)"
  completed: "2026-10-31"
  tasks: 4
---

# Phase quick-261031-c8n Plan 01: D-fix6-03 stmtFill const-dim coordinate fix + v62 merge Summary

**One-liner:** D-fix6-03 fixed in the stmtFill covered-walk only (const positions claim covered bivs by declared extent via the canonical `e<extent>` dim identity, fail-closed on ambiguity), unblocking the v62 honest-FRAG pricing merge — the six const-shaped suite kernels now emit the all-SEQ collapse schedule with suite 14/14, memcheck clean, and the honest before/after table recorded.

**Worktree:** branch `worktree-agent-a1e815ec76e467047` at `/home/ps/Workspace/cuda/project/tcacomp/.claude/worktrees/agent-a1e815ec76e467047` (for the orchestrator's merge-back).

## Tasks Executed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | RED — mixed dynamic/static fill-coordinate unit pin | 3a1020c | internal/cg/cxx/fillcoord_test.go |
| 2 | GREEN — const-covered reconciliation in stmtFill (codegen walk only) | b9d8ac8 | internal/cg/cxx/memory.go, internal/cg/cxx/testdata/fillcoord_mixed.cu |
| 3 | Merge worktree-agent-a895cd8fde2fee08d (v62 pricing + RED pins) | 173577c | merge commit; before-bench.txt staged in the task dir (uncommitted docs artifact) |
| 4 | GPU verification protocol + honest benchmarks + fix6.md §4/§5 | (docs) | design/fix/fix6.md, after-bench.txt, this SUMMARY |

## RED/GREEN Evidence

- **RED (3a1020c):** `TestFillCoordReconstruction/mixed_dynamic/static` fails at Emit with the exact `cxx: fill 146 coordinate reconstruction ran out of axes`; the dynamic-only control and the fully-static probe subtests pass (probe pre-fix logs the same error). The fixture is frontend-faithful: dims R/C/e4, co-traversed x/z loads with DimRefs exactly as `build.go` builds for `[R, C, 4u]` (`{Name:"R"},{Name:"C"},{Extent:4}`), crafted record R=CTA / C=SEQ / e4=SEQ with both loads promoted FRAG (the C1b tier rule forces FRAG: R is a non-covered CTA INS iter) — each fill covers [C, e4] with one axis and two bivs.
- **GREEN (b9d8ac8):** the mixed pin passes with `TensorWindow(x_v, v9, C_it1.Index(), e4_it4.Index())` — coordinate list in TENSOR DIM ORDER (R axis, C biv, e4 biv), pinned by the targeted regex AND the byte golden `testdata/fillcoord_mixed.cu`. The dynamic-only control passes unweakened; the fully-static all-distinct `[512u,4u,1024u]` probe now **emits end-to-end** with tensor-dim-order coordinates.
- **Full hermetic suite green at both commits and post-merge:** 25/25 packages `ok`, `go build`/`go vet` clean, `go list -m all | wc -l` == 1. Every pre-existing golden byte-stable (the const path was always an error path, so no green pin could encode the old behavior).

## The Fix (internal/cg/cxx/memory.go, stmtFill covered branch only)

- Dynamic positions claim the covered biv whose dim param carries the ref's **name** (pre-fix behavior made identity-explicit, so row_sum/softmax's `[R, Ku, C]` — const-covered dim BEFORE the dynamic-covered dim in tensor order — cannot mis-assign the old positional cursor).
- Const positions (`DimRef{Extent>0}`) claim the UNCLAIMED covered biv whose **declared extent** equals the ref's. The declared extent comes from the dim param's canonical `e<extent>` spelling — `build.go` `dimKey` = `pkg.Builder`'s single static-dim declaration source — which is the ONLY extent surviving to codegen (RoleDim params carry no type; tile-lower's round bases are fresh dimvars whose original BIV bounds were RAUW'd away). Documented in the function header.
- Fail-closed, never guessed: zero extent matches → the position is NOT covered and rides its axis (a CTA-assigned const dim is a legal axis — never an error); >1 matches / duplicated covered dim name / double-claim / unclaimed covered biv (non-canonical static name) → NAMED errors. The repeated-extent D-fix6-01 class is never silently resolved (that defect stays open, out of scope).
- Untouched per plan: build.go, the IR, tile-lower/lower_cache, judge/search pricing, register blocking, D-fix6-01.

## Merge Record (Task 3)

- `git merge worktree-agent-a895cd8fde2fee08d --no-edit` — **clean, zero conflicts** (v62 touched `internal/opt/tiling/*` + e2e pin; this fix `internal/cg/cxx/*`).
- Load-bearing post-merge gates, all green unweakened:
  - `TestFix5T2OneTierCollapseEmission` (e2e) — the dynamic row_sum collapse EMITS through the one-tier SeqReducer via the real optimizer, now through the fixed codegen.
  - `TestR3FamilyBGuard` (tiling r3_test.go) — dynamic split-K family untouched.
  - v62 RED pins (`fragwindow_test.go`, `TestSearchRechoiceCollapse`, `TestR3SearchRechoiceConstK`, r3_test §3.1 part-(c)) — green.
- `before-bench.txt` staged in the task dir with the tree-identity note (measured at 2a42865; suite code identical to pre-merge dev since 56c84bb is docs-only) and the 2026-10-29 pinned baselines.

## GPU Verification (Task 4, n63/qwo protocol)

1. **Rebuild FIRST:** `scripts/build.sh` (the gitignored prebuilt bin/tcacomp is stale by design).
2. **Schedule-change detection (sha256 manifest diff vs the uai pinned baseline):**
   - SIX const-shaped kernels DIFFER — all flipped to the all-SEQ collapse (`R=CTA(T=256,LoC=16) C=SEQ(1,T256) Ku=SEQ(1,T256)`, ONE FRAG register-block promoted site `Frag<float,2,256,256>` per kernel, the co-load density-evicted — exactly the v62 search prediction): dup_expr, dup_const, axpy3, scale_copy, row_sum, softmax. Spot-checked fills: scale_copy `TensorWindow(z_v, v14, C_it1.Index(), e4_it4.Index())`; row_sum (const-MIDDLE) `TensorWindow(z_v, v12, e4_it4.Index(), C_it1.Index())` — both tensor-dim-order.
   - layernorm + transpose DIFFER — exactly as v62's recorded evidence (the two dynamic kernels the honest pricing flips to the collapse).
   - Matmul family BIT-IDENTICAL: gemm, mlp, mm_small, mm_mid, mm_splitk, qk_score (family-B dynamic-extent-unchanged constraint holds). NOTE: the plan's "eight matmul/mlp kernels bit-identical" expected 8; measured 6 bit-identical + 8 differing (6 const + layernorm/transpose) — the differing set matches the v62 summary's recorded flip set exactly; the matmul FAMILY (the constraint's actual subject) is fully bit-identical.
3. **Correctness:** suite **14/14 PASS**, canary 0, maxdiff ≤ 5.34e-05 — every kernel within its suite.toml tolerance.
4. **Memcheck (six schedule-changed kernels):**
   - dup_expr, dup_const, axpy3, scale_copy: **clean (0 errors)** via the orchestrator at full suite dims.
   - row_sum: **clean (0 errors)** at full suite dims, run manually against the cached fixture binary without the orchestrator's 300 s subprocess cap (orchestrator run timed out at 300 s — instrumentation inflation only, ~10× over its ~10 s bare runtime; the timeout, not an error, was the failure).
   - softmax: full-dims launch **OOMs under memcheck instrumentation** (`CUDA error ... out of memory` at the launch — the 256 KB/thread Frag local spill × sanitizer shadow exceeds the instrumented launch budget; bare-metal full-dims suite PASS, canary 0, maxdiff 1.9e-06). Memory-safety signal obtained at reduced dims (16×8×32, same kernel/grid structure): **0 errors**, canary 0, maxdiff 1.9e-06. Recorded as the measured consequence of the missing register-blocking codegen (fix6 §3 proper, out of scope) — resource exhaustion under instrumentation, NOT a legality violation.
5. **Honest before/after --jobs 1 --bench-iters 20:** see the table below (after-bench.txt; before = v62's before-bench.txt + 2026-10-29 pins).

## Before/After Benchmark Table (--jobs 1, --bench-iters 20, A6000)

BEFORE = v62's before-bench.txt (tree-identical suite code at 2a42865); pins = 2026-10-29. Full raw logs: `before-bench.txt` / `after-bench.txt` in this directory (after-bench.txt's own run also re-passed the suite 14/14).

| kernel | before (µs) | after (µs) | vs before | vs 10-29 pin |
|---|---:|---:|---:|---:|
| dup_expr | 1,190,420 | 787,319 | −33.9% | −32.6% |
| dup_const | 1,209,020 | 803,706 | −33.5% | −32.1% |
| axpy3 | 2,364,780 | 1,350,980 | −42.9% | −43.2% |
| scale_copy | 2,357,240 | 1,396,520 | −40.8% | −40.5% |
| row_sum | 459,049 | 1,979,530 | **+331.2%** | — |
| softmax | 59,329 | 263,084 | **+343.4%** | — |
| transpose | 336,315 | 180,375 | −46.4% | — |
| layernorm | 492,803 | 460,099 | −6.6% | — |
| mlp | 6,528.1 | 6,401.7 | −1.9% | — |
| mm_small | 2,882.8 | 2,928.8 | +1.6% | — |
| mm_mid | 6,910.9 | 6,986.8 | +1.1% | — |
| mm_splitk | 6,117.7 | 6,330.3 | +3.5% | — |
| gemm | 9,908.4 | 10,067.4 | +1.6% | — |
| qk_score | 17,412.9 | 18,098.7 | +3.9% | — |

## Gate Verdicts

- **No suite kernel >5% slower vs the 2026-10-29 pins: MET** — all four pinned elementwise kernels are 32–43% FASTER (none slower; the comparison script reports `kernels >5% slower vs 2026-10-29 pins: NONE`).
- **No suite kernel >5% slower vs the v62 BEFORE table: VIOLATED on row_sum (+331%) and softmax (+343%) — recorded honestly**, per the plan's explicit instruction, as the measured consequence of the missing register-blocking codegen (fix6 §3 proper, out of scope): the two `[R, Ku, C]` reduce kernels' collapse puts the per-thread `Frag<float,2,256,256>` window (256 KB/thread local spill class) under the reducer bracket, and the window streams from local memory. The matmul family's +1.1…+3.9% is inside the established noise band (v9k measured gemm +3.7% on an IDENTICAL binary = noise; here the family's .cu are bit-identical). transpose −46% and layernorm −6.6% reproduce v62's dynamic-flip numbers (169.7/455.6 ms).
- **fix6 §5.3 ≥10× elementwise gate (vs dup_expr 1.168 s / dup_const 1.184 s / axpy3 2.377 s / scale_copy 2.346 s): NOT MET** — best is axpy3 at 1.76× (2377/1351 ms), the four elementwise land at 1.47–1.76×. Exactly as the plan predicted, the collapse's per-thread Frag window spills to local memory until register-blocking codegen lands (fix6 §3 proper, explicitly out of scope); the ≥10× target needs that slice. The gate's verdict is reported honestly, not gated around — the 1.5–1.8× is real and consistent (the collapse eliminates the per-element WarpReducer bracket even while the window streams from local).

## Deviations from Plan

1. **[Rule 3 — fixture adjustment, sanctioned by the plan]** The single-load crafted fixture produced NO fill: the overlap admission filters a SINGLETON set (filter (a): no co-traversal, no reuse), and the crafted record's empty `Sites` map promotes nothing (the existing reduce3/splitk crafted fixtures emit no fills either — only full-pipeline fixtures do). Fix: added the second co-traversed load z (the minimal admitted shape, mirroring scale_copy's x/z) and stamped `rec.Sites` with promoted FRAG SiteDecisions per the `matmulRecord` idiom (Tie = the CacheDims-clamp product at the outermost level). The frontend DimRef shape (the pinned fact) is untouched.
2. **[Rule 1 — plan wording vs measured evidence]** The plan's "eight matmul/mlp kernels bit-identical" — measured: 6 bit-identical (the whole matmul family incl. mlp), 8 differing (6 const + layernorm/transpose). The differing set is exactly the flip set recorded in the v62 summary (layernorm/transpose flip LEGALLY and FASTER); the family-B constraint's actual subject (the dynamic matmul family) is fully bit-identical. Recorded, not "fixed".
3. **[Rule 3 — harness limit]** row_sum memcheck exceeded the orchestrator's hard 300 s subprocess cap (instrumentation inflation of its slower collapse runtime); rerun manually against the content-hash-cached fixture binary with no timeout — 0 errors at FULL suite dims.
4. **[Honest recording — resource limit under instrumentation]** softmax full-dims memcheck OOMs at launch (256 KB/thread Frag spill × sanitizer shadow); bare-metal full-dims PASS; memcheck clean at reduced dims (same kernel/grid structure). Recorded as the missing-register-blocking consequence, not a legality violation, and not a reason to weaken gates.
5. **Attribution** — merge commit message amended to append the plan-mandated `Co-Authored-By: Claude Opus 5.5` trailer (the `--no-edit` default message carried none).

## Known Stubs

None. The fully-static probe subtest intentionally passes either way by design (probe, not gate — per the plan); it logs a distinct PROBE line the suite output greps.

## Threat Flags

None — pure in-repo compiler codegen fix; no new network/auth/file surface (T-c8n-03 accepted in the plan's threat register).

## Self-Check: PASSED

- Files (7/7 FOUND): internal/cg/cxx/fillcoord_test.go, internal/cg/cxx/testdata/fillcoord_mixed.cu, internal/cg/cxx/memory.go, design/fix/fix6.md, this SUMMARY, before-bench.txt, after-bench.txt.
- Commits (6/6 FOUND): 3a1020c (RED), b9d8ac8 (GREEN), 173577c (merge, incl. v62 816f45d RED + 010ba91 GREEN in ancestry).
- Worktree branch: `worktree-agent-a1e815ec76e467047` at `/home/ps/Workspace/cuda/project/tcacomp/.claude/worktrees/agent-a1e815ec76e467047`, tip 93adab8 — clean tree except the orchestrator-owned docs artifacts (SUMMARY.md, before/after-bench.txt) left uncommitted per the task constraints.
