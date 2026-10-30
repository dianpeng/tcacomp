---
phase: quick-261029-v9k
plan: 01
subsystem: opt/tiling
tags: [tiling, scheduling, r3-warp-axis-freedom, extent-aware-placement, gpu-validation]

requires:
  - phase: quick-261029-qwo
    provides: fix5 T1 (R1 adaptive LoC + R2 one-tier ladder verification + R4 tier-efficiency) and the gate-3 failure diagnosis this task closes
provides:
  - warpDimOf in internal/opt/tiling/search.go — the R3 extent-aware per-LNT WARP-dim ranking (L1 effective extent DESC / L2 stride-1 set membership / L3 non-reduced before reduced / L4 innermost window — a TOTAL comparator), computed once per LNT and passed into digitListOf
  - the per-LNT WARP restriction — digitListOf emits the WARP preset section AND the R1 adaptiveSection(SoCWARP) ONLY for the winner dim; the R-REDUCED prune stands on top; decide-side feasible-area shaping exactly like R-REDUCED (judge/constraints/verify/record/occupancy zero diff)
  - the unit-proof family: ranking pins, restriction pins, the const-K scale_copy re-choice (warps-on-C K=SEQ collapse, ceremony unrepresentable), the family-B guard (dynamic matmul winner unchanged — split-K preserved by construction), the lower-flow guard pins (WARP-on-non-reduced lane-chunk/C1b path flows on unchanged code), the fePin e2e split-k signature surviving untouched
  - the GPU discovery of TWO latent defects in the never-exercised one-tier/WARP-on-non-reduced lowering+emission path (recorded with mechanism diagnosis, per the stop-rule)
affects: [tiling-search, codegen-schedule-selection, fix5-r3, fix6-work-shape-model]

tech-stack:
  added: []
  patterns:
    - "warpDimOf reads only verified IR facts (constExtentOf, SiteFacts.AxesIters innermost position, the reduced map) — never indexes an empty window, nil-safe, fail-closed (T-v9k-01)"
    - "R3 is selection-only exactly like R-REDUCED and the n63 occupancy factor: the ranking narrows the feasible area BEFORE enumeration; legality re-checks (tiling-verify, unchanged) gate every emitted record — a malformed ranking can pick a slow schedule, never an illegal one (T-v9k-02)"
    - "re-derived-pin discipline (n63/qwo precedent): three const-extent matmul pins legitimately moved (their WARP dim is no longer the R3 winner) — re-derived from the ACTUAL post-R3 search output with full derivations in the pin comments; the dynamic-extent family-B winner and the fePin survived unchanged"

key-files:
  created:
    - internal/opt/tiling/r3_test.go
  modified:
    - internal/opt/tiling/search.go
    - internal/opt/tiling/adaptive_test.go
    - internal/opt/tiling/tiling_test.go
    - internal/opt/tiling/lower_test.go
    - internal/opt/tiling/testdata/tile_lowered_matmul.ir
    - .planning/quick/261029-v9k-implement-fix5-t2-r3-warp-lane-axis-free/before-bench.txt
    - .planning/quick/261029-v9k-implement-fix5-t2-r3-warp-lane-axis-free/before-cu-hashes.txt

key-decisions:
  - "R3 implemented exactly per fix5 §1 R3 (amended session-2 ruling): L1 effective extent, L2 stride-1 set (innermost AxesIters position per CS site, within-set innermost window), L3 reduced last-resort, L4 total fallback — the plan's worked outcomes all reproduce at unit level (matmul dynamic → k, frontend const {64,128,64} → N, const-K scale_copy → C)"
  - "digitListOf gained the warpDim winner parameter (plan-sanctioned signature change); all 8 pre-existing call sites updated — the plan said 4, the tree carries 8 (6 in tiling_test.go); per-dim pins pass the queried dim as its own winner, enumerateBest mirrors the pass via the shared warpDimOf"
  - "Three pre-existing pins moved under R3 (TestFix33WorkedExample, TestDecideLowerComposition, the tile_lowered_matmul print golden) — all const-extent matmul fixtures whose old winner placed WARP on a dim that is no longer the R3 winner; re-derived from actual output, never weakened; the family-B guard and fePin survived unchanged as the plan required"
  - "Lower-flow guard pins passed on UNCHANGED tile-lower code (fix5 §3 'already flows' confirmed at IR level) — but the GPU suite then exposed two latent defects in the never-numerically-exercised one-tier emission path (see Gate Verdicts); per the plan's stop-rule both are recorded with mechanism diagnosis, NOT fixed (fixing needs cg emission semantics — outside T2's declared surfaces)"
  - "Gate-3 honest record: the elementwise four stay bit-identical because their [R,C,K] layout makes the REDUCED k the loads' innermost (coalescing) axis — L2 selects k; the collapse needs a compile-time K extent (fix6 D-fix6-01 const-shape specialization). R3 instead re-chose the two [R,K,C] kernels (row_sum, softmax — reduce the MIDDLE dim, coalescing axis = the long non-reduced C), which is the spec working exactly as designed and which exposed the latent defects"

requirements-completed: [QUICK-261029-V9K]

duration: 95min
completed: 2026-10-29
status: complete-with-s31-ruling
---

# Quick 261029-v9k: fix5 T2 — R3 Warp/Lane Axis Freedom with Extent-Aware Placement

**R3 landed exactly as spec'd (warpDimOf four-level total ranking, computed once per LNT, gating digitListOf's WARP section; verify.go/judge.go/constraints.go/runtime//finallower zero diff; full hermetic suite green 25/25) — and the GPU verification produced the task's real finding: R3 re-chose the two `[R,K,C]` suite kernels (row_sum, softmax) onto the collapse schedule precisely as designed, exposing TWO latent defects in the never-numerically-exercised one-tier emission path (store gate + FRAG coordinate), so the suite is 12/14 with canary violations on exactly the two schedule-changed kernels. The six family-B split-K kernels are BIT-IDENTICAL (hard gate satisfied), the fePin survived unchanged, and the four elementwise kernels are also bit-identical — their `[R,C,K]` layout keeps the reduced k as the loads' coalescing axis, so the ≥10× gate-3 requirement fails for the same root cause as qwo (dynamic K), now with the complete mechanism pinned.**

## What Was Built

- **`warpDimOf`** (`internal/opt/tiling/search.go`): the R3 ranking, computed ONCE per
  LNT in `searchScheduleStats` BEFORE the per-dim digit loop:
  - **L1** effective extent DESC — `constExtentOf` const value, else
    `Params.NominalExtent` (the same discipline as R1's `adaptiveLoC` and R4's
    `reducedTierEff`).
  - **L2** stride-1 set membership — `stride1Set` = union over the LNT's CS sites of
    the iters at each site's INNERMOST subscript position (last non-empty
    `AxesIters` entry); in-set dims rank above out-of-set; within the set the
    innermost WINDOW position wins.
  - **L3** non-reduced before reduced (fires only on L1/L2 residuals).
  - **L4** innermost window position (total — window positions are unique; the
    comparator never panics, never ties; nil/empty window → nil, fail-safe).
- **`digitListOf` R3 gate**: the WARP preset section AND the R1
  `adaptiveSection(SoCWARP)` emit ONLY when `iter == warpDim`; the R-REDUCED prune
  stands on top (reduced winner + `ReducedDimWarp=false` → no WARP section at all —
  narrower than the pre-R3 flag=false semantics). Decide-side feasible-area shaping
  exactly like R-REDUCED: enumeration order, judge, eviction, determinism contract,
  tiling-verify all untouched (zero diff: `runtime/`, `finallower/`, `verify.go`,
  `judge.go`, `constraints.go`).
- **Pins** (`r3_test.go`, new): the four-level ranking pins (crafted windows +
  hand-built SiteFacts + the REAL dynamic-matmul SiteFacts), the restriction pins,
  the const-K scale_copy re-choice (winner `R=CTA(16,T256) C=WARP(2,T32) K=SEQ(T32)`
  — the qwo collapse vector, now guaranteed BY CONSTRUCTION with the ceremony
  vector unrepresentable), and the **family-B guard** (dynamic matmul winner
  `i=CTA(128,T32) j=SEQ(T256) k=WARP(32,T256)` with lhs FRAG 32 B reuse 256 and rhs
  evicted — the probe-recorded T2-parent vector, surviving GREEN unchanged).
- **Lower-flow guard pins** (`lower_test.go`, `TestLowerR3WarpNonReducedFlow`): a
  crafted scale_copy-class record (WARP on covered non-reduced C, reduced K SEQ)
  lowers through tile-lower with the WARP template on C (between stride K_d×T_d,
  lane-chunk inside walk, F1-clamped umin bound), the C1b chunk-LOCAL FRAG access
  (iq6 form), the kzv chunk-origin fill window, the SMEM rank-in-index access, the
  per-thread FRAG + cooperative SMEM fills, the single-owner GMEM store, and
  verifier + typeinfer health — ALL PASSING ON UNCHANGED tile-lower code (fix5 §3
  "already flows" confirmed at IR level; no lower fix needed, no deviation).
- **Re-pins** (never weaken): `TestFix33WorkedExample` (M=4096 wins L1 → new argmax
  `M=CTA(128,T32) N=SEQ K=SEQ`, no WARP dim, lhs FRAG 256 B at the F6 cap, rhs
  declined), `TestDecideLowerComposition` (N=2048 wins L1 → `M=CTA(4,T256)
  N=WARP(2,T256) K=SEQ(T32)`, lhs FRAG 128 B reuse 8, rhs SMEM 65536 B), and the
  `tile_lowered_matmul.ir` golden (regenerated via the standard `-update` flow; the
  diff is exactly the R3 schedule change). The **frontend e2e `fePinScheduleSignature`
  survived UNCHANGED** (N wins L1 on the const {64,128,64} shape) — as the plan
  required.

## GPU Verification (A6000, testing/orchestrator.py, --bench-iters 20)

### .cu-diff attribution (the load-bearing result)

| kernel | .cu vs fdf8ef3 | attribution |
|---|---|---|
| mm_small, mm_mid, mm_splitk, gemm, qk_score, mlp (family-B six) | **BIT-IDENTICAL** (sha256 equal) | R3 preserves split-K by construction — the hard EXACT-preservation gate **PASSES** |
| dup_expr, dup_const, axpy3, scale_copy (elementwise four) | **BIT-IDENTICAL** | `[R,C,K]` layout: the reduced k IS the loads' innermost (coalescing) axis → L2 selects k → ceremony vector preserved |
| **row_sum, softmax** | **CHANGED** | `[R,K,C]` layout reducing the MIDDLE dim: the coalescing axis is the long non-reduced C → R3 lands lanes on C with K=SEQ — the spec working EXACTLY as designed |
| transpose, layernorm, mlp-covered others | bit-identical | unchanged |

The plan's expected changed-set was the elementwise four; the actual changed-set is
{row_sum, softmax} — same mechanism, different layout. R3's ranking is layout-faithful:
it frees whichever axis the loads actually coalesce on.

### Correctness (gate 2): **FAILED — 12/14**

12/14 PASS (canary 0, maxdiff in tolerance, all family-B + elementwise + transpose +
layernorm + gemm + qk_score + mlp). The two schedule-changed kernels FAIL:

- `row_sum`: **canary 516096 (unwritten outputs)**; memcheck: **Invalid `__local__`
  read of size 4 bytes (109 errors)**.
- `softmax`: **canary 32256 (unwritten outputs)**; memcheck: **Invalid `__local__`
  read of size 4 bytes (55 errors)**.

Both were memchecked as the plan mandates for every schedule-changed kernel (the
elementwise four ride bit-identical .cu on the prior memcheck runs). The BEFORE
schedules were `R=CTA(T=256,LoC=16) C=sequential(T=32) K=thread/warp(T=32,LoC=2)`
with a WarpReducer bracket; the AFTER schedules are `R=CTA(T=256,LoC=16)
C=thread/warp(T=32,LoC=2) K=sequential(T=32)` with a one-tier SeqReducer — the
emitted collapse kernel is (quoted from the emitted row_sum.cu):

```
reducer0.Init(); ... K walks, Reduce(v27); ... auto v28 = reducer0.Combine();
// single-owner store — the element's owner thread writes (R2/R3)
if (thread_map1.is_first_in_cta()) {
  out_v.Set(v28, (v11 + R_it2.Index()), (v15 + C_it3.Index()));
}
```

### Gate-2 mechanism diagnosis (recorded per the stop-rule; no pin weakened, no out-of-scope fix)

**Defect A — the single-owner store gate is wrong for a one-tier ladder.** The cg
prescan (`internal/cg/cxx/emit.go` ~L407-427) marks EVERY store whose value closure
contains ANY `Combine()` call as `gated`, and `stmtMemStore`
(`internal/cg/cxx/statements.go` L159-169) wraps those in
`is_first_in_cta()`. That is correct for the MULTI-tier WarpReducer (all blockDim
threads compute partial sums of the SAME element; one elected writer stores the
combined result). For the ONE-tier SeqReducer each thread's `Combine()` is an
identity over its OWN accumulator, and the (R_it2, C_it3) walks partition DISTINCT
(r, c) elements across threads — so the gate writes only the first thread's elements
and leaves every other lane's unwritten: exactly the observed canary (row_sum
canary 516096 ≈ all outputs minus the first thread's; grid 16 CTAs × one writer).

**Defect B — the FRAG chunk coordinate reads out of bounds.** The per-thread
`Frag.Get(K_it5.Index(), (C_it3.Index() - v10))` indexes past the fragment's extent
(Invalid `__local__ read, iq6 class) on the real kernels' `[K-round, C-chunk]`
window layout — the chunk-local origin `v10` and the window's chunk-axis extent
don't line up for this configuration (the unit pins lock the chunk-local IR SHAPE,
which is present; the numerical extent arithmetic on the real window layout is the
latent defect).

Both defects are PRE-EXISTING in the one-tier/WARP-on-non-reduced path that NO suite
kernel ever exercised numerically: qwo's T1 changed zero .cu files (its one-tier
path was verified only at IR-emission level by `TestReducerOneTierSeqLadder`), so
R3's re-choice of row_sum/softmax is the first numerical exercise — precisely the
configuration the plan called "the configuration the suite never exercised" and
mandated memcheck for. **The fix touches cg emission semantics (one-tier-aware
gating in the prescan) and the FRAG coordinate arithmetic — outside T2's declared
surfaces (the plan scopes a mechanical fix to tile-lower's lane-chunk/C1b path,
which is NOT where these defects live) — so per the plan's stop-rule they are
recorded here, not fixed.** Recommended follow-up: a fix plan for
`cg/cxx` one-tier gating + the FRAG chunk extent (D-fix5-T2-01/02 style), after
which row_sum/softmax re-run; alternatively the author may revert R3 pending that
fix (the unit pins and family-B evidence survive either way).

### Benchmark (gate 3, avg µs/iter, --bench-iters 20)

| kernel | BEFORE (fdf8ef3) | AFTER | Δ | .cu |
|---|---|---|---|---|
| dup_expr | 1201110.0 | 1198650.0 | −0.2% (noise) | bit-identical |
| dup_const | 1214390.0 | 1215490.0 | +0.1% (noise) | bit-identical |
| axpy3 | 2387140.0 | 2397300.0 | +0.4% (noise) | bit-identical |
| scale_copy | 2387250.0 | 2372740.0 | −0.6% (noise) | bit-identical |
| mlp | 6424.4 | 6280.6 | −2.2% (noise) | bit-identical |
| **row_sum** | 470696.0 | **FAIL** (canary) | — | CHANGED |
| **softmax** | 60090.9 | **FAIL** (canary) | — | CHANGED |
| mm_small | 2883.2 | 2987.7 | +3.6% (identical binary, cache hit — machine noise; under the 5% gate) | bit-identical |
| mm_mid | 6471.7 | 6491.0 | +0.3% (noise) | bit-identical |
| mm_splitk | 6104.9 | 6115.8 | +0.2% (noise) | bit-identical |
| transpose | 335771.0 | 337977.0 | +0.7% (noise) | bit-identical |
| gemm | 10172.0 | 10110.7 | −0.6% (noise) | bit-identical |
| qk_score | 17473.6 | 17552.5 | +0.5% (noise) | bit-identical |
| layernorm | 494086.0 | 493313.0 | −0.2% (noise) | bit-identical |

**GATE-3 HARD GATE — FAILED** (as in qwo): the elementwise four are bit-identical
(±0.6% noise), NOT ≥10×. The mechanism is now completely pinned and is a LAYOUT
consequence: scale_copy/axpy3/dup_expr/dup_const are `[R,C,K]` kernels reducing the
LAST dim — their loads' innermost (coalescing) subscript is the reduced k itself, so
R3's L2 selects k and the ceremony vector is preserved BY CONSTRUCTION. The collapse
requires the loads' coalescing axis to differ from the reduced dim (as in the
`[R,K,C]` row_sum/softmax — where it fired) or a compile-time K extent (const-shape
specialization, fix6 D-fix6-01, out of scope). The no-kernel->5%-slower clause
PASSES for the 12 passing kernels (worst +3.6% on a byte-identical binary — cache
hit, same .cu bytes, environment noise).

## §3.1 RULING ARC (author decision, 2026-10-30 — the failure's true root cause)

The gate-2 failure below was re-investigated on the orchestrator after the
executor's in-branch fixes (722a652 RED / fd72179 GREEN) proved .cu-NEUTRAL
for row_sum (cache-hash identical pre/post fix — neither "defect" was the
canary cause; both remain real, unit-locked one-tier-path fixes). Decoding
the emission against the runtime iterator semantics (ThrIterator step =
blockDim-coupled) and the canary arithmetic:

- row_sum: 8,192 of 524,288 outputs written — EXACTLY 1/64; softmax:
  512 of 32,768 — EXACTLY 1/64.
- The lowering nests R=CTA's coupled-inside walk (ALL 64 threads stride R)
  INSIDE C=WARP's round loop, with the C lane-chunk innermost. Per
  (CTA-round × C-round) the 256×64 region receives only 256 "diagonal"
  stores — the two inside-walk mechanisms consume the SAME threads on
  INDEPENDENT axes; 63/64 of the cross product is never computed.
- This is the iq6 §3.1 CTA×WARP(surviving) hole verbatim ("every (m,n)
  pair carries 16/1024 k values" — also 1/64), fenced with t.Skip on the
  matmul goldens since 261027-iq6. fix5 §3's premise ("the WARP-on-
  non-reduced-dim layout already flows through the existing thread-map
  coordinate machinery — only the CHOICE of dim is new") is GPU-DISPROVEN:
  R3 did not hit an emission bug; it made the search able to SELECT the
  class tile-lower has never been able to lower, and it immediately did
  (row_sum/softmax are the only [R,K,C] middle-reduce kernels — exactly
  where a non-reduced dim wins the extent ranking).

AUTHOR RULING (2026-10-30, iq6 candidate-1): prune surviving-WARP — the
WARP winner restricted to eliminated axes (search side) + a fail-closed
tile-lower mirror. Landed as 8085b84 (pins) + ff929a5 (implementation):

- `reducedAxesOf` (search.go): the ELIMINATED set — reduce-value IVs never
  in any store subscript (distinct from `reducedIters`, whose
  reducer-consuming semantic also contains SURVIVING axes; in
  [R,K,C]->[R,C] every IV feeds the reduce value but only K is eliminated).
- `warpDimOf` §3.1 prune: candidates restricted to eliminated axes; none
  in window -> NO WARP digit; scoped via hasReduce to reduce-carrying
  kernels (every real v0 kernel — map|reduce[|map]|store), crafted
  no-reduce template fixtures keep the full-window R3 ranking.
- `rejectWarpSurvivingDims` (lower.go): the fail-closed mirror, before any
  mutation, unconditional across both ReducedDimWarp modes.
- Consequence (honest): on the const-K fixture the collapse is DOUBLY
  unreachable (surviving-WARP pruned AND all-SEQ FRAG-declined 4096 B >
  256 B), so the R4-penalized ceremony vector is the legitimate winner —
  the elementwise unlock needs the §3.1 cross-product lowering PLUS
  fix6's work-shape/const-shape model.

## Gate Verdicts (fix5 §5) — FINAL (post-ruling)

1. **go build/vet/test exit 0** — PASS (25/25 packages, run twice).
2. **GPU suite 14/14, memcheck clean on every schedule-changed kernel** —
   PASS after the ruling: full suite 14/14 (canary 0 everywhere, maxdiff
   within every tolerance); memcheck CLEAN on row_sum and softmax (the
   recovered kernels; both re-verified under `--sanitizer memcheck`).
3. **Benchmark gate** — the elementwise ≥10× clause remains NOT MET and
   is now fully explained: ALL 14 .cu files are BIT-IDENTICAL to the
   pre-T2 state (sha256-equal to before-cu-hashes.txt — the prune restores
   family-A and preserves family-B by construction), so no timing change
   is possible; the unlock needs §3.1 + fix6. The no-regression clause
   PASSES (worst gemm +3.7% on a bit-identical binary — GPU run noise,
   inside the 5% gate; the rest ±2%).
4. **Correctness invariants** — PASS: tiling-verify green (suite implies
   + hermetic verify tests); `runtime/` zero diff; D1/D2 untouched;
   multi-tier WarpReducer byte-stable; the six family-B .cu sha256-equal
   (the hard EXACT gate); row_sum/softmax sha256-equal to their pre-T2
   hashes (72db430… / df7b8a4a… — full restoration).

## Spec-Recorded Consequences (fix5 §1 R3 / §2 — not defects)

- At most ONE WARP dim per LNT henceforth; MAPM/MALoC accounting unchanged. This
  NARROWS the kzv surviving-WARP unruled case-b reach (side benefit, one line).
- Explicit `ReducedDimWarp=false` + reduced winner → the LNT carries NO WARP dim at
  all (narrower than the pre-R3 flag=false semantics); pinned in
  `TestWarpDimRestriction`.
- tiling-verify: NO new check, `verify.go` untouched (R3 is feasible-area shaping,
  exactly like R-REDUCED; the verify re-checks per-(soc,tile,extent) LoC membership,
  unchanged for the winner's digits).
- MaxCTA stays 512 (fix5 §6 ruling — out of scope).
- fix6 §2 const-K benchmark variants: NOT added (constraint).

## Deviations from Plan

1. **[Plan/Task-2 action — call-site count]**: the plan said "the 4 existing
   digitListOf test call sites (adaptive_test.go ×2, tiling_test.go ×2)"; the tree
   carries 8 (adaptive_test.go ×2, tiling_test.go ×6 — two more per-dim pins, the
   raw-product probe, and the enumerateBest mirror). All 8 updated per the plan's
   rule (each fixture's intended winner; derivations in comments).
2. **[Plan/moved pins — re-derived, never weakened]**: three const-extent matmul
   pins legitimately moved under R3 (fix33, decide-lower composition, the print
   golden) — the plan's anticipated category; re-derived from the ACTUAL post-R3
   search output with full derivations in the pin comments. The family-B guard and
   fePin survived UNCHANGED as the plan required (a ranking bug there would have
   meant fixing the ranking, never the pin).
3. **[Rule — spec-level gap recorded, not fixed] Gate 2 fails on the two
   schedule-changed kernels**: two latent defects (store gate for one-tier ladders;
   FRAG chunk-coordinate OOB) in the never-numerically-exercised collapse emission
   path, precisely located (cg/cxx prescan gating + FRAG coordinate arithmetic).
   Per the plan's stop-rule: no pin weakened, no out-of-scope fix; recorded with the
   mechanism diagnosis and a recommended follow-up fix plan (or author revert of R3).
4. **[Plan/must-have expectation — honest record]**: the plan expected the
   elementwise four to re-choose to the collapse on GPU; they do not (their layout
   makes the reduced dim the coalescing axis — L2 selects k). R3 instead re-chose
   row_sum/softmax, the same mechanism on the opposite layout. The unit-level
   re-choice pin (const-K scale_copy → warps-on-C K=SEQ, ceremony unrepresentable)
   proves the machinery; the GPU-level elementwise win needs const-shape
   specialization (fix6 D-fix6-01) — same conclusion as qwo, now with the complete
   layout-level mechanism.
5. **[Test-fix during RED→GREEN]**: the L2-membership ranking subtest's first draft
   accidentally put BOTH r and k in the stride-1 set (within-set innermost then
   correctly picked k); the fixture was corrected so membership itself is the
   decider (set = {r} only) — the comparator was right, the test premise was wrong.

## Commits

| Hash | Message |
|---|---|
| 6fcb41f | chore(quick-261029-v9k): record BEFORE-state benchmark and .cu hash manifest at fdf8ef3 |
| 5a350e6 | test(quick-261029-v9k): add failing fix5-T2 R3 pins (RED) |
| c3f9c49 | feat(quick-261029-v9k): implement fix5 T2 R3 warp-dim extent-aware placement (GREEN) |
| 33d272e | test(quick-261029-v9k): re-pin const-extent matmul fixtures under R3 (never weaken) |
| ec7ff48 | test(quick-261029-v9k): pin the R3 WARP-on-non-reduced-dim lowering flow |
| 722a652 | test(quick-261029-v9k): add failing one-tier collapse emission pins (RED) |
| fd72179 | fix(quick-261029-v9k): fix the two one-tier collapse emission defects (GREEN — real, unit-locked, but .cu-neutral for row_sum) |
| 8085b84 | test(quick-261029-v9k): §3.1 ruling pins — surviving-WARP prune + fail-closed mirror + re-derivations |
| ff929a5 | fix(quick-261029-v9k): §3.1 ruling — prune surviving-WARP vectors from the feasible area + tile-lower fail-closed mirror |

## Self-Check: PASSED

- internal/opt/tiling/r3_test.go — FOUND (created, 324 lines)
- internal/opt/tiling/search.go — modified (warpDimOf present, verified by tests)
- before-bench.txt / before-cu-hashes.txt — FOUND (14 sha256 entries, 14-kernel table)
- go build/vet/test exit 0 (25/25 packages, final tree)
- zero-diff disciplines: `git diff` empty for runtime/, internal/opt/finallower/,
  internal/opt/tiling/verify.go, judge.go, constraints.go (verified post-commit)
- family-B six .cu sha256-equal to before-cu-hashes.txt (the hard EXACT gate)
- AFTER .cu manifest: changed set = exactly {row_sum, softmax}; elementwise four
  bit-identical (gate-3 failure evidence); BEFORE worktree removed (Task 1)
- memcheck run on every schedule-changed kernel (row_sum ×109, softmax ×55 Invalid
  __local__ reads) — the gate-2 failure evidence
- SUMMARY status: complete-with-s31-ruling (gates 1/4 pass; gate 2 fails on the
  two R3-re-chosen kernels with recorded mechanism; gate 3 fails the elementwise
  ≥10× clause with the layout-level diagnosis, regression clause passes)

## Post-Ruling Benchmark (A6000, --bench-iters 20, --jobs 1, avg µs)

All .cu bit-identical to BEFORE — every delta is GPU run noise. Worst:
gemm +3.7% (bit-identical binary; the v9k BEFORE run itself showed +1.3%
run-to-run on the same binary). No kernel >5% slower — the regression
clause PASSES.

| kernel | BEFORE (fdf8ef3) | AFTER (pruned) | Δ |
|---|---|---|---|
| dup_expr | 1201110 | 1180490 | −1.7% |
| dup_const | 1214390 | 1188790 | −2.1% |
| axpy3 | 2387140 | 2384660 | −0.1% |
| mlp | 6424.4 | 6376.8 | −0.7% |
| softmax | 60090.9 | 60550.7 | +0.8% |
| mm_small | 2883.2 | 2928.2 | +1.6% |
| mm_mid | 6494.4 | 6587.9 | +1.4% |
| mm_splitk | 6099.6 | 6201.7 | +1.7% |
| transpose | 330305 | 335559 | +1.6% |
| row_sum | 470696 | 471961 | +0.3% |
| scale_copy | 2387250 | 2376550 | −0.4% |
| gemm | 9819.2 | 10180.0 | +3.7% (bit-identical binary — noise) |
| qk_score | 17409.5 | 17605.8 | +1.1% |
| layernorm | 487444 | 492873 | +1.1% |

Measurement note: a first benchmark attempt with `--jobs 4` produced
4–6× inflated timings on EVERY kernel (scale_copy spread 2.38–9.95 s) —
four benchmark processes contending for one GPU; the protocol numbers
above are single-job. Bit-identical .cu files make compiler-side timing
change impossible, so the single-job table is the honest record.
