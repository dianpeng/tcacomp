---
phase: quick-261029-qwo
plan: 01
type: execute
wave: 1
depends_on: []
tags: [tiling, scheduling, adaptive-loc, reducer-collapse, tier-efficiency, gpu-validation]
files_modified:
  - internal/opt/tiling/constraints.go
  - internal/opt/tiling/search.go
  - internal/opt/tiling/judge.go
  - internal/opt/tiling/verify.go
  - internal/opt/tiling/record.go
  - internal/opt/tiling/adaptive_test.go
  - internal/opt/tiling/tiling_test.go
  - internal/opt/tiling/lower_test.go
  - internal/opt/finallower/reducer.go
  - internal/e2e/frontend_test.go
  - .planning/STATE.md
autonomous: true
requirements: [QUICK-261029-QWO]

must_haves:
  truths:
    - "Every dim's digit list carries ONE extent-derived adaptive top digit per SoC (R1): LoC_adaptive = clamp(ceil(extent/T_d), 2, MaxCTA) for CTA dims, clamp(..., 2, MaxWarpsPerCTA) for WARP dims, where extent is the dim's const extent when known and Params.NominalExtent otherwise — so a full-grid candidate exists at every tile size"
    - "locPresetOK (constraints.go) accepts the adaptive digit via the SAME shared formula the search enumerates (single source), and tiling-verify stays green on search-chosen adaptive records (fix5 §5 gate 4)"
    - "R4 tier efficiency min(1, extent_k/(32*K_k)) scales the traffic credit of fills covering a warp-sharded reduced dim whose extent cannot feed its lanes — the objective stops paying for degenerate lane-sharding, so the search can choose K=SEQ on small-K kernels"
    - "A one-tier ladder (reduced dim entirely SEQ) lowers with no WarpReducer, no shuffle groups, no SMEM staging, no __syncthreads in the bracket — the SeqReducer per-thread accumulator and the existing single-owner store path (R2; runtime/ untouched)"
    - "go build/vet/test all exit 0 (fix5 §5 gate 1); full GPU suite 14/14 PASS with memcheck clean on every schedule-changed kernel (gate 2)"
    - "Benchmark gate (gate 3, --bench-iters 20 vs the n63-pinned numbers): dup_expr, dup_const, axpy3, scale_copy each >= 10x faster; no suite kernel > 5% slower; results recorded with .cu-diff attribution per family (the n63 protocol)"
    - "No pin was weakened: any drifted pin (tiling_test.go, lower_test.go traffic pins, the fePinScheduleSignature e2e pin) was re-derived from freshly emitted .cu with the derivation in the comment"
  artifacts:
    - path: "internal/opt/tiling/constraints.go"
      provides: "adaptiveLoC — the shared R1 formula (clamp(ceil(extent/tile), 2, SoC cap)) and locPresetOK's adaptive-digit acceptance path"
      contains: "adaptiveLoC"
    - path: "internal/opt/tiling/search.go"
      provides: "digitListOf's one adaptive top digit per (dim, SoC, tile), deduped, in the fixed enumeration order"
      contains: "adaptiveLoC"
    - path: "internal/opt/tiling/judge.go"
      provides: "the R4 tier-efficiency function shared by evalSchedule's score"
      contains: "TierEff"
    - path: "internal/opt/tiling/adaptive_test.go"
      provides: "RED->GREEN pins: adaptive digit values, locPresetOK acceptance/rejection, tierEff values, the small-K re-choice pin"
      min_lines: 80
    - path: ".planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/261029-qwo-SUMMARY.md"
      provides: "the before/after benchmark table vs the n63-pinned numbers, suite verdict, .cu-diff attribution"
  key_links:
    - from: "internal/opt/tiling/search.go (digitListOf)"
      to: "internal/opt/tiling/constraints.go (adaptiveLoC)"
      via: "the digit builder passes the dim's effective extent (constExtentOf with NominalExtent fallback)"
      pattern: "adaptiveLoC\\("
    - from: "internal/opt/tiling/verify.go (locPresetOK re-check)"
      to: "internal/opt/tiling/constraints.go (adaptiveLoC)"
      via: "the SAME formula with the SAME effective extent — verify never re-derives"
      pattern: "adaptiveLoC\\("
    - from: "internal/opt/tiling/search.go (evalSchedule)"
      to: "internal/opt/tiling/judge.go (tier-efficiency)"
      via: "per-site traffic credit scaling folded into score' only; res.traffic stays the raw int pin"
      pattern: "TierEff"
---

<objective>
Implement fix5 T1 (R1 + R2 + R4) per `design/fix/fix5.md` — the authoritative
spec — and its §5 verification gates, with the n63 before/after benchmark
protocol. R1: the fixed LoC preset lists stay as the floor granularity and
every dim's digit list gains ONE extent-derived adaptive top digit
(`clamp(ceil(extent/T_d), 2, MaxCTA)` / `clamp(..., 2, MaxWarpsPerCTA)`), so a
full-grid candidate exists at every tile size and the n63 occupancy factor can
actually bite. R4: a warp-sharded reduced dim whose nominal extent cannot feed
its lanes earns no traffic credit for the fills it fragments (tier efficiency
`min(1, extent_k/(32*K_k))` scales the holders term of K-covering fills) — the
minimal objective amendment that lets the search CHOOSE K=SEQ on small-K
kernels. R2: a one-tier ladder (reduced axis entirely covered by the SEQ
layer) lowers with no reducer ceremony — the existing SeqReducer per-thread
accumulator path, runtime untouched.

OUT OF SCOPE (locked, do NOT implement): R3 warp/lane axis freedom (fix5 T2 —
the §2-table mention of extent-aware WARP ordering in digitListOf belongs to
T2, NOT this task), fix6 work-shape model, any `MaxCTA` change (stays 512 per
the §6 ruling), any runtime/ change.

Purpose: the elementwise/latency-bound kernel family (dup_expr 1.26 s,
dup_const 1.23 s, axpy3 2.45 s, scale_copy 2.38 s) is 100–1000x off because the
search menu is structurally starved (fix5 §0 defects 2 and 3). T1 lands the
interim sane schedule — adaptive-grid CTAs x 1-warp (blockDim 32) x pure
sequential per-thread streams, zero ceremony — expected >= 50x on that family
with no family-B split-K regression (those vectors are multi-tier, unaffected).
Output: search-side implementation + final-lower R2 verification + re-derived
pins + GPU-verified before/after benchmark in the SUMMARY.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@design/fix/fix5.md
@design/fix/fix-opt-tile-search.md
@.planning/quick/261029-n63-factor-sm-occupancy-into-the-tiling-sear/261029-n63-SUMMARY.md
@internal/opt/tiling/search.go
@internal/opt/tiling/constraints.go
@internal/opt/tiling/judge.go
@internal/opt/tiling/record.go
@internal/opt/finallower/reducer.go

<interfaces>
<!-- Key surfaces the executor needs (extracted from the codebase). -->

From internal/opt/tiling/constraints.go:
```go
func locPresetOK(soc SoCType, loc int, p Params) bool // fix4 2.7 re-check #1; called at verify.go:189 and tiling_test.go:574 — SIGNATURE MUST GROW (tile + effective extent) for the R1 adaptive acceptance path
func occupancyOf(iters map[ir.Node]IterDecision, smemBytes int, p Params) occupancy // n63 single source; UNCHANGED
type occupancy struct { grid, activeGrid, ctasPerSM, capacity int; blockDim int; idleFrac, fill, factor float64 }
```

From internal/opt/tiling/search.go:
```go
func digitListOf(iter ir.Node, p Params, reduced map[ir.Node]bool) []schedDigit // SEQ ++ CTA(presetDesc(CtaLoCPresets) x tiles) ++ WARP(...); R-LOCCAP capOf per (dim,tile) via constExtentOf; the R1 adaptive digit is added HERE
// evalSchedule's promoted loop: res.traffic += s.j.bytes * (s.j.reuse - 1) * s.j.holders  (int)
// then: res.score = float64(res.traffic) * occupancyOf(iters, smem, p).factor  — R4 inserts the per-site tierEff into THIS score product only
type schedResult struct { iters map[ir.Node]IterDecision; sites map[uint64]SiteDecision; score float64; traffic int; partism int; smem, frag, promotedCount int }
```
Note: extent, constExtent := constExtentOf(iter) is already derived inside digitListOf (capOf) — reuse that helper for the adaptive digit's effective extent.

From internal/opt/tiling/judge.go:
```go
type siteJudgment struct { promoted bool; class CacheClass; reuse, bytes, holders int; reason string }
func judgeSite(sf SiteFacts, iters map[ir.Node]IterDecision, ctx blockDimCtx, p Params) siteJudgment
type blockDimCtx struct { blockDim int } // = (Π warp K) × 32 — the empty warp product is 1 (one-warp CTA, blockDim 32)
```

From internal/opt/tiling/record.go:
```go
// Params: CtaLoCPresets {2,4,8,16,32,64}, WarpLoCPresets {2,4,8,16,32}, MaxCTA 512, MaxWarpsPerCTA 32, NominalExtent 4096 (the stand-in extent), ReducedDimWarp true. No new Params fields (fix5 §2).
type IterDecision struct { SoC SoCType; LoC int; TileSize int }
```

From internal/opt/finallower/reducer.go:
// newReducerConstruction(prov.Soc, ...) already keys on the reduced dim's SoC:
// SoCSequential -> NewRuntimeSequentialReducer (NO staging row, NO ThreadMapping)
// SoCWARP -> NewRuntimeWarpReducer(init, lambda, threadMap, staging) — the WarpReducer bracket (shuffle + SMEM + __syncthreads)
// R2 for T1 = the one-tier (SEQ) ladder must land on the Sequential path with zero ceremony; the bracket restructure (buildElementLoop / the zero-between flat bracket) is UNCHANGED.
// stagingOf returns nil for KindRuntimeSequentialReducer — the SeqReducer path already allocates no SMEM staging.

From runtime/tca/reducer.h (READ-ONLY reference — fix5 §3: NO runtime change):
// class SeqReducer — "per-thread accumulators, no synchronization; Combine returns acc_": Init() = acc_=init_, Reduce(v) = acc_=fn_(acc_,v), Combine() = acc_. This IS the plain per-thread accumulator R2 wants; the emitted form is `tca::rt::SeqReducer<float> reducer0(c2, CombinerFn);`.
// class WarpReducer — the per-element warp-collective bracket (Init single-writer SMEM slots, Reduce participating-lane fold, Combine __syncthreads + slot fold) — must NOT appear in any one-tier ladder's .cu.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Record the BEFORE-state GPU benchmark, then pin R1/R4 RED</name>
  <files>.planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/before-bench.txt, internal/opt/tiling/adaptive_test.go</files>
  <action>
  STEP 1 — BEFORE-state FIRST, before any code change (the task constraint: before-state measurements recorded BEFORE code changes; the n63 protocol). The current dev tip (6975e07, the fix5-spec docs commit on top of the n63 re-validation) IS the honest before — no feature commits exist yet.
  1. `git worktree add /tmp/tcacomp-qwo-before HEAD` (read-only use; never edit inside it) and `/tmp/tcacomp-qwo-before/scripts/build.sh` — the bin/tcacomp stale-prebuilt gotcha applies: always build, never reuse bin/.
  2. From the MAIN checkout run `python3 -I testing/orchestrator.py --tcacomp /tmp/tcacomp-qwo-before/bin/tcacomp --bench-iters 20` and save the FULL per-kernel output to `.planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/before-bench.txt`. This table is the gate-3 comparison anchor: the n63-pinned elementwise numbers (dup_expr 1.26 s, dup_const 1.23 s, axpy3 2.45 s, scale_copy 2.38 s; family B: mm_small 2.91 ms, mm_mid 6.55 ms, mm_splitk 6.15 ms, gemm 9.89 ms, qk_score 17.6 ms, mlp 6.32 ms) must match it within noise — if they do not, record both and use the freshly measured table.
  3. Sanity: the suite portion of that run reports 14/14 PASS (the before state is known-good).

  STEP 2 — RED pins (TDD; follow the n63 commit discipline: a standalone `test(quick-261029-qwo): add failing fix5-T1 pins (RED)` commit). Create `internal/opt/tiling/adaptive_test.go` pinning, with derivations in comments (per the n63 occupancy_test.go style):
  1. R1 formula pins (function to be named `adaptiveLoC` in constraints.go, shared single source): CTA tile=32, NominalExtent=4096, MaxCTA=512 -> clamp(ceil(4096/32)=128, 2, 512) = 128; WARP same inputs with MaxWarpsPerCTA=32 -> 32; floor: tile >= extent -> ceil 1 -> clamped to 2; cap: tile=8, NominalExtent=4096 -> 512 clamped by MaxCTA (and 32 by MaxWarpsPerCTA).
  2. Effective-extent discipline pin: for a CONST-extent dim (e.g. extent 96, tile 32) the adaptive digit is 3 — NOT a preset value — and digitListOf emits it; this pins the const-extent path end to end (search-side capOf discipline: adaptive = clamp(ceil(extent/tile), 2, cap) computed on the SAME effective extent the verify side will use).
  3. locPresetOK acceptance pins (its signature grows to carry the tile and the effective extent — per the fix5 §2 table, constraints.go hosts the acceptance and shares the formula with verify): (CTA, loc=3, tile=32, extent=96) accepted via the adaptive path; (CTA, loc=3, tile=32, extent=512) REJECTED (adaptiveLoC gives 16 there); preset membership still works unchanged.
  4. R4 tier-efficiency pins (function to live in judge.go, applied in evalSchedule's score only): reduced dim WARP-assigned, extent_k=4, K_k=2 -> min(1, 4/(32*2)) = 0.0625; extent_k=64, K_k=2 -> min(1, 64/64) = 1; reduced dim SEQ or extent_k >= 32*K_k -> exactly 1 (no scaling); dynamic-extent reduced dim falls back to the NominalExtent stand-in.
  5. Search-level re-choice pin (the fix5 §4 headline): an LNT shaped like scale_copy — two outer const-extent dims + one tiny reduced dim (extent 4), ReducedDimWarp true — must, under R1+R4, choose a vector with NO WARP digit on the reduced dim (the K=SEQ collapse vector) and a CTA LoC beyond the old preset top of 64 (the adaptive full-grid digit). Pin via the existing enumerateBest/test harness pattern used by TestTrafficArithmeticAndOptimality; derive the expected winner from the formula, and if the freshly computed winner differs, RE-DERIVE the pin from the actual search output with the score' + tierEff derivation in the comment — never weaken it to force a predetermined vector.
  Run: the new pins MUST FAIL (functions don't exist yet). Do NOT touch production code in this task.
  </action>
  <verify>
    <automated>test -s .planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/before-bench.txt && grep -c "PASS" .planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/before-bench.txt &gt;= 14 && go test ./internal/opt/tiling/ -run 'TestAdaptive|TestLocPresetAdaptive|TestTierEff|TestSearchRechoice' 2&gt;&1 | grep -q "FAIL"</automated>
  </verify>
  <done>before-bench.txt holds the full 14-kernel BEFORE table (--bench-iters 20, 14/14 PASS) from a worktree-at-HEAD build; adaptive_test.go exists with the five pin groups above; the pin run FAILS (RED); a standalone RED commit exists.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement R1 + R4 + the R2 verification, re-derive drifted pins</name>
  <files>internal/opt/tiling/constraints.go, internal/opt/tiling/search.go, internal/opt/tiling/judge.go, internal/opt/tiling/verify.go, internal/opt/tiling/record.go, internal/opt/finallower/reducer.go, internal/opt/tiling/tiling_test.go, internal/opt/tiling/lower_test.go, internal/e2e/frontend_test.go</files>
  <behavior>
    - TestAdaptiveLoCFormula: the §1 formula values from Task 1 group 1.
    - TestDigitListAdaptiveDigit: const-extent 96/tile 32 emits a 3 digit; dedup — when the adaptive value coincides with a preset it appears exactly once.
    - TestLocPresetOKAdaptive: acceptance/rejection per Task 1 group 3.
    - TestTierEff: values per Task 1 group 4; res.traffic (int) is NEVER scaled — only score'.
    - TestSearchRechoice: the scale_copy-shaped LNT re-chooses the K=SEQ adaptive-grid vector.
  </behavior>
  <action>
  Commit discipline: `feat(quick-261029-qwo): implement fix5 T1 R1+R2+R4 (GREEN)` after the pins pass, then a re-pin commit if pins drifted.

  R1 (constraints.go + search.go):
  1. constraints.go: add `adaptiveLoC(soc SoCType, tile, extent int, p Params) int` — clamp(ceilDiv(extent, tile), 2, p.MaxCTA) for SoCTA, clamp(..., 2, p.MaxWarpsPerCTA) for SoCWARP. This is THE single source (fix5 §1: "shared with tiling-verify exactly like every other formula").
  2. Extend `locPresetOK` to carry the tile and the effective extent (change its signature and both call sites — verify.go:189 and tiling_test.go:574 — passing the iter's const extent via the existing constExtentOf helper, falling back to p.NominalExtent, exactly the discipline digitListOf already uses in capOf). Acceptance: preset membership OR loc == adaptiveLoC(soc, tile, effExtent, p). This fallback-to-NominalExtent reading of the spec's "NominalExtent feeds R1" is REQUIRED for search/verify lockstep on const-extent dims (const 96/tile 32 yields adaptive 3, which the bare-NominalExtent formula would reject at verify time) — record the choice in the SUMMARY.
  3. search.go digitListOf: per tile, after the preset loop for each SoC section, append the adaptive digit when it is not already present (dedup — presets win the duplicate). Deterministic placement: tiles ascending, appended after that section's preset digits; update the determinism comment. Respect the existing R-LOCCAP capOf path (const extents keep capping presets as today — the adaptive digit uses the effective extent so it needs no separate cap). R3's extent-aware WARP ordering is NOT part of this task — do not add it.
  4. record.go: extend the NominalExtent doc comment to name its fix5 roles (R1 adaptive digits + R4 tier efficiency) per the fix5 §2 "document in the §2.1-style table" row. NO new Params fields; MaxCTA stays 512 (§6 ruling 2).

  R4 (judge.go + search.go):
  5. judge.go: add the tier-efficiency function (name it `reducedTierEff`): given the LNT's reduced set (lf.reduced), the decision map, and a site's covered set — for each covered dim that is reduced AND WARP-assigned, multiply min(1, float64(effExtent_k)/float64(32*d.LoC)) into the product; everything else contributes exactly 1. effExtent_k = the reduced dim's const extent or p.NominalExtent (same discipline as R1).
  6. search.go evalSchedule: fold the per-site tierEff into the SCORE only — restructure the promoted-site loop to accumulate a float64 scaledTraffic (float64(bytes*(reuse-1)) * float64(holders) * tierEff) alongside the unchanged int res.traffic pin; res.score = scaledTraffic * occupancyOf(...).factor. Non-covering sites carry tierEff 1 (bit-identical to today's score). tiling-verify stays R4-blind (selection-only, the n63 precedent: verify re-checks legality, never the objective).

  R2 (final-lower verification; runtime/ is UNCHANGED per fix5 §3):
  7. With the GREEN build, emit the .cu for the four elementwise kernels (testing/kernels/runtime/{scale_copy,axpy3,dup_expr,dup_const}.tca — use the orchestrator's --keep-builds or the /tmp/tcainspect pattern from fix5 §0) and assert per kernel: the reduced dim's ladder is one-tier (SeqReducer emission, `tca::rt::SeqReducer<...>`), NO WarpReducer construction, NO SmemView staging row for the reducer, NO __syncthreads inside the element bracket, and the epilogue store remains is_first_in_cta-gated single-owner. The SeqReducer register path IS the spec's "plain per-thread accumulator" realization (SSA carries no loop state without phis — an object-free accumulator is impossible compiler-side; the runtime class is already per-thread registers with no synchronization, and nvcc inlines the fixed combiner). If — and only if — the emitted one-tier bracket carries avoidable compiler-side ceremony (per-element re-construction, non-inlined combiner indirection, redundant locals), remove it in internal/opt/finallower/reducer.go and/or the internal/cg/cxx emitter; do not touch runtime/.
  8. Re-derive every drifted pin from freshly emitted output, never weaken: run `go test ./...` and for each failure decide implement-vs-pin per the n63 rule (the implement is illegal -> fix the implement; the legal schedule re-chose -> re-derive the pin with the score'×tierEff derivation in the comment). This explicitly covers the tiling_test.go/lower_test.go traffic-score pins and the fePinScheduleSignature split-k class pin in internal/e2e/frontend_test.go (the frontend fixture may re-choose an adaptive CTA digit while keeping the split-k class; re-derive from its fresh .cu exactly as n63 did).

  Anti-scope-check: diff must NOT contain any R3 machinery (WARP digits on non-reduced dims, extent-aware WARP ordering), no Params fields beyond docs, no runtime/ changes, no MaxCTA change.
  </action>
  <verify>
    <automated>go build ./... &amp;&amp; go vet ./... &amp;&amp; go test ./... 2>&1 | tail -5 | grep -Eq "ok|FAIL" &amp;&amp; go test ./... &gt; /tmp/qwo-test.log 2&gt;&1; ! grep -q FAIL /tmp/qwo-test.log</automated>
  </verify>
  <done>All five behavior groups pass; go build/vet/test exit 0 with every changed pin carrying a re-derivation comment; the elementwise .cu inspection confirms the one-tier SeqReducer shape; the anti-scope diff check is clean.</done>
</task>

<task type="auto">
  <name>Task 3: fix5 §5 verification gates + n63-protocol before/after benchmark + record</name>
  <files>.planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/261029-qwo-SUMMARY.md, .planning/STATE.md</files>
  <action>
  The bin/tcacomp stale-prebuilt gotcha: `scripts/build.sh` BEFORE any GPU run — never trust bin/.
  1. Gate 1 (already green from Task 2): `go build ./...`, `go vet ./...`, `go test ./...` exit 0 — re-run to confirm on the final tree.
  2. Gate 2: `python3 -I testing/orchestrator.py --tcacomp bin/tcacomp` — 14/14 PASS, canary 0 everywhere, maxdiff within every suite.toml tolerance. Then memcheck on EVERY schedule-changed kernel: diff each kernel's freshly generated .cu against the BEFORE worktree's (--keep-builds on both sides, or dump from /tmp/tcacomp-qwo-before) and run `python3 -I testing/orchestrator.py --tcacomp bin/tcacomp --sanitizer memcheck --filter <kernel>` for each changed one — at minimum mm_small and one representative of each moved family (one-tier collapsed elementwise + any family-B kernel whose schedule moved). A verbose run must confirm the compute-sanitizer argv.
  3. Gate 3: `python3 -I testing/orchestrator.py --tcacomp bin/tcacomp --bench-iters 20`. Build the before/after table against before-bench.txt (Task 1) and cross-check the elementwise anchor against the n63-pinned numbers. HARD GATES: dup_expr, dup_const, axpy3, scale_copy EACH >= 10x faster; NO suite kernel > 5% slower (family B's split-K vectors are multi-tier and must be unaffected beyond noise — mm_small, mm_mid, mm_splitk, gemm, qk_score, mlp). fix5 §4's design expectation is elementwise 2.38 s-class -> low tens of ms (>= 50x): if a gate fails, STOP — do not tune by weakening pins; diagnose via the .cu diff (the likely failure mode is the search still not choosing the collapse vector, i.e. an R1/R4 formula bug, or bracket ceremony the Task-2 inspection missed) and either fix forward with a follow-up commit or record the failure prominently in the SUMMARY with the attribution evidence.
  4. Attribution: for every kernel that moved > 10%, dump before/after .cu and record WHICH schedule changed (the n63 .cu-diff attribution per family) — the expected elementwise story: K WARP -> K SEQ collapse + adaptive-LoC full grid, blockDim 32; the expected family-B story: unchanged split-K multi-tier vectors (adaptive digits present in the list but not chosen).
  5. Gate 4: tiling-verify green is implied by the suite (tile-lower refuses an unstamped record) — state that explicitly in the SUMMARY; GlobalReducer/D1/D2 untouched.
  6. Record: write `.planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/261029-qwo-SUMMARY.md` (frontmatter per the n63 SUMMARY: phase quick-261029-qwo, subsystem opt/tiling+finallower, provides/affects, tech-stack patterns, key-files, key-decisions incl. the effective-extent reading of R1 and the SeqReducer-as-plain-accumulator R2 realization, requirements-completed [QUICK-261029-QWO], status complete) with the full before/after table, gate verdicts, and per-family attribution; update `.planning/STATE.md` stopped_at/last_activity/last_updated. `git worktree remove /tmp/tcacomp-qwo-before` when the table is recorded.
  7. Commit: `chore(quick-261029-qwo): record fix5-T1 GPU re-validation and before/after benchmark` (SUMMARY + STATE.md, with the Co-Authored-By trailer per repo settings).
  </action>
  <verify>
    <automated>python3 -I testing/orchestrator.py --tcacomp bin/tcacomp 2>&1 | tail -20 | grep -c "FAIL" | grep -q "^0$" &amp;&amp; test -f .planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/261029-qwo-SUMMARY.md &amp;&amp; grep -q "before" .planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/261029-qwo-SUMMARY.md</automated>
  </verify>
  <done>Gate 1-4 verdicts recorded; 14/14 PASS + memcheck clean on every schedule-changed kernel; the before/after table shows each elementwise kernel >= 10x and zero kernel > 5% slower; per-family .cu-diff attribution in the SUMMARY; STATE.md updated; before worktree removed; record commit landed.</done>
</task>

</tasks>

<verification>
- go build ./... / go vet ./... / go test ./... exit 0 (fix5 §5 gate 1).
- Full orchestrator suite 14/14 on a freshly built bin/tcacomp; memcheck clean on every schedule-changed kernel (gate 2).
- Before/after benchmark (--bench-iters 20) vs the Task-1 BEFORE table: elementwise family >= 10x each, no kernel > 5% slower (gate 3), recorded with .cu-diff attribution (the n63 protocol).
- tiling-verify green on every published record (gate 4, implied by the suite); GlobalReducer unreachable, runtime/ untouched.
</verification>

<success_criteria>
- R1: every dim's digit list carries one extent-derived adaptive top digit per SoC; locPresetOK accepts it via the shared constraints.go formula; verify and search use the same effective extent (const extent or NominalExtent stand-in).
- R4: tier efficiency min(1, extent_k/(32*K_k)) scales the traffic credit of K-covering fills on underfed warp-sharded reduced dims, in score' only; the int traffic pins survive unscaled.
- R2: one-tier ladders emit the SeqReducer register path with no WarpReducer/shuffle/SMEM-staging/__syncthreads ceremony and unchanged single-owner stores; runtime/ has zero diff.
- The scale_copy-shaped search fixture re-chooses the K=SEQ adaptive-grid vector; all drifted pins re-derived (never weakened) with derivations in comments.
- GPU: 14/14 PASS, memcheck clean on changed kernels, elementwise family each >= 10x (design expectation >= 50x to low tens of ms), family B within +/- 5%, all recorded in 261029-qwo-SUMMARY.md.
</success_criteria>

<output>
Create `.planning/quick/261029-qwo-implement-fix5-t1-r1-adaptive-loc-ceilin/261029-qwo-SUMMARY.md` when done (Task 3 writes it with the benchmark evidence and gate verdicts).
</output>
