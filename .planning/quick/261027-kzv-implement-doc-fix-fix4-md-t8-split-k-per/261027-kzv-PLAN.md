---
phase: quick-261027-kzv
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/opt/tiling/record.go
  - internal/opt/tiling/tiling.go
  - internal/opt/tiling/search.go
  - internal/opt/tiling/lower.go
  - internal/opt/tiling/overlap_test.go
  - internal/opt/tiling/lower_test.go
  - internal/opt/tiling/tiling_test.go
  - internal/e2e/frontend_test.go
  - internal/cg/cxx/emit_test.go
  - internal/cg/cxx/testdata/reduce_splitk.cu
  - doc/fix/fix4.md
autonomous: true
requirements: [fix4-T8]

must_haves:
  truths:
    - "Under DefaultParams, a reduce-carrying kernel's searched schedule assigns WARP to the REDUCED dim (split-k, fix4 §3.5) and the full pipeline lowers + emits it: WarpReducer construction with a #warps x sizeof(T) staging row, per-element bracket [Init -> reduced-dim WARP TBL rounds (stride K_d*T_d) with lane-chunk inside walks -> Reduce(value) per element -> Combine()], uniform element walks for the surviving dims, single-owner store"
    - "The split-k matmul is numerically correct on the A6000: frontend matmul_scale BOTH syntax variants (M=CTA x N(reduced)=WARP/2 split-2 x Q=SEQ) canary 0 and maxAbs <= reduceTol (2e-3) vs the CPU reference — the dedicated fix4 §3.5 e2e"
    - "compute-sanitizer memcheck AND racecheck report ZERO errors on the split-k kernel (the cross-warp staging fold and the warp-varying LoadTileCta window bases are the race-freedom-relevant surfaces)"
    - "Explicit ReducedDimWarp=false still FAILS CLOSED at tile-lower on a WARP-assigned reduced dim (the Wave-1 defense becomes the opt-out policy mode; rejectWarpReducedDims code unchanged)"
    - "The unruled iq6 design gap stays fenced: matmulGoldenDesignGapSkip t.Skip lines REMAIN; no surviving-WARP gate is added, widened, or adopted (candidates 1/2/3 untouched)"
    - "Hermetic gate green: go build ./... && go vet ./... && go test ./...; module count 1 (go list -m all | wc -l); existing goldens byte-stable; the ONLY selection that changes is fixtures whose reduced set opens under the flag (verified blast radius: exactly 4 tests)"
  artifacts:
    - path: internal/opt/tiling/record.go
      provides: "DefaultParams ReducedDimWarp false->true (the Wave-2 flip) + truthful comments"
      contains: "ReducedDimWarp:   true,"
    - path: internal/e2e/frontend_test.go
      provides: "re-derived hermetic SPLIT-K schedule pin (launch_bounds 64, dim3(64)/dim3(64)/smem 65544, WarpReducer + staging + ThrIterator N TBL literals)"
      contains: "WarpReducer"
    - path: internal/cg/cxx/testdata/reduce_splitk.cu
      provides: "NEW committed golden: crafted-record split-k emission (WarpReducer + staging + bracket), byte-regenerated"
      contains: "WarpReducer"
    - path: internal/opt/tiling/tiling_test.go
      provides: "TestReducedDimWarpPruning re-pinned: prune under explicit false + the flipped-default split-k winner pinned deterministically"
      contains: "ReducedDimWarp"
    - path: doc/fix/fix4.md
      provides: "§2.1 ReducedDimWarp row + staging note updated to the landed Wave-2 default"
      contains: "Wave 2"
  key_links:
    - from: internal/opt/tiling/record.go (DefaultParams.ReducedDimWarp)
      to: internal/opt/tiling/search.go:303 (R-REDUCED digit prune)
      via: "reduced[iter] && !p.ReducedDimWarp — flag=false prunes WARP digits on reducer-consuming dims"
      pattern: "ReducedDimWarp"
    - from: internal/opt/tiling/record.go (DefaultParams.ReducedDimWarp)
      to: internal/opt/tiling/lower.go:173 (Wave-1 defense)
      via: "if !p.Params.ReducedDimWarp { rejectWarpReducedDims } — stands down at true; UNCHANGED code"
      pattern: "rejectWarpReducedDims"
    - from: internal/e2e/frontend_test.go (compileFrontendKernel)
      to: emitted matmul_scale.cu (WarpReducer split-k emission)
      via: "real pipeline search -> tile-lower -> finallower bracket -> cxx Emit; pin re-derived from the actual bytes, never forced"
      pattern: "compiler\\.CompileKernel"
    - from: emitted Combine bracket
      to: runtime/tca/reducer.h (WarpReducer)
      via: "runtime-owned staging + __syncthreads + replicated fold over warp_count() — §4.4 unchanged, split-k capable; the compiler NEVER emits barriers"
      pattern: "warp_count"
---

<objective>
Implement doc/fix/fix4.md Wave-2 T8 (§8): open the split-k capability — §3.5's reducer wiring — by flipping `Params.ReducedDimWarp` to the Wave-2 default, landing the dedicated split-k matmul e2e (numerical + sanitizers on the A6000), re-pinning the Wave-1 policy tests, and committing the split-k emission golden.

Purpose: Wave 1 (T1–T7) shipped the whole factorization refactor with split-k fenced out of the feasible area (`ReducedDimWarp=false`). The finallower reducer machinery already runs under the flag in hermetic fixtures (261027-e6k), and the runtime WarpReducer is split-k capable by design (§4.4: staging row `#warps × sizeof(T)` + replicated fold over `warp_count()`). T8 is the last staged item of the refactor: turn the capability on, prove it numerically end-to-end, and re-target the Wave-1 policy pins.

PLANNER PROBE GROUND TRUTH (verified 2026-10-27 on a /tmp copy of this exact tree with ONLY the one-line default flip applied — the plan is built on measured outcomes, not predictions):
- Full hermetic suite blast radius of the flip = EXACTLY 4 failing tests: `TestLowerWarpReducedDimRejected`, `TestDefaultParams` (tiling), `TestReducedDimWarpPruning`, `TestFrontendMatmulScalePipeline/written` (e2e). Every golden, every other fixture, green — the committed .cu goldens use crafted records (no search), and the elementwise golden shapes carry an empty reduced set (selection unchanged).
- The REAL frontend matmul_scale (feM=64, feN=128, feQ=64, runtime extents) then selects the split-k vector M=CTA/64/256, Q=SEQ/256, N(reduced)=WARP/2/32 and emits a COMPLETE §3.5 kernel with NO further compiler changes: `__launch_bounds__(64)`, grid dim3(64), blockDim 64 (2 warps), dynamic SMEM pool 65544 B = rhs 65536 B window at the MaxCache cap + 8 B reducer staging (`tca_smem + 65536u`, extent 2u) at a disjoint D8 offset, `tca::rt::WarpReducer<float> reducer0(0.0f, MatMulScaleCombine0, thread_map1, ...)`, per-element bracket `reducer0.Init()` → N tile-between `ThrIterator(thread_map1, 0u, 2u, 0u, 64u, N)` (stride K·T = 64, warp-coordinate rank) with lane-chunk inside walk (init `lane*1`, chunk-local lhs access `N_it5.Index() - lane*1` — the iq6 C1b fix carried) and `reducer0.Reduce(v)` per element → `auto v30 = reducer0.Combine()` after, uniform sequential M/Q element walks, single-owner store gated on `thread_map1.is_first_in_cta()`, `ValidateLaunch(dim3(64u, 1u, 1u), dim3(64u, 1u, 1u), 65544u, 512u)`.
- The TestReducedDimWarpPruning reduce fixture (1024³ const extents) selects M=cta/4/256, N=seq/1/256, K(reduced)=warp/2/32 with lhs promoted FRAG 4 B reuse 256 and rhs promoted SMEM 65536 B reuse 256 — deterministic across runs.

Output: the flipped default with all policy pins re-targeted, a committed split-k emission golden, a green TCACOMP_E2E battery on the A6000 where the frontend matmul IS the dedicated split-k matmul e2e, and doc/fix4.md updated to the landed state.

NOT in scope (fences — read before touching anything):
- The iq6 UNRULED design gap (fix4 §3.1 CTA×WARP(surviving) uncovered cross product): keep both `t.Skip(matmulGoldenDesignGapSkip)` lines in internal/e2e/e2e_test.go EXACTLY as they are. Do NOT adopt candidate rulings 1/2/3 (reduced-only WARP legality prune / template redesign / coverage verifier) as silent decisions — that is the author's call. Do NOT widen any fence: `rejectWarpReducedDims` (lower.go:270-303) keeps its exact semantics (fail-closed whenever the flag is false), and the decide-side `reducedIters` breadth (search.go:357-372) is untouched.
- No runtime/ changes (§4.4 reducers are unchanged by design); no lowering/template redesigns — if the GPU battery surfaces a defect in the split-k emission, fix it MECHANICALLY at the smallest layer with a RED→GREEN pin; if the fix would be a redesign, STOP and report (the lowers-translate-verbs-only ruling).
</objective>

<execution_context>
@HOME/.claude/get-shit-done/workflows/execute-plan.md
@HOME/.claude/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@doc/fix/fix4.md

<interfaces>
The four Wave-1 gates T8 flips or re-pins (all verified line numbers on the current tree):
- internal/opt/tiling/record.go:420 — `ReducedDimWarp:   false,` in DefaultParams (THE flip). Field comment at :388-392 and the DefaultParams doc comment at :401-406 say "Wave 1"/"false" — update both to the Wave-2 truth.
- internal/opt/tiling/search.go:303 — `if reduced[iter] && !p.ReducedDimWarp { return out }` (R-REDUCED digit prune inside digitListOf; comment at :276-277). CODE UNCHANGED — only the "while ReducedDimWarp is false" phrasing becomes "when ReducedDimWarp is false (the Wave-1 policy, now the explicit opt-out)". Same for reducedIters' comment at :351-356 and tiling.go:101-103.
- internal/opt/tiling/lower.go:167-177 — the §3.5 Wave-1 defense `if !p.Params.ReducedDimWarp { rejectWarpReducedDims(...) }` (:173) with `rejectWarpReducedDims` at :270-303. CODE UNCHANGED — re-comment: the defense is now the EXPLICIT-false opt-out mode (DefaultParams no longer trips it).
- internal/opt/tiling/verify.go — NO ReducedDimWarp coupling anywhere (verified); nothing to do.

Test surfaces to re-pin (exact current locations):
- internal/opt/tiling/overlap_test.go:393-410 TestDefaultParams — the `if p.ReducedDimWarp { t.Fatal("...must be false (Wave 1)") }` branch inverts to `!p.ReducedDimWarp → t.Fatal("...must be true (Wave 2, fix4 §3.5)")`; sync the list comment at :390-392.
- internal/opt/tiling/lower_test.go:1035-1082 TestLowerWarpReducedDimRejected — the negative branch constructs `LowerPass{Params: DefaultParams()}`; re-target to explicit `p := DefaultParams(); p.ReducedDimWarp = false`. Keep both positive controls (reduced-dim-SEQ lowers clean; fx3 flag=true stands the defense down) and EXTEND: assert `DefaultParams().ReducedDimWarp == true` so the default-admits policy is pinned at the same site (the defense must NOT fire under the default anymore).
- internal/opt/tiling/tiling_test.go:580-644 TestReducedDimWarpPruning — the `red := runPass(t, redG, DefaultParams())` + "no dim takes WARP" assertion re-targets to an explicit-false params run (add a params-carrying local or set the field on a copy). KEEP the plain (no-reduce) contrast and the reducedIters assertion unchanged. ADD the flipped contrast: under DefaultParams the same reduce fixture deterministically selects the split-k winner M=cta/4/256, N=seq/1/256, K=warp/2/32 with rhs promoted SMEM 65536 B (reuse 256) and lhs promoted FRAG 4 B (reuse 256) — pin all of it (probe-derived; if the pinned values disagree with the run, TRUST THE RUN and re-derive — determinism is guaranteed by the fixed enumeration, but never force a schedule to match a pin).
- internal/e2e/frontend_test.go:229-270 fePinScheduleSignature + :282-308 TestFrontendMatmulScalePipeline — re-derive from the ACTUAL emitted .cu post-flip and re-pin. Expected (probe): rename the helper/comment to the split-k schedule class (fix4 §3.5) and pin at least: `__launch_bounds__(64)`, `ValidateLaunch(dim3(64u, 1u, 1u), dim3(64u, 1u, 1u), 65544u, 512u)`, the literal `<<<dim3(64u, 1u, 1u), dim3(64u, 1u, 1u), 65544u` launch, `tca::rt::WarpReducer<float> reducer0(`, the staging row `tca_smem + 65536u), 2u)`, and the N tile-between `ThrIterator(thread_map1, 0u, 2u, 0u, 64u,` (the K_d·T_d = 64 stride). The helper's own re-derive-first rule stands: never force a schedule to match a pin.

New golden fixture (cxx side — the dedicated hermetic split-k emission pin):
- internal/cg/cxx/emit_test.go — `runTileLowerSchedule` (:126-148) passes `tiling.DefaultParams()` to LowerPass, so post-flip it carries the capability automatically. Add `reduceSplitKGraph` mirroring `reduceWarpGraph` (:150-183: 2-dim chain 1024×512, lhs[m][k] → map(as=[m,k]) → reduce(0.0f,+) → out[m]) with decisions `{SoCTA,100,32}/{SoCWARP,2,32}` on the REDUCED k dim, + typeinfer + runTileLowerSchedule + finallower + `Emit(g, emitOptions())` → `golden.Assert(t, "testdata/reduce_splitk.cu", res.Files["redsplitk.cu"])` + `ScanBracketLaw` (the TestGoldenReduceWarp pattern at :287-297). Name the capability gate in the fixture comment (the finallower runTileLowerScheduleSplitK idiom). Regenerate the golden bytes with the repo's golden-regen mechanism (same as prior sanctioned re-pins) and commit it — numerical backing is the frontend split-k e2e in Task 2.
- internal/e2e/e2e_test.go:242-265 — matmulGoldenDesignGapSkip: DO NOT TOUCH.

GPU battery (all pre-existing, TCACOMP_E2E-gated via skipUnlessE2E: env + cmake (/home/ps/.local/bin/cmake fallback) + nvcc (/usr/local/cuda-12.2/bin/nvcc fallback)):
- TestFrontendMatmulScale (written+omitted) — post-flip THIS IS the dedicated split-k matmul e2e (add a one-line comment saying so); full OUT dump vs CPU reference, canary 0, maxAbs ≤ reduceTol.
- TestFrontendMatmulScaleSanitizer — memcheck + racecheck over the split-k kernel.
- TestReduceWarpE2E/+Sanitizer, TestReduce3E2E/+Sanitizer, TestReduceT1E2E — regression carriers (crafted records, unaffected by the flip).
- TestMatmulScaleE2E/+Sanitizer — stay SKIPPED (design gap).
- Hardware gate: confirm `nvidia-smi` shows the A6000 BEFORE running; if absent, STOP and report (iq6 precedent — T8's GPU half is a hardware gate, not skippable).

Runtime ground truth (unchanged, §4.4): runtime/tca/reducer.h WarpReducer — per-warp accumulators; staging row `#warps × sizeof(T)` in the compiler's dynamic-SMEM pool (the emitter assigns the disjoint offset, here 65536); `Combine()` = `__syncthreads()` FIRST, then every thread folds all `warp_count()` staged slots ascending (deterministic replicated fold), then `__syncthreads()` — runtime-owned barriers, the compiler never emits one. The split-k GPU run is this protocol's first cross-warp NUMERICAL exercise through a real pipeline kernel (conformance_new_api.cu only pins the face at compile/launch).
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: T8a — flip the ReducedDimWarp default, re-pin the Wave-1 policy tests, land the split-k golden (hermetic RED→GREEN)</name>
  <files>internal/opt/tiling/record.go, internal/opt/tiling/tiling.go, internal/opt/tiling/search.go, internal/opt/tiling/lower.go, internal/opt/tiling/overlap_test.go, internal/opt/tiling/lower_test.go, internal/opt/tiling/tiling_test.go, internal/e2e/frontend_test.go, internal/cg/cxx/emit_test.go, internal/cg/cxx/testdata/reduce_splitk.cu</files>
  <behavior>
    - RED (already measured by the planner probe — reproduce locally first): with `ReducedDimWarp: true` in DefaultParams, exactly these fail: TestLowerWarpReducedDimRejected, TestDefaultParams, TestReducedDimWarpPruning, TestFrontendMatmulScalePipeline/written. Nothing else in `go test ./...`.
    - GREEN targets: TestDefaultParams asserts default true (Wave 2); TestLowerWarpReducedDimRejected fails closed only under EXPLICIT ReducedDimWarp=false and stands down under DefaultParams; TestReducedDimWarpPruning pins the prune under explicit false AND the deterministic split-k winner (cta/4/256, seq/1/256, warp/2/32; rhs SMEM 65536 reuse 256, lhs FRAG 4 reuse 256) under the default; TestFrontendMatmulScalePipeline pins the split-k emission signature (launch_bounds 64, dim3(64)/dim3(64)/65544u/512u, WarpReducer + staging + ThrIterator stride-64 N rounds); TestGoldenReduceSplitK byte-pins the crafted-record split-k emission and ScanBracketLaw passes over it.
    - Hermetic gate after: `go build ./... && go vet ./... && go test ./...` green; `go list -m all | wc -l` == 1; existing goldens byte-identical (git status shows ONLY the new testdata/reduce_splitk.cu under testdata).
  </behavior>
  <action>
    1. Flip `ReducedDimWarp: false,` → `true,` at internal/opt/tiling/record.go:420. Update the stale comments to the landed truth: the field comment (:388-392 — the flag is now the OPEN capability; explicit false = the Wave-1 policy), the DefaultParams doc (:401-406 — "ReducedDimWarp true (Wave 2, fix4 §3.5 split-k enabled)"), tiling.go:101-103, search.go:276-277, search.go:351-356, lower.go:167-172 (the defense = the explicit-false opt-out). Comment-only beside :420 — the search/lower/verify CODE paths stay byte-identical (fence: never widen).
    2. Run `go test ./internal/opt/tiling/ ./internal/e2e/` and confirm the failure set is exactly the four probe-named tests — any OTHER failure means an unplanned selection change: investigate before touching anything (do not "fix" an honest all-CTA selection on an unrelated fixture; that is the model's outcome, not a bug).
    3. Re-pin per the interfaces block: overlap_test.go TestDefaultParams (invert the branch + comment); lower_test.go TestLowerWarpReducedDimRejected (explicit-false negative + default-admits assertion); tiling_test.go TestReducedDimWarpPruning (explicit-false prune + flipped split-k winner pin); frontend_test.go fePinScheduleSignature → the split-k schedule pin (rename the helper's comment to the fix4 §3.5 class; re-derive every literal from the freshly emitted matmul_scale.cu via a temporary print of compileFrontendKernel output — expected literals in the interfaces block; pin what is actually there).
    4. Add reduceSplitKGraph + TestGoldenReduceSplitK in internal/cg/cxx/emit_test.go (crafted record {CTA,100,32}/{WARP,2,32} on the reduced k, the reduceWarpGraph shape at 1024×512); regenerate and commit testdata/reduce_splitk.cu; assert ScanBracketLaw over the emitted bytes. This is the byte-stable hermetic pin of the "most intricate emission in the refactor" (§3.5 risk note) — the GPU numerical backing lands in Task 2 before the task closes.
    5. Run the full hermetic gate + module count. Two commits: (a) `feat(tiling): open ReducedDimWarp by default — fix4 §3.5 split-k (Wave 2 T8)` — the flip + comment updates + the three tiling-package and one e2e-package re-pin (atomic RED→GREEN); (b) `test(cxx): pin the split-k emission golden (fix4 §3.5)` — the new fixture + golden.
  </action>
  <verify>
    <automated>go build ./... && go vet ./... && go test ./... && test "$(go list -m all | wc -l)" -eq 1 && git status --porcelain internal/cg/cxx/testdata/ | grep -c reduce_splitk.cu</automated>
  </verify>
  <done>Default ReducedDimWarp=true; the four Wave-1 policy tests re-pinned GREEN with the split-k selections pinned deterministically; reduce_splitk.cu committed and byte-regenerating; hermetic suite + vet + module-count gates green; no fence widened (rejectWarpReducedDims and reducedIters code untouched; both skip sites still present — `grep -c 't\.Skip(matmulGoldenDesignGapSkip)' internal/e2e/e2e_test.go` == 2).</done>
</task>

<task type="auto">
  <name>Task 2: T8b — dedicated split-k matmul e2e on the A6000 (numerical + sanitizers)</name>
  <files>internal/e2e/frontend_test.go</files>
  <action>
    1. HARDWARE GATE: run `nvidia-smi` and confirm an A6000 is present; if not, STOP and report — this task is a hardware gate, not skippable (iq6 precedent).
    2. Mark the dedication: add the one-line comment on TestFrontendMatmulScale that post-flip it is THE dedicated fix4 §3.5 split-k matmul e2e (M=CTA × N(reduced)=WARP/2 × Q=SEQ; the §3.5 protocol end-to-end through the real pipeline). No fixture restructuring.
    3. Numerical battery: `TCACOMP_E2E=1 go test ./internal/e2e/ -count=1 -run 'TestFrontendMatmulScale$|TestReduceWarpE2E|TestReduce3E2E|TestReduceT1E2E' -v -timeout 45m`. Expect: frontend BOTH variants canary 0, maxAbs ≤ 2e-3; all reduce fixtures green. First cold build takes ~15 s of CMake/nvcc (content-addressed fixture cache).
    4. Sanitizer battery: `TCACOMP_E2E=1 go test ./internal/e2e/ -count=1 -run 'TestFrontendMatmulScaleSanitizer|TestReduceWarpSanitizer|TestReduce3Sanitizer' -v -timeout 60m`. Expect memcheck + racecheck ZERO errors on every runnable fixture — the split-k kernel's cross-warp staging fold (`__syncthreads` inside WarpReducer::Combine) and the warp-varying LoadTileCta window bases are exactly what racecheck arbitrates.
    5. Full-suite close-out: `TCACOMP_E2E=1 go test ./internal/e2e/ -count=1 -v -timeout 60m` — the two golden-matmul tests must still SKIP with the design-gap reason (do not delete the skips).
    6. DEFECT PROTOCOL (iq6 pattern) if any numerical or sanitizer check fails on the split-k path: reproduce at the SMALLEST layer hermetically (a RED pin in tiling/finallower/cxx — e.g. the C1b chunk-local precedent 37993de→218334c), fix MECHANICALLY (lower passes translate verbs only — never sweep beyond the verb's rule), regen/re-pin affected goldens in the same commit, re-run the failed GPU battery. If the root cause is a DESIGN-level uncoverability (the §3.1 CTA×WARP(surviving) class — it should NOT be: the flipped selection's only thread-partitioned walk is the REDUCED dim's lane chunks; the surviving M/Q walks are uniform per-thread element walks inside the bracket), or any fix would be a redesign: STOP, leave the fixture failing/skipped with evidence inline, and report — do NOT adopt iq6 candidate rulings 1/2/3.
    7. Commit only if fixes landed (iq6 T6b precedent: a clean battery is commit-free). Fix commits: `fix(<pkg>): <mechanical defect> in the split-k lowering (fix4 3.5)` each carrying its RED→GREEN pin; re-run the full hermetic gate after any fix.
  </action>
  <verify>
    <automated>nvidia-smi --query-gpu=name --format=csv,noheader | grep -c A6000 && TCACOMP_E2E=1 go test ./internal/e2e/ -count=1 -v -timeout 60m 2>&1 | tee /tmp/t8b-e2e.log; grep -cE '^--- (PASS|SKIP)' /tmp/t8b-e2e.log && ! grep -c '^--- FAIL' /tmp/t8b-e2e.log</automated>
  </verify>
  <done>Split-k matmul numerically correct on the A6000 (both syntax variants, canary 0, within tolerance); memcheck + racecheck zero errors on every runnable fixture; golden-matmul skips intact; any surfaced defect fixed mechanically with hermetic RED→GREEN pins; hermetic repo gate re-green after fixes.</done>
</task>

<task type="auto">
  <name>Task 3: T8c — fix4.md state update + SUMMARY with the interaction record</name>
  <files>doc/fix/fix4.md</files>
  <action>
    1. doc/fix/fix4.md: update the §2.1 Params table row `ReducedDimWarp | false (Wave 1)` → `true (Wave 2 — T8 landed; explicit false = the Wave-1 policy)` (also correct that row's stale `— §3.6` meaning column to `— §3.5` — the spec's §3.6 is a stale alias for §3.5, checker-verified at fix4.md:145) and the staging bullet at §1-fix4 header line 7 ("flip to true in Wave 2 when §3.6 lands" → flipped, T8 landed with the dedicated split-k e2e). Keep edits additions/minimal — fix4.md is the spec of record; do not rewrite §3.5.
    2. Write this directory's SUMMARY.md recording at minimum: the flip + the measured blast radius (4 tests); the pinned split-k selections (decide-side const-extent winner and the frontend runtime-extent winner M=cta/64/256, N=warp/2/32, Q=seq/256, pool 65544 = 65536 + 8); the new golden; the GPU battery results (numbers: canary, maxAbs, sanitizer summaries); and THE INTERACTION RECORD the author needs: (a) with the flag open, the decide side now selects reduced-WARP vectors and the pipeline lowers them per §3.5 — live end-to-end; (b) the coarse reducedIters set (ALL dims referenced by a reduce's piped value, search.go:357) previously shielded reduce-carrying kernels from ANY WARP selection — post-flip, SURVIVING-WARP vectors are also enumerable for reduce kernels, so the unruled iq6 case-b silent-wrong class (CTA×WARP(surviving) uncovered cross product) is unchanged in nature but BROADER IN REACH; candidates 1/2/3 remain open, the golden-matmul skips stay; (c) fences kept: rejectWarpReducedDims and reducedIters code untouched, no surviving-WARP gate added.
    3. Update .planning/STATE.md with the completed quick-task row (mirror the 261027-iq6 row shape: one-liner + gates + handoffs). Commit: `docs(fix4): ReducedDimWarp default flipped — Wave 2 T8 landed (split-k e2e green) + summary`.
  </action>
  <verify>
    <automated>grep -c 'true (Wave 2' doc/fix/fix4.md && ls .planning/quick/261027-kzv-implement-doc-fix-fix4-md-t8-split-k-per/261027-kzv-SUMMARY.md && go build ./... </automated>
  </verify>
  <done>fix4.md reflects the landed Wave-2 default; SUMMARY carries the full interaction record (split-k live; case-b reach broadened, ruling open; fences intact); STATE.md row added; all commits conventional and gated.</done>
</task>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| test harness → toolchain subprocesses | e2e invokes cmake/nvcc/compute-sanitizer via the existing no-shell RunCmd harness (5-min timeouts, content-addressed fixture cache) |
| compiler input | local .tca source fixtures only; no untrusted input crosses into this task |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-kzv-01 | Tampering | emitted .cu + goldens | mitigate | byte-exact golden.Assert (new reduce_splitk.cu) + re-derived hermetic schedule pin; determinism pins (fixed enumeration, first-strict-max) |
| T-kzv-02 | Repudiation | GPU battery outcomes | mitigate | machine-parseable CANARY/CHECKSUM/OUT lines + requireZeroSanitizerErrors (present-and-zero SUMMARY lines); log transcript captured in /tmp/t8b-e2e.log |
| T-kzv-03 | DoS | sanitizer runs on shared GPU | accept | existing 60m test timeout; battery is small (4 fixtures + sanitizers, ~5 s warm) |
| T-kzv-SC | Tampering | package installs | mitigate | none in this task; module count 1 gate (`go list -m all \| wc -l` == 1) enforces stdlib-only |
</threat_model>

<verification>
- Hermetic: `go build ./... && go vet ./... && go test ./...` green; module count 1; goldens byte-stable except the new reduce_splitk.cu.
- Fence audit: `grep -c 't\.Skip(matmulGoldenDesignGapSkip)' internal/e2e/e2e_test.go` == 2 (both skip sites remain); `git diff` shows NO code change in rejectWarpReducedDims (lower.go:270-303) or reducedIters (search.go:357-372).
- Policy pins: TestDefaultParams (true), TestLowerWarpReducedDimRejected (explicit-false fail-closed + default-admits), TestReducedDimWarpPruning (prune under false + split-k winner under default) all green.
- GPU (A6000, TCACOMP_E2E=1): frontend matmul both variants canary 0 / maxAbs ≤ 2e-3; sanitizers zero errors; reduce regression fixtures green; golden matmul still SKIPped by design.
</verification>

<success_criteria>
- fix4 §3.5 is LIVE under the default: a reduce-carrying kernel selects and correctly lowers/emits split-k end-to-end (decide → tile-lower → finallower bracket → WarpReducer emission), pinned hermetically (schedule pin + byte golden) and proven numerically on the A6000 with clean sanitizers.
- The Wave-1 defense survives as the explicit-false opt-out, fail-closed and pinned.
- The iq6 surviving-WARP design gap is untouched, still fail-closed where it was, skips in place, interaction honestly recorded for the author.
- Every commit: hermetic build/vet/test green, module count 1, conventional commit format.
</success_criteria>

<output>
Create `.planning/quick/261027-kzv-implement-doc-fix-fix4-md-t8-split-k-per/261027-kzv-SUMMARY.md` when done
</output>
