# Quick Task 261025-of7 — Implement doc/fix3.md (memory IR = three verbs; derived cache placement; mapping-driven loader)

**Status:** COMPLETE — all 9 §4 steps + one Rule-1 pre-fix landed (10 commits); step 9 finished by the orchestrator after the executor died to three consecutive API failures (quota cap 429 ×1, ECONNRESET ×2 — no work lost; every resume resumed from a green commit)

## Commit ladder (step → commit → gates)

| # | Step | Commit | Gates |
|---|------|--------|-------|
| 0 | (Rule 1 pre-fix) memopt rule (c) never sweeps a returned output store | `4ac9a0a` | build/vet/test green; goldens byte-identical; GPU e2e green |
| 1 | D-n perf half: loader row-striped enumeration, vec128 + cp.async, scalar retained | `28d63ca` | build/vet/test green; module count 1; goldens byte-identical; CPU layer 2429/0; GPU e2e green |
| 2 | D-h memory IR unification (9 kinds → 3 verbs + storage class) | `d3293a2` | build/vet/test green; module count 1; goldens byte-identical; six-kind/attr grep clean; astdot ported (in-commit) |
| 3 | D-i fill defusion (New + GMEM tile load + class store composition) | `556a660` | build/vet/test green; goldens byte-identical; derivation==mandate safety net green on all three fill shapes |
| 4 | D-j derived placement (closes fix2 D-c) | `2bf84f5` | build/vet/test green; goldens byte-identical; **TestFrontendMatmulScale passes NUMERICALLY on the A6000 (both variants) — informational GPU run at this step** |
| 5 | D-k memopt generalization (rule (d) folded into the generic rule) | `f32031c` | build/vet/test green; goldens byte-identical; wiring pin proves the generic rule fires on the composition AND a non-fill-shaped tiered store |
| 6 | D-l R-I scope invariant as a verifier check | `eabb3ea` | build/vet/test green; goldens byte-identical; negative pin: stranded consumer fails VERIFY |
| 7 | D-o const exile + TAIL dedup (golden regen 1: reduce_warp) | `dd465ca` | build/vet/test green; module count 1; matmul_store BYTE-IDENTICAL (3e5aa73a…); reduce_warp re-pinned **e82136e93138099f5c43d10f87b14ca5** (the C11-sanctioned drift: the fixture drives its crafted chain without the tail) |
| 8 | D-m/D-n API half: mapping-driven loader call + codegen fusion (golden regen 2: matmul_store) | `145716b` | build/vet/test green; module count 1; CPU layer 2429/0; **GPU e2e green through the new call**; matmul_store re-pinned **229f0eac88ac3b68ac64f7ac2633305b** (exactly the one fill line); reduce_warp at its step-7 pin |
| 9 | D-p Reduce anchor gate + frontend e2e un-skip + battery | `8e0bedf` | build/vet/test green; module count 1; goldens at the D-o/D-m pins; GPU e2e green 32.8s incl. un-skipped TestFrontendMatmulScale (both variants, canary 0) + memcheck/racecheck zero errors; bracket-law pin both variants; grep gate clean |

## Step-1 loader timing measurement (mandated by D-n; one-off bench, not committed)

256×96 f32 SMEM tile (98,304 B ≤ the sm_86 101,376 B opt-in ceiling), 1 CTA × 512
threads, 20,000 fills with rotating row bases (real traffic, not L1-resident
refills), `-O3 -arch=sm_86`:

| Implementation | µs/fill | effective GB/s (read+write) |
|---|---|---|
| old (`copy_window_elem` per-element div/mod + bounds branch) | 9.44 | 20.8 |
| new (row-striped + vec128 + cp.async) | 3.90 | 50.5 |

**2.42× faster.** Correctness cross-check inside the bench: the new loader's
output is byte-identical to the old over a clamped window (row base 448, col
base 1000 → 24-wide valid tail, leading rows in range) — the zero-fill/suppression
semantics match exactly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] TAIL memopt erased the frontend kernel's output store (pre-existing at base — a 10th commit outside the 9-step ladder)**
- **Found during:** Task 1, step 1's GPU gate (`TCACOMP_E2E=1 go test ./internal/e2e/` failed at LINK)
- **Issue:** `pkg/builder.go Build` wires the outputs' stores as the Return's DATA
  operands and the Return's chain edge observes each output tensor's final access.
  tile-lower's P6 `repointChains` legitimately re-points the end node's
  body-internal edges at the nest root — orphaning the output store's region.
  fix2's TAIL memopt instance then applied rule (c) (no effect successor), RAUW'd
  the Return's data use away, and erased the store: the emitted kernel carried no
  output writes, its launcher const-qualified the output tensor, and the e2e
  runner (which declares a mutable output) no longer linked. Latent since fix2
  (the fix2 session never ran the GPU gate; the golden tests use
  `opt.BuildPipeline()`, which has no TAIL).
- **Fix:** rule (c) skips any store the end node holds as a data operand — the
  output-binding convention; observable by contract (INV-D's data-edge reading).
  Regions and rules (a)(b)(d) untouched. A first attempt (preserving the end
  node's chain edges in `repointChains`) was rejected: P6's re-pointing is
  load-bearing for sea-reachability of loop nodes AND the scheduler's end-node
  epilogue placement (8 test pins fail without it).
- **Files modified:** `internal/opt/memopt/memopt.go`
- **Commit:** `4ac9a0a`

**2. [Rule 3 - Blocking] cmd/astdot/main.go ported into the worktree (D-h in-commit)**
- **Found during:** Task 1, step 2 — the D-h verify gate (`grep` over `cmd/`) covers the main
  tree's untracked user WIP; unported, the user's main-tree `go build ./...` breaks post-merge.
- **Action:** ported `/home/ps/Workspace/cuda/project/tcacomp/cmd/astdot/main.go` verbatim with
  ONLY the mechanical kind rename (its two `KindMemLoad, KindMemLoadSMem` / `KindMemStore,
  KindMemStoreSMem` label cases collapsed to the unified kinds), committed inside `d3293a2`
  per the dispatch instruction.

**3. [Scope note] ir-print testdata regenerated at D-h**
- `internal/opt/tiling/testdata/tile_lowered_matmul.ir` (an IR debug-print golden, NOT one of
  the two C11 `.cu` goldens) re-pinned for the new class-suffixed spellings
  (`memstore[frag](...)` etc.) and the one-node-lighter fill (the loader attribute is gone).
  The two C11 goldens stayed byte-identical.

**4. [Rule 3 - Blocking] cxx golden fixtures moved to the real composition (D-o)**
- **Found during:** Task 3, step 7 — with the dedup exiled from final-lower, the golden tests'
  `opt.BuildPipeline()` (registry catalog, no TAIL) path stopped deduping and matmul_store
  drifted — which would have forced an out-of-sanction regen.
- **Fix:** `runPipeline` (emit_test.go) now drives `pipeline.Run` (HIR + LOWER + TAIL) — the
  composition the facade actually wires; the goldens pin the real user path. matmul_store came
  back byte-identical (proof the tail dedup reproduces the old canonicalization for it);
  reduce_warp re-pinned per the C11 sanction (its fixture drives a crafted record chain that
  bypasses the tail, so its measured duplicate-const drift remains by construction).
- **Files:** `internal/cg/cxx/emit_test.go`, `internal/pipeline/pipeline.go`
- **Commit:** `dd465ca`

**5. [Interpretation] step-9 grep gate scoped to the wall-skip text (D-p)**
- The plan's literal automated gate (`! grep -rn "t.Skip" internal/e2e/`) contradicts the
  by-design hermetic skips (design.md/CLAUDE.md: the `TCACOMP_E2E=1` env gate and the
  tool-absence skips in harness.go). Implemented as intended: the 2026-10-24
  "known downstream wall" skip text is gone from internal/e2e (grep for
  "known downstream wall"/"needs its own ruling" is clean); the env/tool skips remain
  (they gate availability, not known defects).

**6. [Rule 3 - Refinement] negative (b)'s severing shape (D-p)**
- The battery's "hand-severed Reduce chain" cannot be a literal const value: a
  loop-invariant producer is structurally rejected EARLIER by final-lower's LR-1
  bracket check ("the piped value's producer sits outside the home loop"), never
  reaching the anchor. The landed pin severs the anchor's OWN predicate instead: the
  producer is a register-only expression over the home IVs (structurally sound,
  inside home, every bracket check green) whose closure reaches NO memory access —
  with a chained lhs load alive in the epilogue consumer only, proving the anchor's
  liveness never crosses memory-chain edges (the load is in the kernel, effect-chained,
  and correctly invisible to the anchor).

## Stops

None so far.

## Open items (carried)

- load→load forwarding: expressible after D-k, stays UNIMPLEMENTED (fix3 §5).
- simplify/lower.go Sweep 3 + the lower-out-of-opt refactor: recorded follow-ups.
