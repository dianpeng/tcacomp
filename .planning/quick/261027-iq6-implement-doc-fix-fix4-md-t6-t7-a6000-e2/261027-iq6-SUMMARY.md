# quick-261027-iq6 — fix4 T6 (A6000 e2e + sanitizers) + T7 (doc pointer stubs) Summary

**One-liner:** fix4 §9(1) closed on the real frontend matmul_scale (both
variants, canary 0, within tolerance; §9(3) sanitizers zero errors), ONE
mechanical C1b emission defect found+fixed on the way (chunk-local FRAG
access, memcheck-proven), and a **design-level coverage gap found and
handed off** (fix4 §3.1's CTA×WARP(surviving) composition leaves the m×k
output cross product 63/64 uncovered — the golden matmul fixture is
STOPPED on it); T7 pointer stubs + runtime_objects row updates landed.

## Task results

| Task | Commit | Outcome |
|------|--------|---------|
| T6a schedule pin + numerical battery | b696cd1 (pin) / 37993de (RED) / 218334c (fix) / 3de2de6 (fixtures) | frontend both variants PASS (canary 0, maxAbs ≤ 2e-3); reduce_warp/reduce3/reducet1 PASS; golden matmul STOPPED (design gap, below) |
| T6b sanitizer battery | (no commit — clean) | memcheck + racecheck ZERO errors on reduce_warp, frontend matmul, reduce3, reducet1 (matmul sanitizer skipped with the fixture) |
| T7 docs | 1e8d401 | pointer stubs on loop_tile_lower/rt/cg (additions-only), runtime_objects.md rows updated to the landed API, final_lower.md verified clean (untouched) |

## e2e run log (A6000, final combined run)

`TCACOMP_E2E=1 go test ./internal/e2e/ -count=1 -v -timeout 60m` — 4.77s
(warm fixture cache; first cold-build battery run took ~15s of CMake/nvcc):

```
--- SKIP: TestMatmulScaleE2E (0.00s)          [design gap — see handoff]
--- PASS: TestReduceWarpE2E (0.18s)
--- SKIP: TestMatmulScaleSanitizer (0.00s)    [same stop]
--- PASS: TestReduceWarpSanitizer (0.73s)
--- PASS: TestFrontendMatmulScalePipeline (0.03s)   [hermetic pin incl. new schedule pin]
--- PASS: TestFrontendMatmulScale (0.29s)            [written + omitted, canary 0]
--- PASS: TestFrontendMatmulScaleSanitizer (1.28s)
--- PASS: TestFrontendMatmulScaleLowerNoTailStillVerifies (0.01s)
--- PASS: TestFrontendMatmulScaleFillCompositionScope (0.01s)
--- PASS: TestReduce3E2E (0.14s)   [re-targeted; canary 0, maxAbs 0]
--- PASS: TestReduceT1E2E (0.12s)  [re-targeted; canary 0, maxAbs 0]
--- PASS: TestReduce3Sanitizer (1.99s)  [reduce3 + reducet1, both tools]
PASS  ok  tcacomp/internal/e2e  4.773s
```

Earlier standalone runs: numerical battery first run 14.58s (3 failures ->
diagnosed below), sanitizer battery 7.16s (clean after fixes).

## Defect findings

### 1. [Rule 1 — mechanical emission bug] C1b FRAG access was not chunk-local — FIXED

- **Layer:** tile-lower (internal/opt/tiling/lower_cache.go, access-coords
  construction).
- **Finding:** a C1b forced-FRAG site's WARP-covered dim buffers only the
  lane's chunk (⌈T_d/32⌉ elements, fix4 §3.3), but the access emitted the
  RAW lane-chunk walk IV (values start at lane·c). GPU evidence:
  compute-sanitizer memcheck on the golden matmul reported 241×
  "Invalid __local__ read of size 4" at the `tca_f0.Get` site (every lane
  ≥ 1), and the fault KILLED the kernel under the sanitizer.
- **Fix:** the coord is now chunk-local (`insideIV − lane·⌈T_d/32⌉`) —
  hermetic RED→GREEN pin TestLowerC1bFragChunkLocalAccess; golden emits
  `Get(d2_it5.Index() - v7)`; memcheck after: 0 errors.
- **Commits:** 37993de (RED pin) → 218334c (fix + goldens re-pinned
  in-commit, sanctioned emission change: matmul_store.cu,
  tile_lowered_matmul.ir).

### 2. [DESIGN-LEVEL GAP — STOPPED + HANDED OFF] fix4 §3.1 CTA×WARP(surviving) composition cannot cover the output cross product

- **Affected fixture:** the golden matmul (TestMatmulScaleE2E + its
  Sanitizer) — the crafted fix4 §3.3-class record with K=WARP (K=2, T=32)
  on a dim that SURVIVES into out[m,n,k] = lhs[m,k]·rhs[n,k]. NOT fixed,
  NOT re-targeted — author ruling required (candidates below).
- **Probe evidence (A6000, 2026-10-27, content-histogram runner over the
  exact fixture binary; probe rig preserved at /tmp/iq6-diag):**
  - Run 1 (pre-fix): canary 1,056,964,608 of 2^30 — exactly 2^24 written;
    sync clean. sample[0] correct, samples[1..7] unwritten.
  - Histograms: m coverage COMPLETE and uniform; n coverage COMPLETE
    (all 4 n-rounds); all 64 k-classes uniform; per-(m,n) k-count
    histogram: **every one of the 2^20 (m,n) pairs carries EXACTLY 16 of
    1024 k values** — precisely {32·w(t)+l(t)+64·j : j∈[0,16)}, the k's
    owned by the single thread t whose coupled-walk rank owns m.
  - Root cause: the M=CTA coupled inside walk strides the WHOLE blockDim
    (all 64 threads, both warps — fix4 §3.1 "step = blockDim"), while the
    K=WARP lane chunks assign each k-tile to ONE warp's lanes. Output
    element (m, k) needs a thread that owns m (coupled rank) AND k (warp
    coord): warp 0 covers m_local ≡ [0,32) mod 64 × k ≡ [0,32) mod 64,
    warp 1 the other diagonal block — the off-diagonal blocks have NO
    owner. 63/64 of every (m,n) k-row is never computed.
  - Counter-hypothesis PROBED AND REFUTED: per-warp coupled walk
    (init lane_id, step 32) raises coverage to only 32/1024 — the hole
    repeats at the lane level (m partitioned over lanes × k partitioned
    over lanes = diagonal only). **Any two thread-partitioned inside
    walks leave the surviving cross product uncovered.**
- **Consequence:** in Wave 1 a WARP dim in a lowered LNT is either (a)
  reduced → the §3.5 fail-closed gate correctly blocks it (this is what
  fired on the pre-re-target reduce3/reducet1 crafted records), or (b)
  surviving → uncovered cross product with the mandatory CTA dim (CuMAP)
  → silent wrong code. fix4 §3.3's own worked example (K=WARP on the
  REDUCED k) is therefore Wave-2-dependent for LOWERING (split-k §3.5),
  consistent with the fail-closed gate; the golden's crafted record
  applied that shape to a NON-reduced k — a combination no gate rejects
  and no template covers. NOTE: the decide pass can still SELECT
  case-(b) schedules for real kernels — silent-wrongness risk until
  ruled.
- **Candidate ruling shapes (handoff):**
  1. **Wave-1 legality ruling:** WARP dims legal ONLY on reduced dims —
     prune surviving-WARP from the decide feasible area + a tile-lower
     fail-closed mirror (the §3.5 pattern, dual direction). Golden
     re-targets to CTA+SEQ; WARP GPU coverage rides Wave-2 split-k.
     Cheapest; matches §3.5's direction.
  2. **Template redesign:** make the CTA dim's inside walk coverable
     alongside WARP dims (e.g. warp-sliced whole-slice walks — each
     thread walks its warp's m-slice wholly × its lane's k-chunk;
     exactly-once coverable) — changes §1.3/§3.1 templates, the
     coupled-walk coalescing story, judge costs, and the 2026-10-23
     flat-index ruling's init. Not mechanical; needs the author.
  3. **Coverage verifier:** regardless of 1/2, a tiling-verify
     cross-product coverage check (every surviving dim pair ≤ 1
     thread-partitioned inside walk) would convert this silent-wrong
     class into a fail-closed error.

### 3. [Plan-anticipated re-target] reduce3/reducet1 crafted records were Wave-1-illegal

- Their hand-crafted records placed WARP on the REDUCED dims — records
  the decide pass cannot produce under ReducedDimWarp=false, and which
  the T4 §3.5 fail-closed gate correctly rejected (exact error text in
  the run log). Re-targeted to SEQ per the T5 deviation precedent
  ({CTA,100,32}/{SEQ,1,64}/{SEQ,1,64} and {CTA,100,2}/{SEQ,1,2}) — the
  fixtures keep pinning the ≥2-between bracket law, SeqReducer numerics,
  and the minimal-legal-tile floor on the GPU (canary 0, maxAbs 0).
  NOTE: the plan's fixture-table intent "WARP-assigned NON-reduced dim
  feeding a reducer" is NOT achievable in Wave 1 (finding 2's hole);
  documented in reduce3_test.go's header. WarpReducer face stays pinned
  by runtime/tests/conformance_new_api.cu.

## Deviations from plan

1. **Golden matmul STOPPED, not green** (finding 2) — per the plan's own
   DESIGN-LEVEL-gap contingency ("STOP that fixture, record probe
   evidence + candidate ruling shapes ... do NOT redesign in this
   task"). §9(1)'s "matmul_scale e2e numerically correct" is carried by
   the REAL frontend matmul_scale fixture (the plan's own §9(1) wording),
   which is green; the golden matmul_store fixture is a separate
   elementwise-store kernel.
2. **reduce3/reducet1 re-targets** (finding 3) — the sanctioned T5
   deviation class; plan anticipated Wave-1 re-targets.
3. **One extra hermetic pin beyond the plan's schedule pin**
   (TestLowerC1bFragChunkLocalAccess) — the plan's defect protocol
   requires a RED→GREEN pin at the smallest reproducing layer for any
   compiler-side emission defect.
4. Plan's fixture-table WARP-coverage expectations on the GPU were
   written pre-probe; Wave-1 runnable schedules are CTA+SEQ only
   (finding 2) — the table's lane-chunk claims ride Wave 2.

## Doc stub inventory (T7, commit 1e8d401)

- loop_tile_lower.md (6 stubs): §3 canonical forms→fix4 §3.1/§1.2; §6 R2
  window sizing→§3.3/§2.4; §6 tier-rule C1b sizing→§3.3; §6bis PHASE
  3→§3.1 (PHASE 0/1/2 + §6 mechanics survive per §3.4 — NOT stubbed);
  §6bis PHASE 4 sizing→§3.3 (+ budget resolution = unreachable
  backstop); §7 v0 searcher policy→§2.2; §7bis counting invariant→§1.3,
  portion grammar→§4.1, launch validation→§4.3/§5.1.
- rt.md (4 stubs): §3 conformance rows→goldens+conformance_new_api.cu;
  §4.2 portion decomposition→§4.1; §4.3 iterator loc→§4.2; §5
  ValidateLaunch→§4.3.
- cg.md (4 stubs): §0 D9/D11 MaxThreadsPerCTA rows marked retired→§2.1/
  §5.1; §6 mapping-ctor rows→§5.1 shapes; §6 C1b chunk-local access
  note; §9 launch derivation→§5.1.
- final_lower.md: verified free of pre-fix4 text (T5 reworked §6) — left
  untouched, per plan.
- runtime_objects.md (living table, ROW UPDATES): inventory rows 1-11,
  23-24, 27-28, 40-43; §1 launch contract (derived dims, MaxCTAPolicy
  512, 4-arg ValidateLaunch, portion tombstone); §2 landed ctors +
  zero-extent + launch-derived totals; §2.3 conventions (coupled walk
  flat+blockDim, lane chunks, C1b chunk-local); §5.4 annotation table
  (absolute LoC/warp-unit K_d/blockDim locs); §6.1/6.2/6.6; §10.4; §11;
  Appendix A/B to golden #163/#165/#166.
- Gates: 4/4 docs carry fix4 pointers; stubbed docs 0 deletions
  (additions-only verified); grep -c ndim runtime_objects.md = 10.

## Gates

- Hermetic `go build ./... && go vet ./... && go test ./...` green at
  every commit; module count 1.
- Goldens byte-stable except the sanctioned Task-1 re-pins (committed
  with the fix + hermetic pin backing).
- Scope fence: ReducedDimWarp default false (record.go), untouched; no
  Wave-2 changes; grep confirms only the pre-existing gates/comments.
- Sanitizers: memcheck + racecheck zero errors on every runnable
  fixture (requireZeroSanitizerErrors enforces present-and-zero SUMMARY
  lines — T-iq6-03 mitigated).

## Known stubs / open items

- The STOPPED matmul fixtures carry `t.Skip(matmulGoldenDesignGapSkip)`
  in internal/e2e/e2e_test.go — one line to delete once finding 2 is
  ruled and landed.
- The decide-side feasible area can still select uncoverable
  CTA×WARP(surviving) schedules (silent-wrong risk) — candidate 1/3
  above address it; needs the author.
- Wave-2 T8 split-k (§3.5, ReducedDimWarp=true) unchanged, as fenced.

## Self-Check: PASSED

- Commits on worktree-agent-a8dd8cf73e2c59f01: 37993de, 218334c, b696cd1,
  3de2de6, 1e8d401 (git log verified).
- Artifacts: 261027-iq6-SUMMARY.md present; the hermetic pin's required
  literal `ValidateLaunch(dim3(64u, 1u, 1u), dim3(32u, 1u, 1u), 0u, 512u)`
  present in internal/e2e/frontend_test.go.
- Duration: ~46 min (05:43:33Z → 06:29:32Z UTC), including GPU batteries
  and sanitizer runs.
