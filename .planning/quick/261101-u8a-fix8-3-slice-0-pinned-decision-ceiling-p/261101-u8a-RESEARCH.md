# fix8 §3 Slice 0 — Pinned-Decision Ceiling Probe: Research

**Researched:** 2026-11-01
**Domain:** tiling-decision injection + e2e emission + bench harness (measurement-only probe)
**Confidence:** HIGH (every claim below verified by reading the live code this session; row_sum emitted from source at HEAD as ground truth)

## Summary

The probe's central question — can the existing Params knobs alone (vehicle a) pin grid ∈ {16,64,128,256,512} — answers **NO, for two independent reasons**. First, the knobs are unreachable: `pipeline.Run` builds every pass from the registry factories, and all three tiling-family factories (`tiling.go:35`, `verify.go:39`, `lower.go:60`) hard-wire `DefaultParams()`; `cmd/tcacomp` exposes only `-o` (`cmd/tcacomp/main.go:64`). Second, even if reachable, presets alone cannot pin: the R1 adaptive digit (`constraints.go:108`) re-introduces smaller LoC values at every tile, and the n63 occupancy factor actively *prefers* them — with `NominalExtent=4096`, a pinned 512-CTA vector scores idleFrac ≈ 0.97 (activeGrid = min(512, ceil(4096/256)) = 16) and loses ~32× to the 16-CTA vector at equal traffic. The pin therefore requires **three coordinated Params overrides per point**: `CtaLoCPresets=[N]`, `MaxCTA=N` (kills the adaptive clamp above N and enforces Π CTA LoC ≤ N), and `NominalExtent ≥ N × maxTile` (forces adaptiveLoC = N at every tile, deduping into the preset, and zeroes idleFrac so the fill term monotonically prefers the pinned grid).

**Primary recommendation:** vehicle **(b)** — a test-only env-gated Params overlay (`paramsFromEnv()`) in a new `internal/opt/tiling/params_env.go`, called at the three existing `DefaultParams()` factory sites. Zero default-path change, provable by sha256 manifest vs the 1c emissions with env unset. Drive the sweep through the **real** compiler facade and the **existing** orchestrator fixture path (`subprocess.run` inherits env; the content-hash cache key includes the emitted `.cu` bytes, so each pin point gets its own fixture with no collision).

**Headline measurement-model finding (STOP-rule candidate, discovered during this research):** the CTA round walk is grid-stride (`CtaIterator(map, dosoc, loc=LoC, init=0, step=LoC×T, bound=extent)`; tile origin = `cta_coord × T` — verified in the emitted row_sum: `CtaIterator(grid_map0, 0u, 16u, 0u, 4096u, v0)` where 4096 = 16×256), and excess CTAs clamp to zero work through the saturating `umin` chain (`v8 = umin(T, extent − min(cta×T + round, extent))` → ThrIterator bound 0). **Today's row_sum (R=CTA T=256 LoC=16 over runtime R=512) has only ceil(512/256) = 2 active CTAs of 16 launched — 64 active threads of 512 launched.** The occupancy model's activeGrid stand-in (NominalExtent=4096) credits 16 active — an 8× overstatement on this kernel. The probe's x-axis must record launched AND active threads; a large part of the 0.47%-of-BW figure may be explained by this alone, before any decision pinning.

## Vehicle Decision: (b) — env-gated Params overlay, applied at the three factories

**Evidence chain [VERIFIED: codebase, this session]:**

| # | Fact | Source |
|---|------|--------|
| 1 | Pipeline constructs passes only via `opt.Lookup(id).Factory()` — no pass injection, no Params threading | `internal/pipeline/pipeline.go:84` |
| 2 | Decide pass factory: `Pass{Params: DefaultParams()}` | `internal/opt/tiling/tiling.go:35` |
| 3 | Verify pass factory: `VerifyPass{Params: DefaultParams()}`; comment mandates mirroring the decide Params | `internal/opt/tiling/verify.go:39-46` |
| 4 | Tile-lower factory: `LowerPass{Params: DefaultParams()}`; "Params must mirror the decide pass's thresholds" | `internal/opt/tiling/lower.go:60-68` |
| 5 | CLI surface is only `-o` | `cmd/tcacomp/main.go:61-67` |
| 6 | Verify re-checks the record against **its own** Params: `locPresetOK(d.SoC, d.LoC, d.TileSize, effExtent, v.Params)`, Π LoC ≤ MaxCTA, budgets, judge cross-check | `internal/opt/tiling/verify.go:196-220,260-287` |
| 7 | Tile-lower performs **no** preset/LoC re-check (no `locPresetOK`/`MaxCTA` hits in `lower*.go`) — the `Verified` stamp gates it | grep over `lower.go`, `lower_cache.go`, `lower_fragchunk.go` |
| 8 | Env-gating is an established house pattern (`TCACOMP_E2E`) | `internal/e2e/` |

Consequences:
- A **decision-override** hook (override `record.Iters` after `searchSchedule`) would be caught by verify's `locPresetOK` for grid > 128: with default Params the legal CTA LoC set is presets {2..64} ∪ adaptive(NominalExtent=4096) = {2..64, 16/32/64/128 by tile} — **256 and 512 are not expressible under default Params at all**. The hook must touch Params anyway; touching Params at the factories keeps search and verify in lockstep *by construction* (both read the same env), with no new override semantics — decisions stay search-chosen, which is the honest "what the current model delivers" measurement.
- Vehicle (c) (throwaway branch) buys nothing over (b): the orchestrator can consume a dev-built `bin/tcacomp` with env vars exactly as easily as a branch binary, and (b) keeps the probe reproducible for slice-1 comparisons. Recommendation: land (b) on dev behind the zero-diff sha256 gate; if the author prefers, the single hook commit can be reverted after the probe (the measurement record does not depend on the hook surviving).

**Exact hook shape (the executor's Task 1):**
- New file `internal/opt/tiling/params_env.go`: `func paramsFromEnv() Params` — starts from `DefaultParams()`, overrides from `TCACOMP_TILING_CTA_LOC` (comma list), `TCACOMP_TILING_TILES` (comma list), `TCACOMP_TILING_MAX_CTA`, `TCACOMP_TILING_NOMINAL_EXTENT`, optionally `TCACOMP_TILING_WARP_LOC`. All-unset → field-identical to `DefaultParams()`. Malformed values → named error (fail-fast at the boundary, house style); `Validate()` still runs in each pass.
- Three one-line factory edits calling `paramsFromEnv()`.
- Unit pins: env unset → `reflect.DeepEqual(paramsFromEnv(), DefaultParams())`; each var set → override applied; malformed → error. Plus one emission pin (e.g. pinned 512 on row_sum emits `dim3(512u…)` + schedule comment `R=CTA(T=…,LoC=512)`).
- Zero-diff gate: with env unset, re-emit all 14 suite kernels and sha256-compare against the 1c manifest (`after-sha256-manifest.txt` in the 261101-oqg task dir) — must be bit-identical.

## Decision → Launch-Dims Map (what exactly gets pinned)

[VERIFIED: `internal/cg/cxx/emit.go:203-239` (`deriveLaunch`), `walk.go:307-332` (schedule comments), emitted row_sum]

| Decision field | Flows to | Mechanism |
|---|---|---|
| `IterDecision{SoC: CTA, LoC: N}` per dim | **gridDim = Π LoC over CTA dims** (per LNT; max across LNTs) | ASTLoopMeta.Loc on CTA betweens → literal `dim3(N)` in both `ValidateLaunch` and `<<<>>>` (`launch.go:145,151`) |
| `IterDecision{SoC: WARP, LoC: K}` on the one warp dim | **blockDim = K × 32** | warpProd over WARP betweens with `Dosoc >= 0` (CTA-coupled insides carry Dosoc −1 and contribute nothing) |
| `IterDecision.TileSize` | chunk shape | window factors → `siteWindowBytes` (judge.go, the ONE shared source) → halving loop in `lower_fragchunk.go:168-186` lands the per-thread chunk under min(raw, MaxFragPerThread=256 B); `snapHalfChunk` biases SEQ halves to 4-float multiples |
| `cg.Options.MaxCTAPolicy = 512` | `ValidateLaunch` 4th arg | constant in `cmd/tcacomp/main.go:24` — independent of tiling.Params; grid 512 passes (≤ check) |

The emitted `.cu` is **self-describing per point**: header comment `// RowSum: launch 16 CTAs x 32 threads …` plus the LNT schedule line `// LNT #0: R=CTA(T=256,LoC=16) C=sequential … R=thread/warp(T=256,LoC=32) …`. The probe reads each point's actual decisions from these lines (never assumes) and asserts the intended grid.

## The Pin Recipe (per sweep point)

Why three knobs and not one [VERIFIED: `constraints.go:108-137`, `search.go:331-403`, `search.go:126-134` + score' arithmetic in `evalSchedule:302`]:

1. `CtaLoCPresets = [N…]` — the menu of CTA LoC digits. Single entry `[N]` for a 1-CTA-dim point; `[a,b]` (a×b=N) to make a 2-dim factorization expressible.
2. `MaxCTA = N` — without it, the R1 adaptive digit `clamp(ceil(NominalExtent/T), 2, MaxCTA)` re-adds large digits at every smaller sweep point and the search jumps to the biggest. With MaxCTA=N the adaptive digit is clamped into the pin.
3. `NominalExtent ≥ N × max(TileSizes)` — forces `adaptiveLoC = N` at every tile (dedup: presets win), and makes `activeGrid = grid` (idleFrac = 0) so the occupancy factor's fill term, `min(1, active/capacity)`, is monotone increasing in grid. Traffic is LoC-neutral for this family (window bytes are tile-based; FRAG holders = blockDim = 32, unchanged), so at equal traffic the max-grid vector wins and the pinned grid is *chosen*, not merely legal. E.g. N=512 → NominalExtent = 131072; N=16 → 4096 (already the default).
4. `TileSizes` (optional, per sub-point) — pins the chunk-shape axis; entries ≥ 2 (T=2 is legal — `record.go:492-495` only requires ≥ 2). Required for fully-active big grids on axpy3/dup_expr (see expressibility table).
5. `WarpLoCPresets = [2]` (optional "thread-spread" point) — the eliminated dim (K, const 4/8) is the only warp-eligible dim (§3.1 prune, `search.go:561-586`); R-LOCCAP caps warp K at `max(2, ceil(K_extent/T))` = 2, so K=2 → blockDim 64 is the only expressible spread. Note this flips the kernel to the split-k shape (WarpReducer machinery — landed and GPU-proven) and R4 tierEff cuts its traffic credit to 1/16 (it will not be chosen unless warp presets are pinned to the exclusion of SEQ on that dim — pin `[2]` does exactly that). Treat as an optional secondary axis, one or two points only.

**Pitfall the recipe defuses:** with only presets pinned and default NominalExtent, the search picks the adaptive 16@T=256 vector over a pinned 512@T=256 vector by ≈ 32× on score' (idleFrac 0 vs 0.97). Every point's emission must be checked against the intended grid; if the search lands elsewhere, the point is re-labeled from the `.cu` and the curve plotted against measured threads.

## Expressibility of the Sweep Points (fixture dims are runner-baked)

[VERIFIED: `testing/runners/row_sum.cu:14` (dR=512, dK=4, dC=1024), `axpy3.cu:14` (512,512,4), `dup_expr.cu:16` (256,256,8); active-CTA math = ceil(extent/T) per CTA dim]

Fully-active requires Π min(LoC_d, ceil(extent_d/T_d)) = grid. Dynamic dims R,C are runtime extents (round bound = the runtime dim value, verified in the emitted CtaIterator bound `v0`).

| kernel (R×C, K const) | grid 16 | 64 | 128 | 256 | 512 (max expressible) |
|---|---|---|---|---|---|
| row_sum (512×1024) | R16/T32 ✓ | R16×C4/T32 ✓ | R16×C8/T32 ✓ | R16×C16/T64 ✓ | R16×C32/T32 ✓ (or single-dim C512/T2) |
| axpy3 (512×512) | R16/T32 ✓ | R16×C4/T32 ✓ | R16×C8/T32 ✓ | R16×C16/T32 ✓ | needs T≤16 (R32×C16/T16) — TileSizes override |
| dup_expr (256×256) | R8... R16 needs ceil(256/T)≥16 → T16 | R16×C4/T16 | R16×C8/T32? ceil(256/32)=8 ✓ | needs T≤16 | needs T≤4 (R64×C8/T4) — TileSizes override |

All 512-CTA points are legal (MALoC Π LoC ≤ MaxCTA=512; ValidateLaunch grid ≤ MaxCTAPolicy=512 — boundary inclusive). Partially-active variants (default tiles, over-provisioned grid) are ALSO legitimate points — plot them against *active* threads; they directly measure the over-provisioned-mapping semantics the STOP rule names. Recommended sweep: primary = presets/MaxCTA/NominalExtent pins with default tiles (the search's own tile/dim choice at each grid), secondary = small-tile fully-active variants at 256/512, optional = one warp-K=2 (blockDim 64) point.

## Emission + Bench Path (existing machinery, no new driver)

[VERIFIED: `testing/orchestrator.py` read this session]

- **Compile:** orchestrator runs `bin/tcacomp compile -o <stage> <kernel.tca>` (`orchestrator.py:212-214`); `subprocess.run` without `env=` inherits the environment → the env pins flow through the real compiler facade (`compiler.CompileKernel` → `pipeline.Run` → registry factories). Prebuilt preference order: `--tcacomp` override → `bin/tcacomp` → `go run ./cmd/tcacomp` (`orchestrator.py:79-87`).
- **Cache safety:** the fixture cache key hashes the **emitted .cu bytes** + runner + common.cuh + CMakeLists (`orchestrator.py:246-252`) — every pin point is a distinct key; no collision, free rebuild management across the sweep.
- **Build:** content-hash cmake fixture (`cmake_template.txt`, nvcc, `-DTCACOMP_RUNTIME_DIR`), 300 s subprocess timeout (ample: pinned kernels run µs–ms).
- **Run/bench:** fixture binary invoked with `--bench 20`; runners emit `BENCH_US avg min max` (`common.cuh:87`); `bench_bytes` from suite.toml gives GB/s (row_sum 18,874,368 B; axpy3 9,437,184; dup_expr 4,456,448 → 768 GB/s lines at 24.6 / 12.3 / 5.8 µs).
- **Protocol:** `--filter <name> --bench-iters 20 --jobs 1`; memcheck via `--sanitizer memcheck` on schedule-changed points (§7 requires it — every pinned point IS schedule-changed).
- **Runner binding:** runners call the D-fix6-02 const-dim wrapper (`RowSumLaunch(uint32_t e4, …)` — the prepended `e<extent>` param); fixture dims are baked in the runner — the probe measures at fixed shapes regardless of pinning. No runner changes needed.

**Probe driver** (small script in the task dir, not committed to testing/): loop pins × kernels × 4 repeats; per point capture BENCH_US, the schedule comment + launch line from the emitted `.cu`, sha256 (manifest discipline), and compute (launched, active, µs, GB/s, × vs BW line). Emit each point once via `bin/tcacomp` to a pinned dir for the manifest, then let the orchestrator do build/run/bench.

## Common Pitfalls

1. **Stale prebuilt** — `scripts/build.sh` first (gitignored `bin/tcacomp` shadows source; documented gotcha, and fix8 §7 item 1).
2. **Over-provisioned CTA semantics** — grid-stride rounds (`step = LoC×T`), excess CTAs clamp to zero via the `umin` chain (v8=0 → zero ThrIterator trips → zero fills). No duplicate traffic, no OOB (historically memcheck-clean). But **today's row_sum already runs 2-of-16 CTAs active** (T=256 over R=512) — always record active threads; a "512-CTA" point at big tiles may be 2 active CTAs in disguise. This is the first STOP-rule candidate, now with a concrete mechanism.
3. **Fill traffic vs grid** — exact partition under grid-stride: each element loaded once per epoch regardless of grid; idle CTAs issue nothing. Chunk reuse per fill is budget-bound (256 B/thread), not grid-bound. No scaling hazard expected — if measurement contradicts, that's STOP-rule candidate 2.
4. **Occupancy stand-in mis-models activity** — `NominalExtent=4096` uniform stand-in credits activeGrid=16 on today's row_sum where reality is 2. The probe's NE override is a measurement instrument (the model input the fixtures' true extents justify), not a model change; slice 1 may want per-dim true extents (fix6 §1 specialization) — record as evidence.
5. **Bench protocol** — `--jobs 1` only (parallel fixtures contend 4–6× on the A6000); 4 runs, honest tables, min-µs is the stable statistic; Chrome GPU spikes inflate averages on bit-identical binaries (1b/1c precedent). Re-measure the UNPINNED point in the same session — do not ratio against the 2026-11-01 table across machine-state drift.
6. **Warp-K=2 point** changes reducer machinery (SeqReducer → WarpReducer, split-k shape): numerics order changes, maxdiff may differ from the pinned-32 points — record honestly, keep within suite tolerance.
7. **Validate floors** — TileSizes/preset entries ≥ 2, MaxCTA ≥ 2, OccEps ∈ (0,1] (`record.go:485-558`); T=2 is legal, T=1 is not.

## STOP Rule Arithmetic

Baseline row_sum 5,238 µs (1c min). 10× → the best pinned point must beat 523.8 µs. §1's arithmetic estimate at 512×32 = 16,384 threads is 40–80 µs (BW line 24.6 µs). If no point beats 523.8 µs, the expressibility analysis is wrong — stop, re-derive against candidates 2/3/4 above (now concretely: active-vs-launched accounting, fill scaling, ThrIterator spread — the coupled walk strides blockDim=32 within a T-sized tile, so per-thread MLP is bounded by T/32 independent loads; small-T points have low per-thread MLP by construction, which the sweep's tile axis will expose directly).

## Assumptions Log

| # | Claim | Section | Risk if wrong |
|---|---|---|---|
| A1 | Score' monotonicity: with NE raised, the search *chooses* the pinned max grid (traffic LoC-neutral for this family) | Pin recipe | Point lands at a smaller grid than intended — defused by reading each point's emission and re-labeling; no re-plan needed |
| A2 | 2-dim pins (`presets=[a,b]`) resolve to a×b rather than a single dim — relies on fill/partism preferring the bigger grid | Expressibility | Same defusal as A1; assert from the `.cu`, adjust presets |
| A3 | Hook lands on dev (vs throwaway branch) | Vehicle | Author-preference only; measurement identical either way |

## Open Questions

1. **Hook disposition** — land `paramsFromEnv` on dev behind the sha256 zero-diff gate (recommended; reproducible for slice 1) vs throwaway branch reverted after the probe. Author call at plan time.
2. **Sweep density** — how many tile sub-variants per grid point (recommend: free default tiles at every point + 2–3 pinned-T variants at 256/512 + one warp-K=2 point; ~20–30 emissions total).
3. Whether to include a **grid > 512 probe** — out of scope by §5 (MaxCTAPolicy revisit rides slice-0 data only if a point is clearly launch-bound at 512).

## Sources

- Codebase (authoritative, read this session): `internal/opt/tiling/{record,constraints,search,tiling,verify,lower,lower_fragchunk}.go`, `internal/pipeline/pipeline.go`, `internal/compiler/compiler.go`, `cmd/tcacomp/main.go`, `internal/cg/cxx/{emit,launch,walk}.go`, `runtime/tca/iterator.h`, `testing/{orchestrator.py,suite.toml,runners/{row_sum,axpy3,dup_expr}.cu,runners/common.cuh}`, `scripts/build.sh`
- Emitted ground truth: `go run ./cmd/tcacomp compile testing/kernels/runtime/row_sum.tca` at HEAD (schedule comments, CtaIterator args, launch line, fill structure)
- Design: `design/fix/fix8.md` (§1 evidence, §2 defect, §3 slice 0, §6 scope, §7 protocol), `design/fix/fix7.md` (§3.3 guard, §9 protocol)
- Baseline numbers: `.planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/` (1c SUMMARY + after-bench + sha256 manifest)

**Environment verified this session:** nvcc 12.2, cmake 4.4.3 (`/home/ps/.local/bin/cmake`), compute-sanitizer present, NVIDIA RTX A6000 49140 MiB visible, orchestrator flags confirmed (`--filter/--bench-iters/--jobs/--sanitizer/--tcacomp`). No missing dependencies; no fallbacks needed.
