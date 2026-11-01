# fix8 — Parallelism buying: an occupancy-directed tiling search (the thread-starvation fix)

- Status: PROPOSAL — recorded 2026-11-01 from the post-1c external audit (the SOTA/ceiling session). **Slice 0 (the measurement probe, §3) LANDED 2026-11-01 (quick-261101-u8a): the STOP rule FIRED at 1.00× — §1's expressibility analysis is contradicted; slice 1 is BLOCKED until the §3 candidate mechanisms are re-derived against the measured record.** Slice 1 (the objective amendment, §4) REQUIRES an author ruling — the traffic-avoided objective is a ruled design (`design/optana/loop-tiling.md`), and this doc's job is to put the evidence for that ruling on the table, never to patch the ruling silently.
- Relationship: fix7 landed the work-shape spine (§3 streaming 1a, §4 bracket amortization 1b, §5 effect-chain vectorization 1c) under the explicit §3.3 scope guard "NO judge/search/pricing changes" — **this doc is the top layer that guard fenced off.** fix6 §2 already proved extent visibility alone is inert ("correct-but-inert": the escape vectors were §3.1-pruned or FRAG-declined, and the const-shape probes benchmarked flat). n63 landed SM occupancy as a **selection-only** score' factor. Nothing in fix1–fix7 buys threads.

## §1 Evidence base (2026-11-01, post-1c; measured against the sha256-manifest-verified 1c emissions)

Launch configs (`bin/tcacomp` rebuilt via `scripts/build.sh`):

- row_sum / dup_expr / dup_const / axpy3 / scale_copy: `dim3(16) × dim3(32)` = **512 threads**.
- gemm: `dim3(128) × dim3(1024)` = **131,072 threads** — there is NO global thread cap in the scheduler. The 512 above is a derived product of the collapse decisions, not a bound; `MaxCTAPolicy = 512` (the `ValidateLaunch` 4th arg) caps **CTAs**, is not binding at 16, and is a fix5-deferred knob.

Achieved rates vs the A6000 (min-µs of the 4-run 1c table, `.planning/quick/261101-oqg-fix7-5-slice-1c-effect-chain-vectorizati/after-bench.txt`; bytes = `testing/suite.toml` `bench_bytes`; A6000 ≈ 768 GB/s HBM2e, 84 SMs × 1,536 threads = 129,024 thread slots, ≈ 38 TFLOP/s FP32 FMA):

| kernel | moved / FLOPs | time | achieved | fraction of ceiling |
|---|---:|---:|---:|---:|
| row_sum | 18.9 MB | 5,238 µs | 3.60 GB/s | 0.47% of BW |
| scale_copy | 9.4 MB | 3,289 µs | 2.87 GB/s | 0.37% |
| axpy3 | 9.4 MB | 3,526 µs | 2.68 GB/s | 0.35% |
| softmax | 2.2 MB | 838 µs | 2.66 GB/s | 0.35% |
| dup_expr | 4.5 MB | 1,749 µs | 2.55 GB/s | 0.33% |
| dup_const | 4.5 MB | 2,749 µs | 1.62 GB/s | 0.21% |
| transpose | 1.3 MB | 950 µs | 1.32 GB/s | 0.17% |
| gemm [128,128,96] | 3.15 MFLOP | 9,678 µs | 0.33 GFLOP/s | ~0.001% of FP32 |
| mm_mid [128,256,64] | 4.19 MFLOP | 6,426 µs | 0.65 GFLOP/s | ~0.002% |

External anchors on the same shapes: a one-thread-per-output hand kernel runs the streaming family in ~10–40 µs (200–600 GB/s); cuBLAS SGEMM at 128×128×96 is launch-bound (~5–15 µs). The internal 668× arc (vs the 2026-10-29 pins) was restoration + machinery, not external competitiveness — recorded here so the doc never overstates itself.

**The expressible ceiling of the CURRENT scheduling model** (no new machinery, no rule changes):

- grid ≤ `MaxCTAPolicy` 512 CTAs; `blockDim = (Π warp-K) × 32` where warp digits may ONLY land on reduced axes (`fix4 ReducedDimWarp` tier rules) — the streaming family's reduced extents are 4–8, so warp-K = 1 and **blockDim = 32 is a structural constant for this family**.
- Max expressible: 512 × 32 = **16,384 threads ≈ 13% of the machine's thread slots**.
- That is ENOUGH for bandwidth-bound at these fixture sizes: Little's law (768 GB/s × ~600 ns ≈ 460 KB in flight) is satisfiable by 16K threads each holding a few independent loads — each thread's outputs are mutually independent, so the existing fill/compute structure supplies the memory-level parallelism.
- Estimated ceilings (ARITHMETIC — slice 0's job is to measure them): row_sum ~40–80 µs (**~70–130× headroom**), the elementwise four ~15–50 µs (**~80–130×**), softmax ~15–30 µs (~40–60×).

## §2 The defect — the search cannot see threads

1. **The objective rewards serialism.** Traffic-avoided scoring with `window-bytes = SEQ-tile`: keeping dims sequential shrinks the priced per-thread window, so LoC=1 is the rational optimum (verified on the mm_small record). Latency exposure is invisible to the objective — a schedule that starves the machine scores perfectly.
2. **n63's occupancy factor is selection-only** — a tiebreak among vectors the traffic objective already ranks, not a term that can make the search pay traffic for threads.
3. **R2 taxes the cheap parallelism**: a load site covered by a CTA dim cannot be a per-thread FRAG window (it forces the SMEM tier with cooperative costs), so CTA-collapsing surviving axes is priced as expensive for these shapes.

Consequence: the search buys 16 CTAs where 512 are offered, 512 threads where 16,384 are expressible — and the kernels run latency-bound at 0.2–0.5% of the bandwidth ceiling on correct, race-free, fully amortized code. The machinery underneath (streaming chunks, amortized brackets, vec16 fills) is built and idle; it has no threads to feed.

## §3 Slice 0 — the pinned-decision ceiling probe (measurement only, no production change)

**STATUS: LANDED 2026-11-01 (quick-261101-u8a) — the STOP rule FIRED. Verdict: STOP, ratio 1.00× (§1's expressibility analysis is contradicted); slice 1 does NOT proceed on this model.** Vehicle (b) as ruled by the research: an env-gated `paramsFromEnv` Params overlay at the three tiling-family registry factories (`internal/opt/tiling/params_env.go`), OFF by default — proven zero-default-path by a sha256 HARD gate (all 14 unpinned suite emissions bit-identical to the 1c manifest; `zero-diff-manifest.txt`); production surface diff = the 3 factory lines; search/judge/record/constraints/cg/runtime ZERO diff. Records in `.planning/quick/261101-u8a-fix8-3-slice-0-pinned-decision-ceiling-p/`: `probe-manifest.txt` (22 points), `probe-readback.txt` (intended-vs-measured grids, launched AND active threads), `probe-bench.txt`, `probe-curve.txt`, `memcheck.txt`, `suite-run.txt` (14/14), `stop-rule-verdict.txt`.

The measured record (min-µs of 4 repeats, `--jobs 1 --bench-iters 20`, in-session unpinned baselines; full table in `probe-bench.txt`):

| point | grid | launched | active | min-µs |
|---|---:|---:|---:|---:|
| row_sum unpinned | 16 | 512 | **64** | 5,239.6 |
| row_sum best pinned (rs-g16 = same schedule) | 16 | 512 | 64 | 5,240.4 |
| row_sum rs-g512 | 512 | 16,384 | **64** | 5,516.2 (worse) |
| axpy3 unpinned → g512t256 | 16→512 | 512→16,384 | 64 at every point | 3,523.3→3,579.2 (worse) |
| dup_expr unpinned | 16 | 512 | **32** | 1,740.6 |
| dup_expr dx-g256 (blockDim-1024 warp landing) | 256 | 262,144 | 262,144 | **998.7 (1.74× — the only bend)** |

**STOP-rule arithmetic:** best pinned row_sum 5,240.4 µs vs in-session unpinned 5,239.6 µs = **1.00×** < 10× (the 10× bar was 523.8 µs; §1 estimated 40–80 µs at 16,384 threads — the measured answer is that 16,384 launched threads carry 64 active ones). Mechanisms, now measured-concrete: (1) active-vs-launched CTA accounting — every default-tile point of all three kernels runs 32–64 ACTIVE threads regardless of grid (T=256 tiles over 512/256-extent dims leave ceil(extent/T)=1–2 CTAs of work; the launched axis is a fiction the occupancy model's NominalExtent stand-in credits at 16); (2) fully-active big grids at blockDim 32 are INEXPRESSIBLE — full activity needs T small, and T<32 zeroes the FRAG reuse (ceil(T/32)=1 trip → zero traffic), so the search either stays at T=256 or degenerates (warp-ceremony landings; dx-g64/dx-g512 select the v0-unreachable GlobalReducer — the known D1 defect class — and fail at launch); (3) per-thread MLP is bounded by the 256 B FRAG budget / T-strided coupled walk. Additional probe findings: the plan's 2-dim CTA presets cannot realize (tier-1 refuses fully CTA-covered loads → zero traffic; single-dim presets + MaxCTA + NominalExtent=N×maxTile pin every grid cleanly), and the R1 adaptive digit collapses to its lowest tile under the MaxCTA clamp, so a grid pin must ride the preset list.

**Purpose:** replace §1's arithmetic ceiling with a measured one — pin the decisions the search is allowed to express and measure what the CURRENT model delivers at each parallelism point.

Protocol: for row_sum, axpy3, dup_expr — sweep the expressible decision space: grid ∈ {16, 64, 128, 256, 512} (R-CTA tile/LoC variants), thread-spread settings, chunk shapes within the existing 256 B budget — respecting the reduced-dim-WARP restriction (blockDim stays 32). Emit, build through the suite fixture path, bench `--jobs 1 --bench-iters 20`, and produce the honest table + the µs-vs-resident-threads curve against the bandwidth-arithmetic line.

Vehicles (in order of preference): (a) existing Params knobs (tile presets, the R1 adaptive-LoC ceiling, `MaxCTAPolicy`) if they alone express the sweep points; (b) a TEST-ONLY decision-override hook in the tiling pass (env-gated, zero default-path change); (c) a throwaway branch. No RED pins required (no production semantics change); the deliverable is the measurement record.

**STOP rule:** if the measured ceiling on row_sum is < 10× vs the 2026-11-01 number (5,238 µs), the §1 expressibility analysis is WRONG somewhere — stop and re-derive (candidates: over-provisioned CTA-mapping semantics, fill traffic scaling with grid, ThrIterator distribution effects) before any objective work. Never build slice 1 on a contradicted model. **→ FIRED at 1.00× (see STATUS above).**

## §4 Slice 1 — the objective amendment (design-first; REQUIRES AUTHOR RULING)

The traffic-avoided objective is a ruling; amending it is the author's call. Candidates to put before the ruling:

- (a) **A resident-threads floor as a CONSTRAINT** — a closed-form, shape-derived minimum parallelism (e.g., threads ≥ min(expressible outputs, Little's-law bound / per-thread in-flight bytes)); schedules below the floor are declined the way the FRAG budget declines over-budget sites. The objective itself stays traffic-pure; parallelism becomes a legality, not a score.
- (b) **An occupancy term in score''** — e.g., penalize traffic-per-resident-thread; promotes n63's factor from selection to score with a pinned weight.
- (c) **Fix the enumeration, not the score** — widen which collapse vectors the search is allowed to enumerate (the fix4 tier rules prune high-LoC vectors early) and let the EXISTING n63 factor select bigger grids.

Non-negotiable constraints: the v62 budget math (Σ blockFootprint ≤ MaxFragPerThread) untouched; R2 tier rules unchanged; the matmul family stays bit-identical unless a slice explicitly re-rules matmul scheduling.

Gates: the streaming family within ~2× of the slice-0 MEASURED ceiling; ≥ 10× vs the 2026-11-01 numbers on row_sum + the elementwise four; no suite kernel > 5% slower; the §7 protocol.

## §5 Slice 2 — MaxCTAPolicy revisit (the fix5-deferred decision)

Binds only once grid demand exceeds 512 CTAs (slice-0 data says when). A one-line Params change + re-measure; decide with numbers, per the fix5 ruling's own deferral.

## §6 Explicitly out of scope

- **warp-K > 1 on surviving axes** — the §3.1-pruned cross-product lowering (GPU-proven 1/64 hole). This is the blockDim=32 wall and `design/nx/nx.md` problem [1]'s subject; it needs the nx design cycle or a dedicated ruling, not a fix8 patch. Fix8's ceiling is 16,384 threads — stated plainly so nobody expects 129K from this doc.
- Tensorization / warp-per-tile / warp specialization (the nx.md ladder 3/4/5/12/13).
- 1d SMEM pipelining + swizzle (fix7 §6) — the matmul-family track, independent.

## §7 Verification protocol

The n63/qwo §9 protocol verbatim: `scripts/build.sh` first (the stale-prebuilt gotcha); hermetic gates (build/vet/test, module count 1); suite 14/14 at `--jobs 1` (canary 0, maxdiff within suite.toml tolerances); memcheck on every schedule-changed kernel at full dims; benchmarks `--jobs 1` only, honest before/after tables in the task dir; no kernel > 5% slower than best-known-good; sha256 manifests for every emitted kernel; violations recorded honestly, never hidden.

## §8 Task breakdown

| # | Slice | Depends | Gates |
|---|---|---|---|
| 0 | pinned-decision ceiling probe (§3) | — | measured ceiling table + curve; STOP rule honored — **DONE 2026-11-01: STOP fired at 1.00×** |
| 1 | objective amendment (§4) | 0's numbers + author ruling | ≥10× streaming vs 2026-11-01; within ~2× of measured ceiling; §7 |
| 2 | MaxCTAPolicy revisit (§5) | 0 (binds only past 512 CTAs) | re-measured, decided with numbers |

Sequencing: 0 → (author ruling on §4 candidates) → 1 → 2. Slice 0 is launchable immediately; slices 1–2 wait on its numbers.
