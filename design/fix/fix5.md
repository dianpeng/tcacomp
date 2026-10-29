# fix5 — Scheduler heuristic vocabulary for sane occupancy schedules (implementation spec)

- Status: READY FOR IMPLEMENTATION. Extends fix4 (the factorization model is unchanged — this spec widens the feasible area's *vocabulary* and the objective's *currency*; no constraint is relaxed, no record field changes meaning).
- **Amendment 2026-10-29 (author rulings, second session): the three §6 open questions are RESOLVED** — (1) T1 accepts 1-warp CTAs as the interim collapsed-kernel shape (duplicate warps rejected: they duplicate loads, ~N× GMEM traffic, reintroducing the ceremony R2 removed); (2) `MaxCTA` stays **512** for T1, decision deferred to the T1 measurement (one-line Params change); (3) WARP placement is **extent-aware — largest nominal extent first, stride-1 as tie-break, the reduced axis as last-resort fallback**, which preserves the n63 family-B split-K wins exactly (large K wins on extent) while freeing small-K kernels to coalesce on C. The provisional R3 ordering below is superseded by this ruling.
- **Built on quick-261029-n63** (SM occupancy in the tiling objective: `score' = traffic × (1−idleFrac) × (ε+(1−ε)·fill)`), which delivered 6.9×–29.1× on the reduce-carrying compute family but **left the elementwise/latency-bound family flat (±1.2%)**: dup_expr 1.26 s, dup_const 1.23 s, axpy3 2.45 s, scale_copy 2.38 s (A6000, `--bench-iters 20`).
- **Conflict rule: where this doc disagrees with `design/fix/fix4.md` on the topics it touches (LoC presets, reducer wiring, thread-map slots), THIS DOC WINS.** fix4's factorization model, constraint arithmetic, record schema, and tiling-verify protocol are inherited verbatim. `design/milestone2/tile_scheduling.md` remains the canonical semantic model; this doc is the v0-search vocabulary to reach it.
- All section-§ references to fix4 (`§1.3`, `§2.1`, …) are to `design/fix/fix4.md`.

## §0 Measured evidence — why the elementwise family is still 100–1000× off

`scale_copy` (`[512,512,4] → [512,512]`, gain/bias affine map, trailing K-reduce forced by the W1 spike wall) emits (`/tmp/tcainspect/scale_copy.cu`, reproduced any time from `testing/kernels/runtime/scale_copy.tca`):

```
launch 16 CTAs × 64 threads          // 1,024 threads ≈ 0.6% of the A6000
LNT: R=CTA(T=256,LoC=16) C=SEQ(T=32,LoC=1) K=WARP(T=32,LoC=2)
per (r,c) element: reducer.Init() → 2 scalar GMEM fills → FMA chain →
                  Reduce() → reducer0.Combine()   // full 2-warp collective
                  → single-owner store
```

Three verified defects, in cost order:

1. **Per-element warp-collective reducer bracket.** Every one of the 262,144 output elements pays Init + two scalar loads + a grouped-shuffle/SMEM-staged Combine whose entire job is adding K=4 numbers. C walk is `SeqIterator(0,1,…)` — one element per bracket. This is >95% of the 2.38 s.
2. **Grid starvation by preset ceiling.** The occupancy term (n63) pushed LoC to the largest non-idle preset value (16 at tile 256), but the digit list tops out at `CtaLoCPresets = {2,…,64}` while the kernel's device capacity is `84 SMs × min(48/2, 1536/64) = 2,016 CTAs`. The best grid the search space can *express* is 3% of capacity. **The objective is optimizing over a structurally starved menu.**
3. **Lane monopoly on the reduced axis.** Warp digits can only land on K (`ReducedDimWarp` capability + the warp-digit placement), so `blockDim = ΠK×32` is anchored to an axis with nominal extent 4: 28 of 32 lanes are dead in the reduce and do nothing useful anywhere else, while C — the long, stride-1, coalescible axis — receives zero threads. The doc `tile_scheduling.md` §8.3 grouping rule already supports lanes sharding *any* axis; the v0 heuristic never uses it.

## §1 Ruled decisions

Rulings from the author session 2026-10-29; they are normative for implementation.

### R1 — Adaptive LoC ceiling replaces the fixed preset top

The fixed `CtaLoCPresets`/`WarpLoCPresets` lists stay as the *floor* granularity, but every dim's digit list gains ONE extent-derived top digit:

```
LoC_adaptive(d) = clamp(ceil(NominalExtent / T_d), 2, MaxCTA)   // CTA dims
K_adaptive(d)   = clamp(ceil(NominalExtent / T_d), 2, MaxWarpsPerCTA)  // WARP dims
```

Deterministic, shared with tiling-verify exactly like every other formula in `constraints.go` (single source). Effect: at every tile size there now exists a full-grid candidate, so the n63 occupancy factor can actually bite (today `fill` is capped by the menu, not the device). `MaxCTA` (512) becomes the real policy ceiling; consider 512→2048 in a follow-up if wave-1 measurements justify it.

### R2 — Degenerate-reducer collapse (SEQ-only ladder)

When the reduced axis is entirely covered by the SEQ layer under the chosen factorization (`F[m][k] = 1` at every layer above SEQ — i.e. no WARP/THREAD digit on the reduced dim), the reducer ladder is **one tier**:

- No `reducer.Init`/`Combine` ceremony, no shuffle groups, no SMEM staging. Each thread owns one accumulator register per surviving output element; the store path (single-owner R2/R3) is unchanged.
- Correctness: the combiner is associative+commutative (tca.md §2.5 invariant); with no cross-thread visibility there is nothing to synchronize. Tail-clamped dynamic extents need nothing special — the SEQ walk rounds over `ceil(K/T_k)` as today.
- Runtime: `WarpReducer` gains a single-tier fast path (or the compiler simply never constructs a reducer object for a one-tier ladder — the latter is preferred: no runtime surface change, final-lower emits a plain accumulator).
- Interaction with R1: collapse removes the `ReducedDimWarp` traffic penalty that currently makes K=SEQ vectors score dead-last (the warp-sharded fills that demote to per-thread FRAG disappear), so the search can now *choose* K=SEQ on small-K kernels.

### R3 — Warp/lane axis freed from the reduced dim, extent-aware placement (RULING AMENDED)

The WARP digit may land on **any** dim (not only the reduced one). Placement is **extent-aware** (author ruling, 2026-10-29 session 2):

```
the WARP digit goes to the dim with the LARGEST nominal extent;
tie → the stride-1 dim (coalescing);
last-resort fallback → the reduced dim (today's behavior).
```

Rationale: this preserves the n63 family-B split-K wins exactly (a large K wins on extent, so warps-on-K is unchanged for the matmul family) while small-K kernels escape to the long coalescible dim (scale_copy: warps on C). The earlier provisional "stride-1 first" ordering would have risked regressing the split-K matmuls and is superseded. This is the `tile_scheduling.md` §8.3 step-2 machinery (group = lanes sharing all thread coords except the k-shard) applied to whichever axis carries the lanes; the reducer grouping rule is unchanged, it just operates on the actual shard layout. With R2+R3, the saner elementwise schedule falls out naturally: R→CTA (adaptive LoC), C→WARP (lanes sweep contiguous C, fully coalesced), K→SEQ per lane (4 FMAs in registers, no ceremony).

### R4 — Work-shape term deferred; tier-efficiency guard ruled IN for R2 only

A full per-thread work model (register tiles, ILP, bracket amortization) is **deferred to fix6** — it is a bigger design (SEQ-layer FRAG windows > 1×1) and must not block R1–R3. But one objective amendment ships with R2: a warp-sharded reduced dim whose nominal extent cannot feed its lanes earns **no traffic credit** for the fills it fragments (tier efficiency `min(1, NominalExtent_k / (32·ΠK))` scales the holders term of K-covering fills). This is the minimal change that stops the objective from *pay* for degenerate lane-sharding — without it, R2's collapse vectors keep losing to the ceremony vectors on traffic grounds.

## §2 Search changes (`internal/opt/tiling`)

| Surface | Change |
|---|---|
| `Params` (record.go) | No new fields; `NominalExtent` (already the stand-in extent) feeds R1's adaptive digits and R4's tier efficiency. Document in the §2.1-style table. |
| `digitListOf` (search.go) | Add the R1 adaptive top digit per SoC; for WARP dims, order candidates by the R3 extent-aware heuristic (largest nominal extent, stride-1 tie-break, reduced axis last-resort — computed once per LNT). |
| `judge.go` | R4: apply tier efficiency to K-covering fill holders when the reduced dim is warp-sharded and `NominalExtent_k < 32·ΠK`. |
| `constraints.go` | `locPresetOK` gains the adaptive-digit acceptance path (same formula, shared with verify). |
| `occupancyOf` | Unchanged; R1 makes its `fill` reachable. |

Feasible area grows (one extra digit per dim per SoC); enumeration cost stays trivial (the search counted hundreds of vectors, not millions). Determinism contract unchanged: enumeration order fixed, ties first-found, `partism` last.

## §3 Lower/codegen changes

| Surface | Change |
|---|---|
| final-lower (`internal/opt/finallower/reducer.go`) | R2: one-tier ladder → no reducer object, plain per-thread accumulator + existing single-owner store. Multi-tier ladders unchanged. The per-element bracket structure survives where a tier exists (bracket amortization is fix6). |
| schedule pass | No structural change; the WARP-on-non-reduced-dim layout already flows through the existing thread-map coordinate machinery (one slot per WARP dim — fix4 §1.3 adopted ruling), only the *choice* of dim is new. |
| runtime (`runtime/`) | **No change required** (R2 prefers not constructing a reducer; existing WarpReducer path untouched for multi-tier). |
| cg (`internal/cg/cxx`) | No emission change for R1–R3. |

## §4 Task breakdown

- **T1 — R1 + R2 + R4** (the coupled small set): adaptive LoC digit, degenerate-reducer collapse, tier-efficiency guard. Lands the interim sane elementwise schedule: adaptive-grid CTAs × 1-warp (blockDim 32) × pure sequential per-thread streams — zero ceremony. Expected: elementwise family 2.38 s → low tens of ms (≥50×); no regression >5% on any suite kernel (family B's split-K vectors are multi-tier and unaffected).
- **T2 — R3**: warp/lane axis freedom with coalescing preference. Expected: elementwise family → single-digit ms (lanes coalesced on C, blockDim 128–1024); improves small-K reduce kernels (mlp, softmax, layernorm) as a side effect.
- **T3 (fix6, out of scope here)**: register blocking / FRAG windows > 1×1, bracket amortization, effect-chain vectorization (float4), RA smem, explicit barriers + cp.async pipelining, swizzle. Sequenced after T1/T2 land and their numbers are pinned.

Each task: TDD pins first (RED), implementation (GREEN), re-derive — never weaken — drifted pins, full suite, honest before/after benchmark via worktree-at-parent vs new build (the n63 protocol).

## §5 Verification gates

1. `go build ./…`, `go vet ./…`, `go test ./…` exit 0.
2. GPU suite `python3.13 testing/orchestrator.py`: 14/14 PASS; memcheck clean on every schedule-changed kernel.
3. Benchmark gate vs the n63-pinned numbers (`--bench-iters 20`): elementwise family (dup_expr, dup_const, axpy3, scale_copy) each ≥10× faster; no suite kernel >5% slower. Results recorded in the task SUMMARY with the .cu-diff attribution per family (the n63 precedent).
4. Correctness invariants: schedules remain legal under the unchanged constraint set (tiling-verify green); D1/D2 untouched (GlobalReducer still unreachable — cross-CTA reduction is NOT part of this spec).

## §6 Resolved rulings (author decisions, 2026-10-29 second session)

The three open questions from the first draft were ruled by the author; the rulings are normative and folded into §1 above.

1. **T1 interim CTA shape — RESOLVED: accept 1-warp CTAs.** Duplicate warps were rejected: they duplicate the loads (~N× GMEM traffic) unless SMEM-shared, reintroducing the ceremony R2 removes. Occupancy is not lost (48 one-warp CTAs fill an SM's 1,536 threads); blockDim 128–1,024 with real per-lane work arrives properly in T2 via R3 lanes-on-C.
2. **`MaxCTA` — RESOLVED: stays 512 for T1; decide from measurement.** The T1 benchmark shows where fill actually binds; the raise to 2,048 (if warranted) is then a one-line Params change, since the occupancy factor already self-regulates toward grid ≈ capacity.
3. **WARP-axis precedence — RESOLVED: extent-aware, largest nominal extent first** (stride-1 tie-break, reduced axis last-resort). Preserves the n63 split-K wins (large K wins on extent) while freeing small-K kernels to coalesce on C.
