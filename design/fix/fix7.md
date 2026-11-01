# fix7 — Work-shape model: streaming register-block lowering + the remaining performance chain

- Status: §3 SLICE 1a LANDED 2026-11-01 (quick-261101-kel) — pins 1-3 RED-recorded then GREEN, §3.4 audit recorded, §3.6 gates ALL MET (restoration: row_sum 37.0 ms < 459 ms, softmax 9.7 ms < 59.3 ms; improvement: the elementwise four 7.0/7.8/21.1/20.8 ms vs 787/804/1351/1397 ms; the >=10x elementwise check vs the 2026-10-29 pins measures 166x/151x/113x/113x — recorded honestly, ahead of 1b). The §3.4 placement gap is CONFIRMED and recorded (fill rides the per-element bracket; priced reuse 8 not realized on the const-innermost four) — it is the slice 1b/fill-placement follow-up, never patched here. Details: `.planning/quick/261101-kel-*/` (SUMMARY, reuse-audit.txt, before/after-bench.txt, sha256 manifest).
- Relationship: fix5 §4 T3 reserved "fix6" for the register-blocking / work-shape model; fix6.md §3 carries the intent. quick-261030-v62 landed the **pricing half** (honest `min(raw, MaxFragPerThread)` block pricing, judge/search/tiling-verify/lower-cache agreeing by construction) and quick-261031-c8n landed the **emission prerequisite** (D-fix6-03). THIS doc is the **lowering half**: making codegen emit what the pricing model assumes, then the rest of the performance chain. fix6 §1 (per-shape specialization) and D-fix6-01 remain tracked there and are re-planned here only as sequencing slots (§8/§9).

## §1 Evidence base — what the quick-task chain established

| Task | Contribution | Key artifact |
|---|---|---|
| 261029-n63 | occupancy-aware score' (selection-only) | `.planning/quick/261029-n63-*/` |
| 261029-qwo | five-link deadlock diagnosis; the 481.5 ceremony score reproduced later by v62 exactly | fix5.md §4 |
| 261029-v9k | fix5 T2 R3 (warp-axis freedom) + §3.1 surviving-WARP pruning (GPU-proven 1/64 hole) | pins 8085b84/ff929a5 |
| 261030-uai | const-shaped the six suite kernels (mixed dynamic/static, D-fix6-02 wrapper) | sha256 manifests |
| 261030-v62 | honest FRAG pricing; collapse selection pinned unit-green; suite flip BLOCKED on emission | branch merged by c8n |
| 261031-c8n | D-fix6-03 fixed (const-covered identity reconciliation); v62 merged; **the honest post-merge numbers** | `261031-c8n-*/{before,after}-bench.txt` |

The 2026-10-31 post-merge baseline (the numbers this doc's gates refer to; `--jobs 1 --bench-iters 20`, A6000):

| kernel | 2026-10-29 pin | 2026-10-31 after | vs pin |
|---|---:|---:|---:|
| dup_expr | 1,168 ms | 787 ms | −32.6% |
| dup_const | 1,184 ms | 804 ms | −32.1% |
| axpy3 | 2,377 ms | 1,351 ms | −43.2% |
| scale_copy | 2,346 ms | 1,397 ms | −40.5% |
| row_sum | — | 1,980 ms (was 459 ms) | **+331% regression** |
| softmax | — | 263 ms (was 59.3 ms) | **+343% regression** |
| transpose | — | 180 ms (was 336 ms) | −46% |
| layernorm | — | 460 ms (was 493 ms) | −6.6% |
| matmul family | — | +1.1…+3.9% (bit-identical .cu; noise band) | — |

## §2 The exact defect — codegen does not honor the streaming contract

The pricing model (`internal/opt/tiling/judge.go:46-51`, v62) defines a promoted FRAG site as a **streamed register block**:

```
blockFootprint = min(rawWindow, MaxFragPerThread)
"the window's peak live per-thread bytes under budget-sized streaming —
 one chunk of at most the whole budget lives in registers at a time"
```

Codegen never implemented the streaming. The emitted kernel (actual `scale_copy.cu`, 2026-10-31):

```cuda
tca::rt::Frag<float, 2u, 256u, 256u> tca_f0;        // 65,536 floats = 256 KB PER THREAD
...
auto v22 = tca_f0.Get(C_it3.Index(), e4_it5.Index());  // runtime indices → cannot be registers
```

`Frag`'s backing store is `T data_[kElems]` (`runtime/tca/view.h:226`). A dynamically-indexed 256 KB array cannot live in registers (sm_86: 256 KB registers per SM, ~8 KB/thread at full occupancy) → nvcc spills the whole array to **local memory (DRAM)**. The budget paid for 256 B; the hardware allocates 256 KB. 32 threads × 256 KB = 8 MB of per-CTA local traffic, refilled per tile-between round.

Three distinct pathologies, all fixed by the same slice (§3):

1. **Raw-window materialization** — the buffer is sized to the RAW covered window (Π T_d) instead of the priced block. The budget gate (`Σ blockFootprint ≤ MaxFragPerThread`, greedy density eviction, `search.go:231`) binds correctly *under the streaming assumption* and is NOT at fault.
2. **Nominal-T sizing ignores true const extents** — the window's e4 dim is sized 256 from the nominal tile param `T=256` while the true const extent is 4: 252/256 of that dim is allocated and never touched (64× over-allocation vs the real 4 KB footprint). Const dims exist precisely so codegen can read true extents.
3. **Refill-per-bracket-iteration structure** — the emitted nest places the fill inside the per-element reducer bracket, so the window is refilled once per inside-walk step (scale_copy: 256 refills per C tile) while the judge priced reuse 8 under the streaming model. Whether this is LICM-correct-but-mis-nested or a placement gap is an open verification item for slice 1a (§3.4) — the stream structure must make the priced reuse the actual reuse.

Consequences today: the collapse is correct (14/14, canary 0), legal (memcheck clean; softmax only OOMs *under instrumentation* — the spill × sanitizer shadow exceeds the instrumented launch budget), and genuinely faster on the elementwise four (bracket elimination outweighs the spill at K=4/8) — but capped at **1.47–1.76×**, and a **3.3× regression** on the [R,Ku,C] reduce pair whose window streams from local memory under the reducer bracket.

## §3 Slice 1a — streaming register-block lowering (THE gate)

**Ruling intent:** the emitted code must realize the priced block. A FRAG buffer's peak live bytes = its chunk; the covered window streams through the chunk in tile-between-round-sized pieces; the existing tile-between rounds ARE the chunk loops.

### §3.1 Chunk sizing

- Chunk dims = covered-dim extents clamped so `sizeof(chunk) ≤ j.bytes` (the block footprint the judge already computes — `siteWindowBytes`, the ONE shared source judge/search/tiling-verify/lower-cache agree on). At most one chunk per site lives at a time → the Σ-blockFootprint budget stays exactly valid; chunking does not change the budget math.
- **True extents for const dims**: the e4 chunk is 4 (const), not T=256 nominal. Kills pathology 2 by construction. (Note the post-typeinfer nil-computed-cell gotcha: const extents come from `ConstTypeOf` / the dim param's canonical `e<extent>` spelling — the c8n identity source — not from `TypeOf`.)
- Dynamic dims keep the tile-param chunk with the runtime's existing tail clamp (`umin(chunk, extent - base)` — already the loader's contract).

### §3.2 Emission shape

The emitted nest already has the streaming skeleton — the tile-between rounds (`SeqIterator(step=T)`) advance the window anchor per chunk (the D-fix6-03 coordinate reconstruction places the round bases correctly). The change is:

1. `Frag` declaration dims = chunk dims (pathology 1). `internal/cg/cxx/walk.go:133` currently prints the buffer type's dims verbatim; the buffer's window factors (set by tile-lower/lower-cache) must BE the chunk — i.e. the fix straddles the lower's window-factor computation and the cxx decl, which read the same buffer node (one source of truth, no divergence).
2. Inside-walk extents clamp to the chunk (the `SeqIterator(0, 1, vN)` walks already clamp via `umin` — the clamp target changes from nominal T to the chunk).
3. The fill stays at the tile-between level, one refill per round — exactly the structure the judge's reuse arithmetic prices.

`LoadTileThread` needs no new machinery: `detail::load_tile_rows` is already row-striped, tail-clamped, zero-fill-padded, with a `uint4` fast path (`runtime/tca/loader.h:128`) — the chunk is just a smaller window per round.

### §3.3 What 1a does NOT touch (scope guard)

- NO judge/search/pricing changes (v62 owns the model; c8n owns emission identity).
- NO schedule-pass changes. If §3.4 confirms a placement gap, that is a SEPARATE follow-up task with its own RED pins.
- NO SMEM/cp.async pipelining/swizzle (§6), NO float4 effect-chain work (§5), NO register blocking beyond the streamed chunk (multi-element register tiles per thread = bracket amortization territory, §4).

### §3.4 Open verification item — priced vs actual reuse

The judge pins reuse per site (`j.reuse`, traffic = rawBytes × (reuse−1) × holders). After 1a, verify on the suite emitters that the actual refill count matches the priced reuse (the c8n/v62 pins plus a new emission pin counting `LoadTileThread` sites per nest path). If actual < priced, the placement is the gap → separate task. Record, never patch silently.

### §3.5 RED pins (TDD, in order)

1. **Unit**: crafted-record fill fixture (the c8n `fillcoord_test.go` idiom) — pin the chunked `Frag` decl dims + the stream-loop structure + per-round window anchors (byte golden).
2. **Unit**: const-extent chunking pin — a covered const dim sizes the chunk by TRUE extent (the 4-not-256 case) — this is the regression test for pathology 2.
3. **e2e emission**: scale_copy + row_sum emitted `.cu` pin — chunked Frag, no 256 KB decl; coordinate order tensor-dim (unchanged c8n semantics).
4. (GREEN realization, not a pin) existing v62/c8n pins must stay green UNWEAKENED — especially `TestFix5T2OneTierCollapseEmission` and the fragwindow suite.

### §3.6 1a gates (the restoration clause)

- Hermetic: build/vet/test green, module count 1, all existing goldens byte-stable except the pinned stream goldens.
- Suite 14/14 at `--jobs 1` after `scripts/build.sh` rebuild; memcheck clean on all schedule-changed kernels — **including softmax at FULL dims** (the 256 KB/thread local array must disappear; the reduced-dims instrumentation protocol from c8n is retired as a c8n-era consequence, not a permanent exemption).
- **Restoration gate**: row_sum < 459 ms and softmax < 59.3 ms (their pre-merge numbers). Without this, the collapse is a net loss on the reduce family and 1a is not done.
- **Improvement gate**: elementwise four strictly better than 787/804/1351/1397 ms.
- Honest before/after `--jobs 1` table recorded in the task dir (n63/qwo protocol).

## §4 Slice 1b — bracket amortization (after 1a pins)

With the window streamed in chunks, the reducer bracket's per-element ceremony (Init + Combine per output element) becomes the dominant residue on the reduce family: today `reducer0.Init()`/`Combine()` wrap every single `C_it3` step (256 brackets per tile), each bracket walking 4 elements. Design direction: one bracket per chunk — Init before the chunk's e4 walk, Reduce per element, Combine after; for multi-chunk reduced dims the combiner folds chunk partials (the SeqReducer is runtime-owned; the runtime gains a chunk-fold entry or the codegen emits an explicit accumulator + one final Combine — DECISION for the 1b plan). Reducer semantics (one-tier, single-owner store) unchanged — this is ceremony count, not semantics. RED pin: bracket-count in the emitted `.cu` (Init/Combine pair count per nest path). Gate: row_sum/softmax beyond the 1a restoration numbers; elementwise ≥10× check becomes meaningful here.

## §5 Slice 1c — effect-chain vectorization (cheap, after 1b)

The loader's `uint4` fast path exists but the emitted chunk shapes may not hit its eligibility (trailing extent %4, 16 B alignment). After 1b: pin chunk trailing extents to multiples of the vector width where the true extents allow (const dims: known at compile time — K=4/8 suites qualify), or emit the runtime eligibility as-is and pin that the fast path fires (a CPU-unit pin on `load_tile_rows` already covers the mechanism; the emission-side pin is new). Scope: emission-side shape choice only. Gate: another honest benchmark table; expected incremental, not gating.

## §6 Slice 1d — RA smem + cp.async pipelining + swizzle (deferred, design-first)

The SMEM class already has cooperative fills with a cp.async fast path and pinned barrier protocol (C9: one `__syncthreads()` before/after, `loader.h:209-218`). What does not exist: double-buffered pipelining across tile-between rounds (prefetch chunk N+1 while computing N), reduction-as-smem staging, and swizzle for bank conflicts. This slice is a matmul-family payoff (the elementwise collapse doesn't need SMEM at all) and interacts with the n63 occupancy factor — it gets its own discuss → plan cycle with 1a–1c numbers pinned first. Explicitly NOT part of the ≥10× elementwise gate.

## §7 D-fix6-01 — per-chain positional identity (sequencing slot)

Still open, fail-closed since c8n (const positions with >1 extent match are a NAMED error, never guessed). Design sketch lives in fix6.md §4; it unblocks fully-static repeated-extent shapes (`[512u, 512u, 4u]`) for fix6 §1. Not on the elementwise performance path — schedule after 1a–1c unless §1 specialization is prioritized.

## §8 fix6 §1 — per-shape specialization + dispatching launcher (sequencing slot)

Per fix6.md §1: frontend shape lists, per-shape compilation (every heuristic reads true extents — R1/R4/n63 finally bite per-variant), one stable dispatching host face (absorbs the D-fix6-02 wrapper proliferation the uai suite now couples to). Depends on §7. The ≥10× gate does NOT depend on this slice — it depends on §3–§5.

## §9 Verification protocol (all slices, n63/qwo)

1. `scripts/build.sh` rebuild FIRST (the gitignored prebuilt shadows source — stale-binary gotcha).
2. Hermetic: `go build`/`go vet`/`go test` exit 0; `go list -m all | wc -l == 1`; pin discipline — never weaken a drifted pin without re-derivation recorded in the pin's comment.
3. GPU correctness: suite 14/14 at `--jobs 1` (canary 0, maxdiff within suite.toml tolerances).
4. Memcheck on every schedule-changed kernel; a full-dims memcheck failure is a HARD gate (the c8n reduced-dims protocol was an explicitly-recorded spill consequence).
5. Benchmarks `--jobs 1` only (parallel fixtures contend on the A6000, 4–6× inflation). Honest before/after tables in the task dir.
6. Global gates: **≥10× elementwise vs the 2026-10-29 pins**; no suite kernel >5% slower than its best-known-good; the matmul-family bit-identical constraint where the slice doesn't touch matmul scheduling.
7. Record violations honestly (the c8n row_sum/softmax precedent: recorded, attributed, never hidden).

## §10 Task breakdown

| # | Slice | Depends | Gates |
|---|---|---|---|
| 1 | **1a streaming block lowering** (§3) | — | restoration + improvement gates (§3.6) |
| 2 | 1b bracket amortization (§4) | 1a pins | row_sum/softmax past restoration; ≥10× check |
| 3 | 1c vectorization (§5) | 1b | incremental, honest table |
| 4 | 1d smem pipeline/swizzle (§6) | 1a–1c numbers | matmul family, own plan |
| 5 | D-fix6-01 (§7) | — (independent) | fully-static repeated-extent suite kernel emits |
| 6 | fix6 §1 specialization (§8) | 5 | per-variant heuristic payoff measured |

Sequencing: 1 → 2 → 3 is the performance spine (the ≥10× gate becomes measurable after 2 and is expected to need all three). 5 → 6 is the specialization spine, independent of the performance spine until §1's per-variant heuristics want the §3–§5 lowering too.

## §11 Risks / open questions

- **Priced-vs-actual reuse** (§3.4): if the fill placement gap is real, 1a's gates may still fail on the reduce pair even with correct chunking — the fallback is the separate placement task, NOT a judge tweak.
- **Chunk-tail overhead**: per-row `umin` clamps already exist; chunking multiplies round count (256/T_chunk refills) — the traffic term prices this, so the search will prefer chunk-friendly vectors once the model and reality agree; expect the search to re-choose some tile params after 1a (record, don't freeze).
- **Occupancy**: post-1a register pressure is the priced 256 B/thread — collapse blockDim 32 is far from any occupancy cliff; 1d's SMEM work is where occupancy bites (n63 owns that factor).
- **Local-memory disappearance**: nvcc's spill of a 256 KB array also inflates the launch's local-memory reservation — after 1a, expect the softmax instrumentation OOM to vanish (§3.6 makes it a gate, so a lingering OOM is a hard failure signal, not a protocol workaround).
