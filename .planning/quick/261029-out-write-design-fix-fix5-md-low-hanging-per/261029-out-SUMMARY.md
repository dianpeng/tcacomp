---
phase: quick-261029-out
plan: 01
subsystem: design
tags: [tcacomp, scheduling, heuristic, occupancy, reducer, fix5]

# Dependency graph
requires:
  - quick: 261029-n63 (SM occupancy in the tiling objective — the occupancy
    factor this spec makes reachable via the adaptive LoC digit)
  - quick: 261028-t4u (the testing suite + benchmark protocol reused for
    the §5 gates)
provides:
  - design/fix/fix5.md — the implementation spec for the low-hanging
    scheduling heuristics (R1 adaptive LoC ceiling, R2 degenerate-reducer
    collapse, R3 lane-axis freedom with coalescing preference, R4
    tier-efficiency guard), task breakdown T1/T2, verification gates,
    and the open author rulings for T2
affects: [internal/opt/tiling, internal/opt/finallower, testing-suite-benchmarks]

# Metrics
duration: ~20 min
completed: 2026-10-29
---

# Quick 261029-out: fix5 design spec Summary

**`design/fix/fix5.md` committed at `6a91298`** — implementation spec for
the scheduler heuristic vocabulary that the occupancy-only objective
(n63) cannot reach: the elementwise family is still 1.2–2.4 s because
(a) every output element pays a per-element warp-collective reducer
bracket (K=4 added by the W1 wall), (b) the LoC preset ceiling ({2..64})
caps the expressible grid at 3% of device capacity, and (c) warp/lane
digits can only land on the reduced axis, stranding the coalescible C
axis with zero threads.

## The four rulings

- **R1 — Adaptive LoC ceiling**: every dim's digit list gains one
  extent-derived top digit `clamp(ceil(NominalExtent/T), 2, MaxCTA)`
  (WARP: capped at MaxWarpsPerCTA), shared with tiling-verify per the
  single-source rule. Full-grid candidates now exist at every tile.
- **R2 — Degenerate-reducer collapse**: when the reduced axis is entirely
  in the SEQ layer, the ladder is one tier — no Init/Combine, no
  shuffles, no SMEM staging; final-lower emits a plain per-thread
  accumulator (preferred over a runtime fast path: no runtime surface
  change).
- **R3 — Lane-axis freedom**: WARP digits may land on any dim, ordered by
  a coalescing heuristic (stride-1 + largest nominal extent); the §8.3
  grouping rule is unchanged, only the axis choice is new.
- **R4 — Tier-efficiency guard**: a warp-sharded reduced dim that cannot
  feed its lanes (`NominalExtent_k < 32·ΠK`) earns no traffic credit for
  its fills — stops the objective paying for degenerate lane-sharding.
  The full per-thread work model (register tiles, bracket amortization,
  vectorization) is explicitly deferred to fix6.

## Task breakdown

- **T1** = R1+R2+R4 (coupled small set) — interim sane elementwise
  schedule, expected ≥50× on the 1.2–2.4 s family, no >5% regressions.
- **T2** = R3 — coalesced lanes on C, blockDim 128–1024, single-digit ms.
- **T3/fix6** (out of scope): register blocking, float4 vectorization,
  RA smem, explicit barriers + cp.async, swizzle.

## Gates

14/14 suite + memcheck clean on schedule-changed kernels; benchmark gate
vs the n63-pinned numbers (elementwise ≥10× each, nothing >5% slower);
tiling-verify green (constraint set untouched); D1/D2 explicitly out of
scope (no cross-CTA reduction here).

## Open rulings for T2

1. Idle-warp duplication accepted as interim T1 shape (blockDim 32)?
2. MaxCTA 512→2048 now or after measurement?
3. Does coalescing always outrank the reduced axis for WARP placement?

## Self-Check: PASSED

- fix5.md committed (`6a91298`); conventions match fix4 (status header,
  conflict rule, §-numbered spec, task/verification structure).
- All quantitative claims traced to today's verified evidence: n63
  benchmark table, scale_copy emitted-CUDA analysis, the occupancy
  capacity math (2,016 CTAs).
