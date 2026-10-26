---
status: complete
quick_id: 261026-ft4
date: 2026-10-26
---

# Quick Task 261026-ft4 — Summary

**Task:** Review doc/fix/fix-opt-tile-search.md (design for the loop-tiling schedule search rewrite) for internal consistency, consistency with settled rulings, and consistency with the implemented internal/opt/tiling.

**Outcome:** COMPLETE — findings written to **doc/review/fix-opt-tile-search.md**. Verdict: direction endorsed (diagnosis of the current search confirmed on all four counts against search.go), but **not implementable as written** — 4 CRITICAL / 5 HIGH / 7 MEDIUM findings.

## Key findings

- **C1** Multi-dim same-category composition admitted (MAPM 3+3) but unpinned — MALoC's sum-of-portions model contradicts the product-based CTA grid decomposition (F4); this was explicitly ruled future-until-pinned. Either keep v0 ≤1+≤1 or pin the portion→extent→product grammar.
- **C2** reuse_score = bytes/MaxSRAM measures SRAM *occupancy*, not traffic avoided (F8 regression); blind to the reuse multiplier (64 KB @ reuse 2 == 64 KB @ reuse 64) and rewards waste.
- **C3** GoC = ΣLoC/goc_max is quota consumption, not granularity — it scores 1 dim @8 and 2 dims @4+4 identically, the exact case its own 28×1024 example argues for; re-introduces the parallelism term deleted as constant-under-exhaustion.
- **C4** FRAG (per-thread) and SMEM (per-CTA) merged into one 64 KB pool — unit error; keep MaxCache + MaxFragPerThread separate (F6).
- **HIGH:** pre- vs post-tiling assignment-slot ambiguity; M=4 vs 5 arithmetic conflict; cross-LNT budget policy unspecified (R3 running floors); eviction is greedy spill with unspecified pop order + a changed error contract (the 2026-10-24 "tiling never disabled" ruling needs a conscious extension); tier rules (R2/C1b refusal + window-bytes group factor) absent.
- **MEDIUM:** determinism/tie-breaking; LoC semantics change breaks the record/verify contract (portion index vs absolute count); F2 coupled-portion floor vs the LoC=8 preset; goc_max formula issues; tile presets {32,64,128} vs current {32,64,128,256}; naming continuity with Params; vacuous constraint (5).

Also endorsed: TBL/TIL structural table, CuMAP, per-dim tiles / searched LoC / eviction salvage / MAPM generalization as goals, deferring site discovery to overlap.go, and the Implementation section's module split (with a concrete mapping onto searchSchedule/evalSchedule/judgeSite/Params).

## Files

- Created: `doc/review/fix-opt-tile-search.md` (the review — deliverable)
- Created: this SUMMARY.md; PLAN.md
- Updated: `.planning/STATE.md` (quick-task row + Last activity)
- **Not touched:** doc/fix/fix-opt-tile-search.md (the author's WIP — review only, per task scope)

## Round 2 (same day) — re-review after the author's revision

- [1] CTA side RESOLVED (absolute-count LoC, Π ≤ MaxLoC.cta, gridDim.x = max-over-LNTs Π = the F4 product model). WARP side regressed into a NEW CRITICAL: MALoC.thr = Mul switches warp composition from the implemented disjoint-shard (sum) model to tensor-product — lower.go/ThreadMap/R2 window-bytes all assume shards; doc now contains both models. Recommend warps stay Σ ≤ 32.
- [2] reuse_score = Π(overlapped T_d): improved (tile volume) but not "actual reuse" — no re-traversal multiplier, no elem bytes, no −1; recommend bytes×(reuse−1) (judgeSite computes both) and drop /MaxSRAM normalization (traffic ≫ budget).
- [3] GoC = partition count: intent ACCEPTED (now encodes the granularity direction; "3+3" comment finally coherent) — but formula text still names the product variables, and goc_max's 2×N arm caps the score at 0.75 for M=4; use goc_max=6.
- [4] Rebutted: registers ≠ L1 on sm_86 (separate 256KB RF; ≤255 regs ≈1KB/thread); counterexample — 60KB/thread FRAG + 4KB SMEM passes the 64KB pool then spills to local (L1/L2) = worse than GMEM. Single pool acceptable ONLY with a per-thread FRAG ceiling retained; two budgets cleanest.
- [5] M=4 ✓ but two stale "5" lines remain.
- NEW: N1 −1 exhaust rule dangling (no quota pool left to exhaust); N2 presets now the parallelism ceiling — ~10× regression (10 CTAs / 8 warps vs 100 CTAs / 16 warps at old defaults); N3 MaxLoC.cta=2^31−1 vacuous; N4 blockDim unit (warps→threads ×32) + name typos; N5 MASRAM scope still unstated.
- Positive: absolute-count LoC converges toward the record contract (M2 narrows to loosening verify's LoC==MaxCTA check); launcher max-over-LNTs + guard-bail is a good addition.

## Deviations

- Review executed inline rather than via spawned gsd-planner/gsd-executor: the analysis required the loop-tiling rulings context already loaded in-session; GSD guarantees (PLAN/SUMMARY/STATE/atomic commits) preserved.
