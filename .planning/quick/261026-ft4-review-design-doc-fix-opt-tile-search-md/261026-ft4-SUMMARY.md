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

## Deviations

- Review executed inline rather than via spawned gsd-planner/gsd-executor: the analysis required the loop-tiling rulings context already loaded in-session; GSD guarantees (PLAN/SUMMARY/STATE/atomic commits) preserved.
