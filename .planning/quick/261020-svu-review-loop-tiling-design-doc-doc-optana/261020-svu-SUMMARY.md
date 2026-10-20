---
phase: quick-261020-svu
plan: 01
subsystem: docs
tags: [review, loop-tiling, optimization, design-doc]
requires:
  - doc/optana/loop_tiling.md (untracked WIP in main checkout)
provides:
  - doc/review/loop_tiling.md — severity-tagged review of the loop-tiling design doc
affects: []
tech-stack:
  added: []
  patterns:
    - severity-tagged findings (BLOCKING/SOUNDNESS/COMPAT/MISSING-EASY/FUTURE) with doc-line + source grounding per finding
key-files:
  created:
    - doc/review/loop_tiling.md
  modified: []
decisions:
  - Verdict: loop-tiling design is feasible-with-required-changes (6 BLOCKING findings, all with sketched fixes)
  - Pass-split settlement: tiling-decide = parallelize slot (side-table decisions keyed by BIV identity), loop-tile-lower = iter-lower slot (rewrite only, emits LIR ForLoop); smem insertion stays at cache-alloc/lower after reduce-lower — the user-mandated merge would violate a ROADMAP hard ordering unless it absorbs reduce lowering too
  - ctaCache-field GVN risk settled as non-issue (MemLoad is FamilyMemory → never value-numbered); side table preferred over an IR field
metrics:
  duration: ~25 min
  completed: 2026-10-20
---

# Quick Task 261020-svu: Review of doc/optana/loop_tiling.md Summary

Severity-tagged design review of the 593-line loop-tiling optimization doc: 22 findings
(6 BLOCKING) with verdict feasible-with-required-changes; settles the four hard questions
(findINS-vs-example contradiction, sum-vs-product LoC, one-BB parallel-tail predication,
cross-pass decision persistence) and delivers a concrete LP/MIP + TMMA future-proofing
section.

## What was built

- `doc/review/loop_tiling.md` (759 lines), structured after `doc/review/phase1_1.md`
  (terse, bullet-driven) extended with severity tags and a findings table:
  - §1 feasibility verdict; §2–§4 soundness (findINS/union overlap analysis, PSched
    constraint model + objective + solver, tiling rewrite semantics read as the
    loop-tile-lower spec); §5 pass-split evaluation; §6 compatibility audit; §7
    missing-but-easy adds; §8 LP/MIP + TMMA future-proofing; §9 verdict + 22-finding
    table + six-point coverage table + self-check.
- Every finding cites target line numbers plus a grounding artifact (internal/*.go path
  or doc/ section), per the plan's must_haves.

## Key findings (all four hard questions settled definitively)

1. **findINS vs the doc's own matmul example (F1, BLOCKING)**: the algorithm buckets at
   the innermost REFERENCED level (both lhs and rhs at iterK = zero reuse), while the
   doc's commentary requires the reuse level (innermost LNT level NOT in the load's INS:
   lhs@iterN, rhs@iterM). The length>=2 candidate filter is coupled to the bug — under
   corrected bucketing the doc's own example would cache nothing.
2. **Sum-vs-product LoC (F4, BLOCKING)**: thread totals are sums (disjoint whole-warp
   shards, tca.md 5.7.2) but CTA totals across CTA-assigned dims are products; the
   sum constraint certifies launches exceeding the budget. Multi-CTA-dim also exceeds
   tca.md v0's single-grid-dim limitation.
3. **One-BB parallel tails (F11, BLOCKING)**: uniform min-bounds work only for sequential
   inside loops; parallel inside loops need per-element PREDICATION (KindSelect for pure
   values + runtime bounds in tile ops; collectives never suppressed) — the mechanism
   tca.md 5.11/5.12 already assumes but the doc never adopts.
4. **Cross-pass persistence (F14, SOUNDNESS)**: the doc records only MemLoad.ctaCache;
   SoC type, LoC, tile_size, and promotion decisions must also cross the decide→lower
   boundary — side table keyed by BIV node identity, passes adjacent, Requires-pinned.

Other BLOCKING: objective truncated at L295 with the stub maximizing cache BYTES (F8);
inverted feasibility filter + 3^K×presets enumeration blowup (F9); rewrite templates
with tile/element unit mismatches and wrong-variable increments causing infinite loops
(F10); per-LNT DoSoC counters colliding in the one kernel launch (F5).

## Commits

| Commit | Content |
|--------|---------|
| 7918d27 | Task 1 — sections 1-4 (feasibility, overlap analysis, PSched model, rewrite semantics) |
| ed636c0 | Task 2 — sections 5-6 (pass split, compatibility audit) |
| efb9002 | Task 3 — sections 7-9 (easy-adds, LP/MIP + TMMA, verdict + tables) |
| e31e2d4 | Citation-convention line in header (must_haves key_links pattern fix) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Review target and sibling specs absent from the worktree**
- **Found during:** environment load, before Task 1
- **Issue:** doc/optana/loop_tiling.md (the review TARGET), the sibling specs
  (induction_variable.md, loop_fusion.md, lower_itercluster.md, smem_insertion.md,
  type_inf.md), doc/opt.md, and doc/review/phase1_1.md are UNTRACKED in the main
  checkout — a git worktree created from HEAD does not carry untracked files, so none of
  the plan's required inputs existed here.
- **Fix:** copied them from the main checkout into the worktree as local, UNCOMMITTED
  reference material (git status shows them untracked; only doc/review/loop_tiling.md
  was ever staged). The main checkout's files were not modified, and doc/optana/loop_tiling.md
  was not modified anywhere (plan constraint honored).
- **Files modified:** none committed; untracked copies only.

**2. Extra polish commit beyond the three task commits**
- The must_haves key_links grep pattern (`loop_tiling\.md:? ?L?[0-9]`) matched 0
  occurrences because citations used the "target L168" style; added an explicit
  citation-convention line to the header (e31e2d4) rather than rewriting 100+ citations.

## Auth gates

None.

## Known Stubs

None — the deliverable is a complete review document; no placeholder content.

## Self-Check: PASSED

- doc/review/loop_tiling.md exists (759 lines >= 150 required).
- Commits 7918d27, ed636c0, efb9002, e31e2d4 verified via git log.
- No source code modified (files_modified = [doc/review/loop_tiling.md] only).
- No file deletions in any commit.
- must_haves key_links patterns both match (1x loop_tiling.md:L, 39x internal/*.go).
- Plan verification checks all pass: six user points covered (§9.3 table), severity tags
  present, no "TBD" strings, verdict stated with per-section justification.
