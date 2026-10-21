---
phase: quick-261021-gca
plan: 01
subsystem: opt-docs
tags: [loop-tiling, d3, composition, runtime-contract, design-doc]
requires:
  - doc/optana/loop_tiling.md (pre-patch revision)
  - doc/optana/loop_tile_lower.md (pre-patch revision)
provides:
  - Final author-ratified constraint set and portion grammar for the tiling decide pass
  - D3 RULED runtime mapping contract (ONE GridMapping + ONE ThreadMap per LNT)
  - 0-based induction + in-body offset rewrite templates
  - Single-term traffic_avoided objective
affects:
  - internal/opt/tiling (searcher must match the patched spec)
key-files:
  created: []
  modified:
    - doc/optana/loop_tiling.md
    - doc/optana/loop_tile_lower.md
tech-stack:
  added: []
  patterns:
    - portion grammar (PORTION_CTA=10 / PORTION_WARP=8, -1 derived remainder)
    - 0-based round enumeration between loops, ranks as in-body offset locals
decisions:
  - D3 closed RULED: per LNT exactly ONE GridMapping + ONE ThreadMap, multi-dimensional, complete partition knowledge
  - Objective is single-term traffic_avoided; parallelism term deleted (constant under full exhaustion)
  - cta_loop is guarded like thr_loop; exact-by-launch wording superseded
  - v0 searcher policy (one CTA dim + one independent WARP dim per LNT) is a POLICY, not a model rule; same-category multi-dim is FUTURE
metrics:
  duration: ~17m
  completed: 2026-10-21T04:06:03Z
---

# Quick Task 261021-gca: Final D3 / Composition Patch for Loop Tiling Docs Summary

Consolidated the author-ruling session into the two design docs: final constraint set with the 10/8 portion grammar and assert-and-trap pre-conditions, single-term memory-reuse objective, per-CTA warp composition semantics with two distinct warp-level codegen forms, 0-based rewrite templates, and D3 closed as RULED in the lowering spec.

## What Was Done

### Task 1 — doc/optana/loop_tiling.md (commit bb0be5e)

- Constraint block replaced with the final model: external static launch config (Ampere defaults, no green context, 512 threads/CTA = 16 warps, CTA count rounded DOWN to portion multiple, 108 -> 100; A6000 84 -> 80 override), named constants PORTION_CTA = 10 / PORTION_WARP = 8, runtime assert-and-trap pre-conditions (#CTAs multiple of 10, threads/CTA multiple of 8 warps = 256 minimum, PORTION_WARP = 8 because each portion is one warp at minimum config), K = 5 innermost SoC search window, portion grammar (CTA portions sum 10, warp portions incl. CTA-coupled halves sum 8, last assignment may be -1 derived remainder, LoC = portion x quantum), per-LNT full-launch exhaustion with future portion-sharding noted.
- Objective rewritten to single-term: score = traffic_avoided = SUM_sites bytes_per_tile x (reuse_count - 1); the weighted multi-objective sum and the parallelism term are deleted (parallelism constant under mandatory full exhaustion). Sample solver code updated to match.
- New "Composition Semantics, CTA x WARP" section: warp allocations PER-CTA with CTA-relative ranks; structural m x n cross product (each CTA-sliced m rank gets a full n space partitioned by that CTA's warps; iteration unit is the thread); two distinct warp-level codegen forms — (a) independent WARP between loop strides #threads x tile_size with each thread owning a whole tile chunk walked by a sequential inside loop, (b) CTA coupled inside loop strides #threads at element granularity.
- v0 searcher policy stated as a POLICY within the constraints, NOT a model rule: at most one CTA dim + one independent WARP dim per LNT; same-category multi-dim (CTA+CTA, WARP+WARP independent) is FUTURE — one short paragraph, combined-slicing arithmetic must be pinned before the searcher may emit such schedules.
- All three rewrite templates moved to 0-based induction + in-body offset form: between loops start at 0 with LoC x T stride (CTA-uniform trip counts, NO map index in any induction variable); ranks appear only as in-body offset locals (d_offset = d_rank x T + round base; CTA ranks from GridMapping uniform across the CTA, WARP ranks from ThreadMap per-thread); tail rounds whole-body no-op via loop_tile_lower.md §4; the stale F11 open-item note replaced by the RULED reference.

### Task 2 — doc/optana/loop_tile_lower.md (commit 588878e)

- §7bis: D3 block replaced — RULED 2026-10-21. Per LNT exactly ONE GridMapping + ONE ThreadMap, each multi-dimensional holding complete partition knowledge (per-dim chained map description superseded as a design typo); tuple coordinates via cuda_linear_id then reverse stride-shape decomposition; warp_index/lane_index/is_first_thread queries for warp-specialized reduce; counting invariant p = m (CTA dims), q = n (independent warp assignments), CTA dims' coupled inside-warps not double-counted, ThreadMap dimensionality = m coupled + n independent; portion/exhaustion grammar and defaults restated from loop_tiling.md; future concurrent-LNT portion sharding noted as the one place chained/shard maps may return.
- §4: amended — ALL parallel loops guarded, cta_loop included; allocated concurrency may exceed the iteration space's need; out-of-range ranks/threads whole-body no-op via the runtime loop materialization; "exact by launch construction, no bailout needed" wording superseded with the reason (portion rounds allocate full stride whether or not the extent fills it).
- §3: normative-forms reference updated to the rewritten 0-based templates (no map index in induction variables, ranks as in-body offset locals, materialized a_tile_max = min(tile_size, A - a_offset)); parallel-loop obligation parenthetical corrected (WARP-dim between loops are the thread-parallel ones; their inside walk is sequential).
- §7: inherited v0 default searcher policy added (POLICY, not a model rule; same-category multi-dim FUTURE).
- §7bis launch validation aligned to the external static portion grammar (replaces "grid = ceil(ext/T) computed at launch").

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Conceptual LoC sketch bounds in loop_tiling.md**
- **Found during:** Task 1, template rewrite
- **Issue:** The earlier conceptual loop sketch (under the LoC explanation) used `i < i_max_tile` as the between-loop bound, contradicting the final 0-based round-enumeration form (`d < D_extent`) being written into the templates — the done criterion requires constraint set and templates to be mutually consistent.
- **Fix:** Bounds changed to `i < I` / `j < J`, a note added that the rank enters in-body only, and the (2.1) inside-loop comment marked sequential over the owned chunk.
- **Files modified:** doc/optana/loop_tiling.md
- **Commit:** bb0be5e

**2. [Rule 1 - Bug] §7bis launch-validation echo of the superseded exact-by-launch model**
- **Found during:** Task 2, §4 amendment
- **Issue:** §7bis still said "grid = ceil(ext/T) computed at launch", which encodes the exact-by-launch construction the §4 amendment supersedes and contradicts the external static portion grammar the D3 RULED block restates.
- **Fix:** Reworded to external static launch config per the portion grammar with runtime assert-and-trap pre-conditions.
- **Files modified:** doc/optana/loop_tile_lower.md
- **Commit:** 588878e

**3. [Rule 1 - Bug] §3(b) parenthetical listed "warp loops" as parallel-inside**
- **Found during:** Task 2, §3 template-reference update
- **Issue:** Under the final templates the WARP dim's inside loop is sequential; its BETWEEN loop is the thread-parallel one. The old parenthetical contradicted the composition semantics.
- **Fix:** Obligation (b) now names "CTA-companion warp inside loops, the thread-parallel between loops of WARP dims".
- **Files modified:** doc/optana/loop_tile_lower.md
- **Commit:** 588878e

## Verification

- Task 1 greps pass: PORTION_CTA/PORTION_WARP present; ASSERT-AND-TRAP present; zero `parallel_weight`/`reuse_weight` occurrences with `score = traffic_avoided` in prose and sample code; PER-CTA present; chunk-granularity stride present; zero `auto a = grid_map...`/`auto a = thread_map...` induction forms remain; v0 policy phrased "NOT a model rule".
- Task 2 greps pass: zero UNDECIDED/OPEN markers in §7bis; "ONE GridMapping" + "ONE ThreadMap"; counting invariant (p = m, q = n, m coupled + n independent); "ALL parallel loops are GUARDED, cta_loop INCLUDED"; v0 policy present.
- Cross-doc: loop_tiling.md tail-guard notes reference loop_tile_lower.md §4; loop_tile_lower.md §3/§7/§7bis reference loop_tiling.md's templates, composition semantics, and portion grammar; no stale Max.cta/Max.threads_per_cta bounds remain; stale F11 open-item note gone.
- doc/review/ and doc/opt.md untouched (per constraints).

## Self-Check: PASSED

- doc/optana/loop_tiling.md modified — FOUND (commit bb0be5e)
- doc/optana/loop_tile_lower.md modified — FOUND (commit 588878e)
- Commits bb0be5e and 588878e present on worktree-agent-ad0cf2bb7030c8ebb — FOUND
- Working tree clean after both commits — VERIFIED
