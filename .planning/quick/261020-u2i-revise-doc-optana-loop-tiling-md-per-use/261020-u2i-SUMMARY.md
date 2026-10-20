---
phase: quick-261020-u2i
plan: 01
subsystem: opt-docs
tags: [loop-tiling, design-doc, review-apply, optana]
requires:
  - doc/review/loop_tiling.md (review findings F1/F4/F5/F8/F9/F10)
  - doc/optana/loop_tiling.md baseline 3fdb50a
provides:
  - Revised doc/optana/loop_tiling.md incorporating author rulings F1/F4/F5/F8/F9/F10
affects:
  - doc/optana/loop_tiling.md
key-files:
  modified:
    - doc/optana/loop_tiling.md
decisions:
  - Cache placement semantics restated as the author's three normative cache-tie rules (TIE / WHOLE DIMENSION LOAD / TILING DEMOTION) with the derived operational tie rule tie(p) = innermost LNT Iter NOT in INS(p); whole-LNT-INS loads tie above the outermost loop (allowed, not excluded)
  - "Author clarification (follow-up): the design distinguishes TWO levels — SET level (overlap analysis: innermost REFERENCED Iter; co-traversal grouping + admission filtering, home of the restored length>=2 set filter) vs TIE level (per-load placement: innermost non-referenced Iter, whole-dim coverage, tiling demotion); reuse lives ABOVE the set level. Supersedes the initial revision's fused single-level bucketing."
  - LoC bounds are externally provided thresholds (Max.cta = gridDim.x, Max.threads_per_cta = blockDim.x, multiple of 32); CTA totals are PRODUCTS across CTA-assigned dims, warp threads are SUMS of disjoint whole-warp shards per CTA; the 1-D grid/block mapping is a RUNTIME CONTRACT, not a compiler-side launch decision
  - Runtime map objects are per-LNT scoped (constructed at that LNT's nest start, gone when it ends); per-LNT DoSoC counters are correct because dimension indices are LNT-local, consumed only by that LNT's intrinsic map nodes
  - Objective completed as GMEM traffic avoided (bytes_per_tile x (reuse_count - 1)), explicitly NOT cache bytes; weighted score = parallel_weight * parallelism + reuse_weight * traffic_avoided
  - Search depth-limited to the innermost K consecutive loops per LNT (K default 5); loops outside the window default to sequential
  - All three rewrite templates use coherent tile-id enumeration (outer counts tile ids, offset = tile_id * tile_size, inside loops honor materialized a_tile_max); per-element tail-tile guarding deferred to predication discipline (review F11 open item, pointer note only)
completed: 2026-10-20
---

# Quick Task 261020-u2i: Revise doc/optana/loop_tiling.md per author rulings Summary

Revised doc/optana/loop_tiling.md applying review findings F1/F4/F5/F8/F9/F10 under the author's rulings: cache-tie placement semantics with tie-rule bucketing, external product/sum concurrency thresholds with per-LNT runtime scoping, a completed traffic-avoided objective with an innermost-K search window, and coherent tile-id rewrite templates.

## Tasks Completed

| # | Task | Commit | Key changes |
|---|------|--------|-------------|
| 1 | F1: cache-tie semantics + bucketing + matmul commentary | bebeb11 | Cache Tie Semantics subsection (3 normative rules + operational tie rule + matmul worked check); corrected lhs/rhs placement commentary; bucketing membership test inverted to tie level; length<2 set filter dropped (singleton sets kept) |
| 2 | F4+F5: constraint model + per-LNT runtime scoping | bc4b6f3 | Max.cta / Max.threads_per_cta external thresholds; product CTA accounting + sum warp-shard accounting; same bounds for every LNT with combined-total guarantee; 1-D mapping as runtime contract; per-LNT map scoping in runtime-object and SoC-numbering sections |
| 3 | F8+F9: objective completion + depth-limited search | 2d2c910 | traffic_avoided = SUM over sites of bytes_per_tile x (reuse_count - 1) with static reuse_count formula; weighted score; innermost-K window (default 5) in solver description and sample; fixed !is_feasiable guard, loc_of(soc) accessor, current_sched typo; sample term (2) now computes traffic avoided |
| 4 | F10: rewrite templates + node-list dedupe | bf04757 | Tile-id enumeration in intro / sequential / CTA / warp templates; a_tile_max bounds in all inside loops; CTA inside loop a_tile += nThreads with a_index = a_tile_offset + a_tile + thread_map.index; warp inside loop a_tile++ (infinite loop fixed); grid_map/thread_map-driven outer loops; predication pointer note (F11 open); IntrinsicGridMap/IntrinsicThreadMap node list appears once |
| 5 | Follow-up: set-vs-tie scoping per author clarification (partially reworks task 1) | 27b3456 | Bucketing reverted to SET level (innermost REFERENCED, "IS in INS" test, co-traversal grouping); length<2 set filter restored with no-partner rationale; TWO-level scoping block added to Cache Tie Semantics; worked check walks both levels ({lhs, rhs} @ iterK -> lhs tie iterN, rhs tie iterM) |

## Follow-up Correction (author clarification)

After the initial 4 tasks, the author clarified the design distinguishes TWO levels that the first revision had wrongly fused; commit 27b3456 applies the correction:

- SET level (MemLoad Overlap Analysis) = innermost REFERENCED Iter. Union-algorithm bucketing restored to the ORIGINAL membership test (scan inner->outer, bucket at first level IS in the load's INS); sets group CO-TRAVERSED loads sharing their innermost referenced dimension (matmul: {lhs, rhs} @ iterK).
- The "If S's length < 2, skip" set filter restored in step (4): a singleton set = a load with no co-traversed partner (e.g. a map that only modifies a single operand) -> no reuse at all -> not worth caching. Per-load 1-dimension-access filter and Max.cs cap unchanged.
- TIE level (per load) = innermost NON-referenced Iter (the operational tie rule) — actual cache placement, whole-dimension coverage, tiling demotion. Reuse lives ABOVE the set level: the set's co-traversed loads get their caches tied above, so the k-sweep turns into SMEM hits that survive the enclosing sweeps.
- Worked check updated to walk BOTH levels: set {lhs, rhs} @ iterK (passes the >=2 filter) -> placements lhs tie->iterN (covers k whole), rhs tie->iterM (covers [n][k] whole).

Net effect vs the original pre-revision doc: the bucketing algorithm text is back to set-level semantics (with co-traversal rationale added), while the tie semantics/placement commentary (task 1's Cache Tie Semantics rules, corrected lhs/rhs placements) stand unchanged.

## Deviations from Plan

Plan tasks 1-4 executed as written (one judgment call within task 3's scope: the solver sample's term (2) was switched from summing cache bytes to computing traffic avoided so the sample matches the objective added in the same task — review F8 flags the old sample as maximizing the wrong quantity). Post-plan author clarification then partially superseded task 1's bucketing/filter changes — applied as follow-up commit 27b3456 (see Follow-up Correction above). Final doc state reflects the clarified two-level (set vs tie) design.

## Out of Scope (per task context)

- F11 predication spec for parallel tails: only a brief pointer note added in the CTA template ("handled by the predication discipline (one basic block), open item from the review (F11), not specced here").
- Review findings other than F1/F4/F5/F8/F9/F10 (F2, F3, F6, F7, F12-F22) remain unaddressed in the doc — separate rulings needed.

## Verification

- F1 (final state after follow-up): 'Cache Tie Semantics' present; RULE 1 (TIE) / RULE 2 (WHOLE DIMENSION LOAD) / RULE 3 (TILING DEMOTION) stated normatively; TWO-level SET vs TIE scoping block present; operational tie rule and two-level worked check ({lhs, rhs} @ iterK -> lhs tie iterN, rhs tie iterM) present; commentary corrected ("cover the k dimension, whole, later demoted to the k tile" / "must cover the n and k dimensions, [n][k]"); bucketing at SET level ("FIRST scanned level whose Iter IS in p's INS"); 'If S's length < 2, skip' filter restored; the 1-dimension-access filter, Max.cs cap, and tile_size==1 exclusions kept.
- F4+F5: Max.cta / Max.threads_per_cta / PRODUCT accounting / RUNTIME CONTRACT / per-LNT scoping greppable in both the runtime-object and SoC-numbering sections; no CTA.max_loc / WARP.max_loc remains.
- F8+F9: objective complete (no line ends at "given by"); innermost-K window stated in the solver description, the hardcoded-loop spec, and the sample; !is_feasiable, loc_of(soc), current_sched fixed; no currrent / total_cache / cache_weight remains.
- F10: a_tile_offset = a * tile_size_a forms, a_tile += increments, a_tile_max bounds present in all inside loops; zero hybrid bound-in-tiles/stride-in-elements forms remain (i+=i_tile_size / a += tile_size_a / a += warp_loc / bare a++ all gone); 'List of new IR nodes' appears exactly once.
- Hygiene: only doc/optana/loop_tiling.md modified; one atomic docs(261020-u2i) commit per task; no file deletions in any commit; working tree clean after the final commit.

## Self-Check: PASSED

- Commits bebeb11, bc4b6f3, 2d2c910, bf04757, 27b3456 verified on branch worktree-agent-abd377c0f14115dec via git log.
- doc/optana/loop_tiling.md is the sole path changed in all five commits (git show --stat verified per commit); file exists and contains all anchors above, including the follow-up's set-level bucketing, restored length<2 filter, and SET/TIE scoping block.
