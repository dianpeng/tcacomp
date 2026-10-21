---
phase: quick-261021-gca
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - doc/optana/loop_tiling.md
  - doc/optana/loop_tile_lower.md
autonomous: true
requirements:
  - R1: Record the final constraint set and scheduler grammar in loop_tiling.md (author-ratified 2026-10-20/21)
  - R2: Record the composition semantics (per-CTA warp space, cross-product structure, two thread-stride codegen forms)
  - R3: Rewrite the rewrite templates to 0-based induction + in-body offset form
  - R4: Close D3 in loop_tile_lower.md §7bis (one map per kind per LNT, RULED) + §4 all-loops-guarded amendment + v0 searcher policy
  - R5: Single-term objective (memory reuse only)

must_haves:
  truths:
    - "loop_tiling.md constraint block states: per LNT at least one CTA-assigned dim (CUDA thread model requirement); at most K=5 innermost dims SoC-assigned (default scheduler config); SUM of CTA portions = 10 and SUM of warp portions = 8 where 10/8 are named constants (future adaptation), the last portion -1 = derived remainder; runtime assert-and-trap pre-condition: provided CTA count multiple of 10, threads per CTA multiple of 8 warps (256 threads)"
    - "loop_tiling.md objective is single-term: maximize memory reuse (GMEM traffic avoided); the weighted multi-objective sum is deleted"
    - "loop_tiling.md states the composition semantics: warp allocations are PER-CTA (every CTA's full thread space re-partitioned; warp ranks are CTA-relative); CTA-rank x per-CTA thread-rank expresses the m*n cross product (each CTA-sliced m rank gets a full n space, partitioned by that CTA's warps); independent WARP dim's between loop strides tile-chunk granularity (#threads x tile_size, each thread owns a tile_size chunk) with SEQUENTIAL inside loop over the chunk's elements; CTA dim's coupled inside loop strides ELEMENT granularity (#threads, each thread owns rows); the two warp-level uses have DIFFERENT codegen"
    - "loop_tiling.md templates use 0-based induction with in-body offsets: between loops start at 0, stride LoC x tile_size (or #threads x tile_size for WARP dims), per-CTA/per-thread ranks appear only as in-body offset locals (rank x tile_size + round base); no map index in any induction variable"
    - "loop_tile_lower.md §7bis marks D3 RULED: per LNT exactly ONE GridMapping + ONE ThreadMap, multi-dimensional, holding the complete partition knowledge; tuple coordinates via reverse stride-shape decomposition of cuda_linear_id; warp_index/lane_index/is_first queries for warp-specialized reduce; counting invariant p = m (CTA assignments = CTA dims) and q = n (independent warp assignments), coupled insides not double-counted; ThreadMap dimensionality = m coupled + n independent"
    - "loop_tile_lower.md §4 is amended: ALL parallel loops are guarded (including cta_loop), since allocated concurrency may exceed what the iteration space needs; out-of-range ranks/threads become whole-body no-ops"
    - "Both docs state the v0 default searcher policy: at most one CTA-assigned dim and at most one independent WARP dim per LNT (a policy within the constraints, not a model rule); same-category multi-dim composition (CTA+CTA, WARP+WARP independent) is marked FUTURE with the note that its combined-slicing arithmetic (product factors of the level's space, portions as selection grammar) must be pinned before the searcher may emit such schedules"
    - "Defaults recorded: Ampere default config rounded DOWN to portion multiple (108 -> 100 CTAs), 512 threads per CTA = 16 warps, no green context; target-specific configs override (e.g. A6000 84 -> 80)"
  artifacts:
    - path: "doc/optana/loop_tiling.md"
      provides: "Final constraint set, composition semantics, corrected templates, single-term objective"
    - path: "doc/optana/loop_tile_lower.md"
      provides: "D3 RULED runtime contract, guarded-loops amendment, v0 searcher policy"
  key_links:
    - doc/opt.md (pipeline authority — unchanged)
    - doc/review/loop_tiling.md (historical grounding; do not modify)

tasks:
  - id: 1
    title: "loop_tiling.md: constraints, objective, composition semantics, templates"
    files: [doc/optana/loop_tiling.md]
    action: |
      Locate by content anchors (line numbers have shifted across revisions).

      1) CONSTRAINT BLOCK (anchor: "Max.cta" / "Max.threads_per_cta" in the PSched
         constraints): replace with the final model:
         - External launch config, static at compile time, from the target GPU
           and green-context SM slicing; defaults: Ampere, no green context,
           512 threads per CTA (= 16 warps), CTA count rounded DOWN to the
           portion multiple (108 -> 100); named constants PORTION_CTA = 10 and
           PORTION_WARP = 8 (future adaptation).
         - Runtime hard pre-condition, assert-and-trap: provided #CTAs is a
           multiple of 10; threads per CTA a multiple of 8 warps (256 threads
           minimum — the reason PORTION_WARP = 8: at minimum config each
           portion is exactly one warp).
         - Per LNT: at least ONE CTA-assigned dim (else the CUDA thread model
           cannot execute it); at most K = 5 innermost dims SoC-assigned
           (default scheduler config, the search window).
         - Portions: CTA-assigned dims' portions SUM to 10; warp-assigned
           (incl. coupled) SUM to 8; the LAST assignment in each category may
           be -1 = derived remainder (searcher decides n-1); LoC of a dim =
           portion x quantum (quantum = total/10 CTAs or total/8 warps).
         - Every LNT under the same bounds; v0 sequential LNTs each remap the
           full launch (per-LNT exhaustion); portion-sharding across LNTs is
           the future concurrent-LNT mechanism.
      2) OBJECTIVE (anchor: "traffic_avoided" / the weighted score): delete the
         weighted multi-objective sum; the objective is SINGLE-TERM: maximize
         memory reuse = GMEM traffic avoided = SUM_sites bytes_per_tile(site) x
         (reuse_count(site) - 1). Note: under mandatory full exhaustion total
         parallelism is constant by construction, so the parallelism term is
         dropped entirely. Update the sample solver code accordingly (score =
         traffic_avoided).
      3) COMPOSITION SEMANTICS — new subsection under the assignment/rewrite
         material:
         - Warp allocations are PER-CTA: every CTA holds the full thread space
           and re-partitions it; warp ranks are CTA-RELATIVE, never global.
         - The m x n cross product arises structurally: CTA ranks slice the
           CTA space; within EACH CTA its warps slice the n space — every
           CTA-sliced m rank (m1/m2/m3...) gets a full n dimension space,
           partitioned by that CTA's warp group. Iteration unit is the thread.
         - Two distinct warp-level codegen forms:
           (a) independent WARP dim's between loop: stride = #threads x
               tile_size — each thread owns a whole tile_size CHUNK of that
               dimension; its inside loop is SEQUENTIAL over the chunk's
               elements;
           (b) CTA dim's coupled inside loop: stride = #threads at ELEMENT
               granularity — each thread owns rows/elements of the tile
               interior.
         - v0 default searcher policy (within the constraints, not a rule):
           at most one CTA-assigned dim and at most one independent WARP dim
           per LNT. Same-category multi-dim (CTA+CTA, independent
           WARP+WARP) is FUTURE: its combined-slicing arithmetic (product
           factors of the level's space; portions as the selection grammar)
           must be pinned before the searcher may emit such schedules.
      4) TEMPLATES (anchor: "Mechanical rewrite for tiling"): rewrite all three
         template code blocks to the 0-based induction + in-body offset form:
         - ALL between loops: `for (a = 0; a < A_extent; a += LoC * T)` —
           induction starts at 0, uniform for every CTA (same trip count —
           the CTA-uniform invariant); NO map index in any induction variable.
         - In-body offset locals: `const auto d_offset = d_rank * T + a;`
           where d_rank is the CTA's/thread's rank in dimension d's
           allocation (rank scaled by tile size + round base). CTA dims:
           rank from GridMapping, uniform across the CTA's threads. WARP
           dims: rank from ThreadMap, per-thread.
         - WARP dim's between loop: stride #threads x tile_size (chunk
           ownership); its inside loop sequential over the chunk.
         - CTA dim's coupled inside loop: element stride #threads.
         - Sequential: unchanged (0-based, uniform bound via materialized
           tile_max).
         - Keep the tail-guard note: out-of-range ranks/threads become
           whole-body no-ops via the loop materialization (loop_tile_lower.md
           §4).
    verify: "greps: PORTION_CTA/PORTION_WARP or the 10/8 constants, assert-and-trap, single-term objective (no parallel_weight), 'PER-CTA', chunk stride, 0-based templates (no 'a = grid_map.index' style induction), v0 searcher policy"
    done: "Constraint set, composition semantics, and templates are mutually consistent"

  - id: 2
    title: "loop_tile_lower.md: D3 RULED, guarded-loops amendment, searcher policy"
    files: [doc/optana/loop_tile_lower.md]
    action: |
      1) §7bis — replace the "OPEN — allocator vocabulary (D3, UNDECIDED)"
         block with RULED content:
         - Per LNT exactly TWO mapping objects: ONE GridMapping (CTA space)
           and ONE ThreadMap (per-CTA thread space); each is multi-dimensional
           and holds the COMPLETE partition knowledge of its space (all CTA
           dims / all warp-category consumers). The earlier per-dim chained
           map description was a design typo — superseded.
         - Tuple coordinates: cuda_linear_id computed first (linear over the
           1-D launch), then reverse stride-shape decomposition per the
           mapping's shape; warp_index / lane_index / is_first_thread queries
           provided for warp-specialized reduce.
         - Counting invariant (author-ruled): p = m (CTA assignments = CTA
           dims), q = n (independent warp assignments); a CTA dim's coupled
           inside-warp is part of its CTA assignment, NOT double-counted;
           ThreadMap dimensionality = m coupled + n independent dims.
         - Portion/exhaustion grammar (from loop_tiling.md): sums 10/8, -1
           remainder, constants, assert-and-trap pre-conditions, defaults.
         - Warp allocations are per-CTA (cross-product composition per
           loop_tiling.md's composition-semantics section).
         - Future concurrent-LNT portion sharding noted as the one place
           chained/shard maps may return.
      2) §4 — amend the primitive descriptions: ALL parallel loops are
         GUARDED (cta_loop included): allocated concurrency may exceed the
         iteration space's need; out-of-range ranks/threads whole-body no-op
         via the runtime loop materialization. Remove/adjust the "exact by
         launch construction, no bailout needed" wording for cta_loop.
      3) §3 or §7 — add the v0 default searcher policy line (at most one CTA
         dim + one independent WARP dim per LNT; same-category multi-dim
         FUTURE pending combined-slicing arithmetic).
      4) Update the 0-based/offset-in-body obligation in §3 if the current
         text still references the old induction form (it should already
         reference loop_tiling.md's templates — ensure the reference holds
         after task 1's rewrite).
    verify: "greps: no 'UNDECIDED' remains in §7bis; 'ONE GridMapping' + 'ONE ThreadMap'; counting invariant; cta_loop guarded; v0 policy present"
    done: "D3 fully RULED; both docs consistent"
