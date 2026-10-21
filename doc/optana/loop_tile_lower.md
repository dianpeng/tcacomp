# Loop Tile Lower

- This is the lowering half of the loop tiling pair
  - [1] Tile rewrite: rewrite each LNT into tile-between + tile-inside loops
  - [2] SoC numbering: derive the DoSoC dimension binding per assigned Iter
  - [3] IntrinsicGridMap/IntrinsicThreadMap materialization
  - [4] SMEM/FRAG cache insertion (the merged smem-insertion content, §6)
- ONE pass performs all four (author-mandated merge)
- The decide half is doc/optana/loop_tiling.md; this spec consumes its decisions

## §0 Header + Position

- Name: Loop Tile Lower
- Pipeline position
  - doc/opt.md is the ONLY authoritative pass-ordering guidance for this spec
    - ROADMAP and doc/tca.md pass orderings are NOT cited here
  - Occupies doc/opt.md's "loop tiling rewrite" entry
    - immediately AFTER "loop tiling" (the decide pass)
    - immediately BEFORE "scheduling" (which runs LICM)
  - Adjacency with tiling-decide is REQUIRED for decision-record lifetime (§1)
    - guaranteed today by the pipeline order itself: no pass runs in between
- Inputs
  - The IR graph post memopt/gvn/tiling-decide
  - The tiling-decide decision record side table (§1)
- Outputs
  - Rewritten LIR loop nests + cache nodes, ready for scheduling

## §1 Decision Record Interface (consumed)

- Side table produced by tiling-decide, per doc/optana/loop_tiling.md
  - (a) per-Iter entries: SoC type (CTA | WARP | seq), LoC, tile_size
    - keyed by BIV node identity
    - BIV identity is GVN-canonical and immune to a later fusion re-run
      deleting Iter prefix nodes
  - (b) per-site entries: promoted flag, tie level, FRAG/SMEM class
    - keyed by MemLoad node id
  - (c) DoSoC dimension binding: NOT in the table
    - derived by this pass's numbering step (§2)
- Record lifetime
  - The analysis cache is invalidate-all-by-default
  - The record survives ONLY because no pass runs between decide and lower
  - If a pass is ever inserted between them, pin
    Requires: tile-lower requires tiling
    - the mechanism exists today (pass-manager precondition checks,
      internal/opt/manager.go)
- No MemLoad.ctaCache field
  - Promotion marks live ONLY in the record; this pass performs the node
    surgery when it consumes the record
  - Rationale: choke-point discipline — a later pass writing a content field
    bypasses MutContext (all sanctioned mutations ride internal/ir/mutate.go)
  - GVN is NOT at risk either way (MemLoad is effectful, never value-numbered),
    the field is simply unnecessary

## §2 SoC Numbering (DoSoC)

- Per LNT, walk the Iter nodes outer → inner
  - Two dimension counters: one for CTA dims, one for WARP/THREAD dims
  - Each counter starts at 0 per LNT
  - Per-LNT restart is CORRECT: the runtime maps are per-LNT scoped objects
    (§7bis) — two LNTs' dim 0 bind to DIFFERENT map instances, no collision
- Sequential Iters get no dimension

## §3 Canonical Lowered Loop Forms

- The normative loop FORMS are the revised templates of doc/optana/loop_tiling.md
  ("Mechanical rewrite for tiling"): 0-BASED round enumeration in the between
  loops (induction starts at 0, stride LoC × T, NO map index in any induction
  variable, CTA-uniform trip counts), ranks appearing only as in-body offset
  locals (offset = rank × tile_size + round base), materialized per-tile
  uniform bounds (a_tile_max = min(tile_size, A − a_offset))
  - This spec does NOT duplicate those templates wholesale; it states the
    pass's obligations on top of them
- Obligations
  - (a) sequential-inside loops: bounded by the materialized uniform bound
  - (b) parallel loops (CTA-companion warp inside loops, the thread-parallel
      between loops of WARP dims)
    - the strided tile coverage of [0, a_tile_max) is normative
    - the materialized uniform bound is normative
    - the TAIL-GUARD is the loop primitive's OWN bailout — RESOLVED (§4,
      ruled 2026-10-20): the thr_loop materialization checks the offset
      global index against the allowed iteration space and skips the WHOLE
      body as a no-op; the compiler emits no guard and no branch
  - (c) every generated loop is emitted as LIR KindForLoop
    - so the NEXT pipeline stage (scheduling, which includes LICM) can hoist
      the tile-bound min/ceildiv expressions out of the tile loops
  - (d) ALL original-loop logic lands in the tile-inside loops
    - every use of an original InductionVariable is patched by RAUW to
      (tile_offset + inside_index) expressions
- SSA rewrite mechanics
  - New tile DimVar/BIV/ForLoop chains are NEW defs; originals are retired by
    RAUW, never edited in place
  - All graph edits ride the MutContext choke points only
    (SetDef/AddUse/RemoveUse/ReplaceAllUsesWith/SetType/Erase,
    internal/ir/mutate.go)
  - No new verifier invariant classes are introduced (§8)
- Integer bounds prerequisite — HARD dependency
  - u32 min/max/ceildiv builtins must exist
    - the registry today has ONLY f32 fminf/fmaxf (internal/ir/builtins.go)
    - without them NO bound expression (a_tile_max, ceil(A/T)) is representable
  - Recipe (three-place extension, doc/ir.md §2 pattern):
    - ir.Builtin registry entry: u32 min, u32 max
    - ceildiv may ride the identity (A + T − 1) / T over the existing BinOp div
    - typeinfer rule for each new builtin
    - codegen mapping

## §4 Loop Materialization & Parallel Tails (Issue #1 — RESOLVED 2026-10-20)

- RULING: the bounds bailout is an operation of the RUNTIME LOOP
  MATERIALIZATION, NOT of the IR
  - ALL parallel loops are GUARDED, cta_loop INCLUDED (amended 2026-10-21):
    allocated concurrency may EXCEED what the iteration space needs, so
    out-of-range ranks/threads whole-body no-op via the runtime loop
    materialization
  - The pass lowers loops to three primitive forms
    - thr_loop — thread-parallel element loop; its materialization checks
      whether the offset global index lies within the allowed iteration
      space; out-of-range iterations become a NO-OP for the ENTIRE iteration
      body (whole-body skip, compiler-inlined pure computation included)
    - cta_loop — CTA-parallel round loop; ALSO GUARDED, same whole-body no-op
      materialization as thr_loop
      - AMENDED (2026-10-21): the portion model allocates a full round stride
        (nCTA × T) whether or not the extent fills it, so an out-of-range CTA
        rank must whole-body no-op — the earlier "exact by launch
        construction, no bailout needed" wording is superseded
    - seq_loop — single-thread sequential; native bounds, no cross-thread
      concerns
  - Branches NEVER enter the IR: divergence exists only inside the
    materialized loop primitives of the generated code
  - Generated shape: the body is wrapped AS A UNIT (thr_loop(map, lo, hi,
    body)); codegen may inline the guard into the primitive's fixed template
    or emit the body as a __device__ function — the unit of no-op is the
    WHOLE body, never individual ops
- Uniformity: CTA-uniform trip counts remain the HARD invariant, enforced by
  construction
  - The bailout lives INSIDE a loop whose iteration count every thread of the
    group agrees on (ceil(tile_max / nThreads))
  - The scaffolding stays convergent; only bodies diverge; the next iteration
    reconverges
- Layering constraint (tile-level vs element-level) — REQUIRED for soundness
  - Tile-level cooperative ops (SMEM buffer fill, tile store-back) are
    emitted OUTSIDE the bailing element loop — a cooperative loadtile needs
    ALL threads of the group to participate (this is how a partial tile gets
    filled), with bounds CLAMPING inside the loader
  - Element-level ops live inside the bailing thr_loop body
  - A cooperative tile op emitted inside a thr_loop body is a spec violation
- Mask contract (the "sane mask" rule for warp-synchronous operations)
  - __shfl_down_sync(mask, ...) is UB if any lane named in mask does not
    execute the instruction — a naive full-warp mask on a tail iteration is
    exactly the deadlock/UB case
  - thread_map is the SINGLE supplier of masks: ANY runtime object that needs
    a participation mask gets it from the map
  - Sane-mask properties (all three REQUIRED)
    - EXACT — names precisely the lanes that will execute the collective
    - UNIFORMLY DERIVED — computed only from map-uniform quantities (base,
      tile_max), so lanes can never disagree about the mask; disagreement is
      the actual UB mechanism
    - PER-ITERATION — a function of the CURRENT iteration's (base, tile_max),
      read from loop-scaffolding state, not a loop-invariant
  - Closed form: ownership is lane-strided and the bound is warp-uniform, so
    the active set is always a LANE PREFIX
    - mask = (tile_max − base ≥ 32) ? 0xFFFFFFFF : ((1u << (tile_max − base)) − 1)
    - pure arithmetic — NO __ballot_sync/__activemask needed to derive it;
      deterministic, divergence-proof, CPU-testable
  - Full-warp fast path: within a full tile the mask is 0xFFFFFFFF — the
    warp-multiple LoC rule guarantees non-tail iterations get the fast path
    for free
  - Warp down-reduce over a prefix mask uses the standard tree pattern
    (boundary lanes may pull undefined values from outside the mask; only
    lane 0's accumulation is consumed)
- Reducer placement (composes with the above)
  - Init BEFORE the element loop; Combine AFTER it
  - Warp down-reduce at Combine time runs over the OWNERSHIP mask (which
    lanes own ≥ 1 partial) — also a prefix, since accumulation is
    lane-strided
- Resolution of the former open facets
  - Q1 IR model: one-BB is PERMANENT — control flow exists only inside
    runtime-materialized loop primitives, never in the IR
  - Q2 uniformity: hard invariant, enforced by the primitive's structure
  - Q3 mask/guard ownership: the loop materialization owns it — wholesale
    body skip; NO compiler predication at all (not even KindSelect wrapping)
  - Q4 predicated stores: DISSOLVED — an element-level store inside a bailing
    body does not execute; only tile-level loaders need internal clamping
    (layering constraint above)
  - Q5 collectives: tile-level collectives UNCONDITIONAL (outside the bailout
    scope); warp-level shuffles use EXACT map masks; together these give
    deadlock freedom

## §5 Intrinsic Nodes

- Two new LIR node kinds
  - IntrinsicGridMap(DoSoC, field) — query the LNT's GridConcurrencyMap
  - IntrinsicThreadMap(DoSoC, field) — query the LNT's ThreadMap
  - field ∈ {"index", "dimension"}
  - Result type: u32
  - Tier: LIR — the kinds exist only post-lowering
  - Semantics: per-LNT scoped — the queried object is the map constructed for
    the enclosing LNT nest (§7bis)
- Payload
  - The field selector is a STRING payload riding the callNode.name precedent
    (mixString hashing, internal/ir/value.go)
  - DoSoC is the SINGLE operand and must be a const u32
- REQUIRED companions
  - A new kind lands in three places by rule (doc/ir.md §2: node struct +
    catalog.go row + allocNode case); these kinds need SIX
  - (1) node struct + catalog.go row
  - (2) allocNode case
  - (3) verify shape case: operand count, DoSoC constness, field vocabulary
    (per-kind shape checks, internal/ir/verify.go)
  - (4) typeinfer rules for BOTH kinds
    - the typing catalog is CLOSED and errors on unknown kinds
      (internal/opt/typeinfer/rules.go) — without these two rules the
      post-pass re-typing (§6 epilogue) FAILS CLOSED
  - (5) legality-window row: the kinds exist only between this pass and codegen
  - (6) contentHash/contentEqual cases mixing DoSoC + field name
    (internal/ir/value.go) so GVN buckets stay faithful
- GVN-CSE
  - CSE of these intrinsics is SOUND (per-launch deterministic pure values)
    and DESIRABLE — repeated (DoSoC, field) queries collapse to one node

## §6 SMEM/FRAG Cache Insertion (merged smem-insertion content)

- Buffer materialization
  - MemNewSMem / FragNew allocation sites at each promoted site's TIE LEVEL
    (the LIR memory family, internal/ir/node_lir_memory.go)
  - Extents from the covered tile shapes AFTER demotion
    (the Cache.dims trailing dimensions, per the decide pass's cache-tie
    semantics)
  - StorageClass payload carried on the LIR memory nodes
- Site Rewrite Mechanics (how a promoted load is lowered)
  - Load-site surgery: the MemLoad's uses are RAUW'd to buffer
    accesses (MemLoadSMem/FragLoad with the covered dims' in-tile
    coordinates); the MemLoad is retired by Erase — this is the "node
    surgery" §1 defers to this pass
  - Fill emission point: inside the innermost covered dim's
    between-round body, immediately before the tile-inside loops
    (§6bis Phase 4); one fill per round
  - Buffer lifetime: ONE buffer per site, declared at the fill point
    (M3), refilled every round — the SMEM buffer is a static
    declaration reused across rounds, never re-allocated
  - Loads-only scope: v0 candidate sites are MemLoads only; stores
    never route through the cache (storetile/write-through is not in
    scope)
  - Buffer layout: plain row-major tile, NO padding/swizzle in v0
    (swizzle deferred to the TMMA future, F22)
- Copy variant selection (per doc/tca.md §5.11's variant table)
  - dims covered by cta_for/thread_for → SMEM cooperative copy:
    tca::rt::LoadTileCta
  - dims under seq_for → FRAG per-thread copy: tca::rt::LoadTileThread
- Sync
  - The runtime owns ALL synchronization — this pass NEVER emits barriers
  - Visibility, reuse safety, store ordering are runtime contract items (§7bis)
  - Compiler obligations: the uniform-call rule + bounds
- Budget enforcement
  - Σ SMEM bytes ≤ Max.cache
  - FRAG sites additionally require:
    - the covered tile dimensions fully unroll — register arrays stay in
      registers only when all indices are compile-time constants after
      unrolling; a non-unrolled FRAG tile spills to local memory, strictly
      worse than the GMEM load it replaced (doc/tca.md §5.8)
    - a per-thread FRAG byte budget (× threads/CTA) beside Max.cache
- Fallback (selection never fails)
  - Halve the reuse tile sizes; if still over budget, drop sites back to GMEM
  - Terminates: halving reaches 1, dropping empties the set
- Post-pass epilogue
  - Relies on the pipeline's typeinfer re-run for newly inserted lowered nodes
    (doc/opt.md "type inference — for newly inserted lowered node"; the pass
    is idempotent)
  - Which REQUIRES §5's typeinfer rules to exist, else the re-run fails closed

## §6bis Mechanical Lowering Procedure (phase by phase)

The full pass algorithm. Every step cites its ruling home; nothing here
introduces a new decision.

- PHASE 0 — IR prerequisites (before this pass can exist)
  - Intrinsic kinds: IntrinsicGridMap(DoSoC, field) /
    IntrinsicThreadMap(DoSoC, field) — LIR tier, string payload
    (callNode.name precedent), result u32, plus ALL SIX companions of
    §5 (node struct + catalog row, allocNode case, verify shape case,
    typeinfer rules — the closed catalog fails without them, legality
    row, contentHash/contentEqual)
  - u32 min builtin (registry + typeinfer + codegen; ceildiv via
    (A + T − 1)/T over existing BinOp div): every materialized
    tile_max = min(T, extent − offset) needs it
    - NOTE: the 0-based element-stride between loops eliminate ceil
      entirely from the loop bounds (bound = the raw extent); only min
      and the primitives' internal ceildiv remain
  - Loop annotations: the rewritten KindForLoop nodes carry their
    SoC / DoSoC / LoC as Input-list KeyedAttribute inputs (M1), so
    codegen selects the cta_loop / thr_loop / seq_loop materialization
    (§4)
- PHASE 1 — record fetch + skeleton
  - Requires: tile-lower requires tiling (pin now — the mechanism
    exists in the pass manager); fetch tiling-record from the analysis
    cache; FAIL CLOSED if absent
  - Re-derive the loopforest; per LNT (Roots() order) resolve every
    Iter -> BIV (DataDeps(iter)[0]) -> {SoC, LoC, TileSize}
    (internal/opt/tiling Record.Iters); a missing entry is an ERROR
    (adjacency guarantees completeness)
- PHASE 2 — DoSoC numbering (§2)
  - Per tree, walk outer -> inner; two counters (CTA dims, warp dims),
    both restarting per LNT; sequential iters get no dimension
- PHASE 3 — loop rewrite (per dim, original order)
  - TileSize == 1 -> NO tiling for that dim: it keeps a single loop
    and its IV maps directly to the global index
  - Otherwise the dim becomes between-round + inside loops; ALL between
    loops nest in dim order, ALL inside loops nest inside them, also in
    dim order (the loop_tiling.md template layout):
    - SEQ dim: between for (a = 0; a < A; a += T) (element stride,
      LoC = 1); inside sequential, bounded by the materialized
      a_tile_max
    - CTA dim: between for (a = 0; a < A; a += LoC*T) with LoC = MaxCTA
      (portion -1/full); in-body cta_rank =
      IntrinsicGridMap(DoSoC_d, "index"); a_offset = cta_rank*T + a.
      Inside = the COUPLED warp walk, ELEMENT stride:
      for (a_i = …; a_i += nThreads_coupled) with
      a_idx = a_offset + a_i + thread_rank_coupled — the coupled
      group's thread count = remaining warp portions x quantum x 32
    - WARP dim: between for (b = 0; b < B; b += nThreads*T) (each
      thread owns a whole T-chunk — the composition-semantics ruling);
      in-body thread_rank = IntrinsicThreadMap(DoSoC_d, "index");
      b_offset = thread_rank*T + b. Inside = SEQUENTIAL walk of the
      owned chunk
  - Materialized bounds before the inside loops:
    d_tile_max = min(T_d, extent_d − d_offset) — bounds ONLY; tail
    guarding is the primitive's whole-body no-op (§4), never an IR
    branch
  - Body rewiring: RAUW every InductionVariable use to the
    (d_offset + d_inside) expression (new defs); retire the original
    Iter chains through the MutContext choke points; zero new
    verifier invariant classes (§8)
- PHASE 4 — cache insertion (per promoted site, keyed by MemLoad id)
  - Recompute the site facts (INS chase, covered set = INS strictly
    inner than the recorded Tie, the CacheDims clamp) with the SAME
    deterministic chase as the decide pass — adjacency makes the
    results identical; the record needs no extension
  - FILL POINT: inside the innermost COVERED dim's between-round body,
    immediately before the tile-inside loops (Tie == nil -> the same
    rule at the outermost position). The tie position fixes WHICH
    rounds refill; the covered rounds fix WHERE the fill sits.
    Worked check: matmul lhs [T_k] is filled per k_o round and re-read
    by every m_i row
  - Buffer + fill AT that point: MemNewSMem (SMEM class) or FragNew
    (FRAG class), extents = product of covered tile sizes x element
    bytes; fill = tca::rt::LoadTileCta (any covered dim group-parallel)
    or LoadTileThread (all covered dims sequential) per §6's variant
    rule; source = the original tensor at the current round offsets
    (INS dims above the tie are baked-in fixed indices); the loader
    owns partial-tile clamping and runs under §4's uniform-call rule.
    NO barriers, ever (§6 sync)
  - Access rewrite: RAUW the MemLoad's uses to
    MemLoadSMem(buf, [covered dims' inside-coords]) / FragLoad(...)
    (both kinds already exist in the LIR catalog — M4); retire the
    MemLoad. Uncovered inner INS dims stay live loop indices BELOW the
    buffer — that re-read IS the reuse
  - Loads-only: the candidate set is MemLoad-based; v0 NEVER routes
    stores through the cache
- PHASE 5 — epilogue
  - Run the verifier (zero new invariant classes); rely on the
    pipeline's typeinfer re-run (Phase 0's rules make it pass);
    deterministic throughout (allocation-order walks, no search — §7)

- IMPLEMENTER DEFAULTS (M1-M4) — normative unless the author overrides
  - M1 loop-annotation mechanism: KeyedAttribute inputs on KindForLoop
  - M2 empty covered set (degenerate scalar cache): filter at the
    DECIDE pass — no reuse value, avoids a degenerate fill
  - M3 buffer node placement: at the fill point (one buffer, refilled
    per round)
  - M4 access node kinds: reuse the existing KindMemLoadSMem /
    KindFragLoad

## §7 Policies

- Dynamic shapes
  - Tile/group sizes are compile-time constants
  - Extents may be runtime values — handled by the remainder guard + the §4
    thr_loop bailout (whole-body skip in the loop materialization)
- Cross-LNT sites
  - Dedupe candidate sites kernel-wide by MemLoad node identity
  - A load whose INS spans multiple LNTs is evaluated against the UNION of the
    referenced iters' assignments
  - Never-bucketed loads are declared non-candidates (explicit, not silent)
- Determinism
  - DoSoC numbering order (outer → inner per LNT, §2) and the decide pass's
    preset ordering are inherited; this pass ADDS NO SEARCH of its own
- Inherited v0 default searcher policy (from the decide pass,
  doc/optana/loop_tiling.md composition semantics)
  - At most ONE CTA-assigned dim and at most ONE independent WARP dim per LNT
    - A POLICY within the constraint set, NOT a model rule
  - Same-category multi-dim composition (CTA+CTA, independent WARP+WARP) is
    FUTURE pending the combined-slicing arithmetic; the searcher may NOT emit
    such schedules until it is pinned
- Re-typing: see §6 epilogue

## §7bis Runtime Contract (what this pass assumes exists)

- Per-LNT scoped maps (RULED)
  - GridConcurrencyMap/ThreadMap are ordinary C++ template objects
  - Constructed at the start of that LNT's generated loop nests, destroyed at
    LNT exit; the next LNT constructs its own remap
  - This per-LNT scoping is WHY the per-LNT DoSoC counters of §2 are correct
- 1-D normalized launch (RULED)
  - gridDim.x = total CTAs, blockDim.x = threads/CTA (whole-warp multiple)
  - The runtime contract by which the maps learn the totals they remap
- Linearization
  - The multi-dim remap of the 1-D launch is runtime-OWNED
  - The compiler never inlines its own copy of the linearization arithmetic
- Allocator vocabulary (D3 — RULED 2026-10-21)
  - Per LNT exactly TWO mapping objects
    - ONE GridMapping over the CTA space
    - ONE ThreadMap over the per-CTA thread space
    - Each is MULTI-DIMENSIONAL and holds the COMPLETE partition knowledge of
      its space, ie all CTA dims for the GridMapping, all warp-category
      consumers for the ThreadMap
    - The earlier per-dim chained map description was a DESIGN TYPO — superseded
  - Tuple coordinates
    - cuda_linear_id computed FIRST, linear over the 1-D launch
    - Then reverse stride-shape decomposition per the mapping's shape
    - warp_index / lane_index / is_first_thread queries provided for
      warp-specialized reduce
  - Counting invariant (author-ruled)
    - p = m: CTA assignments = CTA dims
    - q = n: independent warp assignments
    - A CTA dim's coupled inside-warp is part of its CTA assignment, NOT
      double-counted
    - ThreadMap dimensionality = m coupled + n independent dims
  - Portion/exhaustion grammar, from doc/optana/loop_tiling.md
    - Portion sums 10 (CTA) / 8 (warp), -1 = derived remainder
    - PORTION_CTA / PORTION_WARP constants, assert-and-trap pre-conditions
    - Defaults: Ampere, no green context, 512 threads per CTA = 16 warps, CTA
      count rounded DOWN to the portion multiple (108 -> 100; A6000 84 -> 80)
  - Warp allocations are PER-CTA, cross-product composition per
    doc/optana/loop_tiling.md's composition-semantics section
  - FUTURE concurrent-LNT portion sharding is the ONE place chained/shard maps
    may return
- Tile-op obligations (runtime API must provide)
  - Visibility: SMEM writes visible to the whole CTA after the op returns
  - Reuse safety: a refill waits for prior reads (write-after-read)
  - Store ordering
  - Bounds checking — element-level guarding physically lives in the §4
    thr_loop bailout; tile-level partial-tile clamping lives INSIDE the
    cooperative loaders (both RULED per §4)
  - Masks: warp-synchronous runtime ops obtain participation masks from
    thread_map only, per the §4 mask contract (exact, uniformly derived,
    per-iteration)
  - Uniform call: cooperative ops called by every thread of the CTA the same
    number of times
- Sync ownership (RULED)
  - The runtime owns ALL synchronization/barriers; this pass never emits them
- Launch/budget validation at runtime
  - Launch config is EXTERNAL and static at compile time, per the portion
    grammar of doc/optana/loop_tiling.md; the runtime ASSERT-AND-TRAPs the
    pre-conditions (#CTAs multiple of 10, threads per CTA multiple of 8 warps)
    and guards every parallel loop rank (§4, ALL parallel loops)
  - SMEM budget via device-attribute queries — NEVER hardcode sm_86 ceilings
    (101,376 B); assert ≥ the design budget
  - The compiler emits code + runtime files; it does NOT drive nvcc

## §8 Verification (per-pass obligations)

- ZERO new verifier invariant classes after this pass
  - The use-def/def-use chain checks cover the rewired graph (RAUW keeps the
    chains consistent)
  - The intrinsics are pure leaves for the poset — no new edge classes
  - DCE two-root liveness handles them (pure values, live via consumers)
- Self-check list (pass exit conditions)
  - Every rewritten InductionVariable use has exactly the RAUW'd
    (offset + inside) form — no stale original-IV uses remain
  - Every FRAG site satisfies the fully-unroll condition (§6)
  - The budget fallback terminates (§6)
  - Every parallel-inside loop lowers to thr_loop carrying the §4 bailout
  - No cooperative tile op sits inside a bailing element loop (§4 layering)
  - Every warp-synchronous op's mask comes from thread_map (§4 mask contract)
  - No stale MemLoad remains for any promoted site (its uses are buffer
    accesses; the node is erased)
  - No original InductionVariable use survives (all RAUW'd to
    offset + inside expressions)
  - Every fill sits OUTSIDE bailing element loops (§4 layering)
  - ZERO barriers emitted (§6 sync)
