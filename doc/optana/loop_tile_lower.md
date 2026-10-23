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
    Requires: tile-lower requires tiling AND tiling-verify
    - F7 (ruled 2026-10-21): the per-iter feasibility re-checks (CTA
      LoC == MaxCTA, warp-portion multiples, v0 policy) live in
      tiling-verify; a corrupted record must NOT reach the rewrite
      unvalidated
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
  uniform bounds in the SATURATING form (off' = umin(a_offset, A);
  a_tile_max = umin(tile_size, A − off') — F1, ruled 2026-10-21)
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
  - CLARIFIED (R1, 2026-10-21): a PER-THREAD fill (tca::rt::LoadTileThread,
    FRAG class) is NOT a cooperative op — it is an element-level op and sits
    INSIDE the element loop defining its per-thread coordinate (§6bis
    PHASE 4); only cooperative LoadTileCta fills are bound by the
    outside-the-element-loop rule
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
  - MemNewSMem / FragNew allocation sites at each promoted site's FILL
    POINT, variant-dependent per R1 (the LIR memory family,
    internal/ir/node_lir_memory.go; M3 — one buffer, refilled)
  - Extents per covered dim AFTER demotion (the Cache.dims trailing
    dimensions, per the decide pass's cache-tie semantics), sized by
    the R2 WINDOW rule (ruled 2026-10-21)
    - WARP dim d -> #threads_d x T_d (the whole group's round window,
      NOT the single-thread chunk)
    - SEQ dim -> T_d
    - total buffer bytes = product over covered dims x element size
  - StorageClass payload carried on the LIR memory nodes
- Site Rewrite Mechanics (how a promoted load is lowered)
  - Load-site surgery: the MemLoad's uses are RAUW'd to buffer
    accesses (MemLoadSMem/FragLoad with the covered dims' in-tile
    coordinates); the MemLoad is retired by Erase — this is the "node
    surgery" §1 defers to this pass
  - Fill emission point (RULED 2026-10-21, R1 — VARIANT-DEPENDENT; the
    former unconditional "immediately before the tile-inside loops"
    rule is SUPERSEDED: it placed fills whose coordinates contain
    tile-INSIDE dimvars OUTSIDE the loops defining them — undefined at
    the mandated position, the HI-01 inconsistency)
    - LoadTileThread (FRAG): INSIDE the element loop that DEFINES its
      per-thread coordinate — refills once per epoch of its fill key
    - LoadTileCta (SMEM window): at the innermost covered dim's
      between-round body, immediately before the tile-inside loops;
      CTA-UNIFORM coordinates (round bases) ONLY
    - Full placement mechanics: §6bis Phase 4
  - Buffer lifetime: ONE buffer per site, declared at the fill point
    (M3), refilled per fill-key epoch — per round for LoadTileCta, per
    fill-key element for LoadTileThread (R1); the SMEM buffer is a
    static declaration reused across refills, never re-allocated
  - Loads-only scope: v0 candidate sites are MemLoads only; stores
    never route through the cache (storetile/write-through is not in
    scope)
  - Buffer layout: plain row-major tile, NO padding/swizzle in v0
    (swizzle deferred to the TMMA future, F22)
- Copy variant selection / TIER rule (RULED 2026-10-21, R2; per
  doc/tca.md §5.11's variant table) — classify by the SoC of the
  site's COVERED dims
  - ANY covered dim CTA-assigned -> site REFUSED (not promoted): a
    CTA-covered tile has no coherent fill variant under the v0
    mechanics
  - PRECEDENCE (ruled 2026-10-23, before the WARP tier below): a site
    with a thread-varying NON-covered source coordinate at the fill
    point — ANY referenced iter of a non-covered tensor position
    carrying SoC CTA or WARP — takes the FRAG class REGARDLESS of the
    covered dims' SoC (the worked example's own principle: the
    per-thread coupled/striped walk makes that coordinate vary by
    thread at the fill point, and R1's CTA-uniform-coordinates-only
    rule makes the cooperative SMEM fill incoherent for such sites).
    FRAG over a WARP-covered dim prices and sizes at T_d per thread
    (the per-thread window — NEVER the LoC x 32 group window), the
    access is the plain inside coordinate for EVERY covered dim, and
    the WARP-covered dim's fill coordinate rides the AXES path as its
    round base (rank x T + round; the biv path cannot carry it — the
    fill store's biv operands are pinned to between dimvars)
  - ANY covered dim WARP-assigned (and the precedence above did not
    fire) -> SMEM WINDOW class: cooperative
    tca::rt::LoadTileCta, window-sized buffer (R2 sizing above)
  - ALL covered dims sequential -> FRAG class: per-thread
    tca::rt::LoadTileThread, fill key per-thread (e.g. matmul lhs)
- Sync
  - The runtime owns ALL synchronization — this pass NEVER emits barriers
  - Visibility, reuse safety, store ordering are runtime contract items (§7bis)
  - Compiler obligations: the uniform-call rule + bounds
- Budget enforcement (RULED 2026-10-21: R3 scope + F6 constant)
  - Σ SMEM bytes ≤ Max.cache, KERNEL-WIDE: the sum accumulates over ALL
    LNTs' promoted sites — the decide pass, tiling-verify and this
    pass's budget resolution gate the SAME kernel-wide sum; a per-LNT
    scope is WRONG (two LNTs each at the cap would publish an
    infeasible record)
  - FRAG sites additionally require:
    - the covered tile dimensions fully unroll — register arrays stay in
      registers only when all indices are compile-time constants after
      unrolling; a non-unrolled FRAG tile spills to local memory, strictly
      worse than the GMEM load it replaced (doc/tca.md §5.8)
    - Σ per-thread FRAG bytes ≤ MaxFragPerThread (F6), default 256
      bytes/thread — participates in the same halve/drop fallback as
      the SMEM budget
- Fallback (selection never fails)
  - Halve the reuse tile sizes; if still over budget, drop sites back to GMEM
  - DROP RULE (F4, ruled 2026-10-21): a site is dropped when ANY of its
    covered iters reaches tile 1 — NOT only when all of them do
    (matches the judge's promotion gate; the all-1 variant can strand a
    promoted site at a covered tile of 1 and make the site lowering
    fail closed on feasible input)
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
    tile_max = umin(T, extent − umin(offset, extent)) needs it (the
    SATURATING form, F1, ruled 2026-10-21)
    - NOTE: the 0-based element-stride between loops eliminate ceil
      entirely from the loop bounds (bound = the raw extent); only min
      and the primitives' internal ceildiv remain
  - Loop annotations: the rewritten KindForLoop nodes carry their
    SoC / DoSoC / LoC as Input-list KeyedAttribute inputs (M1), so
    codegen selects the cta_loop / thr_loop / seq_loop materialization
    (§4)
- PHASE 1 — record fetch + skeleton
  - Requires: tile-lower requires tiling AND tiling-verify (F7, ruled
    2026-10-21 — the per-iter feasibility re-checks live in
    tiling-verify and a corrupted record must NOT reach the rewrite
    unvalidated; pin both preconditions now, the mechanism exists in
    the pass manager); fetch tiling-record from the analysis cache;
    FAIL CLOSED if absent
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
      Inside = the COUPLED warp walk, ELEMENT stride — the FLAT-INDEX FORM
      (ruling 2026-10-23, superseding the former
      "a_idx = a_offset + a_i + thread_rank_coupled"): a thread iterator
      with INIT = the NO-ARG flat thread query IntrinsicThreadMap("index")
      (the thread's absolute flattened 1-D rank — never an inlined
      threadIdx.x), STEP = nThreads_coupled (the coupled group's thread
      count = remaining warp portions x quantum x 32), BOUND = a_tile_max:
      for (a_i = flat_index(); a_i < a_tile_max; a_i += nThreads_coupled)
      with a_idx = a_offset + a_i (Index() = flatRank + i*nThreads_coupled).
      The separate +thread_rank_coupled add is DELETED. Threads with
      flatRank >= a_tile_max take ZERO trips — the whole body no-ops via
      the loop materialization: per-thread trip counts ARE the
      single-ownership mechanism (each tile element has exactly one owning
      thread while T <= nThreads_coupled). REGISTERED FUTURE WORK (do NOT
      implement): for T > nThreads_coupled a thread takes multiple trips
      and tid vs tid+coupledThreads overlap — needs group-membership gating
      (is_first-style) when such tiles are ever searched
    - WARP dim: between for (b = 0; b < B; b += nThreads*T) (each
      thread owns a whole T-chunk — the composition-semantics ruling);
      in-body thread_rank = IntrinsicThreadMap(DoSoC_d, "index");
      b_offset = thread_rank*T + b. Inside = SEQUENTIAL walk of the
      owned chunk
  - Materialized bounds before the inside loops — SATURATING form
    (F1, ruled 2026-10-21; the plain min(T, extent − offset) WRAPS in
    u32 on tail rounds — extent − offset underflows toward 4e9 and the
    min picks T, so out-of-range ranks RUN their bodies out of bounds
    instead of no-op'ing):
    - off' = umin(d_offset, extent_d)
    - d_tile_max = umin(T_d, extent_d − off')
    - pure u32, wraps nowhere; d_tile_max == 0 exactly when the rank
      is out of range — bounds ONLY; tail guarding stays the
      primitive's whole-body no-op (§4), never an IR branch
  - Coupled-portion floor (F2, ruled 2026-10-21): an LNT carrying a
    CTA dim must leave >= 1 warp portion for its coupled inside walk
    (WarpPortion <= PORTION_WARP − 1 there); the decide pass's
    Validate/evalSchedule reject violations, and this phase fails
    closed if the coupled thread count < 1 (a zero-stride ForLoop is
    an infinite loop)
  - Branched LNTs (F3, ruled 2026-10-21): if any node of the LNT tree
    has > 1 child, REJECT the rewrite fail-closed — sibling iters must
    NEVER be silently linearized into one nested chain (the body would
    execute a cross product of the siblings' spaces); per-branch
    sub-nests are future work
  - Body rewiring: RAUW every InductionVariable use to the
    (d_offset + d_inside) expression (new defs); retire the original
    Iter chains through the MutContext choke points; zero new
    verifier invariant classes (§8)
- PHASE 4 — cache insertion (per promoted site, keyed by MemLoad id)
  - Recompute the site facts (INS chase, covered set = INS strictly
    inner than the recorded Tie, the CacheDims clamp) with the SAME
    deterministic chase as the decide pass — adjacency makes the
    results identical; the record needs no extension
  - Tier classification (RULED 2026-10-21, R2, + the 2026-10-23
    precedence): classify the site by the SoC of its COVERED dims
    - ANY covered dim CTA-assigned -> site REFUSED (never rewritten)
    - ANY NON-covered referenced iter CTA/WARP-assigned -> FRAG class
      (the thread-varying-coordinate precedence; the re-derivation here
      must agree with the decide-side judge or the record fails closed)
    - ANY covered dim WARP-assigned -> SMEM window class
    - ALL covered dims sequential -> FRAG class (per-thread)
  - FILL PLACEMENT (RULED 2026-10-21, R1 — VARIANT-DEPENDENT; the
    former unconditional rule "inside the innermost covered dim's
    between-round body, immediately before the tile-inside loops" is
    SUPERSEDED — it mandated a position at which the fill's own
    per-thread coordinates (tile-INSIDE dimvars) are undefined, the
    HI-01 inconsistency)
    - LoadTileThread (FRAG class): splice the fill INSIDE the element
      loop that DEFINES its per-thread coordinate; it refills once per
      epoch of the fill key — the innermost loop dim whose change
      invalidates the fragment (e.g. lhs refills per m-element inside
      M.inside)
    - LoadTileCta (SMEM window class): splice the fill at the
      innermost covered dim's BETWEEN-round body, immediately before
      the tile-inside loops (Tie == nil -> the same rule at the
      outermost position); its source coordinates are CTA-UNIFORM
      round bases ONLY — never a per-thread offset
    - ONE structural splice mechanism for BOTH variants: re-point the
      chosen position's body chain so the fill is its body-prefix;
      there is NO bodyRoot == load special case (that path placed a
      cooperative fill INSIDE an element loop — a §4 layering
      violation — while the buffer access sat outside the nest reading
      loop-private dimvars)
    - The tie position fixes WHICH rounds/epochs refill; the fill
      VARIANT fixes WHERE the fill sits
  - SMEM WINDOW sizing + access (RULED 2026-10-21, R2 — the access for
    a memory index inside the tile must use the runtime object
    thread_map.index(d) to obtain the thread index mapping for that
    dimension; a SEQ-assigned dim is a sequential tile loop.
    Rank-in-index => window-in-shape)
    - Buffer extent per covered dim: WARP dim -> #threads_d x T_d (the
      whole group's round window, NOT the single-thread chunk); SEQ
      dim -> T_d; total buffer bytes = product over covered dims x
      element size
    - Access index on a WARP dim d = thread_map.index(d) · T_d + i,
      where d is that dim's SLOT in the LNT's ThreadMap (one map per
      LNT, tuple coordinates, §7bis) — NOT a hardcoded 0 and NOT the
      bare inside-coordinate; access on a SEQ dim = its
      inside-coordinate
    - Fill = ONE cooperative LoadTileCta per round over the group
      window: CTA-UNIFORM source base (the round bases), DISJOINT
      per-thread stripes enumerating the window, tail clamping INSIDE
      the loader
  - Buffer + fill AT the fill point: MemNewSMem (SMEM class) or
    FragNew (FRAG class), extents per the R2 window rule above;
    variant per the tier rule; source = the original tensor at the
    LIVE coordinates of every INS dim in scope at the fill position
    (dims above the fill point contribute their current round/element
    coordinates; the fill key epochs the refill — under R1 nothing is
    "baked above the tie" that is not in scope where the fill sits);
    the loader owns partial-tile clamping and runs under §4's
    uniform-call rule. NO barriers, ever (§6 sync)
  - Access rewrite: RAUW the MemLoad's uses to
    MemLoadSMem(buf, [per-dim access index per R2]) / FragLoad(...)
    (both kinds already exist in the LIR catalog — M4); retire the
    MemLoad. Uncovered inner INS dims stay live loop indices BELOW the
    buffer — that re-read IS the reuse
  - Cross-LNT guard (RULED 2026-10-21, R4): sites recorded as
    non-candidates because their INS spans >= 2 LNTs (Promoted =
    false, explicit reason) are NEVER rewritten; the rewrite REFUSES
    any body referencing foreign iters (another LNT's loop dimvars) —
    fail closed
  - Worked example, ACCEPTED OUTCOME (ruled 2026-10-21; matmul
    M=1024/N=2048/K=512 at defaults — M CTA LoC=100 T=32, N WARP
    8 warps = 256 threads T=32, K SEQ T=32, 100 CTAs/512 threads,
    Max.cache = 64 KB):
    - lhs (tie=iterN, covered={K} all-SEQ) -> FRAG class: 32 x 4 B =
      128 B per thread, LoadTileThread filled INSIDE M.inside — it
      refills per m-element and its reuse is the owning thread's
      32-step n walk. PROMOTED
    - rhs (tie=nil, covered={N WARP, K SEQ}) -> SMEM window class:
      the R2 window is 256 threads x 32 x 32 x 4 B = 1 MB against the
      64 KB kernel-wide budget -> DECLINED by the budget fallback
    - This outcome is INTENTIONAL (R2's accepted consequence): the
      WARP-window rule prices group-level windows honestly; rhs
      returns at smaller shapes or larger budgets
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
    per fill-key epoch — per round for LoadTileCta, per fill-key
    element for LoadTileThread; R1)
  - M4 access node kinds: reuse the existing KindMemLoadSMem /
    KindFragLoad

## §7 Policies

- Dynamic shapes
  - Tile/group sizes are compile-time constants
  - Extents may be runtime values — handled by the remainder guard + the §4
    thr_loop bailout (whole-body skip in the loop materialization)
- Cross-LNT sites
  - Dedupe candidate sites kernel-wide by MemLoad node identity
  - V0 RULED 2026-10-21 (R4 — fail-closed): a load whose INS spans
    >= 2 LNTs is NOT evaluated in v0; it gets an EXPLICIT non-candidate
    SiteDecision entry in the record (Promoted = false, reason), and
    the rewrite REFUSES to rewrite bodies referencing foreign iters
  - The UNION-of-assignments evaluation (judge such a load against the
    union of the referenced iters' assignments) remains the FUTURE
    model — it must NOT be silently approximated by assigning the load
    to one tree and dropping the foreign iters from its INS
  - Never-bucketed loads are declared non-candidates (explicit record
    entries, not silent skips)
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
    - The NO-ARG flat form (ruling 2026-10-23): index() returns the
      holder's absolute flattened 1-D rank (ThreadMapping: threadIdx.x
      analog; GridMapping: blockIdx.x analog); index(d) keeps the tuple
      coordinate. The compiler never inlines threadIdx.x — the map is the
      seam; the coupled-inside walk's IV init is this query
    - warp_index / lane_id / is_first_in_warp / is_first_in_cta /
      warp_count queries PRESENT (2026-10-23 addenda — pure functions of
      index(); runtime-internal control flow for warp-specialized reduce;
      generated code never emits them as conditionals)
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
- Self-check list (pass exit conditions) — STRUCTURAL (F5, ruled
  2026-10-21: every check is computed from the EMITTED graph structure
  and must be able to FAIL on a plausibly-emitted graph; a check that
  inspects a baked constant or a synthetic run-state is vacuous and
  forbidden)
  - Every rewritten InductionVariable use has exactly the RAUW'd
    (offset + inside) form — no stale original-IV uses remain
  - Every FRAG site satisfies the fully-unroll condition (§6) and the
    MaxFragPerThread budget
  - The budget fallback terminates (§6)
  - Every parallel-inside loop lowers to thr_loop carrying the §4 bailout
  - FILL PLACEMENT (structural): every fill is data-reachable from its
    fill-position loop's body operand — LoadTileThread fills sit
    inside the element loop defining their per-thread coordinate;
    LoadTileCta fills sit at the innermost covered dim's between-round
    body, OUTSIDE every bailing element loop (§4 layering), with
    CTA-uniform source coordinates only (R1)
  - FILL SCOPE (structural): every dimvar reachable from a fill's
    source coordinates is defined by a loop that ENCLOSES the fill's
    structural position — no fill reads a loop dimvar from outside
    that loop (the CR-02/HI-01 leakage class)
  - Every warp-synchronous op's mask comes from thread_map (§4 mask contract)
  - No stale MemLoad remains for any promoted site (its uses are buffer
    accesses; the node is erased)
  - No original InductionVariable use survives (all RAUW'd to
    offset + inside expressions)
  - No cooperative tile op sits inside a bailing element loop (§4
    layering) — checked structurally over the emitted fills
  - ZERO barriers emitted (§6 sync) — verified by parsing the loadtile
    attribute back out of every emitted fill's operand list and
    asserting none is live; NEVER by re-reading a baked flag
- Downstream IR verifier invariants (same fix wave, landing AFTER this
  pass's output is corrected): a scoping invariant — a loop's
  IV/dimvar may be used only by nodes reachable from that loop's body
  operand subtree plus the loop's own structure — and a ForLoop dimvar
  positive-const step pin; together they make the fill/rewrite leakage
  class permanently unrepresentable. They are pipeline-level IR
  invariants, not pass-local checks — §8's zero-new-invariant-classes
  rule for THIS pass is unchanged
