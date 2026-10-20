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
  ("Mechanical rewrite for tiling"): tile-id enumeration in the between loops,
  offset = tile_id × tile_size, materialized per-tile uniform bounds
  (a_tile_max = min(tile_size, A − a_tile_offset))
  - This spec does NOT duplicate those templates wholesale; it states the
    pass's obligations on top of them
- Obligations
  - (a) sequential-inside loops: bounded by the materialized uniform bound
  - (b) parallel-inside loops (CTA-companion warp inside loops, warp loops)
    - the strided tile coverage of [0, a_tile_max) is normative
    - the materialized uniform bound is normative
    - the TAIL-GUARD MECHANISM (how per-thread out-of-range elements are
      suppressed) is OPEN — see §4; nothing here may assume an answer
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

## §4 OPEN DESIGN — Control Flow / Parallel Tails (Issue #1)

- STATUS: OPEN, author-designated SERIOUS, explicitly UNDECIDED
  - This section is an open-questions register, NOT a decision record
  - No resolution is baked in anywhere in this spec; every tail-dependent
    obligation elsewhere cross-references this section instead of assuming
- Ground that is D1-independent (NORMATIVE — safe regardless of resolution)
  - (a) parallel tile-inside loops cover [0, a_tile_max) as a STRIDED range of
    the tile; per-thread element counts differ whenever a_tile_max is not a
    multiple of nThreads
    - two sources of raggedness: remainder tiles (extent not divisible by tile
      size) and non-divisible full tiles (a_tile_max not divisible by nThreads)
  - (b) the materialized uniform bound a_tile_max = min(T, A − offset) clamps
    the tile extent AND the global extent into ONE comparison
    - whatever mechanism guards tails can be expressed against this bound
- The five open facets
  - Q1 IR model
    - Question: does v0 stay strictly one-BB (predication-only), or does real
      control flow enter the IR later?
    - Option space: one-BB permanent (tails ride predication only) vs one-BB
      provisional (branches arrive in a later IR version)
    - Consequence: whether the verifier/scheduler/one-BB invariants are
      permanent commitments or provisional scaffolding — the largest-scoped
      consequence of the five
  - Q2 Uniformity
    - Question: are CTA-uniform trip counts a HARD invariant, or may threads
      diverge in iteration count?
    - Option space: hard invariant (every thread of the CTA executes the same
      trip count) vs relaxed (ragged participation allowed)
    - Consequence: the collective/reducer model depends on uniform trip counts;
      relaxing them reopens the deadlock-freedom argument for CTA-scope calls
  - Q3 Mask ownership
    - Question: who owns the active predicate — the compiler (KindSelect
      wrapping, value-level) or the runtime (bounds-checked tile ops,
      effect-level)?
    - Option space: compiler-owned value masks; runtime-owned effect guards;
      split ownership (values by compiler, effects by runtime)
    - Consequence: pure values and effects may have DIFFERENT owners; decides
      how much of the guard appears in the IR at all
  - Q4 Effects
    - Question: a predicated GMEM STORE cannot be expressed as a value-select —
      is a masked/bounds-checked store variant a runtime-API obligation?
    - Option space: runtime bounds-checked store variant; masked-store
      intrinsic; excluding direct GMEM stores from parallel tails
    - Consequence: the runtime API surface and what the compiler must
      guarantee about store operands in tail iterations
  - Q5 Collectives
    - Question: are Init/Combine/cooperative tile ops executed
      UNCONDITIONALLY by every thread, with inactive lanes contributing
      identity?
    - Option space: unconditional execution (ragged warp participation =
      undefined) vs conditional execution with defined ragged semantics
    - Consequence: the uniform-call rule of §6/§7bis and the reducer
      contribution semantics for inactive lanes
- Separability
  - Q2–Q5 are separable from Q1: mask ownership (Q3), effect guarding (Q4),
    and collective rules (Q5) can each be decided while keeping one-BB (Q1
    still open)
- What this spec does while Q1–Q5 are open
  - Bounds (§3), DoSoC numbering (§2), intrinsics (§5), cache insertion (§6)
    are D1-independent and normative as written
  - The cache copy lowering (§6) references the uniform bound WITHOUT
    specifying the tail-guard mechanism

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

## §7 Policies

- Dynamic shapes
  - Tile/group sizes are compile-time constants
  - Extents may be runtime values — handled by the remainder guard + §4
    predication (mechanism open there)
- Cross-LNT sites
  - Dedupe candidate sites kernel-wide by MemLoad node identity
  - A load whose INS spans multiple LNTs is evaluated against the UNION of the
    referenced iters' assignments
  - Never-bucketed loads are declared non-candidates (explicit, not silent)
- Determinism
  - DoSoC numbering order (outer → inner per LNT, §2) and the decide pass's
    preset ordering are inherited; this pass ADDS NO SEARCH of its own
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
- OPEN — allocator vocabulary (D3, UNDECIDED)
  - Question: how do the maps' dimension-vector remaps relate to a kernel-wide
    budget allocator — doc/tca.md §5.7.1's percentage sharding, or a successor?
  - Who computes the per-LNT split of gridDim.x and blockDim.x, and by what
    scheme, is NOT decided here; no option is picked
  - The per-LNT scoped maps themselves and the 1-D launch contract above are
    RULED and independent of this question
- Tile-op obligations (runtime API must provide)
  - Visibility: SMEM writes visible to the whole CTA after the op returns
  - Reuse safety: a refill waits for prior reads (write-after-read)
  - Store ordering
  - Bounds checking — this is where Issue #1's effect-guarding physically
    lives; mechanism OPEN per §4
  - Uniform call: cooperative ops called by every thread of the CTA the same
    number of times
- Sync ownership (RULED)
  - The runtime owns ALL synchronization/barriers; this pass never emits them
- Launch/budget validation at runtime
  - grid = ceil(ext/T) computed at launch
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
  - Every D1-dependent obligation is cross-referenced to the §4 open register
    rather than asserted as decided
