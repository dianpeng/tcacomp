# Review: doc/optana/loop_tiling.md (Loop Tiling design doc)

- Target: `doc/optana/loop_tiling.md` (593 lines, untracked WIP as of 2026-10-20)
- Reviewer method: read the target fully with line numbers; every algorithmic claim checked
  against the live sources — `internal/ir/*` (kind.go, value.go, effect.go, mutate.go,
  builtins.go, node_loop.go, node_memory.go, node_var.go), `internal/analysis/loopforest/`,
  `internal/analysis/sideeffect/`, `internal/opt/*` (registry.go, manager.go, cache.go,
  legality.go, schedule/, simplify/, typeinfer/), and the doc suite (`doc/tca.md` §2.3/2.4/3.3/5.7–5.13,
  `doc/ir.md`, `doc/opt.md`, the sibling `doc/optana/*.md` specs, `.planning/ROADMAP.md`).
- Severity tags: **BLOCKING** (design as written computes the wrong thing / cannot be
  implemented soundly), **SOUNDNESS** (correctness gap with a concrete fix),
  **COMPAT** (conflict with an existing system, cited), **MISSING-EASY** (small, high-value
  addition), **FUTURE** (forward-shaping recommendation, §8).
- Note on the analysis split: the doc keeps LNT enumeration in
  `internal/analysis/loopforest` (its package doc already names loop tiling as first
  consumer — `internal/analysis/loopforest/loopforest.go` L4–L5) and the MemLoad overlap
  analysis local to the pass (target L166). That split is fine; the findings below are
  about the algorithms, not the placement.
- The doc is NOT modified by this review.

## 1. Feasibility verdict

**Verdict: feasible-with-required-changes.**

- The four responsibilities the doc names (target L4–L8: cache placement decision,
  parallelization assignment, problem formulation/search, cache mark + tiling rewrite) all
  have workable mechanical homes in the current system: the loop-forest analysis exists
  (`internal/analysis/loopforest/loopforest.go` — live `KindIter` members, reversed-edge
  trees, read-only), the pass-manager slots exist after memopt-verify@113
  (`internal/opt/registry.go` L20–L31), the mutation choke points needed for a "large
  IRGraph patch" exist (`internal/ir/mutate.go`: SetDef/AddUse/RemoveUse/RAUW/SetType/Erase,
  cataloged in `doc/ir.md` §5.3), and the spec skeleton for tiling semantics already exists
  in `doc/tca.md` §2.4 (`tile(@x/M,T) = iter(@x_o(step=T)/M) * iter(@x_i/T, offset=@x_o)`),
  §3.3 (Tiling Rule), §5.7.4 (tile-between + tile-inside interpretation), §5.9 (iteration
  lowering with tile-loop factoring).
- It is NOT feasible as-designed: the candidate-cache analysis contradicts the doc's own
  matmul commentary (F1), the LoC constraint uses sum semantics where the normalized launch
  implies products (F4), the objective is truncated AND its completed skeleton maximizes
  cache bytes, fighting the budget (F8), the solver pseudo-code inverts its feasibility
  filter (F9), every parallel rewrite template contains defects up to infinite loops
  (F10), and the parallel-tail guard — the one-BB crux — is unspecified; it is solvable by
  predication per `doc/tca.md` §5.12 but the doc must adopt that mechanism explicitly (F11).
- Per-section justification: §2 (overlap analysis) is fixable with a one-line bucketing
  change (F1) plus two filters (F17); §3 (PSched/search) needs a launch-accounting model
  (F4/F5), a real objective (F8), solver repairs (F9); §4 (rewrite) needs the coherent
  tile-id form + predication (F10/F11); the pass split (§5) and compatibility (§6) are
  workable with the state-persistence and pipeline-slot fixes below.

## 2. Soundness — MemLoad overlap analysis (findINS + union, target L164–L198)

### F1 [BLOCKING] findINS/union buckets at the wrong level: contradicts the doc's own matmul example

- Doc: L168 (collection), L172–L178 (findINS), L182–L187 (bucketing: "visiting backwards,
  from inner most Iter node to outer most Iter node… if it contains a Iter node in its INS…
  Record it and remove it from I"), vs the doc's own commentary L152–L153 ("place lhs
  cache at iterN… place rhs cache at iterM").
- Ground: `internal/analysis/loopforest/loopforest.go` L74–L137 (Analyze gives the nest
  with Depth — an innermost-to-outermost walk is directly expressible);
  `doc/tca.md` §5.8 L598–L603 (placement works innermost-outward using the hoisting rule
  to find the reuse scope).
- Analysis: trace the doc's example (L138–L142): nest iterM→iterN→iterK; `lhs[m][k]` has
  INS={iterM,iterK}, `rhs[n][k]` has INS={iterN,iterK}. The algorithm visits innermost
  first: iterK. BOTH loads contain iterK in their INS, so both are recorded and removed at
  the iterK level, producing O = { {lhs, rhs} @ iterK }. That placement caches inside the
  k-loop — every cached element is reloaded every k step, i.e. zero reuse. The commentary
  requires the reuse level: lhs at iterN (reused across n), rhs at iterM (reused across m).
  The reuse level is exactly the **innermost LNT level NOT in the load's INS** (for lhs:
  levels M,N,K minus {M,K} → N; for rhs: minus {N,K} → M). The algorithm as written
  computes innermost-referenced bucketing; the commentary describes outermost-missing
  (reuse) bucketing. They disagree on the doc's only worked example.
- Coupled defect: the downstream filter L193 ("If S's length < 2, skip") passes today only
  because the wrong bucketing merges lhs+rhs into one set; under the corrected bucketing
  each level's set is a singleton ({lhs}@iterN, {rhs}@iterM) and BOTH would be filtered
  out — the doc would then cache nothing on its own flagship example.
- Fix sketch: bucket each MemLoad at its reuse level = the innermost Iter of the LNT NOT in
  its INS (loads whose INS covers the whole LNT have no reuse level → not cacheable at
  this nest, correct: there is no non-referenced dim to reuse across); drop or re-justify
  the length≥2 filter; keep the ≥2-subscript filter (L195) and the tile_size==1 exclusion
  (L283).

### F2 [COMPAT] ValueEqual-on-Iter is a sound membership key today; the subscript-chase assumption must be pinned

- Doc: L178 ("Equality of Iter is defined by ValueEqual, just like GVN"), L176 ("Chasing
  the Input list until hit all InductionVariable").
- Ground: `internal/ir/value.go` L145–L202 (ValueEqual = same kind + same operand def IDs,
  identity keying, no structural recursion), L32–L37 (hash/equal contract relied on by
  gvn-verify); `internal/analysis/sideeffect/sideeffect.go` L183–L186 (bare `KindIter` is
  side-effect free — the loop-fusion reclassification, `doc/optana/loop_fusion.md` [1]);
  `internal/opt/simplify/lower.go` L44/L84/L205 (post simplify-lower@102, every MemLoad
  axis slot resolves to an `InductionVariable(iter)` node).
- Analysis: settle B affirmatively — using ValueEqual for INS membership is consistent
  with GVN's use: two Iters are ValueEqual iff their BIV def and parent def are the same
  nodes (`internal/ir/node_loop.go` L9–L29: ops = [biv] or [biv, parent]), which is
  precisely loop_fusion.md's nested-aware equality, and after gvn@110 canonical Iters are
  already merged so pointer identity ≈ ValueEqual. The subscript chase is exactly ONE step
  today: axes ARE `InductionVariable` nodes after simplify-lower. The assumption to pin:
  (a) findINS runs after simplify (any pipeline slot ≥ 102); (b) when future affine
  subscripts arrive (`doc/optana/induction_variable.md` L33: "not supported in this
  version"), the chase must be defined as a traversal through pure expression nodes only,
  fail-closed on anything not reducible to InductionVariable leaves — "chase the Input
  list" is currently under-specified but not wrong for today's format.

### F3 [SOUNDNESS] Cross-LNT loads are double-collected and never reconciled; unbucketed loads vanish silently

- Doc: L168 ("For a specific LNT, collect all MemLoad that has subscript reference to one
  of the iteration for this LNT"), L182–L187 (bucketing only scans the LNT's own Iters).
- Ground: `doc/tca.md` §5.7.7 L549–L557 (loop nest clusters: one LNT nested inside
  another; a nested map's loads legitimately subscript outer-tree iters);
  `doc/optana/loop_fusion.md` L78–L89 (fusion replaces prefix Iters, making INS
  cross-tree by construction — after fusion the forest sees ONE tree, so that case is
  benign).
- Analysis: two defects. (1) A load subscripted by iters of two different LNTs is
  collected into I for each LNT; each tree's analysis then independently promotes or not,
  with no cross-tree reconciliation — the same site can be promoted twice (two caches) or
  the promotion criterion evaluated against the wrong tree's assignments. (2) A load whose
  INS references an iter of ANOTHER tree is never bucketed in step (3) (only the LNT's own
  Iters are scanned) and stays in I forever — the algorithm does not state it is dropped,
  so O silently misses candidates; that is conservative but unspecified.
- Fix sketch: run collection per LNT but dedupe candidate sites kernel-wide (site key =
  MemLoad node identity); for a load whose INS spans trees, evaluate the promotion
  criterion against the UNION of assignments of all referenced iters; document that
  never-bucketed loads are non-candidates.

## 3. Soundness — PSched constraint model, objective, search (target L200–L341)

### F4 [BLOCKING] Sum-semantics LoC constraints do not model the normalized launch; CTA totals are products

- Doc: L264–L269 ("ALL LoC added for CTA level must be less than CTA.max_loc" /
  "WARP/THREADS level… less than WARP.max_loc"), vs the normalized launch L73–L81
  (gridDim.x = total CTAs, blockDim.x = total threads per CTA, multiple of 32) and the
  whole-warp rule L343–L345.
- Ground: `doc/tca.md` §5.7.1 L441–L452 (CBLId/CTLId linearization; blocks partitioned by
  percentage into shards; threads partitioned in whole warps), §5.7.2 L456–L465 (whole-warp
  groups are disjoint shards — "resolved: whole-warp flat groups only"), §5.7.6 L547
  (v0 grid parallelism is single-dimension; 2-D CTA sharding is future); CLAUDE.md
  Constraints ("every thread group is a whole number of physical warps"); CUDA limits on
  the normalized 1-D launch: gridDim.x ≤ 2^31−1, blockDim.x ≤ 1024 (= 32 warps).
- Analysis: settle D — the two sums are NOT both wrong, but one is. Threads: warp groups
  are DISJOINT whole-warp shards of one CTA's thread budget (tca.md §5.7.2), so
  threads_per_cta = Σ_dim (warp LoC × 32) and "sum ≤ WARP.max_loc (≤ 32 warps / 1024
  threads)" is the right shape. CTAs: when more than one dim of an LNT is CTA-assigned,
  each CTA owns a coordinate TUPLE of tile ids — total CTAs for that LNT is the PRODUCT of
  per-dim CTA LoCs, not the sum; the sum constraint would certify nCTA_0 = nCTA_1 = max
  while the launch needs max². Additionally, multi-dim CTA assignment itself exceeds
  tca.md v0's stated single-grid-dim limitation (§5.7.6 L547) — either restrict the search
  to ≤1 CTA dim per LNT (v0) or model the product (and the gridDim.x limit) explicitly.
- Fix sketch: constraint set per kernel: Σ over all warp-groups (all LNTs) of warp LoC ≤
  32; per LNT with CTA dims {d}: Π nCTA_d ≤ CTA.max_loc (and the kernel-wide grid total ≤
  the device grid limit); document that >1 CTA dim per LNT is post-v0.

### F5 [BLOCKING] No kernel-wide accounting across LNTs; per-LNT DoSoC counters collide

- Doc: L101 ("LNT section #1" — each LNT gets its own GridConcurrencyMap section),
  L364–L386 (numbering pass: "For each LNT… maintain 2 separate dimension index counter…
  Visiting LNT's Iter node in outer -> inner" — counters implicitly restart per LNT).
- Ground: `doc/tca.md` §5.7.1 L448–L452 (the ONE launched grid is partitioned into shards —
  the kernel-wide allocator is the percentage scheme); one kernel = one launch = one
  gridDim.x/blockDim.x pair (target's own L73–L81).
- Analysis: settle E — the launch is per-kernel but constraints are per-PSched (per-LNT):
  nothing bounds Σ_LNT CTAs or Σ_LNT warps against the single launch. Worse, the numbering
  pass assigns dimension indices from per-LNT counters that each start at 0, so LNT1's CTA
  dim 0 and LNT2's CTA dim 0 bind to the SAME dimension of the same runtime map object —
  a collision that silently aliases two unrelated concurrency spaces.
- Fix sketch: a kernel-level allocation step before numbering: one global DoSoC namespace
  (CTA dims and warp dims counted across all LNTs), plus a kernel-wide budget pass
  allocating grid share / warp groups per LNT — the natural model is tca.md §5.7.1's
  percentage sharding, with the numbering pass consuming the kernel-global counters.

### F6 [SOUNDNESS] Promotion criterion: notation ambiguous but coherent under tca.md's two-level CTA semantics; FRAG budget missing (→ F7)

- Doc: L274–L283 ("denote all its subscript access as [x0][x1]…[xn-1][xn]… If all CTA
  assignment happens at iteration <= xn-1… If xn-1 and xn are sequential => ALL in FRAGs…
  Otherwise => SMEM… xn-1 and xn is within a CTA but also not in a single thread").
- Ground: `doc/tca.md` §5.7.3 L467–L475 (`CTA(BlockJ, ThreadK)` is TWO-level: blocks
  outside, thread group INSIDE the CTA — so a CTA-assigned dim is still covered
  thread-parallel within one CTA), §5.8 L604–L606 (storage class: all covered iterations
  sequential → FRAG, otherwise SMEM).
- Analysis: settle F — "iteration" must be read as SUBSCRIPT POSITION, and the coherent
  reading is r1: CTA assignment permitted at positions ≤ n−1 (all but the last subscript),
  because a CTA dim's tile is thread-covered INSIDE each CTA (BlockJ+ThreadK), making
  "xn-1 and xn within a CTA" true even when xn-1 is the CTA dim. Under r1 the matmul case
  works: lhs[m][k] with CTA at m (= x0 = xn-1) → not both sequential → SMEM, matching
  tca.md §5.8. The incoherent reading (CTA also at xn) is correctly excluded: the last
  subscript's tile would span CTAs. The doc must (a) define "iteration" = subscript
  position, (b) state the r1 rule as "no CTA assignment at position xn" plus "CTA at
  xn-1 forces SMEM via the ThreadK half", (c) note WARP assignment at xn-1/xn likewise
  forces SMEM (covered by "Otherwise"). What the criterion genuinely lacks is any
  FRAG/register budget — see F7.

### F7 [MISSING-EASY] No FRAG/register-pressure constraint — only the SMEM Max.cache sum exists

- Doc: L284–L288 (cache-size constraint: "SUMMATION of all cache size CANNOT be larger
  than SMEM size… Max.cache") — nothing for FRAG.
- Ground: `doc/tca.md` §5.8 L587–L589 ("SMEM plus fragment (register) allocation has a
  per-CTA budget, for example 64 KB") and L608 ("register arrays stay in registers only
  when all indices are compile-time constants after unrolling; otherwise they spill to
  local memory. FRAG placement should require fully unrolled inner tiles");
  `.planning/ROADMAP.md` Phase 4 SC2 L109 (64 KB SMEM+FRAG budget).
- Analysis: a FRAG promotion with a tile that does not fully unroll spills to local
  memory — strictly worse than the GMEM load it replaced. The constraint set needs a
  per-thread FRAG bytes cap (× threads/CTA for the CTA aggregate) and a
  FRAG-tile-must-unroll condition. Easy to add to the same constraint block as Max.cache.

### F8 [BLOCKING] Objective function truncated; the completed skeleton maximizes cache bytes — footprint, not reuse — and fights the budget

- Doc: L290–L295 ("(2) SMEM reuse can be given by" — sentence ends mid-line),
  L325–L331 (score = cache_weight × total_cache + parall_weight × p, where total_cache
  sums get_cache… "size in bytes").
- Ground: `doc/tca.md` §5.8 (the quantity of interest is GMEM traffic avoided by the
  cache); §5.7.6 L546 (budget fallback exists because caches must FIT, not be maximal).
- Analysis: settle G — two defects. (1) L295 is truncated: the SMEM-reuse term is never
  defined. (2) The pseudo-code that stands in for it (L327/L331) maximizes total cache
  BYTES: under the Max.cache cap this biases the search toward saturating SMEM regardless
  of benefit, and it conflates footprint with reuse — bytes cached says nothing about how
  often each byte is re-read. Fix: the correct linearizable metric is GMEM traffic
  avoided: Σ_sites bytes(site) × (reuse_count(site) − 1), where reuse_count is the number
  of iterations the cached tile survives at its placement level = the extent product of
  the reuse dims divided by their tile sizes (derivable from the LNT + placement decision,
  no runtime data needed). Optionally minus an occupancy/spill penalty term.

### F9 [BLOCKING] Solver defects: inverted feasibility filter, sloppy accessor, and an enumeration that explodes for realistic nests

- Doc: L314 ("dumps 3 ^ K solution space"), L316 ("if (is_feasiable(each_sched)) continue;"
  — skips FEASIBLE schedules; only infeasible ones get scored), L319–L321 ("auto p = 0 …
  p += get_soc(p)" — accumulator passed where the loop variable belongs; should read
  LoC(soc)), L299–L304 (LoC preset lists per SoC type + tile_size preset list
  32/64/128/256/…).
- Ground: `internal/opt/registry.go` L18–L35 (the pipeline these searches run in —
  compile-time, per kernel); design.md L92–L94 (integer programming named as future
  direction).
- Analysis: settle H — L316 must be `if (!is_feasiable(each_sched)) continue;` as the
  feasibility filter (as written, every scored schedule violates the constraints and every
  feasible one is discarded). L321's accessor conflates the accumulator p with the
  iteration variable. Search size: the space is |SoC|^K × Π_{parallel dims}|LoC presets| ×
  Π_dims |tile presets|; for K=6 with modest lists (4 LoC, 4 tile) that is 3^6 × 4^4 × 4^6
  ≈ 7×10^8 scored schedules per LNT — not compile-time viable as a flat enumeration.
  The constraint structure is per-dim separable except (a) the ≥1-CTA constraint and
  (b) the SMEM/FRAG sum — so prefix pruning (check per-dim constraints as each dim is
  bound, prune before descending) plus fixing LoC/tile presets per dim independently
  reduces this to roughly 3^K × pruned tails; the MIP backend (§8a) is the scale-out
  answer.
- Fix sketch: repair the two pseudo-code lines; restructure the enumeration as a
  depth-first binding over dims with per-dim feasibility pruning and the two global
  constraints checked incrementally (SMEM sum as a running accumulator).

## 4. Soundness — tiling rewrite semantics (target L350–L591), read as the loop-tile-lower spec

This section reviews the rewrite half per the mandated split (decision pass records;
a new lower pass rewrites — see §5). The defects below are in the doc's normative
templates; under the split they become REQUIRED content of the loop-tile-lower spec.

### F10 [BLOCKING] Rewrite templates contain unit mismatches, wrong variables, and infinite loops

- Doc + Ground (`doc/tca.md` §5.12 L793–L804 — the coherent cta_for/thread_for forms this
  review proposes; §5.9.1 L622–L636 — the between/inside emission order):
  1. **Bound/stride unit mismatch** — L36–L37 (intro example), L437–L438 (sequential),
     L501–L505 (CTA), L564–L568 (warp): `for (a = 0; a < ceil(A/tile_size_a); a +=
     tile_size_a)` — the bound counts TILES while the stride advances ELEMENTS, and with
     offset = a × tile_size (L445, L453) the offset double-multiplies. For A=100, tile=10:
     bound 10, a takes 0,10,…,90 → offsets 0,100,…,900 — out of bounds immediately. The
     coherent forms are tile-id enumeration (`for (a = 0; a < ceil(A/T); a++) { offset =
     a*T; }`) or element striding (`for (a = 0; a < A; a += T) { offset = a; }`) — never
     the hybrid.
  2. **Wrong tile variable in a bound** — L449–L451: `b_tile_max = min(tile_size_a, B −
     tile_size_b*b)` — the first argument must be `tile_size_b`. Also L42–L43 (intro
     example): `max_tile_size_i = min(i_tile_size, M − i*tile_size)` — `M` must be `I`
     (the i-loop's extent).
  3. **CTA inside loop is broken three ways** — L515–L519: increments the OUTER variable
     (`a += warp_loc*32*thread_map.dimension`) instead of `a_tile`, so the inner loop
     never advances its own counter (non-terminating) while corrupting the outer stride;
     ignores the computed `a_tile_max` (L508–L511) entirely; and `a_index` (L520) =
     `a_tile_offset + thread_map.index` omits `+ a_tile`, so the strided base never enters
     the index. Coherent form per tca.md §5.12: `for (base = 0; base < a_tile_max; base +=
     nThreads) { idx = a_tile_offset + base + thread_index_in_group; active = …; }`.
  4. **Warp inside loop same wrong-increment** — L580–L584: `for (a_tile = 0; a_tile <
     tile_size_a; a++)` — increments `a`, not `a_tile`: infinite loop; also ignores
     `a_tile_max` (L571–L574).
  5. **Outer construct ambiguity** — L501–L505 strides the tile-between loop by
     `tile_size_a * grid_map.dimension` (a grid-stride flavor over tiles) while L513
     derives the offset from `grid_map.index * tile_size_a` (tile-id enumeration). These
     two are inconsistent as written. Settlement: the CTA case is a tile-id enumeration —
     each CTA's map index IS its tile id — so the coherent outer loop is
     `for (a = grid_map.index; a < ceil(A/T); a += grid_map.dimension) { offset = a*T; }`,
     matching tca.md §5.12's `base = blockIdx.x * T; base += T * gridDim.x`.
- Analysis: none of these are cosmetic — (1), (3), (4) produce wrong indices or
  non-termination in the normative lowering templates. All have the concrete coherent
  forms cited above. The sequential template (L437–L463) is correct in structure
  (a_o → b_o → a_i → b_i, matching tca.md §5.9.1's emission order) once (1)/(2) are fixed.

### F11 [BLOCKING] One-BB crux: parallel tails need predication; the doc's "Ensure bounds check" covers only the sequential case

- Doc: L408–L411 ("Parallel loop… Default use strided style loop… Ensure bounds check —
  Calculate actual tile inside loop's bound before entering into the loop"); the CTA/warp
  templates then use the uniform `tile_size_a` as the inside bound (L517, L582) with no
  per-element guard.
- Ground: CLAUDE.md Constraints ("CFG has exactly one basic block per kernel; no control
  flow in v0"); `doc/tca.md` §5.12 L798–L804 (thread_for tails are PREDICATED, not
  exited: `bool active = (base + threadIdx.x < T) && (idx < M)` predicates loads, stores,
  and Reduce calls so every thread still reaches collective calls), §5.11 L777–L789
  (runtime bounds contract: predicates suppress map invocation but never collective
  Init/Combine or cooperative tile calls), §5.7.8 L581 (invariant 5: CTA-uniform trip
  counts + predicated tails), §2.4 L72 (the remainder guard `@x_o + @x_i < M` is
  implicit in the tile algebra); `internal/ir/kind.go` L49 (`KindSelect` — the typed
  conditional `?:`, the value-level predication carrier available today).
- Analysis: settle J definitively — the uniform `min(...)` bound (a_tile_max) is sound for
  SEQUENTIAL inside loops (one agent executes the whole bound). For PARALLEL inside
  loops, threads cover a strided range of the tile; on a tail tile (extent not divisible
  by tile size) some threads' elements exceed the extent, and trip counts cannot express
  that per-thread — you need either control flow (violates one-BB) or PREDICATION. The
  mechanism exists and is already assumed by the spec stack: value-level `KindSelect`
  predication for pure computations, and runtime-owned bounds/predication inside the
  lowered tile ops (storetile/loadtile) for effects — a predicated GMEM store cannot be
  expressed as value selection, so direct GMEM stores inside parallel tails must either
  be guarded by the runtime's bounds-checked tile variants or lowered to masked forms;
  collective calls must NEVER be predicated away (uniform-call invariant). The doc must
  adopt this explicitly: replace "Ensure bounds check" with (a) uniform a_tile_max for
  sequential-inside, (b) per-element `active` predication for parallel-inside, and
  (c) the collective-exemption rule.
- This is the single most important feasibility question for the rewrite half: answer —
  feasible, by predication; as written, unguarded.

### F12 [COMPAT] New IR nodes IntrinsicGridMap/IntrinsicThreadMap: legal under the v5 append-only rules, but the doc lists them twice and under-specifies the payload

- Doc: L483–L496 and again L546–L559 (the node list is duplicated verbatim — a doc
  defect in itself); signature `IntrinsicGridMap(<DoSoC>, <name of field>)`.
- Ground: `internal/ir/kind.go` L12–L19 + L30 (KindVersion 5, append-only: new kinds
  insert before `KindEnd` in their tier, no renumbering); `doc/ir.md` §2 L103–L106 (a new
  kind lands in exactly three places: struct in the family's node_*.go, catalog.go row,
  allocNode case); `internal/ir/verify.go` L146+ (per-kind shape checks — new kinds need
  cases); `internal/opt/typeinfer/rules.go` L275–L276 (the typing catalog is CLOSED — an
  unknown kind errors, so two new rules are REQUIRED or typeinfer rejects the graph);
  `internal/ir/convention.go` (Input/ImpactBy-only convention is machine-checked).
- Analysis: settle the interface question — the "name of field" selector ("index" vs
  "dimension") may ride EITHER mechanism the suite already sanctions: a string payload
  (the `callNode.name` precedent, hashed via `mixString` in `internal/ir/value.go`
  L79–L84/L262–L263) or an Input-list `KeyedAttribute` operand (the Reduce-init pattern,
  `internal/ir/node_iterfmt.go`). Both are convention-legal; the callNode-style string
  payload is lighter and keeps DoSoC as the single operand. REQUIRED companions, none of
  which the doc mentions: catalog row + allocNode case + verify shape case (DoSoC operand
  must be a const u32; field name ∈ {index, dimension}) + typeinfer rules (result u32) +
  a legality-window row for the stage the lower pass produces + contentHash/contentEqual
  cases mixing DoSoC and the field name so GVN buckets stay faithful. GVN-CSE of these
  intrinsics is sound (per-launch deterministic pure values) and desirable.
