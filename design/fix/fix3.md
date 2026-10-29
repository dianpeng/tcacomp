# fix3 — memory IR is three verbs; cache placement is derived, never mandated; the loader owns the coordination mapping

**Status:** RULED (2026-10-25, author rulings given in the fix2 post-mortem discussion — three rulings: the memory-IR vocabulary, the derived-placement/LICM algorithm, and the loader contract + performance mandate)
**Scope:** `internal/ir` (kind unification + the scope verifier check), `internal/opt/tiling` (fill defusion), `internal/opt/finallower` (the D-c completion under the scope rule), `internal/opt/memopt` (rule generalization), `internal/cg/cxx` (pattern fusion + the loader call), `runtime/tca` (loader API + performance), plus the fix2 carry-overs (canonicalizeConsts exile, the D-f anchor, the un-skip + battery)
**First surfaced by:** fix2's R-D stop at D-c (quick-261025-lmh; probe evidence and the reverted-work record in `.planning/quick/261025-lmh-please-implement-doc-fix2-md/261025-lmh-SUMMARY.md`)
**Baseline:** fix2 §4 steps 1–4 are LANDED and stand (`f7a649a` pipeline, `f9f2c69` dce AST-roots, `52ec1f2` TAIL + facade + migrations, `233291f` verb-death finallower; merge `e7464da`). **fix2 §4 steps 5–8 are SUPERSEDED by this document.**

---

## 0. The rulings (author, 2026-10-25 — normative)

> The IR for lowering only needs MemLoad/Store/New 3 different IRs, but for
> different storage class, ie GMEM/SMEM/FRAG. The cache is still a
> composition of these primitive memory load. […] All memory nodes are just
> new/store/load 3 verbs, and we composes our optimization around them.
> This makes the whole thing composable and cancellable.

> The code must be **lower and lower** when it passes the pipe, instead of
> having those complicated single large fused operations in our pipeline
> unless specialized instruction for performance reason.

> [Tiling] the tile between create an **offset index** which allows tile
> inside loop to make its index **back to original global one** — the new
> one will be `%offset + %tile_index`. This formula changes based on loop
> type, but this is mostly runtime behavior, since these 2 values are coming
> from the iterator. […] [Hoisting a load] will reduce its **dimension
> regarding that loop been hoisted** — move the MemLoad to an iteration that
> **just kills the needed-to-be-cached dimensions**. The final schedule will
> even hoist them to a best position as long as its expression satisfies
> the constraints — the final scheduling has LICM feature.

> [The load runtime primitive needs to] 1) pass in the grid/thread mapping
> objects for loading; 2) pass in source and dest view, which implicitly
> tells offset from global/source, and also dest shape — it is really
> **coordination mapping**; 3) the loading runtime API must use the mapping
> to compose valid loading from source to dest.

Derived rules (lettering continues fix2's R-A…R-F):

- **R-G (three verbs + storage class):** the memory IR is exactly
  `MemNew / MemLoad / MemStore`, parameterized by a storage class
  (GMEM/SMEM/FRAG). A cache (the fill) is a COMPOSITION — New(buffer) +
  tile-shaped MemLoad(GMEM) + plain MemStore(class) — never a fused
  node. Every memory optimization (memopt's whole domain) composes and
  cancels over these three verbs.
- **R-H (lower and lower; fusion only for hardware, only at codegen):** no
  fused mega-operations live in the pipeline. The single allowed fusion is
  at CODEGEN, justified by a specialized instruction: recognizing
  `New + tile-load + window-store` and emitting the cooperative loader call
  (LoadTile / cp.async). The compiler never inlines the loader's thread
  enumeration.
- **R-I (placement is derived — the LICM rule):** a memory item's position
  is always DERIVABLE, at any pipeline point, as the innermost block
  dominated by every surviving loop its post-rewrite coordinate closure
  references, still preceding its memory successors. A pass-local mandated
  position anchor is invalid by construction — R1's variant-dependent fill
  mandate and `siteFill.host` (the "mandated fill-position loop") are
  SUPERSEDED. The offset+index identity (`tiled_k = offset_k +
  tile_index_k`, both from the iterator; the per-loop-type formula is
  runtime behavior) is the semantic anchor for all coordinate rewrites.
- **R-J (the loader contract):** the load runtime primitive takes (1) the
  grid/thread MAPPING objects, (2) source and dest VIEWS that implicitly
  carry the offset and the dest shape, and (3) composes the mapping into a
  valid source→dest copy (coordination mapping). The current
  `LoadTile*(dst, src, bases...)` with raw `threadIdx.x` striping inside
  violates all three; the current per-element copy loop is too slow and is
  rewritten (D-n).

## 1. What exposed it (facts, all verified 2026-10-25)

1. **The D-c wall (fix2's R-D stop):** the fill
   (`fragstore#163`) carries biv `dimvar#120` = **X's own IV** (the
   covered dim's round base). Pre-lower, X (the N-between round loop) is an
   ANCESTOR of the fill's M-walk; final-lower's restructure moves X INSIDE
   the fresh element loops, so any block-relative re-placement rule strands
   the fill outside X's scope — the emitter fails
   (`cxx: dimvar 120 has no enclosing loop in scope`, `expr.go:29`) and the
   per-round semantics break (Tn=64 < N=128: the fill must re-execute each
   X-round). Root cause: R1 mandated positions structurally; fix2's D-c
   ("re-place in order into L′'s list") inherited the same block-relative
   bias; neither derived position from operand scope. The scope invariant
   was checked NOWHERE in the IR — only at the backend.
2. **The kind explosion + the fused store:** `internal/ir/kind.go:43-80`
   carries NINE memory kinds (3 verbs × 3 classes), and the fill role packs
   `[tile-shaped GMEM MemLoad, buffer, bivs…, loadtile-variant]` into ONE
   `KindFragStore`/`KindMemStoreSMem` node meaning "run the cooperative
   copy" — a fused alloc+load+store+variant operation wearing a store's
   clothes. The HIR verbs (`KindMemLoad/Store/New`) already obey R-G; the
   fusion crept in with the LIR memory kinds. It moved work instead of
   removing it: memopt needed a fill-shaped rule (d); the emitter needed
   `stmtFill`'s axes+bivs coordinate reconstruction; store→load forwarding
   (the 2026-10-20 audit's open item) was not even expressible.
3. **The loader is the naive reference:** `runtime/tca/loader.h`
   `copy_window_elem` pays, PER ELEMENT, 2 integer div + 2 mod (runtime
   divisors — no strength reduction), a full-rank bounds-predicate branch,
   and ~6 IMADs of offset math, through per-element `GetAt/SetAt` — tens of
   instructions around what should be one LDG+STS. Consequence: nvcc cannot
   prove contiguity → no LDG.128/STS.128, no cp.async (deferred in the
   header's own comment), and the mapping objects the compiler already
   constructs (`grid_map0`, `thread_map1` in the goldens) are never passed
   in — thread→work assignment is raw `for (i = threadIdx.x; …)`.

## 2. What must NOT change (C-numbering continues fix2's C1–C5)

| # | Ruling | Consequence |
|---|--------|-------------|
| C6 | fix2's landed architecture stands | The pipeline orchestrator, TAIL wiring (memopt factory instance → AST-roots dce), the compiler facade, and the verb-death lowers (`f7a649a`…`233291f`) are untouched except where a step below says so; R-E/R-F remain in force. |
| C7 | fix1's sea-of-nodes model stands | Membership/placement stay IV-derived; ordering rides the memory-effect chain (fix1's chain edge is now UNDERSTOOD AS the ordinary memory chain — no new edge semantics are introduced, the special fill-edge reading is retired). |
| C8 | The bracket law is untouched | Init before X, Reduce inside home, Combine after X, per-element output loops — `ScanBracketLaw` stays the gate. |
| C9 | The loader's synchronization contract is pinned | One `__syncthreads()` before, one after (refill WAR wait + visibility); the runtime owns ALL barriers and tail clamping; bounds predication semantics (out-of-range suppressed = safe padding) unchanged. cp.async's per-thread wait is NOT a barrier. |
| C10 | Runtime surfaces are frozen | WarpReducer protocol, launcher contract, Tensor/TCA_REQUIRE, the CPU unit layer, stdlib-only Go module (`go list -m all \| wc -l` == 1). |
| C11 | Goldens: C5 byte-identity is SUPERSEDED — exactly two deliberate regenerations | Regeneration #1 at the canonicalizeConsts exile (D-o: `reduce_warp.cu` dedup drift — measured; it contains no fill call), #2 at the loader-call API (D-m: `matmul_store.cu`'s `LoadTileThread` text). Each re-pins hashes in its own commit; after #2, byte-identity is re-instated as the standing gate. No other golden drift is tolerated at any step. |

## 3. The ruled fix (D-numbering continues fix2's D-a…D-g)

- **D-h (memory IR: three verbs + storage class):** collapse
  `KindMemLoadSMem`/`KindFragLoad` → `MemLoad(class)`,
  `KindMemStoreSMem`/`KindFragStore` → `MemStore(class)`,
  `KindMemNewSMem`/`KindFragNew` → `MemNew(class)`. The class is a node
  parameter (attribute or leading operand — implementer's choice, uniform
  across kind checks). The `loadtile` loader-selection attribute LEAVES the
  IR (loader selection is derived at codegen from class + mapping shape per
  R-H/R-J). Purely mechanical tree-wide (verifier, memopt, finallower,
  cg/cxx, print, visualization, tests); emitted text unchanged.
- **D-i (the fill is a composition):** tile-lower's fill construction
  (`internal/opt/tiling/lower_cache.go`) emits the three plain nodes —
  `MemNew(class, extents)` + `MemLoad(GMEM, tensor-level, tile-shaped: axes
  + round bases)` + `MemStore(class, buffer, value=the tile load)`; the
  buffer access is a plain `MemLoad(class, buffer, windowed coords)`. The
  tile-shaped source load already exists (`siteFill.src`). Ordering rides
  the ordinary memory-effect chain (the `NewMemLoadChained`-style env
  priming the fill already uses for its source; the accesses chain after
  the store on the buffer's chain). `siteFill` keeps recording facts for
  the §8 checks, but `host` loses its mandate meaning — placement is
  derived (D-j). R1's variant-dependent placement text is struck.
- **D-j (placement is derived — D-c completed under R-I):** in
  `buildElementLoop`'s restructure, each re-placed item's position is
  DERIVED: items whose coordinate closure references only their own fresh
  element loop re-place into L′'s item list in order (the source loads,
  coordinate binops — already working); an item whose closure references a
  SURVIVING inner loop's IV (the fill's biv → X) re-places at the
  innermost block dominated by that loop, still preceding its memory
  successors — on the frontend trigger: inside X, before home. Rebuild
  order is pinned: the re-placed items' rebuild runs AFTER the walk's and
  moved consumers' rebuilds (else an extra emitted local drifts the
  golden — observed and fixed in the reverted work). fix2's D-c text
  ("every Node item … re-places into L′'s item list, in order") is
  SUPERSEDED by this rule; the reverted implementation's candidate shape
  (a) is what the rule derives — recorded for provenance, not as a special
  case.
- **D-k (memopt generalization):** rule (d) collapses into the generic
  three-verb rule — a `MemStore` whose buffer has no surviving `MemLoad`
  is dead, and its source `MemLoad` chain dies with it. The fill-shaped
  special case is deleted. Store→load forwarding becomes EXPRESSIBLE
  (`GMEM tile-load → stored → class load` is now a visible candidate);
  implementing forwarding remains OPEN (separate item, audit-tracked).
- **D-l (the scope invariant becomes a verifier check):** the IR verifier
  gains the R-I invariant: every memory/coordinate consumer must be
  dominated by the loops its operands' coordinate closure references (the
  class of `dimvar … has no enclosing loop in scope` moves from the
  backend emitter to the verifier, failing at the pass that breaks it).
  Post-schedule position only (the check needs SchedBlock stamps).
- **D-m (codegen pattern fusion + the loader call — R-H/R-J):** `cxx`
  recognizes the composition `MemNew + tile-shaped MemLoad(GMEM) +
  MemStore(class, window)` and emits ONE cooperative loader call under the
  NEW runtime signature:
  `LoadTileCta/LoadTileThread(mapping, dstView, srcWindowView)` —
  (1) the ThreadMapping the compiler already constructs is passed in (the
  fill's site must have its LNT cluster's mapping in scope — a site
  without one is an R-D stop, never an emitter patch); (2) the source
  arrives as a WINDOWED view (offset implicit in the view, per R-J);
  (3) `stmtFill`'s axes+bivs coordinate reconstruction is DELETED — the
  composition carries the facts. `matmul_store.cu` regenerates here (C11
  regen #2).
- **D-n (the loader runtime — API + performance, R-J):**
  - API: mapping-object parameter; windowed source view; the runtime
    composes mapping→work (coordination mapping). The compiler never
    inlines thread enumeration (R-H).
  - Performance (signature-internal, independently landable FIRST):
    (i) threads stripe the leading (rank−1) rows — div/mod ONCE per row,
    not per element; (ii) per-row tail clamp via `umin(dstTrailing,
    srcDim − base)` — no per-element bounds branch; (iii) the trailing
    axis is a flat contiguous run (one LDG+STS + tail cmp per element);
    (iv) 128-bit vector path (uint4/float4) when 4-byte T, trailing
    extent %4 == 0, and bases aligned — scalar fallback otherwise;
    (v) cp.async fast path for SMEM dst on `__CUDA_ARCH__ >= 800`
    (16B per thread per LDGSTS, `__pipeline_commit` +
    `wait_prior(0)` before the trailing barrier — C9 preserved);
    (vi) the scalar reference path is RETAINED as the CPU-unit-layer
    and fallback implementation. Verified by the CPU unit layer +
    goldens + e2e; a one-off timing measurement on a big-tile fixture
    (≥256×256 tiles) records the before/after — measurement only, not a
    committed test.
- **D-o (carried from fix2 D-e/R-B — the const exile):** remove
  `canonicalizeConsts` from final-lower. The C5 measurement is TAKEN
  (2026-10-25, recorded in the fix2 SUMMARY): without it BOTH goldens
  drift on duplicate literal consts, so the R-E escape-clause branch
  applies (GVN cannot see post-typeinfer nodes — nil type cells — no
  existing pass covers it): a MINIMAL pipeline-owned tail dedup wired in
  TAIL before the dce instance (so the dce item sweep collects the erased
  duplicates' items). NOT a new registry ID, NOT in final-lower.
  `reduce_warp.cu` regenerates here (C11 regen #1).
- **D-p (carried from fix2 D-f + §4 steps 7–8 — the anchor + un-skip +
  battery):** the anchor check as final-lower's final gate (after
  `ir.Verify`): every live `Reduce(value)` call's value-operand transitive
  OPERAND closure (data-dep edges only — never crossing memory-chain edges
  for liveness) must reach ≥1 memory access whose subscript closure
  references a DimVar that is the IV of a block enclosing the call's
  SchedBlock — fail-closed. `TestFrontendMatmulScale` un-skips (both
  Appendix A variants, maxAbs ≤ reduceTol, canary 0); sanitizer coverage
  extended to the frontend fixture (memcheck + racecheck, zero errors).
  Battery (fundamental only, design.md): positive = the two variants +
  unchanged loopforest/§8 pins; negative = (a) TAIL removed → lower output
  still verifies with garbage present in the sea, (b) the anchor fires on
  a hand-severed Reduce chain, (c) a re-placed composition item lands
  dominated-by-scope (SchedBlock pin) with its memory-chain order intact;
  golden regression vs the NEW pinned hashes; e2e gate
  `TCACOMP_E2E=1 go test ./internal/e2e/`; §4.1 downstream pin —
  `cxx.ScanBracketLaw` over the emitted frontend fixture (no code change
  expected); grep gate: no skip text remains in `internal/e2e/`.

## 4. Implementation plan (ordered; tree green at each step = one commit)

1. **loader perf, internal** (D-n perf half) — `runtime/tca/loader.h` +
   view helpers only; call signature UNCHANGED; goldens byte-identical;
   CPU unit layer pins + e2e green + the timing measurement recorded in
   the summary.
2. **memory IR unification** (D-h) — 9 kinds → 3 + class; mechanical;
   goldens byte-identical (emitted text unchanged).
3. **fill defusion** (D-i) — tile-lower emits the composition; the emitter
   reads the composition and still emits the SAME call text; goldens
   byte-identical; on every existing shape the derived position (D-j)
   must equal today's mandated position — asserted by the suite.
4. **derived placement — D-c completed** (D-j) — the restructure re-places
   by operand scope; the frontend trigger's composition lands inside X
   before home; goldens byte-identical (neither golden shape has a
   between-block fill); `TestFrontendMatmulScalePipeline` green; the
   e2e skip still matches until step 9 (expected).
5. **memopt generalization** (D-k) — rule (d) → the generic three-verb
   rule; the wiring pin updated to the composition shape.
6. **scope verifier** (D-l) — the R-I check + negative pin (an item
   stranded outside its operand scope fails the VERIFIER, not the
   emitter).
7. **const exile + tail dedup** (D-o) — golden regen #1 (documented,
   re-pinned in-commit).
8. **loader API + codegen fusion** (D-m + D-n API half) — mapping in,
   windowed src view, `stmtFill` reconstruction deleted; golden regen #2;
   e2e green.
9. **anchor + un-skip + battery** (D-p) — final acceptance: numerical,
   sanitizers, negative pins, golden re-pin verification, e2e gate, grep
   gate, bracket-scanner downstream pin.

Gates at EVERY step: `go build ./... && go vet ./... && go test ./...`;
goldens per C11 (byte-identical except the two regen steps);
`go list -m all | wc -l` == 1; GPU gate at steps 1, 8, 9
(`TCACOMP_E2E=1 go test ./internal/e2e/ -count=1`).

## 5. Risks / open points

- **ThreadMapping scope at fill sites (D-m):** the emitter tracks the
  enclosing LNT cluster's mapping (`curThread`, `statements.go`); a fill
  whose scope lacks one is an R-D stop for a ruling — never an emitter
  patch.
- **cp.async (D-n):** 16B alignment + fallback discipline; nvcc-only, so
  the CPU unit layer cannot pin it — the goldens + e2e carry it.
- **Tail memopt on the composition (D-k):** the generic rule must fire
  exactly where rule (d) did (the wiring pin is the proof); ordering
  subtleties (an item surviving one round) are ruled then (R-D).
- **Two golden regens (C11):** discipline is the guard — re-pin + document
  in the same commit; any OTHER drift at any step is a blocker.
- **Open, untouched by this fix:** load→load forwarding implementation
  (expressible after D-k, still open); `simplify/lower.go` Sweep 3 (same
  disease — the immediate follow-up quick task); the lower-out-of-opt
  refactor (R-F's recorded coupling); bz9 deferred items (SEQ T==1
  indexExpr scope; T>1 WARP coverage hole); the audit's general
  dead-GMEM-load item.
