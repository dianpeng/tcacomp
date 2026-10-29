# fix2 — lowers are mechanical: final-lower stops sweeping; the pipeline orchestrator wires cleanup; a compiler facade exposes one entry

**Status:** RULED (2026-10-25, author ruling — extends fix1's sea-of-nodes ruling to pass discipline and module architecture)
**Scope:** `internal/opt/finallower` (delete every non-verb erasure; complete the restructure's item re-placement rule), `internal/opt/dce` (AST-item roots), `internal/pipeline` (NEW — the pass orchestrator), `internal/compiler` (NEW — the user-facing facade), `cmd/tcacomp` + `internal/e2e` (migrate to the facade) — plus the frontend e2e un-skip
**First surfaced by:** quick-261024-sdk's open item 1 — the frontend `matmul_scale` emits a reduced walk with no piped producer (all outputs = init·scale). Root-caused 2026-10-24/25 with per-pass IR/AST dumps and a final-lower sub-step trace (probes preserved in `/tmp/fix1-probe/`).

---

## 0. The ruling (author, 2026-10-24/25 — normative)

> Lower is a separate pass that does transformation, DCE is beyond its job.
> The finalization of the compiler is a pipeline of all passes — which means
> if the lower produces garbage, the pipeline must wire another DCE pass
> after the lower. Lower itself just focuses on its **mechanical
> translation**: a pass that follows the rules to mutate the IR graph,
> instead of being smart. […] It should never drop statements. Even if it
> needs optimization, it should use a pass like DCE. When it CANNOT do the
> mechanical translation, stop writing code and prompt back for
> clarification.

Semantic anchor (author's argument): **a reduce's piped value always chains,
through its producing map/reduce to a MemLoad whose subscript uses the loop's
induction variables, into the nest.** The producer chain is inherently
loop-nested; it can never be legitimately dropped or lifted by lowering.

Derived pass-discipline rules:

- **R-A (verb-death only):** the only erasures a lower pass may perform are
  the ones its verb's rule literally prescribes — Map dies in Map Lower
  (RAUW onto body + erase + erase As), Reduce dies in Reducer Lower (RAUW
  onto Combine + erase + erase init KeyedAttribute), the dissolved loop
  headers/IVs die in the element-loop restructure. Everything else — orphan
  sweeps, use-free fixpoint erasure, up-walks collecting "debris" — is
  liveness mutation: DCE's domain, never a lower's.
- **R-B (no optimization in lower):** const canonicalization / literal dedup
  (GVN-family) does not live in final-lower. If the emitted form needs it,
  the pipeline wires the pass that owns it at the position where it can
  fire.
- **R-C (chain edges are load-bearing, never erase criteria):** the
  fill→access ImpactBy chain edge (fix1 ORDERING) carries the placement
  invariant; no lower pass may walk use-lists across chain edges to decide
  erasure.
- **R-D (untranslatable = stop):** when a lower meets a shape its written
  rules do not cover, it fails closed and prompts for a ruling — it never
  bridges the gap with improvised mutation.
- **R-E (composite reuse — 2026-10-25 ruling):** "reuse existed pass
  compositively unless existed pass does not cover the needed
  optimization." A need that an existing pass covers is met by *wiring that
  pass again* at the position where it can fire — never by a new pass kind,
  never by smuggling the behavior into another pass. The escape clause
  (nothing covers it) is the only justification for a new pass.
- **R-F (module architecture — 2026-10-25 ruling):** a **pipeline module**
  is the orchestrator for all analysis + optimization + lower passes: it
  composites them into one exported interface and owns the wiring (e.g.,
  lower followed by memopt to clean the dangling garbage; dce after
  final-lower). External users do not care about pass composition as long
  as the code is efficient and performant. A **compiler module** is the
  single user-facing API — one entry performing syntax → backend-code
  transformation, built on the syntax module and the pipeline module. Lower
  currently lives coupled inside optimization (`internal/opt`); extracting
  it is a recorded follow-up refactor, not part of this fix.

## 1. The defect that exposed it

### 1.1 Symptom

The emitted reduced walk carries no piped producer — no lhs fill read, no
multiply, no `reducer.Reduce(value)` call — so `Combine` returns the init
and every output is 0 (skip signature in `TestFrontendMatmulScale`). The
bracket itself (Init → reduced-dim walks → Combine → store at the element
scope) is correctly placed.

### 1.2 Mechanism (traced frame-by-frame; probes in /tmp/fix1-probe/)

After schedule the graph is complete: the fill (`memload #160` lhs source +
`fragstore #163`) is placed in the **M-walk** (a *between* block of this
reduce), and `fragload #164` + the multiply `#16` sit inside home (the
N-inside walk), chain-wired fill→access. Reducer Lower inserts the Reduce
call `#174 [#reducer, piped]` after the multiply's item — correct. Map
Lower RAUWs the piped operand onto the multiply — correct.

Then `finalizeDissolutions` (final-lower) erases the live chain:

```
iv#77 (dissolved M-walk IV)
  →data→  #94  M-coordinate        (legitimately orphaned: consumers rebuilt)
  →data→  #160 lhs fill load       ← between-block item, abandoned (layer 1)
  →data→  #163 fragstore (fill)    ← same
  →chain→ #164 fragload            ← the up-walk crosses the ImpactBy edge (layer 2)
  →data→  #16  multiply
  →data→  #174 THE REDUCE CALL     ← erased: runtime calls have no SSA users
```

- **Layer 1:** `buildElementLoop`'s restructure re-homes only home's block;
  the between blocks' items (the fill, placed there *by design* per F8/R1)
  are "discarded" with the dissolved wrappers (`reducer.go:386`).
- **Layer 2:** the orphan sweep's up-walk (`pass.go:243-258`) walks
  `ir.UsesOf` on the premise that only coordinate debris remains reachable —
  but the premise is false: the walk crosses the chain edge into live
  consumers and the zero-use fixpoint erases bottom-up from the Reduce call.

The graph was never wrong; the sweep asked a question liveness reasoning
owns, answered it with an unproven premise, and did DCE's job inside a
lower — exactly what R-A/R-C now prohibit.

### 1.3 Why no gate caught it

tile-lower's §8 (ORDERING/LAYERING) ran at 116 when the graph was still
perfect; final-lower's verifier runs *after* the erasure on a consistent
residue; the golden shapes never combine rank ≥ 2 + innermost reduce +
promoted fill site (matmul has no reduce, reduce_warp has no fill). The
combination first occurs in the frontend e2e — where it surfaced.

## 2. What must NOT change

| # | Ruling | Consequence |
|---|--------|-------------|
| C1 | R-A bracket law (261024-bz9 HI-01, restated in fix1 C1) | The bracket placement machinery is untouched; only the item re-placement + sweep deletion change. |
| C2 | F8/R1 fill placement (261021-o61) | Fills sit at variant-mandated positions. The element loop is the *isomorphic* re-placement of the walk: same dim, same clamped bound, same per-element epoch count — the mandate re-derives, it is not overridden. |
| C3 | fix1's one-edge-relation model | Membership/placement stay IV-derived; chain edges keep carrying ORDERING; no new edge semantics. |
| C4 | fix1's sea-of-nodes loop model | Block dissolution re-places items by IV membership — mechanically, no judgment (R-D satisfied: the rule is completed below, not improvised per-case). |
| C5 | Goldens byte-identical | `matmul_store.cu`, `reduce_warp.cu` untouched (neither shape has a between-block fill; the tail wiring must not disturb their emitted form — const dedup measurement decides its form, §3 D-e). |

## 3. The ruled fix

**final-lower translates; the pipeline wires DCE after it.**

- **D-a (finalizeDissolutions → verb-death only):** delete the up-walk and
  the fixpoint orphan sweep. What remains is the prescribed dissolution:
  erase the between loop headers, their IVs, and the dissolved wrappers'
  own header nodes — nothing else. Dead coordinate spines left in the sea
  are DCE's input, not final-lower's business.
- **D-b (maplower → the three-step verb):** `eliminateMap` keeps exactly
  RAUW-onto-body + erase Map + erase As. `sweepDebris` is deleted; the
  lambda subtree's death is DCE's.
- **D-c (the restructure's item re-placement rule):** when between loop L
  dissolves into its fresh element loop L′ (same dim, same clamped bound),
  **every Node item of L's block re-places into L′'s item list, in order**
  (nested Block items map to their own fresh loops — the chain child). The
  existing between-IV→fresh-IV rebuild then covers the re-placed items'
  coordinates exactly as it already covers X/home/moved consumers: effectful
  nodes get their operand slots rebuilt; pure-expression items whose
  consumers read rebuilt clones simply die as sea garbage for DCE. No
  node is dropped by the lower.
- **D-d (DCE gains AST-item roots):** a DCE instance that runs after
  final-lower roots on **all AST items** (the post-schedule truth of
  liveness) plus the end node — this preserves effectful runtime calls
  (Init/Reduce/Combine have no SSA users but are items) and collects every
  non-item sea node. This is DCE-domain liveness reasoning, where it
  belongs.
- **D-e (the tail wiring — ruled Option A, via the pipeline module):** the
  composition becomes `… schedule → final-lower → memopt → dce` at the
  tail. Both tail instances RE-USE the registered factories (R-E): memopt
  again — its region rules (a)–(c) no-op on the LIR window (C1's
  chain-totality gate) while rule (d) fires where fill deadness is settled,
  making rule (d) reachable in-pipeline — it closes the fill-flavored half
  of the 2026-10-20 audit's dead-load finding; the audit's other items
  (load→load forwarding, general dead GMEM loads) remain open and
  unchanged — then dce with AST-item roots collects everything the lower left
  dead (ordering: memopt first, so dce sees the final residue; dce's
  AST-root policy also sweeps the items of memopt's erasures). The wiring
  lives in the pipeline module (R-F), NOT in new registry IDs and NOT in a
  global-priority bump — the registry stays the pass catalog, factories
  construct instances freely. `canonicalizeConsts` leaves final-lower
  (R-B): if C5 measurement shows the goldens need literal dedup, a minimal
  tail dedup is *justified by R-E's escape clause* (GVN cannot see
  post-typeinfer nodes — nil type cells — so no existing pass covers it);
  if the goldens are insensitive, it is deleted. Decided by measurement,
  documented in the summary either way.
- **D-f (the anchor as a hard check):** post-final-lower invariant: every
  `Reduce(value)` call's value chain reaches ≥ 1 memory access whose
  subscripts reference the bracket's loops — fail-closed. This pins the
  author's semantic anchor so the class can never silently emit again.
  (Fill-liveness stays memopt's rule per fix1 D-f, now wired at the tail
  per D-e — ruling Option A, 2026-10-25. No fill is dead on the trigger
  shape; the tail memopt is the composite-reuse wiring that makes the rule
  reachable in-pipeline at all.)

- **D-g (the two modules — R-F):**
  - `internal/pipeline` — the orchestrator. Exports the single
    pass-composition entry over a dematerialized graph; internally wires
    the three windows explicitly: HIR (dce, simplify-lower/inline + verify,
    typeinfer, gvn + verify, memopt + verify — the region rules), LOWER
    (tiling + verify, tile-lower, schedule, final-lower), TAIL (memopt —
    fill liveness — then dce with AST-item roots). It supersedes
    `opt.BuildPipeline()`'s role as the composition owner; the registry
    remains the pass catalog.
  - `internal/compiler` — the facade. One user-facing entry:
    source files → backend files (parse/semacheck via `internal/syntax`,
    kernel build + dematerialize, `pipeline.Run`, `internal/cg` emit,
    options threading). `cmd/tcacomp` and `internal/e2e/frontend_test.go`
    become thin consumers — their duplicated
    parse→build→dematerialize→pipeline→emit sequences are deleted.

## 4. Implementation plan (ordered; tree green at each step = one commit)

1. **pipeline module** (D-g, R-F) — `internal/pipeline` with the three
   windows; at this commit the composition reproduces today's exact pass
   set (HIR + LOWER, no tail yet), so behavior is bit-identical — a pure
   orchestration refactor proven by the full suite + goldens.
   `opt.BuildPipeline()` keeps its registry role for existing tests.
2. **dce: AST-item roots** (D-d) — the root policy extension (AST-attached
   graphs root on items + end). Unit pins: effectful runtime calls survive;
   non-item garbage collected.
3. **tail wiring + compiler facade** (D-e + D-g) — the pipeline grows the
   TAIL window (`memopt` → `dce`, factory re-instantiation per R-E); the
   item maintenance for tail erasures rides the dce sweep. `internal/
   compiler` lands as the single entry; `cmd/tcacomp` and
   `internal/e2e/frontend_test.go` migrate onto it (their duplicated
   sequences deleted). Hermetic pins: a fill-liveness fixture run through
   the REAL pipeline (rule (d) now reachable in-pipeline — pinned for the
   first time at the wiring level; the audit's remaining load→load
   forwarding item stays open, §5).
4. **finallower: delete the sweeps** — D-a + D-b (up-walk, orphan sweep,
   sweepDebris removed; verb-death erasures kept). The tail dce (step 3)
   collects what lower now leaves dead; commit scope is the deletion,
   verified by unit tests + goldens.
5. **finallower: restructure item re-placement** — D-c: between-block items
   re-place into the fresh element loops; the rebuild covers them. This
   commit flips the frontend trigger's producer chain from gone to present;
   `TestFrontendMatmulScale`'s skip condition stops matching (numerical
   acceptance asserted in step 7).
6. **finallower: exile canonicalizeConsts** — R-B/D-e; form decided by the
   C5 measurement (tail dedup only under R-E's escape clause, or deletion).
7. **the anchor pin + un-skip** — D-f check; `TestFrontendMatmulScale`
   un-skips: both variants numerically within `reduceTol`, canary 0,
   `compute-sanitizer` memcheck/racecheck clean.
8. **Tests (fundamental only, per design.md):**
   1. *Positive:* frontend e2e both Appendix A variants pass (above);
      loopforest/§8 pins unchanged.
   2. *Negative pins:* (a) with the tail window removed from a test
      pipeline, lower's output still verifies and simply contains garbage
      (lower never crashes on garbage it no longer makes); (b) the D-f
      check fires on a hand-severed Reduce chain; (c) a re-placed fill
      lands in the same-depth element loop (SchedBlock pin) with its chain
      edge intact.
   3. *Golden regression:* `matmul_store.cu` (`3e5aa73a…`), `reduce_warp.cu`
      (`7e14cba0…`) byte-identical.
   4. *E2E gate:* `TCACOMP_E2E=1 go test ./internal/e2e/` — full suite; no
      skip on the trigger shape remains.

### 4.1 Downstream verification (no code change expected)

The scheduler already places the trigger chain correctly (probe-verified at
014); with the producer chain surviving to codegen, the emitted bracket is
expected to close R-A end-to-end. Pin with the existing bracket scanner
(`internal/cg/cxx/bracket.go`) over the emitted fixture.

## 5. Risks / open points

- **Tail memopt on an AST-attached graph:** memopt's level gate rejects
  materialized/CFG graphs, not AST-attached ones — a tail run passes the
  gate. Its erasures hit nodes that are AST items; the dce tail's sweep
  owns the item maintenance (D-e). If ordering subtleties surface (e.g. an
  item surviving one round), they are ruled then (R-D), never patched into
  the lower.
- **Facade migration:** `cmd/tcacomp` and the e2e test must behave
  identically through `internal/compiler` — the full hermetic suite + both
  goldens + the e2e gate are the proof; any drift is a blocker, not a
  tuning exercise.
- **`simplify/lower.go` Sweep 3** ("erase the projects (use-free after
  2b)") — same disease, different lower; out of this fix's scope, recorded
  as the immediate follow-up quick task.
- **Lower-out-of-opt refactor** (R-F's noted coupling: tiling, schedule,
  final-lower living under `internal/opt`) — recorded follow-up; this fix
  adds the orchestrator that makes the eventual move mechanical.
- **DCE tail + verifier interplay:** dead-but-consistent sea nodes are not
  an invariant violation; if the verifier disagrees, that surfaces in
  unit pins and is ruled then (R-D).
- bz9 deferred items (SEQ T==1 indexExpr scope; T>1 WARP coverage hole) —
  untouched, still awaiting their own rulings.
