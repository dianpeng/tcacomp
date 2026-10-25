# Quick 261025-lmh Summary — implement doc/fix2.md (lowers are mechanical; the pipeline orchestrator wires cleanup; a compiler facade exposes one entry)

**Task:** quick 261025-lmh — implement doc/fix2.md exactly: sweep-free final-lower (R-A/R-B), DCE AST-item roots (D-d), the pipeline tail wiring (D-e, Option A), the internal/pipeline + internal/compiler modules (D-g/R-F), the between-block item re-placement (D-c), the D-f anchor gate, and the frontend e2e un-skip.
**Authoritative spec:** `doc/fix2.md` (2026-10-25 author ruling; lives in the main repo only), `doc/fix1.md` background
**Branch:** `worktree-agent-ad2e5be0d18fee805` (worktree `agent-ad2e5be0d18fee805`)
**Status:** STOPPED AT R-D DURING TASK 3 COMMIT 2 (fix2 D-c) — 4 of 7 commits landed, full hermetic tree green at HEAD `233291f`, both goldens byte-identical throughout; the D-c rule as written produces a scope-invalid (and semantically wrong) graph on the exact frontend trigger it was written to fix — details and probe evidence below. Tasks 4–5 not executed (blocked downstream of the finding); the D-e canonicalizeConsts C5 measurement WAS taken and is recorded below per fix2 D-e's "either way" requirement.

## What was done (fix2 §4 ordered steps → commits)

| §4 step | Scope | Commit | Status |
|---------|-------|--------|--------|
| 1 (D-g/R-F) | `internal/pipeline`: the pass orchestrator — HIR+LOWER windows as explicit ID lists resolved via `opt.Lookup(id).Factory()` (fail-closed, never `Registered()`/`BuildPipeline()`); `Run` = one `opt.NewManager`; composition pinned order-for-order against `opt.BuildPipeline()` | `f7a649a` | DONE |
| 2 (D-d) | `ir.WalkFrom(roots, v)` extraction (Walk delegates, visit order preserved); `dce.NewASTRoots()` — MARK = two-root walk + `WalkFrom` over every AST item node AND every Loop block's header node; post-sweep item maintenance (`RemoveItemsIf` over every block, incl. Prototype); 3 unit pins + the no-AST gate + the plain-sweep contrast pin | `f9f2c69` | DONE |
| 3 (D-e + D-g) | TAIL window `[memopt (SECOND factory instance — R-E composite reuse), dce.NewASTRoots()]` after final-lower; `TailPasses()` exposed; `internal/compiler` facade (`CompileKernel` = Build → Dematerialize → pipeline.Run → cxx.Emit; `CompileFile` merge); `cmd/tcacomp` + `internal/e2e/frontend_test.go` migrated (duplicated sequences deleted); fill-liveness wiring pin (memopt family-11 fixture shape through `[registry gvn, TailPasses()]` under one manager — rule (d) fires in-pipeline, fill + source load erased, item-anchored allocations kept) | `52ec1f2` | DONE |
| 4 (D-a/D-b) | `finalizeDissolutions` → verb-death only (up-walk + fixpoint orphan sweep DELETED; erase the loop header + its OWN header nodes [attrs, BoundedDimVar wrapper, IV] in reverse use order; Erase's live-use refusal is the backstop; the narrowed hard error fires ONLY when a surviving AST ITEM references a dissolved header — sea-only references are tail-DCE garbage); `eliminateMap` → exactly RAUW + erase Map + erase As (`sweepDebris` DELETED); tests re-pinned (dissolved dimvar now dies at the TAIL via the `runTail` helper; the producing Map's lambda pinned dead-in-the-sea/item-anchored instead of erased) | `233291f` | DONE |
| 5 (D-c) | between-block item re-placement | — | **BLOCKED — R-D finding (below)**; the implementation was written, debugged to golden-byte-identity on reduce_warp, and then REVERTED uncommitted when the frontend trigger failed (see finding) |
| 6 (R-B/D-e) | exile canonicalizeConsts | — | NOT EXECUTED (blocked); **C5 measurement taken and recorded below** |
| 7+8 (D-f + battery) | anchor gate + un-skip | — | NOT EXECUTED (blocked; the un-skip's numerical acceptance is unreachable without D-c) |

## R-D FINDING (the stop): fix2 D-c's re-placement rule has no in-scope home for the fill's covered-dim round base on the trigger shape

**Where:** Task 3 commit 2. The plan expected `TestFrontendMatmulScalePipeline` to stay green at this commit ("it only asserts emit succeeds"); with a faithful D-c implementation it does not — `cxx: dimvar 120 has no enclosing loop in scope`. Per fix2 §0 R-D ("when a lower meets a shape its written rules do not cover, it fails closed and prompts for a ruling — it never bridges the gap with improvised mutation") and the executor constraints, the D-c work was reverted and the tree left green at `233291f`.

**Probe evidence** (frontend `matmul_scale`, Appendix A written variant; HIR window + tiling/tile-lower/schedule, pre-final-lower):

- Provenance: M Between=`#152`(CTA)/Inside=`#146`(Thr); Q Between=`#150`/Inside=`#144`; **N (reduced) Between=`#148`=X / Inside=`#142`=home**. Between set (X→home chain) = `[#146 M-inside, #144 Q-inside]`.
- Searched tiling: **all tiles 64**; extents M=64, N=128, Q=64 → **X runs 2 rounds per bracket** (Tn=64 < N=128).
- The schedule tree (pre-lower): `CTA#152 → Seq#150 → Seq#148(X) → Thr#146(M-walk: binop#94, memload#160, fragstore#163 = the fill) → Seq#144 → Seq#142(home: loads, multiply#16, map, reduce)` — the fill sits in the M-walk exactly as fix2 §1.2 states ("placed there by design per F8/R1").
- The fill's operand face: `fragstore#163 = [memload#160 (source load, m-axis — rebuilt onto the fresh element IV ✓), fragnew#159, dimvar#120 (= X's IV — the N-window ROUND BASE, the covered dim's biv), keyedattribute#162 (loadtile)]`.

**What faithful D-c produces** (betweenBlks[i] node items → freshBlks[i], in order, before Init / the nested chain Block, stamped at the fresh depth, joined into the existing rebuild session AFTER the walk/consumer rebuilds — fix2's "the existing rebuild then covers the re-placed items"):

```
SeqLoop#193 (fresh M-element loop, takes X's position)
  binop#94, memload#160, fragstore#163     ← re-placed fill chain (correct per D-c)
  SeqLoop#184 (fresh Q-element loop)
    binop#117 (re-placed)  Init  X#148{...home...}  Combine  moved-consumers
```

Structurally exactly what D-c prescribes (fill precedes Init; chain edge intact; stamps at element depth; `TestReducerBracketPlacement`-style adjacency preserved with re-placed items before Init). It fails twice on this shape:

1. **Scope:** `fragstore#163`'s biv still references `dimvar#120` — X's own IV. X now sits INSIDE the element loops; the fill sits OUTSIDE X. The emitter renders a fill biv as `<X-iterator>.Index()` (LoadTile coordinate, `internal/cg/cxx/memory.go` stmtFill → emitExpr → the KindDimVar case), which requires X's loop to enclose the fill's position → `cxx: dimvar 120 has no enclosing loop in scope`. The between-IV→fresh-IV rebuild cannot cover it: **X's IV is not a between IV** (X survives; the subs keys are only `{#77 (M-inside), #104 (Q-inside)}`).
2. **Semantics (independent of emission):** the lhs FRAG window covers the N-window (its round base IS the biv). With Tn=64 < N=128 the fill must re-execute per X-round (it executed per (m, r) pre-restructure). Re-placed outside X it executes once per (m) element and cannot serve round 1's window — the reduce would combine round-0 data twice.

**Root cause:** fix2 §1.2's mechanism trace followed the fill's DATA chain (`#160 → #163 →chain→ #164`) but never the fill's BIV — the covered dim's round base, which references a SURVIVING loop's IV (X's). C2's "the element loop is the *isomorphic* re-placement of the walk" covers the walk's OWN dim (m) only; the fill's other axis (the n-round) has no isomorphic image in L′ because X moved inside. D-c's rule is sound for between-block items whose foreign coordinates are all between-IV-derived (the source load, the coordinate binops — both rebuild cleanly, and reduce_warp stays golden-byte-identical); it is incomplete for per-round effectful items (the fill).

**Candidate resolution shapes for the ruling (NOT implemented — the executor never improvises a lower):** (a) re-place a between-block item whose operand closure references a non-between surviving loop's IV (X's) into the innermost re-placed position still dominated by that loop — i.e., inside X before home — where both the chain edge and the per-round epoch count are preserved (this generalizes D-c's "in order" by scope rather than by block); (b) split the rule by item class (per-round effectful items vs per-element items); (c) a coordinate remap minted by a ruled rule. Note the rebuild-order subtlety recorded below (Deviation 4) — whichever shape is ruled, the re-placed items' rebuild must stay AFTER the walk/consumer rebuilds to keep C5.

## D-e canonicalizeConsts C5 measurement (recorded per fix2 D-e, "documented in the summary either way")

Measured read-only at HEAD `233291f` (the `canonicalizeConsts` call commented out, goldens run, edit reverted — never committed):

- **Without the dedup BOTH goldens drift**: `TestGoldenMatmulStore` and `TestGoldenReduceWarp` FAIL (`reduce_warp` first difference: duplicate hoisted-const locals and renumbered value locals); `TestReduceShapesEmit` still passes; `TestConstLiteralDedup` fails with e.g. `duplicate literal const: node #3 and node #9 both carry 0 of u32`.
- **Decision per fix2 D-e's rule:** the goldens NEED literal dedup → the R-E escape-clause branch applies (GVN cannot see post-typeinfer nodes — nil type cells — so no existing pass covers it): a MINIMAL tail dedup as a pipeline-owned step wired in TAIL BEFORE the dce instance (so the dce item sweep collects the erased duplicates' items), NOT a new registry ID and NOT in final-lower. Task 4 was not executed (blocked by the R-D stop); the measurement de-risks it for the next session.

## Verification results (per landed commit)

- **Hermetic gate ×4 commits:** `go build ./... && go vet ./... && go test ./...` — all green at each of `f7a649a`, `f9f2c69`, `52ec1f2`, `233291f`; working tree clean at HEAD.
- **C5 golden regression:** `matmul_store.cu` md5 `3e5aa73aad878bf7048584b45feb11f2`, `reduce_warp.cu` md5 `7e14cba02cbd0711e18650cc51563c29` — byte-identical to fix2 §4.8.3's pinned values at every landed commit (and during the D-c work after the rebuild-order fix).
- **Stdlib-only:** `go list -m all | wc -l` == 1.
- **GPU e2e gate:** NOT run — blocked at the R-D stop before Task 5; the trigger's numerical acceptance is unreachable without D-c, and the un-skip was correctly left in place.
- **grep gates:** no `sweepDebris`, no fixpoint orphan loop in `finalizeDissolutions` (Task 3 commit 1 done-criteria grep, clean).

## Deviations from plan

1. **[Rule 3 — Blocking] dce AST-roots needed loop-header roots** (`52ec1f2`): the first tail run over a real scheduled graph erased the ForLoop nodes — nothing roots them (no item references a ForLoop through operand edges; the "nest" attrs point the other way) — and the emitter failed `cxx: loop 150 carries no soc annotation` (it reads each block's `Loop()` operand face). Fixed by adding every Loop block's `Loop()` node to the root set (`astRootNodes`); documented in the pass comment as part of "the post-schedule truth of liveness". Not anticipated by the plan's "every ASTItem.Node" wording; required by its own success criteria.
2. **[Rule 1 — Bug] maplower As-capture ordering** (`233291f`): my first D-b rewrite captured `ir.AsOf(m)` AFTER `lr.mctx.Erase(m)` — but AsOf reads the Map's operand records, which Erase kills, so the trailing As survived (suite caught it: "no live as may survive Map Lower, got 1"). Fixed: capture before the erase, as the original did.
3. **[Rule 3 — Blocking] cxx emitter lambda-closure over-approximation** (`233291f`): the prescan's `markClosure` ran over EVERY surviving lambda — valid only while final-lower's debris sweep erased the producing Maps' lambdas. With D-b they survive, and marking their closures skipped live statements (the load chain; transitively the ThreadMapping items → `cxx: reducer 122 references an unconstructed ThreadMapping`). Fixed to mark ONLY the reducer constructions' combiner-lambda closures (the comment's own documented intent) plus a `KindLambda` no-op statement case. Emitted text unchanged — both goldens byte-identical. The emitter is outside fix2's declared scope; flagged here for the record.
4. **D-c rebuild order (observed during the reverted work, recorded for the ruling):** the re-placed items' rebuild must run AFTER the walk's and the moved consumers' rebuilds (fix2 D-c: "the EXISTING between-IV→fresh-IV rebuild THEN covers the re-placed items"). My first version rebuilt the re-placed binop in place first, so the moved store's rebuild found no sub-key and kept referencing the item → an extra emitted local → reduce_warp golden drift. Reordering restored byte-identity: a pure re-placed item whose consumers read rebuilt clones ends user-free and the emitter's B1 unreferenced-pure-item skip elides it — exactly fix2's "pure-expression items whose consumers read rebuilt clones simply die as sea garbage".

## D-item landing status

| Ruling | Status |
|--------|--------|
| R-A/R-B (verb-death only; no optimization in a lower) | IN FORCE in the landed code (D-a/D-b sweeps deleted; canonicalizeConsts still in final-lower pending Task 4) |
| R-C (chain edges carry ordering) | Preserved (re-placement keeps the fill→access ImpactBy edge; nothing crosses chain edges for erasure) |
| R-D (untranslatable = stop) | **INVOKED** — this document's finding |
| R-E (composite reuse) | LANDED — the tail memopt is the registered factory's second instance; no new registry IDs |
| R-F (pipeline + compiler modules) | LANDED — `internal/pipeline` (orchestrator, three windows) + `internal/compiler` (facade); `opt.BuildPipeline()` untouched as the pass catalog |
| D-a | LANDED (`233291f`) |
| D-b | LANDED (`233291f`) |
| D-c | **BLOCKED — R-D finding** (implementation reverted; evidence above) |
| D-d | LANDED (`f9f2c69` + the loop-header amendment in `52ec1f2`) |
| D-e | PARTIAL — tail wiring (memopt→dce) + facade migrations LANDED (`52ec1f2`); fill-liveness reachable in-pipeline and pinned at the wiring level; canonicalizeConsts exile NOT executed (Task 4 blocked) but the C5 measurement recorded (tail-dedup branch) |
| D-f | NOT EXECUTED (Task 5 blocked) |
| D-g | LANDED (`f7a649a` + `52ec1f2`) |

## Files

- **New:** `internal/pipeline/pipeline.go`, `internal/pipeline/pipeline_test.go`, `internal/compiler/compiler.go`
- **ir:** `internal/ir/walk.go` (WalkFrom extraction)
- **opt:** `internal/opt/dce/dce.go`, `internal/opt/dce/dce_test.go`, `internal/opt/finallower/{pass,maplower,reducer}.go`, `internal/opt/finallower/finallower_test.go`
- **cg:** `internal/cg/cxx/{emit,statements}.go` (the markClosure/lambda-item fix)
- **Consumers:** `cmd/tcacomp/main.go`, `internal/e2e/frontend_test.go` (facade migration; the KNOWN DOWNSTREAM WALL skip is still in place pending D-c+D-f)
- Untouched by design: the registry/factory (`opt.BuildPipeline` keeps its role), the bracket machinery (schedule placement model, tile-lower §8, `ScanBracketLaw`), both golden `.cu` files (checksums above)

## Open items (need the author's ruling)

1. **The D-c gap** (this document's finding): where does a between-block item whose closure references a SURVIVING loop's IV (X's — the fill's covered-dim round base) re-place, given X moves inside the element loops? Candidate shapes (a)/(b)/(c) above. Until ruled, the frontend trigger still emits without its fill (the pre-D-c behavior — items discarded, tree green) and `TestFrontendMatmulScale`'s skip still matches.
2. Task 4's ruled branch is ready to execute as-is once unblocked (tail dedup per the measurement).
3. Pre-existing, untouched per fix2 §5: `simplify/lower.go` Sweep 3; the lower-out-of-opt refactor; the audit's load→load forwarding and general dead-GMEM-load items; bz9 deferred items.

## Self-Check: PASSED

- Files verified present: `internal/pipeline/pipeline.go`, `internal/pipeline/pipeline_test.go`, `internal/compiler/compiler.go`, `internal/ir/walk.go`, `internal/opt/dce/dce.go`, `internal/opt/finallower/maplower.go`, this SUMMARY.
- Commits verified in `git log`: `f7a649a`, `f9f2c69`, `52ec1f2`, `233291f` (HEAD; working tree clean).
- Full hermetic gate green at HEAD; `go list -m all | wc -l` == 1; golden md5s match fix2 §4.8.3's pins.
