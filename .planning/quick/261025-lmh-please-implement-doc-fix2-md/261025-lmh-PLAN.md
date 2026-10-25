---
phase: quick-261025-lmh
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/pipeline/pipeline.go
  - internal/pipeline/pipeline_test.go
  - internal/compiler/compiler.go
  - internal/ir/walk.go
  - internal/opt/dce/dce.go
  - internal/opt/dce/dce_test.go
  - internal/opt/finallower/pass.go
  - internal/opt/finallower/maplower.go
  - internal/opt/finallower/reducer.go
  - internal/opt/finallower/consts.go
  - internal/opt/finallower/finallower_test.go
  - cmd/tcacomp/main.go
  - internal/e2e/frontend_test.go
autonomous: true
requirements:
  - "fix2 D-a: finalizeDissolutions → verb-death only"
  - "fix2 D-b: eliminateMap keeps exactly RAUW + erase Map + erase As"
  - "fix2 D-c: restructure item re-placement into fresh element loops"
  - "fix2 D-d: DCE gains AST-item roots"
  - "fix2 D-e: tail wiring memopt → dce via the pipeline module (Option A)"
  - "fix2 D-f: the semantic anchor as a hard post-final-lower check"
  - "fix2 D-g: internal/pipeline orchestrator + internal/compiler facade"
  - "fix2 R-A/R-B/R-C/R-D/R-E/R-F: pass discipline + module architecture rules"
must_haves:
  truths:
    - "final-lower performs only verb-death erasures; the up-walk, the fixpoint orphan sweep, and sweepDebris no longer exist"
    - "internal/pipeline owns the composition (HIR + LOWER + TAIL); the TAIL wires memopt then an AST-item-roots dce instance, both via registered factories"
    - "internal/compiler is the single syntax→backend entry; cmd/tcacomp and internal/e2e/frontend_test.go contain no duplicated parse→build→dematerialize→pipeline→emit sequence"
    - "Between-block Node items re-place into the fresh element loops (fill precedes Init; chain edge intact; SchedBlock stamped at the element-loop depth)"
    - "The D-f anchor check fires fail-closed on a hand-severed Reduce chain"
    - "TestFrontendMatmulScale both Appendix A variants pass numerically (reduceTol, canary 0) with NO skip; memcheck+racecheck clean"
    - "matmul_store.cu and reduce_warp.cu goldens byte-identical throughout"
  artifacts:
    - path: "internal/pipeline/pipeline.go"
      provides: "the pass orchestrator: three explicitly-wired windows over the registry factories"
      contains: "func Run"
    - path: "internal/compiler/compiler.go"
      provides: "the user-facing facade: source/kernel → backend files"
      contains: "func Compile"
    - path: "internal/opt/dce/dce.go"
      provides: "AST-item-roots DCE instance constructor + item maintenance on erasure"
      contains: "NewASTRoots"
    - path: "internal/opt/finallower/pass.go"
      provides: "sweep-free finalizeDissolutions + the D-f anchor gate"
    - path: "internal/opt/finallower/reducer.go"
      provides: "D-c between-block item re-placement in buildElementLoop"
  key_links:
    - from: "internal/pipeline"
      to: "opt.Lookup(id).Factory()"
      via: "explicit per-window ID lists, fail-closed on unknown ID"
      pattern: "opt\\.Lookup\\("
    - from: "internal/compiler"
      to: "internal/pipeline.Run + internal/cg/cxx.Emit"
      via: "Build → Dematerialize → pipeline.Run → Emit"
      pattern: "pipeline\\.Run"
    - from: "internal/pipeline TAIL"
      to: "memopt rule (d) + dce.NewASTRoots()"
      via: "opt.NewManager over [memopt factory instance, dce AST-roots instance]"
      pattern: "NewASTRoots"
---

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<objective>
Implement doc/fix2.md exactly — the author ruling that makes lowers mechanical and moves cleanup ownership to the pipeline.

Purpose: close quick-261024-sdk's open item 1 — the frontend `matmul_scale` emits a reduced walk with no piped producer because final-lower's orphan sweeps did DCE's job inside a lower and erased the live fill→access chain. The fix: final-lower only translates verbs (R-A/R-B), a new pipeline module wires the cleanup tail (R-E/R-F), a compiler facade exposes one entry (R-F), and the semantic anchor becomes a hard check (D-f).

Output: `internal/pipeline`, `internal/compiler`, sweep-free `internal/opt/finallower`, AST-item-roots DCE, migrated `cmd/tcacomp` + `internal/e2e/frontend_test.go`, the §4.8 fundamental test battery, and a numerically-correct un-skipped frontend e2e.
</objective>

<context>
@doc/fix2.md — THE normative spec. Its §0 rules (R-A..R-F), §2 constraints (C1-C5), §3 rulings (D-a..D-g), and §4 ordered plan are authoritative. When this plan and fix2.md ever appear to disagree, fix2.md wins and R-D applies: stop and surface the conflict, never improvise.
@doc/fix1.md — the prior sea-of-nodes ruling this extends (background only).
@.planning/STATE.md

Codebase facts verified 2026-10-25 (read these files first; names are current):

- `internal/opt/finallower/pass.go` — `Run` order: insertRuntimeObjects → lowerReducers → lowerMaps → `finalizeDissolutions` (the up-walk at ~L247-258 + fixpoint orphan sweep at ~L265-280 to DELETE per D-a) → rewriteQueries → `canonicalizeConsts` (consts.go) → `sweepErasedItems` → ir.Verify. `sweepErasedItems` (L414) is the RemoveItemsIf-over-blocks pattern the tail dce will own.
- `internal/opt/finallower/maplower.go` — `eliminateMap` = RAUW + erase Map + erase As + `sweepDebris` (L93, DELETE per D-b).
- `internal/opt/finallower/reducer.go` — `buildElementLoop` (L292): builds fresh element loops innermost-first, inserts Init at elemBlk index 0 (L378), DISCARDS between-block items ("their pure items are discarded", L384-411 — this is what D-c replaces), rebuilds X/home/moved-consumer coordinates via `rebuildOperands`/`rebuildMemo` (L439-458, L839-919), retargets "nest" attrs, queues `dissolveJob`s. `moveCombineConsumers` at L581.
- `internal/opt/dce/dce.go` — level gate refuses `g.Materialized() || g.CFG() != nil` (an AST-attached scheduled graph passes — fix2 §5 confirms); MARK via `ir.Walk` (Level-1 roots: end node + params); SWEEP to fixpoint. `dce/verify.go` is the separate dce-verify pass @101 (untouched).
- `internal/ir/walk.go` — `Walk(g, v)`; the child-relation traversal (operands + payload uses) is package-private. Exporting a rooted-walk substrate (e.g. `WalkFrom(roots []Node, v Visitor)` with `Walk` delegating) is the mechanical enabler for AST-item roots. `internal/ir/astcfg.go` — `ASTItem{Node, Block}`, `ASTBlock.Items()/InsertItemAt/RemoveItemsIf`, `Graph.AST()`.
- `internal/opt/registry.go` + `factory.go` — `opt.Register/Lookup/Registered`, `opt.BuildPipeline()` (factories ARE re-invokable: registry_test pins fresh instances — R-E's mechanism). Registered defaults: dce@100, dce-verify@101, simplify-lower@102, simplify-inline@103, simplify-verify@104, typeinfer@105, gvn@110, gvn-verify@111, memopt@112, memopt-verify@113, tiling@114, tiling-verify@115, tile-lower@116, schedule@900, final-lower@950. `opt/all` blank-imports everything. `Manager.Run` enforces `Requires` against pass IDs already run in the SAME pipeline (tail memopt's Requires ["gvn"] is satisfied by the HIR window's gvn in one manager run; two same-ID instances are fine).
- `internal/opt/memopt/memopt.go` — level gate identical shape; rule (d) = fill liveness (261024-sdk D-f); `memopt_test.go` has the direct-invocation fill fixture to lift for the wiring pin.
- `cmd/tcacomp/main.go` `compileKernel` (L100-126) and `internal/e2e/frontend_test.go` `compileFrontendKernel` (L154-199) — the duplicated sequences to delete; `TestFrontendMatmulScale` L296-312 holds the skip contingency to remove.
- Goldens: `internal/cg/cxx/emit_test.go` → `testdata/matmul_store.cu`, `testdata/reduce_warp.cu` via `golden.Assert` (+ `ScanBracketLaw` in `internal/cg/cxx/bracket.go` — the R-A textual scanner).
- Fixture helpers in `internal/opt/finallower/finallower_test.go`: `newRedFx`/`runRedFx` (tiled reduce through tile-lower+schedule+final-lower with explicit IterDecisions), `liveOfKind`, `walkItems`, `TestTiledReduceEndToEnd`, `TestConstLiteralDedup`.
- e2e: `skipUnlessE2E`, `BuildFixture`, `RunCmd`, sanitizer pattern in `e2e_test.go` ~L308-340 (`compute-sanitizer --tool memcheck|racecheck`).
- Constraint check: NO `internal/pipeline` or `internal/compiler` package exists yet (verified). Go module `tcacomp`, stdlib-only (keep `go list -m all | wc -l` == 1).

OUT OF SCOPE (fix2 §5 — do NOT touch): `simplify/lower.go` Sweep 3; the lower-out-of-opt refactor; bz9 deferred items; the audit's load→load forwarding and general dead-GMEM-load items.
</context>

<tasks>

<task type="auto">
  <name>Task 1: pipeline module + DCE AST-item roots (fix2 §4 steps 1+2 — two commits)</name>
  <files>internal/pipeline/pipeline.go, internal/pipeline/pipeline_test.go, internal/ir/walk.go, internal/opt/dce/dce.go, internal/opt/dce/dce_test.go</files>
  <action>
COMMIT 1 (`refactor(261025-lmh): pipeline module — orchestration refactor, composition bit-identical`) — fix2 §4 step 1 / D-g / R-F:
- New package `internal/pipeline`: the pass orchestrator. Export `Run(g *ir.Graph) error` over a dematerialized graph, running ONE `opt.NewManager(...)` over the concatenation of three windows. Each window is an explicit ID list resolved through `opt.Lookup(id).Factory()`, fail-closed with a named error on an unknown ID — NEVER `opt.Registered()`/`opt.BuildPipeline()` (the registry is process-global and tests register fakes; the composition must be deterministic). Blank-import `internal/opt/all` so the factories are registered.
- Window lists at this commit (HIR + LOWER only, NO tail yet): HIR = [dce, dce-verify, simplify-lower, simplify-inline, simplify-verify, typeinfer, gvn, gvn-verify, memopt, memopt-verify]; LOWER = [tiling, tiling-verify, tile-lower, schedule, final-lower]. This must equal today's `opt.BuildPipeline()` ID sequence exactly — pin it with a test in `pipeline_test.go` asserting order-for-order ID equality against `opt.BuildPipeline()`.
- `opt.BuildPipeline()` and the registry are UNTOUCHED (registry stays the pass catalog for existing tests, per D-g).
- Gate: full hermetic suite + both goldens green (behavior must be bit-identical — this commit adds orchestration only).

COMMIT 2 (`feat(261025-lmh): dce gains AST-item roots (fix2 D-d)`) — fix2 §4 step 2 / D-d:
- `internal/ir/walk.go`: extract the iterative child-relation traversal so it can run from arbitrary roots — export `WalkFrom(roots []Node, v Visitor)` (same Visitor/VisitResult, same children relation: operand defs then payload uses, live records only) and make `Walk(g, v)` delegate to it with its level-aware root set. Pure restructuring; no behavior change to `Walk`.
- `internal/opt/dce/dce.go`: give `Pass` an unexported mode (`astRoots bool`); registry Factory keeps returning the zero `Pass{}` (priority-100 behavior unchanged). Add exported constructor `NewASTRoots() Pass` returning the tail instance. In AST-roots mode, `Run` additionally: requires `g.AST() != nil` (named error otherwise — the mode exists only post-schedule); MARK = the existing `ir.Walk` (end node + params) PLUS `ir.WalkFrom` over every AST item node collected recursively through all blocks (Groups, Loop blocks, the Prototype — every `ASTItem.Node`); after the sea sweep, perform item maintenance: `RemoveItemsIf(it.Node != nil && it.Node.Erased())` recursively over every block (the `sweepErasedItems` pattern, now owned by the tail dce per D-e — this is what collects the items of memopt's erasures). Level gate unchanged (an AST-attached graph is neither materialized nor CFG-carrying — it passes).
- Unit pins in `dce_test.go` (fundamental only): (i) an AST-attached graph whose RuntimeCallMethod is an AST item with zero SSA users SURVIVES the AST-roots instance (the D-d headline — effectful runtime calls are roots); (ii) a non-item sea node unreachable from items/end/params is COLLECTED; (iii) erased nodes' AST items are swept. Construct graphs directly through ir constructors (the memopt/finallower test pattern).
  </action>
  <verify>
    <automated>go build ./... && go vet ./... && go test ./internal/pipeline/ ./internal/opt/dce/ ./internal/ir/ ./internal/cg/cxx/ ./... </automated>
  </verify>
  <done>Two commits landed; pipeline.Run reproduces BuildPipeline's exact composition (pinned); dce.NewASTRoots preserves item-anchored effectful calls and collects non-item garbage; full hermetic suite + goldens byte-identical.</done>
</task>

<task type="auto">
  <name>Task 2: tail wiring + compiler facade + migrations (fix2 §4 step 3 — one commit)</name>
  <files>internal/pipeline/pipeline.go, internal/pipeline/pipeline_test.go, internal/compiler/compiler.go, cmd/tcacomp/main.go, internal/e2e/frontend_test.go</files>
  <action>
COMMIT (`feat(261025-lmh): pipeline TAIL window (memopt→dce) + compiler facade`) — fix2 §4 step 3 / D-e + D-g / R-E + R-F:
- `internal/pipeline` grows the TAIL window, wired AFTER final-lower: [memopt (a SECOND instance from `opt.Lookup("memopt").Factory()` — R-E composite reuse, never a new registry ID), dce (`dce.NewASTRoots()`)]. Order is fixed: memopt FIRST so dce sees the final residue and its item sweep collects memopt's erasures (D-e). Expose the tail composition for tests (e.g. an exported `TailPasses() []opt.Pass` returning the exact instances `Run` wires) — `Run` stays the single composition entry. Tail memopt's region rules (a)-(c) no-op on the LIR window (C1 chain-totality gate) while rule (d) (fill liveness) is now reachable in-pipeline — this closes the fill-flavored half of the 2026-10-20 audit's dead-load finding; the other audit items stay open and untouched.
- New package `internal/compiler`: the facade (R-F). One user-facing flow: parse/semacheck via `internal/syntax`, per-kernel `Build` + `Dematerialize`, `pipeline.Run`, `internal/cg/cxx.Emit`, options threading. Minimal API: `CompileKernel(k *syntax.Kernel, opts cg.Options) (*cg.Result, error)` plus a path-level convenience (`CompileFile(src string, opts cg.Options) (*syntax.Module, *cg.Result, error)` or equivalent) — keep it small; the facade owns NO pass logic.
- Migrate consumers, deleting their duplicated sequences: `cmd/tcacomp/main.go` `compileKernel` becomes flag/IO + the facade call; `internal/e2e/frontend_test.go` `compileFrontendKernel` becomes fixture-writing + the facade call + its .cu filtering. Behavior must be IDENTICAL (fix2 §5: any drift is a blocker, not a tuning exercise) — proven by the full hermetic suite + both goldens + the hermetic `TestFrontendMatmulScalePipeline`.
- Hermetic wiring pin in `pipeline_test.go`: a fill-liveness fixture (lift the direct-invocation rule-(d) fixture shape from `memopt_test.go` — FragStore with no live buffer loads + its source GMEM load) run through the REAL tail composition (`TailPasses()` via `opt.NewManager`) asserting the fill AND its source load are erased in-pipeline — rule (d) pinned at the wiring level for the first time. The load→load forwarding audit item is NOT covered and stays open.
  </action>
  <verify>
    <automated>go build ./... && go vet ./... && go test ./... && go list -m all | wc -l</automated>
  </verify>
  <done>TAIL wired with factory-reused memopt + AST-roots dce; internal/compiler is the only place composing syntax→backend; cmd/tcacomp and the e2e test contain no pipeline sequence; fill-liveness fires through the real tail wiring (pinned); hermetic suite + goldens green; module count 1.</done>
</task>

<task type="auto">
  <name>Task 3: finallower — delete the sweeps, then re-place between-block items (fix2 §4 steps 4+5 — two commits)</name>
  <files>internal/opt/finallower/pass.go, internal/opt/finallower/maplower.go, internal/opt/finallower/reducer.go, internal/opt/finallower/finallower_test.go</files>
  <action>
COMMIT 1 (`refactor(261025-lmh): finallower deletes every non-verb erasure (fix2 D-a/D-b)`) — fix2 §4 step 4:
- `pass.go` `finalizeDissolutions`: DELETE the transitive up-walk (`up`, the `ir.UsesOf` walk ~L247-258) and the fixpoint orphan sweep (~L265-280). What remains per D-a: erase the between loop header node, then its own header nodes — the IV, the loop's BoundedDimVar wrapper, its KeyedAttributes — each through `lr.mctx.Erase` (Erase's live-use refusal is the backstop; the dim PARAM and bound-max are shared with the fresh element loop and are never candidates). Dead coordinate spines left in the sea are the tail dce's input. Narrow the dissolved-IV hard error: it fires ONLY when a surviving AST ITEM still references the dissolved IV (the emitter would have no iterator for it); sea-only references are garbage DCE owns (lower never crashes on garbage it no longer makes — negative pin 8.2(a)'s premise).
- `maplower.go`: `eliminateMap` keeps EXACTLY the three-step verb — RAUW onto body, erase Map, erase the trailing As. DELETE `sweepDebris` and its call; the lambda subtree's death is DCE's (D-b). Update the file-header comment that promises "full debris disposal".
- Update finallower tests that pinned debris disposal (e.g. the lambda-debris assertions in `TestMapLowerEliminatesMap` / `TestTiledReduceEndToEnd`'s "exactly the combiner lambda survives"): garbage now survives final-lower and dies at the TAIL — re-pin by running the tail composition after `runFinalLower` in those tests (import `internal/pipeline`'s `TailPasses()`), or assert the verb only. Keep every placement pin unchanged.
- Gate: full hermetic suite + goldens (unaffected shapes have no between-block fill; garbage collection equivalence is the tail's job).

COMMIT 2 (`fix(261025-lmh): finallower re-places between-block items into the element loops (fix2 D-c)`) — fix2 §4 step 5:
- `reducer.go` `buildElementLoop`: replace the "between blocks … are discarded" behavior with the D-c re-placement rule. For each betweenBlks[i] ↔ fresh element loop i (the 1:1 outer→inner mapping the chain construction already builds): every NODE item of betweenBlks[i] re-places into fresh loop i's item list, IN ORDER, positioned BEFORE Init — elemBlk becomes [re-placed items…, Init, X-block, Combine, moved consumers…]; outer fresh loops become [re-placed items…, nested chain Block]. Rationale to record in a comment: the fill must precede its readers (the chain edge carries ORDERING, R-C/C3), and R-A's "Init immediately before X" is preserved (the bracket scanner is the gate). BLOCK items of a between block are NOT re-placed — the chain child maps to its own fresh loop, which the chain construction already nests.
- Stamp every re-placed item's node `SetSched(depth_i, freshBlk_i)` (same depth discipline as the fresh loop headers) — negative pin 8.2(c) asserts this.
- Extend the between-IV→fresh-IV rebuild to cover the re-placed items: their operand slots go through the same `rebuildOperands`/`subs`/`rebuildMemo` session as X/home/moved consumers (cloned spines, shared leaves — one clone per def per session).
- This commit flips the frontend trigger's producer chain from gone to present: `TestFrontendMatmulScalePipeline` stays green (it only asserts emit succeeds); the numerical acceptance is asserted in Task 5 (the skip still matches until then — expected, tree stays green).
  </action>
  <verify>
    <automated>go build ./... && go vet ./... && go test ./internal/opt/... ./internal/e2e/ -run 'TestFrontendMatmulScalePipeline|TestTiledReduce|TestMap|TestReducer' && go test ./internal/cg/cxx/</automated>
  </verify>
  <done>finallower contains zero liveness-mutation sweeps (grep: no `sweepDebris`, no fixpoint orphan loop in finalizeDissolutions); between-block items survive into the element loops with rebuilt coordinates and correct stamps; goldens byte-identical; hermetic suite green.</done>
</task>

<task type="auto">
  <name>Task 4: exile canonicalizeConsts — measurement decides (fix2 §4 step 6 — one commit)</name>
  <files>internal/opt/finallower/pass.go, internal/opt/finallower/consts.go, internal/opt/finallower/finallower_test.go, internal/pipeline/pipeline.go</files>
  <action>
COMMIT (`refactor(261025-lmh): canonicalizeConsts leaves final-lower (fix2 R-B/D-e)`) — fix2 §4 step 6:
- Remove the `canonicalizeConsts` call from finallower's `Run` (R-B: const canonicalization / GVN-family dedup does not live in a lower). THEN MEASURE C5 before deciding the landing form:
  - Run both golden tests (`go test ./internal/cg/cxx/`). If `matmul_store.cu` and `reduce_warp.cu` are byte-identical WITHOUT any dedup: DELETE `consts.go` outright; delete/re-pin `TestConstLiteralDedup` with a comment recording the measurement.
  - If either golden drifts: the goldens NEED literal dedup — implement the MINIMAL tail dedup under R-E's escape clause (GVN cannot see post-typeinfer nodes — nil type cells — so no existing pass covers it). It is a pipeline-owned step wired in TAIL BEFORE the dce instance (so the dce item sweep collects the erased duplicates' items), NOT a new registry ID and NOT in final-lower (R-F/D-e). Relocate the routine (e.g. export it from a small home the pipeline calls, or inline the walk in `internal/pipeline`); re-pin the dedup invariant post-tail.
- Document the decision (measurement result + chosen form) in the commit message; the full detail goes in the task summary either way (fix2 D-e requires the summary record).
- Gate: hermetic suite + goldens byte-identical at the end of the commit regardless of branch taken.
  </action>
  <verify>
    <automated>go build ./... && go vet ./... && go test ./... </automated>
  </verify>
  <done>final-lower no longer canonicalizes consts; the C5 measurement is recorded; goldens byte-identical; suite green.</done>
</task>

<task type="auto">
  <name>Task 5: anchor pin + un-skip + the fundamental battery (fix2 §4 steps 7+8 — two commits)</name>
  <files>internal/opt/finallower/pass.go, internal/opt/finallower/finallower_test.go, internal/pipeline/pipeline_test.go, internal/e2e/frontend_test.go, internal/e2e/e2e_test.go</files>
  <action>
COMMIT 1 (`feat(261025-lmh): the D-f anchor check + the frontend e2e un-skips`) — fix2 §4 step 7 / D-f:
- `pass.go`: add the anchor check as the final gate of finallower's `Run` (after `ir.Verify`): for every live RuntimeCallMethod "Reduce" call, the call's value operand's transitive OPERAND closure (data-dep edges only — NEVER cross chain/ImpactBy edges, R-C) must contain at least one memory-access node (GMEM/SMEM/FRAG load family) whose subscript closure references at least one DimVar that is the IV of a block enclosing the call's SchedBlock (the bracket's loops: the element loops, X, home). Fail-closed with a named error naming the Reduce call — the author's semantic anchor ("a reduce's piped value always chains … into the nest") can never silently emit again.
- Un-skip `TestFrontendMatmulScale` in `internal/e2e/frontend_test.go`: DELETE the "KNOWN DOWNSTREAM WALL" skip contingency block (L296-312's `zeros == len(full)` skip) — both Appendix A variants must pass numerically (maxAbs ≤ reduceTol, canary 0). If the numerical gate fails, that is a new finding with its own ruling (R-D) — stop and report, do not tune.
- Sanitizers: extend the e2e sanitizer coverage (the `compute-sanitizer --tool memcheck|racecheck` pattern in `e2e_test.go` ~L308-340) to the frontend matmul fixture — both tools, zero errors.
- Gate on the GPU: `TCACOMP_E2E=1 go test ./internal/e2e/`.

COMMIT 2 (`test(261025-lmh): fix2 §4.8 fundamental battery`) — fix2 §4 step 8 (design.md: fundamental tests ONLY, nothing more):
- Positive: frontend e2e both variants (Task 5 commit 1) + loopforest/tile-lower §8 pins unchanged (gate only, no edits).
- Negative pins: (a) in `pipeline_test.go` — with the TAIL window removed, run HIR+LOWER over the trigger-shaped fixture and assert final-lower's output still verifies clean AND dead nodes remain present in the sea (lower never crashes on garbage it no longer makes); (b) in `finallower_test.go` — the D-f check fires (named error) on a hand-severed Reduce chain (redFx fixture, sever the value→memory-access chain before running final-lower's gate); (c) in `finallower_test.go` — a re-placed fill item's SchedBlock IS the fresh element-loop block at the same depth, with the fill→access ImpactBy chain edge intact (lift the fill into the redFx fixture's between block or use the trigger shape).
- Golden regression: both goldens byte-identical (existing `emit_test.go` tests are the gate — no edits).
- E2E gate: `TCACOMP_E2E=1 go test ./internal/e2e/` — full suite, and NO skip on the trigger shape remains (grep gate: `grep -rn "known downstream wall" internal/e2e/` returns nothing).
- Downstream verification per fix2 §4.1: pin the emitted frontend fixture with `cxx.ScanBracketLaw` (the bracket scanner over the emitted .cu — no code change expected).
  </action>
  <verify>
    <automated>go build ./... && go vet ./... && go test ./... && TCACOMP_E2E=1 go test ./internal/e2e/ -count=1 && grep -rn "known downstream wall" internal/e2e/ ; test $? -eq 1</automated>
  </verify>
  <done>D-f anchor gate live and negatively pinned; TestFrontendMatmulScale passes both variants numerically with sanitizers clean and no skip; negative pins (a)(b)(c) in place; goldens byte-identical; full e2e gate green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| frontend source → lexer/parser | Already exists; unchanged by this fix (the facade reuses internal/syntax verbatim) |

No new trust boundaries, no external services, no package installs (stdlib-only Go module — unchanged). The STRIDE surface is unchanged from the current compiler; every mutation continues to flow through MutContext choke points and the verifier gates.
</threat_model>

<verification>
Per-step gates (fix2 §4: tree green at EACH step = one commit; run after every commit):
- Hermetic: `go build ./... && go vet ./... && go test ./...`
- Goldens: `go test ./internal/cg/cxx/` (byte-identical `matmul_store.cu`, `reduce_warp.cu` — C5, non-negotiable at every step)
- GPU (final gate): `TCACOMP_E2E=1 go test ./internal/e2e/ -count=1` — full suite, no skip on the trigger shape, sanitizers clean
- Stdlib-only: `go list -m all | wc -l` == 1
Bracket machinery untouched (C1): no edits to schedule's placement model, tile-lower §8, or the bracket law — `ScanBracketLaw` stays green over both goldens.
</verification>

<success_criteria>
- All 8 §4 steps landed as ordered atomic commits (5 tasks, 7 commits) with the tree green at each.
- Every D-a..D-g ruling implemented; no deferred §5 item pulled into scope.
- `TestFrontendMatmulScale` both Appendix A variants numerically within reduceTol, canary 0, memcheck+racecheck clean, skip deleted.
- Goldens byte-identical throughout; hermetic suite green; e2e gate green.
- The canonicalizeConsts decision (measurement + form) recorded in the summary.
</success_criteria>

<output>
Create `.planning/quick/261025-lmh-please-implement-doc-fix2-md/261025-lmh-SUMMARY.md` when done, following the quick-task summary conventions in `.planning/STATE.md`'s Quick Tasks Completed table (description covering each D-item, gates run, goldens result, the D-e dedup measurement decision, commit hash).
</output>
