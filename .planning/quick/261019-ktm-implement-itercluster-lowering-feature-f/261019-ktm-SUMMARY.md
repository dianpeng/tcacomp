---
phase: quick-261019-ktm
plan: 01
subsystem: ir-lowering
tags: [ir, lowering, itercluster, simplification, pass, ssa]
requires:
  - "KindVersion 3 IR suite (quick-261018-bl0)"
  - "opt pass registry + dce/gvn pass pattern (quick-261018-l36, quick-261018-nga)"
provides:
  - "Five Input-list node kinds: IterClusterProject, KeyedAttribute, As, InductionVariable, TensorDimOf (KindVersion 4)"
  - "Map/Reduce/IterCluster Input-list layout with AsOf/ReduceInitOf/IterClusterProjectIndex read faces"
  - "internal/opt/simplify: simplify-lower (102), simplify-inline (103), simplify-verify (104), composed by opt/all"
affects:
  - internal/ir (kind, catalog, node, node_lambda, node_loop, node_iterfmt, mutate, verify, print, value)
  - internal/opt/all (pipeline composition)
  - internal/visualization (DOT golden regenerated)
tech-stack:
  added: []
  patterns:
    - "Input-list rule: every incoming stage reference rides the operand list (no hidden payload pointers)"
    - "Wrapper-node identity keys GVN (D13): distinct As/KeyedAttribute nodes keep identical stages apart"
key-files:
  created:
    - internal/ir/node_iterfmt.go
    - internal/ir/node_iterfmt_test.go
    - internal/opt/simplify/lower.go
    - internal/opt/simplify/inline.go
    - internal/opt/simplify/verify.go
    - internal/opt/simplify/simplify_test.go
  modified:
    - internal/ir/kind.go
    - internal/ir/catalog.go
    - internal/ir/node.go
    - internal/ir/node_lambda.go
    - internal/ir/node_loop.go
    - internal/ir/mutate.go
    - internal/ir/verify.go
    - internal/ir/print.go
    - internal/ir/value.go
    - internal/opt/all/all.go
    - internal/opt/all/all_test.go
    - internal/visualization/testdata/matmul_scale.dot
decisions:
  - "D1-D13 implemented exactly as recorded in the plan's reconciliation block (see Deviations for the three additive ir APIs the passes needed)"
  - "pkg matmul_scale.ir golden regenerated deliberately and came out byte-identical (statement IDs are builder-reserved ahead of chain-node allocation; stage spellings preserved through the unwrap paths)"
  - "Fully-inlined maps (zero declared params) are a legitimate verifier state — the input-count contract applies while params remain declared"
metrics:
  duration: "~31 min"
  completed: 2026-10-19
  commits: 4
  files-changed: 24
  tests: "13 packages green (clean test cache)"
---

# Quick Task 261019-ktm: Implement IterCluster lowering (Iter format + Simplification pass) Summary

**One-liner:** Lowered the frontend's IterCluster format to data-dependency Iter format — five new Input-list node kinds (KindVersion 4), Map/Reduce/IterCluster operand-list restructure, and a three-pass `internal/opt/simplify` (lower 102 / inline 103 / verify 104) that rewrites the doc's myfoo example into per-dim Iter+TensorDimOf+InductionVariable nodes with fully inlined lambda bodies, slotted between dce-verify and gvn in the default pipeline.

## What Was Built

### Task 1 — IR layer: Input-list restructure (commits e4b00d7 RED, 87fa79f GREEN)

- **Five node kinds** (D4/D2/D1/D6): `iterClusterProjectNode` (idx payload, GVN content field), `keyedAttributeNode` (key payload), `asNode` (variadic dim operands), `tensorDimOfNode` (interned TU32 axis const at operand 1), `inductionVariableNode` — all FamilyAbstract (pure, DimVar precedent), dispatched from `allocNode`, catalog rows + String spellings + legality-visible from day one.
- **Enum**: KindIterClusterProject/KeyedAttribute/As inserted after KindReturn (HIR); KindTensorDimOf/KindInductionVariable appended before KindEnd (LIR); **KindVersion 4** with the one-time-renumber note. catalog_test's tier-boundary pin updated to the new HIR tail (KindAs).
- **Map/Reduce/IterCluster**: `mapNode.as` payload deleted — NewMap appends a trailing As node (wrapper allocated before the stage, deterministic order); NewReduce wires `[piped, KeyedAttribute("init", init)]` with the signature unchanged; `NewIterCluster(origin, dims, inputs...)` variadic. All pkg/chain.go call sites compiled with zero changes (D11's compile-compatibility truth).
- **Read faces**: `AsOf` (backwards operand scan), `ReduceInitOf` (attr unwrap), `IterClusterProjectIndex`.
- **Verifier** (fail-closed): project→cluster, iv→iter, tdim→(RoleTensor, const axis), as→RoleDim operands, reduce operand 1 attr, As unique-and-last, map input count == param count (while params remain declared).
- **Printer**: reduce reads piped from operand 0 (D12) + init via ReduceInitOf; map as-clause from AsOf; TensorDimOf-bounded iters render `@a/tensordimof(lhs, 0)` instead of `?` (D5); new-kind value renderings; multi-input clusters join their inputs' stage text.
- **GVN identity** (D13): contentHash/Equal cover idx and key; wrapper-node identity keeps constructor-built stages apart (pinned, including the white-box shared-wrapper reduce that still proves the lambda machinery).

### Task 2 — Simplification passes (commits 90b58ee RED, bbf50ef GREEN)

- **simplify-lower (102)**: per-cluster dim derivation from projecting maps' As operands (map allocation order, As order, dedup); per-dim interned `TensorDimOf → Iter(TF32 0/1) → InductionVariable` triples, first MemLoad axis pairing wins (shared dims yield ONE iv); MemLoad axis slots rewired to IVs via SetDef; project uses rewired to cluster inputs (bounds-checked); projects then processed clusters erased with propagated Erase refusals. Bidirectional fail-closed coverage: an as-dim without a pairing AND a MemLoad RoleDim axis not covered by any As both error naming the node. Legacy builder clusters (no projects, non-load inputs) are a true no-op — lazy const allocation means not even a dead node appears (fingerprint-stable, pinned).
- **simplify-inline (103)**: D9 positional binding (KeyedAttribute elements unwrap, trailing As binds nothing); per paramRef: RAUW onto the binding + Erase, with body-root paramRefs first rewired through the new `SetLambdaBody` choke point; `ClearLambdaParams` strips dead declarations while keeping params with live non-declaration uses (acc-style survivors); use-free lambda InputParams erased. One MutContext, allocation-order determinism, idempotent.
- **simplify-verify (104)**: no live IterClusterProject; no lambda-body paramRef resolving to a still-declared param of the same stage (body-subtree DFS + param-side resolution); Task-1 placement/chain invariants via ir.Verify. Errors name node ID and kind.
- **Wiring**: `opt/all` blank-imports simplify; BuildPipeline order is dce(100) → dce-verify(101) → simplify-lower(102) → simplify-inline(103) → simplify-verify(104) → gvn(110) → gvn-verify(111) → schedule(900); Manager runs the L1 prefix over the doc example end-to-end and the result Materializes and verifies (pinned by test).

## Deviations from Plan

None of the D1–D13 decisions proved impossible — all implemented as recorded. Implementation-level deviations (Rules 1–3), all auto-applied:

### Auto-fixed Issues

**1. [Rule 3 - Blocking API] `ir.LambdaParams(n Node) [][]SSAUse` added (internal/ir/print.go)**
- **Found during:** Task 2 — the inline and verify passes must enumerate a stage's lambda parameter DECLARATIONS in positional order; the params live in the Lambda record (not operand slots) and no exported face exposed them (`LambdaBodies` covers only bodies). Reconstructing order from per-param use lists is impossible (records append into different vars).
- **Fix:** exported read face beside `LambdaBodies`, same pattern.
- **Files:** internal/ir/print.go
- **Commit:** bbf50ef

**2. [Rule 1 - Bug + Rule 3 - Blocking API] `(*MutContext).SetLambdaBody` added (internal/ir/mutate.go)**
- **Found during:** Task 2 — the plan's inline step ("RAUW the paramRef, then Erase it") corrupts any graph whose lambda body ROOT is a paramRef (an identity map, e.g. `map(%x, lambda(v): v)`): `Lambda.Body` is a raw pointer RAUW cannot reach, so erasing the ref leaves the body dangling at a tombstone and ir.Verify flags "erased node is still reachable".
- **Fix:** sanctioned choke point for the one raw-pointer edge; inline rewires the body before erasing such refs. Pinned by TestInlineReduce (identity piped map).
- **Files:** internal/ir/mutate.go, internal/opt/simplify/inline.go
- **Commit:** bbf50ef

**3. [Rule 3 - Blocking API] `ir.IterClusterProjectIndex(n Node) (int, bool)` added (internal/ir/node_iterfmt.go)**
- **Found during:** Task 2 — lowering must read a project's cluster input-slot index; the idx payload was write-only.
- **Files:** internal/ir/node_iterfmt.go
- **Commit:** bbf50ef

**4. [Rule 2 - Correctness] Map input-count invariant accepts the fully-inlined form (internal/ir/verify.go)**
- **Found during:** Task 2 — after simplify-inline, a map declares zero params but keeps its value inputs; the Task-1 count check as written made the pass's own output unverifiable, so the pipeline could never pass its post-pass legality/verify steps.
- **Fix:** the count contract (Input[i] ↔ param i, D4) applies while params remain declared; `len(params) == 0` is the inlined shape. The Task-1 RED battery's mismatch case (0 inputs, 1 param) still fires.
- **Files:** internal/ir/verify.go
- **Commit:** bbf50ef

**5. [Rule 2 - Correctness] Lowering coverage is bidirectional (internal/opt/simplify/lower.go)**
- **Found during:** Task 2 — the plan's behavior block demands "a cluster whose MemLoad axis dim is not covered by any projecting map's As returns an error naming the dim", but the action text only described the as-dim→pairing direction; a load axis dim absent from every As would silently keep an uncovered iteration axis.
- **Fix:** every MemLoad axis that is a RoleDim InputParam must appear in the derived dim list, else error naming the axis node (non-dim axis values, e.g. constants, pass through untouched).
- **Files:** internal/opt/simplify/lower.go
- **Commit:** bbf50ef

### Notable confirmations (no deviation)

- **D11 goldens**: `pkg/testdata/matmul_scale.ir` was regenerated through the golden harness (`go test ./pkg/ -update`) and came out **byte-identical** (empty diff) — better than the plan's minimum. Statement numbers are builder-reserved ahead of chain-node allocation, and the new As/KeyedAttribute wrapper nodes render through the unwrap paths, so no textual delta exists. The DOT golden (`internal/visualization/testdata/matmul_scale.dot`) regenerated with exactly the expected delta: the new As node (n7) plus the now-operand-reachable dim param (n1), numbering shifts only, stage spellings unchanged.
- **D13 premise correction**: the plan asserted "none [assert] map-map merging today", but `internal/ir/value_test.go` TestValueLambdaMachinery did assert constructor-built map/reduce value equality. Resolved strictly within D13's recorded principle without weakening: the lambda machinery is still proven (as-less maps; a white-box reduce sharing the attr wrapper + combiner atoms), and NEW assertions pin the wrapper-identity consequence (distinct As/KeyedAttribute wrappers ⇒ unequal).
- **Mechanical expectation updates** (all tracking the new canonical structure, several strictly stronger): walk orders gain the As node; catalog HIR boundary KindReturn→KindAs; small-chain statement numbers #23/#24→#25/#26; gvn reduce operand layout `ops[1]`→`ops[0]` + ReduceInitOf check added; all_test membership list 2→8 IDs; cfg_test reads the as-order through AsOf and its fixture gained a matching operand (the new binding contract).

## Threat Model Dispositions

| Threat | Status |
|--------|--------|
| T-ktm-01 (tampering via direct mutation) | Mitigated — every rewiring rides SetDef/RAUW/Erase/ClearLambdaParams/SetLambdaBody; Erase refusals propagate; no direct field writes from the pass |
| T-ktm-02 (malformed Input-list shapes) | Mitigated — fail-closed verifier checks run via the manager after every pass; lowering bounds-checks project indices and errors naming the node |
| T-ktm-03 (nil/materialized DoS) | Mitigated — level gates evaluated before any mutation on all three passes |
| T-ktm-04 (silent corruption reaching codegen) | Mitigated — simplify-verify re-asserts no-live-projects, full inline, and placement invariants; pinned by the corruption tests |
| T-ktm-05 (info disclosure in diagnostics) | Accepted as designed — errors carry node IDs/kinds only |

## Verification

- `go build ./... && go vet ./... && go test ./...` green after each task (final run on a clean test cache: 13 packages ok).
- The doc's worked example (two rank-2 tensors, dims a/b/c, cluster of two MemLoads, map with two projects) lowers and inlines end-to-end through the registered default pipeline's L1 prefix, then Materializes and verifies — asserted by `TestSimplifyPipelineWiring`, not by inspection.
- No assertion weakened anywhere; the only golden deltas are the reviewed DOT regeneration and the byte-identical pkg golden.

## Self-Check: PASSED

- Files exist: internal/ir/node_iterfmt.go, internal/ir/node_iterfmt_test.go, internal/opt/simplify/{lower,inline,verify,simplify_test}.go, internal/opt/all/all.go — all present in the worktree.
- Commits exist: e4b00d7, 87fa79f, 90b58ee, bbf50ef (git log fc62ae0..HEAD).

## Revision Round — Input/ImpactBy-only Convention (commits 2264739 RED, 9249a4e GREEN)

User amendment after reviewing the round-1 IR: the convention is universal —
a node's inputs live ONLY in its Input list (data dependencies) and its
ImpactBy list (side-effect dependencies); no node may hide SSA/Node
references in record fields. Every node kind, loops included.

### Execution note

The executor hit an API rate limit mid-GREEN (print.go half-refactored, its
remaining patch script unapplied). The orchestrator session completed the
round in the same worktree: applied the intended print.go refactor (adapted
to the real file layout — round-1's tensorDimOfNode case sits between the
blocks the leftover patch expected), wrote the convention enforcement from
scratch, fixed three integration bugs the restructure surfaced, and updated
stale round-1 test expectations to the revision contracts.

### Audit table (node kind × previously hidden state × resolution)

| Node kind | Hidden state | Resolution |
|---|---|---|
| paramRefNode | `use SSAUse` payload | ops = [param] (D14) |
| iterNode | `dimUse`/`boundUse` payloads | ops = [dim, bound?, init, step] — bound slot only in bounded form (D14) |
| dimVarNode | `dimUse` payload | ops = [dim, init, step] (D14) |
| iterClusterNode | `dims` payload | dims LEAD the input list; ClusterDims/ClusterInputs derive the split (D14) |
| memStoreNode | `target access{tensorUse, axes}` payload | ops = [value, tensor, axes...]; payload = ImpactBy only (D14) |
| mapNode | `lam Lambda` record (SSAUse params + Body pointer) | trailing KindLambda node (D15) |
| reduceNode | `combiner Lambda` record | ops = [piped, KeyedAttribute, lambda] (D15) |
| keyedAttributeNode | — | value already operand 0; name/literal stay metadata |
| inputParamNode | — | name/typ/role are debug/diagnostic payload, not references — legal |
| effect bearers (memLoad/memStore/return) | `effectIns []SSAUse` | THE sanctioned ImpactBy spelling (direct []SSAUse on effect bearers) |
| const/binop/call/cast/... | — | already operand-only |

### New decisions (appended to the plan's D1–D13)

- **D14** — operand-only layouts for paramRef/iter/dimvar/itercluster/memStore;
  new verifier shape checks: iter/dimvar operand 0 is a dim InputParam,
  paramRef operand 0 is an InputParam, cluster dims lead the input list.
- **D15** — KindLambda node (ops = [params..., body]) carried as Map/Reduce's
  trailing operand; the `Lambda` struct is demoted to a transient
  builder-only construction record; LambdaParams/LambdaBodies/SetLambdaBody/
  ClearLambdaParams are reimplemented over lambda-node operand storage;
  As placement is "last value operand, before the lambda".
- **D16** — the convention is machine-checked: `internal/ir/convention.go`
  (hiddenRefFieldType + checkConvention reflect sweep wired into Verify).
  Type-level and fail-closed: a forbidden field trips even when zero-valued;
  nil types classify hidden. GVN contract narrowed accordingly — the lambda
  compares by identity like any other operand; the pre-revision structural
  body recursion is gone (cycle guards structurally unnecessary, pinned by
  the rewritten termination tests).

### Integration bugs the restructure surfaced (all fixed in 9249a4e)

1. **Scheduler false cycles** — kernel params now ride store operand slots,
   so first-wins statement ownership of shared tensor/dim params manufactured
   st2→st1 edges colliding with the effect chain's st1→st2 (cycle, stuck
   list). Fix: InputParam deps create no statement edges (params are
   entry-available leaves produced by no statement).
2. **simplify-lower wrong bindings** — project rewiring resolved raw
   OperandsOf indices, which now lead with dims (map inputs became the dim
   params instead of the loads). Fix: resolve via ClusterInputs.
3. **simplify-inline never erased params** — the keep-predicate filtered
   uses by the STAGE node, but post-D15 the declaring user is the stage's
   lambda node, so every param looked live. Fix: treat the stage's own
   lambda node as the declaration user.

### Test-contract updates (round-1 expectations → revision contracts)

Reduce ops [piped, attr] → [piped, attr, lambda]; map ops [vals..., As] →
[vals..., As, lambda]; walk orders include the lambda node (ID 9); DOT
golden regenerated (wrapper nodes now visible as ordinary nodes — statement
IDs unchanged, pinned); catalog HIR boundary KindAs → KindLambda; store
payload = ImpactBy only; DataDeps(store) = [value, tensor, axes...]; GVN
twin tests pin identity keying (operand-sharing twins equal, fresh lambda
nodes unequal). One genuinely inverted test condition fixed
(TestConventionReflectCoversAllKinds' Lambda assertion fatal'd on the
correct classification).

### Revision verification

- `go build ./... && go vet ./... && go test -count=1 ./...` — 13 packages
  ok on a clean cache, including the new TestConvention* suite (reflect
  enforcement, no-hidden-payload-bearers, shape checks, lambda node layout,
  identity, choke points).
- No assertion weakened: every updated expectation pins strictly more
  structure than its round-1 counterpart (the lambda/As/attr nodes are now
  asserted where they were previously invisible).

