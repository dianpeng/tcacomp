---
phase: 261016-mpu-refactor-phase-1-ir-from-ast-style-to-ss
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/ir/node.go
  - internal/ir/ssa.go
  - internal/ir/kind.go
  - internal/ir/construct.go
  - internal/ir/mutate.go
  - internal/ir/walk.go
  - internal/ir/verify.go
  - internal/ir/print.go
  - internal/ir/doc.go
  - internal/ir/cfg.go
  - internal/ir/dimvar.go
  - internal/ir/tile.go
  - internal/ir/loop.go
  - internal/ir/dce.go
  - pkg/builder.go
  - pkg/chain.go
  - pkg/expr.go
  - pkg/instrument.go
  - doc/tca.md
autonomous: true
requirements: [P1R-01, P1R-02, P1R-03, P1R-04, P1R-05, P1R-06, P1R-07, P1R-08]
must_haves:
  truths:
    - "No name-keyed Param record is used as a def identity anywhere in internal/ir (grep for the old Param struct returns nothing); every kernel parameter is an InputParam IR node referenced exclusively through SSAUse"
    - "A BasicBlock stores statements in an intrusive doubly-linked list: removing a statement given its pointer is O(1) (no position search, no slice reindex)"
    - "pkg.Builder resolves every name through an on-the-fly symbol table at build time; the Graph it produces contains no unresolved name references"
    - "Building-phase DCE exists: after building, unreferenced side-effect-free SSADefs are marked dead, death propagates by dataflow to fixpoint, and dead statements are wiped"
    - "DimVar and BoundedDimVar are real IR nodes; BIV equality and include/belong relations from doc/optana/induction_variable.md are implemented and unit-tested"
    - "loadtile/storetile and a lowered for-loop node exist as IR kinds in the single module (FamilyTileOp and FamilyLoop no longer empty), each in its own submodule file"
  artifacts:
    - path: "internal/ir/cfg.go"
      provides: "CFG, BasicBlock with intrusive statement list, statement types"
    - path: "internal/ir/dimvar.go"
      provides: "DimVar / BoundedDimVar nodes and IV/BIV relations"
    - path: "internal/ir/tile.go"
      provides: "loadtile/storetile node kinds"
    - path: "internal/ir/loop.go"
      provides: "lowered for-loop node kind"
    - path: "internal/ir/dce.go"
      provides: "building-phase mark-dead + fixpoint + wipe"
    - path: "internal/ir/construct.go"
      provides: "InputParam node constructors; no name-keyed Param-as-def"
  key_links:
    - from: "pkg/builder.go"
      to: "internal/ir symbol table"
      via: "on-the-fly resolution during statement construction"
      pattern: "symtab|symbol"
    - from: "internal/ir/dce.go"
      to: "SSAVar.refs"
      via: "RefSize-based dead marking"
      pattern: "RefSize\\(\\)"
    - from: "internal/ir/cfg.go"
      to: "Statement"
      via: "prev/next intrusive pointers"
      pattern: "prev, next \\*Statement"
---

<objective>
Rewrite the Phase 1 IR from an AST-materialization shape into an optimizer IR shape, per the user's review `doc/review/phase1_1.md` (the authoritative requirements source) and `doc/optana/induction_variable.md`.

Purpose: The current IR carries name-keyed `Param` records as def identities (`paramRefNode.param`, `iterNode.dim/bound`, `storeNode.target Access{Tensor, Axes}`). An optimizer IR must have every reference be an `SSAUse` and every definition be an `SSADef`; names are debug payload only. Statements must support O(1) removal (intrusive list). DimVar/BoundedDimVar and the tile/loop lowering kinds must exist now so later phases lower into the same single-tier module instead of bolting them on.

Output: A refactored `internal/ir` (single Go module, organized into per-concern files) plus a reworked `pkg` builder that resolves symbols on the fly and runs building-phase DCE; IR unit tests extended (design.md: only fundamental tests — IR unit tests + e2e, no suites); goldens regenerated deterministically.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@doc/review/phase1_1.md
@doc/optana/induction_variable.md
@doc/tca.md
@internal/ir/doc.go

<interfaces>
Current handle layer (KEEP — review confirms the use-list/RAUW model is right; only its consumers change):

From internal/ir/ssa.go:
```go
type SSAVar struct { id uint64; node Node; define Value; refs []*ssaRefRecord }
func (v *SSAVar) RefSize() int          // live ref count — DCE's liveness probe
func (v *SSAVar) replaceAllReference(nv *SSAVar) // RAUW whole-chain move
type SSADef struct { ssa *SSAVar }      // zero = unwired
type SSAUse struct { rec *ssaRefRecord } // zero = unwired; killed records read dead
func UsesOf(def Node) []SSAUse
```

From internal/ir/mutate.go (the five MutContext choke points — keep all five; Erase must switch to the intrusive-list removal):
```go
func (ctx *MutContext) SetDef(u SSAUse, newVal Node) error
func (ctx *MutContext) AddUse(user Node, val Node) SSAUse
func (ctx *MutContext) RemoveUse(u SSAUse)
func (ctx *MutContext) ReplaceAllUsesWith(old, newVal Node) error
func (ctx *MutContext) Erase(n Node) error // currently scans block.statements slice — MUST become O(1)
```

What is being REMOVED (P1R-01): the name-keyed `Param` record and every payload field of kind `Param` / `[]Param` / `Access{Tensor Param, Axes []Param}`:
`paramRefNode.param`, `iterNode.dim/bound`, `iterClusterNode.dims`, `mapNode.as`, `Lambda.Params`, `storeNode.target`, `Graph.params`, `Add{Dim,Tensor,Scalar}Param`, `NewLambdaParam`, `NewParamRef`.

Single-BB fact (P1R-03): v0 has exactly one basic block per kernel and no control flow — SSA resolution is single-pass with NO phi nodes. `doc/tca.md` §2.3 defines the induction variable: defined by `(init, step)`; bounded form `@x/M` = `@x < M`.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Core IR reshape — CFG + intrusive statements + InputParam nodes (P1R-01, P1R-02, P1R-06, P1R-07)</name>
  <files>internal/ir/cfg.go, internal/ir/node.go, internal/ir/kind.go, internal/ir/construct.go, internal/ir/mutate.go, internal/ir/walk.go, internal/ir/verify.go, internal/ir/print.go, internal/ir/doc.go</files>
  <behavior>
    - A `BasicBlock` holds statements via `head`/`tail` `*Statement` and each `Statement` has `prev`/`next *Statement` — removal given a statement pointer is O(1) and does not touch any other statement.
    - `Block.statements []*Statement` slice is GONE; iteration is `for s := bb.First(); s != nil; s = s.Next()`. Existing iteration-order-sensitive consumers (printer, walk, verify) are updated to the list walk and still produce deterministic order.
    - A new node kind `KindInputParam` exists. Each kernel parameter (dim, tensor, scalar) is one `InputParam` node owned by the graph's param list (NOT a block statement). It carries: name string (DEBUG ONLY — never used for resolution or identity), an interned `Type` (nil for dim params), and a role discriminator. Its SSAVar is its SSADef pointee.
    - Every former name-reference becomes an SSAUse slot: map/store operand tensor references and axis references, iter dim/bound references, lambda-parameter references, map `as`-order references. Concretely: `paramRefNode` now holds `use SSAUse` (pointing at a lambda InputParam); `iterNode` holds `dimUse, boundUse SSAUse`; `Access` becomes `{tensorUse SSAUse, axes []SSAUse}`; `Lambda.Params` becomes `[]SSAUse` of lambda InputParam nodes; `mapNode.as` becomes `[]SSAUse`.
    - `Graph.Params()` returns the InputParam nodes; there is no exported name→param lookup on Graph.
    - `MutContext.Erase` unlinks a statement from the block in O(1) via the intrusive links (no slice scan).
    - Construction (`construct.go`) wires every one of these uses exactly as it wires operand uses today (fresh ref record in the target's SSAVar), so a freshly built graph is fully resolved with zero name lookups.
  </behavior>
  <action>
    Create `internal/ir/cfg.go`: define `CFG` (v0: exactly one `*BasicBlock`; struct exists so multi-block extension later is additive), `BasicBlock` with head/tail + O(1) `InsertBefore`/`InsertAfter`/`Remove`, and rehome `Statement` there as an intrusive node with `prev/next *Statement`, its existing `id/def/rhs`. Statement kinds split per the review: an assign statement (has LHS `SSADef` + RHS) vs a pure side-effect statement (RHS is a side-effecting op like store; still occupies the list). Keep the `Value()`/none-Value classification exactly as documented in doc.go — do not re-litigate it.

    In `node.go`/`kind.go`/`construct.go`: delete the `Param` record and all `Add{Dim,Tensor,Scalar}Param`/`NewLambdaParam` constructors; add `KindInputParam`, `inputParamNode`, and `NewInputParam(origin, role, name, typ)` graph-owned constructors. Convert the payload fields listed in <behavior> from `Param`/`[]Param`/`Access{Param,...}` to `SSAUse`/`[]SSAUse`/`Access{SSAUse, []SSAUse}`; update `fixedArity` (InputParam arity 0), `wireOperands`, the `Lambda` record, and every constructor (`NewIter`, `NewIterCluster`, `NewMap`, `NewReduce`, `NewStore`, `NewParamRef`) to accept use-pointing nodes and wire them through `val.BasePtr().ssa.addRef(...)` exactly like operand wiring. Update `walk.go` (lambda children), `verify.go` (use-def invariant now also covers param/axis/dim/as uses; surplus handling unchanged), and `print.go` (render InputParam name from the node's debug payload; access rendering now goes through use→def→InputParam instead of a Param field — output text for the oracle programs must remain semantically identical; golden text changes only where the representation genuinely changed, and the diff must be reviewed, not rubber-stamped).

    Do NOT widen any interface (doc.go directive 4 stands: Node stays BasePtr+KindOf). Do NOT touch the SSAVar/SSADef/SSAUse handle layer — it is correct per the review. Update `internal/ir/doc.go`'s design notes to describe the new shape.

    Downstream compile breakage in pkg/ is expected after this task; Task 2 fixes pkg/. This task is done when internal/ir compiles and its tests (updated in place) pass.
  </action>
  <verify>
    <automated>go build ./internal/... && go test ./internal/ir/ -count=1</automated>
  </verify>
  <done>Old `Param`-record grep over internal/ir returns zero hits; intrusive list removal is exercised by a unit test asserting O(1) unlink (unlink s2 from [s1,s2,s3], remaining order s1,s3, no slice anywhere); InputParam nodes exist per param and every reference test asserts `use.SSA().Node()` is the InputParam node; verifier passes on oracle-shaped graphs.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: On-the-fly symbol resolution in the builder + building-phase DCE (P1R-03, P1R-08)</name>
  <files>pkg/builder.go, pkg/chain.go, pkg/expr.go, pkg/instrument.go, pkg/builder_test.go, pkg/instrument_test.go, pkg/oracle_test.go, pkg/determinism_test.go, internal/ir/dce.go, internal/ir/dce_test.go</files>
  <behavior>
    - `pkg.Builder` keeps its public typed handles (Dim/Tensor/Scalar/Operand/Expr) unchanged in shape, but each handle now holds the resolved `SSAUse` pointing at the kernel's InputParam node (created eagerly at declaration time) instead of an `ir.Param` record.
    - Builder construction is a single pass: declaring a name creates its InputParam node AND enters (name → use/def) into a per-builder symbol table; every later name occurrence resolves through the table into SSAUse/SSADef. After Build(), the graph has no name-keyed references left — names survive only in InputParam debug payloads.
    - Building-phase DCE (internal/ir/dce.go, review's algorithm verbatim): (1) after building, for each statement whose SSADef is side-effect-free and has RefSize()==0, mark it dead; (2) run a dataflow worklist to fixpoint — a def whose every use is dead becomes dead (dead-use propagation), and a statement is dead only if it is side-effect-free; side-effecting statements (stores to known GMEM outputs, and the side-effect node the review describes for RHS-with-side-effect) are never marked dead; (3) wipe: unlink dead statements from the intrusive list in one pass and Erase their node chains. `map(...) output(%abc)` where %abc is unreferenced is DCEed; `map(...)` dummy ops with no traced output are dropped; `map(...) store(%input,...)` on known memory survives.
    - DCE reports deterministically: worklist processes statements in intrusive-list order, ties broken by node ID.
  </behavior>
  <action>
    Rewrite pkg/builder.go's declaration half: `Dim/Tensor/Scalar` handles gain a `use ir.SSAUse` field (replacing `param ir.Param`); declaration rungs call `g.NewInputParam(...)` immediately and store the returned node's def-use handle in the handle and in the builder's `map[string]...` tables (which now map name → handle-with-use, i.e., the on-the-fly symbol table the review mandates — note chain.go's header already documents that Phase 5's parser lowers onto these rungs, so this table IS the future AST-translation resolution path: keep it in pkg, exported nowhere).

    Update `pkg/chain.go` and `pkg/expr.go`: every place that built `ir.Access{Tensor: p, Axes: ...}`, `ir.Lambda{Params: ...}`, `NewIter(dim, bound, ...)`, `NewMap(..., as)`, `NewStore(target, ...)` now passes the resolved uses from the handles (axis binding validation — structural subscripts, arity — stays exactly as-is). `pkg/instrument.go` (oracle/instrumentation) updated to the same pattern. All eager checks and deferred-error plumbing (D-10) unchanged.

    Implement `internal/ir/dce.go`: `func (g *Graph) markDeadAndPropagate() map[*Statement]bool` + exported `func (g *Graph) BuildingDCE(ctx *MutContext) error`. Side-effect classification: a statement is side-effecting iff its RHS root is a store/side-effect kind OR any node reachable from the RHS has a side-effecting kind (walk once per statement, memoized by node ID); everything else (const/binop/select/cast/call/param-ref/iter/map/reduce) is side-effect-free. Fixpoint: worklist of newly-dead statements; for each dead statement, visit the defs its operand uses point at and re-test `RefSize()` counting only live (non-dead-statement) users — when a def's live-user count hits zero and it is side-effect-free, mark dead and enqueue. Wipe in intrusive-list order, calling the existing `MutContext.Erase` (now O(1)) for node cleanup. `pkg.Builder.Build()` runs BuildingDCE as its final step before sealing.

    Update pkg tests to the new handles where they touched `ir.Param`; add internal/ir/dce_test.go covering the review's three cases (side-effect store survives; unreferenced map-output DCEed; dummy side-effect-free op dropped) plus fixpoint propagation (a chain a→b→c where only a is unreferenced kills all three).
  </action>
  <verify>
    <automated>go build ./... && go test ./internal/ir/ ./pkg/ -count=1</automated>
  </verify>
  <done>Builder-produced oracle graphs verify clean and contain zero name-keyed references; DCE unit tests show the three review cases and fixpoint chain-kill; pkg tests all pass with deterministic output (determinism_test green).</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Specialized nodes — DimVar/BoundedDimVar, loadtile/storetile, lowered for-loop + tests/docs (P1R-04, P1R-05)</name>
  <files>internal/ir/dimvar.go, internal/ir/tile.go, internal/ir/loop.go, internal/ir/kind.go, internal/ir/construct.go, internal/ir/verify.go, internal/ir/print.go, doc/tca.md, internal/ir/doc.go, internal/ir/testdata/*, pkg/testdata/*, internal/golden/*</files>
  <behavior>
    - `KindDimVar` node: operands `[init, step]` (SSAUse), payload `name string` (debug only). Two DimVars are EQUAL iff (init, step) tuples are equal — `SameIV(a, b Node) bool` compares through operand uses. This is the IV EQ/NE relation from induction_variable.md.
    - `KindBoundedDimVar` node: operands `[dimvar, max]` (SSAUse — dimvar points at a DimVar node, max at the bound value/InputParam). BIV equality: same dimvar AND same max (`SameBIV(a, b Node) bool`). Include/belong: for normalized IVs (init=0, step=1), `Includes(a, b)` iff a.max <= b.max; `BelongsTo(b, a)` iff `Includes(a, b)`. Non-normalized operands return false (documented; range-inference analysis arrives later per the doc's own note).
    - `KindLoadTile`: operands `[tensor, iv-or-biv per tiled axis]` — a tile read from a tensor param at bounded-IV coordinates. `KindStoreTile`: operands `[value, tensor, iv-or-biv per tiled axis]`. Both join FamilyTileOp (no longer empty).
    - `KindForLoop`: the LOWERED loop node — operands `[iv(DimVar), bound, body]`; body is a single child subgraph root (lambda-style bearer or operand, choose one and document it). Joins FamilyLoop (no longer empty). All new kinds append BEFORE `KindEnd` and `KindVersion` bumps to 2; existing constants never renumber (IR-03 append-only rule).
    - All five kinds live in the SAME `internal/ir` module but in separate files: dimvar.go, tile.go, loop.go (P1R-05's submodule organization).
    - Every golden file that legitimately changed is regenerated by the pinned printer and remains byte-deterministic across two consecutive runs (determinism_test).
  </behavior>
  <action>
    Add the five kinds to kind.go (append-only, KindVersion=2, String() names: "dimvar", "bdimvar", "loadtile", "storetile", "forloop"; Family() routes dimvar/bdimvar to FamilyAbstract — they are mid-level, not scalar exprs — and the tile/loop kinds to their now-populated families). Implement nodes + typed constructors in dimvar.go/tile.go/loop.go following the exact allocNode/wireOperands pattern; the relation helpers (SameIV, SameBIV, Includes, BelongsTo) live in dimvar.go as pure functions over operand uses. Extend verify.go: dimvar arity 2, bdimvar operand 0 must resolve to a DimVar node (verifier flags otherwise — fail-closed), tile kinds require tensor operand 0 resolving to a tensor-typed InputParam, forloop operand 0 resolves to a DimVar. Extend print.go with deterministic rendering for the new kinds (match doc/tca.md §2.3-2.4 notation: `@x(init=0, step=1)`, `@x/M`, `loadtile(...)`/`storetile(...)`).

    Update doc/tca.md: add a short "IR node suite" subsection listing the node families and pointing at the submodule files, and record the IV/BIV relational definitions (EQ/NE on (init,step) and (dimvar,max); include `~>`, belong `<~` on normalized BIVs) citing doc/optana/induction_variable.md. Update internal/ir/doc.go design notes. Regenerate goldens by running the pinned printer over the oracle programs, then run the full suite twice and confirm byte-identical output (determinism gate).

    Do NOT implement the passes that would PRODUCE tile/loop nodes (cache-lower, iter-lower) — this task lands the node suite only; phases 3-4 lower into them.
  </action>
  <verify>
    <automated>go build ./... && go test ./... -count=1 && go test ./pkg/ -run TestDeterminism -count=2</automated>
  </verify>
  <done>All 5 kinds construct, verify, print deterministically, and round-trip in unit tests (including SameIV/SameBIV/Includes/BelongTo true AND false cases, and non-normalized Includes=false); FamilyTileOp/FamilyLoop non-empty; goldens regenerated and stable across double-run; doc/tca.md and doc.go updated.</done>
</task>

</tasks>

<verification>
- `go vet ./...` clean; `go test ./... -count=1` green.
- Grep gate: `grep -rn "type Param struct" internal/ pkg/` returns nothing; `grep -rn "\.param\b" internal/ir/` returns nothing outside InputParam debug payloads.
- Oracle programs still compile through pkg.Builder, verify clean, and print deterministically (goldens stable across two runs).
</verification>

<success_criteria>
Every review requirement (P1R-01..P1R-08) is implemented at full fidelity: symbols gone from the IR graph (names debug-only), every reference an SSAUse / every definition an SSADef, CFG→BB→intrusive-statements shape, on-the-fly single-pass SSA resolution at build time, DimVar/BoundedDimVar with the documented relations, loadtile/storetile + lowered for-loop kinds in a single module organized into submodule files, O(1) statement removal, mutation support reviewed and preserved (five MutContext choke points intact, Erase now O(1)), and building-phase DCE (mark dead → dataflow fixpoint → wipe) implemented and tested. Project constraints hold: pure Go stdlib (no new imports beyond stdlib), single-tier SSA IR, design.md's test discipline (fundamental IR unit tests only).
</success_criteria>

<output>
After completion, create `.planning/quick/261016-mpu-refactor-phase-1-ir-from-ast-style-to-ss/261016-mpu-SUMMARY.md`
</output>
