---
phase: quick-261017-vdj
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/ir/effect.go
  - internal/ir/effect_test.go
  - internal/ir/construct.go
  - internal/ir/node.go
  - internal/ir/schedule.go
  - internal/ir/schedule_test.go
  - internal/ir/walk.go
  - internal/ir/verify.go
  - internal/ir/dce.go
  - internal/ir/print.go
  - internal/ir/doc.go
  - internal/ir/cfg_test.go
  - internal/ir/walk_test.go
  - internal/ir/print_test.go
  - internal/ir/dce_test.go
  - internal/ir/verify_test.go
  - internal/ir/use_test.go
  - internal/visualization/dot.go
  - pkg/builder.go
  - pkg/chain.go
  - pkg/builder_test.go
  - pkg/oracle_test.go
  - pkg/determinism_test.go
  - pkg/instrument_test.go
autonomous: true
requirements: []
must_haves:
  truths:
    - "A graph under construction and between optimization passes has NO BasicBlock/Statement structure; it is a pure sea of nodes held by the Graph"
    - "Ordering of side-effecting nodes (stores per tensor) is explicit via ImpactBy effect-input edges, not via statement list position"
    - "After Materialize runs (post-passes, pre-codegen), the graph has exactly one BasicBlock whose statements print byte-identically to the current goldens"
    - "pkg.Builder's public API (Stmt/Build and handle types) is unchanged; Build returns a materialized, verified graph"
    - "go test ./... passes with zero diff under all testdata/ golden directories"
  artifacts:
    - path: "internal/ir/effect.go"
      provides: "Effect verdict + ImpactBy effect-input payload + construction Env tracking per-tensor effect chains"
    - path: "internal/ir/schedule.go"
      provides: "Level-2 materialization: deterministic producer-first scheduling of the sea-of-nodes into CFG/BasicBlock/Statement"
    - path: "internal/ir/node.go"
      provides: "Graph without live CFG at Level 1; materialized flag; level-aware accessors"
  key_links:
    - from: "pkg.Build"
      to: "ir.Materialize"
      via: "Build calls BuildingDCE (sea-of-nodes) then Materialize then pass-manager VerifyEach"
      pattern: "Materialize\\("
    - from: "internal/ir/walk.go roots"
      to: "params + effect chain heads"
      via: "Level-1 Walk roots are graph params plus terminal effect nodes, not BB statements"
      pattern: "effectRoots|chainHeads"
---

<objective>
Adapt tcacomp's phase-1 SSA IR (`internal/ir`) from its current "statements into a
BasicBlock at construction" shape to the two-level design documented in the
tinycudaopto skeleton comments (`/home/ps/Workspace/cuda/project/tinycudaopto/internal/ir/`):

- **Level 1 — sea-of-nodes**: the frontend/builder and ALL optimization passes
  operate on a statement-free node graph. Data deps are operand SSAUse slots
  (already the case); side-effect total order becomes explicit **ImpactBy**
  effect-input edges maintained by a construction-time **environment** object
  (restrict-alias rule: distinct tensor names never alias, per skeleton
  `node.go`/`memory.go` comments).
- **Level 2 — CFG/BB/Instructions**: produced ONLY by a final scheduling step
  after all passes, before codegen. One CFG, one BasicBlock (v0 has no control
  flow), intrusive Statement list — the existing `cfg.go` types are REUSED
  unchanged; only their *time of creation* moves.

This is a MODIFICATION of the existing `internal/ir` package (per quick-task
context: not a rewrite). The SSA handle layer (`ssa.go`: SSAVar/SSADef/SSAUse,
ref records, RAUW) already matches the skeleton's [5.1]/[5.2] model and stays
as-is. `pkg.Builder`'s public API stays pin-compatible.

Purpose: optimization passes must be able to mutate/reorder freely without a
statement list pretending to be an ordering; ordering is re-derived once, late,
by the scheduler.
Output: two-level IR in `internal/ir`; builder producing sea-of-nodes;
byte-identical golden output.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

<!-- Design reference (READ ALL COMMENTS — they are the spec): -->
@/home/ps/Workspace/cuda/project/tinycudaopto/internal/ir/node.go
@/home/ps/Workspace/cuda/project/tinycudaopto/internal/ir/memory.go
@/home/ps/Workspace/cuda/project/tinycudaopto/internal/ir/loop.go
@/home/ps/Workspace/cuda/project/tinycudaopto/internal/ir/bb.go
@/home/ps/Workspace/cuda/project/tinycudaopto/internal/ir/effect.go

<!-- Existing implementation to modify: -->
@internal/ir/ssa.go
@internal/ir/node.go
@internal/ir/cfg.go
@internal/ir/construct.go
@internal/ir/kind.go
@internal/ir/walk.go
@internal/ir/verify.go
@internal/ir/dce.go
@internal/ir/print.go
@internal/ir/doc.go
@pkg/builder.go
@pkg/chain.go

<interfaces>
<!-- Existing contracts the executor must preserve (from internal/ir): -->

SSA handle layer (ssa.go — UNCHANGED by this plan):
```go
type SSAVar struct { id uint64; node Node; define Value; refs []*ssaRefRecord }
type SSADef struct { ssa *SSAVar }                 // value-type handle; zero = unwired
type SSAUse struct { rec *ssaRefRecord }           // value-type handle; copies share the record
func (u SSAUse) SSA() *SSAVar / Def() Node / IsKilled() bool
func UsesOf(def Node) []SSAUse
func DefOf(u SSAUse) Node
```

Graph/Node surface (node.go — Level-1 accessors change):
```go
type Node interface { BasePtr() *Base; KindOf() Kind }
type Value interface { Node; IsNone() bool }
type Base struct { id; kind; loc; lastTouchedBy; ssa *SSAVar; erased; surplus; graph *Graph; stmt *Statement }
// stmt stays NIL through construction and all passes; set only by Materialize.
```

CFG surface (cfg.go — REUSED UNCHANGED, created later):
```go
type Graph struct { name; params []Node; cfg *CFG; nextID }  // cfg becomes nil until Materialize
func (g *Graph) Block() *BasicBlock  // nil pre-materialization
func (g *Graph) AddStatement(rhs Node) *Statement      // moves to schedule-time path
func (g *Graph) ReserveStatementID() uint64            // to be removed/repurposed (pkg stops reserving)
```

Payload-use wiring (construct.go):
```go
func wirePayloadUse(user Node, val Node) SSAUse   // effect inputs use THIS mechanism
type payloadBearer interface { forEachPayloadUse(fn func(SSAUse)) }  // verifier/Walk already follow these
```

Golden dirs that MUST NOT diff: internal/ir/testdata, internal/visualization/testdata, pkg/testdata
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add Effect verdict + ImpactBy effect inputs + construction Env (additive; no behavior change)</name>
  <files>internal/ir/effect.go, internal/ir/effect_test.go, internal/ir/construct.go, internal/ir/node.go, internal/ir/doc.go</files>
  <behavior>
    - storeNode gains effect-input payload (`effectIns []SSAUse`); its forEachPayloadUse enumerates tensor/axes uses THEN effect inputs, so Verify and Walk treat ordering edges as first-class graph edges in both chain directions.
    - Effect verdict per node: effect-free vs effectful (store is effectful; return is effectful-via-operand but carries no chain of its own — it reads the store through its operand, so no separate effect inputs).
    - Env: construction-time environment mapping tensor identity (the RoleTensor InputParam NODE, never a name — review P1R-01) to the current chain head SSAUse. NewStore through the env path wires ImpactBy = current head(s) for its target tensor, then updates the chain head to the new store.
    - Two stores to the same tensor chain second.ImpactBy -> first; stores to different tensors are isolated (independent chains).
  </behavior>
  <action>
    Create internal/ir/effect.go:
    - `Effect` verdict type with Free/Effectful constants and `Has()` (mirrors the skeleton's effect.go shape but tcacomp-local naming; do NOT import from tinycudaopto — it is a separate module, reference only).
    - `ImpactBy()` accessor on the nodes that carry effect inputs (storeNode first; leave a `effectBearer`-style internal interface or extend payloadBearer — prefer extending storeNode's existing forEachPayloadUse so verify.go/walk.go need zero changes).
    - `Env` struct: `NewEnv()`, `NoteStore(tensor Node, store Node)` / lookup keyed by the tensor InputParam node pointer. Restrict-alias rule goes in the doc comment (kernel tensors never alias — skeleton memory.go rationale).
    - Env-aware store constructor, e.g. `NewStoreChained(origin, env, tensor, axes, value)` — wires `wirePayloadUse(n, chainHead)` for each relevant chain head into effectIns, then env.NoteStore. Existing `NewStore` stays (un-chained) so current callers/tests compile.
    Add Graph-level retention of terminal effect nodes: `effectRoots []Node` on Graph, appended by the env-aware path, so Level-1 Walk/verify have non-statement roots (Task 2 consumes this; add the field now, populate in NewStoreChained).
    Update doc.go design notes to record the two-level shape decision.
    Do NOT change: ssa.go, cfg.go, walk.go, verify.go, dce.go, print.go, pkg/. Everything existing must still compile and pass.
  </action>
  <verify>
    <automated>go build ./... && go test ./internal/ir/ ./pkg/ 2>&1 | tail -5</automated>
  </verify>
  <done>New effect.go compiles; effect_test.go proves same-tensor stores chain (second store's ImpactBy resolves to first store), cross-tensor stores stay isolated; full pre-existing suite still green (zero testdata diff).</done>
</task>

<task type="auto">
  <name>Task 2: Sea-of-nodes Graph (no CFG at Level 1) + Materialize scheduler + migrate in-package consumers</name>
  <files>internal/ir/node.go, internal/ir/construct.go, internal/ir/schedule.go, internal/ir/schedule_test.go, internal/ir/walk.go, internal/ir/verify.go, internal/ir/dce.go, internal/ir/print.go, internal/ir/doc.go, internal/ir/cfg_test.go, internal/ir/walk_test.go, internal/ir/print_test.go, internal/ir/dce_test.go, internal/ir/verify_test.go, internal/ir/use_test.go, internal/visualization/dot.go</files>
  <behavior>
    - NewGraph creates NO CFG: g.cfg == nil, materialized == false. Block()/CFG() return nil pre-materialization (nil-safe; no panic paths — T-1.1-03 style).
    - Level-1 Walk roots: g.params (declaration order) + g.effectRoots (chain heads, effect order) — replaces the BB-statement root scan. Determinism preserved (no map iteration).
    - Materialize(g): deterministic producer-first schedule (reuse/refactor print.go's StmtNodes emission-order DFS — chain stages producer-first via pipeInput — into a shared internal helper used by BOTH printer and scheduler), creates statements with IDs from the graph's monotonic counter IN EMISSION ORDER, links the intrusive list, sets Base.stmt, sets g.cfg + materialized=true.
    - Statement numbering must reproduce today's dense construction-order numbering so canonical print stays byte-identical.
  </behavior>
  <action>
    node.go: Graph gains `effectRoots []Node` (set in Task 1) and `materialized bool`; cfg field stays but is nil until Materialize. Block()/CFG() documented as Level-2-only, nil before.
    schedule.go (new): `func (g *Graph) Materialize() error` — refuses if already materialized; schedules via the shared emission-order helper; assigns statement IDs from g.NewID() in emission order (dense: the scheduler runs before any further node construction, so numbering matches what the reserved-ID scheme produced before — VERIFY against goldens, do not assume); builds CFG{entry: bb} and appends StmtAssign/StmtEffect statements (classifyStmt reused). Erase's O(1) stmt-unlink path stays valid (stmt nil-check).
    construct.go: remove the statement-emitting body of AddStatement/AddStatementAs/ReserveStatementID from the construction path (keep thin versions used ONLY by Materialize internally, unexported if possible); NewGraph no longer creates the block. Keep every other constructor untouched.
    walk.go: level-aware roots — Level 1: params + effectRoots (push order mirrored so pop order keeps params first, matching today's determinism contract); Level 2 (materialized): today's statement scan. Update the file's design comment.
    dce.go: BuildingDCE rewritten for sea-of-nodes — mark-sweep: mark reachable from Level-1 roots (params + effectRoots) following operand slots, payload uses (now including ImpactBy), and lambda bodies; Erase unreachable. Post-materialization DCE (if any test uses it) keeps the statement-list sweep.
    print.go / visualization/dot.go: require materialized graph (assert non-nil Block with a clear error, not a panic); emission logic itself unchanged — it already iterates statements, which now exist only post-Materialize. Refactor StmtNodes emission order into the shared helper WITHOUT changing its output.
    verify.go: no structural change needed (it walks via Walk; ImpactBy already covered as payload) — but update doc comments to state both levels are verifiable.
    Migrate in-package tests: any test that calls AddStatement/Block() pre-materialization must instead build nodes (env-chained stores) and call Materialize first. cfg_test.go becomes the Materialize test home (statement order, intrusive list ops, O(1) Remove).
  </action>
  <verify>
    <automated>go build ./... && go test ./internal/... 2>&1 | tail -8 && git diff --stat -- '**/testdata' | wc -l</automated>
  </verify>
  <done>internal/... suite green; internal/ir/testdata and internal/visualization/testdata diffs are EMPTY (0 lines from the git diff --stat pipe); Materialize produces one BB whose statement order matches the golden canonical print byte-for-byte.</done>
</task>

<task type="auto">
  <name>Task 3: Rewire pkg builder to sea-of-nodes Build + full golden verification</name>
  <files>pkg/builder.go, pkg/chain.go, pkg/builder_test.go, pkg/oracle_test.go, pkg/determinism_test.go, pkg/instrument_test.go, internal/ir/doc.go</files>
  <behavior>
    - pkg.Builder public API unchanged: Dim/Tensor/ScalarParam/At/Stmt/Build signatures and handle types (Dim/Tensor/Operand/Expr) are pin-compatible — callers cannot tell Level 1 happened.
    - Stmt chains no longer reserve statement IDs; chains accumulate nodes in the sea (store construction goes through the env-aware chained path from Task 1 so multi-store kernels keep correct ordering edges).
    - Build(): emit return nodes (as today) → BuildingDCE (sea-of-nodes) → Materialize → pass.Manager with empty pass list + VerifyEach (unchanged D-10 path) → return graph. Build returns an error (never a half-built graph) on any verification failure.
    - Statement numbering in the oracle goldens (statement #1 / #2 doc-pinned expectation) is reproduced by the scheduler's emission-order IDs.
  </behavior>
  <action>
    chain.go: replace its AddStatement call(s) with env-chained node construction (thread the ir.Env through the chain; the chain's store rung calls NewStoreChained). Drop statement-ID reservation from Stmt() (remove ReserveStatementID usage; the reserved retID machinery in builder.go's output struct goes away — returns are nodes now, materialization assigns statement IDs).
    builder.go: Build() gains the Materialize step between BuildingDCE and the pass-manager verify; keep the io.Discard VerifyEach instrumentation path exactly (D-10: verification only via the pass-manager path, never direct Verify calls from pkg).
    Update pkg tests: anything asserting on g.Block()/statement counts directly builds then Materializes via Build (Build already returns materialized). Oracle/determinism tests must pass with byte-identical goldens.
    Finish doc.go: record the final two-level architecture (Level-1 sea-of-nodes + env/effect chains; Level-2 scheduled CFG; where optimization vs codegen plug in) per the skeleton's design notes.
    Update .planning/STATE.md quick-task table entry on completion (via the summary step).
  </action>
  <verify>
    <automated>go test ./... 2>&1 | tail -5 && echo "TESTDATA DIFF LINES:" && git status --porcelain -- '**/testdata/**' 'pkg/testdata/**' | wc -l</automated>
  </verify>
  <done>go test ./... fully green; `git status --porcelain` shows zero changes under internal/ir/testdata, internal/visualization/testdata, pkg/testdata; Build returns a materialized verified graph; builder public API source-compatible with all callers.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| pkg caller → ir constructors | builder/frontend input crosses here; malformed handles are the injection class (already handled by eager checks, D-10) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-quick-01 | Tampering | sea-of-nodes mutation during passes | mitigate | All chain edits stay behind the five existing MutContext choke points; ImpactBy records are ordinary payload records, so the verifier's existing use-def/def-use invariants cover them with zero new checkers |
| T-quick-02 | DoS (panic) | nil CFG accessed pre-materialization | mitigate | Block()/CFG() return nil instead of panicking; printer/dot assert with clear errors; all accessors nil-safe (T-1.1-03 pattern) |
| T-quick-03 | Elevation/invalid state | env wired inconsistently by hand-rolled NewStore | accept | NewStore (un-chained) remains for rejection tests; restrict-alias rule is a documented precondition per skeleton, enforced when frontend lands |
</threat_model>

<verification>
- go build ./... clean
- go test ./... fully green
- `git status --porcelain -- '**/testdata/**'` empty (goldens byte-identical across internal/ir, internal/visualization, pkg)
- go vet ./internal/ir ./pkg clean
</verification>

<success_criteria>
- Graph is statement-free between construction and Materialize; passes mutate pure sea-of-nodes
- Store ordering is explicit ImpactBy edges maintained by Env; verifier covers them via existing payload-use invariants
- Materialize produces the single-BB CFG in deterministic producer-first order, byte-identical canonical print
- pkg.Builder API pin-compatible; Build returns materialized+verified graph through the D-10 pass-manager path only
- Full test suite green, zero golden diffs
</success_criteria>

<output>
After completion, create `.planning/quick/261017-vdj-adapt-phase-1-ssa-ir-to-new-two-level-de/261017-vdj-SUMMARY.md`
</output>
