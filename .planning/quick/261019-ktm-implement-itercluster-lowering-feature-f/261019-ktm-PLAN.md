---
phase: quick-261019-ktm
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/ir/kind.go
  - internal/ir/catalog.go
  - internal/ir/node_iterfmt.go
  - internal/ir/node_iterfmt_test.go
  - internal/ir/node_loop.go
  - internal/ir/node_lambda.go
  - internal/ir/mutate.go
  - internal/ir/verify.go
  - internal/ir/print.go
  - internal/ir/value.go
  - internal/opt/simplify/lower.go
  - internal/opt/simplify/inline.go
  - internal/opt/simplify/verify.go
  - internal/opt/simplify/simplify_test.go
  - internal/opt/all/all.go
  - pkg/testdata/matmul_scale.ir
autonomous: true
requirements: ["QUICK-261019-ktm"]
tags: [ir, lowering, itercluster, simplification, pass, ssa]

must_haves:
  truths:
    - "A graph built in the IterCluster format (IterCluster with MemLoad inputs, Map/Reduce whose Input lists carry IterClusterProject operands, Reduce's KeyedAttribute('init', ...), and a trailing As node) is lowered by the registered Simplification passes into the Iter format: one Iter per As-listed dim bound by TensorDimOf(tensor, axis), InductionVariable nodes indexing the rewired MemLoad axes, and Map/Reduce Input slots pointing at the loads directly"
    - "After simplify-lower no live IterClusterProject or IterCluster-format cluster remains; after simplify-inline no paramRefNode inside a Map lambda body resolves to a parameter still declared in that Map's Lambda record; ir.Verify reports zero violations after each sub-pass"
    - "Every reference into Map/Reduce rides the Input list: the as-order is a trailing KindAs node found by scanning inputs (ir.AsOf), Reduce's init is a KeyedAttribute at input position 1 (ir.ReduceInitOf unwraps it), and the verifier enforces As-last/uniqueness and the attribute position fail-closed"
    - "The legacy builder path compiles with zero source changes (NewMap/NewReduce/NewIterCluster signatures keep their parameter shapes; NewIterCluster's inputs become variadic), builds the same kernel, and pkg goldens are regenerated deliberately with only numbering/new-node deltas — no semantic reordering"
    - "simplify-lower (102), simplify-inline (103, requires lower), simplify-verify (104, requires inline) self-register from func init() in internal/opt/simplify and appear in opt/all BuildPipeline in that order between dce-verify (101) and gvn (110)"
    - "go build ./..., go vet ./..., go test ./... all green with no assertion weakened"
  artifacts:
    - path: "internal/ir/node_iterfmt.go"
      provides: "The five new node types: iterClusterProjectNode (ops=[cluster], idx payload), keyedAttributeNode (ops=[value], key payload), asNode (ops=dims), inductionVariableNode (ops=[iter]), tensorDimOfNode (ops=[tensor, axis-const]) with constructors NewIterClusterProject/NewKeyedAttribute/NewAs/NewInductionVariable/NewTensorDimOf"
      exports: ["NewIterClusterProject", "NewKeyedAttribute", "NewAs", "NewInductionVariable", "NewTensorDimOf"]
      min_lines: 120
    - path: "internal/ir/node_lambda.go"
      provides: "Restructured mapNode (lam payload only — as field removed, As rides the operand list) and reduceNode (ops = [piped, KeyedAttribute, As?]) with unchanged constructor signatures, plus AsOf and ReduceInitOf accessors that scan the Input list"
      exports: ["NewMap", "NewReduce", "AsOf", "ReduceInitOf"]
    - path: "internal/ir/node_loop.go"
      provides: "iterClusterNode with variadic input operands (was single body operand); dims payload unchanged"
      exports: ["NewIterCluster"]
    - path: "internal/ir/mutate.go"
      provides: "(*MutContext).ClearLambdaParams — the sanctioned choke point that kills a stage's dead lambda-param payload records (the params live in the Lambda record, not operand slots, so RemoveUse cannot reach them)"
      exports: ["ClearLambdaParams"]
    - path: "internal/opt/simplify/lower.go"
      provides: "Pass simplify-lower: As-driven Iter/TensorDimOf/InductionVariable creation, MemLoad axis rewiring, project re-homing, project+cluster erasure — all through MutContext"
    - path: "internal/opt/simplify/inline.go"
      provides: "Pass simplify-inline: positional lambda-param binding to Input-list value bindings (attributes unwrap to their value), RAUW + Erase of param refs, dead-param stripping via ClearLambdaParams"
    - path: "internal/opt/simplify/verify.go"
      provides: "Pass simplify-verify mirroring dce-verify: no live projects, As/attribute placement invariants, Map bodies fully inlined"
  key_links:
    - from: "internal/ir/node.go allocNode"
      to: "internal/ir/node_iterfmt.go"
      via: "five new Kind cases dispatching to the new structs (the one place a kind joins the concrete suite)"
      pattern: "KindIterClusterProject|KindKeyedAttribute|KindAs|KindInductionVariable|KindTensorDimOf"
    - from: "internal/opt/simplify"
      to: "internal/opt registry"
      via: "func init() { opt.Register(opt.PassSpec{...}) } in each of lower.go/inline.go/verify.go"
      pattern: "opt\\.Register\\(opt\\.PassSpec"
    - from: "internal/opt/all/all.go"
      to: "internal/opt/simplify"
      via: "blank import _ \"tcacomp/internal/opt/simplify\""
      pattern: "_ \"tcacomp/internal/opt/simplify\""
    - from: "internal/ir/print.go stageText/pipeInput"
      to: "AsOf/ReduceInitOf"
      via: "reduce renders init from ReduceInitOf and piped from operand 0; map renders its as-clause from AsOf's operands; pipeInput reduce case reads operand 0"
      pattern: "AsOf|ReduceInitOf"
    - from: "internal/opt/simplify/lower.go"
      to: "internal/ir MutContext"
      via: "SetDef for axis/slot rewiring and Erase for project/cluster removal — no direct field writes"
      pattern: "\\.SetDef\\(|\\.Erase\\("
---

<objective>
Implement doc/optana/lower_itercluster.md: lower the IterCluster format to the Iter format.

Purpose: The frontend will emit map/reduce stages in the IterCluster format (a cluster of MemLoads bridged into Map/Reduce through IterClusterProject nodes, attributes and as-orders riding the Input list). Simplification — run almost immediately in the pipeline — rewrites that graph into the data-dependency Iter format (Iter + TensorDimOf + InductionVariable) and inlines lambda parameter references, so every later analysis/optimization pass sees plain SSA data flow with no hidden references.

Output: five new IR node kinds (IterClusterProject, KeyedAttribute, As, InductionVariable, TensorDimOf), restructured Map/Reduce/IterCluster nodes whose incoming references all ride the Input list, a new internal/opt/simplify package with two registered sub-passes plus a verify pass, and a green full test suite with deliberately regenerated goldens.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@doc/optana/lower_itercluster.md
@doc/optana/induction_variable.md

@internal/ir/node.go
@internal/ir/kind.go
@internal/ir/catalog.go
@internal/ir/node_loop.go
@internal/ir/node_lambda.go
@internal/ir/node_memory.go
@internal/ir/node_dimvar.go
@internal/ir/mutate.go
@internal/ir/verify.go
@internal/ir/print.go
@internal/ir/value.go
@internal/opt/registry.go
@internal/opt/dce/dce.go
@internal/opt/dce/verify.go
@internal/analysis/sideeffect/sideeffect.go
@pkg/chain.go
</context>

<interfaces>
<!-- Current signatures the executor builds against. Verified against the codebase 2026-10-19. -->

From internal/ir/node_lambda.go:
```go
type Lambda struct { /* params []SSAUse; Body Node */ }
func NewLambda(params []Node, body Node) Lambda
func (l Lambda) Params() []SSAUse
func (g *Graph) NewMap(origin string, ops []Node, lam Lambda, as []Node) Node   // KEEP this signature
func (g *Graph) NewReduce(origin string, init Node, piped Node, combiner Lambda) Node // KEEP this signature
```

From internal/ir/node_loop.go:
```go
func (g *Graph) NewIter(origin string, dim, bound Node, init, step Node) Node // ops=[init,step]; dim/bound are payload uses
func (g *Graph) NewIterCluster(origin string, dims []Node, body Node) Node    // BECOMES (origin, dims []Node, inputs ...Node)
```

From internal/ir/mutate.go (the ONLY sanctioned mutation paths):
```go
func (ctx *MutContext) SetDef(u SSAUse, newVal Node) error
func (ctx *MutContext) ReplaceAllUsesWith(old, newVal Node) error
func (ctx *MutContext) Erase(n Node) error // refuses while live uses remain — propagate the error
```

From internal/ir/ssa.go + walk.go:
```go
func UsesOf(def Node) []SSAUse // u.rec.user names the using node
func OperandsOf(n Node) []Node
func Walk(g *Graph, v Visitor)
func (g *Graph) AllocatedNodes() []Node // allocation order = deterministic iteration order
```

From internal/opt (pass contract — mirror internal/opt/dce exactly):
```go
type Pass interface { ID() string; Requires() []string; Produces() Stage; Preserved() []string; Run(*Context, *ir.Graph) error }
func Register(spec PassSpec) // panics on empty ID / nil Factory / duplicate / ID mismatch
// registry priority anchors: dce=100, dce-verify=101, gvn=110, gvn-verify=111, schedule=900
```
</interfaces>

<reconciliation>
## Doc-vs-code reconciliation decisions (RECORDED — implement exactly these)

The doc is a design sketch written ahead of implementation; the current IR differs. Where they conflict, these are the binding resolutions (constraint from the task: the plan decides and records):

**D1 — As rides the Input list (user amendment, HARD requirement).**
New node `KindAs` (struct `asNode`, ops = the dim uses, variadic). Map/Reduce carry it as the LAST Input-list element. `mapNode.as []SSAUse` payload field is REMOVED; the accessor `ir.AsOf(n Node) Node` scans `OperandsOf(n)` backwards for the KindAs node — no hidden pointer. Reduce Input layout: `[piped, KeyedAttribute("init", init), As?]`; Map Input layout: `[value operands..., As?]`. The trailing As is optional at the IR level (zero-dim/legacy graphs), but unique-and-last when present (verifier-enforced). Iter creation order and dim extraction in simplify-lower come from the As node's operands, per the amendment.

**D2 — KeyedAttribute for Reduce's init (doc).**
`KindKeyedAttribute` (ops = [value]; `key string` debug payload, "init" for Reduce). `NewReduce` KEEPS its signature `(origin, init, piped, combiner)` but wires ops as `[piped, NewKeyedAttribute(origin, "init", init)]` internally — zero call-site churn in pkg/chain.go and tests. New accessor `ir.ReduceInitOf(n Node) Node` unwraps the attribute's value operand. Verifier: reduce operand 1 must be a keyedAttributeNode. The printer unwraps it so the rendered text `reduce(init=..., <piped>, lambda(...))` keeps today's spelling.

**D3 — IterCluster gets an Input list (doc).**
`iterClusterNode` ops change from `[body]` to variadic `[inputs...]`; the dims payload stays. `NewIterCluster(origin string, dims []Node, inputs ...Node)` — the two pkg/chain.go call sites and all test call sites compile unchanged (the legacy single-map wrapper is `inputs=mapNode`). Catalog arity 1 → -1. The doc's cluster-level "EffectList(lhs, rhs)" is DROPPED as redundant: per-tensor ordering already rides the individual MemLoads' ImpactBy chains (effect.go's chain-to-head-node model); a cluster-level effect list pointing at params contradicts that model. Recorded.

**D4 — IterClusterProject (doc).**
`KindIterClusterProject` (HIR, FamilyAbstract; ops = [cluster]; `idx int` — structural, participates in GVN content hash/equal). Verifier: operand 0 must be an iterClusterNode. In the IterCluster format, Map Input slot i holds `IterClusterProject(cluster, i)`; positional correspondence Input[i] ↔ lambda param i is the binding contract (verifier-checked for Map: value-operand count == len(lam.params)).

**D5 — Iter format reuses the existing iterNode (doc's Iter(dim, TensorDimOf(...))).**
No iterNode struct change. The lowered Iter is `NewIter(pass, dimParam, tensorDimOfNode, const0, const1)`: dim payload = RoleDim InputParam; bound payload WIDENED to accept any value node (today it is only documented as a dim param; nothing enforces it); init/step = TF32 Const 0/1 matching the established DimVar test convention (TypeInference is future work per the doc). Printer: when the bound's def is not an InputParam, render `@name/<value>` via r.value instead of the `?` fallback paramNameOf produces today.

**D6 — InductionVariable and TensorDimOf are LIR-tier pure value nodes.**
`KindInductionVariable` (ops = [iter]; verifier: operand 0 must be an iterNode) — the u32 current-iteration value; participates in normal expressions. `KindTensorDimOf` (ops = [tensor, axisConst]; constructor `NewTensorDimOf(origin, tensor, axis int)` creates the TU32 axis const internally; verifier: operand 0 must be a RoleTensor InputParam, operand 1 a constNode) — the shape upper bound. Both FamilyAbstract → pure under EffectOf and the accumulated side-effect analysis (no analysis change needed; they follow the DimVar precedent). GVN-eligible; two InductionVariables over the same Iter dedupe by operand identity.

**D7 — Enum placement and KindVersion.**
Insert KindIterClusterProject, KindKeyedAttribute, KindAs after KindReturn (end of the HIR section); append KindTensorDimOf, KindInductionVariable before KindEnd (LIR section). The HIR insertions renumber the LIR constants once, deliberately: bump KindVersion to 4 and extend the kind.go header comment stating why (catalog_test's "HIRKinds == kinds < KindForLoop" invariant still holds). Add catalog rows, String() spellings (`iterclusterproject`, `keyedattribute`, `as`, `tensordimof`, `inductionvariable`), and Family classification.

**D8 — Lambda record keeps its params declaration list.**
The doc's "Lambda struct MUST NOT contain params[]SSAUse" note targets hidden BINDINGS. The record's params are the parameter DECLARATIONS (ordered, Walk-anchored); the binding is the Input-list correspondence (D4), now verifier-enforced. Removing declarations entirely would orphan unused params and lose positional order. Recorded as the resolution; revisit only if a later pass needs it.

**D9 — Inline binding rule (uniform, positional, attribute-unwrapping).**
For both Map and Reduce: lambda param i binds to the i-th Input-list VALUE binding, where a KeyedAttribute element binds its wrapped value operand and the trailing As node binds nothing. So Reduce `[piped, attr(init)]` binds `[acc, now]` as acc↔piped, now↔init-value; Map `[ops..., As]` binds params↔ops. This is the doc's literal positional rule, made uniform. NOTE (recorded tension): the builder's (acc, now) combiner convention means the eventual builder migration to this format should emit combiner params in input order — a one-line change when that lands; the builder path never runs Simplification today (pkg builds run opt.NewManager(nil)), so no current graph is affected.

**D10 — Simplification is one sub-module, three registered passes.**
`internal/opt/simplify`: `simplify-lower` (Priority 102), `simplify-inline` (103, Requires ["simplify-lower"]), `simplify-verify` (104, Requires ["simplify-inline"]), each self-registered from func init(), mirroring dce/dce-verify structure (level gates: nil graph and materialized/CFG-carrying graph are errors evaluated before any mutation; Produces StageConstructed; Preserved nil). STATE.md's pipeline order (simplify first among transformations) is honored: they slot after dce cleanup, before gvn — so GVN dedupes the lowered TensorDimOf/Iter/IV nodes.

**D11 — Legacy graphs pass through; goldens regenerate deliberately.**
Legacy builder output (cluster wrapping a Map, reduce without projects) is skipped by simplify-lower (no IterClusterProject anywhere; non-MemLoad cluster inputs untouched) — the pass is a documented no-op there. BUT the Input-list restructure changes the canonical print's SSA numbering: StmtNodes descends operands, so the new As and KeyedAttribute operand nodes take numbers and shift subsequent %N references. pkg/testdata/matmul_scale.ir MUST be regenerated. The executor verifies the regenerated diff shows ONLY numbering shifts plus the new nodes' presence — same chain order, same stage spellings, same as-clauses.

**D12 — pipeInput reduce case flips to operand 0.**
print.go's pipeInput reads `KindReduce → opAt(n, 1)` today (init-first layout). Under [piped, attr, as?] it must read operand 0. Map stays operand 0 (values-first; add a defensive nil when operand 0 is an As/KeyedAttribute). IterCluster stays operand 0.

**D13 — GVN identity updates.**
value.go contentHash/contentEqual gain cases: iterClusterProjectNode mixes/compares `idx`; keyedAttributeNode mixes/compares `key`; the other three new kinds have no content fields (operands carry identity). mapNode loses the as payload enumeration (As rides operand IDs) — two structurally identical maps no longer value-equal (distinct As nodes); this is accepted and recorded (stage-level GVN dedup becomes As-node-aware future work; gvn-verify's re-derivation is unaffected). Existing gvn tests that assert map dedup would catch any unintended merge — none assert map-map merging today; fingerprint/node-count assertions gain the new As/attr nodes and are updated mechanically.
</reconciliation>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: IR layer — five Input-list node kinds, Map/Reduce/IterCluster restructure, full integration</name>
  <files>internal/ir/kind.go, internal/ir/catalog.go, internal/ir/node_iterfmt.go, internal/ir/node_loop.go, internal/ir/node_lambda.go, internal/ir/mutate.go, internal/ir/verify.go, internal/ir/print.go, internal/ir/value.go, internal/ir/node_iterfmt_test.go, internal/ir/node.go, pkg/testdata/matmul_scale.ir</files>
  <behavior>
    - Test: constructing each new node via its constructor and calling ir.Verify on a graph rooting it through NewReturn reports ZERO violations (shapes wired correctly).
    - Test: a boundedDimVar-style fail-closed battery (mirror node_dimvar_test.go TestDimvarVerifyShapes) — IterClusterProject whose operand 0 is a const, InductionVariable whose operand 0 is a const, TensorDimOf whose operand 0 is a scalar param or whose operand 1 is a binop, As with a non-dim operand, Reduce whose operand 1 is a bare const (not KeyedAttribute), Map with two As nodes or an As in non-last position — each reports its specific verifier message via assertViolation.
    - Test: NewReduce keeps producing `reduce(init=<v>, <piped>, lambda(...): ...)` text identical to the pre-change spelling after Materialize+Print of a small chain (unwrap path); NewMap's ` as (@m, @k)` clause renders identically from the As node.
    - Test: iterNode with a TensorDimOf bound renders `@a/<tensordimof rendering>` (no `?`).
    - Test: ValueHash/ValueEqual — two IterClusterProjects over the same cluster with idx 0 vs 1 hash/compare UNEQUAL (pure graph, sideeffect analysis attached); two KeyedAttributes with keys "init" vs "other" unequal; two InductionVariables over the same iter equal (via a GVN-style bucket check or direct nodeValueEqual through the exported Value interface with a stub SideEffectAnalysis).
    - Test: pkg oracle path still builds matmul_scale and the regenerated testdata matches Print byte-for-byte (existing golden test machinery), with the regenerated file diff showing only numbering/new-node deltas.
  </behavior>
  <action>
    1. **kind.go** (D7): add KindIterClusterProject, KindKeyedAttribute, KindAs after KindReturn (HIR section); KindTensorDimOf, KindInductionVariable before KindEnd (LIR section). Bump KindVersion to 4 with a header note: "v4 = Input-list restructure (quick-261019-ktm): HIR additions renumbered the LIR constants once, deliberately." Add String() cases (`iterclusterproject`, `keyedattribute`, `as`, `tensordimof`, `inductionvariable`) and Family() classification (all FamilyAbstract except keep TensorDimOf/InductionVariable FamilyAbstract too — pure value nodes, DimVar precedent; IterClusterProject/KeyedAttribute/As FamilyAbstract).
    2. **catalog.go**: five rows with correct tier/family/arity — IterClusterProject (HIR, Abstract, 1), KeyedAttribute (HIR, Abstract, 1), As (HIR, Abstract, -1), TensorDimOf (LIR, Abstract, 2), InductionVariable (LIR, Abstract, 1); flip KindIterCluster arity 1 → -1; update KindReduce arity 2 → -1 (optional trailing As); extend row docs.
    3. **node_iterfmt.go (NEW, ~200-260 lines)**: the five structs + constructors, following node_dimvar.go's commenting discipline:
       - `iterClusterProjectNode{Base, operandList, idx int}` — `NewIterClusterProject(origin string, cluster Node, idx int)`; cluster wired as operand 0.
       - `keyedAttributeNode{Base, operandList, key string}` — `NewKeyedAttribute(origin, key string, value Node)`; value wired as operand 0.
       - `asNode{Base, operandList}` — `NewAs(origin string, dims ...Node)`; dims wired as operands in order.
       - `tensorDimOfNode{Base, operandList}` — `NewTensorDimOf(origin string, tensor Node, axis int)`: allocates with 2 slots, wires tensor at 0, creates and wires a TU32 Const (InternScalar(TU32)) at 1 via NewConst.
       - `inductionVariableNode{Base, operandList}` — `NewInductionVariable(origin string, iter Node)`.
       Export read helpers here or in node_lambda.go: `AsOf(n Node) Node` (scan OperandsOf backwards for KindAs), `KeyedAttributeOf`/`ReduceInitOf(n Node) Node` (Reduce: operand 1 must be keyedAttribute; return its operand-0 def, nil otherwise).
    4. **node.go allocNode**: five new cases dispatching to the structs (the single place kinds join the suite).
    5. **node_loop.go**: `NewIterCluster(origin string, dims []Node, inputs ...Node)` — wire each input as an operand slot (NewNode with the variadic operands; iterClusterNode type assertion retained). Keep dims payload. Update the struct doc comment.
    6. **node_lambda.go**: 
       - mapNode: DELETE the `as []SSAUse` field and its forEachPayloadUse enumeration (lam params remain the only payload). NewMap keeps signature; when len(as) > 0, append `g.NewAs(origin, as...)` to the wired operands.
       - reduceNode: NewReduce keeps signature `(origin, init, piped, combiner)`; internally ops = [piped, g.NewKeyedAttribute(origin, "init", init)] (attr node allocated BEFORE wiring so allocation order is deterministic: attr first, then reduce). Add optional trailing As support later only if needed — D1 makes Reduce's As optional; do NOT add an as parameter now (no caller supplies one; extending later is additive).
       - fixupLambdaUsers flow unchanged.
    7. **mutate.go**: add `func (ctx *MutContext) ClearLambdaParams(n Node, keep func(Node) bool) error` — iterates n's lambda records (lambdaBearer), kills each param record whose def fails `keep` (removeRef from the def's var, killed = true), rebuilds the record's params slice with the kept uses only, stamps n. This is the choke point for Task 2's inline (lambda params live in the Lambda record, not operand slots — RemoveUse cannot reach them).
    8. **verify.go checkNodeShape** additions (fail-closed, mirroring existing checks):
       - *iterClusterProjectNode: operand 0 must be an *iterClusterNode ("iterclusterproject operand 0 must be an itercluster").
       - *inductionVariableNode: operand 0 must be an *iterNode ("inductionvariable operand 0 must be an iter").
       - *tensorDimOfNode: operand 0 must pass checkTensorParam; operand 1 must be a *constNode ("tensordimof operand 1 must be a const axis index").
       - *asNode: every operand must be a RoleDim InputParam ("as operand i must be a dim InputParam").
       - *reduceNode: operand 1 must be a *keyedAttributeNode ("reduce operand 1 must be a KeyedAttribute (init)").
       - *mapNode / *reduceNode placement: at most one KindAs operand and, when present, it must be the LAST operand ("as must be the last input"); plus mapNode's value-operand count (excluding a trailing As) must equal len(lam.params) ("map input count must equal lambda parameter count").
    9. **print.go**:
       - stageText *reduceNode: piped = opAt(t, 0); init value = ReduceInitOf; text spelling unchanged.
       - stageText *iterClusterNode: render inputs by joining each input's stageText with ", " after the `iter(@dims) -> ` prefix (single input keeps today's bytes).
       - asClause consumers: render the map's as-clause from AsOf(map)'s OperandsOf (same ` as (@m, @k)` text).
       - value(): cases for *tensorDimOfNode → `tensordimof(<tensorName>, <axis constText>)`; *inductionVariableNode → `inductionvariable(<iter value-render>)`; *keyedAttributeNode → `keyedattribute(<key>=<value>)`; *iterClusterProjectNode → default %N numbering (transient pre-lowering node — do not special-case). iterNode bound: when boundUse.Def() is not an *inputParamNode, append "/" + r.value(boundDef) instead of paramNameOf's "?".
       - pipeInput: KindReduce → opAt(n, 0) (D12); KindMap stays opAt(n, 0) but return nil when that operand is an As/KeyedAttribute (defensive); KindIterCluster stays opAt(n, 0).
    10. **value.go**: contentHash/contentEqual cases for idx (mixUint64) and key (mixString), mirroring callNode.name's treatment.
    11. **Tests** (node_iterfmt_test.go + targeted additions to print_test/value_test): implement the behavior block. Mirror node_dimvar_test.go's ivGraph/assertViolation helpers.
    12. **Full-suite pass**: `go build ./... && go vet ./... && go test ./...`. Mechanically update the expectations that legitimately change: node counts in gvn/dce/walk/effect/cfg/dot/sideeffect tests (+1 As node per NewMap with non-empty as, +1 attr node per NewReduce), walk-order assertions on map children (As now an operand child), and REGENERATE pkg/testdata/matmul_scale.ir via the existing golden-generation path. Rule: never weaken an assertion — update it to the new canonical structure. Inspect the golden diff: numbering shifts and new nodes only; any stage reordering is a bug in your pipeInput/stageText changes, not an expectation update.
  </action>
  <verify>
    <automated>cd /home/ps/Workspace/cuda/project/tcacomp && go build ./... && go vet ./... && go test ./...</automated>
  </verify>
  <done>All five kinds constructible and verifier-shaped fail-closed; Map/Reduce/IterCluster restructured with every incoming reference in the Input list; AsOf/ReduceInitOf exported; print keeps stage spellings with regenerated goldens; value identity covers idx/key; entire test suite green with zero weakened assertions.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Simplification pass — lower IterCluster to Iter, inline lambdas, verify</name>
  <files>internal/opt/simplify/lower.go, internal/opt/simplify/inline.go, internal/opt/simplify/verify.go, internal/opt/simplify/simplify_test.go, internal/opt/all/all.go</files>
  <behavior>
    - Test (lower, doc example): build the doc's myfoo shape — dims a/b/c (RoleDim params), tensors lhs/rhs (rank-2 RoleTensor params, f32), cluster = NewIterCluster(dims=[a,b,c], inputs=[MemLoad(lhs,[a,b]), MemLoad(rhs,[a,c])]), map = NewMap(ops=[Project(cluster,0), Project(cluster,1)], lambda(l,r)=BinOp(l*r), as=[a,b,c]) — run simplify-lower, then assert: exactly 3 iters exist, keyed by dim in AS ORDER (a←TensorDimOf(lhs,0), b←TensorDimOf(lhs,1), c←TensorDimOf(rhs,1)); each MemLoad axis slot now resolves to an InductionVariable whose iter's dim payload is the original axis dim; the map's operand slots 0/1 resolve to the two MemLoads; no live IterClusterProject or IterCluster remains (cluster erased via Tombstones); ir.Verify(g) is empty.
    - Test (lower, shared dim): a dim bound by two loads (a in lhs[0] and rhs[0]) yields ONE iter bound by the FIRST pairing (lhs, 0) — deterministic first-wins, pinned.
    - Test (lower, legacy no-op): a builder-shaped graph (cluster wrapping a Map, no projects) passes through with AllocatedNodes alive-set unchanged and Verify clean.
    - Test (lower, fail-closed): a cluster whose MemLoad axis dim is not covered by any projecting map's As returns an error naming the dim; a nil graph and a materialized graph return errors without mutation.
    - Test (inline, map): after lower, run simplify-inline on the doc example — every paramRefNode that referenced lambda params l/r is gone (erased); the lambda body's binop operands resolve directly to the two MemLoads; the map's Lambda record params list is empty; the lambda InputParams are erased; Verify clean.
    - Test (inline, reduce): a reduce with ops [piped=mapNode, attr(init=const)] and combiner lambda [acc, now] whose body references now: after inline, now's refs point at the attr's VALUE per D9's positional rule; params still referenced by the body stay declared (ClearLambdaParams keeps live ones).
    - Test (inline, idempotence): running lower then inline a second time changes nothing (compare alive-node fingerprint before/after).
    - Test (verify pass): a hand-corrupted post-inline graph (re-add a Project via NewNode wiring, or leave a Map body ref pointing at a still-declared param) makes simplify-verify return an error; a clean lowered graph passes.
    - Test (pipeline): opt/all BuildPipeline contains dce, dce-verify, simplify-lower, simplify-inline, simplify-verify, gvn in exactly that priority order; Manager.Run over the doc-example graph with the full default pipeline succeeds end-to-end.
  </behavior>
  <action>
    1. **lower.go** — package simplify, `Pass{}` with ID "simplify-lower", Priority 102, no Requires, Produces StageConstructed, Preserved nil; init() self-registration mirroring dce.go. Run:
       a. Level gate (nil / materialized / CFG → error, before any mutation).
       b. Snapshot g.AllocatedNodes(); collect clusters (KindIterCluster) and projects (KindIterClusterProject) in allocation order; index projects by their operand-0 cluster.
       c. For each cluster WITH at least one projecting map: derive the ordered dim list from the projecting maps' AsOf operands (map allocation order, As order, dedup by node identity). For each dim, scan the cluster's MemLoad inputs in order, each axis slot in order, for the first axis whose def is that dim → create (interned by dim node identity, one table for the whole Run) `NewTensorDimOf(pass, tensor, axisIdx)`, `NewIter(pass, dim, tdim, const0, const1)` (TF32 consts interned once per Run), `NewInductionVariable(pass, iter)`. A dim covered by no axis → fail-closed error naming the dim. Clusters with ZERO projecting maps: MemLoad inputs → error (live uncovered iteration space); non-MemLoad inputs only (legacy wrapper) → skip the cluster entirely.
       d. Second sweep: for each cluster processed in (c), for each MemLoad input, for each axis slot whose def has an interned iter: `mctx.SetDef(axisUse, ivOfDim)`.
       e. For each map/reduce operand slot whose def is a Project P: `mctx.SetDef(slotUse, clusterInput(P.idx))` (cluster input = the cluster's operand slot P.idx def; bounds-check and error fail-closed).
       f. Erase each Project (now use-free), then each processed cluster — PROPAGATE Erase errors (a hybrid graph still piping a cluster surfaces the refusal; never force-erase). All mutations through one MutContext{Pass: "simplify-lower"}.
    2. **inline.go** — ID "simplify-inline", Priority 103, Requires ["simplify-lower"]. Run: level gate; for each map/reduce in allocation order: compute valueBindings per D9 (OperandsOf in order; stop at a trailing KindAs; a KindedAttribute operand contributes its operand-0 def; other operands contribute themselves). For i, paramUse in the record's Params(): if i < len(valueBindings): for each use in ir.UsesOf(paramUse.Def()) whose rec.user is a paramRefNode R: `mctx.ReplaceAllUsesWith(R, valueBindings[i])` then `mctx.Erase(R)` (Erase's live-use refusal is the safety net — propagate errors). Then `mctx.ClearLambdaParams(stage, keep)` where keep = the param still has live uses (acc-style survivors stay declared). Finally Erase lambda InputParams that are now use-free (skip graph-owned params — RoleLambda params are never in g.params). Deterministic throughout; one MutContext.
    3. **verify.go** — ID "simplify-verify", Priority 104, Requires ["simplify-inline"], mutating nothing: (a) no live IterClusterProject in AllocatedNodes; (b) every live map's lambda body contains no paramRefNode resolving to a param still declared in that map's record (walk each lambda Body subtree via a local DFS over OperandsOf + payloadBearer + lambdaBearer, mirroring hasEffectNode's shape); (c) As/attribute placement invariants from Task 1 hold on live stages. Error messages name node ID and kind (dce-verify style).
    4. **all.go**: add blank import `_ "tcacomp/internal/opt/simplify"`; extend all_test's registered-pass assertions if it enumerates IDs.
    5. **simplify_test.go**: implement the behavior block. Graph builders construct the IterCluster format DIRECTLY via the ir constructors (the established pass-test pattern — dce_test/gvn_test never go through pkg). Helper: `docExampleGraph(t)` returning the doc's myfoo shape plus node handles for assertions. Use g.Tombstones()/AllocatedNodes() for liveness assertions and a fresh sideeffect.Analyze where a SideEffectAnalysis is needed.
    6. Full suite: `go build ./... && go vet ./... && go test ./...` green.
  </action>
  <verify>
    <automated>cd /home/ps/Workspace/cuda/project/tcacomp && go test ./internal/opt/simplify/ ./internal/opt/all/ && go test ./...</automated>
  </verify>
  <done>internal/opt/simplify registered as three passes at 102/103/104 and composed by opt/all; the doc's example graph lowers to Iter format with interned per-dim Iters, TensorDimOf bounds, InductionVariable-indexed loads, project/cluster erasure, and inlined lambda bodies; ir.Verify stays clean after every sub-pass; legacy graphs pass through; full suite green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| test/hand-built IR graphs → Simplification pass | The only inputs the new pass sees today (the pkg builder path runs an empty pipeline). Malformed graphs are untrusted input. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-ktm-01 | Tampering | simplify lower/inline mutations | mitigate | All rewiring through MutContext choke points (SetDef/RAUW/Erase/ClearLambdaParams); Erase live-use refusals propagate as errors — no force-erasure, no direct field writes |
| T-ktm-02 | Tampering | malformed Input-list shapes (As mid-list, duplicate As, reduce without attr, project at wrong cluster slot) | mitigate | Verifier shape checks added in Task 1 are fail-closed and run via VerifyEach after every pass; the pass itself bounds-checks project indices and errors naming the node |
| T-ktm-03 | Denial of Service | pass run on nil/materialized graphs | mitigate | Level gates evaluated before any mutation (dce's refusal pattern) |
| T-ktm-04 | Repudiation | post-lowering corruption silently reaching codegen | mitigate | simplify-verify (104) re-asserts no-live-projects, full Map inline, and placement invariants; ir.Verify chain checks cover both SSA directions |
| T-ktm-05 | Information Disclosure | diagnostics | accept | Error messages carry node IDs/kinds only — no source text or names beyond debug payload already in-repo |
</threat_model>

<verification>
- `go build ./... && go vet ./... && go test ./...` green after each task.
- The doc's worked example (two rank-2 tensors, dims a/b/c, cluster of two MemLoads, map with two projects) lowers and inlines end-to-end through the registered default pipeline — asserted by test, not by inspection.
- pkg goldens: regenerated file diff reviewed — numbering/new-node deltas only; stage spellings and chain order unchanged.
- No assertion weakened anywhere in the suite (expectation updates must track the new canonical structure, never delete checks).
</verification>

<success_criteria>
- All must_haves truths hold and are pinned by tests in internal/ir and internal/opt/simplify.
- The five reconciliation-critical accessors/nodes exist and are used by printer, verifier, and pass: NewAs+AsOf, NewKeyedAttribute+ReduceInitOf, NewIterClusterProject, NewInductionVariable, NewTensorDimOf.
- internal/opt/all BuildPipeline order includes simplify-lower(102) → simplify-inline(103) → simplify-verify(104) between dce-verify and gvn.
- Full test suite green; goldens regenerated deliberately with a reviewed diff.
</success_criteria>

<output>
After completion, create `.planning/quick/261019-ktm-implement-itercluster-lowering-feature-f/261019-ktm-SUMMARY.md`
</output>
