---
phase: quick-261021-jay
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/ir/kind.go
  - internal/ir/catalog.go
  - internal/ir/node_intrinsic.go
  - internal/ir/node.go
  - internal/ir/verify.go
  - internal/ir/value.go
  - internal/ir/builtins.go
  - internal/ir/node_loop.go
  - internal/ir/print.go
  - internal/ir/builtins_test.go
  - doc/ir.md
  - internal/opt/typeinfer/rules.go
  - internal/opt/registry.go
  - internal/opt/tiling/lower.go
  - internal/opt/tiling/lower_rewrite.go
  - internal/opt/tiling/lower_cache.go
  - internal/opt/tiling/lower_test.go
autonomous: true
requirements: [loop-tile-lower-pass]
must_haves:
  truths:
    - "Every decided LNT is rewritten: no live KindIter or KindInductionVariable of a rewritten LNT survives; every generated loop is a KeyedAttribute-annotated KindForLoop"
    - "DoSoC bindings are derived per LNT with CTA and ThreadMap counters restarting per LNT; sequential iters get no dimension"
    - "Every promoted MemLoad is erased with its uses RAUW'd to MemLoadSMem/FragLoad over a MemNewSMem/FragNew buffer declared at the fill point"
    - "Every materialized tile bound is a umin call representable and typable in the IR (u32 builtin registered, typeinfer rule passes)"
    - "After the pass ir.Verify reports zero violations and a typeinfer re-run succeeds on the rewritten graph"
    - "The pass fails closed when the tiling-record is absent"
  artifacts:
    - path: "internal/ir/node_intrinsic.go"
      provides: "IntrinsicGridMap/IntrinsicThreadMap node structs, constructors, payload read face"
    - path: "internal/opt/tiling/lower.go"
      provides: "tile-lower pass at priority 116: registration, Requires tiling, record fetch fail-closed, per-LNT skeleton, PHASE 5 epilogue"
    - path: "internal/opt/tiling/lower_rewrite.go"
      provides: "DoSoC numbering + loop rewrite (between/inside forms, materialized bounds, IV RAUW, chain re-pointing)"
    - path: "internal/opt/tiling/lower_cache.go"
      provides: "site fact recomputation, fill point, buffer+fill emission, access rewrite, budget fallback"
    - path: "internal/opt/tiling/lower_test.go"
      provides: "pinned tests for numbering, rewrite, cache insertion, matmul end-to-end, fail-closed gates"
  key_links:
    - from: "internal/opt/tiling/lower.go"
      to: "tiling.Record (record.go)"
      via: "ctx.Cache.Get(tiling RecordName) cast to *Record, fail closed"
      pattern: "Cache\\.Get\\(RecordName\\)"
    - from: "internal/opt/tiling/lower_rewrite.go"
      to: "internal/ir/mutate.go"
      via: "every graph edit rides MutContext (SetDef/AddUse/RemoveUse/RemoveImpactBy/ReplaceAllUsesWith/SetType/Erase)"
      pattern: "MutContext\\{Pass: \"tile-lower\"\\}"
    - from: "internal/opt/tiling/lower_cache.go"
      to: "internal/opt/tiling/overlap.go"
      via: "same-package reuse of the decide pass's deterministic INS chase (findINS/analyzeOverlap/treeItersOf/levelsInnerToOuter)"
      pattern: "analyzeOverlap\\("
    - from: "generated loop bounds"
      to: "internal/ir/builtins.go"
      via: "umin calls for materialized tile_max = umin(T, extent - offset)"
      pattern: "NewCall\\(.*\"umin\""
---

<objective>
Implement the loop-tile-lower optimization pass — the lowering sequel to the loop-tiling decide pass — per the normative, self-contained spec doc/optana/loop_tile_lower.md (§6bis mechanical procedure, PHASE 0-5; §6bis was completed by quick task 261021-j24 precisely so this pass can be implemented from the doc alone).

Purpose: convert the decide pass's side-table decisions into IR node surgery — rewritten tiled LIR loop nests (KindForLoop) carrying SoC/DoSoC/LoC annotations, IntrinsicGridMap/IntrinsicThreadMap queries, and SMEM/FRAG cache insertion — so the pipeline advances from "decided" to "lowered, ready for scheduling".

Output: two new LIR intrinsic node kinds with all six companions, u32 min/max builtins, the tile-lower pass at priority 116 in internal/opt/tiling (lower.go, lower_rewrite.go, lower_cache.go), pinned tests, full-repo gates green (build/vet/test).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@doc/optana/loop_tile_lower.md
@doc/optana/loop_tiling.md
@doc/opt.md

<interfaces>
<!-- Key contracts the executor needs. Extracted from the codebase — do not re-explore. -->

From internal/opt/tiling/record.go (SAME PACKAGE as the new pass — reuse, do not duplicate):
```go
const RecordName = "tiling-record"
type SoCType int // SoCSequential=0, SoCTA=1, SoCWARP=2
type CacheClass int // ClassNone=0, ClassSMEM=1, ClassFRAG=2
type IterDecision struct { SoC SoCType; LoC int; TileSize int } // LoC units: CTA dims -> CTAs; WARP dims -> WARPS (x32 = threads)
type SiteDecision struct { Promoted bool; Tie ir.Node; Class CacheClass } // Tie == nil => above outermost
type Record struct { Iters map[ir.Node]IterDecision; Sites map[uint64]SiteDecision } // Iters keyed by BIV node identity; Sites keyed by MemLoad BasePtr().ID()
func (r Record) IterDecisionOf(iter ir.Node) (IterDecision, bool) // resolves DataDeps(iter)[0] first
func (r Record) SiteDecisionOf(load ir.Node) (SiteDecision, bool)
const ( PortionCTA = 10; PortionWarp = 8; AutoWarpPortion = 4 )
type Params struct { MaxCTA, MaxThreadsPerCTA, WarpPortion, MaxCache, CacheDims, MaxCS, WindowK int; TileSizes []int; NominalExtent int }
func DefaultParams() Params // 100 CTAs, 512 threads, 64KB, CacheDims 2, WindowK 5
```

From internal/opt/tiling/overlap.go (package-private, same package — reuse):
```go
type SiteFacts struct { Load, Tensor, SetLevel, Tie ir.Node; Covered, INS, Levels []ir.Node; AxesIters [][]ir.Node }
type lntFacts struct { Tree *loopforest.Tree; Levels []ir.Node; CS []SiteFacts } // Levels inner->outer
func analyzeOverlap(g *ir.Graph, f loopforest.Forest, se sideeffect.SideEffects, p Params) []lntFacts
func findINS(load ir.Node, se sideeffect.SideEffects) []ir.Node
func treeItersOf(root *loopforest.Tree) []ir.Node  // DFS preorder = outer->inner along each branch
func levelsInnerToOuter(root *loopforest.Tree) []ir.Node
func siteBytesPerTile(sf SiteFacts, iters map[ir.Node]IterDecision) int  // search.go
func judgeSite(sf SiteFacts, iters map[ir.Node]IterDecision, p Params) (bool, CacheClass) // search.go
```
From internal/opt/tiling/verify.go: `func bivOf(iter ir.Node) (ir.Node, error)`.

From internal/analysis/loopforest: `Analyze(g) Forest`, `Forest.Roots() []*Tree`, `Tree{Iter ir.Node; Parent *Tree; Children []*Tree; Depth int}`, `Forest.NodeOf(iter)`.
From internal/analysis/sideeffect: `Analyze(g) SideEffects` (implements ir.SideEffectAnalysis).

From internal/ir/mutate.go — the ONLY sanctioned mutation surface (project rule):
```go
type MutContext struct { Pass string }
func (ctx *MutContext) SetDef(u SSAUse, newVal Node) error
func (ctx *MutContext) AddUse(user Node, val Node) SSAUse
func (ctx *MutContext) RemoveUse(u SSAUse)
func (ctx *MutContext) RemoveImpactBy(u SSAUse) error
func (ctx *MutContext) ReplaceAllUsesWith(old, newVal Node) error
func (ctx *MutContext) SetType(n Node, ty Type) error // idempotent on Equal types
func (ctx *MutContext) Erase(n Node) error // REFUSES while any live use remains; erase users first
```

internal/ir constructors (node_xxx.go files; construction is exempt from MutContext):
```go
g.NewForLoop(origin, iv, bound, body Node) Node        // node_loop.go — Task 1 makes attrs variadic
g.NewDimVar(origin, dim, init, step Node) Node         // operand 0 must be a dim InputParam (verifier)
g.NewBoundedDimVar(origin, iv, max Node) Node          // operand 0 must be a dimvar (verifier)
g.NewIter(origin, bdv, parent Node) Node
g.NewInductionVariable(origin, iter Node) Node
g.NewKeyedAttribute(origin, key string, value Node) Node // node_iterfmt.go; key is string payload, value is 1 operand
g.NewConst(origin, t Type, v float64) Node             // u32 consts: ir.InternScalar(ir.TU32)
g.NewBinOp(origin, op BinOpCode, a, b Node) Node       // BinOpAdd/Sub/Mul/Div/...
g.NewCall(origin, name string, args ...Node) Node      // node_expr.go; name payload; registry-checked only by typeinfer
g.NewMemLoad(origin, tensor Node, axes []Node) Node
g.NewMemLoadSMem(origin, src Node, bivs ...Node) Node  // src must be MemNewSMem (verifier)
g.NewMemStoreSMem(origin, value, dst Node, bivs ...Node) Node
g.NewMemNewSMem(origin, extent Node) Node
g.NewFragLoad / g.NewFragStore / g.NewFragNew           // same shapes, FragNew buffer
```
Read faces: `ir.OperandsOf(n) []Node`, `ir.ImpactBy(n) []SSAUse`, `ir.UsesOf(n) []SSAUse`, `ir.DataDeps(n) []Node`, `ir.OperandsOf(u.Def())` for operands; `SSAUse.Def() Node`, `SSAUse.User() Node`; `ir.TypeOf(n)`, `ir.IsErased(n)`, `g.AllocatedNodes() []Node`, `ir.Verify(g) []Violation`.

Pass registration pattern (tiling.go): `opt.Register(opt.PassSpec{ID, Priority, Factory})` in init(); Pass implements `ID/Requires/Produces/Preserved/Run`. Manager checks Requires against earlier-run passes, runs `checkLegality(p.Produces(), g)` after Run, then `cache.Invalidate(p.Preserved())` (invalidate-all default — an empty Preserved drops the record, which is what the lower pass wants AFTER consuming it).
</interfaces>
</context>

<decisions_pinned_by_this_plan>
The spec is normative everywhere it speaks. It leaves a small number of representation choices to the implementer (M1-M4 cover annotations, empty-set, buffer placement, access kinds only). This plan pins the remainder — implement exactly these, citing them in code comments:

- **P1 intrinsic payload/shape**: `field` is a string payload on the node struct (callNode.name precedent, node_expr.go) read via a total `IntrinsicFieldOf(n) string` face; `DoSoC` is the single operand and must be a const-u32 KindConst (verifier shape case checks operand count == 1, DoSoC constness, field in {"index", "dimension"}). Family: FamilyAbstract (the TensorDimOf/InductionVariable pure-value precedent, so EffectOf is free); tier LIR; kinds insert between KindInductionVariable and KindEnd (append-only rule, NO KindVersion bump); kind strings "intrinsicgridmap"/"intrinsicthreadmap".
- **P2 builtin names**: `"umin": {2, TU32, TU32}` and `"umax": {2, TU32, TU32}` (the fminf/fmaxf type-prefix precedent). §3's recipe lists both; umin is the one the pass emits. ceildiv is NOT added — §6bis PHASE 0 note: 0-based between loops eliminate ceil from loop bounds entirely.
- **P3 ForLoop annotations (M1)**: KindForLoop catalog arity changes 3 → -1 (variadic); `NewForLoop(origin, iv, bound, body Node, attrs ...Node)` (no other callers exist — verified). Annotations are trailing KeyedAttribute inputs with u32-const values, keys exactly: "soc" (0=seq, 1=cta, 2=thr — the materialization primitive per §4: cta_loop/thr_loop/seq_loop, encoding mirrors SoCType order), "dosoc" (the loop's map dimension), "loc" (that loop's materialization concurrency: CTA between = MaxCTA; CTA coupled inside = coupled thread count; independent WARP between = LoC×32; sequential = 1). Verifier shape case extended: forLoop trailing operands beyond slot 2 must be KeyedAttribute nodes. print.go's forLoop case renders the trailing attributes.
- **P4 bound operand**: each generated ForLoop's bound is a fresh NewBoundedDimVar over the loop's DimVar with the bound expression as max (no production code builds ForLoops today — this pass sets the convention inside the existing shape rules; the verifier pins only operand 0 = dimvar).
- **P5 fill representation** (the one node the spec names only as a runtime API): the fill is a `MemStoreSMem(value, buf, coveredRoundBases...)` (SMEM class) or `FragStore(value, buf, coveredRoundBases...)` (FRAG class) whose VALUE operand is a fresh tile-shaped `NewMemLoad(tensor, fillAxes)` — a partial deref (k < rank) typing as a tensor slice, so the existing typeinfer store rule (store types as its value operand) passes. The cooperative-vs-thread variant rides one KeyedAttribute annotation input on the store: key "loadtile", u32-const value 1 = LoadTileCta (any covered dim group-parallel), 2 = LoadTileThread (all covered dims sequential), per §6's variant rule. The tile-interior enumeration and partial-tile clamping are runtime-owned (§4/§6); the compiler emits no loop and no barrier for the fill. MemStoreSMem/FragStore are variadic kinds, so the trailing annotation input needs no arity change.
- **P6 liveness and effect-chain re-pointing**: KindForLoop is data-reachable only downward (loop → body), so rewritten loop-tree roots are unreachable from the two-root set unless chained. Rule: after an LNT's rewrite, every live ImpactBy payload edge (any bearer's effectIns record) whose target lies inside the LNT's rewritten body is re-pointed via `MutContext.SetDef(edge, lntRoot)`; for an erased promoted MemLoad, its incoming per-tensor chain edges re-point to that site's fill source MemLoad (the fill's source load chains on the source tensor exactly as NewMemLoadChained would — its effectIns wired at construction is impossible post-construction, so the pass re-points the erased load's incoming edges to the fill source load, preserving order and liveness). This keeps return.ImpactBy reaching every loop root, ordering "after the whole loop" — semantically the correct total order.
- **P7 no companion verify pass**: §8 phrases the self-checks as "pass exit conditions" — implement them as an in-pass final sweep in Run (any violation = non-nil error), NOT a separate tile-lower-verify pass. The pair-pass pattern is available later if the author asks.
- **P8 no self-typing**: per §6bis PHASE 5 the pass does NOT stamp types on inserted nodes; it relies on the pipeline's typeinfer re-run. The default pipeline has no post-lower typeinfer slot yet (final-lower lands in a later phase), so lower_test.go exercises the contract explicitly: run typeinfer over the rewritten graph and assert success (Phase 0's rules make it pass — §5 companion 4).
- **P9 legality companion (§5 companion 5) as implementable today**: only StageConstructed carries a window (legality.go); constructedWindow marks every AllKinds() kind Legal, so the new kinds' catalog rows ARE their legality row today. Pin the window contract in the catalog row doc text ("LIR, exists only between tile-lower and codegen; Legal at StageConstructed by the catalog-iteration rule — the revision point when stage windows land"), mirroring legality.go's own revision-point note. The pass Produces StageConstructed like every current pass.
</decisions_pinned_by_this_plan>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: PHASE 0 — IR prerequisites (intrinsic kinds + six companions, umin/umax, annotated ForLoop)</name>
  <files>internal/ir/kind.go, internal/ir/catalog.go, internal/ir/node_intrinsic.go, internal/ir/node.go, internal/ir/verify.go, internal/ir/value.go, internal/ir/builtins.go, internal/ir/node_loop.go, internal/ir/print.go, internal/ir/builtins_test.go, internal/opt/typeinfer/rules.go, doc/ir.md</files>
  <behavior>
    - Test: constructing IntrinsicGridMap/IntrinsicThreadMap with a const-u32 DoSoC and field "index" or "dimension" yields u32 under typeinfer; any other field string is a verifier shape violation; a non-const or multi-operand intrinsic is a verifier shape violation.
    - Test: two intrinsic nodes with the same (DoSoC const value, field) over the same operand ValueHash/ValueEqual equal; differing field or differing DoSoC def do not.
    - Test: ir.Builtin("umin") and ir.Builtin("umax") return {2, TU32, TU32}; a u32-arg umin call types u32 under the existing KindCall rule (registry-driven — no rule change needed beyond the closed-switch membership).
    - Test: NewForLoop with trailing KeyedAttribute inputs verifies (no surplus violation); a forLoop with a trailing NON-KeyedAttribute operand is a shape violation; the 3-operand legacy shape still verifies.
  </behavior>
  <action>
    Implement §6bis PHASE 0 exactly, per pinned decisions P1-P4, P9. All edits follow the exact layering of existing kinds (doc/ir.md §2's "three places" plus the spec's six companions):

    1. kind.go: add KindIntrinsicGridMap and KindIntrinsicThreadMap between KindInductionVariable and KindEnd (append-only within the LIR tier — do NOT renumber existing constants, do NOT bump KindVersion); add String() cases "intrinsicgridmap"/"intrinsicthreadmap"; Family() leaves them in the FamilyAbstract default arm (add a comment line naming them beside TensorDimOf/InductionVariable).
    2. catalog.go: two TierLIR/FamilyAbstract rows — KindIntrinsicGridMap arity 1, doc "grid-map query: (DoSoC const u32); string field payload index|dimension; result u32"; KindIntrinsicThreadMap likewise. Doc text carries the P9 legality-window note.
    3. NEW internal/ir/node_intrinsic.go: intrinsicNode struct (*Base + operandList + field string), total read face IntrinsicFieldOf(n) string, constructors NewIntrinsicGridMap/NewIntrinsicThreadMap(origin, field string, dosoc Node) — mirror callNode/NewCall in node_expr.go (name payload set after NewNode; one variadic-safe slot).
    4. node.go allocNode: two cases allocating intrinsicNode.
    5. verify.go checkNodeShape: new case — operand count exactly 1; operand 0 a KindConst with InternScalar(TU32) payload type (use ConstTypeOf; nil operand left to the generic slot check per file convention); IntrinsicFieldOf in {"index","dimension"}. Extend the forLoopNode case: every operand past index 2 must be a *keyedAttributeNode (message "for-loop trailing operands must be KeyedAttribute annotations (M1)"). These are extensions of the EXISTING per-kind shape class — zero new invariant classes (§8).
    6. value.go: contentHash case mixing mixString(valueHashOffset, field); contentEqual case comparing field equality (DoSoC is an operand — identity-keyed automatically). This is §5 companion 6 so GVN buckets stay faithful (§5 GVN-CSE of the intrinsics is sound and desirable).
    7. builtins.go: add "umin"/"umax" entries (P2) and update the file comment ("every entry is (f32,...f32)->f32" now has the u32 min/max exception). Update builtins_test.go's pinned table.
    8. node_loop.go: P3 — NewForLoop gains variadic attrs; allocate 3+len(attrs) slots; doc comment states the M1 annotation contract and the key vocabulary. Update print.go's forLoop case to render trailing KeyedAttribute inputs (mirror how reduce renders its init attribute) and add print cases for the two intrinsic kinds rendering field + DoSoC operand (mirror callNode's name rendering). kind.go's String cases cover canonical spelling.
    9. internal/opt/typeinfer/rules.go: add both kinds to the u32 class case (the KindIter/KindForLoop/KindDimVar/KindBoundedDimVar/KindInductionVariable/KindTensorDimOf arm) — §5 companion 4; without it the post-pass re-typing fails closed. Update the file-header kind count comment (31 → 33).
    10. doc/ir.md: update the sentinel value mention (KindEnd 32 → 34), the "All 31 kinds" count and listing table with the two new rows, and the ForLoop arity mention if the doc lists arities.

    Write the behavior tests FIRST in the existing test files' style (node_convention_test.go / value_test.go / builtins_test.go / verify_test.go as fitting; a new internal/ir/node_intrinsic_test.go is acceptable) — RED, then implement to GREEN. Do not touch anything under internal/opt/tiling in this task.
  </verify>
    <automated>cd /home/ps/Workspace/cuda/project/tcacomp && go build ./... && go vet ./... && go test ./internal/ir/... ./internal/opt/typeinfer/... </automated>
  </verify>
  <done>Both intrinsic kinds exist with all six companions (struct+catalog, allocNode, verify shape, typeinfer u32 rule, legality-note row, contentHash/contentEqual) plus kind-string/print rendering; umin/umax registered and u32-typed; NewForLoop carries M1 KeyedAttribute annotations with verifier shape enforcement; full ir + typeinfer packages green.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: PHASE 1-3 — pass skeleton, record fetch, DoSoC numbering, loop rewrite</name>
  <files>internal/opt/tiling/lower.go, internal/opt/tiling/lower_rewrite.go, internal/opt/tiling/lower_test.go, internal/opt/registry.go</files>
  <behavior>
    - Test: tile-lower without the record in the cache fails closed with a named error; Requires() == ["tiling"] and the manager rejects a pipeline missing it.
    - Test (numbering): on a chain fixture with one CTA dim, one independent WARP dim, one sequential dim: the CTA dim gets GridMapping dim 0; the CTA dim's coupled warp and the independent warp dim get DISTINCT ThreadMap dims (coupled = 0, independent = 1 per D3's m+n counting); the sequential dim gets none; a second LNT restarts both counters.
    - Test (rewrite, TileSize==1): the dim keeps a single annotated for-loop and its original IV uses map directly to the loop index expression.
    - Test (rewrite, SEQ dim T>1): between loop (iv step T, bound = raw extent), inside sequential loop bounded by umin(T, extent − offset) — a NewCall "umin" node is reachable; the original KindInductionVariable has zero live uses; the original KindIter is erased.
    - Test (rewrite, CTA dim): between loop step LoC×T with LoC==MaxCTA; in-body IntrinsicGridMap(dosoc,"index") rank expression; inside loop element-stride with the coupled thread count; index = offset + a_i + IntrinsicThreadMap(coupledDim,"index").
    - Test (rewrite, WARP dim): between loop step (LoC×32)×T; in-body IntrinsicThreadMap(dosoc,"index"); inside sequential over the owned chunk.
    - Test (annotations): every generated ForLoop carries soc/dosoc/loc KeyedAttribute inputs per P3; loops of a sequential dim carry soc=0/loc=1.
    - Test (graph health): ir.Verify reports zero violations after Run; BuildingDCE-style reachability — every generated ForLoop root is live via a re-pointed ImpactBy edge (P6); Run is deterministic (two runs over fresh copies produce identical printed graphs).
  </behavior>
  <action>
    Implement §6bis PHASE 1-3 + §2 + §3 obligations + §4's compiler-side shape, as a new pass in the SAME package tiling (the simplify package's multi-pass precedent: one package, several registered passes), per pinned decisions P4, P6. doc/optana/loop_tile_lower.md §6bis is the step-by-step authority; loop_tiling.md's "Mechanical rewrite for tiling" templates are the normative loop forms.

    1. internal/opt/tiling/lower.go — pass plumbing: init() registers ID "tile-lower", Priority 116 (immediately after tiling-verify 115; update registry.go's anchor comment block with the 116 line). Requires ["tiling"] — the §1 record-lifetime pin. Produces StageConstructed (P9). Preserved: EMPTY (the record describes pre-rewrite iters keyed by nodes this pass erases — it must not outlive the pass). Gates mirror tiling.go: nil graph and materialized/CFG-carrying graph are deterministic errors before anything is read; Params validated (reuse DefaultParams; the pass needs MaxCTA/MaxThreadsPerCTA/WarpPortion for the static strides and coupled thread counts).
    2. PHASE 1 skeleton in Run: fetch ctx.Cache.Get(RecordName), cast *Record, FAIL CLOSED (spec §6bis: a nil ctx/cache or wrong type is an error, not a silent skip). Re-derive loopforest.Analyze + sideeffect.Analyze + analyzeOverlap (same-package helpers — the SAME deterministic chase). Per root in Roots() order, resolve every iter via bivOf to its IterDecision; a missing entry is an ERROR (adjacency guarantees completeness).
    3. internal/opt/tiling/lower_rewrite.go — DoSoC numbering (§2/§6bis PHASE 2): per LNT walk treeItersOf order (DFS preorder = outer→inner); one GridMapping counter (increments on SoCTA iters), one ThreadMap counter (increments on the COUPLED warp of each SoCTA iter at that iter's position, and on each SoCWARP iter — D3's counting invariant: ThreadMap dimensionality = m coupled + n independent, the coupled warp is NOT a CTA-space dim); both restart per LNT; SoCSequential gets neither. Store per-iter: soc, gridDim (CTA iters), threadDim (coupled + independent), loc.
    4. Loop rewrite (§6bis PHASE 3), per LNT in dim order, building NEW defs only ( originals retired by RAUW, never edited — §3): all between loops nest in dim order, all inside loops nest inside them in dim order. Per SoC + TileSize:
       - TileSize == 1: single annotated seq loop, IV maps directly to the global index.
       - SEQ: between DimVar(init 0, step T), bound = the ORIGINAL extent node (the iter's BIV max operand — 0-based strides eliminate ceil: bound is the raw extent); inside sequential loop bounded by materialized tile_max = umin-call(T_const, extent − offset_expr).
       - CTA: between step LoC×T (LoC = record LoC == MaxCTA); in-body cta_rank = IntrinsicGridMap(gridDim, "index"); offset = cta_rank×T + between-IV; inside = the COUPLED warp walk at ELEMENT stride: step = coupled thread count (static: (PortionWarp − independentPortions) × quantum × 32, quantum = (MaxThreadsPerCTA/32)/PortionWarp; mirror search.go's WarpPortion auto logic for the portion split), index = offset + inside-IV + IntrinsicThreadMap(coupledThreadDim, "index"); bound = tile_max.
       - WARP: between step (LoC×32)×T (LoC is a warp count — tiling-verify's unit); in-body thread_rank = IntrinsicThreadMap(threadDim, "index"); offset = thread_rank×T + between-IV; inside SEQUENTIAL over the owned chunk bounded by tile_max.
       Every generated loop: NewDimVar over the ORIGINAL dim InputParam (reuse the iter's dim), NewForLoop(iv, NewBoundedDimVar(iv, boundExpr), body, attrs...) with P3 annotations. NO map index in any induction variable — ranks appear only as in-body offset locals; NO guard, NO branch, NO barrier ever (§4: tail guarding is the primitive's whole-body no-op).
    5. Body rewiring + retirement: RAUW every InductionVariable use of the LNT's iters to the (offset + inside) expression per template (for the innermost body placement: the LNT's innermost effectful node — the memory op the deepest loop level owns — becomes the innermost inside loop's body operand; each outer loop's body is the next loop node). Retire the originals through MutContext only: after RAUW, erase the InductionVariable nodes, then the Iter nodes, then the old BIV/DimVar objects, dependency order (Erase refuses live users — users first). Apply the P6 chain re-pointing: every live ImpactBy edge targeting a node now inside the rewritten body re-points (SetDef) at the LNT's outermost loop root. All walks deterministic (allocation order, tree order — no map-iteration emission; §7).
    6. lower_test.go: behavior tests above, fixtures in overlap_test.go's style (newChain/matmul are same-package and reusable). Write the numbering + TileSize==1 + fail-closed tests FIRST (RED), then implement, then the CTA/WARP/annotation/determinism tests. PHASE 4/5 (cache insertion) are NOT in this task — promoted-site surgery is Task 3; this task's fixtures should use Params/records where sites end up unpromoted (e.g. a chain with a single load = no co-traversed set) or assert rewrite-only invariants.
  </verify>
    <automated>cd /home/ps/Workspace/cuda/project/tcacomp && go build ./... && go vet ./... && go test ./internal/opt/... </automated>
  </verify>
  <done>tile-lower@116 registered, requires tiling, fetches the record fail-closed; every decided LNT rewrites into annotated tiled ForLoop nests per the three templates with materialized umin bounds, map-free inductions, and rank/offset intrinsics; all original IV/iter chains retired through MutContext; P6 chain re-pointing leaves every loop root live; ir.Verify clean; deterministic.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: PHASE 4-5 — SMEM/FRAG cache insertion, epilogue self-checks, matmul end-to-end</name>
  <files>internal/opt/tiling/lower_cache.go, internal/opt/tiling/lower_test.go</files>
  <behavior>
    - Test (matmul end-to-end): decide → tiling-verify → tile-lower over the matmul fixture: lhs and rhs MemLoads erased with zero live uses; a MemNewSMem (or FragNew per the record's class) exists per promoted site declared at the fill point; one fill (P5 store with "loadtile" annotation and tile-shaped source MemLoad) sits inside the innermost covered dim's between-round body before the tile-inside loops; every former use of the load reads MemLoadSMem/FragLoad over the buffer with the covered dims' inside coordinates; uncovered inner INS dims stay live loop indices below the buffer; ir.Verify zero violations; a typeinfer Run over the rewritten graph SUCCEEDS (P8).
    - Test (worked check §6bis): the lhs [T_k] buffer fills per k_o round and the fill's m-coordinate is the enclosing m index expression (INS above the tie baked as per-round fixed values), not a loop over the buffer.
    - Test (loads-only): stores never route through the cache — the out store survives as a GMEM MemStore.
    - Test (budget fallback §6): with MaxCache forced tiny, the halve-then-drop loop terminates — tile sizes halve while over budget, sites drop back to GMEM (their MemLoads survive unrewritten) when halving reaches 1; with DefaultParams the fallback is a no-op (tiling-verify already asserted Σ ≤ MaxCache).
    - Test (self-checks §8): each exit condition individually provable — an injected stale-IV use or a fill placed inside an inside loop of a parallel dim makes Run return a named error.
  </behavior>
  <action>
    Implement §6bis PHASE 4-5 + §6 site-rewrite mechanics + §8 self-check list, per pinned decisions P5-P8.

    1. internal/opt/tiling/lower_cache.go — per promoted site (Record.Sites keyed by MemLoad id, run AFTER the loop rewrite so coordinates exist):
       - Recompute the site facts with the SAME deterministic chase (analyzeOverlap's SiteFacts — same package; adjacency makes the results identical, §6bis). Cross-check the recomputed Tie/Covered against the record; mismatch = error (fail closed).
       - FILL POINT (§6bis PHASE 4 + M3): inside the innermost COVERED dim's between-round body, immediately before the tile-inside loops; Tie == nil → the same rule at the outermost position. Buffer + fill AT that point: MemNewSMem (ClassSMEM) or FragNew (ClassFRAG), extent = product of covered tile sizes × element bytes (element size from the tensor param's payload type — f32 = 4); fill per P5: MemStoreSMem/FragStore whose value is a fresh NewMemLoad(tensor, fillAxes) partial deref — covered dims at their round-base offset expressions, INS-above-tie dims at their enclosing rewritten index expressions, uncovered trailing dims absent (the partial deref keeps them as the slice's trailing dims); variant annotation KeyedAttribute("loadtile", 1|2) per §6's rule (any covered dim group-parallel → 1 = LoadTileCta; all covered dims sequential → 2 = LoadTileThread). NO barriers, ever.
       - ACCESS REWRITE: RAUW the MemLoad's uses to NewMemLoadSMem(buf, coveredInsideCoords...) / NewFragLoad(buf, ...) (M4); retire the MemLoad via Erase (users already RAUW'd); per P6, the erased load's incoming per-tensor chain edges re-point to the fill's source MemLoad. Uncovered inner INS dims stay live loop indices BELOW the buffer — that re-read IS the reuse.
       - BUDGET (§6): recompute Σ SMEM bytes (siteBytesPerTile over the final tiles); while over MaxCache: halve the reuse tile sizes of the offending sites (rebuild their buffers/fills/accesses), and when a tile size reaches 1 drop that site back to GMEM (undo its surgery — keep the GMEM load, erase buffer/fill/access). Terminates by construction. FRAG re-assert: covered dims all sequential SoC (the judge's criterion re-checked — the fully-unroll condition's precondition).
    2. PHASE 5 epilogue in lower.go's Run: after all LNTs and sites — run ir.Verify(g), returning any violation as an error; run the §8 self-check sweep (P7): no live KindInductionVariable/KindIter of any rewritten LNT; every parallel-inside loop carries the materialization annotation (thr/cta); no fill inside an inside loop of a parallel dim (layering, §4); no stale MemLoad for any promoted site; zero barriers (no sync-emitting node was ever constructed — assert the fill-store inventory only); every FRAG site sequential-covered. All checks read-only over the finished graph, deterministic order.
    3. lower_test.go additions: the behavior tests above, matmul() as the worked example (SET {lhs,rhs}@iterK, lhs tie iterN covers [·][K], rhs tie nil covers [N][K] — assert both buffers' placements per the worked check). Full-pipeline test: tiling → tiling-verify → tile-lower → ir.Verify → typeinfer re-run (P8). Write the matmul end-to-end and budget-fallback tests FIRST (RED), then implement.
  </verify>
    <automated>cd /home/ps/Workspace/cuda/project/tcacomp && go build ./... && go vet ./... && go test ./... </automated>
  </verify>
  <done>Promoted sites lowered to buffers+fills+accesses per §6/P5 with the variant annotation; loads-only scope holds; budget fallback terminates; §8 self-checks enforced as pass exit conditions; matmul end-to-end green through verify + typeinfer re-run; full-repo build/vet/test green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| IR graph in → pass out | In-process compiler data; no external input crosses at runtime |
| analysis cache → pass | The tiling-record is consumed in-process; a missing/corrupted record must fail closed |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-jay-01 | Tampering | record fetch in lower.go | mitigate | Fail-closed cast+presence check on ctx.Cache.Get(RecordName); missing entry or wrong type = named error before any mutation |
| T-jay-02 | Tampering | all graph mutations | mitigate | MutContext choke points only (internal/ir/mutate.go) — project rule; no direct field writes anywhere in the pass |
| T-jay-03 | DoS | budget fallback + rewrite loops | mitigate | Halving reaches 1 / dropping empties the set (§6 termination); rewrite adds NO search of its own (§7 determinism) — loops are bounded by LNT count |
| T-jay-04 | Repudiation | pass diagnostics | mitigate | Errors name node IDs and pass "tile-lower" (LastTouchedBy stamps ride every choke point) |
| T-jay-05 | Information disclosure | generated code | accept | No secrets exist in compiler IR; intrinsics query only launch-map state |
</threat_model>

<verification>
- Task gates: go build ./... && go vet ./... && go test ./... (full-repo, the project's gate convention per STATE.md quick-task history).
- Structural: grep -c '"tile-lower"' internal/opt/tiling/lower.go confirms registration; go test ./internal/opt/ -run 'TileLower|Matmul' pins the pass behavior.
- Spec-fidelity self-audit before summary: every §6bis PHASE 0-5 step has a code home; every §8 exit condition has a check; all six §5 companions present for both intrinsic kinds; M1-M4 honored; P1-P9 decisions commented at their implementation sites.
</verification>

<success_criteria>
- The decide→lower pair composes: tiling → tiling-verify → tile-lower runs clean over the matmul worked example, producing annotated tiled loop nests + intrinsics + SMEM/FRAG caches, verifier-clean, typeinfer-re-run-clean, deterministic.
- Zero new verifier invariant classes (§8) — only extensions of the existing per-kind shape class.
- Pure Go stdlib only; no new module dependencies; conventional commits per task (feat/test scopes).
</success_criteria>

<output>
Create `.planning/quick/261021-jay-implement-loop-tile-lower-optimization-p/261021-jay-SUMMARY.md` when done
</output>
