---
phase: quick-261018-nga
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/ir/node.go
  - internal/ir/value.go
  - internal/ir/value_test.go
  - internal/ir/effect.go
  - internal/ir/effect_test.go
  - internal/analysis/sideeffect.go
  - internal/analysis/sideeffect_test.go
  - internal/opt/gvn/gvn.go
  - internal/opt/gvn/verify.go
  - internal/opt/gvn/gvn_test.go
  - internal/opt/all/all.go
  - internal/opt/registry.go
autonomous: true
requirements: ["QUICK-261018-nga"]
tags: [ir, optimization, gvn, cse, analysis]

must_haves:
  truths:
    - "Two pure computations with pairwise-identical SSA operands collapse to a single representative after the gvn pass; all former users resolve to the representative through their SSAUse records"
    - "No node the side-effect analysis marks effectful is ever replaced, merged, or erased by gvn (every FamilyMemory node — loads included, nil ImpactBy included — control-flow nodes, and everything the order-sensitive closure taints)"
    - "The side-effect analysis is accumulated through order-sensitive kinds ONLY (FamilyMemory, FamilyLoop, Return, future collectives); pure kinds (ScalarExpr, Map, Reduce, Const, Param, InputParam) never propagate taint, so address/index math over iter values is GVN-eligible"
    - "EffectOf marks every FamilyMemory node node-local EFFECTFUL (nil ImpactBy included) while staying self-scoped; DCE (root-reachability based) and scheduler behavior are unchanged, proven by their existing suites"
    - "Operands are keyed by SSA node identity in hash and equality: `i*2` computed twice over the same iter node dedups; `i1*2` vs `i2*2` over distinct iter nodes NEVER dedups"
    - "Two structurally identical loads against one tensor do NOT merge (memory read-read CSE is explicitly out of scope, deferred to a future materialization/load-store pass)"
    - "gvn-verify fails on a graph where two pure ValueEqual nodes both remain, where the accumulated analysis is not a fixpoint over the graph, or where any use record dangles"
    - "BuildPipeline (via internal/opt/all) runs gvn then gvn-verify between dce-verify and schedule"
  artifacts:
    - path: "internal/ir/value.go"
      provides: "ValueHashOneOfAKind sentinel, SideEffectAnalysis query interface, promoted *Base ValueHash/ValueEqual implementing the identity-keyed hash/equality contract"
      min_lines: 120
    - path: "internal/analysis/sideeffect.go"
      provides: "Analyze(g) accumulated side-effect analysis implementing Analysis (Name) and ir.SideEffectAnalysis, with the order-sensitive closure"
      exports: ["Analyze", "SideEffects"]
    - path: "internal/opt/gvn/gvn.go"
      provides: "GVN/CSE pass: global hash table keyed on ValueHash/ValueEqual, dedup via RAUW + Erase"
      exports: ["Pass"]
    - path: "internal/opt/gvn/verify.go"
      provides: "Post-GVN invariant verifier pass"
      exports: ["VerifyPass"]
  key_links:
    - from: "internal/opt/gvn/gvn.go"
      to: "internal/analysis"
      via: "analysis.Analyze(g) consumed as ir.SideEffectAnalysis"
      pattern: "analysis\\.Analyze"
    - from: "internal/ir/value.go"
      to: "ir.SideEffectAnalysis"
      via: "one-of-a-kind gate consults the accumulated analysis before hashing/comparing"
      pattern: "HasSideEffect"
    - from: "internal/opt/gvn/gvn.go"
      to: "internal/ir MutContext"
      via: "ReplaceAllUsesWith + Erase through the mutation choke point (references move via SSAUse records)"
      pattern: "ReplaceAllUsesWith"
    - from: "internal/opt/all/all.go"
      to: "internal/opt/gvn"
      via: "blank import firing the self-registration init"
      pattern: "opt/gvn"
---

<objective>
Implement CSE via GVN (Global Value Numbering) on the TCA sea-of-nodes IR.

Purpose: the sea of nodes has no BB/CFG, so once every node carries a side-effect
label, global value numbering dedups pure computations graph-wide in one hash
table pass — the cheapest high-value optimization pass in the pipeline.

Output: (1) `ValueHash`/`ValueEqual` on the IR `Value` interface taking the
side-effect analysis as input; (2) the accumulated side-effect analysis in the
EXISTING `internal/analysis` package; (3) `internal/opt/gvn` — the GVN pass plus
a dedicated GVN-invariant verifier pass, self-registered into the default
pipeline.

AMENDED SPEC (user review decisions A–G; they override the corresponding parts
of the original spec items):

- **A — EffectOf verdicts CHANGE:** every FamilyMemory node (GMEM
  MemLoad/MemStore/MemNew, MemLoadSMem/MemStoreSMem/MemNewSMem,
  FragLoad/FragStore/FragNew) is node-local EFFECTFUL even with nil ImpactBy.
  Rationale: the memory model treats each memory region as a single slot;
  subscripts are too coarse to disprove aliasing between a load and a write,
  so all ops on one memory node must serialize. Task 1 edits the verdicts AND
  audits/re-runs the DCE + scheduler consumers (legality has no EffectOf
  consumer). DCE liveness is two-root-set reachability (R3), not EffectOf
  based, so it must be proven unaffected via the existing suite (pin test if
  the suite lacks one).
- **B — memory read-read CSE OUT of scope:** deferred to a future
  materialization/load-store pass. Negative pin: two structurally identical
  loads against one tensor do NOT merge.
- **C — taint relaxed:** accumulated side effect propagates through
  order-sensitive kinds ONLY (FamilyMemory, FamilyLoop = Iter/IterCluster/
  ForLoop, Return, future collectives). Pure kinds (ScalarExpr family, Map,
  Reduce, Const, Param, InputParam) do NOT propagate — address/index math over
  iter values MUST be GVN-eligible.
- **D — identity keying (load-bearing consequence of C):** ValueHash and
  ValueEqual key EVERY operand by SSA node identity (the representative's
  node ID; equality = node identity), NEVER recursive structural equality over
  operands. Pure nodes with side-effectful operands participate; two nodes
  whose operands are distinct SSA defs (e.g. two one-of-a-kind iter nodes with
  identical fields) must NEVER merge. The former "InputParam atoms" rule
  extends to ALL operands of ALL kinds.
- **E — KindReduce participates in GVN:** ops=[init, piped] plus combiner
  payload deep-equality via the same lambda-comparison machinery as Map
  (doc §2.5: reduce is stateful but has no observable side effects).
- **F — location:** the analysis lives in the EXISTING `internal/analysis`
  package (`sideeffect.go` there), not a new `internal/ana` package —
  `internal/analysis` already hosts the Analysis contract internal/opt builds
  on. No import-cycle blocker exists: `internal/analysis` → `internal/ir` is
  the only edge (`internal/ir` does not import `internal/analysis`), and the
  query face `ir.SideEffectAnalysis` is defined in ir precisely so analysis can
  implement it without a cycle.
- **G — unchanged (kept):** fresh-object rule (MemNew/MemNewSMem/FragNew
  effectful — merging distinct allocations is unsound; now also implied by A),
  lambda-body conservative scope (body roots skipped, documented upgrade
  path), priorities 110/111, verifier's side-effect-never-replaced assertion.

Spec-fidelity matrix (original items [1]–[8] as amended, plus amendment-only
rows; every row Full):

| Spec item | Delivered by | Notes |
|---|---|---|
| [1] GVN-based CSE on sea-of-nodes | Task 3 | global table, no BB/CFG |
| [2].1 non-zero ImpactBy ⇒ side effect | Task 2 | base verdict, `len(ir.ImpactBy(n)) > 0` (future-bearer hook; all current bearers are memory/return, covered by A) |
| [2].2 memory op ⇒ side effect — AMENDED (A) | Tasks 1+2 | EVERY FamilyMemory node, loads included, node-local effectful with nil ImpactBy; single-slot memory model |
| [2].3 implicit control flow (Iter) ⇒ side effect | Task 2 | Iter, IterCluster, ForLoop, Return (conservative closure of "Iter node") |
| [2].4 operand transitivity — AMENDED (C) | Task 2 | propagation through order-sensitive kinds ONLY (FamilyMemory, FamilyLoop, Return, future collectives); pure kinds never taint |
| [2].5 side-effect nodes never participate in GVN | Tasks 1+3 | ^uint64(0) sentinel gate in the pass; ^0/false contract in the IR |
| [3].1 ValueHash, simple combiner — AMENDED (D) | Task 1 | operands keyed by SSA node identity (representative's node ID), no structural recursion into operands |
| [3].2 ValueEqual deep equality — AMENDED (D+E) | Task 1 | operand equality = node identity, ALL kinds; recursion ONLY through lambda payloads, via the shared Map/Reduce lambda-comparison machinery |
| [3].3 side-effect: hash = uint64(-1), equal = false | Task 1 | literal, incl. self-comparison |
| [4].1 standalone analysis — AMENDED (F) | Task 2 | internal/analysis/sideeffect.go (existing package); gvn only uses it |
| [4].2 Value interface methods take the analysis as input | Task 1 | `SideEffectAnalysis` param on both methods |
| [4].3 Effect field = self state, not accumulated — AMENDED (A) | Task 1 | verdicts CHANGE (all FamilyMemory effectful) but stay self-scoped (no lattice meet); accumulation lives only in internal/analysis; DCE + scheduler audited and re-run |
| [4].4 global hash table (no BB/CFG) | Task 3 | `map[uint64][]ir.Node`, buckets resolved by ValueEqual |
| [4].5 table records Value→SSA; ALL references via SSAUse | Task 3 | representative node IS the SSA def; RAUW moves ref records |
| [4].6 dedup during table building | Task 3 | duplicate found ⇒ RAUW + Erase along the way |
| [4].7 GVN verifier + during-pass assert | Task 3 | VerifyPass + hard assert before every RAUW |
| [4].8 tests for both passes | Tasks 1–3 | value contract, accumulated analysis, gvn + gvn-verify |
| AMEND-B memory read-read CSE deferred | Task 3 | negative pin: two identical loads do NOT merge |
| AMEND-D identity keying called out | Tasks 1+3 | `i1*2` vs `i2*2` (distinct iter defs) never merge; shared-operand duplicates do |
| AMEND-E KindReduce participates | Tasks 1+3 | not gated out; combiner compared via the Map lambda machinery |
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@doc/tca.md
@internal/ir/node.go
@internal/ir/effect.go
@internal/ir/ssa.go
@internal/ir/mutate.go
@internal/opt/manager.go
@internal/opt/registry.go
@internal/opt/dce/dce.go
@internal/opt/dce/verify.go

<interfaces>
<!-- Key contracts the executor builds against. Extracted from the codebase. -->

From internal/ir/node.go — the interface being extended (Task 1):
```go
type Value interface {
    Node        // BasePtr() *Base; KindOf() Kind
    IsNone() bool
    // ADD: ValueHash(se SideEffectAnalysis) uint64
    // ADD: ValueEqual(other Value, se SideEffectAnalysis) bool
}
// Every concrete node embeds *Base; promoted methods satisfy the interface.
// Base.ssa.node is the back-pointer to the concrete node (set in allocNode).
```

From internal/ir/effect.go — verdicts AMENDED per decision A (Task 1 changes them):
```go
func EffectOf(n Node) Effect
// CURRENT: Full for KindMemStore, KindReturn; Free otherwise.
// AMENDED:  Full for EVERY FamilyMemory kind (MemLoad, MemStore, MemNew,
//           MemLoadSMem, MemStoreSMem, MemNewSMem, FragLoad, FragStore,
//           FragNew) plus KindReturn — node-local, nil ImpactBy included.
func ImpactBy(n Node) []SSAUse        // effect-input uses (may be empty/nil)
// EffectOf consumers (grep-verified): internal/opt/schedule/scheduler.go:463,511
// (belt-and-braces after the licmHoistableKinds whitelist) and effect_test.go.
// DCE does NOT consume EffectOf (two-root-set reachability, R3); Materialize's
// isEffectKind (internal/ir/dce.go) and legality do not consult it either.
```

From internal/ir (poset.go / print.go) — exported child enumeration (Task 2 uses these):
```go
func DataDeps(n Node) []Node      // live operand-slot defs, slot order
func EffectDeps(n Node) []Node    // live ImpactBy defs
func LambdaBodies(n Node) []Node  // map/reduce lambda body roots
func OperandsOf(n Node) []Node    // operand defs incl. unwired nil slots
```
Non-effect payload uses (store targets, iter dims/bounds, as-orders, lambda
params, paramRef) target InputParam atoms only — for the C-relaxed closure the
relevant child relation is DataDeps + LambdaBodies.

From internal/ir/mutate.go — the ONLY sanctioned mutation path (Task 3):
```go
type MutContext struct{ Pass string }
func (ctx *MutContext) ReplaceAllUsesWith(old, newVal Node) error
func (ctx *MutContext) Erase(n Node) error   // refuses while live uses remain
```

From internal/ir/mark.go:
```go
func (g *Graph) AllocatedNodes() []Node  // non-erased, allocation order (def-before-use)
func (g *Graph) Tombstones() []Node
```

From internal/opt (manager.go / registry.go) — pass contract (Task 3):
```go
type Pass interface {
    ID() string; Requires() []string; Produces() Stage
    Preserved() []string; Run(*Context, *ir.Graph) error
}
func Register(spec PassSpec)  // panics on empty ID / nil Factory / duplicate / ID mismatch
// Priority scale: 100 dce, 101 dce-verify, ..., 900 schedule. Smaller = earlier.
```

From internal/analysis/analysis.go — the package Task 2 extends:
```go
type Analysis interface{ Name() string }  // opt.Cache compatibility
// Currently dependency-free; sideeffect.go adds the single import of
// internal/ir (no cycle: ir never imports analysis).
```

From internal/ir/node_lambda.go — THE ONE NON-SSAUse EDGE (critical):
```go
type Lambda struct {
    params []SSAUse
    Body   Node  // raw pointer, NOT a use record — RAUW cannot rewire it
}
```

Node-kind quick map (internal/ir/kidn.go → kind.go): FamilyMemory = MemLoad,
MemStore, MemNew, MemLoadSMem, MemStoreSMem, MemNewSMem, FragLoad, FragStore,
FragNew. FamilyLoop = Iter, IterCluster, ForLoop. ScalarExpr = Const, Param,
BinOp, UnaryOp, Select, Cast, Call. Abstract others: Map, Reduce, InputParam,
Return, DimVar, BoundedDimVar.
</interfaces>
</context>

<design_decisions>
Discretionary choices and amendment implementations (documented in code where
they land):

1. **Single-slot memory model (A):** each memory region is one slot; tensor
   subscripts cannot disprove aliasing between a load and a write, so every
   FamilyMemory node is node-local effectful (nil ImpactBy included). This
   also subsumes the fresh-object rule (G kept for the MemNew* rationale:
   merging distinct allocations is unsound — each allocation denotes a
   distinct memory object; LLVM never CSEs allocas).
2. **Identity keying (D):** ValueHash mixes each operand/payload-target's
   NODE ID; ValueEqual compares operands by node identity — for ALL kinds,
   not just InputParam atoms. No structural recursion over operands. Two
   one-of-a-kind iter nodes with identical fields therefore never merge their
   consumers (`i1*2` vs `i2*2`); shared-operand duplicates do (`i*2` twice).
   The RAUW-during-processing cascade makes this compose like classic value
   numbering: once a duplicate def merges, its users' operand records point
   at the representative, so user-level duplicates become identity-equal and
   merge in the same pass.
3. **Order-sensitive closure (C):** `effectful(n) = base(n) OR
   (orderSensitive(n) AND ∃ effectful child)` with orderSensitive =
   FamilyMemory ∪ FamilyLoop ∪ {Return} ∪ (future collectives hook). Today
   base ⊇ order-sensitive (A), so the propagation clause mainly arms future
   collective kinds — it is the specified mechanism and is pinned by the
   negative tests (pure kinds with effectful operands stay pure).
4. **Reduce participates (E):** KindReduce is in the GVN domain; its combiner
   lambda compares via the same machinery as Map's (params by identity, body
   by the recursive identity-keyed equality). Distinct maps/reduces carry
   distinct lambda-param atoms, so they conservatively rarely merge — that is
   correct, not a gap.
5. **Location (F):** internal/analysis/sideeffect.go; the only new edge is
   internal/analysis → internal/ir (no cycle, stated in the plan per F).
6. **Lambda body roots are skipped by GVN** (G, pass-level policy):
   `Lambda.Body` is the one raw-pointer edge; RAUW cannot rewire it, so
   deduping a body root would strand a dangling `lam.Body`. Upgrade path
   (documented in gvn.go): a MutContext body-rewire choke point when
   lambda-body CSE ever matters. Body INTERIORS still dedup (their users are
   record-holders inside the body).
7. **Erase duplicates after RAUW** ("CSE along the way", [4].6).
8. **Deterministic processing order:** `AllocatedNodes()` in allocation order
   (construction is def-before-use topological; lowest-ID duplicate survives).
9. **Memory read-read CSE deferred (B):** all memory ops are effectful (A),
   hence excluded from GVN; the future materialization/load-store pass owns
   read dedup against the effect chains.
10. **Verify does full pairwise O(N²)** unique-representative checking over
    participating nodes (not bucket-limited) so hash/equality contract drift
    is also caught. Fine at v0 kernel scale (hundreds of nodes).
11. **No analysis publishing to the Cache:** gvn mutates the graph and
    preserves nothing; a pre-run analysis would be stale and a post-run put is
    wiped by `Invalidate(nil)`. gvn-verify recomputes fresh (independence).
12. **Invalid kinds fail closed:** `KindInvalid` / `Kind >= KindEnd` hash to
    the ^0 sentinel and compare false — mirrors the pass manager's fail-closed
    stance on unknown kinds.
</design_decisions>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: IR value contract (identity-keyed ValueHash/ValueEqual) + amended EffectOf verdicts</name>
  <files>internal/ir/value.go, internal/ir/value_test.go, internal/ir/node.go, internal/ir/effect.go, internal/ir/effect_test.go</files>
  <behavior>
    - Test: a pure node under a pure-stub analysis returns a non-sentinel hash; the SAME node under an effectful-stub returns ^uint64(0) and ValueEqual(other) == false even when other == the node itself ([3].3 literal).
    - Test (D — identity keying): two binops with the same opcode and the SAME operand nodes — same hash, ValueEqual true; same fields but DISTINCT operand nodes (two separately-built equal consts as operands) — ValueEqual false AND hashes differ.
    - Test: leaf pooling survives (no operands): two NewConst(f32,2) equal; two NewParamRefs to the SAME InputParam equal; to different params, unequal; two distinct InputParams never equal.
    - Test (D + participation): a pure binop whose operand is an effectful node (e.g. an iter) under an analysis marking the iter effectful — the binop still returns a non-sentinel hash and can be equal to an identically-keyed binop (pure nodes with side-effectful operands participate).
    - Test: KindInvalid and a Kind >= KindEnd node (g.NewNode) return the sentinel and compare false; pure composite equals itself; hash deterministic across calls.
    - Test (lambda machinery, E): two maps (and two reduces) with distinct lambda-param InputParams are ValueEqual false (conservative); lambda body recursion terminates under a body cycle closed via MutContext.AddUse (cycle guard, fail-safe false).
    - Test (A — amended EffectOf): EffectOf returns Full for EVERY FamilyMemory kind built with nil ImpactBy (MemLoad, MemStore, MemNew, MemLoadSMem, MemStoreSMem, MemNewSMem, FragLoad, FragStore, FragNew) and for Return; Free for scalar-expr, map, reduce, dimvar/bdimvar kinds. Existing effect_test.go pins updated accordingly.
    - Test (A audit): internal/ir, internal/opt/dce, internal/opt/schedule suites all green after the verdict change; if no existing DCE test pins that an unreferenced memory op (e.g. an SMEM store off the root paths) is still swept, add that pin here.
  </behavior>
  <action>
    Create internal/ir/value.go (new file, package ir):
    - `const ValueHashOneOfAKind uint64 = ^uint64(0)` — the NOT-INTERESTED sentinel ([3].3).
    - `type SideEffectAnalysis interface { HasSideEffect(n Node) bool }` — the query face the accumulated analysis (internal/analysis, per F) implements. Defined in ir to avoid an import cycle (analysis imports ir); documented as such.
    - Methods on `*Base` (promoted to every concrete node; the concrete node is reachable as `b.ssa.node`), IDENTITY-KEYED per amendment D:
      - `func (b *Base) ValueHash(se SideEffectAnalysis) uint64`:
        1. One-of-a-kind gate: `se == nil` (fail-closed: no analysis ⇒ no GVN), `se.HasSideEffect(node)`, or invalid kind (`KindInvalid || k >= KindEnd`) ⇒ return `ValueHashOneOfAKind`.
        2. `KindInputParam` atom ⇒ FNV-1a-style mix of kind + node ID.
        3. Composite ⇒ simple hash combiner over: kind; per-kind content fields via in-package type switch (constNode: Type + `math.Float64bits(val)`; binOpNode.op; unaryOpNode.op; castNode.typ; callNode.name; LIR memory nodes: StorageClass); then each operand slot's DEF NODE ID in canonical slot order (live records only); then each payload use's TARGET NODE ID in payload order; then per lambda: param target IDs + the body's recursive hash. NEVER recurse structurally into operands (D) — the only recursion is through lambda bodies (the shared Map/Reduce machinery, E), with a cycle guard contributing 0 on re-entry.
        4. If the computed hash equals the sentinel, perturb (`h ^= 0x9e3779b97f4a7c15`) so no pure node collides with it.
        - Types: hash interned `*NumericType` by content (elem + dims, mirroring typeContentHash in types.go); non-NumericType Type implementations contribute a constant (future components must add theirs; ValueEqual still guards via Type.Equal).
      - `func (b *Base) ValueEqual(other Value, se SideEffectAnalysis) bool`:
        1. nil other, se == nil, either side effectful per se, either side invalid kind ⇒ false ([3].3: side-effect nodes are never equal, even to themselves).
        2. Atoms (InputParam): node-pointer identity.
        3. Composites: same concrete kind; content fields equal (Type.Equal for types — never panics, see types.go; `math.Float64bits` equality for const values — keeps the hash/equality contract for ±0.0; opcode equality; name equality); same operand count; operand defs pairwise NODE-IDENTICAL (`u.Def()` pointer/Base identity — the representative, post-RAUW); payload targets pairwise identical; lambda bearers (Map AND Reduce, E): same param count, params pairwise identical, bodies recursively equal via the SAME lambda-comparison machinery. Cycle guard on body recursion: in-progress pair set, re-entry ⇒ false.
      - Document the CONTRACT on both methods: `ValueEqual(a,b) == true` MUST imply `ValueHash(a) == ValueHash(b)` (gvn-verify relies on it); hash is best-effort, collisions resolved by ValueEqual; identity keying (D) is what makes pure nodes with side-effectful operands safely GVN-eligible.
    internal/ir/node.go — extend the Value interface:
    ```go
    type Value interface {
        Node
        IsNone() bool
        ValueHash(se SideEffectAnalysis) uint64
        ValueEqual(other Value, se SideEffectAnalysis) bool
    }
    ```
    Update the Value doc comment: the two methods are the GVN value-numbering
    surface (quick-261018-nga, user decision) and take the accumulated
    side-effect analysis as input per spec [4].2; note this widens Value
    deliberately while Node stays at two methods (directive 4).
    internal/ir/effect.go — AMENDED VERDICTS (A) + audit:
    - `EffectOf`: return `NewEffectFull()` for every FamilyMemory kind (all nine: MemLoad, MemStore, MemNew, MemLoadSMem, MemStoreSMem, MemNewSMem, FragLoad, FragStore, FragNew — easiest via `k.Family() == FamilyMemory`) plus `KindReturn`; everything else Free. Verdicts stay node-local/self-scoped ([4].3: still no lattice meet, no accumulation).
    - Rewrite the file doc paragraph that claimed "SMEM/FRAG stores and loads are FREE ... dead-code-eligible": state the single-slot memory model (each memory region is one slot; subscripts cannot disprove load/write aliasing; all ops on one memory node serialize), and note that DCE dead-code eligibility is governed by two-root-set reachability (R3), NOT by EffectOf — unreferenced memory ops remain sweepable.
    - Update effect_test.go verdict pins; then AUDIT consumers: re-run `go test ./internal/...` and specifically confirm (i) scheduler call sites 463/511 are belt-and-braces after `licmHoistableKinds` (memory kinds are not whitelisted, so hoisting behavior is unchanged); (ii) DCE (opt/dce + ir's build-time sweep) is root-reachability based and unaffected — add the SMEM-store sweep pin test if the suite lacks one; (iii) Materialize/isEffectKind and legality do not consult EffectOf (grep-verified true; state it in the commit-worthy audit note).
    AVOID: hashing names of anything except callNode.name (names are debug payload, P1R-01); recursive structural equality over operands (D forbids it); making the hash security-sensitive (FNV only, cf. the T-1-01-04 note in types.go); any lattice-meet/accumulation inside EffectOf.
  </action>
  <verify>
    <automated>cd /home/ps/Workspace/cuda/project/tcacomp && go build ./... && go test ./internal/ir/ ./internal/opt/dce/ ./internal/opt/schedule/</automated>
  </verify>
  <done>Value interface carries both identity-keyed methods; amended EffectOf verdicts green with DCE/scheduler proven unaffected; all pre-existing internal/ir tests pass (interface widening breaks nothing — concrete nodes satisfy it via *Base promotion).</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: accumulated side-effect analysis in internal/analysis (order-sensitive closure)</name>
  <files>internal/analysis/sideeffect.go, internal/analysis/sideeffect_test.go</files>
  <behavior>
    - Test (A): every FamilyMemory kind is effectful with NIL ImpactBy — including plain unchained MemLoad, MemLoadSMem, FragLoad, and the MemNew/FragNew/MemNewSMem allocations (fresh-object rationale).
    - Test rule [2].1: an env-chained access (ImpactBy present) is effectful (subsumed by A for current bearers; keeps the hook for future effectBearers).
    - Test rule [2].3: Iter, IterCluster, ForLoop, Return are effectful.
    - Test (C — pure kinds never propagate): a binop over an ITER node's value stays PURE (GVN-eligible); a cast/select/call chain over effectful operands stays pure; a Map piping a MemLoad operand stays pure; a Map whose lambda body references effectful nodes stays pure (body children do not taint a pure-kind bearer).
    - Test (C — order-sensitive propagation machinery): a memory/loop kind with an effectful child reports effectful (trivially true via base today; the clause exists for future collectives — documented vacuity pinned by determinism: two Analyze runs agree and the result equals the base set on standard fixtures).
    - Test: HasSideEffect on an erased node or a nil node returns true (conservative); Analyze(nil) yields an all-conservative result.
  </behavior>
  <action>
    Create internal/analysis/sideeffect.go (package analysis — the EXISTING package per F; its only new import is internal/ir, no cycle: ir never imports analysis):
    - `type SideEffects struct{ ... }` — unexported `map[ir.Node]bool`; implements BOTH `Analysis` (`Name() string` → `"side-effect"`, opt.Cache compatibility) and `ir.SideEffectAnalysis` (`HasSideEffect(ir.Node) bool`). Unknown/erased/nil nodes query true (fail-closed).
    - `func Analyze(g *ir.Graph) SideEffects` — read-only over live nodes (`g.AllocatedNodes()`); never mutates (the analysis package contract).
    - Base (self) verdicts:
      1. `len(ir.ImpactBy(n)) > 0` ⇒ effectful ([2].1, future-bearer hook);
      2. every FamilyMemory kind ⇒ effectful ([2].2 as amended by A — loads included, nil ImpactBy included; the fresh-object rationale for MemNew* documented per G);
      3. implicit control flow: `KindIter, KindIterCluster, KindForLoop, KindReturn` ⇒ effectful ([2].3; cluster/forloop/return are the conservative closure of "Iter node").
    - Accumulated closure ([2].4 as amended by C): `effectful(n) ||= orderSensitive(n) && ∃ effectful child`, over children = `ir.DataDeps(n)` ∪ `ir.LambdaBodies(n)`. `orderSensitive` = FamilyMemory ∪ FamilyLoop ∪ {KindReturn} ∪ a documented hook for future collective kinds. Pure kinds (ScalarExpr family, Map, Reduce, Const, Param, InputParam, DimVar/BoundedDimVar) NEVER propagate — address/index math over iter values must stay GVN-eligible. Memoized DFS with three states; in-progress re-entry ⇒ effectful (conservative, cycle-safe, mirroring the scheduler's pinnedMemo pattern). O(N+E), deterministic.
    - File doc comment: contrast with `ir.EffectOf` (node-local structural verdict — now also memory-effectful per A) vs THIS accumulated GVN gatekeeper whose taint is order-sensitive-only (C); document why the propagation clause is future-arming (base ⊇ order-sensitive today).
    Follow the repo conventions: small file, comprehensive doc comments citing the spec/amendment items, stdlib + internal/ir imports only.
  </action>
  <verify>
    <automated>cd /home/ps/Workspace/cuda/project/tcacomp && go build ./... && go test ./internal/analysis/</automated>
  </verify>
  <done>Accumulated analysis green on all amended-rule tests (memory-effectful-with-nil-ImpactBy, pure-kinds-never-propagate); implements both interfaces; no IR mutation.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: GVN pass + GVN verifier pass, registered into the default pipeline</name>
  <files>internal/opt/gvn/gvn.go, internal/opt/gvn/verify.go, internal/opt/gvn/gvn_test.go, internal/opt/all/all.go, internal/opt/registry.go</files>
  <behavior>
    - Test gates: Run(nil,nil) errors "gvn: nil graph"; a materialized graph is refused with "materialized" in the message and left untouched (node count unchanged) — mirror dce tests.
    - Test basic CSE (shared operands): diamond `((a*b) + (a*b))` over the SAME a,b nodes — the duplicate mul is erased, every former user's operand Def() resolves to the surviving representative, node count drops by exactly 1.
    - Test cascade (identity keying + RAUW): C1, C2 two equal consts; B1=binop(C1,C1), B2=binop(C2,C2) — gvn merges C2 into C1 AND then B2 into B1 in one run (B2's records were rewired to C1 by the first RAUW, making the binops identity-equal).
    - Test leaf pooling: two NewConst(f32,2) dedup; two paramRefs to the same param dedup; differing const values / opcodes do NOT dedup.
    - Test iter-arith (C+D, the amendment's named case): `i*2` built twice over the SAME iter node i — the two binops dedup, the iter node itself survives untouched; `i1*2` vs `i2*2` over DISTINCT iter nodes — NO merge, both survive.
    - Test side-effect safety ([2].5/[4].7 as amended): two structurally identical loads against one tensor (nil ImpactBy, same tensor/axes) do NOT merge (A+B — memory read-read CSE is out of scope); two env-chained stores with identical tensor/axes/value BOTH survive; two identical NewIter survive; two identical MemNew survive.
    - Test (E — Reduce participates): a reduce node survives gvn untouched and is NOT gated out; a reduce piping a duplicated pure value gets its piped operand rewired to the representative (participation observed through its operand); two reduces with identical init/piped operands but distinct combiner-param atoms do NOT merge (conservative lambda identity).
    - Test lambda-body-root skip: a node structurally equal to a map's lambda body root survives (body roots are one-of-a-kind; interiors still dedup — pin a duplicate binop INSIDE a shared body context merging).
    - Test idempotence: a second Run changes nothing (node count and print output stable).
    - Test gvn-verify: green after gvn on every fixture; RED — a graph holding two pure ValueEqual nodes WITHOUT gvn run makes VerifyPass.Run error; RED — Manager pipeline of only [gvn-verify] fails the "requires gvn" precondition.
    - Test pipeline wiring: Registered() orders dce(100) < dce-verify(101) < gvn(110) < gvn-verify(111) < schedule(900); BuildPipeline under opt/all contains gvn and gvn-verify.
  </behavior>
  <action>
    Create internal/opt/gvn/gvn.go (package gvn, structural mirror of internal/opt/dce/dce.go):
    - `func init()` — `opt.Register(opt.PassSpec{ID: "gvn", Priority: 110, Factory: func() opt.Pass { return Pass{} }})`.
    - `type Pass struct{}` — ID "gvn"; Requires nil (usable standalone; the registry priority supplies default order); Produces `opt.StageConstructed` (erases/rewires only, introduces no new kinds); Preserved nil (invalidate all).
    - `Run(_ *opt.Context, g *ir.Graph) error`:
      1. Gates BEFORE any mutation: nil graph ⇒ `errors.New("gvn: nil graph")`; `g.Materialized() || g.CFG() != nil` ⇒ `errors.New("gvn: graph already materialized (gvn runs on the Level-1 sea)")`.
      2. `se := analysis.Analyze(g)` (import `tcacomp/internal/analysis`, per F).
      3. Precompute bodyRoots: for every allocated node, mark `ir.LambdaBodies(n)` — the one non-SSAUse edge; RAUW cannot rewire `lam.Body`, so body roots are one-of-a-kind for GVN (design decision 6; document the upgrade path). Body interiors still dedup normally.
      4. `table := make(map[uint64][]ir.Node)`; `mctx := &ir.MutContext{Pass: "gvn"}`.
      5. For each n in `g.AllocatedNodes()` (allocation order; deterministic, lowest-ID duplicate survives):
         - skip if bodyRoots[n];
         - `v, ok := n.(ir.Value)`; `!ok` ⇒ error (foreign node — fail loudly);
         - `h := v.ValueHash(se)`; `h == ir.ValueHashOneOfAKind` ⇒ continue ([2].5: side-effect nodes do not participate — with A this excludes every memory op, loop node, and the end node);
         - scan bucket `table[h]`: on `e.(ir.Value).ValueEqual(v, se)` (identity-keyed per D — a match requires pairwise-identical operand defs):
           * ITEM-7 HARD ASSERT (G, kept): `if se.HasSideEffect(n) || se.HasSideEffect(e)` ⇒ return error naming both nodes ("gvn: side-effect node %d (%s) would be replaced by %d (%s)") — unreachable unless the sentinel gate and the analysis disagree, i.e. exactly the drift worth failing on;
           * `mctx.ReplaceAllUsesWith(n, e)` — every reference moves via the SSAUse ref records ([4].5); this is also what makes the Task-3 cascade test work (users of the erased duplicate become identity-equal to the representative's users);
           * `if uses := ir.UsesOf(n); len(uses) > 0` ⇒ return error (post-RAUW bookkeeping corruption); else `mctx.Erase(n)` ([4].6: CSE performed along the way);
           * break;
         - if no match: insert n into the bucket (the representative — its SSAVar IS the recorded SSA, [4].5).
    Create internal/opt/gvn/verify.go (mirror of internal/opt/dce/verify.go):
    - Register ID "gvn-verify", Priority 111, `Requires: []string{"gvn"}`, Produces StageConstructed, Preserved nil.
    - `VerifyPass.Run` — mutation-free, same gates, fresh `se := analysis.Analyze(g)` + fresh bodyRoots. Three checks, first failure returns a diagnostic naming nodes (IDs + kinds):
      1. DANGLING: every use of every allocated node (`ir.UsesOf`) resolves to a non-erased def with a non-nil user (post-RAUW/Erase SSA integrity).
      2. ANALYSIS-CLOSURE (as amended by C): the accumulated analysis is a fixpoint over the actual graph — every PURE node has no base verdict AND is not an order-sensitive kind with an effectful child (`ir.DataDeps` ∪ `ir.LambdaBodies`); every EFFECTFUL node has a base verdict OR is order-sensitive with an effectful child.
      3. UNIQUE-REPRESENTATIVE ([4].7 post-hoc): over participating nodes (allocated, valid kind, not bodyRoot, pure per se) full pairwise — no two distinct nodes are ValueEqual. O(N²) documented as fine at v0 kernel scale; stronger than bucket-limited checking because it also catches hash/equality contract drift.
    internal/opt/all/all.go — add `_ "tcacomp/internal/opt/gvn"`.
    internal/opt/registry.go — extend the priority-scale comment: `110 global CSE (gvn), 111 gvn-verify — between early cleanup and late scheduling`. Comment only.
    internal/opt/gvn/gvn_test.go — external package `gvn_test` following dce_test.go conventions (fixture builders returning named nodes, exact gate-error strings, nodeSet-style survivors assertions, ir.NewEnv-chained construction).
  </action>
  <verify>
    <automated>cd /home/ps/Workspace/cuda/project/tcacomp && go build ./... && go vet ./... && go test ./internal/...</automated>
  </verify>
  <done>GVN dedups pure computations with identical SSA operands globally and rewires all references through SSAUse records; memory ops (loads included), loops, the end node, and everything the order-sensitive closure taints are provably never replaced (assert + verifier + tests); distinct-operand lookalikes never merge; gvn/gvn-verify registered at 110/111 and included in BuildPipeline; full internal tree green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none) | Pure in-process compiler transformation over internally constructed IR; no external input crosses a trust boundary in this change |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-nga-01 | Tampering | ValueHash/ValueEqual | accept | FNV mixing is not collision-resistant by design (cf. T-1-01-04 note in types.go); correctness never relies on hash uniqueness — every bucket hit is confirmed by deep ValueEqual |
| T-nga-02 | DoS | gvn/gvn-verify | accept | O(N²) verify pairwise and lambda-body hash recursion bounded by graph size; graphs are kernel-sized; cycle guards prevent infinite recursion on malformed input |
| T-nga-03 | Tampering | side-effect gate | mitigate | Fail-closed defaults: nil analysis ⇒ one-of-a-kind; unknown/erased nodes ⇒ effectful; during-pass assert aborts the pipeline if a side-effect node could ever be replaced |
| T-nga-04 | Tampering | EffectOf verdict change (A) | mitigate | Regression risk contained by the audit: DCE is root-reachability based (pinned by its suite), scheduler call sites are belt-and-braces behind the hoist whitelist, legality/Materialize do not consume EffectOf — all re-run in Task 1's verify |
</threat_model>

<verification>
- `go build ./...` — clean compile (interface widening breaks no implementor: all in-repo nodes satisfy Value via *Base promotion; grep confirmed no external ir.Value implementors).
- `go vet ./...` clean.
- `go test ./internal/...` — all packages green, including pre-existing ir/opt suites (goldens, DCE, scheduler unaffected by the amended EffectOf verdicts — proven in Task 1's audit run; print paths untouched).
- Manual sanity: `go test ./internal/opt/all/ -v` shows the default pipeline containing dce, dce-verify, gvn, gvn-verify, schedule in priority order.
</verification>

<success_criteria>
- Spec items [1]–[8] as amended by decisions A–G delivered at full fidelity (see matrix in objective); amendment-only rows (B, D, E) pinned by their named tests.
- A graph with duplicate pure computations over shared SSA operands collapses them (cascade included); lookalikes over distinct defs (two iter nodes) never merge; duplicate loads/stores/iters/allocations never merge.
- `i*2` twice over one iter dedups; `i1*2` vs `i2*2` does not (the amendment's named case).
- EffectOf marks every FamilyMemory node node-local effectful with DCE + scheduler suites green (audit recorded).
- gvn-verify detects: remaining duplicates, non-fixpoint analysis, dangling uses.
- Default pipeline includes both new passes; stdlib-only constraint intact (hash/fnv or inline FNV, math — all stdlib); no new go.mod entries.
- Files follow repo conventions: layered small files, doc comments citing spec/amendment items, external _test packages, exact-error gate tests.
</success_criteria>

<output>
After completion, create `.planning/quick/261018-nga-cse-via-gvn-side-effect-analysis-global-/261018-nga-SUMMARY.md`
</output>
