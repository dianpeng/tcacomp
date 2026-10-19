---
phase: quick-261019-uoq
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/analysis/sideeffect/sideeffect.go
  - internal/analysis/sideeffect/sideeffect_test.go
  - internal/opt/gvn/verify.go
  - internal/opt/gvn/gvn.go
  - internal/opt/gvn/gvn_test.go
  - internal/ir/value.go
  - internal/analysis/loopforest/loopforest.go
  - internal/analysis/loopforest/loopforest_test.go
  - internal/analysis/all/all.go
  - internal/analysis/all/all_test.go
autonomous: true
requirements: [LF-1, LF-2, LF-3, LF-4, LF-5, LF-6]

must_haves:
  truths:
    - "Two live Iter nodes whose kind and operand-slot def identities match (same dim, same bound after const CSE, same init, same step, same parent chain) FUSE under one gvn run: the duplicate is erased, every former user's operand resolves to the survivor, and gvn-verify passes"
    - "A bare Iter (no effectful operand) queries PURE from sideeffect.Analyze; an Iter with an accumulated-effectful operand (e.g. a MemLoad init) queries EFFECTFUL and never enters GVN"
    - "IterCluster, ForLoop, and Return remain base-effectful; loads/stores/allocations still never merge"
    - "Nested iters fuse only whole-nest: two identical parent-wired chains fuse in one run; inner iters wired to DIFFERENT parents never fuse; unbounded (3-operand) and bounded (4-operand) iters never fuse"
    - "loopforest.Analyze returns a deterministic forest over live KindIter nodes: parent = first live-Iter operand (doc's reversed edge), roots and children in allocation order, depth from roots; nil graph yields an empty forest; erased iters are skipped; nothing is mutated"
    - "After importing internal/analysis/all, analysis.Lookup(\"loop-forest\") succeeds and BuildAll constructs the forest under that name"
  artifacts:
    - path: "internal/analysis/sideeffect/sideeffect.go"
      provides: "KindIter removed from baseVerdict (side-effect free per doc/optana/loop_fusion.md); package comment updated: the propagation clause is now LIVE for Iter"
      contains: "KindIterCluster, ir.KindForLoop, ir.KindReturn"
    - path: "internal/opt/gvn/verify.go"
      provides: "baseVerdictMirror drops KindIter in the same commit — ANALYSIS-CLOSURE stays a fixpoint"
      contains: "KindIterCluster, ir.KindForLoop, ir.KindReturn"
    - path: "internal/opt/gvn/gvn_test.go"
      provides: "inverted fusion pins (TestGVNIterArithmetic 2nd half, TestGVNNeverMergesSideEffects) + new TestGVNIterFusionBounded + TestGVNIterFusionNestedAwareness"
      contains: "TestGVNIterFusion"
    - path: "internal/analysis/loopforest/loopforest.go"
      provides: "Forest/Tree types, Analyze per the doc's forest algorithm, Name() loop-forest, init self-registration"
      exports: ["Tree", "Forest", "Analyze", "Roots", "NodeOf"]
      min_lines: 60
    - path: "internal/analysis/loopforest/loopforest_test.go"
      provides: "flat/nested/two-nest/erased/nil/determinism/read-only/name pins + post-GVN single-survivor integration"
      min_lines: 100
    - path: "internal/analysis/all/all.go"
      provides: "blank import firing loopforest registration"
      contains: "analysis/loopforest"
  key_links:
    - from: "internal/opt/gvn/gvn.go Run"
      to: "sideeffect.Analyze gate"
      via: "se.HasSideEffect(iter) now false for bare iters — iters enter the identity-keyed table"
      pattern: "sideeffect\\.Analyze"
    - from: "internal/opt/gvn/verify.go ANALYSIS-CLOSURE"
      to: "internal/analysis/sideeffect baseVerdict"
      via: "baseVerdictMirror must equal baseVerdict — KindIter dropped in BOTH in one commit"
      pattern: "baseVerdictMirror"
    - from: "internal/analysis/loopforest init()"
      to: "internal/analysis registry"
      via: "analysis.Register(AnalysisSpec{Name: \"loop-forest\", ...})"
      pattern: "loop-forest"
    - from: "internal/analysis/all/all.go"
      to: "internal/analysis/loopforest"
      via: "blank import so registration fires"
      pattern: "opt|analysis/loopforest"
---

<objective>
Implement loop fusion via GVN per doc/optana/loop_fusion.md, plus a standalone loop-nest-forest analysis (the future consumer is loop tiling, doc/optana/loop_tiling.md).

The fusion mechanism already exists: gvn.go is fully generic (identity-keyed ValueHash/ValueEqual over kind + content + operand-slot def identities; RAUW cascade). Iter nodes fail to fuse ONLY because `sideeffect.baseVerdict` lists KindIter as base-effectful. The doc says: "The Iter node is itself side effect free". THE change is the side-effect gate reclassification — remove `ir.KindIter` from the base verdict (keep IterCluster/ForLoop/Return; keep FamilyLoop in `orderSensitive` so an Iter with an effectful operand still taints out of GVN). No gvn.go algorithm change.

Purpose: unlock loop fusion (identical loop headers collapse; equal-valued bounds CSE first; whole nests fuse through parent-operand identity) and provide the forest analysis that loop tiling will consume.
Output: reclassified side-effect analysis + inverted/pinned GVN tests + new `internal/analysis/loopforest` package wired into the analysis registry and composition root.

GVN soundness verdict was already established by the orchestrator: SOUND (no GVN restructuring; only the gate reclassification + mirror sync).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@doc/optana/loop_fusion.md
@doc/optana/loop_tiling.md
@internal/analysis/sideeffect/sideeffect.go
@internal/opt/gvn/gvn.go
@internal/opt/gvn/verify.go
@internal/opt/gvn/gvn_test.go
@internal/analysis/sideeffect/sideeffect_test.go

<interfaces>
<!-- Key contracts the executor needs — extracted from the codebase. Do not re-derive. -->

From internal/ir (all exported, stable):
```go
func (g *Graph) NewIter(origin string, dim, bound Node, init, step Node) Node
  // bound == nil => 3-operand node [dim, init, step]; bound != nil => 4-operand [dim, bound, init, step]
func (g *Graph) NewNode(k Kind, origin string, operands ...Node) Node // generic constructor — use for parent-wired iters
func DataDeps(n Node) []Node   // live operand-slot defs in slot order (skips killed/unwired records)
func OperandsOf(n Node) []Node // operand defs in slot order, nil for unwired
func UsesOf(n Node) []SSAUse
func (g *Graph) AllocatedNodes() []Node // allocation order = def-before-use topological for builder graphs
func ImpactBy(n Node) []SSAUse
func Verify(g *Graph) []error
type MutContext struct{ Pass string } // Erase / ReplaceAllUsesWith choke points
// KindIter: TierHIR, FamilyLoop. KindIterCluster/KindForLoop also FamilyLoop.
// ir/verify.go pins ONLY operand 0 of an Iter to a dim InputParam — extra (parent) operands appended after the standard slots are legal.
// ir.EffectOf (node-local) already says Iter effect-FREE — do not touch it.
```

From internal/analysis (registry pattern, mirrors internal/opt):
```go
type Analysis interface { Name() string } // read-only contract: never mutate the IR
type AnalysisSpec struct {
    Name    string
    Factory func(g *ir.Graph) Analysis // must tolerate nil graph (Register cross-checks Factory(nil).Name())
}
func Register(spec AnalysisSpec)          // panics on duplicate/empty/nil/mismatched factory
func Lookup(name string) (AnalysisSpec, bool)
func BuildAll(g *ir.Graph) map[string]Analysis
```
Self-registration precedent (sideeffect.go:46-51):
```go
func init() {
    analysis.Register(analysis.AnalysisSpec{
        Name:    "side-effect",
        Factory: func(g *ir.Graph) analysis.Analysis { return Analyze(g) },
    })
}
```

From internal/opt/gvn:
```go
func (Pass{}).Run(_ *opt.Context, g *ir.Graph) error       // the dedup; processes AllocatedNodes() in order
func (VerifyPass{}).Run(_ *opt.Context, g *ir.Graph) error // ANALYSIS-CLOSURE re-derives base/order-sensitive tables
// gvn.go:139-142 already hard-asserts post-RAUW ZERO remaining uses before Erase — the doc's
// "all iter nodes in Prefix(A) must have ZERO USE" verification requirement is covered by this.
```

Test helpers in gvn_test.go (reuse): scaffold, runGVN, runGvnVerify, assertGone, assertKept, nodeSet.
Test helper in sideeffect_test.go (reuse): fixture() returns (g, m, out, alpha, c1); mustAnalyze.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Reclassify KindIter as side-effect free — GVN loop fusion (RED first, then GREEN)</name>
  <files>internal/analysis/sideeffect/sideeffect_test.go, internal/analysis/sideeffect/sideeffect.go, internal/opt/gvn/verify.go, internal/opt/gvn/gvn_test.go, internal/opt/gvn/gvn.go, internal/ir/value.go</files>
  <behavior>
  Analysis-level (sideeffect_test.go):
  - Bare Iter (no effectful operands) queries PURE: HasSideEffect(iter) == false
  - IterCluster, ForLoop, Return still query EFFECTFUL (base verdict unchanged for them)
  - Iter with an accumulated-effectful operand (init = a MemLoad) queries EFFECTFUL (orderSensitive(KindIter) via FamilyLoop + effectful child) — the fusion soundness gate
    - NEW test TestIterTaintsThroughEffectfulOperand pins this
  GVN-level (gvn_test.go):
  - Two identical NewIter over the same dim/init/step const nodes: duplicate erased, consumers rewired (TestGVNIterArithmetic 2nd half INVERTS)
  - Two bounded iters whose bounds are equal-valued but DISTINCT const nodes: bounds CSE first, then the iters fuse, in ONE run
  - Unbounded (3-operand) vs bounded (4-operand) iter over the same dim/init/step: NEVER fuse (slot-count mismatch)
  - Two identical parent-wired nests ([dim, init, step, parent] with parent appended LAST): consts CSE, parents fuse, children fuse through the rewired parent operand — whole nest in one run
  - Inner iters sharing dim/init/step but wired to DIFFERENT parents (parents differ by step const): NEVER fuse
  - Loads, stores, allocations still never merge; st-chain ImpactBy untouched (TestGVNNeverMergesSideEffects with iters moved out of the exclusion set)
  </behavior>
  <action>
  STEP A — RED (invert and extend the tests first; commit when the new expectations fail):

  1. internal/analysis/sideeffect/sideeffect_test.go:
     - TestImplicitControlFlowEffectful (lines 103-127): SPLIT. Keep itercluster/forloop/return effectful. Move `iter` out and pin it PURE — either a new TestIterIsPure or restructure in place; cite doc/optana/loop_fusion.md [1] ("The Iter node is itself side effect free").
     - TestPureKindsNeverPropagate fixture guard (lines 151-154): `if !se.HasSideEffect(iter) || !se.HasSideEffect(load)` currently asserts iter effectful — INVERT to assert iter PURE (keep load effectful). The body/comment "references the effectful iter" needs rewording; the map/binop/cast/select/call pure assertions are unchanged.
     - TestOrderSensitivePropagation vacuity pin (lines 204-213): the `want` formula includes `n.KindOf() == ir.KindIter` as base-effectful — REMOVE KindIter from the formula (it is no longer base-effectful; on this fixture no Iter has an effectful child, so accumulated == base-set still holds). Update the test comment: the propagation clause is no longer vacuous in general — it is live for Iter-with-effectful-operand.
     - ADD TestIterTaintsThroughEffectfulOperand: `load := g.NewMemLoad("ctor", out, []ir.Node{m})`; `it := g.NewIter("ctor", m, nil, load, step)` (init operand is a MemLoad — structurally legal, the analysis does not type-check). Assert `se.HasSideEffect(it)` is TRUE and a bare iter (init = plain const) is FALSE. This pins that the taint propagation (orderSensitive FamilyLoop) keeps effectful-fed iters out of GVN.

  2. internal/opt/gvn/gvn_test.go:
     - File header comment (lines 19-22): iters move from the "never merge" list to the fusion list — cite doc/optana/loop_fusion.md.
     - TestGVNIterArithmetic second half (g2 graph, lines 276-291): INVERT. After runGVN: `assertGone(t, g2, i2, x2)`; `assertKept(t, g2, i1, x1, cons2)`; cons2's operands must both resolve to x1. (First half — b1/b2 over the SAME iter — is unchanged.)
     - TestGVNNeverMergesSideEffects (lines 298-343): move it1/it2 OUT of the exclusion set. New expectations: `assertGone(t, g, it2)`, `assertKept(t, g, it1)`; keep `assertKept(t, g, load1, load2, st1, st2, alloc1, alloc2)`; node count must drop by exactly 1 (`after != before-1` fails); keep the st2 ImpactBy→st1 chain assertion and runGvnVerify. (it1/it2 have no users in this fixture — post-RAUW zero-use assert at gvn.go:139-142 is trivially satisfied.)
     - ADD TestGVNIterFusionBounded — two graphs in one test (match the file's existing style):
       (a) b1,b2 := two Const(f32,64); it1 := NewIter(m, b1, zero, one); it2 := NewIter(m, b2, zero, one); consumer := BinOp(Add, it1, it2); scaffold. After runGVN: assertGone(b2, it2); assertKept(b1, it1, consumer); consumer operands both it1; ir.Verify clean; runGvnVerify. (Bounds CSE first, iters fuse in the same run — pins "DimVar and Bounds are both equal" via equal-valued const CSE.)
       (b) itU := NewIter(m, nil, zero, one) (3 operands); itB := NewIter(m, bnd, zero, one) (4 operands, bnd a const); assertKept(itU, itB) — slot-count mismatch prevents fusion.
     - ADD TestGVNIterFusionNestedAwareness — two graphs in one test. Parent-wired iters are built with the parent APPENDED as the FINAL operand via the generic constructor: `g.NewNode(ir.KindIter, "ctor", dim, init, step, parent)` → [dim, init, step, parent]. (Slot 1 is the bound in the current layout; the doc's "second input" phrasing predates the operand-layout revision. Position is not load-bearing — GVN keys on full slot identity and the forest scans all operands — but the tests must use ONE consistent layout: parent last.)
       (a) Whole-nest fusion: z1,z2 := Const(f32,0) pair; s1,s2 := Const(f32,1) pair; p1 := NewIter(m, nil, z1, s1); p2 := NewIter(m, nil, z2, s2); c1 := NewNode(KindIter, n, z1, s1, p1); c2 := NewNode(KindIter, n, z2, s2, p2); scaffold over a consumer of c1/c2. Allocate in def-before-use order (consts, parents, children). After ONE runGVN: assertGone(z2, s2, p2, c2); assertKept(z1, s1, p1, c1); consumer operands resolve to c1; runGvnVerify. (Const CSE → parents identity-equal → p2 erased → c2's parent slot rewired to p1 → children identity-equal → c2 erased: the RAUW cascade fuses the whole nest in one run.)
       (b) Different parents block inner fusion: sA := Const(f32,1), sB := Const(f32,2) (distinct values, never CSE); pA := NewIter(m, nil, z, sA); pB := NewIter(m, nil, z, sB); childA := NewNode(KindIter, n, z, sA, pA); childB := NewNode(KindIter, n, z, sA, pB) — children share dim/init/step, differ ONLY in the parent slot. After runGVN: assertKept(pA, pB, childA, childB); runGvnVerify. (Pins nested awareness: inner equality requires the whole parent chain equal.)
     Run: `go test ./internal/analysis/sideeffect/... ./internal/opt/gvn/...` — the inverted/new tests FAIL. Commit: `test(quick-261019-uoq): invert iter side-effect pins to GVN fusion semantics per doc/optana/loop_fusion.md (RED)`.

  STEP B — GREEN (the reclassification; ALL in one commit so the ANALYSIS-CLOSURE mirror never drifts):

  3. internal/analysis/sideeffect/sideeffect.go:
     - baseVerdict switch (lines 178-183): remove `ir.KindIter` — becomes `case ir.KindIterCluster, ir.KindForLoop, ir.KindReturn:` with the comment updated: Iter reclassified side-effect free per doc/optana/loop_fusion.md ("The Iter node is itself side effect free"); IterCluster/ForLoop keep the conservative closure ([2.3]).
     - Package comment: rewrite the "FUTURE-ARMING ... today base ⊇ order-sensitive ... the closure adds nothing on real graphs" paragraph (lines 28-34) — after the reclassification the propagation clause is LIVE: KindIter is order-sensitive (FamilyLoop) but NOT base-effectful, so an Iter with an effectful operand accumulates effectful and stays out of GVN (the fusion soundness gate), while a bare Iter is pure and GVN-eligible.
     - `orderSensitive`: NO CHANGE — the FamilyLoop family check already covers Iter (that is the taint path).
  4. internal/opt/gvn/verify.go: baseVerdictMirror switch (lines 159-163): drop `ir.KindIter` in the SAME commit — becomes `case ir.KindIterCluster, ir.KindForLoop, ir.KindReturn:`. Without this, gvn-verify's ANALYSIS-CLOSURE check fails loudly on every graph holding a bare iter (mirror base=true vs accumulated=false). The mirror's independence is the point: it must be edited in lockstep, never trusted to auto-derive. Update its comment likewise.
  5. internal/opt/gvn/gvn.go: DOC-COMMENT ONLY (no code change). On the Pass doc (or Run doc), add: Iter nodes participate in value numbering via identity keying with nested awareness — two iters are one value iff kind, operand count, and every operand-slot def identity match (dim, bound, init, step, and any parent-Iter operands appended for nested loops), so equal loop headers fuse and whole nests fuse through the parent-operand RAUW cascade; effectful-fed iters stay excluded via the order-sensitive taint. Cite doc/optana/loop_fusion.md.
  6. internal/ir/value.go: COMMENT ONLY. The identity-keying doc (lines 16-19) says "Two one-of-a-kind defs with identical fields (two iters, two allocations) therefore never merge their consumers" — the "(two iters" example is now stale (iters with matching operand identities DO merge). Swap the example to "(two stores, two allocations)". No code change.

  Run `go test ./...` — ALL green. Commit: `feat(quick-261019-uoq): reclassify KindIter side-effect free — GVN loop fusion (sideeffect + gvn-verify mirror)`.

  Do NOT edit anything under doc/optana/ (user's docs). Do not touch ir/verify.go, dce.go, effect.go, schedule.go, typeinfer — verified unaffected (kind-based checks; node-local EffectOf already says Iter free; scheduler LICM whitelists scalar-expr kinds only; pipeline order already correct: simplify-lower (102) creates Iters, typeinfer (105), gvn (110) fuses).
  </action>
  <verify>
    <automated>go test ./internal/analysis/sideeffect/... ./internal/opt/gvn/... ./internal/ir/... && go test ./... && go vet ./... && gofmt -l internal/ | grep -q . && echo GOFMT-DIRTY || echo OK</automated>
  </verify>
  <done>
  - Bare iters fuse under gvn (duplicate erased, users rewired, gvn-verify green); bounded iters with equal-valued const bounds fuse; whole parent-wired nests fuse in one run; different parents / different arities block fusion
  - IterCluster/ForLoop/Return still effectful; loads/stores/allocs still never merge; Iter with effectful operand still effectful (taint pin)
  - sideeffect.go baseVerdict and verify.go baseVerdictMirror both exclude KindIter, edited in one commit; package comments accurate (no stale "base ⊇ order-sensitive" claim)
  - Full `go test ./...` green
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: loopforest analysis package + registry wiring (the loop-nest forest per the doc)</name>
  <files>internal/analysis/loopforest/loopforest_test.go, internal/analysis/loopforest/loopforest.go, internal/analysis/all/all.go, internal/analysis/all/all_test.go</files>
  <behavior>
  - Analyze(g) over live KindIter nodes only (skip erased; IterCluster/ForLoop/DimVar are NOT forest members)
  - Parent = the FIRST live-Iter operand in slot order (IR edge x→y reversed to forest edge y parent-of x); x leaves the root set
  - Roots() and each Tree.Children are in ALLOCATION order; Depth counts from roots (root = 0); two Analyze runs produce identical structure
  - Flat graphs (today's only builder shape — no iter→iter operand edges) yield all-singleton trees
  - Nil graph → empty forest (Roots() empty, NodeOf false for everything, Name() still "loop-forest"); erased iters skipped; foreign/nil nodes → NodeOf false
  - Read-only: node set and use counts identical before/after (analysis-package contract — no MutContext, no mutation)
  - The forest is a SEPARATE tree structure, never baked into IR nodes; computed on the fly per demand
  - Post-GVN integration: after gvn fuses two identical iters, the forest holds exactly one survivor tree
  - analysis/all wiring: Lookup("loop-forest") succeeds; BuildAll constructs it under "loop-forest"
  </behavior>
  <action>
  STEP A — RED: create internal/analysis/loopforest/loopforest_test.go pinning the contract (package loopforest_test; import tcacomp/internal/analysis, tcacomp/internal/analysis/loopforest, tcacomp/internal/ir; for the integration case tcacomp/internal/opt/gvn). Tests:
  - TestName: zero-value Forest Name() == "loop-forest"; a Forest from Analyze satisfies analysis.Analysis (mirror sideeffect's TestName).
  - TestFlatGraphAllSingletons: two unrelated unbounded iters (NewIter over distinct dims or the same dim — either way no iter→iter operand) → Roots() has 2 trees, both Depth 0, empty Children, NodeOf true for each; an IterCluster, a ForLoop, and a DimVar in the same graph → NodeOf false (only KindIter is collected).
  - TestNestedChain: outer := NewIter(m, nil, c0, c1); inner := g.NewNode(ir.KindIter, "ctor", n, c0, c1, outer); innermost := g.NewNode(ir.KindIter, "ctor", k, c0, c1, inner) — parent appended LAST (same layout convention as Task 1). Expect ONE root (outer); Depths 0/1/2; NodeOf(inner).Parent == NodeOf(outer); outer.Children == [inner's tree]; inner.Children == [innermost's tree].
  - TestTwoNestsTwoRoots: two disjoint outer/inner chains → Roots() has 2, each with one child.
  - TestErasedIterSkipped: two childless iters, Erase one via ir.MutContext → forest holds exactly the survivor.
  - TestNilGraphEmpty: Analyze(nil).Roots() empty; NodeOf(nil) and NodeOf(anyNode) false; Name() == "loop-forest" (Register cross-checks Factory(nil).Name()).
  - TestDeterministicAndReadOnly: run Analyze twice; render each forest to a fingerprint string (root iter IDs in order, per-tree children IDs in order, depths) and require equality; require node count and ir.UsesOf counts unchanged across Analyze.
  - TestForestAfterGVNFusion: two identical iters + a consumer; run gvn.Pass; Analyze → Roots() has exactly 1 tree and NodeOf(duplicate) is false (the duplicate was erased) — the bridge pin tying fusion to the forest invariant.
  Commit: `test(quick-261019-uoq): pin loop-forest analysis contract per doc/optana/loop_fusion.md (RED)` (build failure — package absent — is the red state).

  STEP B — GREEN: create internal/analysis/loopforest/loopforest.go (package loopforest; imports tcacomp/internal/analysis + tcacomp/internal/ir only; file stays well under the 400-line norm):

  ```go
  // Tree is one loop-nest tree node: a live KindIter plus its forest links.
  type Tree struct {
      Iter     ir.Node // the KindIter this tree node represents
      Parent   *Tree   // nil for roots
      Children []*Tree // iters that have this iter as an operand, allocation order
      Depth    int     // root = 0, child = parent + 1
  }

  // Forest is the loop-nest forest over one graph snapshot (read-only analysis).
  type Forest struct {
      trees map[ir.Node]*Tree // live iter -> its tree node
      roots []*Tree           // allocation order
  }

  func (Forest) Name() string { return "loop-forest" }
  func Analyze(g *ir.Graph) Forest
  func (f Forest) Roots() []*Tree
  func (f Forest) NodeOf(iter ir.Node) (*Tree, bool)
  ```

  Analyze implements the doc's algorithm VERBATIM (doc/optana/loop_fusion.md "Loop Nests/Forrest analysis" — note the forest section is the live spec; the doc marks the older "Loop Fusion" prose above it DEPRECATED, GVN supersedes it):
  1. Collect: iterate g.AllocatedNodes() in order; every node with KindOf() == ir.KindIter and !BasePtr().Erased() gets a Tree; record it as a root candidate in that same (allocation) order.
  2. Link (reversed edges): for each live iter x in allocation order, scan ir.DataDeps(x) in slot order; the FIRST dep y that is a live Iter collected in pass 1: set tree(x).Parent = tree(y), append tree(x) to tree(y).Children, remove x from the root set. Doc: IR edge iter(x) -> iter(y) (x nested in y) becomes forest edge y -> x, and x leaves the root set. Single-parent convention: the doc's forest assumes each nested iter has ONE parental loop riding its operand list — first-iter-operand wins, deterministic.
  3. Depth: DFS from each root in Roots() order (root 0, child = parent+1). Cycle-safe by construction: mutually-referencing iters (impossible under def-before-use allocation, defensive anyway) remove each other from the root set and are never reached by the DFS — they keep Depth 0 and the algorithm still terminates.
  4. Nil g → zero Forest (nil map, nil roots — Roots() returns nil, NodeOf false). Analyze never mutates: no MutContext, only read faces (AllocatedNodes, DataDeps, KindOf, BasePtr).

  Self-register exactly like sideeffect.go:46-51:
  ```go
  func init() {
      analysis.Register(analysis.AnalysisSpec{
          Name:    "loop-forest", // stable public id — opt.Cache key convention
          Factory: func(g *ir.Graph) analysis.Analysis { return Analyze(g) },
      })
  }
  ```
  Package doc: state the purpose (enumerate all loop nests for future consumers — loop tiling per doc/optana/loop_tiling.md), the separate-structure rule (never baked into IR nodes, computed on demand), the read-only contract, and the parent-operand convention.

  STEP C — wiring: internal/analysis/all/all.go — add the blank import `_ "tcacomp/internal/analysis/loopforest"` (accumulating composition root). internal/analysis/all/all_test.go — extend TestAllRegistersEveryAnalysis with `analysis.Lookup("loop-forest")` membership, and add a BuildAll assertion: BuildAll(ir.NewGraph("t"))["loop-forest"] exists and type-asserts to loopforest.Forest (import the package in the test).

  Run `go test ./...` — green. Commits: `feat(quick-261019-uoq): implement loopforest analysis (doc/optana/loop_fusion.md forest algorithm)` then `feat(quick-261019-uoq): wire loopforest into analysis/all composition root` (or one commit if preferred — keep conventional-commit format).

  Do NOT edit doc/optana/. Note: the registry is process-global and tests may register later — never assert exact registry size in tests (existing convention in all_test.go).
  </action>
  <verify>
    <automated>go test ./internal/analysis/... ./internal/opt/gvn/... && go test ./... && go vet ./... && gofmt -l internal/ | grep -q . && echo GOFMT-DIRTY || echo OK</automated>
  </verify>
  <done>
  - internal/analysis/loopforest exists with Tree/Forest/Analyze/Roots/NodeOf; Name() "loop-forest"; registered via init(); blank-imported by analysis/all
  - Forest correct on flat (all singletons), nested chains (reversed edges, depths), two nests, erased-iters, and nil graphs; deterministic across runs; provably non-mutating
  - Post-GVN fusion leaves exactly one survivor iter in the forest
  - all_test.go pins Lookup("loop-forest") membership and BuildAll construction
  - Full `go test ./...` green
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

None introduced. Both tasks are internal compiler analysis/pass changes over in-memory IR; no external input, no I/O, no new dependencies (pure Go stdlib, per project constraint).

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-uoq-01 | Tampering | gvn dedup of iters | mitigate | Fusion only removes PROVEN duplicates: identity-keyed ValueEqual + side-effect hard assert (gvn.go:132) + post-RAUW zero-use assert (gvn.go:139-142) + gvn-verify ANALYSIS-CLOSURE/UNIQUE-REPRESENTATIVE mirrors — all retained unchanged |
| T-uoq-02 | Tampering | sideeffect reclassification unsoundness (an effectful-fed iter entering GVN) | mitigate | orderSensitive FamilyLoop taint retained; pinned by TestIterTaintsThroughEffectfulOperand; gvn-verify mirror drop is lockstep-committed so drift fails loudly |
| T-uoq-03 | DoS | loopforest on malformed graphs (cycles, pass-mutated allocation order) | mitigate | Cycle-safe by construction (root-set removal terminates; DFS from roots never enters a cycle); nil/erased/foreign nodes fail closed (NodeOf false) |
</threat_model>

<verification>
- `go test ./...` fully green (unit + integration; e2e gate is env-guarded and unaffected)
- `go vet ./...` clean; `gofmt -l internal/` empty
- Manual spot check (executor): build two identical iters in a scratch test, run the L1 prefix pipeline (dce → dce-verify → gvn → gvn-verify via opt.BuildPipeline, mirroring TestGVNL1PrefixPipelineInterop) — one iter survives, graph verifies
- No files under doc/optana/ modified (`git status` check)
</verification>

<success_criteria>
- LF-1: KindIter absent from sideeffect baseVerdict AND gvn-verify baseVerdictMirror (one commit); package comments accurate
- LF-2: identical iters (flat, bounded-with-CSE'd-bounds, whole nested chains) fuse in one gvn run; arities and parent chains correctly block fusion
- LF-3: Iter with effectful operand stays effectful (order-sensitive taint), pinned by test
- LF-4: gvn-verify green on every new fixture (ANALYSIS-CLOSURE fixpoint holds with the reclassification)
- LF-5: loopforest implements the doc's forest algorithm (live-iters collection, reversed edges, root set, allocation-order determinism, separate structure, on-demand, read-only)
- LF-6: "loop-forest" registered, composed via analysis/all, membership + BuildAll pinned in all_test.go
</success_criteria>

<output>
After completion, create `.planning/quick/261019-uoq-implement-loop-fusion-via-gvn-loop-nest-/261019-uoq-SUMMARY.md`
</output>
