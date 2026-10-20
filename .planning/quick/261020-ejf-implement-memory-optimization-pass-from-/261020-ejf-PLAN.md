---
phase: quick-261020-ejf
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/ir/mutate.go
  - internal/ir/effect.go
  - internal/ir/node_memory.go
  - internal/ir/node_end.go
  - internal/ir/mutate_test.go
  - internal/opt/memopt/memopt.go
  - internal/opt/memopt/verify.go
  - internal/opt/memopt/memopt_test.go
  - internal/opt/all/all.go
  - internal/opt/registry.go
  - doc/optana/memory-opt.md
autonomous: true
requirements:
  - C1 (chain-totality eligibility guard — skip, never error)
  - C2a (zero-live-use store rule + RAUW-to-value + fixpoint sweep)
  - C2b (store-store needs same proven access path; walk stops at first load)
  - C2c (regions keyed by tensor operand node identity; MemInit = KindMemNew)
  - C3 (MutContext.RemoveImpactBy choke point)
  - C4 (GMEM HIR only, Level-1 sea, slot after gvn, erase/rewire-only)
  - INV-A..G (post-conditions asserted by memopt-verify and the property tests)

must_haves:
  truths:
    - "A chained load whose first-reached store behind it has the same proven access path is erased and its users see the stored value"
    - "An earlier store shadowed by a later same-path store (only stores between them) is erased with its chain edge spliced via SetDef, or dropped via RemoveImpactBy when it is chain-first"
    - "A zero-live-use store is erased after its data uses are RAUW'd to its value operand, cascading to a fixpoint; a Return-wired store is never erased"
    - "A region whose accesses do not form ONE linear chain is skipped entirely — no error, no mutation"
    - "memopt-verify is green after memopt on every fixture, flags a planted forwardable load, and rejects reversed pipeline order"
    - "A second memopt run changes nothing; every mutation stamps Pass \"memopt\" via one MutContext"
  artifacts:
    - path: "internal/opt/memopt/memopt.go"
      provides: "memopt pass — priority 112, Requires [gvn], fixpoint forward/eliminate/sweep over eligible regions"
      contains: "opt.Register"
      min_lines: 80
    - path: "internal/opt/memopt/verify.go"
      provides: "memopt-verify pass — priority 113, Requires [memopt], fixpoint + integrity assertions over eligible regions"
      contains: "opt.Register"
    - path: "internal/ir/mutate.go"
      provides: "RemoveImpactBy choke point (C3)"
      contains: "RemoveImpactBy"
    - path: "internal/opt/memopt/memopt_test.go"
      provides: "the review's 10 property-test fixture families"
    - path: "internal/opt/all/all.go"
      provides: "blank import firing memopt's init registration"
      contains: "tcacomp/internal/opt/memopt"
  key_links:
    - from: "internal/opt/memopt/memopt.go"
      to: "internal/ir/mutate.go"
      via: "mctx.RemoveImpactBy when the erased store/load is chain-first"
      pattern: "RemoveImpactBy"
    - from: "internal/opt/memopt/memopt.go"
      to: "internal/analysis/sideeffect"
      via: "se := sideeffect.Analyze(g) feeding Base.ValueEqual for axes comparison (gvn.go's pattern)"
      pattern: "ValueEqual"
    - from: "internal/opt/all/all.go"
      to: "internal/opt/memopt"
      via: "blank import; BuildPipeline then places memopt after gvn-verify"
      pattern: "_ \"tcacomp/internal/opt/memopt\""
    - from: "internal/opt/memopt/verify.go"
      to: "internal/opt/memopt/memopt.go"
      via: "shared package-internal chain-derivation helper (mirrors gvn's shared code)"
      pattern: "func derive"
---

<objective>
Implement the memory optimization pass from doc/optana/memory-opt.md exactly as
specified by the completed algorithm review (261020-ejf-REVIEW.md — verdict SOUND
with binding conditions C1-C4, invariants INV-A..G, and a 10-family property-test
spec). Deliverables: the memopt pass (store-to-load forwarding, store-store
elimination, dead-store elimination), the sibling memopt-verify pass, the new
MutContext.RemoveImpactBy IR choke point (C3), and the property tests.

Purpose: the pass is the doc's stated enabler for loop-fusion intermediates and
the frontend's local-tensor model; today's builder graphs rarely fire the rules,
so the pass's own tests construct chain-carrying graphs directly via the ir
constructors (same as dce_test.go) — that is expected and fine.
Output: internal/opt/memopt/{memopt,verify,memopt_test}.go, the ir choke point,
registration wiring, and the doc's appended verdict section.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@./CLAUDE.md
@.planning/STATE.md
@.planning/quick/261020-ejf-implement-memory-optimization-pass-from-/261020-ejf-REVIEW.md
@doc/optana/memory-opt.md
@internal/opt/registry.go
@internal/opt/dce/dce.go
@internal/opt/dce/verify.go
@internal/opt/dce/dce_test.go
@internal/ir/mutate.go
@internal/ir/effect.go
@internal/ir/node_memory.go
@internal/ir/node_end.go

<interfaces>
<!-- Key contracts the executor needs. Extracted from the codebase. -->

internal/ir — MutContext choke points (mutate.go):
  type MutContext struct{ Pass string }
  (*MutContext).SetDef(u SSAUse, newVal Node) error   // moves the record between vars — used for SPLICE
  (*MutContext).RemoveUse(u SSAUse)                   // kills record + drops OPERAND slot only (the C3 gap)
  (*MutContext).ReplaceAllUsesWith(old, newVal Node) error
  (*MutContext).Erase(n Node) error                   // REFUSES while any live use of n remains
  // NEW in Task 1: (*MutContext).RemoveImpactBy(u SSAUse) error

internal/ir — effect-chain model (effect.go, node_memory.go, node_end.go):
  func ImpactBy(n Node) []SSAUse                      // copy of effectIns handles; nil = chain-first / none
  func UsesOf(def Node) []SSAUse
  (SSAUse).Def() Node / .User() Node / .IsKilled() bool
  (*Base).Erased() bool / .ID() uint64 / .KindOf() Kind
  type effectBearer interface{ impactBy() []SSAUse }  // implemented by memLoadNode, memStoreNode, returnNode
  operandList.removeSlot(u SSAUse) bool               // record-identity drop (slices.Delete) — the pattern to mirror
  func wirePayloadUse(user, val Node) SSAUse

Node layouts (operand order is LOAD-BEARING):
  memLoadNode:  ops = [tensor, axes...];   effectIns = ImpactBy
  memStoreNode: ops = [value, tensor, axes...] (value is operand 0 — store returns input#0)
  returnNode:   ops = returned values;     effectIns = one ImpactBy edge per wired region's final access

Constructors for fixtures (graph methods):
  g.NewInputParam(origin, ir.RoleDim|RoleTensor|RoleScalar, name, ty)
  ir.InternScalar(ir.TF32) / ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "m"}})
  g.NewConst / g.NewBinOp / g.NewMemNew(extent)
  g.NewMemStore(origin, tensor, axes, value)          // UN-chained (C1-violation fixtures)
  g.NewMemStoreChained(origin, env, tensor, axes, value)
  g.NewMemLoadChained(origin, env, tensor, axes)
  g.NewReturnChained(origin, env, []Node{tensors...}, values...) // wires End's ImpactBy at current heads
  ir.NewEnv() / ir.NewGraph(name) / ir.Verify(g) -> []string / g.AllocatedNodes() / g.End()

Value identity for axes comparison (value.go, node.go):
  func (b *Base) ValueEqual(other Value, se SideEffectAnalysis) bool
  // se plumbing: se := sideeffect.Analyze(g)  (tcacomp/internal/analysis/sideeffect — gvn.go line 111)

Pass surface (opt):
  opt.Register(opt.PassSpec{ID, Priority, Requires, Factory})
  opt.Pass: ID() / Requires() / Produces() opt.Stage / Preserved() / Run(*opt.Context, *ir.Graph) error
  opt.StageConstructed / opt.NewManager([]opt.Pass) / opt.NewVerifyEach(nil)
  opt.Lookup(id) (PassSpec, bool) / opt.BuildPipeline() / opt.Registered()
  Priorities: 100 dce, 101 dce-verify, 105 typeinfer, 110 gvn (Requires nil), 111 gvn-verify, 900 schedule
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: MutContext.RemoveImpactBy choke point (C3) with ir-level tests</name>
  <files>internal/ir/mutate_test.go, internal/ir/mutate.go, internal/ir/effect.go, internal/ir/node_memory.go, internal/ir/node_end.go</files>
  <behavior>
    - Happy path: on a chained store pair, RemoveImpactBy(successor's edge) empties the successor's ImpactBy list, kills the record (IsKilled), removes it from the predecessor's ref list, leaves ir.Verify green.
    - Nil use: deterministic error "ir: RemoveImpactBy on nil use".
    - Wrong slot class: a use held in an ordinary operand slot (e.g. a BinOp input) is rejected fail-closed with an error naming the user's node ID and kind; graph untouched; ir.Verify green.
    - Uniform hook: the Return node's ImpactBy edge is removable the same way (pins returnNode coverage).
  </behavior>
  <action>
    Write internal/ir/mutate_test.go FIRST (RED), following the package declaration and fixture style of the neighboring internal/ir/effect_test.go / node_memory_test.go — build small chained graphs exactly like dce_test.go's middleLinkFixture (dim param, tensor param, consts, NewMemStoreChained x2, NewReturnChained). Then implement (GREEN):

    1. effect.go: extend the unexported effectBearer interface with a removal hook, removeImpactUse(u SSAUse) bool — drop the handle whose record identity matches u.rec from the bearer's effectIns slice, mirroring operandList.removeSlot (compare s.rec == u.rec, slices.Delete; only the handle copy is dropped, per CR-01/CR-02).

    2. node_memory.go + node_end.go: implement the hook on memLoadNode, memStoreNode, and returnNode. Do NOT change the constructors or the effectIns field layout — the hook only removes; the chained constructors keep their direct append wiring.

    3. mutate.go: add the exported choke point following IR-04/D-17 conventions (doc comment in the repo's explanatory style, cross-referencing the review's C3 and dce's hinge-6 gap): func (ctx *MutContext) RemoveImpactBy(u SSAUse) error. Semantics: nil record → error "ir: RemoveImpactBy on nil use"; resolve u.rec.user; if the user is not an effect bearer or the hook returns false (the record is not held in its effectIns), return a fail-closed fmt.Errorf naming the user's node ID and KindOf (mirror SetLambdaBody's error style); on success detach the record from its def's ref list (def.removeRef), set u.rec.killed = true, drop the handle via the hook, and stamp the user and the def's node (mirror RemoveUse's stamping). Document the division of labor: RemoveImpactBy only DROPS an edge (the chain-first removal case); re-pointing a successor at the removed node's predecessor stays a plain SetDef splice.

    Keep it minimal: no changes to RemoveUse/Erase/SetDef behavior. Run the full internal/ir suite to prove zero regression.
  </verify>
    <automated>go test ./internal/ir/ -run 'TestMutateRemoveImpactBy' -v && go test ./internal/ir/</automated>
  </verify>
  <done>RemoveImpactBy exists, is exported, fail-closed on misuse with node-ID/kind diagnostics, removes handles from all three effect bearers by record identity, kills the record, stamps nodes; new tests pin all four behaviors; the entire internal/ir suite stays green.</done>
</task>

<task type="auto">
  <name>Task 2: memopt pass (112) + memopt-verify pass (113) + registration</name>
  <files>internal/opt/memopt/memopt.go, internal/opt/memopt/verify.go, internal/opt/all/all.go, internal/opt/registry.go</files>
  <action>
    Create package internal/opt/memopt with the repo's pass convention (dce is THE template: package comment citing the quick task + doc/optana/memory-opt.md + the review, init() self-registration, Pass struct with ID/Requires/Produces/Preserved/Run, level gates FIRST, all mutations through one ir.MutContext{Pass: "memopt"}).

    memopt.go — Pass{}, init registers opt.PassSpec{ID: "memopt", Priority: 112, Factory: ...}. Requires() returns ["gvn"] (C4: runs after GVN so identity-keyed axes make ValueEqual maximally effective); Produces() opt.StageConstructed; Preserved() nil. Run, in order:

    1. LEVEL GATES before any mutation: nil graph → errors.New("memopt: nil graph"); g.Materialized() || g.CFG() != nil → deterministic error mentioning "materialized" (dce's wording pattern). A refused graph is untouched.

    2. DERIVE REGIONS (C1, shared helper reused by verify.go — mirror gvn's shared-code pattern): iterate g.AllocatedNodes() in allocation order; group live (non-Erased) KindMemLoad/KindMemStore by their tensor operand NODE identity (load operand 0, store operand 1; key by BasePtr — never by kind or name, per C2c). Per region verify ONE LINEAR CHAIN: each access has at most one ImpactBy edge and its def is in the set; each access has at most one effect-user successor inside the set ∪ {g.End()} (enumerate via each candidate user's ImpactBy records); the edges connect the whole set as a single path. A region that fails = legal-but-unoptimized → SKIP it silently: no error, no mutation (effect.go documents un-chained construction as legal).

    3. FIXPOINT over eligible regions only (progress-flag loop bounded like dce's sweep, e.g. total-access-count+1 rounds), all mutations through the one MutContext created after the gates. Determinism (INV-G): regions processed in first-access allocation order, accesses in chain order; maps used for grouping only, never for decision order.

       a. FORWARD LOADS: for load L, walk ImpactBy backwards — pass through loads; the FIRST store reached is B; forward only when B exists and the access paths are PROVEN equal (C2b discipline): same tensor node identity + same axes arity + pairwise axes ValueEqual, with se := sideeffect.Analyze(g) as the SideEffectAnalysis argument (gvn.go's plumbing; axes are pure kinds so verdicts are analysis-independent). On fire: mctx.ReplaceAllUsesWith(L, B's operand 0) (the stored value; covers Return data operands too), then splice L's successor edges (the next access's ImpactBy record, and End's ImpactBy record when L is the region's final access): plain mctx.SetDef(edge, L's own ImpactBy def) when L has a predecessor, else mctx.RemoveImpactBy(edge) — L was chain-first; then mctx.Erase(L). Propagate Erase refusals as errors (fail-closed, they name the node).

       b. STORE-STORE ELIMINATION (C2b): for store S, walk ImpactBy backwards: a load stops the walk (it may read that position); at each store S0 reached, if sameProvenPath(S, S0): RAUW S0's data uses to its operand 0, splice the just-traversed edge (def == S0) to S0's predecessor via SetDef or RemoveImpactBy when S0 is chain-first, Erase S0, continue the walk from S0's predecessor (same-position cascades die in one walk); an unequal-position store is walked PAST (sound under the restrict-alias rule — the review's argument 2).

       c. DEAD-STORE SWEEP (C2a): for each store in an eligible region with NO effect-user successor (no payload use from the region set and none from End): first RAUW its data uses to its operand 0 (a store returns input#0, so the rewire is value-preserving), then Erase when it reaches zero live uses. Sweep to fixpoint so unwired tail cascades (S1→S2 both unread) fully die. A Return-wired store always keeps its payload use and is NEVER erased (INV-D).

    The pass is erase/rewire-only (C4): no new nodes, no new kinds; GMEM HIR only — LIR SMEM/FRAG families carry no effect chains and are out of scope by construction.

    verify.go — VerifyPass{}, init registers opt.PassSpec{ID: "memopt-verify", Priority: 113, Requires: ["memopt"], Factory: ...}; Requires() ["memopt"], Produces StageConstructed, Preserved nil; mutates NOTHING (dce/verify.go template). Level gates ("memopt-verify: nil graph" / materialized refusal) before anything. Re-derive regions/chains with the shared helper, then assert over ELIGIBLE regions only: (1) no killed/dangling use record held by a live memory node and no use of an erased def; (2) FIXPOINT — no forwardable load remains (first-store-reached + proven-equal path), no eliminable store pair remains (proven-equal pair with only stores between), no zero-use store remains (no-successor store that still carries only RAUW-able data uses) — legally-unoptimizable (C1-skipped) regions must NOT fail it; (3) every access's effect users ∈ accesses ∪ {End}. All failures are fail-closed errors naming node IDs and kinds.

    Wiring: internal/opt/all/all.go gains the blank import _ "tcacomp/internal/opt/memopt" (both init()s fire from one import). internal/opt/registry.go: extend the priority-scale comment with the 112/113 anchor lines (112 memory optimization memopt — forwarding + store elimination, post-GVN; 113 memopt-verify — the post-memopt invariant/fixpoint check) between the 111 and "..." lines. Note: registry_test.go's BuildPipeline test asserts relative order, not an exact pass list — adding 112/113 must keep it green unchanged.
  </verify>
    <automated>go build ./... && go vet ./internal/opt/... ./internal/ir/... && go test ./internal/opt/ ./internal/ir/</automated>
  </verify>
  <done>Both passes registered and resolvable via opt.Lookup ("memopt" Priority 112, "memopt-verify" Priority 113 Requires memopt); after the all import, BuildPipeline places memopt immediately after gvn-verify; gates refuse nil/materialized graphs untouched before any mutation; every mutation rides the single MutContext{Pass: "memopt"}; the fixpoint implements C1/C2a/C2b/C2c exactly; all existing suites stay green.</done>
</task>

<task type="auto">
  <name>Task 3: Property tests (the review's 10 fixture families) + doc verdict appendix</name>
  <files>internal/opt/memopt/memopt_test.go, doc/optana/memory-opt.md</files>
  <behavior>
    - Family 1 gates: nil graph exact error; materialized graph refused with node count unchanged.
    - Family 2 forwarding fires: S(p,v)→L(p) with L returned — L erased, Return data operand becomes v, successor edges spliced to S, ir.Verify green.
    - Family 3 forwarding blocked: different position; intervening unequal store (most-recent-store discipline); no predecessor store — loads all kept.
    - Family 4: two same-position loads both forwarded; hinge-6-shaped middle load chain-spliced through.
    - Family 5 store-store: same position no read → earlier erased; interleaved load blocks; different positions kept; S0(p)→S1(p')→S2(p) walk-past-unequal (S0 dies, S1 stays); 3-store cascade collapses.
    - Family 6 dead stores: unwired S1→S2 tail cascade both erased; data-use store RAUW'd to its value then erased; Return-wired tail NEVER erased.
    - Family 7 C1 guard: one access built via un-chained NewMemStore on an otherwise-chained region → region skipped, nothing erased, Run returns nil.
    - Family 8 head removal: erased earlier store is chain-first → successor's ImpactBy edge DROPPED via RemoveImpactBy, ir.Verify green.
    - Family 9: idempotence (second Run: node count unchanged); Manager interop [dce, gvn, memopt, memopt-verify] + NewVerifyEach(nil) runs clean; registry/BuildPipeline include memopt after gvn-verify.
    - Family 10 memopt-verify: green after Run on every fixture from families 2-8; flags a planted forwardable load; reversed [memopt-verify, memopt] pipeline rejected on "requires"; its own level gates.
  </behavior>
  <action>
    Write internal/opt/memopt/memopt_test.go in dce_test.go's exact style: external test package importing tcacomp/internal/ir, tcacomp/internal/opt, tcacomp/internal/opt/dce, tcacomp/internal/opt/gvn, tcacomp/internal/opt/memopt; local nodeSet/assertGone/assertKept helpers plus an assertImpactBy equivalent of dce_test.go's; fixtures built ONLY through the ir constructors (chained constructors for well-formed regions; plain NewMemStore to plant C1 violations); every mutating fixture asserted with ir.Verify green, and the forwarding/elimination fixtures additionally re-Run for idempotence. Implement the 10 families exactly as the review's "Property tests" section and the behavior list above specify — including the exact gate error strings ("memopt: nil graph", "memopt-verify: nil graph") mirroring TestDCEGateNilGraph, the untouched-refusal node-count check, the Return data operand update assertion in family 2 (Return's operand list is readable via g.End()), and family 9's registry assertion via opt.Lookup/Registered priorities (112/113) plus BuildPipeline ordering (gvn-verify before memopt before memopt-verify). For family 10's planted-flag test, build the forwarding fixture and run ONLY VerifyPass — expect an error naming the load's node ID and kind.

    Then append to doc/optana/memory-opt.md a "## Review verdict (implemented corrections)" section (pointer-style summary, the review file stays the full record): C2a (the sound rule is the SSAVar zero-live-use test + RAUW-to-value + fixpoint — not "no MemLoad after"), C2b (store-store needs the SAME PROVEN access path and stops at the first load), C2c (MemInit = KindMemNew; regions keyed by tensor operand node identity), C1 (chain-totality eligibility guard; failing regions are skipped, never errors), C3 (the RemoveImpactBy choke point), C4 (GMEM HIR, Level-1 sea, slot after gvn, erase/rewire-only), and the INV-A..G post-conditions. Do not rewrite the doc's prose — append, so the original algorithm text stays auditable.
  </verify>
    <automated>go test ./internal/opt/memopt/ -v && go test ./...</automated>
  </verify>
  <done>All 10 fixture families pinned and green under go test ./internal/opt/memopt/; the full repo suite (go test ./...) is green; doc/optana/memory-opt.md carries the verdict appendix so doc and implementation agree.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| none | In-process compiler transform over an already-verified IR; no user input, network, or file I/O crosses a boundary in this code. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-q-01 | Tampering (soundness) | memopt forwarding/elimination rules | mitigate | C1 chain-totality guard skips suspicious regions; INV-C property tests pin memory-state preservation; memopt-verify re-derives and asserts the fixpoint |
| T-q-02 | Denial of service | fixpoint loop | mitigate | rounds bounded by total access count + 1 with a progress flag (dce sweep pattern) |
| T-q-03 | Information disclosure | error messages | accept | diagnostics name node IDs/kinds only — no user data exists in the IR layer |
</threat_model>

<verification>
- go test ./internal/ir/ — choke point tests green, zero regression in the IR suite.
- go test ./internal/opt/ — registry/BuildPipeline ordering green with memopt at 112 and memopt-verify at 113.
- go test ./internal/opt/memopt/ -v — all 10 review fixture families green.
- go test ./... — full suite green (stdlib-only project; no external installs anywhere).
- Manual spot check: opt.Lookup("memopt") and opt.Lookup("memopt-verify") resolve with the review's priorities and Requires wiring.
</verification>

<success_criteria>
- Every binding condition implemented as specified: C1 skip-not-error, C2a zero-live-use + RAUW + fixpoint, C2b proven-path + stop-at-load, C2c node-identity region keys, C3 RemoveImpactBy, C4 scope and pipeline slot.
- INV-A (Verify green), INV-B (subsequence chains), INV-D (Return-wired stores survive), INV-E (erase/rewire-only, one MutContext), INV-F (idempotence), INV-G (allocation/chain-order determinism) are each pinned by at least one test in memopt_test.go.
- No changes to existing choke-point or constructor semantics; no new node kinds; no external dependencies.
- doc/optana/memory-opt.md and the implementation tell the same story.
</success_criteria>

<output>
Create `.planning/quick/261020-ejf-implement-memory-optimization-pass-from-/261020-ejf-SUMMARY.md` when done.
Commit convention: test(quick-261020-ejf) / feat(quick-261020-ejf) prefixes per repo history.
</output>
