---
phase: quick-261024-sdk
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/ir/node_loop.go
  - internal/ir/kind.go
  - internal/ir/catalog.go
  - internal/ir/print.go
  - internal/ir/loopspec.go
  - internal/ir/loopspec_test.go
  - internal/ir/verify.go
  - internal/analysis/loopforest/loopforest.go
  - internal/analysis/loopforest/loopforest_test.go
  - internal/analysis/loopforest/loopforest_lir_test.go
  - internal/opt/tiling/lower.go
  - internal/opt/tiling/lower_rewrite.go
  - internal/opt/tiling/lower_cache.go
  - internal/opt/tiling/verify.go
  - internal/opt/gvn/gvn.go
  - internal/opt/gvn/verify.go
  - internal/opt/gvn/gvn_test.go
  - internal/opt/schedule/nest.go
  - internal/opt/memopt/memopt.go
  - internal/opt/memopt/verify.go
  - internal/opt/memopt/memopt_test.go
  - internal/ir/testdata/golden_smoke.txt
  - internal/cg/cxx/testdata/matmul_store.cu
  - internal/cg/cxx/testdata/reduce_warp.cu
  - internal/e2e/e2e_test.go
autonomous: true
requirements: [FIX1-01]
must_haves:
  truths:
    - "ForLoop has no body operand; NewForLoop signature is (origin, iv, bound, attrs...)"
    - "loopforest derives LIR nesting from IV data-dep references, not BodyRoot containment"
    - "tile-lower §8 checkFillPlacement uses ORDERING (chain predecessor) + LAYERING (no inside-IV), not body-reachability"
    - "GVN has no lambdaBodyRoots / body-root exemption"
    - "schedule pass uses loopforest IV-membership, not local body/reach derivation"
    - "memopt removes unconsumed buffer stores (fills with no live buffer reads) together with their source loads"
    - "matmul_store.cu and reduce_warp.cu goldens are byte-identical to before"
    - "Frontend e2e trigger fixture passes tile-lower (no longer fails §8)"
  artifacts:
    - path: "internal/ir/node_loop.go"
      provides: "ForLoop node without body slot; NewForLoop(origin, iv, bound, attrs...)"
      min_lines: 80
    - path: "internal/ir/loopspec.go"
      provides: "LoopNesting interface without BodyRoot method; IsLoop + ParentEdge only"
      contains: "type LoopSpec interface"
    - path: "internal/analysis/loopforest/loopforest.go"
      provides: "IV-data-dep-derived forest for LIR forloops; no BodyRoot calls"
      contains: "func Analyze"
    - path: "internal/opt/tiling/lower_rewrite.go"
      provides: "buildNest without body chaining; lw.inside re-derived from IV closure; bodyRootOf deleted"
    - path: "internal/opt/tiling/lower.go"
      provides: "checkFillPlacement with ORDERING (chain predecessor) + LAYERING (no inside-IV); PLACEMENT-by-reachability deleted"
    - path: "internal/opt/gvn/gvn.go"
      provides: "GVN without lambdaBodyRoots or body-root exemption"
    - path: "internal/opt/schedule/nest.go"
      provides: "schedule pass forest derivation via loopforest IV-membership relation"
    - path: "internal/opt/memopt/memopt.go"
      provides: "fill liveness removal — unconsumed buffer stores removed with source loads"
  key_links:
    - from: "internal/ir/node_loop.go"
      to: "internal/opt/tiling/lower_rewrite.go"
      via: "buildNest calls NewForLoop"
      pattern: "NewForLoop"
    - from: "internal/ir/loopspec.go"
      to: "internal/analysis/loopforest/loopforest.go"
      via: "LoopSpec interface consumed by Analyze"
      pattern: "spec\\.BodyRoot|spec\\.IsLoop|spec\\.ParentEdge"
    - from: "internal/analysis/loopforest/loopforest.go"
      to: "internal/opt/schedule/nest.go"
      via: "schedule pass derives forest from loopforest"
    - from: "internal/opt/tiling/lower.go"
      to: "internal/opt/memopt/memopt.go"
      via: "dead fill removal (D-f) owned by memopt, not DCE"
---

<objective>
Remove all loop body operands from the IR per the 2026-10-24 author ruling in doc/fix1.md: loops are control-flow nodes with no body; membership is via IV back-references; ordering is via data + chain edges; tile-lower §8 checks are recast around the chain/IV-membership relation. Seven ordered commits, each leaving the tree green.

Purpose: The body-operand model violates the sea-of-nodes design (never a direct link from a control-flow node to an expression). It caused tile-lower §8 to reject correctly-spliced graphs when a load (not a store) was elected body root on rank>=2 LNTs with innermost reduction + promoted load site.
Output: Body slot removed from ForLoop, LoopSpec.BodyRoot deleted, loopforest derives nesting from IV data-dep closure, tile-lower checkFillPlacement recast to ORDERING+LAYERING-IV, GVN lambdaBodyRoots removed, schedule pass uses loopforest IV-membership, memopt owns fill liveness removal, goldens for unaffected shapes byte-identical, frontend e2e passes tile-lower.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@doc/fix1.md (THE authoritative spec — rulings, constraints C1–C5, derived consequences D-a–D-g, §5 implementation plan with 7 ordered steps)
@doc/tca.md (IR design context, §5.10 for loop/reduce model)
@internal/ir/node_loop.go (current NewForLoop signature: (origin, iv, bound, body, attrs...))
@internal/ir/loopspec.go (current LoopNesting interface with BodyRoot method)
@internal/ir/verify.go (checkLoopScope — current body-operand containment form)
@internal/analysis/loopforest/loopforest.go (current BodyRoot-derived linking)
@internal/opt/tiling/lower_rewrite.go (bodyRootOf, buildNest, reachableFrom, lw.inside)
@internal/opt/tiling/lower.go (checkFillPlacement — current PLACEMENT-by-reachability form)
@internal/opt/gvn/gvn.go (lambdaBodyRoots + body-root exemption)
@internal/opt/gvn/verify.go (bodyRoots mirror)
@internal/opt/schedule/nest.go (schedule forest derivation)
@internal/opt/memopt/memopt.go (dead-load elimination — fill liveness extends this)
@internal/opt/memopt/verify.go (memopt fixpoint checker)

Implement exactly per doc/fix1.md §5 — the 7 steps are strictly ordered, each = one commit, each leaves the tree green. Do NOT reorder, combine across step boundaries in a way that breaks per-step greenness, or skip any step.
</context>

<tasks>

<task type="auto">
<name>Task 1: IR layout + loopforest + tiling (Steps 1–3 of §5)</name>
<files>
internal/ir/node_loop.go, internal/ir/kind.go, internal/ir/catalog.go, internal/ir/print.go,
internal/ir/loopspec.go, internal/ir/loopspec_test.go, internal/ir/verify.go,
internal/analysis/loopforest/loopforest.go, internal/analysis/loopforest/loopforest_test.go, internal/analysis/loopforest/loopforest_lir_test.go,
internal/opt/tiling/lower.go, internal/opt/tiling/lower_rewrite.go, internal/opt/tiling/lower_cache.go, internal/opt/tiling/verify.go,
internal/ir/testdata/golden_smoke.txt
</files>
<action>

Execute the first three steps of doc/fix1.md §5 as three sequential commits (each commit leaves the tree green):

**Step 1 — ir: ForLoop layout** (commit 1 of 7)
- Change `NewForLoop` signature in internal/ir/node_loop.go from `(origin, iv, bound, body Node, attrs ...Node)` to `(origin, iv, bound Node, attrs ...Node)` — drop the body parameter. Operand layout becomes `[iv, bound, attrs...]`.
- Update the layout comment in internal/ir/kind.go for KindForLoop to remove the body slot description.
- Update the FamilyLoop row in internal/ir/catalog.go to reflect new operand count (no body).
- Update internal/ir/print.go so ForLoop printing shows the new operand layout (no body operand line).
- Delete `BodyRoot(n Node) Node` from the LoopNesting interface in internal/ir/loopspec.go. The interface keeps IsLoop + ParentEdge only.
- Delete the forLoopNode BodyRoot implementation in internal/ir/loopspec.go.
- Delete or update the body-related portions of loopspec comments and doc comments (the ones that describe body-operand containment).
- Delete TestLoopSpecBodyRoot from internal/ir/loopspec_test.go (it pins the deleted behavior).
- Update `checkLoopScope` in internal/ir/verify.go per D-c: delete the body-operand containment test. The residual invariant (a node using L's IV ordered after L's completion) may be vacuous — if so, delete checkLoopScope entirely rather than keeping a tautology. The tiling §8 SCOPE check already guards the placement-level hazard.
- Update all NewForLoop call sites. The only producer is tiling's buildNest (Step 3 updates it — for this step, temporarily pass `nil` or adjust call sites to match new signature so the build compiles; the real wiring comes in Step 3).
  Actually, since Step 1 must leave the tree green and buildNest still passes a body arg, update buildNest's call in Step 1 to drop the body argument (it will be properly rewired in Step 3). The goal is compile + test green.
- Regenerate goldens that print forloops mechanically — only operand-count deltas are acceptable. Check internal/ir/testdata/golden_smoke.txt and any other IR-printing goldens.
- Verify: `go build ./... && go vet ./... && go test ./internal/ir/... ./internal/analysis/...` all pass.
- Commit message: `ir(261024-sdk): drop ForLoop body slot — sea-of-nodes: loops have no body operands`

**Step 2 — analysis/loopforest** (commit 2 of 7)
- Rewrite internal/analysis/loopforest/loopforest.go to derive LIR loop-to-loop nesting from **IV data-dep references** (D-b), not BodyRoot containment.
- Rule: loop L_in is nested in L_out iff L_out's IV dimvar is in L_in's data-dep operand closure. Effect edges are NEVER linking input (C3: completion edges order, never nest).
- Delete all calls to `spec.BodyRoot()` from loopforest.go. Delete the bodyCaches field and any body-closure machinery.
- Replace with: for each loop, collect its IV dimvar; for each candidate child loop, walk its data-dep (non-chain, non-effect) operand closure; if the parent's IV dimvar appears in that closure, link parent→child.
- Update loopforest comments to document the IV-membership relation.
- Update internal/analysis/loopforest/loopforest_test.go and loopforest_lir_test.go: re-pin tests to the new forest derivation. HIR iter tests (which use ParentEdge, not BodyRoot) should be unaffected. LIR forloop tests need new pins showing IV-derived nesting (inner loop under outer because inner's data-dep closure references outer's IV dimvar).
- Verify: `go build ./... && go test ./internal/analysis/loopforest/...` passes.
- Commit message: `analysis(261024-sdk): loopforest derives nesting from IV data-dep closure, not BodyRoot`

**Step 3 — opt/tiling** (commit 3 of 7)
- In internal/opt/tiling/lower_rewrite.go:
  - Delete `bodyRootOf` entirely (D-g).
  - Rewrite `buildNest` to create loops with `[iv, bound, attrs]` and no body chaining. The inner variable no longer needs a body root to start from — instead, loops are constructed as a nest with no body operand, and membership is implicit via IV references.
  - Re-derive `lw.inside` (the P6 inside-set, D-g): nodes whose data-dep closure references any of the LNT's loops' IVs — the same relation loopforest now uses. Delete `lw.reachableFrom(body)` derivation; replace with the IV-closure membership test.
  - Update all references to `lw.bodyRoot` / `lw.inside` throughout lower_rewrite.go and lower_cache.go.
  - The `reachableFrom` function may still be needed for other purposes (e.g., chain walks) — audit callers. If it is only used for body-derived inside sets, delete it. If used elsewhere, keep it but rename/repurpose so no walk silently changes meaning (§6 Risks).
- In internal/opt/tiling/lower.go:
  - Rewrite `checkFillPlacement` per D-e: delete PLACEMENT-by-reachability and body-subtree LAYERING.
  - ORDERING becomes the structural check: every live buffer access of a site has the fill among its chain predecessors (ImpactBy). Verify this by walking chain predecessors of each access and checking the fill's store node is among them.
  - LAYERING is recast: a cooperative fill's operand-closure references no inside-loop IV. (SCOPE and UNIFORMITY are unchanged — already IV-based.)
  - EPOCH is no longer a separate check — the fill's coordinates are the promoted load's own axes by construction, so epoch membership is inherent.
- In internal/opt/tiling/lower_cache.go: remove the `bodyRoot == load` special-case comment/code (HI-02 path is deleted — D-e's new LAYERING check covers this correctly).
- In internal/opt/tiling/verify.go: update any invariants that referenced body operands or body-root containment.
- This commit flips the frontend e2e trigger from fail to pass at tile-lower (per §5 step 3 description).
- Verify: `go build ./... && go test ./internal/opt/tiling/...` passes.
- Commit message: `opt(261024-sdk): tiling — delete bodyRootOf, recast checkFillPlacement to ORDERING+LAYERING-IV`

Constraint C5: after each step, verify that `matmul_store.cu` and `reduce_warp.cu` goldens are byte-identical to before this plan. Run `go test ./internal/cg/cxx/...` and confirm no golden changes for these two files.
</action>
<verify>
<automated>go build ./... && go vet ./... && go test ./internal/ir/... ./internal/analysis/loopforest/... ./internal/opt/tiling/... ./internal/cg/cxx/...</automated>
</verify>
<done>
Steps 1-3 complete with 3 green commits: ForLoop has no body slot, LoopSpec.BodyRoot deleted, loopforest uses IV data-dep closure, tiling's checkFillPlacement uses ORDERING+LAYERING-IV, bodyRootOf deleted, buildNest has no body chaining, lw.inside re-derived from IV closure, matmul_store.cu and reduce_warp.cu are byte-identical.
</done>
</task>

<task type="auto">
<name>Task 2: GVN + schedule + memopt (Steps 4–6 of §5)</name>
<files>
internal/opt/gvn/gvn.go, internal/opt/gvn/gvn_test.go, internal/opt/gvn/verify.go,
internal/opt/schedule/nest.go, internal/opt/schedule/scheduler.go,
internal/opt/memopt/memopt.go, internal/opt/memopt/memopt_test.go, internal/opt/memopt/verify.go
</files>
<action>

Execute steps 4-6 of doc/fix1.md §5 as three sequential commits (each commit leaves the tree green):

**Step 4 — opt/gvn** (commit 4 of 7)
- In internal/opt/gvn/gvn.go: delete `lambdaBodyRoots` function and the body-root exemption in the main GVN loop (the `bodyRoots[n]` skip check). Per D-d: with no body operands there is nothing to exempt.
- In internal/opt/gvn/verify.go: delete the bodyRoots mirror check (the `bodyRoots[n]` skip in the verifier).
- In internal/opt/gvn/gvn_test.go: delete TestGVNLambdaBodyRootSkip (it pins the deleted exemption behavior).
- Update GVN comments that reference body operands or lambda body roots.
- Verify: `go build ./... && go test ./internal/opt/gvn/...` passes.
- Commit message: `opt(261024-sdk): gvn — delete lambdaBodyRoots and body-root exemption`

**Step 5 — opt/schedule** (commit 5 of 7)
- In internal/opt/schedule/nest.go: replace local body/reach derivations with the loopforest IV-membership relation.
- The schedule pass currently builds its own ForLoop nest forest (derived from body operands). Replace this with a call to or reuse of loopforest.Analyze, which now correctly derives nesting from IV data-dep closure.
- Fail-closed gates (branched nest, KindIterCluster, etc.) are unchanged.
- Update comments that reference body-operand containment.
- Verify: `go build ./... && go test ./internal/opt/schedule/...` passes.
- Commit message: `opt(261024-sdk): schedule — use loopforest IV-membership relation for nest derivation`

**Step 6 — memopt fill liveness** (commit 6 of 7)
- In internal/opt/memopt/memopt.go: wire fill liveness removal into memopt's dead-load/store elimination (D-f).
- A fill whose buffer has no live reads is dead code. A buffer store (fragstore / the fill's store side) with zero live buffer loads (fragload accesses) is dead together with its source load (the promoted GMEM load).
- This is the same class as memopt's existing dead-load elimination. Extend the dead store / dead load logic to handle buffer (SMEM/FRAG) memory regions, not just GMEM.
- Specifically: when a FragStore (or MemStoreSMem) has no consumers (no FragLoad / MemLoadSMem reads it in the chain-eligible region), remove the store AND its source load (the fill's load). This cascades naturally if the existing dead-store + dead-load elimination is extended to buffer memory classes.
- DCE stays untouched (per the author correction in §4: "Not DCE, we have memory optimization which should remove it").
- In internal/opt/memopt/verify.go: ensure the fixpoint checker covers buffer dead-store elimination.
- Add tests in memopt_test.go for the fill-removal case (covered more thoroughly in Step 7's negative pins).
- Verify: `go build ./... && go test ./internal/opt/memopt/...` passes.
- Commit message: `opt(261024-sdk): memopt — remove unconsumed buffer stores (fills) with source loads`

Constraint C5: after step 5, verify `go test ./internal/cg/cxx/...` shows matmul_store.cu and reduce_warp.cu still byte-identical. After step 6, same check.
</action>
<verify>
<automated>go build ./... && go vet ./... && go test ./internal/opt/gvn/... ./internal/opt/schedule/... ./internal/opt/memopt/... ./internal/cg/cxx/...</automated>
</verify>
<done>
Steps 4-6 complete with 3 green commits: GVN has no lambdaBodyRoots/exemption, schedule uses loopforest IV-membership, memopt removes unconsumed fills with source loads, DCE untouched, goldens byte-identical.
</done>
</task>

<task type="auto">
<name>Task 3: Tests — fundamental only (Step 7 of §5)</name>
<files>
internal/opt/tiling/lower_test.go,
internal/analysis/loopforest/loopforest_lir_test.go,
internal/opt/memopt/memopt_test.go,
internal/cg/cxx/testdata/matmul_store.cu,
internal/cg/cxx/testdata/reduce_warp.cu,
internal/e2e/e2e_test.go
</files>
<action>

Execute step 7 of doc/fix1.md §5 as one commit (the 7th and final commit). Tests are fundamental only per design.md.

**7.1 Positive tests:**
- Frontend trigger fixture passes the recast §8 suite: ensure the e2e test (added in 261024-k2i) for matmul_scale passes tile-lower. If it was previously marked Skip or expected-failure due to this defect, remove that marking. The test should now pass tile-lower §8.
- Loopforest unit tests pin IV-derived nesting: add or verify a test showing an inner LIR forloop is correctly nested under an outer forloop because the inner's data-dep closure references the outer's IV dimvar.

**7.2 Negative pins (non-vacuity):**
- (a) ORDERING: a fill wired without the chain edge to an access fails checkFillPlacement. Add a test in tiling/lower_test.go that constructs a fill + access pair with the chain edge missing and asserts ORDERING check failure.
- (b) LAYERING: a cooperative fill carrying an inside-IV coordinate fails LAYERING. Add a test in tiling/lower_test.go where the fill's operand-closure references an inside-loop IV (not just enclosing ones), and assert LAYERING check failure.
- (c) memopt-removal: an unconsumed fill is removed by memopt. Add a test in memopt/memopt_test.go with a buffer store that has no corresponding buffer load, and assert both the store and its source load are eliminated.

**7.3 Golden regression (C5):**
- Verify `matmul_store.cu` and `reduce_warp.cu` in internal/cg/cxx/testdata/ are byte-identical to before this fix. No code semantics changed for these shapes (no-reduce matmul and single-dim reduce have no promoted-load-with-innermost-reduce shape).
- If they differ, investigate — only mechanically-regenerated operand-count deltas in forloop-printing goldens (not these .cu files) are acceptable.

**7.4 E2E gate:**
- Run `TCACOMP_E2E=1 go test ./internal/e2e/ -run Frontend -v` (both Appendix A variants of matmul_scale full-pipeline).
- Per §5 step 7 contingency: downstream walls (schedule/final-lower on the reduce shape) are new findings with their own rulings — this fix is NOT judged by them. If the test fails at a stage AFTER tile-lower (e.g., schedule, final-lower, codegen), that is expected and NOT a defect of this fix. The key assertion is that tile-lower no longer fails §8.
- If the e2e test was gated with a SkipIf or t.Skip for the tile-lower defect, update it to skip only on downstream failures (with a clear comment linking to the downstream defect), or to assert pass at tile-lower and accept downstream failure as a known limitation.

Verify full repo green at hermetic level: `go build ./... && go vet ./... && go test ./internal/...` (non-e2e, i.e., without TCACOMP_E2E=1).

Commit message: `test(261024-sdk): fundamental tests — positive/negative pins, golden regression, e2e gate`
</action>
<verify>
<automated>go build ./... && go vet ./... && go test ./internal/... && TCACOMP_E2E=1 go test ./internal/e2e/ -run Frontend -v 2>&1 | head -80</automated>
</verify>
<done>
Step 7 complete with 1 green commit: positive pins pass (frontend trigger passes tile-lower, loopforest nesting correct), negative pins pass (ORDERING/LAYERING/memopt-removal failures correctly detected), golden regression confirms matmul_store.cu + reduce_warp.cu byte-identical, e2e gate reaches past tile-lower (downstream walls accepted as out-of-scope per §5 contingency), full repo hermetic green.
</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| IR node layout → pass consumers | Internal compiler boundary — incorrect operand layout causes panics or wrong code |
| Tile-lower §8 checks → downstream passes | Structural validation gate — wrong checks let unsound schedules through |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-sdk-01 | Tampering | ForLoop operand layout | mitigate | Per-step greenness: every commit runs go build + go vet + targeted tests; golden byte-identity check for unaffected shapes (C5) |
| T-sdk-02 | Tampering | checkFillPlacement recast | mitigate | Negative pins (7.2a/b) prove the recast checks still reject unsound graphs; ORDERING failure on missing chain edge, LAYERING failure on inside-IV |
| T-sdk-03 | Denial of Service | memopt fill liveness (D-f) | mitigate | Fixpoint checker in memopt/verify.go covers buffer-store elimination; removal is bounded by chain-region size |
| T-sdk-04 | Information Disclosure | loopforest IV-closure derivation | accept | Analysis-only pass, no user-facing output; correctness pinned by unit tests |
| T-sdk-05 | Repudiation | Step ordering | mitigate | Each step = one commit with clear message; per-step greenness provable by git bisect to any step |
| T-sdk-SC | Tampering | package installs | n/a | No new packages — pure Go stdlib only (project constraint) |
</threat_model>

<verification>
Per-step verification (each commit independently):
- `go build ./...` — compiles clean
- `go vet ./...` — no vet warnings
- `go test ./internal/ir/... ./internal/analysis/loopforest/... ./internal/opt/tiling/... ./internal/opt/gvn/... ./internal/opt/schedule/... ./internal/opt/memopt/... ./internal/cg/cxx/...` — targeted packages pass

Final verification (after all 7 commits):
- Full repo hermetic: `go test ./internal/...` passes
- Golden regression: matmul_store.cu and reduce_warp.cu are byte-identical (C5)
- E2E gate: TCACOMP_E2E=1 go test ./internal/e2e/ -run Frontend passes tile-lower §8 (downstream walls accepted per §5 contingency)
</verification>

<success_criteria>
- 7 commits in sequence, each leaving the tree green (go build + go vet + targeted tests pass)
- ForLoop body slot removed from NewForLoop signature, kind.go, catalog.go, print.go, loopspec.go
- LoopSpec.BodyRoot method deleted; interface is IsLoop + ParentEdge only
- loopforest derives LIR nesting from IV data-dep closure, not BodyRoot
- tile-lower checkFillPlacement: ORDERING (chain predecessor) + LAYERING (no inside-IV), no body-reachability
- GVN lambdaBodyRoots deleted; no body-root exemption
- Schedule pass uses loopforest IV-membership relation
- memopt removes unconsumed buffer stores + source loads (D-f); DCE untouched
- matmul_store.cu + reduce_warp.cu goldens byte-identical (C5)
- Positive + negative test pins pass
- Full repo hermetic green
</success_criteria>

<output>
Create `.planning/quick/261024-sdk-fix-bug-described-in-doc-fix1-md/261024-sdk-SUMMARY.md` when done
</output>
