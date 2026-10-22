---
phase: 261022-b2m-fix-schedule-pass-per-261021-ta3-code-re
plan: 01
subsystem: compiler-opt-scheduling
tags: [schedule-pass, ast-cfg, licm, effect-chains, epilogue, fail-closed, go]

requires:
  - phase: 261021-ta3-implement-final-optimization-pass-schedu
    provides: the schedule pass (sea-of-nodes → AST CFG) whose review findings this task fixes
  - phase: 261021-o61-fix-tile-lower-per-review-rulings-runtim
    provides: the tile-lower fill/RAUW/repointChains machinery the uniform relation mirrors
provides:
  - uniform-relation placement model (two-case rule, effectful floor with allocation exemption, epilogue class with home resolution)
  - fail-closed cross-block ordering net (net.go verifyBlockOrder — leftover subsumed)
  - fail-closed guards: branched nests, KindIterCluster, non-const dosoc/loc, stale-sched reset
  - ASTBlock.Nest() back-pointer API + NewASTGroupInNest + ASTKernel.LNTs []*ASTLNT (ruling 7)
  - complementary SchedSN/SchedBlock semantics (ruling 6) consumed by final_lower's reducer dual route
  - committed full-pipeline matmul integration test incl. the CR-01 K=32768 fill-inside-host regression
affects: [final_lower, codegen, tiling, schedule]

tech-stack:
  added: []
  patterns:
    - "uniform membership relation: DataDeps + ImpactBy defs (mirrors tiling reachableFrom) for cap AND ownership, operand-only closure kept for forest linking"
    - "two-case placement without clamping: in-loop at SN vs epilogue-of-contributing-nest (LAST contributing root)"
    - "post-placement empirical net: linear block index + PosetDeps order check with the loop-body containment carve-out"

key-files:
  created:
    - internal/opt/schedule/net.go
    - internal/opt/schedule/pipeline_test.go
    - internal/opt/schedule/order_net_internal_test.go
  modified:
    - internal/opt/schedule/scheduler.go
    - internal/opt/schedule/nest.go
    - internal/opt/schedule/scheduler_test.go
    - internal/ir/astcfg.go
    - internal/ir/astcfg_test.go
    - internal/ir/node.go
    - internal/ir/effect.go

key-decisions:
  - "ImpactBy treated identically to input edges throughout the scheduling model (cap + ownership uniform; linking stays operand-only because repointChains makes loop nodes uniform-reachable from foreign nests)"
  - "No clamp: SN <= capUni places at SN; SN > capUni places in the contributing nest's epilogue keeping the propagated SN (depth requirement) — tile-lower already encodes inside-vs-after on the edge target"
  - "HI-02 floor exempts allocations (MemNew/MemNewSMem/FragNew): shared declarations hoist; only memory-ACCESS kinds floor at operand containment"
  - "Epilogue home = LAST contributing root among live PosetDeps homes (ForLoop dep → its root; in-loop dep → owner; epilogue dep → its home), resolved by a bounded deterministic fixpoint"
  - "The cross-block net carves out exactly the ForLoop→body containment edge; statements keep the full guard (SN propagation makes statement dep-in-own-subtree impossible)"

patterns-established:
  - "Empirical-guard net: a placement model gets a machine-checked invariant (linear block order vs PosetDeps), not just a proof"
  - "Fail-closed shape gates mirror the producer pass's contracts (branched nests ↔ tile-lower F3; KindIterCluster ↔ KindIter)"

requirements-completed: [CR-01, CR-02, HI-01, HI-02, ME-01, ME-02, ME-03, LO-01, LO-03, LO-04]

duration: 13min
completed: 2026-10-22
status: complete
---

# Quick Task 261022-b2m: Fix schedule pass per 261021-ta3 code review Summary

**Uniform-relation placement (two-case rule, no clamp) with epilogue landing zones, an effectful floor with allocation exemption, and a fail-closed cross-block ordering net — the K=32768 SMEM fill now executes once per round inside its host loop (was: hoisted to the once-only prologue), and every unenclosed consumer (reduction shapes, post-nest stores, the end node of every kernel) executes after its producing nest.**

## What Was Built

### Task 0 (prep commit a406225) — LO-04 pulled forward

gofmt of all nine flagged files (the six pre-existing ones the plan listed for
Task 4 — internal/analysis/doc.go, internal/ir/effect_test.go, internal/ir/kind.go,
internal/ir/ssa.go, internal/opt/memopt/memopt_test.go, pkg/expr.go — plus Task 1's
three touched files). Whitespace/alignment/doc-comment-list normalization only;
no semantic tokens changed. Pulled ahead of T4 so the orchestrator's
every-commit gate (`gofmt -l internal pkg` empty) holds from the first task
commit onward.

### Task 1 (28c7fec) — uniform-relation placement model + fail-closed guards

- **nest.go**: new `reach` field + `reachSetOf` — the transitive closure from
  ops[2] over `ir.DataDeps` ∪ `ir.ImpactBy` defs, the exact relation of tiling's
  `lowerRun.reachableFrom` (lower_rewrite.go:498; the plan cited lower_cache.go —
  same function). Cap AND ownership use `.reach`; forest LINKING stays on
  `.body` with the documented reason (repointChains re-points post-nest
  consumers' effect edges at foreign nest roots, so linking on `.reach` would
  cross-nest the forest; LO-02 stays unfixed by design).
- **scheduler.go**: the containment clamp is deleted. Two caps (capOp from
  `.body`, capUni from `.reach`); SN propagation maxed over DataDeps+EffectDeps
  (design intent, cited); HI-02 floor (effectful non-allocation nodes floor at
  capOp) followed by a second propagation so floors flow to consumers;
  two-case classification (SN ≤ capUni → in-loop at SN; SN > capUni → epilogue
  class); uniform ownership (first-claim at depth == final SN over `.reach`);
  epilogue home resolution (LAST contributing root, bounded deterministic
  fixpoint); per-root placement builds the `*ir.ASTLNT` record FIRST, creates
  the pads via `ir.NewASTGroupInNest(lnt)`, places ranks 0..maxDepth, then
  collects the root's epilogue nodes (allocation order, topoRank with
  maxDepth+1 as a diagnostic label only) stamping `SetSched(sn, epilogue)`
  KEEPING each node's propagated SN (ruling 6). Entry guards: stale-sched
  reset (LO-01), KindIterCluster refusal (ME-03), branched-nest refusal
  (HI-01, mirrors tile-lower F3).
- **ir/astcfg.go**: NewASTLoop fails closed on non-const `dosoc`/`loc` (ME-02,
  all three keys); `ASTBlock.nest` + `Nest()` accessor; `NewASTGroupInNest`;
  `ASTKernel.LNTs []*ASTLNT`; epilogue-contract + complementary SN/block docs.
- **ir/node.go**: schedSN/schedBlock field comments refreshed to ruling 6.
- **ir/effect.go**: stale "licmHoistableKinds whitelist" consumer note rewritten
  (the schedule pass gates effectful hoisting itself now).

### Task 2 (d530d93) — post-placement cross-block ordering net

`net.go verifyBlockOrder(k, nodes, live)`: flattens the kernel's blocks into a
linear execution index (DFS entry order), then fail-closes on live leftovers
(subsumes the old net, same message shape) and on any cross-block inversion
(user's block executing before a live PosetDeps def's block — first violation
in deterministic scan order, naming both IDs and kinds; equal indices legal).
Wired post-placement pre-attach (ASTKernel built first, verified, then
attached); the old leftover-only block is deleted. Plus the loop-body
containment carve-out and a foreign-block fail-closed (see Deviations 3-4).

### Task 3 (95b8ad5) — updated + new pins

Updated in place (with T1, see Deviation 2): TestSingleNestLICM end-node,
TestMultipleLNTs end-node/epilogues. New: TestReductionShapeEpilogue (THE
ruling-4/6 dual-route pin), TestPostNestStoreEpilogue, TestEffectfulNoHoist
(HI-02 + allocation exemption), TestBranchedNestRefusal (HI-01),
TestIterClusterGate (ME-03), TestUnreachableForLoopRefusal (ME-01),
TestSchedStateReset (LO-01, byte-identical dumpAST vs clean twin),
order_net_internal_test.go (topoRank self-dependency cycle via
MutContext.SetDef; verifyBlockOrder inversion + leftover),
astcfg_test.go TestASTLoopNonConstAttrsFailsClosed + TestASTGroupNestAccessor.

### Task 4 (ed7cbd1) — committed full-pipeline integration test

`pipeline_test.go`: one clean builder `matmulGraphK(name, kExtent)` (u32
extents 1024/1024/kExtent, M→N→K iter chain, IV-wired lhs[m][k]/rhs[n][k],
chained store of lhs·rhs over [m,n,k]); matmulGraphBounded NOT ported;
dumpAST lives in scheduler_test.go (same package). Pins:
TestPipelineMatmulSchedule (success, twin-run byte-identical dumps,
single placement + linkage, one LNT, six-loop CTA>Seq>Thr>Thr>Seq>Seq chain,
6 params / 0 gmem / 1 lnt); TestPipelineFillInsideHost (THE CR-01 regression
at K=32768: exactly one live SMEM fill; fill SN 3 inside its K-between host
ThrLoop whose loop node also sits at SN 3; all live fill deps ≤ 3; src
KindMemLoad at SN 1; KindMemNewSMem buffer at SN 0 in the FIRST prologue; end
node in the LAST LNT's epilogue); TestPipelineEndNodeEpilogue (rulings 6-7:
end in the last cluster's epilogue at SN 6 — the observed stable propagated
depth requirement — nest recoverable via Nest() by Root identity).

The scratch probe zz_scratch_probe_test.go never existed in this worktree
(uncommitted main-tree file; the orchestrator handles the deletion).

## Verification

- `go build ./... && go vet ./... && go test ./... -count=1` — full repo green
  before EVERY commit (the orchestrator's gate), and at HEAD.
- `test -z "$(gofmt -l internal pkg)"` — clean.
- CR-01 pin: K=32768 fill at SN 3 inside its host Loop block (was SN 0 prologue).
- CR-02 pins: reduction shape, post-nest store, end node in epilogue Groups
  after their producing nests (end node: LAST LNT), keeping propagated SNs.
- Determinism: twin pipeline runs render byte-identical AST dumps.
- All seven must-have truths are pinned by named tests (see each test header).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - blocking gate] LO-04 gofmt pulled forward from T4 into a prep commit**
- Found during: pre-Task-1 gate check
- Issue: the orchestrator's gate requires `gofmt -l internal pkg` empty before
  EVERY commit, but the plan scheduled the six pre-existing files' gofmt for
  Task 4 — Tasks 1-3 commits would have failed the gate.
- Fix: separate style prep commit a406225 gofmt-ing all nine flagged files
  (six pre-existing + T1's three touched files) before T1.
- Files: internal/analysis/doc.go, internal/ir/{astcfg.go, effect_test.go, kind.go, node.go, ssa.go}, internal/opt/memopt/memopt_test.go, internal/opt/schedule/nest.go, pkg/expr.go
- Commit: a406225

**2. [Rule 3 - blocking gate] Two stale CR-02 pins updated in T1 instead of T3**
- Found during: Task 1 verification
- Issue: the plan allowed TestSingleNestLICM/TestMultipleLNTs failures to
  remain until Task 3, but the every-commit gate requires a green suite.
- Fix: the two pin updates specified by Task 3 were pulled into T1's commit;
  Task 3 added all the NEW tests unchanged.
- Files: internal/opt/schedule/scheduler_test.go
- Commit: 28c7fec

**3. [Rule 1 - bug] net.go loop-body containment carve-out**
- Found during: Task 2 verification (TestNestedNestSinglePlacement failed)
- Issue: the parent ForLoop's dependency on its child loop (the body operand)
  is CONTAINMENT (the body runs inside the loop), not before-use ordering —
  the naive index rule fired a false positive (parent loop block index < child
  block index).
- Fix: a dep whose block lives in the user loop-block's subtree is legal when
  the user is a ForLoop. Statements keep the full guard: SN propagation makes
  a statement with a dep in a strictly deeper block of its own subtree
  impossible, so no real inversion is masked.
- Files: internal/opt/schedule/net.go
- Commit: d530d93

**4. [Rule 2 - soundness] net.go foreign-block fail-closed**
- Found during: Task 2 implementation
- Issue: a live node whose SchedBlock is not part of the flattened kernel AST
  would read a default index of 0 and could mask real inversions.
- Fix: explicit membership check — "landed in a block outside the kernel AST".
- Files: internal/opt/schedule/net.go
- Commit: d530d93

**5. [Plan-detail] epilogue-home fixpoint bound is len(nodes)+1, not len(roots)+1**
- The plan's stated bound can spuriously fail closed: epilogue CONSUMER
  CHAINS (st3 after st2 after an in-loop store) resolve one layer per pass
  and can exceed the root count. Implemented as until-stable with the
  len(nodes)+1 hard terminator; the defensive no-home failure is unchanged.
- Files: internal/opt/schedule/scheduler.go
- Commit: 28c7fec

**6. [Plan-detail] verifyBlockOrder runs on a constructed ASTKernel**
- The plan's wiring point ("before the kernel metadata/AttachAST") and its
  signature (*ir.ASTKernel) required building the kernel struct first, then
  verifying, then attaching. Behavior identical: an error still leaves the
  AST unattached.
- Files: internal/opt/schedule/scheduler.go
- Commit: d530d93

**7. [Reference nit] reachableFrom lives at lower_rewrite.go:498**
- The plan cited lower_cache.go:498; the function (and its repointChains
  comment at :517) is in lower_rewrite.go. nest.go's comment cites the actual
  file.

## Out of Scope (follow-ups, untouched)

- The tile-lower side-finding: a matmul variant with K-less store axes fails
  fill placement (a tiling bug, not a schedule bug).
- LO-02: forest-link conflation via value consumption (linking stays
  operand-only by design, documented in nest.go).
- IN-01: loop-attr nodes land in the first prologue while iv/bound land in
  the loop block (final_lower must know to look in the prologue).
- IN-02: AttachAST/AppendItem robustness nits, SetSched doc nuance.

## Known Stubs

None — no stub patterns were introduced; every landing zone (epilogue Groups)
receives real placed nodes from this pass or stays empty as the documented
D-04 pad for final_lower.

## Threat Flags

None — no new trust-boundary surface: the pass reads in-process IR and emits
an AST or deterministic fail-closed errors (T-261022b2m-01/-02 mitigated as
planned; no new dependencies).

## Self-Check: PASSED

- All 10 created/modified source files exist on disk.
- All 5 commits present: a406225 (style prep), 28c7fec (T1), d530d93 (T2),
  95b8ad5 (T3), ed7cbd1 (T4).
- Full repo build/vet/test green and gofmt clean at HEAD.
- SUMMARY.md left uncommitted (the orchestrator handles the docs commit).

