---
phase: quick-261021-ta3
plan: 261021-ta3
subsystem: compiler-ir
tags: [sea-of-nodes, ast-cfg, licm, scheduling-number, forloop-nest, gsd-quick]

# Dependency graph
requires:
  - phase: 261021-o61 / 261021-jay lineage
    provides: "tile-lower's annotated KindForLoop nests (soc/dosoc/loc KeyedAttributes), the legacy ir.Graph/poset read faces, and the loopforest analysis structure mirrored by nest.go"
provides:
  - "ASTKernel/ASTBlock/ASTItem/ASTGmemAlloc/ASTLoopMeta/ASTLNT IR types (internal/ir/astcfg.go) with BlockKind Group/Seq/CTA/ThrLoop classification from the soc KeyedAttribute"
  - "Graph.AST()/AttachAST() — the separate AST CFG field (D-01), nil until the schedule pass attaches it"
  - "Base.schedSN/schedBlock + SchedSN()/SchedBlock()/SetSched() — persistent per-node scheduling state readable after the pass (D-03)"
  - "the rewritten schedule pass: ForLoop-nest forest, SN anchoring + max-propagation + containment clamp, per-LNT LBBN with prologue/epilogue Groups, per-rank stable topo placement, Params/Gmem/LNTs kernel metadata"
  - "pass-level behavior tests pinning the spec's edge cases (internal/opt/schedule/scheduler_test.go)"
affects: ["final_lower (doc/optana/final_lower.md — consumes the AST CFG, per-node SN/block linkage, and the preserved tiling-record cache)", "codegen", "reducer lowering (ASTLNT SoC metadata)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "AST-style CFG as a separate Graph field, recursive Statement=Node|Block items"
    - "scheduling-number LICM by construction: anchors at loop-header depth, max-propagation, containment clamp"
    - "body-operand containment forest for ForLoop nesting (vs the iter forest's reversed-edge linking)"

key-files:
  created:
    - internal/ir/astcfg.go
    - internal/ir/astcfg_test.go
    - internal/opt/schedule/nest.go
    - internal/opt/schedule/order.go
  modified:
    - internal/ir/node.go
    - internal/opt/schedule/scheduler.go
    - internal/opt/registry.go
    - internal/opt/gvn/gvn_test.go

key-decisions:
  - "\"Iteration node\" (spec 3.1) anchors the loop's header — the ForLoop node plus its iv/bound operands — at the loop's nest depth, fixed during propagation; a body node only reaches the anchor through def chains (the loop→body edge is use→def), and the iv is exactly what loop-varying index math depends on."
  - "A node's final SN is clamped to its deepest enclosing loop depth (unenclosed nodes clamp to 0): unclamped max-propagation misplaces the cross-nest effect-chained nodes tile-lower's repointChains creates (later-LNT chain heads and the end node point at earlier nests' roots); the clamp keeps SN consistent with the enclosing structure and makes the ownership assignment provably total."
  - "Ownership: a rank-r node belongs to the LNT of the FIRST loop (allocation order) at depth r whose body subtree contains it — containment, not mere closure membership, is the correctness-bearing condition."
  - "Absent soc fails soft to BlockSeqLoop (the tiling.SoCType zero-value convention); soc outside {0,1,2} fails closed."

patterns-established:
  - "Pass-state cells on ir.Base (schedSN/schedBlock) follow the ty discipline: never printed/serialized/verified."
  - "Forest membership is non-erased AND reachable: an unreachable live ForLoop is refused fail-closed rather than scheduled as garbage."

requirements-completed: [QUICK-261021-ta3]

# Metrics
duration: 26min
completed: 2026-10-21
---

# Quick 261021-ta3: Final optimization pass schedule (AST-style CFG + SN LICM) Summary

**Sea-of-nodes → AST-style CFG (Kernel → Group/Seq/CTA/ThrLoop blocks) with scheduling-number LICM: persistent per-node SN + block linkage on ir.Base, replacing the deleted statement-level scheduler as the final_lower transition point**

## Performance

- **Duration:** ~26 min
- **Started:** 2026-10-21T13:42:30Z
- **Completed:** 2026-10-21T14:08Z
- **Tasks:** 3/3
- **Files modified:** 8 (4 created, 4 modified; the old scheduler.go rewritten in place and scheduler_test.go deleted+replaced)

## Accomplishments

- The `schedule` pass (priority 900, Requires `tile-lower`, Preserved `tiling-record`) consumes the Level-1 sea after tile-lower and attaches a complete ASTKernel — Body clusters [prologue Group, LBBN root, epilogue Group] per LNT in forest-root allocation order, Params, Gmem allocations with extents, and per-LNT SoC metadata — without touching the legacy cfg/Materialize/printer/goldens (D-01) and without inserting runtime-object/glue nodes (D-04).
- LICM by construction: loop/header nodes anchor at nest depth, max-propagation lifts loop-varying nodes to their rank, invariants stay at SN 0 and land in the LNT prologue Group; every live node carries its persistent SN + landing block on `ir.Base` for final_lower (D-03).
- Fail-closed surface: nil/materialized/already-attached graphs, live raw KindIter, unreachable live ForLoops, rank dependency cycles (naming stuck IDs), and any live node landing in no block all refuse deterministically with the AST left unattached.

## Task Commits

Each task was committed atomically (code only):

1. **Task 1: IR structures — ASTKernel/ASTBlock types + persistent SN/block fields** - `9a423cb` (feat)
2. **Task 2: The schedule pass — delete the old scheduler, build the AST CFG from the sea-of-nodes** - `efe3ce6` (feat)
3. **Task 3: Pass-level scheduling tests — the spec's edge cases** - `b1ba203` (test, incl. the iv/bound header-stamping fix found by the new tests)

## Files Created/Modified

- `internal/ir/astcfg.go` — ASTKernel/ASTBlock/ASTItem/ASTGmemAlloc/ASTLoopMeta/ASTLNT types, NewASTGroup/NewASTLoop (soc classification + defaults + fail-closed soc error), Graph.AST()/AttachAST(), Base.SchedSN()/SchedBlock()/SetSched()
- `internal/ir/node.go` — Graph.ast field (D-01) and Base.schedSN/schedBlock fields
- `internal/ir/astcfg_test.go` — construction/accessor/classification/AttachAST-refusal/SetSched tests
- `internal/opt/schedule/nest.go` — Step-0 ForLoop-nest forest (collect/link-by-body-containment/depth DFS), body-set computation, maxDepth
- `internal/opt/schedule/scheduler.go` — the rewritten pass: gates, live set, anchors, propagation, clamp, ownership, LBBN + prologue/epilogue, rank placement, kernel metadata, AttachAST
- `internal/opt/schedule/order.go` — per-rank stable Kahn topo with allocation-order tiebreak and the cycle fail-closed error
- `internal/opt/schedule/scheduler_test.go` — the 9 behavior groups (old statement-level tests deleted with the old pass, D-02)
- `internal/opt/registry.go`, `internal/opt/gvn/gvn_test.go` — priority-scale and stale interop comment updates only

## Decisions Made

- **Anchor set = ForLoop + its iv/bound header** at the loop's nest depth (see Deviations): without header anchoring the spec's propagation lifts nothing — a body node's dependency closure never contains its loop (the loop→body edge is use→def), and loop-varying index math depends on the iv.
- **Containment clamp SN = min(propagated, deepest-enclosing-loop-depth)**: keeps SN and structure consistent on the real tile-lower output (repointChains wires end/chain-heads at nest roots via ImpactBy, which unclamped propagation would lift out of place) and makes ownership totality provable (a nest's containing loops chain through every depth 1..cap).
- **Ownership by first allocation-order rank-matching loop whose body contains the node** — deterministic, single placement, no duplication.
- **The iv/bound header lands in its Loop block** (stamped during LBBN construction, deepest sharing loop wins for shared headers) — the header is part of the spec's loop structure section 1.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Anchor the loop header (iv/bound) — the plan's literal "ForLoop-only" anchoring lifts no body node**
- **Found during:** Task 2 design review against the real tile-lower output shape (verified in `internal/opt/tiling/lower_rewrite.go` repointChains/buildNest: body stores never depend on their enclosing ForLoop; the loop→body edge is use→def).
- **Issue:** With only ForLoop nodes anchored at depth, every body node's SN stays 0 under the spec's max-propagation (its dependency closure contains no anchored node — the iv it actually depends on was never anchored), so all loop bodies would land in the prologue and the loop blocks would be empty — contradicting the plan's own truths ("the varying node gets SchedSN 1", "consumes the sea-of-nodes after tile-lower"). The spec's "iteration node" is read as the iteration-header unit: ForLoop + iv/bound (ops[0]/ops[1]).
- **Fix:** Anchor ops[0]/ops[1] at the loop's nest depth alongside the loop node (max-wins for shared headers); propagate from there.
- **Files modified:** internal/opt/schedule/scheduler.go
- **Verification:** TestSingleNestLICM (varying node SN 1 in ThrLoop), TestNestedNestSinglePlacement (shared node SN 2), full repo green.
- **Committed in:** efe3ce6 (T2)

**2. [Rule 1 - Bug] Clamp propagated SN to the deepest enclosing loop depth; ownership by containment, not closure membership**
- **Found during:** Task 2 design review of the multi-LNT chain wiring repointChains emits (later-LNT chain heads and the end node carry ImpactBy edges at earlier nests' roots).
- **Issue:** Unclamped max-propagation gives those unenclosed/shallow nodes a rank their structure cannot host (e.g. the end node would land inside a loop block); the plan's ownership rule ("first loop with matching SN in the node's closure") would also misplace cross-nest-chained body stores into the wrong LNT. Under the clamp, ownership = first allocation-order rank-matching loop whose BODY contains the node, which is provably total (a nest's containing loops chain through every depth 1..cap).
- **Fix:** SN = min(propagated, containment cap); ownership requires body containment. All nine plan test groups pass unchanged under this reading.
- **Files modified:** internal/opt/schedule/scheduler.go
- **Verification:** TestMultipleLNTs (cluster order + shared invariant placement), TestEffectOnlyChain, the leftover fail-closed net (unplaceable nodes name themselves), full repo green.
- **Committed in:** efe3ce6 (T2)

**3. [Rule 1 - Bug] Stamp the anchored iv/bound header into its Loop block during LBBN construction**
- **Found during:** Task 3 — TestSingleNestLICM et al. tripped the new leftover fail-closed net ("2 live nodes landed in no block"): the rank placement never claims anchored header nodes, so their SchedBlock stayed nil.
- **Fix:** buildLBBN stamps ops[0]/ops[1] with the loop's own block (the loop whose depth equals the header's max-wins anchored SN — the deepest sharing loop).
- **Files modified:** internal/opt/schedule/scheduler.go
- **Verification:** All schedule tests pass; the `iv/bound land in the Loop block` assertion added to TestSingleNestLICM pins it. Committed in b1ba203 (T3).

**4. [Rule 2 - Missing critical] Fail-closed net for unplaced live nodes + unreachable-live-ForLoop refusal**
- **Found during:** Task 2 (threat model T-ta3-01/03 hardening).
- **Issue:** The plan's algorithm had no explicit guard for a live node that ends up in no block (future reduction shapes chaining outer-level reads after inner stores would silently vanish from the AST CFG) nor for a non-erased ForLoop unreachable from the sea roots.
- **Fix:** After placement, any live node with nil SchedBlock fails closed naming the sorted IDs; forest members not in the Walk live set are refused by name. The graph's AST is never partially emitted.
- **Files modified:** internal/opt/schedule/scheduler.go
- **Verification:** Exercised by the Task-3 test suite (the leftover net caught deviation #3 during development). Committed in efe3ce6 (T2).

---

**Total deviations:** 4 auto-fixed (3 bugs, 1 missing-critical)
**Impact on plan:** All four are correctness requirements for the plan's own stated truths and tests on the real tile-lower output shape; no scope creep. The plan's mechanisms (anchor/propagate/own/place) are intact — the deviations tighten the anchor set, clamp the propagation to structure, and add the two fail-closed nets the threat model demands.

## Issues Encountered

- The design spec (doc/optana/schedule.md) exists only as an untracked file in the main checkout, not in this worktree — read from the main checkout path for reference; not committed here (docs artifacts are the orchestrator's responsibility).
- The old `internal/opt/schedule/` directory disappeared entirely after `git rm` (both files were its only contents); recreated with the three new files.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- final_lower (doc/optana/final_lower.md) can now consume `g.AST()`: the Body clusters, per-node `SchedSN()`/`SchedBlock()` linkage, `ASTKernel.Gmem` (launcher buffers), and `ASTLNT` SoC metadata are all in place; the preserved `tiling-record` cache entry carries the per-LNT runtime-mapping data (ir does not import tiling, per the plan contract).
- Deferred/future shapes the fail-closed nets will surface rather than mis-schedule: reduction-style effect chains crossing loop depths (outer-level reads chained after inner stores) need reducer lowering semantics before they can be placed.
- Not run end-to-end: no pipeline test executes tile-lower → schedule on a real lowered graph yet (all BuildPipeline usages are presence/order checks); wiring that integration test is natural follow-up work for the final_lower phase.

## Self-Check: PASSED

- internal/ir/astcfg.go, internal/ir/astcfg_test.go, internal/opt/schedule/{scheduler,nest,order,scheduler_test}.go exist (created/modified as claimed).
- Commits 9a423cb (T1), efe3ce6 (T2), b1ba203 (T3) present on `worktree-agent-a2af708a62043628d`.
- `go build ./... && go vet ./... && go test ./...` — full repo green (exit 0); legacy Materialize/golden tests untouched and passing; `grep -rn internal/opt/schedule` shows only the blank import in all.go (plus a doc comment and the registry test import — no import breaks).

---
*Plan: quick-261021-ta3*
*Completed: 2026-10-21*
