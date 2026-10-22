---
phase: 261021-ta3-implement-final-optimization-pass-schedu
reviewed: 2026-10-21T23:05:00Z
depth: deep
files_reviewed: 9
files_reviewed_list:
  - internal/ir/astcfg.go
  - internal/ir/astcfg_test.go
  - internal/ir/node.go
  - internal/opt/schedule/nest.go
  - internal/opt/schedule/order.go
  - internal/opt/schedule/scheduler.go
  - internal/opt/schedule/scheduler_test.go
  - internal/opt/registry.go
  - internal/opt/gvn/gvn_test.go
findings:
  critical: 2
  high: 2
  medium: 3
  low: 4
  info: 3
  total: 14
status: issues_found
---

# Quick Task 261021-ta3: Code Review Report

**Reviewed:** 2026-10-21T23:05:00Z (commits 9a423cb, efe3ce6, b1ba203 — range 05674fe..b1ba203)
**Depth:** deep (per-file + cross-file: tile-lower repointChains/buildNest/lower_cache F8, effect.go/poset.go/walk.go read faces, final_lower.md contract)
**Files Reviewed:** 9
**Status:** issues_found
**Tree:** `go build ./...` and `go test ./internal/...` green before and after review (review-only; no source changes made)

Verification method: findings CR-01/CR-02/HI-01/HI-02 were confirmed empirically with temporary
scratch tests in `internal/opt/schedule/` (written, run, deleted; final tree re-verified green),
plus a full default-pipeline run (tiling → tile-lower → schedule) on the matmul shape, including
a K=32768 multi-round variant.

## Summary

The rewrite is well-structured and its deterministic machinery (forest derivation, anchoring,
fixpoint propagation, per-rank Kahn topo with allocation tiebreak) checks out — no map-iteration
order leaks into output, single placement holds on chain nests, and the fail-closed entry gates
work. However, the pass's core placement decision — the containment clamp
`SN = min(propagated, deepest-enclosing-loop-depth)` with rank 0 mapped to the FIRST LNT's
prologue — is unsound in two ways that are provably reachable:

1. **In the real pipeline, today**: tiling's cache-insertion fills are loop members via effect
   chains (F8), NOT via the body-operand subtree; `bodySetOf` excludes ImpactBy edges, so every
   fill has containment cap 0 and the clamp hoists it to the once-only kernel prologue even when
   its coordinates depend on a depth-3 loop IV (verified: fill SN=0 while its own operand
   dimvar has SN=3). Multi-round K / fill-key-epoch kernels get stale SMEM windows → wrong
   numerical results once final_lower/codegen consumes the AST.
2. **As a class**: any unenclosed node whose value or effect dependency is in-loop is scheduled
   BEFORE that dependency (the prologue executes first). This includes the reduction shape that
   final_lower.md's §"Reducer Lower Integration" explicitly builds on, and the kernel end/Return
   node of EVERY kernel. The committed tests pin this inversion as intended behavior.

No fail-closed net validates cross-block ordering — the leftover net only checks that a node has
*some* block — so all of the above is silent.

## Critical Issues

### CR-01: Operand-only containment + clamp hoists per-epoch SMEM fills out of their host loops (in-pipeline wrong schedule)

**File:** `internal/opt/schedule/scheduler.go:163-170, 199-208`; `internal/opt/schedule/nest.go:96-123`
**Issue:** `bodySetOf` computes containment as the transitive **operand** closure from each loop's
`ops[2]` and explicitly excludes ImpactBy edges ("containment is operand structure only"). But
tile-lower's cache insertion (PHASE 4, `internal/opt/tiling/lower_cache.go:38-47`) deliberately
places fills OUTSIDE the body-operand subtree — "the fill is placed [via the F8 splice] NOT by
making it the body operand subtree; the fill becomes a member through the pass-built [effect]
chains … the fill re-executes per fill-key epoch". The two passes hold contradictory models of
loop membership. Consequence: every fill's containment cap is 0, and the clamp at
scheduler.go:201-208 pulls its SN back to 0 no matter how deep its dependencies sit.

**Concrete failure scenario (verified, full default pipeline, matmul with K=32768):**
the SMEM fill store lands at `SN=0` in the kernel prologue Group while its own data dependency
`dimvar#125` (the K between-round loop IV) is anchored at `SN=3`. The fill must execute once per
round; the schedule executes it exactly once, before the nest. Round 2+ reads the stale SMEM
window → wrong numerical results. This also silently breaks lower_cache.go's documented F8
contract ("re-executes per fill-key epoch") and R1 ("runs once per round"). Even the single-round
K=1024 pipeline dump shows the fill in the prologue (`memstoresmem#130[sn=0]`), which is only
accidentally correct while the fill happens to be epoch-invariant.

**Fix:** Containment must agree with tile-lower's membership model. Options (any one):
- compute the cap from the loop bodies' **data+payload** closure (the same edge set
  `lower_rewrite.reachableFrom` uses — `DataDeps` ∪ `ImpactBy`), not operand-only; or
- fail closed when `propagated SN > cap` for any node (the clamp currently converts a
  structural inconsistency into silent misplacement); or
- record each fill's mandated host loop in the preserved tiling-record and honor it during
  placement instead of deriving containment.
Also add a post-placement invariant check: for every live node, its block must not execute
before any of its `PosetDeps`-containing block (see CR-02's net suggestion).

### CR-02: The clamp schedules unenclosed nodes BEFORE their in-loop value/effect dependencies — reduction shape, post-nest stores, and the Return of every kernel

**File:** `internal/opt/schedule/scheduler.go:199-208, 267-293`
**Issue:** Rank 0 means "the very first prepended Group BB" (spec 3.1) — a block that executes
BEFORE every loop. A node whose containment cap is 0 but whose propagated SN ≥ 1 (its value or
effect dependency is inside a loop) is clamped to 0 and emitted in the FIRST LNT's prologue,
i.e., before the loop that produces its dependency. The spec's max-propagation exists precisely
to prevent this; the clamp systematically undoes it for unenclosed consumers.

**Concrete failure scenarios (all verified with scratch tests):**
- **Reduction shape (review-focus 2a):** `R = n * 2` at kernel level consuming in-loop `n` —
  R lands at SN 0 (prologue) while `n` sits at SN 1 inside the ThrLoop. This is exactly the
  shape final_lower.md §"Reducer Lower Integration" depends on ("Using Reducer's own SN to learn
  its tighted Loop BB — OR using reducer's first operand"): the reducer's own SN (0) and its
  operand's SN (1) now disagree, and any emission at the reducer's block reads the operand
  before it is computed.
- **Post-nest store (review-focus 2b):** a kernel-level store chained (ImpactBy) after an
  in-loop store on the same tensor — the per-tensor effect order tile-lower's repointChains
  establishes ("after the whole loop") is inverted; the emitted program leaves the LOOP's value
  in memory instead of the later store's. The probe dump additionally shows a data-flavored
  instance: `memstore#11[sn=0]` in the prologue whose value `binop#10[sn=1]` stays in the loop.
- **The end/Return node (every kernel, in-pipeline today):** the builder wires
  `Return.values = output stores` (pkg/builder.go Build) and repointChains wires
  `Return.ImpactBy → nest roots`, so the Return always has in-loop deps, always clamps to 0,
  and always lands in the first prologue. `scheduler_test.go:213-215` ASSERTS this
  ("the unenclosed end node clamps to rank 0 (the prologue)") — the inversion is pinned as
  intended behavior.
- **Multi-LNT:** the rank-0 collection (`if rank == 0 || owner[n] == r`, scheduler.go:276)
  claims ALL rank-0 nodes for the FIRST root, so a node whose deps live in a LATER nest is
  emitted before that nest. Later LNTs' prologues can never receive nodes (pinned by
  `TestMultipleLNTs:331-333`) and **epilogues are permanently empty** — no code path ever
  appends to them, despite the epilogue being the natural landing pad for exactly these
  "after the whole nest" nodes.

No fail-closed net catches any of this: the leftover net (scheduler.go:304-313) only checks
`SchedBlock() != nil`, never cross-block ordering.

**Fix:** distinguish "invariant" (propagated SN 0) from "out-of-structure consumer"
(propagated SN ≥ 1, cap 0). Land the latter after the deepest nest their dependencies live in —
minimally the LAST root's epilogue Group, ideally the epilogue of the nest owning their deepest
dep — or fail closed on `cap(n) < max(dep SN)` instead of clamping. Additionally add a
post-placement ordering net: for each live node and each live `PosetDeps` def, assert the def's
block is not scheduled after the user's block in execution order (block position is derivable
from the AST walk). Update `TestSingleNestLICM`'s end-node assertion accordingly.

## Warnings

### HI-01: Branched nests — `byDepth` conflates sibling loops at the same depth, silently cross-producting their bodies (verified)

**Severity:** HIGH
**File:** `internal/opt/schedule/scheduler.go:258-266, 280-283`
**Issue:** `byDepth[t.depth] = t.block` keeps only the LAST child stamped at each depth, and
`owner[n]` records only the nest ROOT (`nestRoot(t)`), not the owning loop. With two sibling
loops at the same depth under one parent, both siblings' rank-r nodes target one sibling's
block. Verified with a scratch test: outer CTA loop with two inner Seq siblings — BOTH
siblings' stores landed in sibling-2's block; sibling-1's loop block was left empty. Sibling
A's body executes inside sibling B's iteration space (cross product), silently — no error.
Not reachable through the current pipeline (tile-lower rejects branched LNTs fail-closed, F3),
but the pass fail-closes other shape violations (unreachable ForLoop, live KindIter), so the
inconsistent defensiveness is a trap for direct consumers (final_lower, tests, future pass
changes that lift F3).

**Fix:** Either reject branched nests fail-closed in `forLoopForest`/`Run` (mirroring F3), or
make ownership record the owning LOOP (not the root) and place nodes into
`ownerLoop.block` instead of `byDepth[rank]`.

### HI-02: LICM hoists effectful nodes (loads/stores/allocations) — the scalar-only whitelist of the old scheduler is gone, and effect.go's audit note is now stale

**Severity:** HIGH
**File:** `internal/opt/schedule/scheduler.go:176-208`; `internal/ir/effect.go:100-105`
**Issue:** Any live node whose propagated SN is 0 is hoisted to rank 0 regardless of effect
verdict. Verified: a `MemStore` with invariant operands inside a loop body is hoisted to the
prologue, leaving the loop block empty. The deleted statement-level scheduler restricted LICM
to `licmHoistableKinds` (scalar-expression kinds only); `internal/ir/effect.go:102-105` still
asserts "the scheduler's LICM calls sit strictly BEHIND the licmHoistableKinds whitelist" —
that claim is false for the rewritten pass. Hoisting a store is unsound when the loop may
execute zero iterations (a zero-extent bound makes the hoisted store unconditional when the
loop body never ran) and it reorders per-tensor chains relative to in-loop accesses.

**Fix:** Gate rank-0 hoisting for `EffectOf(n).Has()` nodes: only hoist effectful nodes whose
loop is guaranteed ≥ 1 trip, or don't hoist them at all (keep them in their owner loop body).
Update the stale consumer note in effect.go.

## Medium Issues

### ME-01: Test suite does not pin the fail-closed nets the SUMMARY claims are exercised, and pins the CR-02 inversion as correct

**Severity:** MEDIUM
**File:** `internal/opt/schedule/scheduler_test.go` (whole file); SUMMARY claim at `.planning/quick/261021-ta3-implement-final-optimization-pass-schedu/261021-ta3-SUMMARY.md:131`
**Issue:** The SUMMARY states the leftover net and unreachable-live-ForLoop refusal are
"Exercised by the Task-3 test suite". They are not: `TestFailClosedGates` covers only
materialized / live-iter / second-run. Untested refusal paths: unreachable ForLoop
(scheduler.go:127-132), leftover net (304-313), topoRank dependency cycle (order.go:82-95).
Also untested: shared iv/bound headers across sibling loops (the max-depth anchoring +
single-stamp logic at scheduler.go:364-370), the reduction/post-nest placement shapes (CR-02),
and any tile-lower → schedule end-to-end run (the SUMMARY admits this last one). Finally,
`scheduler_test.go:213-215` asserts the end-node-in-prologue placement, i.e., encodes CR-02's
inversion as spec.

**Fix:** Add tests for each refusal path; once CR-02's placement rule is fixed, replace the
end-node assertion with the corrected landing (epilogue/kernel tail); add one pipeline-level
matmul → schedule test (the scratch probe in this review demonstrates it runs and is stable
across two identical runs).

### ME-02: NewASTLoop is fail-closed on a non-const `soc` but silently skips a non-const `dosoc`/`loc`

**Severity:** MEDIUM
**File:** `internal/ir/astcfg.go:182-206`
**Issue:** A non-const `soc` value errors; a non-const `dosoc` or `loc` value is silently
dropped (`if v, ok := ConstValueOf(val); ok` with no else), leaving the block at the default
(-1 / 1) and propagating that default into `ASTLoopMeta` — metadata final_lower uses for
per-LNT runtime mapping and reducer specialization. The inconsistency means a corrupted
annotation is caught for one key and masked for the other two. (tile-lower always emits const
u32 values, so this is a robustness gap, not a live bug.)

**Fix:** Treat non-const `dosoc`/`loc` as errors (same message shape as the soc case), or
document the deliberate fail-soft split with a rationale.

### ME-03: The raw-HIR gate refuses `KindIter` but not `KindIterCluster`

**Severity:** MEDIUM
**File:** `internal/opt/schedule/scheduler.go:96-104`
**Issue:** The gate rejects any live `KindIter`, but a live `KindIterCluster` (the legacy
builder format; `internal/opt/tiling` never references clusters, so nothing retires them)
passes and would be scheduled as an ordinary AST statement — an abstract HIR header emitted
into the final CFG. The refusal's rationale ("raw HIR iters must have been lowered by
tile-lower") applies equally to clusters.

**Fix:** Extend the gate to `KindIterCluster` (and audit for any other HIR-only kinds that
survive tile-lower).

## Low Issues

### LO-01: Failed-run sched state is never reset; a retried Run can inherit poisoned SNs

**Severity:** LOW
**File:** `internal/opt/schedule/scheduler.go:141-208`
**Issue:** On a mid-placement error the pass correctly leaves the AST unattached, but the
`SetSched` state on nodes persists. A retry passes the `g.AST() != nil` gate, and since
propagation is monotone-up only (`if sn > n.BasePtr().SchedSN()`), a stale SN higher than the
correct value but ≤ the node's cap is never corrected (the clamp only lowers to cap). Fresh
graphs (the normal manager re-run) are unaffected.

**Fix:** At Run entry, reset sched state for all allocated nodes (`SetSched(0, nil)`), or clear
it in the error paths via defer.

### LO-02: Forest linking conflates "enclosed by" with "consumed as a value by" another nest's body

**Severity:** LOW
**File:** `internal/opt/schedule/nest.go:57-74`
**Issue:** First-container-wins links loop x under any loop y whose body-operand closure
REACHES x — including reachability via a use (x consumed as a value inside y's body, the
`NewReturnChained(..., loopA, loopB)` pattern). Such an x would be mis-nested under y and
emitted inside y's loop block (re-executed per y iteration). Not reachable from tile-lower's
constructor shapes (loops are referenced only by parent loops' ops[2] and the Return, which no
body set contains), but the doc's safety claim ("x's own subtree never contains its ancestors
by construction") is asserted, not checked.

**Fix:** Verify the link is an enclosure (x is reachable from y's ops[2] via BODY structure —
e.g., x is an ancestor-consumed operand only through loop nesting edges), or fail closed on a
loop node appearing in more than one nest's body set as a non-child.

### LO-03: Cross-nest owner/dependency mismatch can silently invert same-rank ordering

**Severity:** LOW
**File:** `internal/opt/schedule/scheduler.go:216-225, 267-293`; `internal/opt/schedule/order.go:39-42`
**Issue:** A rank-r node owned by nest B is excluded from nest A's rank-r topo collection even
when its same-rank dependency is owned by A (`owner[n] == r` fails; the dep is not in the
`inSet` closure, so no edge is recorded). If B's cluster is emitted before A's, the node runs
before its dependency. Requires cross-nest body-set sharing at rank ≥ 1, which tile-lower's
per-LNT sequential construction makes unlikely today (all of nest A's loops are allocated
before nest B's), but nothing guards it — same missing cross-block net as CR-02.

**Fix:** The CR-02 ordering net covers this. Alternatively, when collecting rank r for root r,
also require each node's same-rank PosetDeps to be co-located or fail closed.

### LO-04: New/modified files are not gofmt-clean

**Severity:** LOW
**File:** `internal/ir/astcfg.go`, `internal/ir/node.go`, `internal/opt/schedule/nest.go`
**Issue:** `gofmt -l` flags all three (struct comment-column alignment drift introduced by the
new fields, e.g., `Graph.ast` in node.go and `forNest`/`ASTItem`/`ASTKernel` alignment). The
SUMMARY's green-report (`go vet`, tests) did not include gofmt. (Several untouched repo files
share the pattern, but the new files add to it rather than fixing it.)

**Fix:** `gofmt -w` the three files.

## Info

### IN-01: Loop attribute nodes land in the first prologue while iv/bound land in the loop block

**File:** `internal/opt/schedule/scheduler.go:151-158, 364-370` vs `internal/opt/schedule/nest.go:107-121`
**Issue:** Headers are anchored (iv/bound → their Loop block), but the trailing `soc/dosoc/loc`
`KeyedAttribute` operands are not in any body set (closure starts at ops[2]) and land at rank 0
in the first prologue (visible in the pipeline dump: `keyedattribute#53/55/57[sn=0]`). Harmless
today (annotations, duplicated in `ASTLoopMeta`), but final_lower must know to look in the
prologue for them.

### IN-02: AttachAST/AppendItem accept malformed ASTItems; SetSched doc comment overclaims

**File:** `internal/ir/astcfg.go:133, 269-287, 308-311`
**Issue:** `AppendItem` accepts items with both `Node` and `Block` set or both nil; `AttachAST`
validates only the Body kind. `SetSched`'s "so the two can never disagree" is only true at
placement time — during anchoring/propagation SN is set with a nil block by design (the comment
does acknowledge the phase split, so this is a nit).

### IN-03: Untracked scratch debris in the working tree

**File:** `internal/opt/schedule/zz_scratch_probe_test.go` (untracked; not part of the reviewed commits)
**Issue:** A review-scratch probe file from a concurrent review session sits untracked in the
package. It passes tests but must not be committed. (This review's own scratch file was already
deleted; final tree re-verified green.)

---

_Reviewed: 2026-10-21T23:05:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
