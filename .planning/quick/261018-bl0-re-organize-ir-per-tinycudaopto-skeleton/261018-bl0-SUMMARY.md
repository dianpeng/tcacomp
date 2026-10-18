# Quick 261018-bl0: Re-organize the two-level IR per the tinycudaopto skeleton — Summary

**One-liner:** `internal/ir` re-layered into per-family node files with a HIR/LIR kind catalog and v3 renumber (R4), the sea-of-nodes effect model rebuilt around ONE end node and a two-root-set liveness (R1–R3), and scheduling promoted from Materialize's minimal default to an optimization pass `pass.Scheduler` with stable topo sort + LICM (R5) — goldens byte-stable except the one sanctioned `store(` → `memstore(` spelling sweep.

**Plan:** `.planning/quick/261018-bl0-re-organize-ir-per-tinycudaopto-skeleton/261018-bl0-PLAN.md`

## Tasks and commits

| Task | Commit  | What landed                                                                                          |
| ---- | ------- | ---------------------------------------------------------------------------------------------------- |
| 1    | 32cb839 | Layered node files (node_lambda/node_expr/node_var/node_memory/node_loop/node_dimvar/node_lir_memory/node_end), catalog.go (KindTier/Tier/HIR/LIR/AllKinds, `len(AllKinds()) == int(KindEnd)-1` pinned), KindVersion 3 renumber (HIR before LIR; generic Load/Store + LoadTile/StoreTile removed; SMEM/FRAG families with StorageClass), NewStore→NewMemStore with no aliases, lowercase canonical spellings, goldens regenerated ONCE |
| 2    | 9b9bac6 | One-end-node effect model (R2: first `NewReturn` is `Graph.end`; `NewReturnChained` wires every mem-param final chain state via ImpactBy), two-root-set DCE (R3: sea roots = RoleTensor params + end; stores never roots; SMEM/FRAG unobservable), poset.go read face (R1: DataDeps/EffectDeps/PosetDeps/SeaRoots), `Env.FinalStates`, `Graph.NewStatement` factory, pkg.Build single-end wiring, verify fails closed on multiple end nodes |
| 3    | ffb1497 | `pass.Scheduler` (ID "schedule", produces StageConstructed): stable Kahn topo over the statement-ownership poset → LICM → topo; LICM hoists whitelisted pure invariant computations before the enclosing nest; cycle remainder fails closed; 8 new tests incl. manager+VerifyEach integration and idempotence |

All three tasks: `go build ./... && go vet ./... && go test ./...` green (uncached), gofmt clean (excluding pre-existing `internal/ir/ssa.go` / `pkg/expr.go` violations at base — out of scope), **zero testdata drift in tasks 2 and 3** (`git status --porcelain -- '**/testdata/**'` empty before each commit).

## Task-1 golden review record (per the flipped golden policy)

Goldens regenerated exactly once in task 1 and reviewed hunk by hunk:

- `pkg/testdata/matmul_scale.ir`: every hunk is the sanctioned spelling sweep only — statement #8's stage chain now reads `memstore(%output[@m][@k], %0)`, statement #9 unchanged in shape (`return %output`; the store renders through its own access rule, so the end-node wiring of task 2 needed no further golden change).
- `internal/visualization/testdata/matmul_scale.dot`: one hunk — the store node's label `memstore(%output[@m], %0)` (kind-spelling only); all IDs, ranks, and edges unchanged.
- No numeric, ordering, ID, or structural drift anywhere.

## Deviations from plan

### Auto-fixed

**1. [Rule 1 – Bug] BinOpCode/UnaryOpCode tables dropped at base, build broken**
- **Found during:** Task 1
- **Issue:** The base commit's kind.go no longer carried the `BinOpCode`/`UnaryOpCode` operator tables (§7.6 mirror) that node_expr.go depends on; the tree did not build as-committed.
- **Fix:** Restored the full opcode tables in `internal/ir/node_expr.go` (their new home alongside the scalar-expression nodes), values and order unchanged.
- **Commit:** 32cb839

**2. [Rule 2 – Correctness] building-DCE seeds ALL graph params, not just the RoleTensor sea roots**
- **Found during:** Task 2
- **Issue:** R3's two-root set limits *observable* roots to RoleTensor params, but the def-use verifier audits every graph-owned param; seeding liveness with only the RoleTensor subset would let building DCE sweep un-erased dim/scalar params (signature debris that later passes still read).
- **Fix:** `markReachable` seeds the end node plus ALL `g.params`; the narrower SeaRoots() read face still reports the R3 observable set for analysis consumers.
- **Commit:** 9b9bac6

**3. [Rule 3 – Blocking] printer could not render pass-inserted value statements (task 3 prerequisite)**
- **Found during:** Task 3 (LICM's "Print shows %N" criterion unreachable without it)
- **Fix:** Three golden-safe extensions to `internal/ir/print.go`, all restricted so committed goldens cannot change (verified: zero drift):
  - `chainOf` seeds a one-element chain for a non-stage statement RHS (value-rooted statements did not exist before passes could create them);
  - statement-boundary rendering: a node that roots a DIFFERENT `StmtAssign` statement prints as `%N` from other statements' trees (effect roots keep their established spellings, e.g. `return %output`; DOT label path untouched);
  - `operandPrec`: operands rendering as `%N` bind as atoms, so hoisted chains print `%0 * 3.0`, never `(%0) * 3.0`.
- **Commit:** ffb1497

**4. [Rule 1 – Bug] Remove+Append reschedule round-trip destroyed stmt back-pointers**
- **Found during:** Task 3 (idempotence test: second Run duplicated hoisted statements; body rendered hoisted nodes inline)
- **Issue:** `BasicBlock.Remove` clears the root's `stmt` back-pointer (pinned behavior, cfg_test) but `Append`/`InsertBefore`/`InsertAfter` never restored it — the scheduler's collect-remove-reappend apply step silently un-owned every statement.
- **Fix:** The three link operations now maintain the back-pointer on link (Remove clears, links restore — a symmetric invariant; no existing test pins otherwise).
- **Commit:** ffb1497

### Design adjustments within task 3 (documented, no plan change needed)

- **`Base.Statement()` exported accessor added** (new ir API): the pass package must detect already-scheduled nodes (idempotence) and attribute statement ownership; reaching the unexported field from `internal/pass` is impossible. Read-only, no behavior change.
- **LICM leaf narrowing:** the plan's whitelist `{Const, Param, BinOp, UnaryOp, Select, Cast, Call}` is kept as the safety set, but candidates must also bind at least one operand — hoisting bare const/param leaves would add no computation and pollute the schedule with one-constant statements. The plan's happy path (invariant *const binop chain*) is unaffected.
- **LICM insertion point:** "immediately before the loop's statement" is refined to "immediately before the earliest statement whose tree already contains the candidate". StmtNodes order meets consumers before producers; the refinement guarantees producers land left of consumers, which keeps the statement-ownership map acyclic for the closing topo sort (a plain insert-before-loop produced an Hc1<->Hc2 ownership cycle through shared operand consts in the very first run).

## Requirement traceability

- **R1** (sea-of-nodes Input+ImpactBy poset): `internal/ir/poset.go` — DataDeps/EffectDeps/PosetDeps, pinned by poset_test.go.
- **R2** (ONE end node): `Graph.end`, `NewReturnChained`, `Env.FinalStates`; verify fails closed on a second reachable return (verify_test.go `TestVerifierFlagsMultipleEndNodes`).
- **R3** (two-root-set): `SeaRoots()`; stores live only via the end node's operand/ImpactBy paths (`TestWalkUnchainedStoreInvisible`, `TestMaterializeSkipsErasedAndUnreachable`).
- **R4** (layered node files + catalog): task 1, `catalog.go` + `catalog_test.go`.
- **R5** (scheduler is an optimization pass): `internal/pass/scheduler.go` with stable topo + LICM, manager/legality/VerifyEach integration pinned.
- **R6** (goldens byte-stable): task-1 sanctioned spelling sweep only; tasks 2–3 zero drift.

## Known stubs

None. No placeholder text, hardcoded empty payloads, or unwired data sources were introduced.

## Threat flags

None. No new network endpoints, auth paths, file-access patterns, or trust-boundary schema changes; the scheduler mutates an in-memory IR through the existing exported mutation choke points.

## Follow-ups (not in this plan)

- **doc/tca.md §5.1 sync:** the canonical printer now emits lowercase kind spellings (`memstore(`, `memloadsmem(`, `fragstore(`, …); the doc's notation section still shows the old spellings. Deliberately deferred as a doc-only task (flagged in the task-1 commit message).
- Pre-existing gofmt violations in `internal/ir/ssa.go` and `pkg/expr.go` (present at base 16198ab) remain out of scope.

## Self-Check: PASSED

- Files created in this plan exist: `internal/ir/poset.go`, `internal/ir/poset_test.go`, `internal/pass/scheduler.go`, `internal/pass/scheduler_test.go`, layered `internal/ir/node_*.go`, `internal/ir/catalog.go` — FOUND.
- Commits exist on the worktree branch: 32cb839 (task 1), 9b9bac6 (task 2), ffb1497 (task 3) — FOUND (`git log --oneline -3`).
- Golden drift: `git status --porcelain -- '**/testdata/**'` empty after tasks 2 and 3 — ZERO DRIFT confirmed.
- Full suite green with cold cache: `go clean -testcache && go build ./... && go vet ./... && go test ./...`.
