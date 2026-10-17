---
phase: quick
plan: 261017-vdj
subsystem: ir
tags: [ir, ssa, sea-of-nodes, materialize, two-level, golden-preserving]
dependency-graph:
  requires: [quick-261016-mpu (SSA/CFG IR refactor)]
  provides: [Level-1 sea-of-nodes Graph, Materialize scheduler, level-aware Walk/DCE/Verify, construction Env with ImpactBy effect inputs]
  affects: [internal/ir, internal/visualization, internal/pass tests, pkg]
tech-stack:
  added: []
  patterns: [two-level IR — sea-of-nodes at construction/passes, scheduled single-block CFG only after Materialize; builder-pinned statement IDs (Scheme X) preserving golden numbering]
key-files:
  created: [internal/ir/schedule.go, internal/ir/schedule_test.go, internal/ir/effect.go]
  modified: [internal/ir/node.go, internal/ir/construct.go, internal/ir/walk.go, internal/ir/dce.go, internal/ir/mutate.go, internal/ir/verify.go, internal/ir/print.go, internal/ir/cfg.go, internal/ir/doc.go, internal/visualization/dot.go, pkg/builder.go, pkg/chain.go]
decisions:
  - "Scheme X: pkg.Stmt() KEEPS reserving statement IDs (pinned via Graph.PinStatementID on effect roots) instead of the plan's drop-the-reservation — required to keep committed goldens byte-identical (oracle statement #8/#9, dot cluster_10)"
  - "Effect roots registered kind-based at node allocation (store/tile-store/return), not only via NewStoreChained — Walk/DCE/Materialize roots all derive from the same registration-order list"
  - "Erase kills payload records (as-order, cluster dims) as well as operand slots; def-use verifier skips ref records owned by unvisited (dead, pre-DCE) users"
metrics:
  duration: "single session (quick task)"
  completed: 2026-10-17
  tasks: 3
  files: 31
---

# Quick 261017-vdj: Adapt Phase-1 SSA IR to Two-Level Design Summary

Phase-1 IR is now a true two-level IR: Level 1 sea-of-nodes during construction and optimization (no CFG/statements; Block()/CFG() nil until Materialize; side-effect order explicit via effect roots + ImpactBy effect-input edges), Level 2 scheduled single-block CFG produced only by `Graph.Materialize()` after passes, before codegen. All committed goldens (`pkg/testdata/matmul_scale.ir`, `internal/visualization/testdata/matmul_scale.dot`, `internal/ir/testdata/golden_smoke.txt`) are byte-identical — zero diff under every `testdata/` directory. `pkg.Builder`'s public API (Stmt/Build/handles) is unchanged.

## Task Results

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | Effect verdict + ImpactBy effect-input payload + construction Env | `a5b9c83` | Done |
| 2 | Sea-of-nodes Graph, Materialize scheduler, level-aware walk/DCE | `1440816` | Done |
| 3 | Final verification + hard golden gate | — (this summary) | Done |

## Verification Results

- `go test ./...` — all packages pass (`internal/ir`, `internal/pass`, `internal/visualization`, `pkg`, `internal/diagnostic`), including `TestMatmulScaleGolden` and `TestDotGolden`.
- `go vet ./...` — clean.
- Hard golden gate: `git status --porcelain -- '**/testdata/**'` — empty. Zero diff under all testdata golden directories.
- `gofmt -l` — only pre-existing drift in files untouched by this plan (`internal/ir/kind.go`, `internal/ir/ssa.go`, `pkg/expr.go`); out of scope, not fixed.

## must_haves Verification

| Truth | Status |
|-------|--------|
| No BasicBlock/Statement structure under construction / between passes (pure sea held by Graph) | PASS — `NewGraph` builds no CFG; `Block()` nil-safe; `Materialized()` flag; new `TestMaterializeGuards`, `TestSingleBlock` |
| Side-effect ordering explicit via ImpactBy effect-input edges, not statement position | PASS — `internal/ir/effect.go` (task 1); ImpactBy records ride `storeNode.forEachPayloadUse`, covered by existing def-use verifier with zero new checkers |
| After Materialize: exactly one BasicBlock, statements print byte-identically to current goldens | PASS — golden tests + zero testdata diff |
| pkg.Builder public API unchanged; Build returns materialized, verified graph | PASS — Stmt/Build signatures unchanged; Build = pin returns → sea DCE → Materialize → pass-manager `VerifyEach` only (D-10) |
| `go test ./...` passes with zero diff under all testdata/ golden directories | PASS |

## New Test Coverage (Level-1 contracts)

- `schedule_test.go`: Materialize guards (nil/double), effect-root registration order, pinned vs counter IDs (pin 40; unpinned return = next monotonic ID 6), erased roots schedule nothing, Print refuses unmaterialized graphs.
- `dce_test.go`: mark/sweep sea DCE, fixpoint chain kill, `TestBuildingDCESeaWipesAbandonedChain` (payload-record killing regression).
- `walk_test.go`: pinned L1 order vs L2 parity, Stop/SkipChildren, cyclic guard.
- `cfg_test.go`: intrusive list ops via Materialize, L1+L2 `InputParamReferences`, `EraseIsO1`.

## Deviations from Plan

### Decision Deviation (documented, justified)

**1. Scheme X — Stmt() reservation KEPT via PinStatementID (plan said drop the reservation)**
- **Found during:** Task 2 design, before coding.
- **Issue:** The plan's literal shape (Stmt() no longer reserves IDs; Materialize draws all statement IDs at schedule time) shifts the committed canonical goldens: the builder oracle expects `statement #8`/`statement #9`, but without reservation the chain's 22 nodes (IDs 10–31) and return node (32) would force statement IDs ≥ 33. The orchestrator made byte-identical goldens a HARD gate.
- **Fix:** `Stmt()` keeps calling `ReserveStatementID()` (now documented as consumed by `PinStatementID`); the chain's store and each output's return carry `Base.stmtID`; `Materialize()` emits each live effect root's statement under its pinned ID, drawing from the monotonic counter only when unpinned (raw-IR users). Proven byte-identical by `TestMatmulScaleGolden`, `TestDotGolden`, and the zero-diff testdata gate.
- **Files:** `internal/ir/node.go`, `internal/ir/construct.go`, `internal/ir/schedule.go`, `pkg/builder.go`, `pkg/chain.go`.
- **Commit:** `1440816`.

### Auto-fixed Issues

**2. [Rule 1 - Bug] Erase left live-looking payload records on DCE-erased nodes**
- **Found during:** Task 2 (new `TestBuildingDCESeaWipesAbandonedChain` exposed it immediately).
- **Issue:** `MutContext.Erase` killed only operand-slot records; a node's payload records (map as-order, tile cluster dims) stayed live in other nodes' ref lists, tripping the def-use invariant for any graph where DCE erased a node with payload refs. Erase's statement unlink also assumed a linked block (nil-dereference pre-Materialize).
- **Fix:** Erase now kills payload records via the same `kill` closure over `forEachPayloadUse`; stmt unlink guarded (`s != nil && g.cfg != nil`).
- **Files:** `internal/ir/mutate.go`.
- **Commit:** `1440816`.

**3. [Rule 1 - Bug] def-use verifier false-flagged over-links from unreachable (dead, pre-DCE) users**
- **Found during:** Task 2 (`TestBuildingDCEWipesUnreferenced` precondition).
- **Issue:** At Level 1 the walk roots at params + effect roots, so legitimately dead nodes (abandoned chains, pre-DCE debris) are unvisited; their records in a live def's ref list were reported as "use that no operand slot owns".
- **Fix:** `checkDefUse` builds the walked set and skips ref records owned by unvisited users (unreachable subgraph's bookkeeping, killed when DCE erases the user). Materialized graphs are unaffected — every user is visited.
- **Files:** `internal/ir/verify.go`.
- **Commit:** `1440816`.

### Scope Notes (no code impact)

**4. Effect-root registration is kind-based at allocation, not only NewStoreChained**
- The plan named store chains as the effect-root source; implementing registration per-kind in `allocNode` (stores, tile stores, returns) is what makes the review's P1R-08 dead-chain scenario and raw-IR tests work with one uniform root list for Walk, DCE, and Materialize. Behavior for the builder path is identical.

**5. Test-code-only ID shifts; goldens untouched**
- Tests that anchor exact node IDs shifted where the old flow interleaved AddStatement IDs into construction order (e.g. print test's small chain now numbers `#23`/`#24` because the return node consumes ID 22 first; irx ids `[1,2,3,4,5,6,8]`). These are test-local expectations; no committed golden file changed.

## Known Stubs

None — every constructor in the touched surface wires real data; no placeholder/empty payload flows to rendering.

## Threat Flags

None — no new network surface, auth paths, or trust-boundary schema changes (compiler-internal IR refactor).

## Self-Check

- Files created: `internal/ir/schedule.go` FOUND, `internal/ir/schedule_test.go` FOUND, `internal/ir/effect.go` FOUND (committed in `a5b9c83`).
- Commits: `a5b9c83` FOUND, `1440816` FOUND (`git log --oneline --all`).
- Golden gate re-verified after final commit: `git status --porcelain -- '**/testdata/**'` empty.

## Self-Check: PASSED
