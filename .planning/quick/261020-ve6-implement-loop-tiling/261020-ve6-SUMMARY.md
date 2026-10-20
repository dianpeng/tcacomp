---
status: complete
task: 261020-ve6
plan: Implement doc/optana/loop_tiling.md (decide half)
tasks_executed: 3
commits:
  - ac5defc
  - 52e2bb2
  - 103e390
files_touched:
  - internal/ir/node_expr.go
  - internal/opt/tiling/record.go
  - internal/opt/tiling/overlap.go
  - internal/opt/tiling/overlap_test.go
  - internal/opt/tiling/search.go
  - internal/opt/tiling/tiling.go
  - internal/opt/tiling/tiling_test.go
  - internal/opt/tiling/verify.go
  - internal/opt/tiling/verify_test.go
  - internal/opt/all/all.go
  - internal/opt/registry.go
---

# Quick Task 261020-ve6: Loop-Tiling Decide Pass Summary

Implemented the DECIDE half of the loop-tiling pair per
doc/optana/loop_tiling.md: a new `internal/opt/tiling` sub-module whose
`tiling` pass (priority 114) computes — without mutating the IR — the
MemLoad overlap facts (findINS, SET/TIE two-level model, admission filters,
CacheDims clamp), the innermost-K flat PSched search (3^K x uniform-tile
enumeration, greedy LoC completion, CTA-product/warp-sum/SMEM-budget
gates), the FRAG/SMEM promotion judge, and the weighted
parallelism+traffic-avoided objective; plus the `tiling-verify`
record-consistency companion (priority 115) and the decision-record side
table (loop_tile_lower.md §1 contract) published through the opt analysis
cache as "tiling-record". One small ir read face (`ConstValueOf`) landed
next to `ConstTypeOf`.

## Tasks Executed

### Task 1: Decision-record contract + MemLoad overlap analysis (ac5defc)

- `internal/ir/node_expr.go`: exported `ConstValueOf(n) (float64, bool)`
  read face (twin of the package-private `constValueOf`) for reading static
  BIV extents outside package ir.
- `internal/opt/tiling/record.go`: `Record` (`Iters` keyed by BIV node
  identity, `Sites` keyed by MemLoad `BasePtr().ID()`, `Name()` =
  "tiling-record" satisfying analysis.Analysis structurally — no
  internal/analysis import, no registry entry), `SoCType`/`CacheClass`
  enums, lookup helpers `IterDecisionOf`/`SiteDecisionOf`, and `Params`
  with the doc-pinned defaults (`MaxCTA` 65535, threads 1024, 64 KB SMEM,
  `CacheDims` 2, `MaxCS` 16, `WindowK` 5, LoC presets, tile presets, unit
  weights, `NominalExtent` 4096) plus fail-fast `Validate`.
- `internal/opt/tiling/overlap.go`: `findINS` (deterministic Input-list
  worklist to InductionVariable leaves, ValueEqual membership with the
  side-effect analysis threaded in), LNT ownership (deepest INS member;
  ties by root allocation order then iter id), SET bucketing at the
  innermost referenced level, doc-ordered admission filters (set >= 2,
  rank >= CacheDims fail-closed, MaxCS cap), TIE placement (operational
  rule + outermost-tie-to-nil canonicalization + CacheDims clamp), and
  `elemBytes`.
- Tests: the doc's matmul worked example verbatim (SET {lhs,rhs}@iterK;
  lhs tie iterN covering [·][K]; rhs above-outermost covering [N][K]),
  compound-axis INS, all three filters, rank-3 clamp, `ConstValueOf`,
  `Params.Validate` (14 error classes).

### Task 2: PSched search, promotion judge, objective, tiling pass @114 (52e2bb2)

- `internal/opt/tiling/search.go`: spine window (root -> deepest leaf,
  last WindowK levels; non-spine iters sequential), flat enumeration (SoC
  odometer x tiles ascending, zero-CTA vectors skipped, <= 3^5 x 5 = 1215
  schedules at defaults), greedy LoC completion inner->outer (exact for
  the LoC sub-problem — LoC never enters traffic), promotion judge
  (CTA-at-last-position gate, covered-tile-1 gate, last-two-positions-
  sequential => FRAG else SMEM), traffic arithmetic (bytes_per_tile x
  (reuse_count - 1), extents via `ConstValueOf` with `NominalExtent`
  stand-in), SMEM-only budget gate, strict-first-best deterministic
  selection.
- `internal/opt/tiling/tiling.go`: init self-registration (`ID` "tiling",
  priority 114, `Requires` [simplify-lower], `Produces` StageConstructed,
  `Preserved` ["tiling-record"]); level gate + Validate + per-root
  overlap/search in allocation order; record published via
  `Context.Cache.Put`; NO MutContext — the pass mutates nothing (ruling
  D2). Exported `New(Params)` constructor.
- `internal/opt/all/all.go`: blank import wired in import-block order.
  `internal/opt/registry.go`: priority-scale anchors 114/115 (doc/opt.md
  itself needed no change — its "loop tiling" slot is the authority).
- Tests: matmul end-to-end (BIV keys, feasibility, carried ties, judge
  consistency recomputed over the chosen schedule), tight CTA/warp
  budgets, zero-CTA-never, 7-deep window pin (2 outermost seq/1/1),
  judge truth table, traffic arithmetic + optimality cross-check by
  independent test-local enumeration of the whole space, determinism/
  idempotence, no-mutation (node set + LastTouchedBy identical), pipeline
  adjacency after memopt-verify.

### Task 3: tiling-verify companion @115 (103e390)

- `internal/opt/tiling/verify.go`: init self-registration (`ID`
  "tiling-verify", priority 115, `Requires` [tiling], `Preserved`
  ["tiling-record"] so the record outlives the verify pass's own
  invalidation); re-derives forest + overlap facts (re-RUN derivation,
  never shared state — the memopt pattern) and checks COVERAGE (every
  live iter has an entry under its BIV; every site key resolves to a live
  MemLoad; every candidate site carries an entry), per-LNT FEASIBILITY
  (>= 1 CTA, product/sum bounds, sequential LoC 1, LoC/TileSize >= 1,
  known SoC), and SITE well-formedness (tie a live same-LNT iter or nil,
  class in {SMEM, FRAG}, recomputed SMEM footprint <= MaxCache). Claims
  consistency, not optimality.
- Tests: green paths (matmul, empty CS, empty forest), four planted
  corruption classes each failing with a tiling-verify-prefixed error,
  Requires enforcement via Manager, record survival past the verify pass,
  115 adjacency pin.

## Verification

- `go build ./...` clean; `go vet ./internal/opt/tiling/` clean; gofmt
  clean on all touched files.
- `go test ./...` — all 17 packages green (18 tests in the new suite).
- `grep -c "Priority: 11[45]"` finds exactly the two registrations
  (tiling.go: 114, verify.go: 115); `opt/all` import count = 1.
- Pinned must-haves all hold: matmul worked example, per-LNT feasibility,
  record retrievable by name and surviving passes 114+115, no IR mutation
  and no `ctaCache` field anywhere, outside-window loops sequential/1/1,
  idempotent re-runs, verify green/flagging/rejecting.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Stale closure in the tie clamp**
- **Found during:** Task 1 GREEN (caught by TestTieClampRank3 failing)
- **Issue:** the covered-set closure captured the pre-clamp `tie` local,
  so the clamp loop recomputed coverage against the original tie and
  walked the tie all the way to the innermost level.
- **Fix:** parameterized the closure (`coveredAt(t)`) so each clamp step
  recomputes against the current tie.
- **Files modified:** internal/opt/tiling/overlap.go
- **Commit:** ac5defc

### Documented Interpretations (in-code, spec-conformant readings)

1. **Tie canonicalization** (overlap.go header): the plan's matmul pin
   demands `tie(rhs) = nil` while the doc's operational rule computes
   iterM for that load. A tie AT the outermost level is recorded as nil —
   physically identical placement (directly above the outermost loop) and
   provably identical reuse arithmetic (all-LNT-iters minus covered). A
   tie landed AT the outermost level by the clamp stays a real node (its
   covered set is a strict subset of INS; nil would mean "all of INS").
2. **Tensor type read** (overlap.go `tensorRankOf`/`elemTypeOf`): reads
   the typeinfer-stamped cell (`ir.TypeOf`) first, falling back to the
   tensor InputParam's declared payload (`ir.InputParamTypeOf`). The
   plan's literal `TypeOf`-only read is pass state and is nil before
   typeinfer runs, which would fail-closed drop every site on
   construction-shaped graphs.
3. **ClassNone zero value** (record.go): the plan's "zero value =
   unclassified/unpromoted" requires a named zero distinct from ClassSMEM;
   `ClassNone` was added for it.
4. **Internal test package** (tiling_test.go): package-internal tests
   exercise the judge/search helpers directly; `opt/all` cannot be
   imported there (import cycle through the test binary), so the registry
   adjacency pin imports the memopt sub-module directly — the same
   registrations opt/all accumulates.

### Process Notes

- TDD RED/GREEN was executed per task in-workspace (tests written first,
  verified failing, then implemented to green). The binding constraint
  "all green required before each commit" forbids committing the RED
  state, so each task landed as ONE atomic `feat` commit containing tests
  and implementation together, instead of separate `test(...)`/`feat(...)`
  commits.
- Pre-existing gofmt drift in four unrelated files
  (internal/analysis/doc.go, internal/ir/effect_test.go,
  internal/ir/ssa.go, internal/opt/memopt/memopt_test.go) was left alone —
  out of scope.

## Known Stubs

None. Every component is fully implemented; the out-of-scope surface
(tile rewrite, DoSoC numbering, IntrinsicGridMap/IntrinsicThreadMap, cache
insertion surgery, LIR emission, runtime C++ objects, u32 builtins)
belongs to the future loop-tile-lower pass by the plan's own OUT OF SCOPE
section, and the record it will consume is complete and verified.

## Self-Check: PASSED

- Files: all 11 files_touched exist on disk (10 under internal/, all
  committed).
- Commits: ac5defc, 52e2bb2, 103e390 present on the worktree branch on
  top of base a6ff882.
- Full suite green at HEAD.
