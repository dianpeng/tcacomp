---
phase: quick-261024-bz9
plan: 01
subsystem: finallower/tiling/runtime/e2e
tags: [finallower, tiling, runtime, reducer, e2e, goldens, bracket-law]
requires: [261023-l12]
provides: [ra-bracket-law, hi03-tile-gates, warp-reducer-2026-10-24-protocol, t1-sweep-endpoint, rank3-multi-between-e2e, warp-single-mirror-gate]
tech-stack:
  added: []
  patterns: [brace-depth R-A scanner over emitted text, per-reduce rebuild memoization (one clone per shared spine), triple-gated tile<=coupled assert-and-trap family]
key-files:
  created:
    - internal/cg/cxx/bracket.go
    - internal/e2e/reduce3_test.go
  modified:
    - internal/opt/finallower/reducer.go
    - internal/opt/finallower/pass.go
    - internal/opt/finallower/finallower_test.go
    - internal/opt/tiling/record.go
    - internal/opt/tiling/verify.go
    - internal/opt/tiling/verify_test.go
    - internal/opt/tiling/lower_rewrite.go
    - internal/opt/tiling/lower_test.go
    - internal/opt/tiling/overlap_test.go
    - internal/opt/tiling/tiling_test.go
    - doc/tca.md
    - doc/rt.md
    - runtime/tca/reducer.h
    - runtime/tests/cpu_tests.cpp
    - internal/cg/cxx/emit_test.go
decisions:
  - "R-A bracket law implemented literally (Init immediately before X in X's parent = the innermost element loop; Reduce inside Y; Combine immediately after X); at len(betweenBlks)==1 elemBlk == outermostBlk so the golden shape is bit-identical"
  - "Mirror-gate cap operand is the EXTENT, not the literal constant 1 — the plan's own numerical acceptance gate is the oracle (a 1-trip cap under-covers every extent > LoC*32, exactly shape B's 512-element axis at 256 threads); the F1-saturating zeroing mechanism is otherwise exactly as ruled"
  - "Shape B is the rank-2 T==1 endpoint ({CTA,100,1}/{WARP,8,1}) — the plan's literal rank-3 variant AND its documented fallback both hit a pre-existing tile-lower wall (see Deferred); all three stated shape-B purposes still exercised"
metrics:
  duration: ~3h
  completed: 2026-10-24
---

# Quick Task 261024-bz9: Fix 3 HIGH review findings + WarpReducer protocol + e2e expansion Summary

**One-liner:** the R-A bracket law landed mechanically (IR + emitted-text proofs for 1-between / >=2-between / T==1), tile>coupled rejected at three gates, WarpReducer reworked to the activemask/slot protocol with a racecheck-decided trailing barrier, and two new full-pipeline reduce shapes run clean on the A6000.

## Gates

| Gate | Result |
| --- | --- |
| Hermetic `go build ./...` + `go vet ./...` + `go test ./...` + gofmt | all packages ok, zero failures, formatting clean |
| Runtime CPU unit layer (`tcacomp_cpu_tests`) | 2029 checks, 0 failures (was 167; + the bracket simulation + mask arithmetic) |
| `TCACOMP_E2E=1 go test ./internal/e2e/` | all 7 tests PASS (matmul + reduce goldens, reduce3 + reducet1 numerics, 2 sanitizer tests over 4 fixtures) |
| matmul e2e (regression) | exact checksum/samples, canary 0 — unchanged |
| reduce e2e (regression, new protocol) | canary 0, within 2e-3 — unchanged |
| reduce3 e2e (NEW, rank-3 multi-between) | **canary 0, max abs diff 1.22e-4** (tol 2e-3), 2048 outputs |
| reducet1 e2e (NEW, T==1 sweep endpoint) | **canary 0, max abs diff 9.16e-5** (tol 2e-3), 1024 outputs |
| compute-sanitizer | memcheck + racecheck clean on all four fixtures (after the contingency below) |
| Golden disposition (R-B) | reduce_warp.cu regenerated via `-update` → **byte-identical**; matmul goldens byte-identical; `git status` over testdata clean |

## The 1-between R-A audit (R-B) — outcome: CONFORMANT, with evidence

Mechanical evidence, both levels:

1. **IR level** — `TestReducerBracketPlacement` (extended) asserts on the
   live pipeline output: Init's containing block IS X's parental block (the
   element loop), Init's item index == X's block index − 1, Combine == X's
   index + 1 in the same block, Reduce's innermost containing block == home
   (Y). Green before and after the Task-1 code changes (at
   len(betweenBlks)==1 `elemBlk == outermostBlk`, so the fix could not and
   did not move anything in this shape).
2. **Emitted-text level** — `TestGoldenReduceWarp` now runs
   `cxx.ScanBracketLaw` (the brace-depth scanner) over the emitted text;
   green. The regeneration through the normal golden mechanism
   (`go test ./internal/cg/cxx/ -run TestGolden -update`) rewrote the file
   to **zero diff** against the committed golden.

Verdict per the R-B resolution protocol: the audit found the committed
golden's bracket placement **R-A-conformant** — the author's "the golden
is buggy" suspicion resolved NEGATIVE by measurement. The golden did not
change; no diff to explain. (Consistent with the plan's own pre-audit
reading of lines 46–64; the review's HI-01 was about `len>=2` shapes,
where the bug was real and is now fixed and e2e-proven.)

## The R-A bracket law (HI-01) + the T==1 consumer move (HI-02)

- `buildElementLoop` now captures the INNERMOST element block (`elemBlk`,
  the first built) as X's parent host: `[Init, X{home{...Reduce}},
  Combine, moved consumers]` with exact item adjacency; the outermost
  block keeps the parental insertion slot. Doc tca.md §5.10(2) states the
  law verbatim-in-substance.
- The T==1 flat bracket keeps its (already lawful) Init/X/Combine
  placement and now relocates the qualifying epilogue consumers in
  immediately after Combine, stamped into the same block — the Combine
  local and the consuming store share one C++ lexical scope. Pinned at IR
  level (`TestReducerBracketT1`: Init/X/Combine/store adjacency +
  `SchedBlock(combine) == SchedBlock(store)`) and emitted-text level (the
  scanner's same-scope check over the emitted kernel).
- Shared move logic (`moveCombineConsumers`): reduced-dim poison check →
  thread-partitioned refusal → optional between-IV rebuild → generalized
  P-12 enclosure assertion → relocate after Combine.

### The thread-partitioned-consumer refusal

A moved consumer whose operand closure references the single IV of a
WARP-SoC dim fails closed naming the dim: the `is_first_in_cta` store gate
yields single-owner stores only over CTA-replicated element walks, so a
WARP-partitioned non-reduced output dim would leave a coverage hole.
Defensive fail-closed for an unruled adjacent shape — unreachable when the
reduced dim is WARP (the one-WARP-dim policy), and the reduced dim's own
single IV is caught earlier by the reduced-dim poison check. Pinned by
`TestReducerWarpSingleConsumerRefusal`.

## The single-form mirror gate (T==1 WARP)

In a CTA-carrying LNT the WARP-single bound becomes the F1-saturating
expression over `offset = index(threadDim) + index(ctaThreadDim)·extent`,
zeroing the mirrored coupled half's trips; the IV keeps init 0 / step
LoC·32. **One deliberate deviation from the plan's literal text:** the cap
operand is the **extent**, not the constant 1 ("reuse the tileMax helper
with tile 1"). The fused single loop enumerates whole rounds at stride
LoC·32, so a 1-trip cap would under-cover every extent > LoC·32 — exactly
the T==1 e2e's 512-element last axis at a 256-thread group; the plan's own
numerical acceptance gate (shape B within 2e-3) is the oracle, and with
the extent cap it passes at 9.16e-5. The owning half's bound
(`extent − rank`) is also strictly tighter than the old raw extent (no
OOB phantom rounds). Pinned by `TestRewriteWarpSingleMirrorGate` (bound
shape, cap operand, gate/rank queries, IV stride) and by the e2e numerics.

## WarpReducer — the 2026-10-24 protocol, and the racecheck-decided contingency

Implemented exactly as ruled: no per-thread accumulator in device code
(host-only `host_acc_` guarded by `#ifndef __CUDACC__`, absent from both
nvcc passes); `Init()` resets ALL slots single-writer (each warp's first
lane rewrites its own slot); `Reduce(v)` captures `__activemask()` and
runs the guarded fn down-tree over the participating lanes inside Reduce
(`lane_id + off < __popc(mask)` discards out-of-prefix reads; the
prefix-alignment contract is pinned in a comment); lane 0 folds
`slot[warp] = fn(slot[warp], warpPartial)`; `Combine()` is barrier-first
then every thread folds all `warp_count()` slots ascending. SeqReducer and
GlobalReducer untouched; ctor face and IR/codegen contract untouched.

**T-bz9-05 outcome: the contingency was needed and applied.** The ruled
protocol measured racecheck-DIRTY on both new fixtures —
`Race reported between Write access at 0x580 … and Read access at
0xd30/0xd40/0xd50/0xd70` (reduce3; write = the next bracket's Init/Reduce
slot writes, reads = the straggler's Combine fold) and the mirrored pair
on reducet1 (`write@0x500` vs `reads@0x1060–0x10a0`, plus
`read@0x1070` vs `write@0x950/0xb80`). This is exactly the anticipated
slot-WAR between a straggler fold and the next bracket's Init — the old
protocol's Init wrote no SMEM (register acc_), so the window is new. Per
the pre-ruled contingency, ONE trailing `__syncthreads()` was added at the
end of Combine (after the fold); after it, **memcheck + racecheck are
clean on all four fixtures** and the numerics are unchanged. This is
recorded as a ruled-protocol amendment candidate for the author: the
protocol's "ONLY sync" claim is now two barriers per bracket, and
doc/rt.md §4.6 states this with the evidence.

## HI-03 — three fail-closed tile ≤ coupled gates

`Params.Validate` (gate 1, after the positivity loop), tiling-verify's
per-LNT feasibility sweep (gate 2, warpPortion resolved from Params
exactly as `coupledThreads()` — never the record's warp LoC), and the
rewrite-side backstop beside the F2 floor (gate 3) all reject
`TileSize > coupledThreads` with named errors carrying the tile and the
coupled count (512 vs 256 at defaults). `coupled == 0`
(WarpPortion == PortionWarp) skips the comparison — the F2 floor owns that
configuration. DefaultParams (max tile 256 == coupled 256) stays valid.
Dedicated tests: `TestParamsValidateCoupledTileGate`,
`TestVerifyCoupledTileGate`, `TestLowerCoupledTileBackstop`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `betweenBlocks` returned nil for the legal empty between set**
- **Found during:** Task 1 (the T==1 fixture)
- **Issue:** `append([]*ir.ASTBlock(nil), path...)` with zero elements returns nil — the T==1 flat-bracket path (`len(betweenBlks) == 0`) was UNREACHABLE its entire life: every T==1 reduce died as "home block is not nested inside X's block (corrupted schedule?)". The deepest-layer reason HI-02 was never even reachable.
- **Fix:** a non-nil empty slice on success; comment pins the trap.
- **Files:** internal/opt/finallower/reducer.go
- **Commit:** 0cc076b

**2. [Rule 1 - Bug] Per-user rebuild clones broke the loop-scope invariant on the >=2-between shape**
- **Found during:** Task 1 (the rank-3 fixture failed final-lower's verifier: a store-private rebuilt spine referenced the SEQ dim's between IV from outside its operand closure)
- **Fix:** `rebuildWithSubs` memoized per reduce (one clone per shared spine): the moved store shares the in-nest load's clone, which is inside the defining loops' closures through the load. Emitted text unchanged (both clones rendered identical inline text — the golden stayed byte-identical).
- **Files:** internal/opt/finallower/{reducer,pass}.go
- **Commit:** 0cc076b

**3. [Rule 3 - Blocking] Two existing tiling fixtures violated the new gate-1 Params surface**
- **Found during:** Task 1 (`TestFeasibilityBounds/minimal warp budget` at 256 threads / tiles {..256} vs coupled 128; `TestCoupledPortionFloor/verify rejects starved record` at WarpPortion 7 / tiles {..256} vs coupled 64)
- **Fix:** tile sets shrunk to the legal bound ({32,64,128} and {32,64}); each subtest's original intent (warp-budget cap; record-level F2) preserved.
- **Files:** internal/opt/tiling/tiling_test.go
- **Commit:** 0cc076b

**4. [Rule 1 - Bug] `typeinfer` rejects a `KindKernelMeta` created before it on the crafted-record path**
- **Found during:** Task 3 (fixture build)
- **Fix:** the e2e fixture sets the meta AFTER typeinfer and before schedule (the full pipeline's pre-105 passes handle the meta; the standalone pass's catalog is closed). Fixture-local; no production code changed.
- **Commit:** 688a4a5

### Plan-shape deviations (ruled intent over literal text)

**5. Mirror-gate cap operand = extent, not the constant 1** — see the
dedicated section above; the plan's literal formula fails the plan's own
numerical gate.

**6. Shape B is rank-2, not rank-3** — the plan's literal rank-3 T==1 shape
and its documented fallback BOTH die at tile-lower time on a pre-existing,
independent wall (the SEQ-single raw-dimvar index expression cannot serve
an epilogue consumer under the loop-scope invariant). Per the plan's own
fallback clause the shape was re-targeted to the rank-2 T==1 endpoint,
which exercises all three stated purposes; the wall is recorded in
deferred-items.md as an author-level fix. The finallower T==1 pin, the
emit pin, and the e2e all cover the re-targeted shape.

## Deferred Issues

See `deferred-items.md` (same directory): (1) the SEQ-single epilogue wall;
(2) the unruled T>1 WARP-non-reduced-dim coverage-hole analog of the
refusal; (3) a suggested audit of other `append(empty)` nil-return traps.

## Commits

| Commit | Content |
| --- | --- |
| `0cc076b` | Task 1 — R-A bracket law (elemBlk/outermostBlk), T==1 consumer move + shared gates + refusal, betweenBlocks empty-path fix, rebuild memoization, HI-03 gates 1–3, WARP-single mirror gate, tca.md §5.10, IR pins (R-A adjacency ×3 shapes, refusal probe, gate probes, mirror-gate bound pin) |
| `016cb6e` | Task 2 — WarpReducer 2026-10-24 protocol (activemask guarded down-tree inside Reduce, single-writer slot fold, Init resets all slots, barrier-first ascending Combine, host-only acc_), doc/rt.md §4.6, CPU bracket simulation (tolerance-only per R-C) + mask arithmetic — 2029 checks |
| `917c0c3` | chore — ignore runtime/tests/build |
| `688a4a5` | Task 3 — bracket.go R-A scanner + golden wiring (audit: conformant, byte-identical), reduce3/reduceT1 emit builders + pins, the two new e2e shapes + sanitizers, T-bz9-05 trailing-barrier contingency on measured racecheck evidence |

## Worktree metadata

- worktree_path: `.claude/worktrees/agent-a3a683e13157faf66`
- branch: `worktree-agent-a3a683e13157faf66`
- expected_base: `b760c0ab8dadc06f71aea51ecf1d1e02159e7e10` (plan revision bdd6644 merged in first)

## Self-Check: PASSED

All created files exist (bracket.go, reduce3_test.go, SUMMARY, deferred-items);
all four commits present on the worktree branch (0cc076b, 016cb6e, 917c0c3,
688a4a5); testdata byte-identical; final gates re-verified green.
