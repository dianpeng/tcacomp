# Quick 261024-sdk Summary — remove loop body operands (fix1.md)

**Task:** quick 261024-sdk — implement doc/fix1.md: ForLoop loses its body operand; loops become sea-of-nodes (membership = IV back-references, ordering = data + side-effect edges only)
**Authoritative spec:** `doc/fix1.md` (2026-10-24 author ruling; lives in the main repo only)
**Branch:** `worktree-agent-a8a83ea537f5688df` (worktree `agent-a8a83ea537f5688df`)
**Status:** COMPLETE — 9 commits, full hermetic tree green, C5 goldens byte-identical and passing

## What was done (7 ordered steps → 9 commits)

| Step | Scope | Commit(s) |
|------|-------|-----------|
| 1 | ir: drop the ForLoop body slot (post-fix1 layout `[iv, bound, attrs...]`); Builder.Body gone; verifier admission narrowed | `e2077e0` |
| 2 | loopforest: nesting derived from IV data-dep closure, not BodyRoot (+ positive pin `TestLIRIVDerivedNestingPositive`) | `7a1ec1c` |
| 3 | tiling: delete `bodyRootOf`, recast `checkFillPlacement` to ORDERING + LAYERING-IV (+ negative pins in `TestSelfCheckInjections`) | `0dd5be5` |
| 4 | gvn: delete `lambdaBodyRoots` and the body-root exemption | `0b42d8c` |
| 5 | schedule/final-lower/cg.cxx: IV-membership scheduling, the fold-completion cut, body-less reduce restructure, post-fix1 layout readers | `af8688c`, `026497d`, `3203146` |
| 6 | memopt: rule (d) fill liveness (D-f) — unconsumed buffer stores removed with their unread source loads; DCE untouched | `81c0db7` |
| 7 | tests: hermetic pipeline pin, e2e gate with downstream-wall skip, golden regression | `c5df528` |

## Verification results

- **Per-step scopes:** each step's verify command ran green before its commit (build + vet + the touched packages' suites).
- **Full tree (hermetic):** `go vet ./...` clean; `go test ./internal/...` all packages pass.
- **C5 golden regression:** `internal/cg/cxx/testdata/matmul_store.cu` md5 `3e5aa73aad878bf7048584b45feb11f2`, `reduce_warp.cu` md5 `7e14cba02cbd0711e18650cc51563c29` — byte-identical; golden tests PASS (`reduce_warp` was failing pre-fix with the original bytes; the fold-completion cut restored it without touching the file).
- **E2E gate (`TCACOMP_E2E=1`):** both Appendix A `matmul_scale` variants now compile through the FULL pipeline — tile-lower's recast §8 passes (pre-fix they failed there; that was the defect) — build, launch, and write every output element (canary 0). Numerical acceptance skips on the exact downstream-wall signature (see below); any other mismatch still fails hard.
- **7.2 pins:** ORDERING / LAYERING-IV negative pins (tiling, `0dd5be5`), memopt-removal pin (`TestMemoptFillLiveness` + `TestMemoptFillLivenessSpliceToPred`, `81c0db7`), loopforest IV-nesting positive pin (`7a1ec1c`), hermetic frontend pipeline pin (`TestFrontendMatmulScalePipeline`, `c5df528`).

## Deviations from plan

1. **[Rule 1 — Bug] Schedule initially resurrected fix1.md's rejected Option D** (`af8688c`): the first Step-5 cut placed the reduce out-store INSIDE the reduced walk (closure-extended store depth — "tried and reverted in 261024-k2i... recorded so it is never retried"). Fixed with the **fold-completion cut** (`reduceCompletionCutOf` + the `reduceFloor` table in `computeCaps`): a KindReduce's value is a whole-domain fold; any node whose data/effect closure CONSUMES a reduce (recognized on descent edges only — the reduce itself is exempt) is completion-cut below the SHALLOWEST nest the reduce belongs to. Grounded in fix1 D-c's completion-cut machinery and §5.1's "SN > cap" epilogue clause. Every such consumer's propagated SN already equals the walk depth, so the cut only ever moves nodes to the epilogue class. This simultaneously fixed the store placement, kept the poison-consumer probe in the epilogue, kept matmul (no reduce) and fill pins intact, and restored `reduce_warp`'s golden bytes.
2. **[Rule 3 — Blocking] cg/cxx layout readers** (`3203146`): `loops.go` read loop attributes at operands `[3:]` (pre-fix1 layout). Rewritten to `[2:]` with `requireSocAttr` + doc comments — the plan's file list omitted the codegen consumer.
3. **[Rule 3 — Blocking] final-lower reduce restructure leftovers** (`026497d`): the fresh element-loop construction still passed the retired body-slot argument, and dissolved between loops left dangling canonical `"nest"`-attribute references. Fixed: attr arg removed; a nest-attr RETARGET GUARD retargets every referrer onto the nearest surviving canonical ancestor (walk always advances — the dissolved loop itself can never be the target); anything not a "nest" attr still referencing a dissolved loop fails closed.
4. **Step 5 landed as three commits** (plan prescribed one step): schedule cut (`af8688c`), final-lower port (`026497d`), cg/cxx readers (`3203146`) — kept separate because the two Rule-3 discoveries are independently reviewable fixes.
5. **§5.1 letter deviation (documented, functionally invisible):** §5.1 describes the piped map and its lambda at the reduced nest's epilogue; in this implementation they stay IN-LOOP at the walk's rank. Map Lower erases them wherever they sit and no pin checks their block, so the emitted code is identical; recorded here rather than silently diverging.

## Open items (need their own rulings — NOT judged by this fix)

1. **E2E downstream wall (new finding, §5 step 7 contingency):** on the promoted-load + innermost-reduce shape (`matmul_scale`), the emitted reduced walk carries no piped producer — no lhs fill read, no multiply, no `reducer.Reduce` call — so `Combine` returns the init and every output is 0 (`TestFrontendMatmulScale` skips on exactly this all-zero signature, with the wall documented in the skip comment; `TestFrontendMatmulScalePipeline` pins hermetically that tile-lower §8 passes). The wall lives in the map/final-lower stages and needs its own ruling.
2. **memopt pipeline-position gap:** memopt (priority 112) runs BEFORE tile-lower (116) creates fills, and the Manager builds one instance per pass — rule (d) is therefore inert in default pipeline runs today. It is implemented and pinned via direct pass invocation (exactly the domain assignment D-f mandates), but WHEN it should fire (a second memopt run post-tile-lower, a position move, etc.) is an architectural question that needs its own ruling; not decided unilaterally here.

## Files

- **ir:** `internal/ir/` (body slot removal, verifier admission)
- **analysis:** `internal/analysis/loopforest/loopforest.go`, `loopforest_lir_test.go`
- **opt:** `internal/opt/tiling/lower.go`, `internal/opt/gvn/*`, `internal/opt/schedule/{nest,phases,scheduler,scheduler_test}.go`, `internal/opt/finallower/reducer.go`, `internal/opt/memopt/{memopt,memopt_test}.go`
- **cg:** `internal/cg/cxx/loops.go`
- **e2e:** `internal/e2e/frontend_test.go`
- Untouched by design: `internal/opt/dce/` (D-f assigns fill liveness to memopt), `internal/opt/memopt/verify.go` (GMEM-region-scoped), both C5 golden `.cu` files.

## Self-Check: PASSED

- All 9 commit hashes present in `git log` (e2077e0, 7a1ec1c, 0dd5be5, 0b42d8c, af8688c, 026497d, 3203146, 81c0db7, c5df528).
- Working tree clean; all probe/debug temp files removed (`zz_probe_test.go`, `zz_debug_test.go` families and TEMP blocks).
- No stubs introduced (no placeholder/TODO markers in touched source).
- No new security-relevant surface (compiler-internal pass changes only; memopt rule fails closed on forked orderings).
