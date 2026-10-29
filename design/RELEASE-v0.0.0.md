# tcacomp v0.0.0 — Release Notes

- **Tag:** `v0.0.0` (annotated, local only — the user pushes when ready)
- **Tagged commit:** `e9e46ae` (branch `dev`)
- **Date:** 2026-10-29

tcacomp v0.0.0 is the first tagged MVP of the correctness-first TCA
tensor-kernel compiler: a working end-to-end flow from frontend source
through the single-tier SSA IR and its optimization passes to CUDA C++ plus a
C++20 runtime, GPU-verified on the A6000 (sm_86, CUDA 12.2). See
[README.md](../README.md) for the project overview,
[doc/syntax.md](../doc/syntax.md) for the frontend language,
[doc/tca.md](../doc/tca.md) for the IR design, and
[doc/ir.md](../doc/ir.md) for the IR catalog.

## What v0.0.0 is

The simple MVP definition from the project brief: the core value — documented
example kernels (e.g. `matmul_scale`) compile end-to-end and run correctly
on the target GPU with numerical results matching a CPU reference — is
achieved and reproducible, from frontend source through the TCA IR and its
optimization passes to CUDA C++ plus runtime, in one automated flow.

Scope of this tag is **correctness, not performance**. v0.0.0 proves the
architecture end-to-end; making the generated kernels fast is the explicit
job of the next milestone (see "Explicitly NOT in v0.0.0").

## In scope (what's in the tag)

### Frontend

Hand-rolled lexer (go/scanner structure, no `text/scanner`), recursive-descent
parser with precedence climbing, semacheck rule catalog, module loader with
`@include` and cycle detection, AST-to-IR lowering onto the pkg rungs, and the
`cmd/tcacomp` compile CLI. Pure Go standard library — no external modules, no
parser generators, no cgo. See `internal/syntax/` and
[doc/syntax-design.md](syntax-design.md).

### IR + verifier

Single-tier SSA node suite (no high/mid/low IRs; lowering introduces node
kinds into the same suite), two levels: L1 sea-of-nodes plus L2 scheduled
AST-style CFG built by the schedule pass. Verifier pass after every
transformation pass; mutation through choke points only. See
[doc/ir.md](ir.md) and [doc/tca.md](tca.md).

### Optimization pipeline

The real pass table from `internal/pipeline/pipeline.go`:

- **HIR window:** dce (100) → dce-verify (101) → simplify-lower (102) →
  simplify-inline (103) → simplify-verify (104) → typeinfer (105) →
  gvn (110, incl. loop-nest fusion) → gvn-verify (111) → memopt (112) →
  memopt-verify (113).
- **LOWER window:** tiling (114, decide: overlap analysis + combinatorial
  search + decision record) → tiling-verify (115) → tile-lower (116, tiled
  nests + SMEM/FRAG cache insertion) → schedule (900, sea-of-nodes → AST
  CFG with scheduling-number LICM) → final-lower (950, reducers/maps/runtime
  objects over the AST CFG).
- **TAIL window:** pipeline-owned const dedup → memopt (second instance,
  fill liveness) → dce (AST-item roots).

See [doc/opt.md](opt.md) and `doc/optana/`.

### CUDA C++ codegen

Deterministic emitter (`internal/cg/cxx`) producing the artifact trio: kernel
`.cu`, extern "C" launcher with baked launch totals and `ValidateLaunch`, and
a generated `CMakeLists.txt` pinned to `sm_86`. The `tca::Tensor<T, Rank>`
host/device contract with compiler-emitted shape validation. See
[doc/cg.md](cg.md).

### C++20 header-only runtime

`runtime/tca/` — GridMapping/ThreadMapping portion decompositions, uniform
iterators, bounds-predicated TensorView/SmemView/Frag views, cooperative tile
loaders (LoadTileCta/LoadTileThread), Seq/Warp reducers, host Tensor +
TCA_REQUIRE + ValidateLaunch (smem ceiling queried, never hardcoded). Built
as a CMake INTERFACE target. See [doc/rt.md](rt.md) and
[doc/runtime_objects.md](runtime_objects.md).

### Testing surface

Go unit + golden tests across `internal/` (IR, opt passes, syntax, pipeline,
e2e, pkg), the `TCACOMP_E2E`-gated GPU harness in `internal/e2e/`, and the
repo-root `testing/` suite: 14 kernels across 5 categories (dedup, fusion,
tiling, runtime, llm), 15 runner TUs, a TOML registry, and a stdlib-only
Python 3.13 orchestrator with benchmark and compute-sanitizer modes.

## Validation evidence

Measured on 2026-10-29 at commit `e9e46ae`, on the A6000 (sm_86, CUDA 12.2,
driver 580.178). Every claim below is reproducible with the commands in
"Reproducing the validation".

- `go build ./...` — green.
- `go vet ./...` — green.
- `go list -m all | wc -l` == 1 — the stdlib-only constraint holds (the
  module itself is the only entry).
- All Go unit tests pass (`go test ./...`).
- GPU suite `python3.13 testing/orchestrator.py` — **14 passed / 0 failed /
  0 skipped**, canary 0, maxdiff 2.4e-7..5.3e-5 within per-test tolerances
  (none ever broadened), total 22.8s.
- compute-sanitizer **memcheck 14/14 zero errors** and **racecheck 14/14
  zero hazards**, both with the numerical checks enforced (not sanitizer-only
  passes).

Benchmark mode (`--bench-iters 20`) also ran; see the performance caveat in
"Known issues" before reading any GB/s figure from it.

## Known issues

Shipped documented per the release decision (D-02): the two defects below and
the frontend spike walls are recorded here and in
[261028-t4u-SUMMARY.md](../.planning/quick/261028-t4u-build-a-comprehensive-compiler-testing-s/261028-t4u-SUMMARY.md);
fixes land in the post-0.0.0 backend optimization/scheduling milestone.

### D1 — GlobalReducer selection (compiler defect)

The schedule pass selects the v0-unreachable `KindRuntimeGlobalReducer` for
single-input and rank-1-output reductions. Legal frontend programs (e.g. the
plain single-input row sum `map on(x[i][j]) | reduce as (i)`) compile cleanly
but the emitted kernel asserts at launch (`"GlobalReducer: not enabled in
v0"`). Repro: any single-input reduce kernel, corpus pre-reshape versions in
commit `3086e59`. Root cause: the tiling search admits CTA digits on reduced
dims with no pruning (only WARP has the `ReducedDimWarp` gate), and the
policy comment at `internal/opt/finallower/reducer.go:848` does not match the
implemented feasible area.

### D2 — transpose permutation lowering miscompiles (compiler defect)

Both legal spellings of a transpose — written `as (j, i, k)` on
`on(x[i][j][k], ...)` and the swapped binding `on(x[j][i][k]) as (i, j, k)`
— compile but produce identical wrong output. Tiny-case dump (R=4, C=3, K=2):
expected `out[i][j] = Σ_k x[j][i][k]²`; observed
`out[j][i] = Σ_k x[0][j+i][k]²` (axis coordinates fold along anti-diagonals).

### Spike walls W1–W6 (frontend expressiveness limits)

v0 frontend walls that forced documented per-kernel fallbacks in the test
corpus; each kernel file's header records the wall(s) that shaped it. W7 ==
D1.

- **W1** — map-only rungs require a reduction before any store
  (`internal/syntax/build/build.go:389`). Pure elementwise chains are not
  expressible; fallback moved maps into a root map over a trailing reduced
  axis.
- **W2** — a piped map requires a preceding reduction (`build.go:355`).
  Fallback: `map|reduce|map` over a trailing axis.
- **W3** — the mlp piped map with a tensor binding `on(bias[j])` is rejected
  by final-lower (P-12: moved consumer references a dimvar whose loop does
  not enclose the bracket block). Fallback: bias-free GELU MLP.
- **W4** — rank-2 dynamic intermediate lets are not constructible
  (`"a multiplicative GMEM size is not constructed"`). Fallback: RMS-flavored
  layernorm; softmax restructured.
- **W5** — reading a rank-1 dynamic intermediate trips a typeinfer invariant
  (`"memload operand is not a tensor parameter"`), and an unused reduce-let
  trips final-lower's erase. Fallback: the shipped softmax is the exp+sum
  reduction core with a weight operand.
- **W6** — a reduce must drop exactly one axis (`relocate`, `build.go:424`);
  no identity reduce exists, so a rank-preserving final map (the softmax
  divide) is unexpressible. Folded into W5's fallback.

### Performance caveat

v0 codegen is **latency-bound**: per-thread SEQ loops dominate, so per-kernel
times are large and the benchmark-mode GB/s figures are **nominal**
(`bench_bytes / elapsed`), exceeding device bandwidth for compute-bound
kernels. The benchmark table exists for **regression tracking only** — do not
read it as bandwidth or performance analysis. Real performance work is the
next milestone.

## Explicitly NOT in v0.0.0

What this tag does **not** contain, so the release is not misread:

- Any performance-oriented backend work — that is the milestone being
  planned next.
- Warp-level parallel performance tuning.
- TMA / `cp.async.bulk` (sm_90+ hardware; the A6000 is sm_86 — out of scope
  by hardware).
- Multi-CTA grid sync / cooperative launch / split-k forms beyond the
  current single one.
- The D1/D2 fixes.
- The frontend expressiveness lifts behind W1–W6.
- Roadmap items still open in `ROADMAP.md` phases 2–5 checklists. Note: the
  checkboxes there **lag actual progress** (Phases 1–5 landed largely ahead
  of the checkbox state); this doc is a release statement, not a progress
  report.

## Reproducing the validation

From the repo root, at commit `e9e46ae` (or tag `v0.0.0`), on a machine with
the A6000 + CUDA 12.2 toolchain (nvcc, cmake, ninja, compute-sanitizer):

```sh
go build ./...
go vet ./...
go list -m all | wc -l          # expect: 1
go test ./...                   # hermetic unit + golden tests
TCACOMP_E2E=1 go test ./internal/e2e/   # GPU e2e (skipped without the env gate)
python3.13 testing/orchestrator.py              # 14 passed / 0 failed
python3.13 testing/orchestrator.py --sanitizer memcheck    # 14/14 zero errors
python3.13 testing/orchestrator.py --sanitizer racecheck   # 14/14 zero hazards
```

The GPU suite requires the A6000 (or compatible sm_86 device) and the CUDA
12.2 toolchain; the `TCACOMP_E2E` gate keeps unit-test runs hermetic when no
GPU is present. Note: `bin/tcacomp` in a fresh checkout is a stale gitignored
prebuilt — rebuild with `scripts/build.sh` if you need the CLI binary.
