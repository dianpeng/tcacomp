---
status: complete
phase: 01-ir-core-pass-infrastructure-tooling
source:
  - 01-01-SUMMARY.md
  - 01-02-SUMMARY.md
  - 01-03-SUMMARY.md
  - 01-04-SUMMARY.md
  - 01-05-SUMMARY.md
started: 2026-10-16T00:19:08Z
updated: 2026-10-16T00:31:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Cold Start Smoke Test
expected: From a clean state, `go test ./... -count=1` passes in every package, `go vet ./...` is clean, and `go list -m all | wc -l` outputs 1 (stdlib-only module).
result: pass

### 2. matmul_scale Oracle Golden
expected: `go test ./pkg/ -run TestMatmulScaleGolden -count=1 -v` passes — the programmatic matmul_scale kernel built through the pkg/ builder API prints byte-identically to the committed doc-pinned oracle `pkg/testdata/matmul_scale.ir`.
result: pass

### 3. Canonical Printer Output
expected: Building a small graph through pkg/ and printing it shows the canonical form: `statement #N:` headers, chain stages joined by ` |`, stage-local `%0` pipe inputs, bare kernel params (`scale`), `%name` stored bindings, tensor refs rendered structurally (`lhs[@m][@n]`), float consts like `0.0` (no f suffix).
result: pass

### 4. Integer Const Full-Precision Printing (post-fix behavior)
expected: After the WR-02 fix, two near-max int32 constants that differ only in low bits print distinctly — `Int32(2147483647)` and `Int32(2147483600)` no longer both render as the same rounded float text. Bool consts render as `true`/`false`.
result: pass

### 5. Verifier Surplus-Operand Flag (post-fix behavior)
expected: After the WR-01 fix, a node whose constructor receives more operands than its kind's arity is caught by `ir.Verify` with a violation naming the node, the invariant (`invUseDef` family), and the last-touched-by pass — instead of silently accepting the surplus.
result: pass

### 6. DOT Emission
expected: `go test ./internal/visualization/ -count=1 -v` passes and `internal/visualization/testdata/matmul_scale.dot` starts with `digraph`, contains one `subgraph cluster_<id>` per statement, node labels mirroring the text printer's canonical rendering, and renders with `dot -Tsvg` without errors.
result: pass

### 7. Instrumentation Through Manager
expected: Running a pipeline with `pkg.DefaultInstrumentations(w)` wired: with DumpAfter set, output contains the real IR text (`statement #N:` + stage content), NOT the placeholder lines (`// [ir dump:` / `// [dot dump:`); with DumpGraphviz on, output is a real `digraph`; with PassTiming on, a sorted `pass nop: <duration>` line appears; on a broken graph, the pipeline-end verify error names the node.
result: pass

### 8. Determinism (VER-08)
expected: `go test ./pkg/ -run 'TestDeterminism' -count=10` passes — two fresh in-process builds of the oracle print byte-identical text AND byte-identical DOT.
result: pass

## Summary

total: 8
passed: 8
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
