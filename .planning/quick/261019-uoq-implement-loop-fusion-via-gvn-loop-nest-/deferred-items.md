# Deferred Items — quick-261019-uoq

Out-of-scope discoveries logged during execution (pre-existing at baseline
40c449e; NOT fixed per the scope-boundary rule — only issues directly
caused by this task's changes were auto-fixed).

## Pre-existing gofmt violations (found via the plan's `gofmt -l internal/` verification)

| File | State at baseline 40c449e | Notes |
|------|---------------------------|-------|
| `internal/analysis/doc.go` | gofmt-dirty | Unrelated to this task; untouched |
| `internal/ir/effect_test.go` | gofmt-dirty | Unrelated to this task; untouched |
| `internal/ir/ssa.go` | gofmt-dirty | Unrelated to this task; untouched |

All files touched by quick-261019-uoq are gofmt-clean (checked per package
and per file after every commit).
