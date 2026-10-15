---
phase: 1
slug: ir-core-pass-infrastructure-tooling
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-10-15
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: `01-RESEARCH.md` §Validation Architecture (all test names illustrative — planner owns final names).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Go stdlib `testing` (Go 1.26.2) — zero external modules |
| **Config file** | none — the only "config" is the golden `-update` flag flow (D-12) |
| **Quick run command** | `go test ./internal/ir/ ./internal/pass/ ./pkg/ -count=1` |
| **Full suite command** | `go test ./... -count=1` |
| **Estimated runtime** | ~5 seconds (hermetic unit + golden tests; no GPU/exec calls in Phase 1) |

---

## Sampling Rate

- **After every task commit:** Run `go test ./internal/ir/ ./internal/pass/ ./pkg/ -count=1`
- **After every plan wave:** Run `go test ./... -count=1`
- **Before `/gsd-verify-work`:** Full suite green + `go vet ./...` + `go list -m all | wc -l` == 1 (stdlib-only enforcement) + golden CI mode (no `-update`)
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-* | 01 | 1 | IR-01, IR-02 | T-1-01 / — | N/A (no external surface) | unit | `go test ./internal/ir/ -run TestUseChain -count=1` | ❌ W0 | ⬜ pending |
| 1-01-* | 01 | 1 | IR-01, IR-04, PASS-06 | T-1-02 / — | Malformed graph fails loudly, never partial state | unit (corruption) | `go test ./internal/ir/ -run TestVerifierCatchesCorruption -count=1` | ❌ W0 | ⬜ pending |
| 1-02-* | 02 | 2 | IR-03, IR-07 | T-1-03 / — | Unknown Kind rejected, never silently skipped | unit | `go test ./internal/pass/ -run TestLegalityRejectsUnknownKind -count=1` | ❌ W0 | ⬜ pending |
| 1-02-* | 02 | 2 | PASS-01, PASS-02, TOOL-06 | — / — | N/A | unit | `go test ./internal/pass/ -run 'TestPreconditionRejectsOrder|TestInstrumentation' -count=1` | ❌ W0 | ⬜ pending |
| 1-02-* | 02 | 2 | PASS-03 | — / — | N/A | unit | `go test ./internal/pass/ -run TestAnalysisInvalidation -count=1` | ❌ W0 | ⬜ pending |
| 1-03-* | 03 | 2 | IR-05, TOOL-01, TOOL-03, TOOL-04 | T-1-04 / — | Printer emits no environment-derived strings (paths/hostnames) | unit + golden | `go test ./internal/ir/ -run TestCanonicalPrint -count=1 && go test ./internal/diagnostic/ ./internal/visualization/ -count=1` | ❌ W0 | ⬜ pending |
| 1-04-* | 04 | 3 | FE-01, TOOL-05 | — / — | N/A | golden | `go test ./pkg/ -run TestMatmulScaleGolden -count=1` | ❌ W0 | ⬜ pending |
| 1-04-* | 04 | 3 | VER-08 | — / — | N/A | unit + golden | `go test ./pkg/ -run TestDeterminism -count=10` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `go.mod` (module `tcacomp`, `go 1.26.2`, zero requires) — repo has no Go files at all yet
- [ ] Package skeletons: `internal/{ir,pass,analysis,flags,diagnostic,log,visualization,golden}`, `pkg/`
- [ ] `internal/golden` harness — needed by every other plan's tests
- [ ] First golden file `pkg/testdata/matmul_scale.ir` lifted from `doc/tca.md` §5.1 (diff lifted text against §9.3 for consistency first)
- [ ] `internal/flags` defaults for `VerifyEach/DumpAfter/DumpGraphviz/PassTiming` — needed by pass-manager tests

*Greenfield — no existing test infrastructure to build on.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | — | — | All phase behaviors have automated verification |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
