# Phase 1: IR Core, Pass Infrastructure & Tooling - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-15
**Phase:** 01-ir-core-pass-infrastructure-tooling
**Areas discussed:** IR printer format, Builder API ergonomics, Golden-test harness shape, Node representation

---

## IR printer format

| Option | Description | Selected |
|--------|-------------|----------|
| Doc-faithful notation | tca.md §5.1 statement form, SSA values %N by definition order | ✓ |
| LLVM-style SSA | `%0 = map ...` one line per node | |
| Hybrid invented form | Doc structure + LLVM-ish operand syntax | |

**User's choice:** Doc-faithful notation
**Notes:** What you read in tca.md is what dumps show; debugging doesn't require translating notations.

| Option | Description | Selected |
|--------|-------------|----------|
| Canonical re-render | Expr trees re-rendered with normalized parens/spacing; identical graphs print identically | ✓ |
| Source-preserved text | Original token text kept when built from source | |

**User's choice:** Canonical re-render
**Notes:** Byte-stability is the property golden tests and the Phase 5 parser-oracle comparison rely on.

| Option | Description | Selected |
|--------|-------------|----------|
| Stable minimal default | No locs by default; verbose modes behind --dump flags | ✓ |
| Always verbose | Loc + last-touched-by-pass on every node always | |

**User's choice:** Stable minimal default
**Notes:** Golden diffs show only real IR changes.

| Option | Description | Selected |
|--------|-------------|----------|
| Print-only Phase 1 | Round-trip = byte-identical printing across runs; golden tests compare strings | ✓ |
| Parse-back too | IR-text parser so goldens parse back to IR | |

**User's choice:** Print-only Phase 1
**Notes:** Rejected parse-back: a second grammar to maintain before the real frontend exists, risking divergence from the Phase 5 parser target.

| Option | Description | Selected |
|--------|-------------|----------|
| Stable statement IDs | `statement #N` persists across passes; diffs track evolution | ✓ |
| Flat node stream | No statement grouping | |

**User's choice:** Stable statement IDs

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror printer structure | .dot uses same numbering, statement clusters, use-def edges, deterministic order | ✓ |
| Rich styled graphs | Color per family, styled effect edges | |

**User's choice:** Mirror printer structure

## Builder API ergonomics

| Option | Description | Selected |
|--------|-------------|----------|
| Fluent pipe-chains | Chain mirrors `\|`-separated stages; parser lowering is mechanical | ✓ |
| Explicit node assembly | Constructor calls per stage | |

**User's choice:** Fluent pipe-chains

| Option | Description | Selected |
|--------|-------------|----------|
| First-class typed handles | `b.Dim`, `b.Tensor` return Go handles; no string lookups | ✓ |
| Plain-name resolution | Strings + shape structs resolved at build time | |

**User's choice:** First-class typed handles

| Option | Description | Selected |
|--------|-------------|----------|
| Expr-builder DSL | `b.Mul(l, r)` returns typed expr nodes; lambdas are Go funcs | ✓ |
| String lambdas | `b.Lambda("l, r", "l * r")` parsed from strings | |

**User's choice:** Expr-builder DSL
**Notes:** String lambdas rejected — embeds a mini-frontend with its own error surface before the real frontend.

| Option | Description | Selected |
|--------|-------------|----------|
| Eager cheap checks | Cheap invariants checked per call; full verifier post-build via PM path | ✓ |
| Verify-at-end only | All validation in post-build verifier | |

**User's choice:** Eager cheap checks
**Notes:** Fail fast at call site, but one authoritative verification path.

## Golden-test harness shape

| Option | Description | Selected |
|--------|-------------|----------|
| Side-by-side .golden files | Files under testdata/; content is reviewable data | ✓ |
| Inline CHECK comments | Expectations in Go test source | |

**User's choice:** Side-by-side .golden files

| Option | Description | Selected |
|--------|-------------|----------|
| -update flag | `go test -update` rewrites; CI without flag fails stale | ✓ |
| Env var / make target | UPDATE_GOLDEN=1 or make target | |

**User's choice:** -update flag

| Option | Description | Selected |
|--------|-------------|----------|
| Exact byte compare | Strongest; feasible because printer is canonical | ✓ |
| Pattern subset matching | FileCheck CHECK/DAG — tolerant but weak | |

**User's choice:** Exact byte compare
**Notes:** Pattern matching rejected — a golden test can pass while IR changes unnoticed.

| Option | Description | Selected |
|--------|-------------|----------|
| One generic harness | Same -update flow/diff/layout for IR, .dot, diagnostics | ✓ |
| Per-kind harnesses | Separate irtest/dottest/diatest | |

**User's choice:** One generic harness

## Node representation

| Option | Description | Selected |
|--------|-------------|----------|
| Interface families + base | Families as Go interfaces; shared embedded base with ID/Loc/use-lists | ✓ |
| Single struct + Kind tag | One Node struct, payload bag | |

**User's choice:** Interface families + base

| Option | Description | Selected |
|--------|-------------|----------|
| Intrusive Use records | LLVM-style Use objects with next/prev + back-pointers; RAUW O(uses) | ✓ |
| Per-def user slices | Slice of users per def | |

**User's choice:** Intrusive Use records

| Option | Description | Selected |
|--------|-------------|----------|
| Required mutation context | Every choke-point mutation takes pass-context; last-touched-by automatic | ✓ |
| Optional self-tagging | Passes tag nodes where they remember | |

**User's choice:** Required mutation context
**Notes:** PASS-06 diagnostics become structural, not advisory.

| Option | Description | Selected |
|--------|-------------|----------|
| Interned pointer types | Identity == equality; map keys; deterministic ordering | ✓ |
| Comparable value types | Struct tag + fields | |

**User's choice:** Interned pointer types

## Claude's Discretion

- Legality-window declaration mechanism (per-pass vs central table) — research-phase input, per ROADMAP.md research flag
- Pass-registration style, diagnostics error plumbing, .dot styling details, internal/log facade shape
- Analysis-cache internals beyond PASS-03 contract

## Deferred Ideas

None.
