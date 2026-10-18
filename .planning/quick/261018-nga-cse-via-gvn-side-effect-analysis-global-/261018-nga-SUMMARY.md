---
phase: quick-261018-nga
plan: 01
subsystem: ir-optimization
tags: [ir, optimization, gvn, cse, analysis, side-effects]
requires:
  - "quick-261018-l36 (pass registry / mark-sweep DCE substrate)"
provides:
  - "ir.ValueHash / ir.ValueEqual identity-keyed value-numbering contract on every node (via *Base promotion)"
  - "ir.SideEffectAnalysis query face + analysis.Analyze accumulated side-effect analysis (order-sensitive closure)"
  - "opt/gvn Pass (global CSE via one hash table, RAUW+Erase dedup) at priority 110"
  - "opt/gvn VerifyPass (dangling / analysis-closure / unique-representative invariants) at priority 111"
affects:
  - "internal/ir/effect.go verdicts (amendment A: every FamilyMemory node + Return now EFFECTFUL)"
  - "internal/opt/all default pipeline (gvn, gvn-verify between dce-verify and schedule)"
  - "internal/opt/schedule LICM belt-and-braces EffectOf consults (behavior unchanged — whitelisted kinds stay free)"
tech-stack:
  added: []          # stdlib-only constraint intact (math only); no go.mod changes
  patterns:
    - "identity-keyed value numbering (operands by SSA node identity, recursion only through lambda payloads)"
    - "fail-closed analysis query face defined in ir, implemented in analysis (no import cycle)"
    - "independent-mirror verifier (gvn-verify re-derives the fixpoint tables locally to catch drift)"
key-files:
  created:
    - internal/ir/value.go
    - internal/ir/value_test.go
    - internal/analysis/sideeffect.go
    - internal/analysis/sideeffect_test.go
    - internal/opt/gvn/gvn.go
    - internal/opt/gvn/verify.go
    - internal/opt/gvn/gvn_test.go
  modified:
    - internal/ir/node.go
    - internal/ir/effect.go
    - internal/ir/effect_test.go
    - internal/opt/all/all.go
    - internal/opt/registry.go
decisions:
  - "A: single-slot memory model — every FamilyMemory node node-local EFFECTFUL (nil ImpactBy included); DCE unaffected (root-reachability based, pinned by existing suite)"
  - "C+D: taint propagates only through order-sensitive kinds; operands keyed by SSA node identity — pure math over iter values is GVN-eligible while i1*2 vs i2*2 never merges"
  - "F: analysis lives in the existing internal/analysis package; ir.SideEffectAnalysis defined in ir to break the cycle"
  - "Body roots are one-of-a-kind for GVN (Lambda.Body is the one non-SSAUse edge); interiors dedup through record-holding users"
metrics:
  duration: "15m"
  completed: "2026-10-18T09:26:21Z"
  tasks: 3
  commits: 6
  files-touched: 12
  tests-added: 33
---

# Phase quick-261018-nga Plan 01: CSE via GVN + side-effect analysis Summary

GVN-based global CSE on the sea-of-nodes IR: an identity-keyed `ValueHash`/`ValueEqual` contract on every IR node (operands keyed by SSA node identity, recursion only through Map/Reduce lambda payloads), an accumulated side-effect analysis whose taint propagates only through order-sensitive kinds (so `i*2` over one iter dedups while lookalikes over distinct defs never merge), and a `gvn`/`gvn-verify` pass pair at priorities 110/111 — with `EffectOf` amended so every memory op (loads included, nil ImpactBy included) is node-local effectful under the single-slot memory model.

## What Was Built

### Task 1 — IR value contract + amended EffectOf verdicts (commits 1eacebd, 3124018)

- `internal/ir/value.go` (444 lines): `ValueHashOneOfAKind` sentinel (`^uint64(0)`), the `SideEffectAnalysis` query interface (defined in ir so `internal/analysis` can implement it without an import cycle — amendment F), and the promoted `*Base` methods:
  - `ValueHash(se)`: one-of-a-kind gate (nil analysis / effectful / invalid kind → sentinel), InputParam atoms by kind+ID, composites via a simple FNV-style combiner over kind, per-kind content fields (const type + `math.Float64bits`, opcodes, cast type, call name, LIR StorageClass), operand slot count + per-slot def node IDs, payload target IDs, and per-lambda param IDs + recursive body hash with a cycle guard contributing 0 on re-entry. Sentinel-collision perturbation (`h ^= 0x9e3779b97f4a7c15`). No structural recursion into operands, ever (amendment D).
  - `ValueEqual(other, se)`: false on nil/unanalyzable/either-side-effectful ([3].3 literal — including self-comparison)/invalid kinds; atom identity; composites by same kind + content + pairwise def identity + payload target identity + lambda machinery (params by identity, bodies recursive, in-progress pair set fails safe on cycles). The CONTRACT (equal ⇒ same hash) is documented on both methods — gvn-verify's bucket math relies on it.
- `internal/ir/node.go`: Value interface widened with both methods; doc comment records the deliberate asymmetry (Node stays at two methods per directive 4).
- `internal/ir/effect.go` (amendment A): `EffectOf` returns Full for `k.Family() == FamilyMemory || k == KindReturn` — all nine memory kinds, loads and allocations included, nil ImpactBy included. File doc rewritten to the single-slot memory model; explicitly notes DCE eligibility is root-reachability based (R3), not EffectOf based.
- **Audit (T-nga-04):** scheduler LICM call sites (scheduler.go:463,511) sit strictly behind `licmHoistableKinds` (scalar-expr kinds only — all still Free), so hoisting behavior is unchanged and the suite proves it; DCE (opt/dce + ir's build-time sweep) never consults EffectOf — the existing `TestDCEHingeDeadSMemPair` pin proves an unreferenced SMEM store is still swept despite now reading "effectful"; Materialize's `isEffectKind` and the legality windows do not consult EffectOf (grep-verified). Full tree green after the change.

### Task 2 — accumulated side-effect analysis (commits b9ff595, 8a2faf9)

- `internal/analysis/sideeffect.go` (183 lines): `SideEffects` implementing BOTH `analysis.Analysis` (`Name() == "side-effect"`) and `ir.SideEffectAnalysis`; `Analyze(g)` — read-only, memoized three-state DFS over `DataDeps ∪ LambdaBodies`, O(N+E), deterministic.
  - Base verdicts: `len(ImpactBy(n)) > 0` ([2.1] future-bearer hook), every FamilyMemory kind ([2.2] as amended by A), Iter/IterCluster/ForLoop/Return ([2.3]).
  - Closure ([2.4] as amended by C): `effectful(n) ||= orderSensitive(n) && ∃ effectful child`, orderSensitive = FamilyMemory ∪ FamilyLoop ∪ {Return} ∪ the FamilyCollective hook. Pure kinds never propagate. In-progress re-entry reads effectful (cycle-safe, conservative).
  - Fail-closed queries: nil/erased/unknown nodes → true; `Analyze(nil)` → all-conservative (T-nga-03).

### Task 3 — GVN pass + verifier + pipeline wiring (commits 0bc8d9a, d8bce2e)

- `internal/opt/gvn/gvn.go` (173 lines): registered at priority 110. Level gates mirror dce (nil graph / materialized refusal, untouched on refusal). One global `map[uint64][]ir.Node` table, allocation-order sweep (lowest-ID duplicate survives; def-before-use construction means a node's operands have already resolved to their final representatives before it is hashed — which is what makes the RAUW cascade (`C2→C1` then `B2→B1` in one run) compose like classic value numbering). ITEM-7 hard assert before every RAUW; post-RAUW zero-uses check; `Erase` along the way ([4.6]). Lambda body roots skipped (the one non-SSAUse edge — `RAUW` cannot rewire `lam.Body`; documented upgrade path), interiors still dedup. No analysis published to the cache (documented rationale: a pre-run analysis goes stale on the first Erase; `Invalidate(nil)` wipes post-run puts anyway).
- `internal/opt/gvn/verify.go` (194 lines): registered at 111 with `Requires: ["gvn"]`. Mutation-free; recomputes the analysis and body-root set fresh (independence). Three checks: DANGLING (every live use resolves to non-erased def + non-erased non-nil user), ANALYSIS-CLOSURE (the accumulated verdicts are a fixpoint of base ∪ order-sensitive propagation over the actual graph, checked against an INDEPENDENT local mirror of the tables so drift between analysis and verifier is caught), UNIQUE-REPRESENTATIVE (full pairwise ValueEqual over participating nodes — O(N²) documented fine at v0 kernel scale; catches hash/equality contract drift that bucket-limited checking would miss).
- `internal/opt/all/all.go`: `_ "tcacomp/internal/opt/gvn"` blank import; `internal/opt/registry.go`: priority-scale comment extended with the 101/110/111 anchors.
- 33 new tests across the three files; full `go build ./... && go vet ./... && go test ./...` green.

## Spec-Fidelity Notes

Every row of the plan's spec matrix is delivered at full fidelity, including the amendment-only rows: AMEND-B (two identical loads never merge — pinned), AMEND-D (`i*2` over one iter dedups / `i1*2` vs `i2*2` never — pinned), AMEND-E (Reduce participates; distinct combiner-param atoms keep reduces apart — pinned). The pipeline wiring truth holds: BuildPipeline under opt/all runs gvn (110) then gvn-verify (111) between dce-verify (101) and schedule (900).

## Deviations from Plan

### Interpretation clarifications (no spec change)

**1. EffectOf keeps loop kinds Free — the plan's own action text over its behavior prose**
- **Found during:** Task 1 GREEN
- **Issue:** Task 1's behavior list mentions iter under the verdict tests, but the action text is explicit: "Full for every FamilyMemory kind ... plus KindReturn; everything else Free". Loop-kind effectfulness ([2.3]) belongs to the ACCUMULATED analysis (Task 2), not the node-local verdict ([4].3 self-scoped).
- **Fix:** Implemented per the action text; the effect_test pins iter/forloop as Free under EffectOf with a pointer to the accumulated analysis, and Task 2's tests pin them effectful under `Analyze`.
- **Files:** internal/ir/effect.go, internal/ir/effect_test.go

**2. Idempotence probe uses a structural fingerprint instead of print output**
- **Found during:** Task 3 test design
- **Issue:** `ir.Print` on a Level-1 graph is the constant marker "ir: graph not materialized" (gvn refuses materialized graphs), so print-output comparison is vacuous.
- **Fix:** The idempotence test compares node count plus a structural fingerprint (per-node kind, operand def IDs, use-list user IDs) — a strictly stronger stability probe.
- **Files:** internal/opt/gvn/gvn_test.go

**3. Full-pipeline interop runs the L1 prefix**
- **Found during:** Task 3 test design
- **Issue:** The full BuildPipeline cannot run end-to-end today: dce/gvn refuse materialized graphs while schedule demands one — Materialize slots between gvn-verify and schedule in the caller's flow (repo reality predating this plan).
- **Fix:** The interop test runs [dce, dce-verify, gvn, gvn-verify] sliced from BuildPipeline in registry order, asserting dead code swept + duplicates merged + end state verifies. The 100<101<110<111<900 ordering itself is pinned separately.
- **Files:** internal/opt/gvn/gvn_test.go

### Auto-fixed issues

**4. [Rule 1 - Bug] Test-fixture bugs caught by the RED→GREEN loop**
- **Found during:** Tasks 1 and 3 GREEN
- **Issue:** Four fixture bugs in the new tests: `NewBinOp` arity (raw `NewNode` needed for operandless binops); Go untyped constant `-0.0` collapses to `+0.0` (used `math.Copysign`); wrong expected operand const in the leaf-pooling assertion; the side-effect exclusion fixture's `one` const duplicated `val` const (legitimate pooling, not an exclusion violation).
- **Fix:** All four corrected against the implementation contract; no production code was changed to accommodate them.
- **Files:** internal/ir/value_test.go, internal/opt/gvn/gvn_test.go
- **Commit:** 3124018, d8bce2e

None of the deviations changed the plan's scope, contracts, or file layout.

## Threat Flags

None. The threat register's mitigations landed as specified: T-nga-03 (fail-closed gates: nil analysis ⇒ sentinel, unknown/erased nodes ⇒ effectful, during-pass assert), T-nga-04 (verdict-change audit recorded above). No security-relevant surface beyond the plan's model (pure in-process compiler transformation; stdlib-only intact — `math` is the only new import; zero go.mod entries).

## Test Evidence

- `go build ./...` clean; `go vet ./...` clean; `go test ./...` all green (ir, analysis, opt, opt/all, opt/dce, opt/gvn, opt/schedule, diagnostic, visualization, pkg).
- Named-case pins: cascade (one run merges C2→C1 then B2→B1), iter-arith (shared vs distinct iter defs), identical loads/stores/iters/allocs never merge with chains untouched, Reduce participation through its piped operand, lambda-body-root skip with interior dedup, gvn-verify RED on a duplicate-holding graph and on a [gvn-verify]-only pipeline.

## Self-Check: PASSED

All 13 created/modified files exist on disk; all 6 task commits verified in `git log` (1eacebd, 3124018, b9ff595, 8a2faf9, 0bc8d9a, d8bce2e). Per the quick-task constraints, SUMMARY.md and other docs artifacts are left uncommitted for the orchestrator's Step 8 docs commit; ROADMAP.md untouched.
