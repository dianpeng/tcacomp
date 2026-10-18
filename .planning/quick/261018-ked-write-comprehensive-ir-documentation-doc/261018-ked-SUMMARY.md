---
phase: quick
plan: 261018-ked
subsystem: docs
tags: [docs, ir, reference, examples]
dependency-graph:
  requires: [quick-261017-vdj (two-level IR), quick-261018-bl0 (tinycudaopto re-org: layered files, one-end-node, two-root DCE, pass.Scheduler)]
  provides: [doc/ir.md — implementation-faithful IR reference with worked examples]
  affects: [doc/]
tech-stack:
  added: []
  patterns: [docs-only quick task; verbatim-golden anchoring; printer-rule-derived snippets with explicit `# illustrative` markers]
key-files:
  created: [doc/ir.md]
  modified: []
decisions:
  - "Single deliverable doc/ir.md; doc/tca.md untouched (its §5.1 `store(` notation drift is flagged in doc/ir.md §1.2, sync remains the deferred follow-up)"
  - "Every IR snippet is either verbatim from a committed golden or derived from print.go rules; non-derivable lines marked `# illustrative` (§9.4 cache sketch)"
metrics:
  duration: "single session (interrupted: executor hit API quota mid-task-2; orchestrator completed the accuracy pass, gates, and commit)"
  completed: 2026-10-18
  tasks: 2
  files: 1
---

# Quick 261018-ked: Comprehensive IR Documentation Summary

`doc/ir.md` (749 lines) is the implementation-faithful IR reference, grounded
in the post-261018-bl0 codebase and cross-referenced against doc/tca.md's
decision IDs (IR-01..IR-07, D-02, D-04, D-05, D-15..D-17, P1R-08, R1..R6,
X-3/X-4) so the two docs interlock instead of drifting.

## Contents

- **§1 Overview** — what the IR is; §1.2 catalogs the known drift vs
  doc/tca.md (the §5.1 `store(` → `memstore(` spelling, plus lambda-param
  renderings) with a do-not-edit-here note.
- **§2 File organization** — what every file in internal/ir owns
  (node_*.go layering, catalog.go, ssa.go, effect.go, poset.go, walk.go,
  verify.go, dce.go, schedule.go, cfg.go, print.go, mutate.go, types.go)
  and why.
- **§3 Two-level design** — Level-1 sea-of-nodes (Input data deps + ImpactBy
  effect deps = the poset), Level-2 single-block CFG (default Materialize vs
  pass.Scheduler).
- **§4 Node catalog** — all 25 kinds in enum order with tier, family, arity,
  operands/payload, semantics; HIR 16 / LIR 9; arity + shape-check notes.
- **§5 SSA handle layer** — SSAVar/SSADef/SSAUse over one pointee; payload
  uses are uses; mutation choke points.
- **§6 Effect model** — verdicts (X-3), per-tensor chains + restrict-alias
  rule, the ONE end node (R2) carrying return values + final chain states.
- **§7 DCE** — two-root-set liveness (end + RoleTensor params, R3/P1R-08);
  why stores/in-program memory are never roots.
- **§8 Poset + scheduling** — DataDeps/EffectDeps/PosetDeps/SeaRoots read
  face; default Materialize (D-7) vs pass.Scheduler topo+LICM (R5); printer
  vs Walk (D-5).
- **§9 Worked examples** — the deliverable's centerpiece:
  - 9.1 matmul_scale oracle: builder source, the 8-line golden **verbatim**,
    node-by-node stage walkthrough (ID pinning #8/#9, `%0` pipe bindings,
    end-node wiring through the store's returned value).
  - 9.2 trivial scale kernel: derived IR with full ID-allocation derivation.
  - 9.3 two-store effect chains: printed IR plus the ImpactBy edge lists the
    printer hides (isolated per-tensor chains; second-ImpactBy-first; end
    node final chain heads; last-store-wins operand update).
  - 9.4 LIR cache-placement sketch: Mem*SMem/Frag* operand shapes, marked
    `# illustrative` where no golden pins the rendering.

## Verification

- `go build ./... && go vet ./... && go test ./...` green; zero testdata
  drift (`git status --porcelain -- '**/testdata/**'` empty).
- Accuracy pass: HIR/LIR/All kind counts run against catalog.go (16/9/25 ✓);
  §9.4 LIR operand orders verified against node_lir_memory.go ✓; §9.1 golden
  byte-identical to pkg/testdata/matmul_scale.ir ✓; §4 table diffed against
  catalog.go rows ✓.
- Docs-only: no code, test, testdata, or CLAUDE.md changes.

## Deviations / interruptions

- The executor hit an API quota error after committing task 1 and drafting
  task 2's section 9 (uncommitted). The orchestrator verified the drafted
  §9 against the code (catalog counts, LIR operand orders, golden bytes),
  finished the accuracy pass and gates, and committed task 2 (b44eb18).
  No plan-content changes were needed — the draft was accurate.

## Commits

- `8a7f137` docs(quick-261018-ked): doc/ir.md sections 1-8
- `b44eb18` docs(quick-261018-ked): doc/ir.md section 9 — worked examples
