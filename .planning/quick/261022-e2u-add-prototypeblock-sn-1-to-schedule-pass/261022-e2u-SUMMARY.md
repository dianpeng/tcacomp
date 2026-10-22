---
phase: quick-261022-e2u
plan: 261022-e2u
subsystem: compiler-ir
tags: [ast-cfg, prototype-block, sn-minus-1, placeholder-anchoring, gsd-quick]

# Dependency graph
requires:
  - phase: 261022-b2m / 261021-ta3 lineage
    provides: "the uniform-relation placement model, ASTKernel/ASTBlock/Nest() API, the cross-block ordering net, and the pipeline integration test the prototype pins extend"
provides:
  - "BlockPrototype (append-only kind 4) + PrototypeSN = -1 + NewASTPrototype() (internal/ir/astcfg.go)"
  - "ASTKernel.Prototype — the kernel-signature block, validated FIRST by AttachAST, rendered before the body"
  - "TOTAL SN→lexical-scope mapping: -1 signature / 0 kernel / r depth-r loop, with zero special conditions (max-propagation makes a -1 dep contribute nothing)"
  - "Placeholder anchoring in the schedule pass: live InputParam + KindMemNew anchor at -1 into the Prototype block; the ordering net flattens Prototype-first with a signature-edge carve-out"
affects: ["final_lower (renders the Prototype block as the kernel signature; Params/Gmem stay as structured metadata views, D-5)", "codegen"]

key-files:
  created: []
  modified:
    - internal/ir/astcfg.go
    - internal/ir/astcfg_test.go
    - internal/ir/node.go
    - internal/opt/schedule/scheduler.go
    - internal/opt/schedule/net.go
    - internal/opt/schedule/scheduler_test.go
    - internal/opt/schedule/pipeline_test.go
    - internal/opt/schedule/order_net_internal_test.go
    - doc/optana/schedule.md

key-decisions:
  - "Prototype is a dedicated ASTKernel field rendered/validated FIRST, NOT a Body item — Body keeps its [pro, lbbn, epi] cluster shape"
  - "Anchoring reuses the loop-header discipline exactly (fixed during propagation, excluded from rank collection/ownership by the anchored set) — the only new mechanism is the anchor call itself"
  - "KindMemNewSMem/KindFragNew and extent consts stay normally scheduled (SN 0) — in-kernel declarations and values, not placeholders (D-4)"
  - "Ordering-net signature-edge carve-out: a prototype placeholder referencing a scheduled value (MemNew's extent const) is a declaration edge, not an execution-order edge — mirrors the ForLoop containment carve-out; prototype-landed DEPENDENTS stay fully guarded"

requirements-completed: [QUICK-261022-e2u]

# Metrics
duration: interrupted-executor + ~20min inline completion
completed: 2026-10-22
---

# Quick 261022-e2u: PrototypeBlock (SN -1) Summary

**MemNew (GMEM) + InputParam placeholders anchor at SN -1 into a kernel-signature
Prototype block rendered first — the TOTAL SN→lexical-scope mapping with zero
special conditions in the scheduling algorithm.**

## Execution note

The dispatched executor (worktree-agent-ac2de…) was cut off mid-Task-1 by an
API rate limit (429, resets 10:54). Its Task-1 work was COMPLETE and gated
green in its worktree; the orchestrator adopted it (applied the verified diff
to master, appended the amendment to the original untracked schedule.md —
which the worktree could not see — rather than replacing it), removed the
orphaned worktree, and completed Task 2 inline.

## Task Commits

1. **T1 (17b9fed, feat)** — BlockPrototype/PrototypeSN/NewASTPrototype/ASTKernel.Prototype,
   AttachAST Prototype-first validation, SchedSN -1 semantics, the schedule.md
   amendment, constructor/refusal/round-trip tests.
2. **T2 (98677fc, feat)** — scheduler placeholder anchoring (max-wins anchor
   helper, allocation-order prototype items), no-loop `!anchored` filter,
   net Prototype-first flatten + signature-edge carve-out; pin updates
   (TestNoLoops 5+2, TestEffectfulNoHoist GMEM-at--1, TestKernelMetadata
   prototype view) + new pipeline pins (param anchoring, whole-kernel
   linkage walk).

## What Was Built

- **The total SN mapping**: -1 = signature scope (Prototype), 0 = kernel scope
  (prologue/kernel Group), r ≥ 1 = depth-r loop. Verified end-to-end on the
  matmul pipeline: 6 InputParams at SN -1 in the Prototype; params no longer
  appear as body items.
- **Zero special conditions**: anchoring rides the existing loop-header
  mechanism; propagation's max makes -1 deps contribute nothing; the anchored
  set excludes placeholders from rank collection/ownership automatically.
- **D-4 carve-out respected**: KindMemNewSMem stays at SN 0 (allocation
  exemption, pinned by the pipeline fill test); GMEM KindMemNew anchors at -1;
  extent consts stay scheduled at SN 0.

## Verification

- Full repo gate at HEAD: build/vet green, `go test ./... -count=1` all 17
  packages ok, `gofmt -l internal pkg` empty.
- Ad-hoc dump (probe deleted): Prototype(6 inputparams, sn=-1) → [prologue,
  CTA>Seq>Thr>Thr>Seq>Seq, epilogue]; fill memstoresmem at sn=3 inside its
  K-between host; body at sn=6.

## Out of Scope

None beyond the plan's fences (tile-lower K-less side-finding, LO-02 remain
open follow-ups from 261022-b2m).

## Self-Check: PASSED

---
*Plan: quick-261022-e2u*
*Completed: 2026-10-22*
