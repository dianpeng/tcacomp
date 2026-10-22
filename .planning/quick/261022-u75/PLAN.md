---
status: complete
created: 2026-10-22
type: quick
---

# Complete doc/cg.md — CUDA C++ backend codegen design

## Description

User has a 31-line seed design doc (doc/cg.md) for codegen. Final-lower
(quick-261022-l1o) is done; codegen is the next stage. Task: review the
existing codebase + specs and EXPAND doc/cg.md into a complete, implementable
design for the CUDA C++ backend (internal/cg/cxx). Issues found during review
are surfaced to the user for confirmation BEFORE finalizing. The C++ runtime
library itself is OUT OF SCOPE (follow-up task) but its API surface must be
pinned by this design so the library can be implemented from it.

## Inputs

- doc/cg.md (seed), doc/tca.md §5.10-5.13, doc/optana/final_lower.md,
  doc/runtime_objects.md (44-entry runtime contract inventory),
  doc/optana/loop_tile_lower.md §4/§7bis, doc/review/final_lower.md
- Codebase: internal/ir (node kinds, print, verify), internal/opt/finallower,
  internal/opt/schedule (Loop BB Cluster / Group structure), pkg (builder),
  scalfold reference runtime

## Steps

1. [x] Read seed doc + runtime_objects.md + final_lower.md + tca.md §5.10-13
2. [x] Dispatch explorer: codegen-ready IR node inventory + pipeline entry
3. [x] Dispatch explorer: scalfold runtime API shapes
4. [x] Surface design issues to user for confirmation — 4 rulings given
       (D1 full trio artifact / D2 tca::rt / D3 TensorView / D4 kernel-top
       hoist) + mid-turn shim mandate (D6) folded in; D5 dynamic-smem
       correction and D7 slot-IV Combine documented as flagged rulings
5. [x] Write the completed doc/cg.md (seed structure preserved, 11 sections)
6. [x] Commit atomically; update STATE.md quick-task table

## Non-goals

- Implementing internal/cg/cxx (design only)
- Implementing the C++ runtime library (next task after codegen)
