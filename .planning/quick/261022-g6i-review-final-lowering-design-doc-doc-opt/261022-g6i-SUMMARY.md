---
phase: quick-261022-g6i
plan: 01
subsystem: docs/design-review
tags: [design-review, final-lower, reducer-lowering, runtime-objects, mlir]
requires:
  - doc/optana/final_lower.md (review target, untracked working copy)
  - doc/runtime_objects.md inventory + §11 spec'd-future table
  - internal/ir/astcfg.go scheduled-CFG state (SchedSN/SchedBlock/Nest)
provides:
  - doc/review/final_lower.md — structured compatibility review of the Final Lowering design doc
affects: []
tech-stack:
  added: []
  patterns:
    - element-verdict tags (CONFIRMED/GAP/CONTRADICTION/OUT-OF-SCOPE-OK) + severity tags per doc/review/loop_tiling.md conventions
key-files:
  created:
    - doc/review/final_lower.md (772 lines)
  modified: []
decisions:
  - "E3 adjudicated CONTRADICTION: the doc's currentLoop resolves to the tile-INSIDE loop, so insertion points 2.2/2.4 put Init/Combine inside the tile-between loop on every tiled reduced dim — per-tile partials, contradicting tca.md §5.10 L723-728 (FL-1, the only BLOCKING finding)"
  - "E9 adjudicated COMPLEMENTARY: RuntimeGridMapping/RuntimeThreadMapping construction nodes complete the already-emitted query intrinsics (runtime_objects.md anticipated construction from KernelInfo + portions); the gaps are the portion data source (tiling-record cache entry) and the query-to-object lexical-enclosure binding"
  - "E2 (Reduce-inline exemption) CONFIRMED coherent: D9's acc<->piped/now<->init binding is wiring, not combiner semantics; ripple surface enumerated (simplify-verify invariant split is the required companion)"
  - "Cross-doc Reducer-interface divergence named: tca.md Init() argument-free vs final_lower.md/runtime_objects.md Init(init, lambda) — recommendation: the latter wins, tca.md gets a one-line amendment"
  - "Overall verdict: compatible-with-named-fixes — 28 findings (1 BLOCKING, 5 CONTRADICTION, 13 GAP, 7 COMPAT, 2 MINOR), 12 numbered required fixes; no architectural change required"
metrics:
  duration: ~17 min
  completed: 2026-10-22
---

# Quick Task 261022-g6i: Review final_lower.md design doc — Summary

**One-liner:** Evidence-cited compatibility review of the Final Lowering design doc
(reducer lowering, Map elimination, runtime-object construction) against the live IR,
pass pipeline, and runtime-object specs — verdict compatible-with-named-fixes, with
the reducer placement algorithm's tile-between/tile-inside ambiguity as the single
BLOCKING defect.

## What was done

- Task 1 (commit f67c2aa): doc/review/final_lower.md sections 1–4 — verdict summary +
  coverage table, method/evidence base, per-element audit E1–E10 (one verdict tag
  each, dual citations), 11-item mechanical-translation completeness walk.
- Task 2 (commit af25f84): sections 5–9 — two-direction runtime-object boundary
  cross-check vs doc/runtime_objects.md inventory, per-construct MLIR mapping table
  (overall MEDIUM), consolidated FL-1..FL-28 findings table, overall verdict +
  12-item required-fix list, self-check.

## Key findings

1. **FL-1 (BLOCKING, E3):** the doc's reducer-lowering algorithm identifies
   "currentLoop" via the Reducer's SN / first operand's SchedBlock — which resolves
   to the tile-INSIDE loop of the reduced dimension — and then places Init/Combine
   around THAT loop (points 2.2/2.4), inside the tile-between loop. tca.md §5.10
   L723-728 requires Init/Combine to wrap the tile-BETWEEN loop so one reduction
   spans all tiles. On the default tiled path (K=512/T=64 in the repo's own worked
   example) this silently produces per-tile partials.
2. **FL-3 (GAP, E4):** no reduce<@r>-equivalent annotation exists in the IR today;
   the doc never states how the reduced dimension is identified (the op0 → producing
   Map → AsOf → last-as-operand derivation is the only mechanism, and it must run
   before Map elimination).
3. **FL-8/FL-11 (CONTRADICTION, E5):** RuntimeCallMethod's method name is drawn as a
   string OPERAND (operands are nodes; both precedents are string payloads), and the
   doc sketches both an explicit lambda-call node AND reducer-internal lambda
   application — runtime_objects.md §11 records internal-only.
4. **FL-19 (COMPAT, E9):** the Runtime*Mapping construction nodes are complementary
   to the already-emitted KindIntrinsicGridMap/ThreadMap query nodes — the system
   was designed expecting them (inventory entries 7/8/41); gaps are the portion
   source (the preserved "tiling-record" cache entry) and the query→object binding.
5. **Mechanical translation (§4):** five true information gaps a backend would have
   to invent (query-object binding, accumulator tiles/validity tags, fused-consumer
   re-iteration after Combine, lambda-as-callable, Reducer<T> T-binding), two
   runtime-library contract gaps (launch configuration, dispatch mechanism).

## Deviations from Plan

None — plan executed exactly as written. Two environment notes, handled without
deviation:

- doc/optana/final_lower.md, doc/cg.md, and doc/opt.md are untracked in the main
  checkout and therefore absent from this worktree; they were read read-only from
  the main checkout (line numbers cited from those working copies). The
  untouched-target gate was satisfied worktree-side (the file cannot be modified
  where it does not exist; only Read operations were ever issued against the main
  checkout's copy).
- The plan's verify gates invoke `git -C <main-repo> diff --quiet`, which the
  worktree sandbox refuses; the equivalent worktree-local checks were run instead
  (all counts passed; target absent/unmodified).

## Verification

- Task 1 gate: 16 headers (≥8), 22 verdict-tag lines (≥5), 51 internal/ file-citation
  lines (≥8), tca.md cited, target untouched — PASS.
- Task 2 gate: 24 headers (≥14), MLIR cited, runtime_objects.md cited, 91 FL-n lines
  (≥8), verdict keyword present, 772 lines (≥200), target untouched — PASS.
- Post-commit: no file deletions in either commit; no stray untracked files.
- Only file created: doc/review/final_lower.md. doc/optana/final_lower.md NOT
  modified. No files under internal/, pkg/, runtime/ touched.

## Self-Check: PASSED

- doc/review/final_lower.md exists (772 lines) — FOUND.
- Commit f67c2aa (sections 1-4) — FOUND. Commit af25f84 (sections 5-9) — FOUND.
- All five user evaluation points covered by the §1 coverage table — verified.
