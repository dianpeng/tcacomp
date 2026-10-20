---
phase: quick-261020-ux8
plan: 01
subsystem: docs/optana
tags: [loop-tiling, lowering, smem, frag, pass-spec]
requires:
  - doc/optana/loop_tiling.md (decide pass — decision record producer)
  - doc/review/loop_tiling.md (findings F3/F7/F11/F12/F13/F14/F16/F18/F20 + §5.4)
provides:
  - doc/optana/loop_tile_lower.md (implementable merged loop-tile-lower pass spec)
  - doc/optana/smem_insertion.md (pointer stub to the merged spec)
affects: []
tech-stack:
  added: []
  patterns:
    - open-questions register pattern (§4) — normative D1-independent ground + undecided facets
key-files:
  created:
    - doc/optana/loop_tile_lower.md
  modified:
    - doc/optana/smem_insertion.md
decisions:
  - "§4 (Issue #1, control flow/parallel tails) written as OPEN-questions register: D1-independent ground normative, five facets Q1-Q5 with option spaces and consequences, no resolution; every other section cross-references §4"
  - "§7bis allocator vocabulary (D3) left OPEN; per-LNT scoped maps and 1-D normalized launch contract written as RULED"
  - "All author-ratified items written normative: no ctaCache field, FRAG must-unroll + byte budget, u32 min/max/ceildiv hard prerequisite, string intrinsic payload + six companions, runtime-owns-all-barriers, halve-then-drop fallback, cross-LNT dedupe/union, LIR KindForLoop emission, zero new verifier invariant classes"
metrics:
  duration: ~20 min
  completed: 2026-10-20
---

# Quick Task 261020-ux8: Draft doc/optana/loop_tile_lower.md (merged pass spec) Summary

Merged loop-tile-lower pass spec (tile rewrite + DoSoC numbering + map intrinsics + SMEM/FRAG cache insertion in one pass) with the two contested items — Issue #1 control-flow/predication (§4) and D3 allocator vocabulary (§7bis) — preserved as explicit OPEN registers; smem_insertion.md converted to a pointer stub.

## Tasks Completed

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 | Write doc/optana/loop_tile_lower.md — the merged pass spec | 96af88e | doc/optana/loop_tile_lower.md (296 lines, §0-§8 + §7bis) |
| 2 | Convert smem_insertion.md to pointer stub | e299914 | doc/optana/smem_insertion.md |

## OPEN-discipline verification (the two items that must read as undecided)

- §4 is framed "open-questions register, NOT a decision record": D1-independent ground (strided tile coverage of [0, a_tile_max); the uniform bound min(T, A − offset) folding tile+extent clamp) stated as normative; Q1-Q5 each presented as question + option space + consequence with NO marked resolution; Q2-Q5 separability from Q1 noted.
- Every tail-touching obligation elsewhere cross-references §4 instead of assuming an answer: §3(b) tail-guard mechanism, §6 compiler obligations (uniform call + bounds only), §7 dynamic-shapes predication, §7bis bounds checking, §8 self-check item.
- §7bis presents the allocator question (who computes the per-LNT gridDim.x/blockDim.x split, percentage sharding vs successor) without picking; the RULED items (per-LNT scoped maps, 1-D launch contract, runtime-owned linearization and sync) are labeled RULED and kept separate.

## Verification results (plan greps — all present)

- 'loop tiling rewrite' (opt.md position cite, §0), LoadTileCta, MemNewSMem, ceildiv, "fully unroll", Requires, allocator vocabulary (§7bis OPEN), Q1-Q5 facets — all found; 10 sections; 296 lines (min 150).
- Pipeline position cites doc/opt.md ONLY; §0 explicitly disclaims ROADMAP/tca.md ordering.
- All cited internal/ paths verified to exist: internal/ir/kind.go (KindForLoop, KindSelect, KindVersion 5), internal/ir/node_lir_memory.go (MemNewSMem/FragNew/StorageClass), internal/opt/typeinfer/rules.go (closed catalog, fails on unknown kinds), internal/ir/mutate.go (RAUW + choke points), internal/ir/builtins.go (only f32 fminf/fmaxf — confirming the u32-builtin gap).
- LoadTileCta/LoadTileThread grounded as tca::rt:: runtime variants per doc/tca.md §5.11 L767-768 (not IR node kinds).

## Deviations from Plan

None — plan executed exactly as written (amended v2 plan, task 1's action block followed section by section).

## Self-Check: PASSED

- doc/optana/loop_tile_lower.md exists (15,281 bytes, 296 lines)
- doc/optana/smem_insertion.md exists (pointer stub, no residual old content)
- Commits 96af88e and e299914 present on worktree-agent-ab5fe14fd65f0d8d1, directly atop base dd2e81a
