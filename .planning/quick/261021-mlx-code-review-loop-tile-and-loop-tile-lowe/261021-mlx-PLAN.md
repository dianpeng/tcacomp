---
id: 261021-mlx
type: quick
description: Code review loop_tile and loop_tile_lower passes
date: 2026-10-21
status: planned
tasks_total: 1
---

# Quick Task 261021-mlx: Code review loop_tile and loop_tile_lower passes

Review-only task (no source changes). Verify the tiling decide/lower implementation
is sane against its specs and recorded design rulings.

## Task 1: Review tiling passes

**files:**
- internal/opt/tiling/tiling.go, search.go, overlap.go, record.go, verify.go (decide half, `tiling`@114 + `tiling-verify`@115)
- internal/opt/tiling/lower.go, lower_cache.go, lower_rewrite.go (lower half, new)
- IR prerequisites: internal/ir/node_intrinsic.go, node_lir_memory.go, node_loop.go, kind.go, builtins.go, value.go, verify.go, print.go, catalog.go; internal/opt/typeinfer/rules.go
- Tests as coverage evidence: lower_test.go, tiling_test.go, overlap_test.go, verify_test.go

**action:** gsd-code-reviewer audit vs doc/optana/loop_tile_lower.md (§6bis mechanical
procedure, §6 rewrite mechanics, M1–M4 defaults) and doc/optana/loop_tiling.md
decide-half semantics; check pipeline wiring vs doc/opt.md ordering; check the
author's recorded rulings (fail-closed record fetch, copy-variant rule, runtime-owned
barriers, budget fallback, cross-LNT dedupe, dynamic shapes, TileSize==1 skip).

**verify:** REVIEW.md produced in this directory with severity-classified findings.

**done:** Findings relayed to user; docs committed.
