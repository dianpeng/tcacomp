---
phase: quick-261022-l1o
plan: 01
subsystem: opt/final-lower + ir/runtime-nodes
tags: [final-lower, reducer-lowering, map-lower, runtime-objects, tiling-provenance]
requires:
  - doc/optana/final_lower.md (as amended — design review 261022-g6i verdict GO, all 28 findings closed in-doc)
  - doc/tca.md §5.10 (reduce lowering law)
  - internal/opt schedule/tile-lower passes (AST CFG, SchedSN/SchedBlock linkage, tiling decisions)
provides:
  - internal/opt/finallower — the final-lower pass (priority 950, Requires schedule), the last pass before codegen
affects:
  - internal/ir (8 new runtime node kinds, Intrinsic→Runtime rename, InsertItemAt/RemoveItemsIf AST utilities)
  - internal/opt/tiling (tiling-provenance record publication)
  - internal/opt/schedule (Preserved tiling-provenance; anchorSchedule skips RoleLambda params)
  - internal/opt/simplify (Reduce-inline exemption, Map-only verify invariant)
  - internal/opt/typeinfer (runtime kind rules)
tech-stack:
  added: []
  patterns:
    - tiling provenance as analysis-cache DATA (ir knows nothing about opt/tiling)
    - dedicated construction nodes + two generic mechanical-literal nodes (RuntimeCallMethod/RuntimeGetAttribute) with closed string payloads
    - expression-spine rebuild with substitution (tile-lower rebuildAxis discipline) for sound coordinate re-basing on shared expressions
key-files:
  created:
    - internal/ir/node_runtime.go
    - internal/opt/tiling/record.go
    - internal/opt/finallower/pass.go
    - internal/opt/finallower/maplower.go
    - internal/opt/finallower/objects.go
    - internal/opt/finallower/reducer.go
  modified:
    - internal/ir (kind/catalog/verify/node/print/effect/astcfg + rename lockstep across the repo)
    - internal/opt/{registry,all,schedule,tiling,simplify,typeinfer} touch points
    - internal/opt/finallower/finallower_test.go
decisions:
  - "Executor terminated mid-Task-3 by an API rate limit (429); the orchestrator resumed in-worktree and finished Task 3 — three defects fixed: (1) the re-iteration qualifying set narrowed to Combine-value CONSUMERS per doc/tca.md §5.10 point 4 (the between-IV-only criterion wrongly captured the scheduler-anchored piped Map projection, a producer); (2) the re-iteration bound gate scope-checks only DimVar leaves (computed SSA values are legally reused after X closes; the bound node itself is plan-pinned to be shared with the between loop); (3) betweenBlocks excludes home itself — the reduced dim's home loop is the terminus, never a slot-index space nor a re-iteration target"
  - "The interim live-Reduce gate from Task 2 was replaced by the real lowerReducers call ordered BEFORE lowerMaps (LR-2: @r derivation reads Reduce.op0 → producing Map → AsOf)"
metrics:
  duration: ~2.5 h (planner 23 min; executor tasks 1-2 + terminated task 3; orchestrator completion of task 3)
  completed: 2026-10-22
---

# Quick Task 261022-l1o: Implement final lowering pass — Summary

**One-liner:** The final-lower pass (internal/opt/finallower, priority 950 after schedule)
is implemented per doc/optana/final_lower.md as amended — reducers become specialized
runtime-object constructions with bracket-law Init/Reduce/Combine placement and
fused-consumer re-iteration, Maps are eliminated with full debris disposal, every kernel
gains a KernelInfo first statement and every LNT its mapping pair, and every launch-map
query becomes an explicit RuntimeGetAttribute — full repo gate green.

## What was done

**Task 1 — Runtime node taxonomy + plumbing** (`091630b`):
- Intrinsic→Runtime rename repo-wide (KindRuntimeGridQuery/ThreadQuery, node_runtime.go,
  RuntimeAttributeOf; enum values unchanged).
- 8 new kinds: RuntimeKernelInfo, GridMapping/ThreadMapping (portion grammar),
  RuntimeSequentialReducer/WarpReducer/GlobalReducer (construction carries ALL state),
  RuntimeCallMethod/RuntimeGetAttribute (closed payload vocabularies); FamilyRuntime new,
  FamilyCollective's first members; six companions complete (catalog, allocNode, verify,
  typeinfer, print, hash/equal).
- Effect model: CallMethod ordered side effect, GetAttribute pure (GVN-sound).
- Reduce-inline exemption in simplify-inline + Map-only simplify-verify invariant split.
- Tiling provenance record (per-dim Between/Inside/Soc/Tile keyed by dim InputParam,
  per-LNT mapping plans) published by tile-lower under `tiling-provenance`, preserved
  through schedule; anchorSchedule skips RoleLambda params.

**Task 2 — final-lower part 1** (`533ec2f`):
- astcfg.go: InsertItemAt + RemoveItemsIf generic block utilities.
- Pass contract: final-lower@950, Requires schedule, fail-closed gates (nil graph,
  no AST, provenance fetch), Run = gates → reducers → maps → objects → queries →
  erased-item sweep → ir.Verify.
- Map Lower (LR-9): RAUW onto the body expression, explicit As erasure, conservative
  local debris sweep (never a whole-graph DCE — runtime-call statements have no users).
- Runtime objects (LR-11/LR-12): KernelInfo as kernel Body item 0; per-LNT
  GridMapping+ThreadMapping prepended to the OWNING prologue; every query rewritten to
  RuntimeGetAttribute(%mapping, field, dosoc) — query kinds vanish.
- Interim live-Reduce gate (replaced in Task 3).

**Task 3 — Reducer lower** (`186c3bd`, completed by the orchestrator after the executor
was terminated by an API rate limit mid-task):
- @r derivation fail-closed (LR-2), bracket law (Init before X / Reduce inside home with
  between-loop slot IVs / Combine after X, one reduction spanning ALL tiles),
  Soc specialization, homogeneity (LR-8), Combine RAUW, fused-consumer re-iteration
  (LR-14/P-12) with coordinate-spine rebuild onto fresh re-iteration dimvars.
- Three defects fixed on completion (see decisions above) — the qualifying set, the
  bound enclosure gate, the between-set off-by-one, plus a test that located X's item
  via the loop node instead of its Loop block.
- Doc conformance sweep: every normative line of doc/optana/final_lower.md mapped to
  code; zero gaps.

## Verification

- GATE: `go build ./... && go vet ./... && go test ./... -count=1 && gofmt -l internal pkg` — green.
- Plan grep checks: rename complete (only rename-note comments), runtime kinds present
  across all kind-agnostic companions, tiling-provenance publish/preserve/fetch exactly
  as specified, registry+blank import present.
- Behavioral: bracket placement / Combine RAUW / re-iteration shape / negative probes
  (no provenance, type mismatch, reduced-dimvar poison) / query-outside-LNT /
  map-only + tiled-reduce end-to-end with twin-run determinism, ir.Verify green.

## Commits

| Task | Commit | Subject |
|------|--------|---------|
| 1 | `091630b` | feat(261022-l1o): Runtime node taxonomy, rename, companions, provenance plumbing |
| 2 | `533ec2f` | feat(261022-l1o): final-lower pass part 1 — contract, AST items, Map Lower, runtime objects, query rewrite |
| 3 | `186c3bd` | feat(261022-l1o): reducer lower — bracket law, specialization, slot IVs, re-iteration |

## Known gaps / follow-ups

- None against the plan. The CTA (SoC 1 → RuntimeGlobalReducer) specialization is
  registered scaffolding, unreachable under the v0 scheduling policy (no CTA-parallel
  reduced dims) — per the design doc.
- Codegen consuming this pass's output is a future phase (out of scope).
