# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-15)

**Core value:** `matmul_scale` (and the other documented example kernels) compiles end-to-end and runs correctly on the target GPU with numerical results matching a CPU reference — frontend source → TCA IR → optimization passes → CUDA C++ + runtime, in one automated flow.
**Current focus:** Phase 1 — IR Core, Pass Infrastructure & Tooling

## Current Position

Phase: 1 of 5 (IR Core, Pass Infrastructure & Tooling)
Plan: 0 of 4 in current phase
Status: Ready to plan
Last activity: 2026-10-15 — Roadmap created (5 phases, 63/63 v1 requirements mapped); research summary validated the correctness-first phase shape

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: -

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: 5-phase correctness-first shape confirmed (IR+pass infra → naive codegen+runtime+e2e → parallelization+reduce → fusion/merge+caches → frontend+CLI); each phase e2e-verifiable on the A6000.
- [Roadmap]: Pass pipeline staged across phases; full PM-enforced order is simplify → merge → fusion → effect-linearize → parallelize → iter-lower → reduce-lower → cache-alloc → cache-lower → codegen. Phase 3 validates the parallelize→lower half on builder oracle graphs (pipes materialized to SSA at construction); Phase 4 lands the front half and validates the full order.
- [Roadmap]: Reducer contract (Init/Reduce/Combine, validity tags) defined in Phase 2; Phase 3 swaps only the collective implementation.
- [Roadmap]: compute-sanitizer harness mode built in Phase 2, required CI from Phase 3 on (VER-03 mapped to Phase 2).

### Pending Todos

None yet.

### Blockers/Concerns

- REQUIREMENTS.md header counted 57 v1 requirements; actual count is 63. Coverage block corrected during roadmap creation (all 63 mapped).
- Research flags for planning: Phase 1 legality-window API shape (MEDIUM confidence); Phase 2 CUDA 12.2 header verification (`__pipeline_*` signatures, C++20-subset corners).
- Doc-pinned policy gaps to close in-plan: FP/tolerance policy (Phase 2), ownership-predicate representation (Phase 3 design discussion), 64 KB budget padded-size interpretation (Phase 4).

## Session Continuity

Last session: 2026-10-15
Stopped at: Roadmap, state, and requirements traceability written; awaiting Phase 1 planning
Resume file: None
