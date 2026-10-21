---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Phase 01.1 complete (verified 8/8), next Phase 02
last_updated: "2026-10-16T02:30:00.000Z"
last_activity: 2026-10-20 - Completed quick task 261020-ve6: implement loop-tiling decide pass per doc/optana/loop_tiling.md
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 6
  completed_plans: 1
  percent: 17
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-15)

**Core value:** `matmul_scale` (and the other documented example kernels) compiles end-to-end and runs correctly on the target GPU with numerical results matching a CPU reference — frontend source → TCA IR → optimization passes → CUDA C++ + runtime, in one automated flow.
**Current focus:** Phase 01.1 — ir-architecture-refinement (verification next), then Phase 02

## Current Position

Phase: 02 (naive-codegen-runtime-e2e) — NEXT
Plan: 0 of 4
Status: Phase 01.1 complete (verified 8/8, goldens byte-identical); ready for code review
Last activity: 2026-10-21 - Completed quick task 261021-mlx: code review of loop_tile + loop_tile_lower — decide half sound, lower half's PHASE 4 cache insertion not shippable (2 CRITICAL: u32 tail wrap, fills never placed at fill point); 20 findings in REVIEW.md

Progress: [██░░░░░░░░] 17%

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

### Roadmap Evolution

- Phase 01.1 inserted after Phase 1: IR architecture refinement — extensible Type interface (NumericType), SSAUse/SSADef/SSAVar handle redesign, visitor-pattern graph walk (URGENT; user feedback from IR construction pass review, 2026-10-16)

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

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261016-mpu | Refactor phase 1 IR from AST-style to SSA/CFG optimizing-compiler IR per doc/review/phase1_1.md | 2026-10-16 | c470396 | [261016-mpu-refactor-phase-1-ir-from-ast-style-to-ss](./quick/261016-mpu-refactor-phase-1-ir-from-ast-style-to-ss/) |
| 261017-vdj | Adapt phase-1 SSA IR to new two-level design (sea-of-nodes L1, scheduled CFG L2) per tinycudaopto skeleton | 2026-10-17 | 1440816 | [261017-vdj-adapt-phase-1-ssa-ir-to-new-two-level-de](./quick/261017-vdj-adapt-phase-1-ssa-ir-to-new-two-level-de/) |
| 261018-bl0 | Re-organize IR per tinycudaopto skeleton (layered node_xxx.go files, HIR/LIR catalog, one-end-node model, two-root-set DCE, pass-level topo+LICM scheduler) | 2026-10-18 | ffb1497 | [261018-bl0-re-organize-ir-per-tinycudaopto-skeleton](./quick/261018-bl0-re-organize-ir-per-tinycudaopto-skeleton/) |
| 261018-ked | Write comprehensive IR documentation (doc/ir.md — organization, two-level design, catalog, SSA, effects, DCE, poset, scheduling, worked examples) | 2026-10-18 | b44eb18 | [261018-ked-write-comprehensive-ir-documentation-doc](./quick/261018-ked-write-comprehensive-ir-documentation-doc/) |
| 261018-l36 | Implement mark-sweep DCE pass on TCA IR + reorganize internal/opt pass manager (registry/factory/all, init self-registration, priority ordering, shared analysis cache) | 2026-10-18 | 84eef54 | [261018-l36-implement-mark-sweep-dce-pass-on-tca-ir-](./quick/261018-l36-implement-mark-sweep-dce-pass-on-tca-ir-/) |
| 261018-l36-f1 | fast: add dce-verify post-sweep invariant pass + chain-integrity regression tests (middle-link conservatism, dead-tail detachment) | 2026-10-18 | ca26d2a | — |
| 261018-nga | Implement CSE via GVN: identity-keyed ValueHash/ValueEqual value contract, amended EffectOf verdicts (all FamilyMemory effectful), accumulated side-effect analysis in internal/analysis, GVN pass + verifier at 110/111 | 2026-10-18 | d8bce2e | [261018-nga-cse-via-gvn-side-effect-analysis-global-](./quick/261018-nga-cse-via-gvn-side-effect-analysis-global-/) |
| 261019-jck | Refactor internal/analysis to mirror internal/opt registry pattern: top-level registry/abstract-factory infra, sideeffect/ sub-module with init self-registration, all/ composition root | 2026-10-19 | e451b2d | [261019-jck-refactor-internal-analysis-to-mirror-int](./quick/261019-jck-refactor-internal-analysis-to-mirror-int/) |
| 261019-ktm | Implement IterCluster lowering per doc/optana/lower_itercluster.md (5 new node kinds, As/KeyedAttribute Input-list layout, simplify-lower/inline/verify passes) + revision: universal Input/ImpactBy-only convention machine-checked by the verifier (D14-D16) | 2026-10-19 | 9249a4e | [261019-ktm-implement-itercluster-lowering-feature-f](./quick/261019-ktm-implement-itercluster-lowering-feature-f/) |
| 261019-tu3 | Implement type inference pass per doc/optana/type_inf.md: IR type cell (Base.ty, TypeOf/IsErased, MutContext.SetType choke point, ir.Builtin registry) + typeinfer pass at priority 105 (DAG post-order compute-then-stamp, full 31-kind catalog rules, idempotent, fail-closed) | 2026-10-19 | 1a4beb0 | [261019-tu3-implement-type-inference-pass-from-doc-o](./quick/261019-tu3-implement-type-inference-pass-from-doc-o/) |
| 261019-uoq | Implement loop fusion via GVN per doc/optana/loop_fusion.md: KindIter reclassified side-effect free (sideeffect baseVerdict + gvn-verify mirror in lockstep) so identical headers/bounded headers/whole parent-wired nests fuse in one run; + internal/analysis/loopforest ("loop-forest" registered analysis, doc's reversed-edge forest algorithm, on-demand forest-of-trees for future loop tiling) | 2026-10-19 | 7fc95e6 | [261019-uoq-implement-loop-fusion-via-gvn-loop-nest-](./quick/261019-uoq-implement-loop-fusion-via-gvn-loop-nest-/) |
| 261020-b4f | Restructure KindIter to the sound 2-parameter formation NewIter(bdv, parent?) per doc/optana/{loop_fusion,induction_variable}.md (unbounded form deleted; BIV as the compared value object); simplify-lower builds wired DimVar→BIV→Iter nest chains per as-order interned (dim, parent) — forest shows real nest trees not singletons; GVN prefix fusion (ABC∩ABCD) + post-fusion forest chain pinned | 2026-10-20 | 90d66bb | [261020-b4f-pin-gvn-prefix-fusion-tests-wire-loop-ne](./quick/261020-b4f-pin-gvn-prefix-fusion-tests-wire-loop-ne/) |
| 261020-ejf | Implement memory optimization pass per doc/optana/memory-opt.md (review verdict SOUND w/ conditions C1-C4): memopt@112 store→load forwarding + store-store + dead-store elimination over chain-eligible regions, memopt-verify@113 fixpoint checker, MutContext.RemoveImpactBy splice choke point, 10 property-test families | 2026-10-20 | 64f8413 | [261020-ejf-implement-memory-optimization-pass-from-](./quick/261020-ejf-implement-memory-optimization-pass-from-/) |
| 261020-svu | Review loop tiling design doc doc/optana/loop_tiling.md (verdict: feasible-with-required-changes; 22 findings, 6 BLOCKING — findINS bucketing bug vs matmul example, sum-vs-product LoC, one-BB predication, decide-vs-lower pass split + decision persistence; LP/MIP-solver + TMMA/tensor-core future-proofing section) → doc/review/loop_tiling.md | 2026-10-20 | deb5888 | [261020-svu-review-loop-tiling-design-doc-doc-optana](./quick/261020-svu-review-loop-tiling-design-doc-doc-optana/) |
| 261020-u2i | Revise doc/optana/loop_tiling.md per author rulings on review 261020-svu findings: F1 cache-tie semantics (TIE/WHOLE-DIM-LOAD/DEMOTION rules + set-vs-tie two-level model: SET=innermost referenced for co-traversal grouping + length≥2 filter, TIE=innermost non-referenced for placement), F4 external Max.cta/Max.threads_per_cta thresholds (product CTA / sum warp accounting, 1-D mapping as runtime contract), F5 per-LNT scoped runtime maps, F8 traffic-avoided objective, F9 innermost-K=5 search window, F10 coherent tile-id rewrite templates | 2026-10-20 | 943712d | [261020-u2i-revise-doc-optana-loop-tiling-md-per-use](./quick/261020-u2i-revise-doc-optana-loop-tiling-md-per-use/) |
| 261020-fd1 | fast: Cache.dims configurable cached-dimension count (dimension level reuse, default 2) + tensor-rank < Cache.dims filter at candidate set formation in doc/optana/loop_tiling.md | 2026-10-20 | 9f09a8b | — |
| 261020-ux8 | Draft doc/optana/loop_tile_lower.md — merged loop-tile-lower pass spec (tile rewrite + DoSoC numbering + intrinsics + SMEM/FRAG insertion) at opt.md's "loop tiling rewrite" slot; 11 author rulings normative (side-table decision record, FRAG unroll+budget, u32 min/max/ceildiv, string intrinsic payload + 6 companions, LoadTileCta/Thread rule, runtime-owns-sync, halve-then-drop fallback, cross-LNT dedupe, dynamic shapes, LIR KindForLoop); Issue #1 control-flow register (Q1-Q5) + D3 allocator vocabulary OPEN; smem_insertion.md → pointer stub | 2026-10-20 | 26a1288 | [261020-ux8-draft-doc-optana-loop-tile-lower-md-merg](./quick/261020-ux8-draft-doc-optana-loop-tile-lower-md-merg/) |
| 261020-fd2 | fast: resolve loop_tile_lower.md Issue #1 — runtime loop materialization owns the bounds bailout (thr_loop whole-body no-op / cta_loop exact / seq_loop native); one-BB permanent; tile-level cooperative ops outside bailing loops; mask contract (exact, uniformly-derived, per-iteration, lane-prefix closed form from thread_map); reducer Init/Combine placement; Q1-Q5 all resolved — only D3 allocator vocabulary remains OPEN | 2026-10-20 | 9b124ff | — |
| 261021-gca | Final D3/composition patch per author rulings: loop_tiling.md — per-LNT constraints (≥1 CTA dim, ≤5 innermost SoC dims, ΣCTA portions=PORTION_CTA/10, Σwarp portions=PORTION_WARP/8, −1 remainder, assert-and-trap multiples, defaults 100 CTAs/512 threads/16 warps), single-term traffic-avoided objective (weighted sum deleted), Composition Semantics section (per-CTA warp space, structural m×n cross product, chunk-stride WARP-between vs element-stride CTA-coupled-inside), all templates 0-based induction + in-body rank offsets; loop_tile_lower.md — D3 RULED (one GridMapping + one ThreadMap per LNT, tuple coords via reverse stride-shape, warp/lane/is_first queries, counting invariant p=m/q=n, ThreadMap dims m+n), §4 all-loops-guarded incl. cta_loop, v0 searcher policy (≤1 CTA dim + ≤1 independent WARP dim; same-category multi-dim FUTURE) | 2026-10-21 | ecd8620 | [261021-gca-final-d3-composition-patch-for-loop-tili](./quick/261021-gca-final-d3-composition-patch-for-loop-tili/) |
| 261021-i1o | Refactor internal/opt/tiling (tiling@114) from legacy constraint/objective model to the final loop_tiling.md design: Params surface (PortionCTA=10/PortionWarp=8/AutoWarpPortion=4 constants, WarpPortion field, weights + LoC presets deleted, defaults 100 CTAs/512 threads, assert-and-trap divisibility validation), search (portion completion replacing greedy preset loops, single-term score=traffic, v0 policy filter ≤1 CTA + ≤1 independent WARP dim, CTA-carrying fallback per the ≥1-CTA ruling), tiling-verify@115 invariants (CTA LoC==MaxCTA, warp LoC portion-multiple, policy on records), tests pinned to final model; overlap/record/window/promotion-judge confirmed already-final and untouched (matmul worked example zero-diff); gates green: build/vet/test full-repo | 2026-10-21 | d611190 | [261021-i1o-refactor-internal-opt-tiling-implementat](./quick/261021-i1o-refactor-internal-opt-tiling-implementat/) |
| 261021-j24 | Add the complete mechanical lowering procedure to doc/optana/loop_tile_lower.md: new §6bis (Phase 0 IR prerequisites — intrinsic kinds + six companions, u32 min builtin, KeyedAttribute loop annotations; Phase 1 record fetch fail-closed; Phase 2 DoSoC numbering; Phase 3 loop rewrite mechanics — TileSize==1 skip, between/inside forms per SoC with 0-based strides and coupled/thread chunk semantics, materialized bounds, IV RAUW; Phase 4 cache insertion — fact recomputation, fill point at innermost covered round body, MemNewSMem/FragNew + LoadTileCta/LoadTileThread, MemLoad→MemLoadSMem/FragLoad access rewrite, loads-only; Phase 5 epilogue) + §6 Site Rewrite Mechanics block (RAUW surgery, fill point, buffer lifetime, loads-only, row-major layout) + four §8 mechanical exit conditions + M1-M4 implementer defaults — doc now self-contained for implementing the lower pass | 2026-10-21 | adb391f | [261021-j24-add-the-complete-mechanical-lowering-pro](./quick/261021-j24-add-the-complete-mechanical-lowering-pro/) |
| 261020-ve6 | Implement loop-tiling decide pass per doc/optana/loop_tiling.md — internal/opt/tiling (tiling@114 + tiling-verify@115): MemLoad overlap analysis (findINS via Input-chase to InductionVariable, SET bucketing at innermost referenced iter, rank<Cache.dims + set-size≥2 + Max.cs filters), TIE placement (innermost non-referenced iter, above-outermost allowed), PSched 3^K flat search over innermost-K=5 window (LoC presets, tile presets, CTA-product ≤ Max.cta / warp-sum×32 ≤ Max.threads_per_cta / SMEM-bytes ≤ Max.cache feasibility, parallelism + traffic-avoided weighted objective), D2 decision-record side table (per-Iter by BIV identity, per-site by MemLoad id, NO ctaCache field, zero IR mutation) exposed via analysis cache with Preserved ["tiling-record"]; matmul worked example pinned (SET {lhs,rhs}@iterK, lhs tie iterN covers [·][K], rhs above-outermost covers [N][K]); 17 packages green, 18 new tests | 2026-10-20 | 2a776aa | [261020-ve6-implement-loop-tiling](./quick/261020-ve6-implement-loop-tiling/) |
| 261021-mlx | Code review loop_tile + loop_tile_lower passes vs specs and author rulings (20 findings: 2 CRITICAL, 6 HIGH, 5 MEDIUM, 7 LOW; build/vet/test green throughout) — decide half fundamentally sound (boundary issues only: per-LNT vs kernel-wide SMEM budget, §7 cross-LNT union unimplemented, coupled≥1 floor unenforced); lower half PHASE 2-3 loop rewrite solid but PHASE 4 cache insertion not shippable: CR-01 u32 tileMax tail arithmetic wraps (OOB for every real extent, whole-body no-op never triggers), CR-02 fills never structurally placed at §6bis fill point (float outside every loop, one-shot at Materialize), HI-01/02 scope-inconsistent fill coordinates, HI-03 incoherent WARP-site lowering, HI-05 branched LNTs silently mis-lowered; green tests pin node shapes not semantics → REVIEW.md | 2026-10-21 | dbb9baf | [261021-mlx-code-review-loop-tile-and-loop-tile-lowe](./quick/261021-mlx-code-review-loop-tile-and-loop-tile-lowe/) |

## Session Continuity

Last session: 2026-10-15T14:13:51.541Z
Stopped at: Phase 1 context gathered
Resume file: .planning/phases/01-ir-core-pass-infrastructure-tooling/01-CONTEXT.md
