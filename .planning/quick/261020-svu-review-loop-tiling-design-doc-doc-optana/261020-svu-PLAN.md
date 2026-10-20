---
phase: quick-261020-svu
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - doc/review/loop_tiling.md
autonomous: true
requirements:
  - R1 (feasibility verdict on the design as written, user point 2)
  - R2 (soundness of findINS/union candidate-cache analysis + PSched constraint model + promotion criteria + rewrite pseudo-code, user points 2+4)
  - R3 (pass-split evaluation: tiling = decide/record vs new "loop tile lower" merged with smem_insertion, user point 3)
  - R4 (compatibility audit vs IR/verifier/SSA/one-BB, neighboring passes, runtime, user point 4)
  - R5 (missing-but-easy additions, user point 5)
  - R6 (future-proofing: LP/MIP solver-friendly reshaping + TMMA/tensor-core tile-shape and multi-stage SMEM/FRAG constraints, user point 6)

must_haves:
  truths:
    - "Every finding in doc/review/loop_tiling.md cites doc/optana/loop_tiling.md line numbers AND a grounding artifact (internal/ path or doc/ path with section)"
    - "The review states one of: feasible-as-designed / feasible-with-required-changes / infeasible-as-designed, with per-section justification"
    - "The findINS/union algorithm is checked against the doc's own matmul example (doc/optana/loop_tiling.md lines 135-163 vs algorithm lines 171-198) and the agreement or contradiction is settled precisely"
    - "The pass-split (decision-recording pass vs IR-rewriting loop-tile-lower merged with smem_insertion) is evaluated: what state must persist between the two passes, where it lives, and which pipeline slots each pass occupies"
    - "A concrete LP/MIP + TMMA section exists that names every non-linearity in the current formulation and the linearization/decision-variable restructuring for each"
    - "All six user points map to review sections (explicit coverage table)"
  artifacts:
    - path: "doc/review/loop_tiling.md"
      provides: "Structured review with severity-tagged findings, matching doc/review/phase1_1.md general style"
      min_lines: 150
  key_links:
    - from: "doc/review/loop_tiling.md soundness findings"
      to: "doc/optana/loop_tiling.md"
      via: "line-number citations"
      pattern: "loop_tiling\\.md:? ?L?[0-9]"
    - from: "doc/review/loop_tiling.md compatibility findings"
      to: "internal/ir/*, internal/analysis/loopforest/*, internal/opt/*"
      via: "file path citations"
      pattern: "internal/(ir|analysis|opt)/[a-z_/]+\\.go"
---

<objective>
Review the loop-tiling optimization design doc (doc/optana/loop_tiling.md, 593 lines) for feasibility, correctness, soundness, and compatibility with the existing system — evaluated WITH the user-mandated pass split in mind (tiling pass only decides and records; the IR rewrite moves to a new "loop tile lower" pass merged with the later SMEM-insertion pass).

Purpose: this is the compiler's core optimization pass; a wrong candidate-cache analysis, an unsound constraint model, or an IR-rewriting scheme that violates the one-BB SSA invariants would poison every downstream phase. The review must catch design-level defects before implementation begins.

Output: doc/review/loop_tiling.md — structured review, severity-tagged findings (BLOCKING / SOUNDNESS / COMPAT / MISSING-EASY / FUTURE), verdict, and a concrete LP/MIP + TMMA future-proofing section.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@doc/optana/loop_tiling.md (review TARGET — read fully, with line numbers)

Ground-truth artifacts the executor MUST verify the doc's assumptions against (read each once; grep for specifics rather than re-reading):
- doc/tca.md — IR/semantics spec: sections 2.3 (induction vars), 2.4 (iterations), 3.3 (Tiling Rule), 5.7.x (Automatic Parallelization: 5.7.1 remapping the CUDA thread hierarchy, 5.7.2 warp-based assignment, 5.7.3 parallelism property)
- doc/ir.md — implementation-faithful IR reference for internal/ir (node catalog, Input/ImpactBy-only convention, effect model, DCE two-root-set, poset/scheduling); section 1.2 lists known drift vs doc/tca.md
- doc/opt.md — pipeline overview (loop tiling and smem/frag insertion listed as separate stages)
- doc/optana/{induction_variable,loop_fusion,lower_itercluster,memory-opt,smem_insertion}.md — neighbor specs. NOTE: smem_insertion.md is a 3-line stub — the merged "loop tile lower" pass has no spec today
- internal/analysis/loopforest/loopforest.go — Forest/Tree/Roots()/NodeOf(); package doc explicitly names loop tiling as first consumer; members are live KindIter only; read-only analysis
- internal/ir/: kind.go (KindVersion 5, 31 kinds; HIR KindIter/KindInductionVariable/KindMemLoad; LIR KindForLoop/KindMemLoadSMem/KindFragLoad etc.), value.go (ValueHash/ValueEqual identity contracts), verify.go + convention.go (Input/ImpactBy-only convention machine-checked; verifier runs after every pass), effect.go (effect verdicts), node_loop.go / node_memory.go / node_var.go (Iter(bdv,parent), MemLoad, InductionVariable shapes), mutate.go (MutContext choke points), types.go (TypeOf/IsErased), builtins.go (ir.Builtin registry — check whether min/ceil/max exist), dce.go/mark.go (two-root-set liveness)
- internal/opt/: registry.go (pass priorities: gvn, memopt@112, typeinfer@105, simplify), manager.go + cache.go (analysis invalidation), legality.go (pass legality windows)
- /home/ps/.claude/projects/-home-ps-Workspace-cuda-project-tcacomp/memory/opt-pipeline-audit-rulings.md — prior audit rulings (memopt dead-load gap open; remaining stages WIP)
- .planning/ROADMAP.md — phases 3-4 pipeline order (parallelize → iter-lower → reduce-lower → cache-alloc → cache-lower) that the split passes must slot into
- pkg/ and design.md — runtime scaffolding status; the doc's GridConcurrencyMap/ThreadMap runtime objects presuppose runtime features (arbitrary multi-dim remap, whole-warp granularity, launch validation) that may not be specced yet

Style reference for the deliverable: doc/review/phase1_1.md (terse, structured, bullet-driven), extended with severity tags and a findings table as the user requested.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Core-algorithm review — feasibility + soundness of candidate-cache analysis, PSched model, solver, rewrite semantics</name>
  <files>doc/review/loop_tiling.md (create: header, method note, sections 1-4)</files>
  <action>
Read doc/optana/loop_tiling.md fully with line numbers, then verify each of its algorithmic claims against the ground-truth artifacts listed in context. Write sections 1-4 of doc/review/loop_tiling.md: (1) Feasibility verdict; (2) Soundness of the MemLoad overlap analysis (findINS + union/grouping, doc lines 164-198); (3) Soundness of the PSched constraint model, objective, and search (lines 200-341); (4) Soundness of the tiling-rewrite semantics (lines 350-591) — reviewed as the spec of the future loop-tile-lower pass per the split.

MUST-verify checklist (verify or refute each with line + code citations; these are candidate findings from pre-reading, not conclusions — the executor settles each):

A. findINS/union vs the doc's own example. The algorithm (lines 182-188) scans Iter nodes innermost-to-outermost and buckets each MemLoad at the FIRST (innermost) iter in its INS. Trace the doc's matmul example (lines 135-163): lhs[m][k] has INS={iterM,iterK}, rhs[n][k] has INS={iterN,iterK}; innermost-first bucketing puts BOTH loads at the iterK level. But the doc's own commentary (lines 152-155) requires lhs cached at the iterN level and rhs at the iterM level — i.e., placement derived from the innermost NON-referenced dim (the reuse level), which is the outermost-missing-dimension logic, not innermost-referenced bucketing. Determine precisely whether the algorithm computes what the commentary claims; if not, this is a BLOCKING/SOUNDNESS finding and the easy fix (bucket by reuse level = innermost iter of the LNT NOT in the load's INS, restricted to loads with >=2 subscript dims) should be sketched.
B. ValueEqual-on-Iter assumption (line 178, 184): confirm internal/ir/value.go's ValueEqual contract actually covers KindIter and KindInductionVariable the way loop_fusion.md defines nested-aware equality, and that using it for INS membership is consistent with GVN's use. Note the assumption that subscript operands are directly InductionVariable nodes (per lower_itercluster.md format) — doc says "chasing the Input list until hit all InductionVariable" which implies chasing through expression trees; state the exact assumption and what happens for future affine subscripts (induction_variable.md says unsupported today).
C. Cross-LNT loads: a MemLoad whose subscripts reference iters of two different LNTs is collected by step (1) for each LNT (line 168). Is double-collection sound? Also: does the grouping survive fusion (identical-header nests fused by GVN share Iter nodes — is INS then per-tree or cross-tree)?
D. PSched constraint shape (lines 264-269): "ALL LoC added for CTA level must be less than CTA.max_loc" uses SUM semantics, but with normalized 1-D launches (lines 73-81) total CTAs = PRODUCT of per-dim CTA LoCs across assigned dims, and total threads = product of warp LoCs x 32 (whole-warp rule, lines 343-345). Check against real CUDA limits (gridDim.x, 1024 threads/block, sm_86) and tca.md 5.7.1-5.7.3. Sum-vs-product is a potential SOUNDNESS finding.
E. Multi-LNT kernel accounting: each LNT gets its own GridConcurrencyMap section (line 101) but constraints are per-PSched while the LAUNCH is per-kernel — cross-LNT total CTA/thread accounting is unaddressed. Also the DoSoC numbering pass (lines 364-386) maintains per-LNT counters — what prevents dimension collisions across LNTs sharing one grid/thread map?
F. Cache promotion criterion (lines 273-288): untangle the index notation ("all CTA assignment happens at iteration <= xn-1" — which subscript positions exactly; is xn the last subscript?). Check both readings against tca.md 5.7.3 parallelism property and the FRAG=per-thread vs SMEM=per-CTA justification (lines 277-281). Note the missing FRAG/register-pressure budget (only SMEM Max.cache exists, lines 287-288) — register spill would silently destroy the FRAG strategy.
G. Objective function: line 295 "(2) SMEM reuse can be given by" is TRUNCATED mid-sentence — the objective is incomplete as written. Additionally line 331 maximizes cache_weight * total_cache (total cache BYTES), which conflates footprint with reuse and directly fights the Max.cache budget constraint. Propose the correct linearizable metric (e.g., GMEM traffic avoided per promoted site).
H. Solver defects: line 316 "if (is_feasiable(each_sched)) continue;" inverts the feasibility filter (skips FEASIBLE schedules); line 322 get_soc(p) signature sloppiness; search-space size of the 3^K enumeration (line 314) x LoC presets x tile-size presets (lines 299-304) — give the actual complexity for a depth-K nest and judge feasibility for realistic K (3-6).
I. Rewrite pseudo-code defects (these move to loop-tile-lower under the split, but the review must still settle the semantics): line 437-438 bound/stride unit mismatch (a < ceil(A/tile) vs a += tile — counts tiles but strides in elements); line 449-451 b_tile_max uses min(tile_size_a, B - tile_size_b*b) — wrong tile variable; CTA case lines 515-519 inner loop increments a instead of a_tile and ignores the computed a_tile_max; warp case lines 580-584 same wrong-increment bug; CTA outer loop strides by tile*nCTA (line 501-505) while the offset uses grid_map.index * tile (line 513) — determine whether the outer construct is a real grid-stride loop or a disguised tile-id enumeration (these two are inconsistent as written; propose the coherent form via index decomposition in GridConcurrencyMap).
J. One-BB crux: the IR has exactly one basic block and no control flow (project constraint). Sequential tile-inside loops can materialize a uniform bound (min expression) — but PARALLEL strided loops give each thread a different trip count when the bound is not divisible; the doc's "Ensure bounds check" note (lines 408-411) only covers the uniform case. Per-thread guards require either control flow (violates one-BB) or predication. Check whether KindSelect (kind.go) / masked forms can carry the predication, and what tca.md 5.7.3 assumes. This is the single most important feasibility question for the rewrite half — treat accordingly.

Each finding gets: severity, doc line(s), grounding citation (file:line or doc section), one-paragraph analysis, and a concrete fix sketch where one exists.
  </action>
  <verify>
    <automated>test -f doc/review/loop_tiling.md && grep -c "L\?[0-9]\{2,3\}" doc/review/loop_tiling.md >/dev/null && grep -ci "severity\|BLOCKING\|SOUNDNESS" doc/review/loop_tiling.md</automated>
  </verify>
  <done>Sections 1-4 exist with every checklist item A-J settled (verified or refuted) and cited; at least the one-BB predication question (J) and the findINS-vs-example question (A) have definitive answers</done>
</task>

<task type="auto">
  <name>Task 2: Pass-split evaluation + compatibility audit against IR, verifier, neighboring passes, runtime</name>
  <files>doc/review/loop_tiling.md (append sections 5-6)</files>
  <action>
Write section 5 (pass split) and section 6 (compatibility audit) of the review.

Section 5 — the split the user mandates (user point 3): loop tiling = analysis + decision recording; loop tile lower = new pass combining the tile rewrite (doc lines 359-591), the SoC-numbering/DoSoC pass (lines 364-386), and the existing smem_insertion stage (doc/opt.md; smem_insertion.md is a 3-line stub). Note the doc itself half-anticipates this at line 413-417 ("Separate a specific pass, well written, fully tests for this dedicated goal"). Evaluate concretely:
- Decision persistence: the doc records ONLY MemLoad.ctaCache (lines 352-358) — but tile_size, SoC type, and LoC per Iter must also survive until the lower pass, and DoSoC is computed at lower time. Identify every piece of state that must cross the pass boundary and where it can live (IR-visible marks vs opt.Cache side tables vs re-derivable): check internal/opt/cache.go invalidation rules and whether node identity (ValueEqual) is stable across intervening passes (gvn before, memopt@112 — consult registry.go priorities and the memory file opt-pipeline-audit-rulings.md). If intervening passes can erase/merge Iter nodes after decisions are recorded, that is a SOUNDNESS finding for the split; propose pipeline placement (e.g., tiling immediately before tile-lower, after memopt) reconciled with doc/opt.md order AND .planning/ROADMAP.md phases 3-4 (parallelize / iter-lower / cache-alloc / cache-lower slots).
- What the merged loop-tile-lower spec must contain that smem_insertion.md currently lacks (list it — this feeds R5).
- Interface risk: ctaCache as a MemLoad field changes ValueHash/ValueEqual identity (a marked and unmarked load of the same address must NOT GVN-merge) — verify value.go's contract and flag the required contract change.

Section 6 — compatibility audit against existing systems (user point 4). For each, cite the internal/ file + line that proves or disproves compatibility:
- IR node surface: the two new kinds IntrinsicGridMap/IntrinsicThreadMap (doc lines 483-496, 547-559 — note the doc duplicates the node list twice) vs kind.go KindVersion-5 catalog rules (LIR insertion before KindEnd, version bump policy), the Input/ImpactBy-only convention (convention.go, machine-checked per quick-261019-ktm — the "name of field" selector must ride the Input list, KeyedAttribute pattern exists at KindKeyedAttribute), verifier rules (verify.go — new kinds need verify cases), typeinfer (fail-closed 31-kind catalog per quick-261019-tu3 — two new rules needed or the pass rejects the graph), sideeffect verdicts (effect.go — intrinsics are pure/deterministic so GVN may CSE them; is that sound for grid/thread index queries?), DCE two-root-set liveness (mark.go/dce.go — are intrinsics reachable from the end node via ImpactBy or value chains?), and poset/scheduling (schedule.go — intrinsics as pure leaves).
- Verifier/one-BB/SSA invariants: the rewrite must preserve SSA (new tile induction variables are new defs; old InductionVariable uses must be rewritten to offset+inside-tile expressions via RAUW-style splice — check mutate.go MutContext choke points exist for this scale of patch; the doc calls it "large IRGraph patch", lines 413-415).
- Neighboring passes: loop fusion (fusion runs before tiling — forest must be stable; loop_fusion.md), memopt (store-to-load forwarding vs promoted-to-cache loads: does the tiling decision need to see pre-memopt loads or post? memory-opt.md), typeinfer rerun for lowered nodes (doc/opt.md lists a second type inference stage), scheduling/LICM (tile rewrite output must be LICM-able; check internal/opt/schedule).
- Runtime compatibility: GridConcurrencyMap arbitrary multi-dim remap (doc lines 83-125) vs what exists under pkg/ (check) and design.md's runtime contract (scalfold reference: explicit storage tiers, whole-warp groups, launch/shape validation, compiler-does-not-drive-build). Flag every runtime feature the design silently requires. Also cross-check the normalized-launch claims against tca.md 5.7.1-5.7.2 and the CLAUDE.md constraint that every thread group is a whole number of warps.
  </action>
  <verify>
    <automated>grep -c "internal/" doc/review/loop_tiling.md && grep -ci "pass split\|tile lower\|tile-lower" doc/review/loop_tiling.md</automated>
  </verify>
  <done>Section 5 names every piece of cross-pass state with a storage mechanism and a pipeline slot reconciled with ROADMAP phases 3-4; Section 6 covers IR kinds/convention/verifier/typeinfer/GVN/sideeffect/DCE, fusion+memopt+schedule neighbors, and runtime — each with an internal/ or doc/ citation</done>
</task>

<task type="auto">
  <name>Task 3: Missing-easy-adds, LP/MIP + TMMA future-proofing, verdict and self-check</name>
  <files>doc/review/loop_tiling.md (append sections 7-9, finalize)</files>
  <action>
Write sections 7-9 and finalize the document.

Section 7 — missing but easy to add (user point 5). Consolidate the gaps found in Tasks 1-2 plus: FRAG/register budget constraint (only SMEM Max.cache exists today, doc lines 287-288); determinism/tie-breaking in the search (identical scores must produce identical code — reproducible builds); dynamic-shape policy (TensorDimOf bounds are runtime values, but the search treats tile sizes as compile-time constants — state the static-shape assumption or the parametric-tile alternative); min/ceil/max expression availability for bound materialization (check internal/ir/builtins.go; if absent, adding them to the ir.Builtin registry is the easy fix); Max.cs truncation order-determinism (lines 190-198); multi-LNT grid accounting (Task 1 item E). Each item: what is missing, why it matters, the minimal addition.

Section 8 — future-proofing the formulation (user point 6). Two subsections, concrete enough to implement against:
(a) LP/MIP-solver-ready reshaping. The doc already gestures at linearity (lines 339-341). Enumerate every non-linearity in the current formulation: tile_size x tile_size products in the SMEM budget constraint (bilinear in decision variables); the promotion criterion as a logical implication over assignment variables (indicator constraints); LoC product-vs-sum launch totals; reuse-metric ratios in the objective. For each, give the standard linearization: categorical tile variables chosen from small discrete sets per dim (keeps cache-size terms linear once the category is fixed), binary SoC-type assignment variables x_{d,type} with exactly-one constraints, integer LoC variables with product terms linearized via big-M or eliminated by per-dim enumeration, indicator/big-M encodings for promotion implications, and a linear traffic-reduction objective. State the key architectural recommendation: define PSched feasibility and score as a decoupled constraint interface (predicate + objective terms over decision vectors) so the current hardcoded enumeration (lines 297-338) and the future MIP backend are interchangeable implementations of the same interface — the formulation stays solver-friendly without a solver dependency today (stdlib-only constraint).
(b) TMMA / tensor-core awareness. Near-future requirements: tile_size must satisfy divisibility/compatibility with hardware MMA instruction shapes (and the multi-stage TMMA copy instruction shapes on newer targets); SMEM/FRAG caches must be involved in async multi-stage copies (cp.async-style staging today per CLAUDE.md sm_86 ceiling, TMMA multi-stage copies later), which introduces buffer-stage-count decision variables, layout/swizzle constraints (alignment, bank-conflict avoidance) on the cached tiles, and producer/consumer tile-shape coupling through the SMEM/FRAG tier. Recommend the decision-variable restructuring needed NOW so these land as added constraints rather than a reformulation: per-dim tile categories tagged with instruction-shape compatibility classes, per-site promote + layout + stage-count variables in the decision record, and cache-size constraints expressed per (site, layout, stage) so multi-buffering just multiplies a linear term. Cite CLAUDE.md's sm_86-vs-sm_90 ceiling rules where relevant (cp.async now, TMA/TMMA later).

Section 9 — verdict and self-check: overall verdict (feasible-as-designed / feasible-with-required-changes / infeasible-as-designed) justified in one paragraph; findings summary table (ID, severity, section, one-line description); explicit coverage table mapping the user's six verbatim points to review sections. Re-read the finished document once for consistency (severities coherent across sections, no unresolved "TBD"), and confirm the general format matches doc/review/phase1_1.md's structured style.
  </action>
  <verify>
    <automated>grep -c "MIP\|LP" doc/review/loop_tiling.md && grep -c "TMMA" doc/review/loop_tiling.md && grep -ci "verdict" doc/review/loop_tiling.md && ! grep -ci "TBD" doc/review/loop_tiling.md</automated>
  </verify>
  <done>Sections 7-9 complete: every easy-add has a minimal-addition sketch; the LP/MIP subsection names each non-linearity and its linearization; the TMMA subsection gives the decision-variable restructuring; verdict + findings table + six-point coverage table present; document is internally consistent</done>
</task>

</tasks>

<threat_model>
Documentation-only review task: no runtime code, no untrusted input, no package installs. Trust boundaries: none crossed. STRIDE: not applicable — no security-relevant surface is produced or modified by this plan.
</threat_model>

<verification>
- doc/review/loop_tiling.md exists, >= 150 lines, structured like doc/review/phase1_1.md but with severity tags
- Every finding cites doc/optana/loop_tiling.md line numbers plus a grounding artifact (internal/*.go path or doc/*.md section)
- All six user points covered: (1) target doc identified in header; (2) feasibility section; (3) pass-split section; (4) soundness + compatibility sections; (5) missing-easy-adds section; (6) LP/MIP + TMMA section
- No source code is modified by this task (files_modified contains only the review doc)
</verification>

<success_criteria>
- Review doc written to doc/review/loop_tiling.md with a defensible verdict
- The four hard questions each have a definitive answer: findINS-vs-matmul-example agreement; sum-vs-product LoC semantics; one-BB predication for parallel tails; cross-pass decision persistence under the split
- Findings are severity-tagged (BLOCKING / SOUNDNESS / COMPAT / MISSING-EASY / FUTURE) with a summary table
- The LP/MIP + TMMA section is concrete (names non-linearities, gives linearizations, restructures decision variables) rather than advisory prose
</success_criteria>

<output>
Create .planning/quick/261020-svu-review-loop-tiling-design-doc-doc-optana/261020-svu-SUMMARY.md when done
</output>
