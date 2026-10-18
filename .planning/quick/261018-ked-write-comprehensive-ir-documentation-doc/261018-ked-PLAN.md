---
phase: quick-261018-ked
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - doc/ir.md
autonomous: true
requirements: []
must_haves:
  truths:
    - "doc/ir.md exists and documents how internal/ir is organized: every file (node_xxx.go layering, catalog.go, ssa.go, effect.go, poset.go, walk.go, verify.go, dce.go, schedule.go, cfg.go, print.go, mutate.go, types.go, doc.go, kind.go, construct surface) with what each owns and why"
    - "doc/ir.md explains the two-level design: Level-1 sea-of-nodes with Input data deps + ImpactBy effect deps forming a poset, and Level-2 CFG/BB/Statement materialized by the default Materialize or re-optimized by pass.Scheduler (stable topo + LICM)"
    - "doc/ir.md documents the node catalog: every Kind categorized HIR vs LIR with per-kind semantics, matching internal/ir/catalog.go exactly"
    - "doc/ir.md documents the SSA handle layer (SSAVar/SSADef/SSAUse, use-def/def-use intrusive records) and the effect model (EffectOf verdicts, per-tensor Env effect chains, restrict-alias rule, the ONE end node via NewReturnChained carrying all observable effects)"
    - "doc/ir.md documents DCE (two-root-set liveness: end node + RoleTensor params; stores never roots; SMEM/FRAG unobservable) and the poset read face (DataDeps/EffectDeps/PosetDeps/SeaRoots)"
    - "doc/ir.md contains MULTIPLE conceptual-code → IR mapping examples: the REAL pkg/testdata/matmul_scale.ir golden quoted verbatim and annotated node-by-node, plus at least 3 smaller teaching examples (trivial scale kernel, two-store chain showing ImpactBy chaining, LIR cache-placement sketch with MemLoadSMem/FragLoad); every IR snippet is either verbatim from a committed golden or derived by reading the printer in internal/ir/print.go, with any illustrative-only line explicitly annotated"
    - "doc/ir.md cross-references doc/tca.md decision/requirement IDs (IR-01..IR-07, D-01..D-18 where relevant) and explicitly notes where doc/tca.md §5.1 notation predates the memstore( lowercase spelling (the follow-up flagged in quick-261018-bl0)"
  artifacts:
    - path: "doc/ir.md"
      provides: "Comprehensive IR documentation: organization, two-level design, node catalog, SSA layer, effect model, DCE, poset, scheduling, worked conceptual-code → IR examples"
      contains: "memstore("
  key_links:
    - from: "doc/ir.md worked examples"
      to: "pkg/testdata/matmul_scale.ir"
      via: "verbatim quote + annotation"
      pattern: "statement #8"
    - from: "doc/ir.md"
      to: "internal/ir/catalog.go"
      via: "node catalog section enumerates the same Kind set"
      pattern: "HIR|LIR"
---

<objective>
Write ONE in-depth, comprehensive documentation file `doc/ir.md` covering the TCA IR as it exists TODAY (post quick-261018-bl0): file organization, the two-level sea-of-nodes/CFG design, the full HIR/LIR node catalog, the SSA handle layer, the effect/ImpactBy model with the one-end-node rule, two-root-set DCE, the poset read face, and scheduling (default Materialize + pass.Scheduler with topo + LICM) — anchored by multiple conceptual-code → IR mapping examples, with the committed `pkg/testdata/matmul_scale.ir` golden quoted verbatim and annotated.

Purpose: doc/tca.md §5 specifies the algebra and pipeline but not the concrete Go IR. doc/ir.md is the implementation-faithful companion: every claim verifiable against current source, cross-referenced to doc/tca.md decisions (IR-01..IR-07, D-01..D-18) so the two docs interlock instead of drifting.
Output: `doc/ir.md` only. No code changes. No testdata changes.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/quick/261017-vdj-adapt-phase-1-ssa-ir-to-new-two-level-de/261017-vdj-SUMMARY.md
@.planning/quick/261018-bl0-re-organize-ir-per-tinycudaopto-skeleton/261018-bl0-SUMMARY.md
@.planning/phases/01-ir-core-pass-infrastructure-tooling/01-CONTEXT.md
@doc/tca.md
@doc/syntax.md
@pkg/testdata/matmul_scale.ir
</context>

<accuracy_discipline>
Every claim in doc/ir.md must be verifiable against the current code (post-261018-bl0, HEAD = ffb1497). Rules for the executor:
1. READ the actual source before writing each section. The full inventory to read (in this order):
   - `internal/ir/catalog.go` + `catalog_test.go` — the Kind list, Tier(), families, arities (this is the node-catalog section's source of truth)
   - `internal/ir/kind.go`, `doc.go` — KindVersion 3 numbering, package intent
   - `internal/ir/node.go` — Base struct, payload records, stmtID pinning, effect-root registration at alloc
   - `internal/ir/node_lambda.go`, `node_expr.go`, `node_var.go`, `node_memory.go`, `node_loop.go`, `node_dimvar.go`, `node_lir_memory.go`, `node_end.go` — per-family node semantics
   - `internal/ir/ssa.go` — SSAVar/SSADef/SSAUse handle layer
   - `internal/ir/effect.go` — EffectOf verdicts, ImpactBy records, Env, NewReturnChained, Env.FinalStates
   - `internal/ir/poset.go` — DataDeps/EffectDeps/PosetDeps/SeaRoots read face
   - `internal/ir/walk.go` — visitor walk, root sets, level-aware behavior
   - `internal/ir/dce.go` — mark/sweep, two-root-set liveness
   - `internal/ir/schedule.go` — default Materialize, PinStatementID
   - `internal/ir/cfg.go` — BasicBlock/Statement, intrusive list, NewStatement factory
   - `internal/ir/verify.go` — invariants, fail-closed end-node rule
   - `internal/ir/print.go` — canonical printer rules (chainOf, %N numbering, statement-boundary rendering, operandPrec) — REQUIRED for deriving teaching-example IR text
   - `internal/ir/mutate.go` — choke-point mutation, Erase record-killing
   - `internal/ir/types.go` — interned types
   - `internal/pass/scheduler.go` — pass.Scheduler (topo + LICM)
   - `pkg/builder.go`, `pkg/chain.go` — the construction surface the examples are written against
2. For IR example snippets: quote `pkg/testdata/matmul_scale.ir` VERBATIM for the worked example. For the smaller teaching examples, derive the printed form by reasoning over constructors + the printer source (chainOf / %N numbering / stage rendering). DO NOT invent IR text the printer could not produce; any line that is illustrative-only (not derivable) must be marked as such explicitly (e.g. `# illustrative` annotation or a prose caveat).
3. NO code changes, NO new tests, NO testdata regeneration. Docs only. If a snippet cannot be derived with confidence from print.go + goldens, mark it illustrative rather than adding code to verify it.
4. Cross-reference doc/tca.md §-numbers and decision IDs. The decision-ID tables (IR-01..IR-07, D-01..D-18) live in `.planning/phases/01-ir-core-pass-infrastructure-tooling/01-CONTEXT.md` — read it for exact ID meanings. Where doc/tca.md §5.1 notation predates the lowercase `memstore(` spelling (the follow-up flagged in 261018-bl0's summary), note this explicitly in doc/ir.md (call it a known drift; do NOT edit doc/tca.md — that sync is a separate deferred task).
</accuracy_discipline>

<tasks>

<task type="auto">
  <name>Task 1: Source-grounding read pass + write doc/ir.md sections 1-8 (organization through scheduling)</name>
  <files>doc/ir.md</files>
  <action>
Read the full source inventory listed in &lt;accuracy_discipline&gt; (catalog.go, kind.go, doc.go, node.go, all node_*.go, ssa.go, effect.go, poset.go, walk.go, dce.go, schedule.go, cfg.go, verify.go, print.go, mutate.go, types.go, internal/pass/scheduler.go, pkg/builder.go, pkg/chain.go) plus 01-CONTEXT.md for decision IDs. Then create doc/ir.md with this section skeleton (section numbers are guidance, content is mandatory):

1. **Overview &amp; how to read this doc** — what the IR is, its place between builder/frontend and passes/codegen, relationship to doc/tca.md (algebra spec) vs doc/ir.md (implementation-faithful IR spec), and a drift note: doc/tca.md §5.1 notation predates the lowercase `memstore(` spelling (follow-up from quick-261018-bl0; doc/tca.md sync deferred).
2. **File organization** — what each internal/ir file owns and WHY: node_*.go per-family layering (lambda/expr/var/memory/loop/dimvar/lir_memory/end), catalog.go (KindTier/Tier/HIR/LIR/AllKinds), kind.go (KindVersion 3 renumber, HIR before LIR), node.go (Base, payload records, stmtID pinning, effect-root registration), ssa.go, effect.go, poset.go, walk.go, verify.go, dce.go, schedule.go, cfg.go, print.go, mutate.go, types.go, doc.go. One short paragraph or bullet per file; explain the layering rationale (per quick-261018-bl0 task 1).
3. **The two-level design** — Level-1 sea-of-nodes: no CFG/statements during construction or between passes; deps = Input (data) edges + ImpactBy (effect) edges; the poset. Level-2: exactly one BasicBlock of Statements, produced only by Graph.Materialize() (default, correctness-preserving, pinned statement IDs per D-04 / Scheme X) or re-optimized by pass.Scheduler. Explain why two levels (passes mutate a sea; scheduling is an optimization decision deferred to a pass).
4. **Node catalog** — enumerate EVERY Kind from catalog.go, categorized HIR vs LIR, one line of semantics per kind: tier, family, operands/arity, payload, what it means. Include the v3 numbering note (generic Load/Store + LoadTile/StoreTile removed; SMEM/FRAG families with StorageClass; HIR GMEM family = MemLoad/MemStore/MemNew; LIR = MemLoadSMem/MemStoreSMem/MemNewSMem/FragLoad/FragStore/FragNew). Note `len(AllKinds()) == int(KindEnd)-1` invariant.
5. **SSA handle layer** — SSAVar/SSADef/SSAUse, intrusive Use records with next/prev links and back-pointers (D-16), use-def/def-use traversal, RAUW semantics. Reference IR-01.
6. **Effect model** — EffectOf verdicts, per-tensor effect chains via construction Env, the ImpactBy edge payload, restrict-alias rule (a tensor's effects chain in order; the effect-input side is the aliasing contract), the ONE end node: first NewReturn becomes Graph.end; NewReturnChained wires every mem-param final chain state via ImpactBy so the single return carries ALL observable effects (IR: return value(s) + effect chains). Env.FinalStates. Reference quick-261017-vdj/261018-bl0 decisions.
7. **DCE** — two-root-set liveness: sea roots = RoleTensor (memory) params + the end node; stores are NOT roots (they chain into memory params' effect chains); SMEM/FRAG state unobservable unless it feeds the end node; building-DCE seeds ALL params for the def-use verifier (the 261018-bl0 deviation — markReachable seeds end + all g.params). mark/sweep + fixpoint chain kill; Erase kills payload records too.
8. **Poset + scheduling** — DataDeps/EffectDeps/PosetDeps/SeaRoots read face (poset.go); default Materialize (registration-order roots, pinned vs monotonic statement IDs); pass.Scheduler (ID "schedule", produces StageConstructed): stable Kahn topo over the statement-ownership poset → LICM → topo; LICM whitelist {Const, Param, BinOp, UnaryOp, Select, Cast, Call} narrowed to candidates binding ≥1 operand; insertion point = before earliest consuming statement; cycle remainder fails closed; Remove/Append back-pointer invariant. Cross-reference D-04/D-05.

Keep each section tight and factual; do not editorialize. Cross-reference doc/tca.md §-numbers and IR-xx/D-xx IDs inline wherever a design decision originates there.
  </action>
  <verify>
    <automated>test -f doc/ir.md &amp;&amp; grep -c "^## " doc/ir.md | awk '$1>=8{exit 0}{exit 1}' &amp;&amp; grep -q "memstore(" doc/ir.md &amp;&amp; grep -q "ImpactBy" doc/ir.md &amp;&amp; grep -q "SeaRoots" doc/ir.md</automated>
  </verify>
  <done>doc/ir.md exists with sections 1-8 covering every must-have truth except the worked examples; every file in internal/ir is accounted for in the file-organization section; catalog section enumerates the same kinds as catalog.go; decision IDs (IR-01..IR-07, D-04, D-05, D-16) cited inline.</done>
</task>

<task type="auto">
  <name>Task 2: Worked examples section + full accuracy/consistency pass over doc/ir.md</name>
  <files>doc/ir.md</files>
  <action>
Add section 9 "Conceptual code → IR: worked examples" to doc/ir.md:

**9.1 The matmul_scale oracle (real golden, verbatim).** Quote `pkg/testdata/matmul_scale.ir` (8 lines, statement #8 + statement #9) VERBATIM in a code block. Show the conceptual kernel (from doc/syntax.md / the matmul_scale oracle: `output[m][k] = sum_n lhs[m][n] * rhs[n][k] * scale` style) side by side. Then a node-by-node / stage-by-stage walkthrough: the iter→map stage (lhs/rhs GMEM loads, lambda l*r), the reduce stage (init=0.0, lambda acc+now), the second map (v * scale), the memstore to %output[@m][@k], and statement #9's `return %output` — explaining how the store chains into %output's effect chain and the end node carries it via ImpactBy. Explain the `%0` pipe-local numbering (chain-local per doc/tca.md §2.6 — both reduce and scale-map inputs print %0) and statement #8/#9 pinned IDs (Scheme X). Optionally reference `internal/visualization/testdata/matmul_scale.dot` for the graph view.

**9.2 Teaching example: trivial scale kernel.** Conceptual: `out[i] = in[i] * 2.0` style. Derive its canonical IR by reasoning over the builder (b.Stmt().Iter(...).Map(...).Store(...)) + print.go rules. Every line must be derivable from printer rules; annotate the derivation briefly (why %0, why statement numbering).

**9.3 Teaching example: two-store chain (ImpactBy chaining).** A kernel storing to two tensors (or two stores to one tensor) whose effect order matters — show how the second store's effect-input (ImpactBy) points at the first store's chain head, and how the end node reaches both via the per-tensor final states. This example exists to make the effect chain VISIBLE in printed/stated IR terms since the printer renders stores by their own access rule (per 261018-bl0: statement #9 unchanged in shape because the store renders through its own access rule) — state the edges in prose/edge-list form if the text printer does not render them, and say so explicitly.

**9.4 Teaching example: LIR cache-placement sketch.** Show an HIR memory node lowered to MemLoadSMem/FragLoad with StorageClass — a sketch: annotate clearly which lines are illustrative-only (the LIR lowering passes are future phases; derive only what catalog.go + node_lir_memory.go + print.go can support, and mark everything else `# illustrative`).

Then run the full accuracy/consistency pass over the whole doc:
- Re-open catalog.go and diff the doc's kind list against it (every kind present, right tier, right spelling — lowercase canonical: `memstore(`, `memloadsmem(`, `fragstore(`).
- Re-open print.go and re-check every IR snippet in section 9 against chainOf/%N/operandPrec/statement-boundary rules.
- Check every doc/tca.md §-reference and IR-xx/D-xx ID resolves (IDs per 01-CONTEXT.md).
- Confirm the drift note about doc/tca.md §5.1 vs lowercase `memstore(` is present.
- Confirm no code/test/testdata files were touched: `git status --porcelain` shows only doc/ir.md (plus this plan directory's files).
</action>
  <verify>
    <automated>grep -q "statement #8" doc/ir.md &amp;&amp; grep -q "statement #9" doc/ir.md &amp;&amp; grep -q "illustrative" doc/ir.md &amp;&amp; grep -q "IR-01" doc/ir.md &amp;&amp; git status --porcelain -- '**/testdata/**' | wc -l | awk '$1==0{exit 0}{exit 1}' &amp;&amp; go build ./... &amp;&amp; go test ./internal/ir ./pkg 2>&1 | tail -1 | grep -q "ok"</automated>
  </verify>
  <done>doc/ir.md complete: sections 1-9; matmul_scale golden quoted verbatim and annotated; ≥3 teaching examples with derivations or explicit illustrative annotations; kind list byte-consistent with catalog.go; decision IDs cross-referenced; tca.md §5.1 drift note present; zero testdata drift; build + tests still green (untouched).</done>
</task>

</tasks>

<verification>
- `doc/ir.md` exists, comprehensive (sections 1-9), every claim traceable to source read this session.
- Worked example quotes `pkg/testdata/matmul_scale.ir` verbatim (byte-check by eye against the golden: 8 lines, statement #8 stage chain, statement #9 `return %output`).
- All teaching-example IR text either printer-derivable or explicitly annotated illustrative.
- `git status --porcelain -- '**/testdata/**'` empty; `go build ./...`, `go vet ./...`, `go test ./...` all green (nothing changed in code).
- Cross-references to doc/tca.md sections and IR-xx/D-xx IDs all resolve.
</verification>

<success_criteria>
- ONE new file: doc/ir.md. No other repo changes.
- Covers: file organization, two-level design, full HIR/LIR node catalog, SSA handle layer, effect/ImpactBy model + one-end-node, two-root-set DCE, poset, scheduling (Materialize + pass.Scheduler LICM).
- Multiple conceptual-code → IR mappings, anchored on the real matmul_scale golden, annotated node-by-node.
- Explicit note where doc/tca.md §5.1 notation predates lowercase memstore(.
- Docs interlock with doc/tca.md via decision/requirement ID cross-references.
</success_criteria>

<output>
After completion, create `.planning/quick/261018-ked-write-comprehensive-ir-documentation-doc/261018-ked-SUMMARY.md`
</output>
