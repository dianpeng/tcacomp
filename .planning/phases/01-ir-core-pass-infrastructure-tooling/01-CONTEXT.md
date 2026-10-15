# Phase 1: IR Core, Pass Infrastructure & Tooling - Context

**Gathered:** 2026-10-15
**Status:** Ready for planning

<domain>
## Phase Boundary

A verified, debuggable Go IR and pass-pipeline foundation with a programmatic graph-builder API — the substrate every later phase hangs off. Scope is exactly the four roadmap plans: (01-01) SSA IR core with use-def/def-use chains and choke-point mutation, (01-02) legality windows + verifier framework + pass manager + flags/analysis skeletons, (01-03) diagnostics/log/visualization/canonical-printer/golden-test tooling, (01-04) `pkg/` builder API + `matmul_scale` oracle graph + determinism byte-compare. No codegen, no runtime, no frontend parser.
</domain>

<decisions>
## Implementation Decisions

### IR Printer Format
- **D-01:** The canonical textual printer emits **doc-faithful notation** — statements print as in `doc/tca.md` §5.1 (`iter(@m,@k,@n) -> map(lhs[@m][@n], rhs[@n][@k], lambda(l, r): l * r) as (@m, @k, @n) | reduce(...) | ...`), with SSA values numbered `%N` by deterministic definition order.
- **D-02:** Lambdas / scalar expressions are **canonically re-rendered** from the IR with normalized parens/spacing. Two structurally identical graphs always print byte-identically — this property is what golden tests and the Phase 5 parser-oracle comparison rely on.
- **D-03:** **Stable minimal default** output: no locs in the printed text by default; verbose modes (pass-origin annotations, source spans) are available behind `--dump-after`-style flags. Golden tests pin the minimal default so diffs show only real IR changes.
- **D-04:** **Stable statement IDs**: `statement #N` persists across passes for the same logical statement, so pass-to-pass diffs track a statement's evolution.
- **D-05:** `.dot` emission **mirrors the text printer**: same SSA numbering, statement structure rendered as clusters, explicit use-def edges, fully deterministic node/edge order.
- **D-06:** Phase 1 is **print-only** — no IR-text parser. "Round-trip" means byte-identical printing across repeated runs/processes; golden tests compare strings; graphs are always constructed via the builder API.

### Builder API Ergonomics (`pkg/`)
- **D-07:** **Fluent pipe-chains** mirror the doc's `|`-separated stages: `b.Stmt().Iter(m,k,n).Map(lhs, rhs, func(l, r) { return l * r }).As(m, k, n).Reduce(init, fn).Map(...).Store(out)`. Phase 5 parser lowering is a near-mechanical mapping from syntax to builder calls.
- **D-08:** Dims/axes/tensors are **first-class typed Go handles** from the builder (`b.Dim("m", 128)`, `b.Tensor("lhs", f32, m, n)`) — no string lookups; the Go compiler catches shape mismatches at call sites.
- **D-09:** Lambda bodies use a small **expr-builder DSL**: `b.Mul(l, r)`, `b.Add(acc, v)` return typed expr nodes composed bottom-up; lambdas are Go funcs over expr builders. Mirrors `doc/syntax.md` operators 1:1 and re-renders canonically. (String-parsed lambdas rejected — they would embed a mini-frontend before the real one.)
- **D-10:** The builder performs **eager cheap invariant checks** at each call (defined operands, arity, dim counts) and returns errors; the full verifier still runs on the completed graph through the normal pass-manager path. Fail fast at the call site, but one authoritative verification path.

### Golden-Test Harness
- **D-11:** Golden expectations live in **side-by-side files under `testdata/`** (Go convention), e.g. `testdata/matmul_scale.ir`.
- **D-12:** Update flow is the standard **`-update` flag** (`go test ./internal/ir/ -update`); CI runs without the flag so stale goldens fail loudly.
- **D-13:** Comparison is **exact byte compare** — feasible and strongest because the printer is canonical and deterministic (D-02).
- **D-14:** **One generic golden harness** serves all output kinds (IR text, `.dot`, diagnostics) with the same `-update` flow, diff output, and `testdata/` layout.

### Node Representation & Mutation API
- **D-15:** IR nodes are **Go interface families** (abstract / loop / tile-op / collective / scalar-expr per IR-03) with a shared embedded base struct carrying ID, Loc, and use-list links — the `go/ast` shape applied to SSA. Passes type-switch over families.
- **D-16:** Use-def/def-use chains are **LLVM-style intrusive `Use` records** (each operand use is a `Use` object with next/prev links and back-pointers to def and user; RAUW splices lists in O(uses)) — the llir/llvm model named in CLAUDE.md as the definitive pure-Go reference.
- **D-17:** **Required mutation context**: every choke-point mutation (SetDef/AddUse/RemoveUse/RAUW/Erase) takes a pass-context argument and the IR records last-touched-by automatically — PASS-06 diagnostics (naming the last-touched-by pass) are structural, not advisory.
- **D-18:** Types are **interned, pointer-compared values** (element types, shapes referencing dim objects): identity == equality, map keys work, deterministic emission ordering needs no tie-breaking.

### Claude's Discretion
- Legality-window declaration mechanism (per-pass declared windows vs central table) — flagged MEDIUM-confidence in ROADMAP.md; the research phase settles the API shape within the MLIR-style constraint of IR-07.
- Pass-registration style (declarative registry vs explicit pipeline lists), diagnostics error-type plumbing, `.dot` cluster styling details, `internal/log` facade shape over `slog`.
- Analysis-cache internals beyond PASS-03's invalidate-all default + preservation whitelist contract.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Authoritative design docs
- `doc/tca.md` — IR algebra, §5.1 graph-construction/SSA model, pass pipeline, node families; §2 notation the printer must reproduce
- `doc/design.md` — module requirements, verification-per-pass requirement, stdlib-only constraint
- `doc/syntax.md` — v0 frontend spec; §11.1 builder-first ordering (FE-01); operator surface the expr-builder DSL mirrors

### Planning artifacts
- `.planning/REQUIREMENTS.md` — IR-01..IR-07, PASS-01..PASS-06, TOOL-01..TOOL-06, FE-01, VER-08 (this phase's requirement set)
- `.planning/ROADMAP.md` §Phase 1 — success criteria and the four plans 01-01..01-04; research flag on legality-window API shape
- `CLAUDE.md` — (a) lexer/parser reference decisions, (b) IR/SSA data-structure decisions (mutable pointer graph, intrusive use-lists, stable IDs, verifier-after-every-pass), stack constraints

### External reference scaffold (read-only, do not copy)
- `/home/ps/Workspace/cuda/project/scalfold` — reference DSL for runtime philosophy (storage tiers, generators); Phase 1 relevance is limited to naming/tier vocabulary consistency

</canonical_refs>

<code_context>
## Existing Code Insights

Greenfield repository — no source exists yet. Only docs and `.planning/`. Every Phase 1 decision sets precedent; package layout is prescribed by PROJECT.md: `runtime/` (not in this phase), `pkg/` (public builder API), `internal/` (`ir`, `pass`, `opt`, `analysis`, `codegen` later, `diagnostic`, `visualization`, `log`, `flags`, `syntax` later), `cmd/` (not in this phase).

### Reusable Assets
- None (greenfield).

### Established Patterns
- None in code. Patterns come from the docs and CLAUDE.md decisions (LLVM-style use-lists, go/ast-style node interfaces, go/scanner-structure lexer later).

### Integration Points
- `pkg/` builder API is the future target of the Phase 5 parser (`internal/syntax`) — decisions D-07..D-10 shape that contract.
- The golden harness (D-11..D-14) is reused by every later phase's golden IR tests (Phases 3–5 success criteria depend on it).
- The canonical printer (D-01..D-06) is the comparison oracle for the Phase 5 parser-identical-to-oracle gate.
</code_context>

<specifics>
## Specific Ideas

- Printer output should read like `doc/tca.md` — debugging a dump means reading the doc's own notation, not translating from LLVM-isms. (User picked doc-faithful over familiar LLVM style explicitly.)
- The builder chain should look like the pipe syntax a kernel author would write — the parser's job in Phase 5 should feel like a transcription, not a translation.
- Rejected during discussion: string-parsed lambdas in the builder (would embed a mini-frontend with its own error surface before the real frontend exists); inline CHECK comments (Go files balloon); pattern-subset matching (weak — passes while IR changes unnoticed); IR-text parse-back in Phase 1 (second grammar before the real one).
</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

### Reviewed Todos (not folded)
None.
</deferred>

---

*Phase: 01-ir-core-pass-infrastructure-tooling*
*Context gathered: 2026-10-15*
