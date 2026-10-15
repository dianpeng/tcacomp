# Phase 1: IR Core, Pass Infrastructure & Tooling - Research

**Researched:** 2026-10-15
**Domain:** Compiler IR infrastructure in pure Go (SSA graph core, pass manager, diagnostics/printing/golden-test tooling)
**Confidence:** HIGH (all stack facts verified against the local Go 1.26.2 toolchain and official MLIR/LLVM/Graphviz documentation fetched this session); MEDIUM on the legality-window API shape (explicit research flag — no off-the-shelf precedent, settled below as applied design judgment)

## Summary

Phase 1 is greenfield, stdlib-only Go infrastructure. The correct substrate for every design question here already exists and was verified this session: LLVM's `Value`/`User`/`Use` intrusive use-list model (documented in the LLVM Programmer's Manual, confirmed verbatim including `replaceAllUsesWith` "traverses the use list"), MLIR's pass-instrumentation hook set (`runBeforePass`/`runAfterPass`/`runBeforePipeline`/`runAfterAnalysis`/...), MLIR's invalidate-all-by-default analysis preservation contract, MLIR's `ConversionTarget` legality-action model (`Legal`/`Dynamic(callback)`/`Illegal` + unknown handling), LLVM `opt`'s `-verify-each` / `-time-passes` / `-print-after` flag semantics, and Go's own stdlib golden-file pattern (`gofmt_test.go` package-level `var update = flag.Bool("update", ...)`, and `go help packages` confirming `testdata/` directories are ignored by the go tool). Everything the phase needs can be built directly on these verified patterns with zero external modules — which is a hard project constraint anyway.

The two ROADMAP research flags resolve as follows. (1) **Legality-window API**: adopt a **stage-centered legality table** — a central `map[Stage]LegalityWindow` where each window maps node `Kind` → `Legal` | `Illegal` | `Dynamic(predicate)`, each pass declares the stage it produces (`Produces() Stage`), and the pass manager checks the whole graph against that window after every pass run. Unknown `Kind`s are rejected (satisfies IR-03). This is the ConversionTarget model narrowed from "per-conversion-pass target" to "per-pipeline-stage window", which is what IR-07's wording ("per pipeline stage, checked by the pass manager after every pass") actually asks for, and it avoids duplicating window declarations in every pass. (2) **Mutation-API surface**: LLVM-style intrusive `Use` records (one per operand slot, next/prev links in the def's use list, back-pointers to def and user), with a choke-point `MutContext` carrying the pass name so `LastTouchedBy` is stamped structurally on every mutation (D-17), plus explicit no-op/self-RAUW and erase-with-uses guards.

**Primary recommendation:** Build `internal/ir` as a direct Go port of LLVM's use-list discipline (not a redesign), put legality windows and instrumentation in `internal/pass` as MLIR-derived but drastically simplified single-tier structures, and implement the golden harness exactly as Go's own stdlib does (`flag.Bool("update")` + `testdata/` + byte compare) — then pin the `matmul_scale` oracle print from `doc/tca.md` §5.1 as the first golden file.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**IR Printer Format**
- **D-01:** The canonical textual printer emits **doc-faithful notation** — statements print as in `doc/tca.md` §5.1 (`iter(@m,@k,@n) -> map(lhs[@m][@n], rhs[@n][@k], lambda(l, r): l * r) as (@m, @k, @n) | reduce(...) | ...`), with SSA values numbered `%N` by deterministic definition order.
- **D-02:** Lambdas / scalar expressions are **canonically re-rendered** from the IR with normalized parens/spacing. Two structurally identical graphs always print byte-identically — this property is what golden tests and the Phase 5 parser-oracle comparison rely on.
- **D-03:** **Stable minimal default** output: no locs in the printed text by default; verbose modes (pass-origin annotations, source spans) are available behind `--dump-after`-style flags. Golden tests pin the minimal default so diffs show only real IR changes.
- **D-04:** **Stable statement IDs**: `statement #N` persists across passes for the same logical statement, so pass-to-pass diffs track a statement's evolution.
- **D-05:** `.dot` emission **mirrors the text printer**: same SSA numbering, statement structure rendered as clusters, explicit use-def edges, fully deterministic node/edge order.
- **D-06:** Phase 1 is **print-only** — no IR-text parser. "Round-trip" means byte-identical printing across repeated runs/processes; golden tests compare strings; graphs are always constructed via the builder API.

**Builder API Ergonomics (`pkg/`)**
- **D-07:** **Fluent pipe-chains** mirror the doc's `|`-separated stages: `b.Stmt().Iter(m,k,n).Map(lhs, rhs, func(l, r) { return l * r }).As(m, k, n).Reduce(init, fn).Map(...).Store(out)`. Phase 5 parser lowering is a near-mechanical mapping from syntax to builder calls.
- **D-08:** Dims/axes/tensors are **first-class typed Go handles** from the builder (`b.Dim("m", 128)`, `b.Tensor("lhs", f32, m, n)`) — no string lookups; the Go compiler catches shape mismatches at call sites.
- **D-09:** Lambda bodies use a small **expr-builder DSL**: `b.Mul(l, r)`, `b.Add(acc, v)` return typed expr nodes composed bottom-up; lambdas are Go funcs over expr builders. Mirrors `doc/syntax.md` operators 1:1 and re-renders canonically. (String-parsed lambdas rejected — they would embed a mini-frontend before the real one.)
- **D-10:** The builder performs **eager cheap invariant checks** at each call (defined operands, arity, dim counts) and returns errors; the full verifier still runs on the completed graph through the normal pass-manager path. Fail fast at the call site, but one authoritative verification path.

**Golden-Test Harness**
- **D-11:** Golden expectations live in **side-by-side files under `testdata/`** (Go convention), e.g. `testdata/matmul_scale.ir`.
- **D-12:** Update flow is the standard **`-update` flag** (`go test ./internal/ir/ -update`); CI runs without the flag so stale goldens fail loudly.
- **D-13:** Comparison is **exact byte compare** — feasible and strongest because the printer is canonical and deterministic (D-02).
- **D-14:** **One generic golden harness** serves all output kinds (IR text, `.dot`, diagnostics) with the same `-update` flow, diff output, and `testdata/` layout.

**Node Representation & Mutation API**
- **D-15:** IR nodes are **Go interface families** (abstract / loop / tile-op / collective / scalar-expr per IR-03) with a shared embedded base struct carrying ID, Loc, and use-list links — the `go/ast` shape applied to SSA. Passes type-switch over families.
- **D-16:** Use-def/def-use chains are **LLVM-style intrusive `Use` records** (each operand use is a `Use` object with next/prev links and back-pointers to def and user; RAUW splices lists in O(uses)) — the llir/llvm model named in CLAUDE.md as the definitive pure-Go reference.
- **D-17:** **Required mutation context**: every choke-point mutation (SetDef/AddUse/RemoveUse/RAUW/Erase) takes a pass-context argument and the IR records last-touched-by automatically — PASS-06 diagnostics (naming the last-touched-by pass) are structural, not advisory.
- **D-18:** Types are **interned, pointer-compared values** (element types, shapes referencing dim objects): identity == equality, map keys work, deterministic emission ordering needs no tie-breaking.

### Claude's Discretion
- Legality-window declaration mechanism (per-pass declared windows vs central table) — flagged MEDIUM-confidence in ROADMAP.md; the research phase settles the API shape within the MLIR-style constraint of IR-07. **Settled in this research: stage-centered central table (see Architecture Patterns → Legality Windows).**
- Pass-registration style (declarative registry vs explicit pipeline lists), diagnostics error-type plumbing, `.dot` cluster styling details, `internal/log` facade shape over `slog`.
- Analysis-cache internals beyond PASS-03's invalidate-all default + preservation whitelist contract.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| IR-01 | SSA graph core: SSADef/SSAUse/SSAVar, use-def + def-use chains, trace uses/defs, replace-def updates refs, delete | LLVM use-list model verified (ProgrammersManual: `use_begin`/`replaceAllUsesWith` semantics); llir/llvm named as pure-Go design reference in CLAUDE.md; mapping to Go sketched in §Architecture Patterns |
| IR-02 | One basic block per kernel; statement RHS is SSAUse or Value | tca.md §5.1 model (verified in doc read); Graph→BB→Statement structure sketched below |
| IR-03 | Versioned stable Kind enum covering all node families; PM rejects unknown kinds | MLIR "unknown op" handling verified (DialectConversion doc); forward-compat pattern in §Architecture Patterns |
| IR-04 | All mutation via choke-point API (SetDef/AddUse/RemoveUse/RAUW/Erase) | LLVM/llir mutation set verified; choke-point + MutContext design below; enforcement = analyses receive read-only view (no MutContext) |
| IR-05 | Stable integer node IDs; deterministic iteration/emission order | Verified Go pitfall: `maps.Keys` order explicitly unspecified (go doc); ordering rules in §Pitfalls + printer design |
| IR-06 | Minimal `Loc{frontend span, pass origin}` on every node | Pass-origin stamped structurally via D-17 MutContext; span zero-value in Phase 1 (no frontend) |
| IR-07 | Node-family legality windows per pipeline stage, PM-checked after every pass | MLIR ConversionTarget model verified; stage-centered adaptation designed below (research flag #1, MEDIUM confidence) |
| PASS-01 | PM runs ordered pipelines with enforced preconditions | `Requires()` precondition mechanism designed below; failure aborts pipeline (MLIR `signalPassFailure` semantics verified) |
| PASS-02 | Instrumentation structural: verify-after-every-pass, IR dumps, pass timing as PM callbacks | MLIR `PassInstrumentation` hook set + standard instrumentations verified verbatim; mapping below |
| PASS-03 | Conservative analysis cache: invalidate-all default, preservation whitelist | MLIR preservation contract verified verbatim (invalidate-by-default, `markAllAnalysesPreserved`/`markAnalysesPreserved`) |
| PASS-04 | Global flags package = single source of truth | `internal/flags` typed-struct design below; stdlib `flag` package (verified available) |
| PASS-05 | Analysis package skeleton, read-only / mutation-free rule | Enforcement design: analyses get read-only interface, never MutContext (below) |
| PASS-06 | Transformation passes declare before/after invariants; verifier checks with rich diagnostics (node ID, last-touched-by, invariant) | Invariant-registry design below; D-17 makes last-touched-by structural; corrupted-chain white-box test strategy below |
| TOOL-01 | Diagnostic package: severity model, source spans, structured context | Severity enum + Bag design below (3 severities per requirement text) |
| TOOL-02 | Log infra for compiler-scale debug output (per-pass, per-node) | `log/slog` verified (`slog.New`, `NewTextHandler`, `slog.Group` via go doc); facade pattern per CLAUDE.md |
| TOOL-03 | Graphviz `.dot` emission for any pipeline stage | DOT language spec verified (graphviz.org): digraph grammar, `subgraph cluster*` convention; emit-only, no dependency |
| TOOL-04 | Canonical deterministic textual IR printer | Determinism rules + `%N` numbering design below; `strings.Builder` + `fmt.Fprintf` per CLAUDE.md stack table |
| TOOL-05 | Golden IR regression tests with update flow | Go stdlib golden pattern verified in local toolchain source (`cmd/gofmt/gofmt_test.go:18`); `testdata/` convention verified (`go help packages`) |
| TOOL-06 | Pass dump/trace flags: `--dump-after`, `--dump-graphviz`, `--pass-timing`, `--verify-each=0` | LLVM `opt -verify-each`/`-time-passes` semantics verified (opt CommandGuide); flag plumbing via `internal/flags` |
| FE-01 | `pkg/` elaborator/graph-builder API producing parser-target-identical graphs, available from Phase 1 as oracle | Fluent typestate chain design below (D-07..D-10); oracle = tca.md §5.1 `matmul_scale` text |
| VER-08 | Determinism byte-compare: fixed compiled kernel → bitwise-identical outputs across runs | Test design below: in-process double-build byte compare + committed golden as the cross-process anchor |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

Directives the planner must not contradict (extracted from the project CLAUDE.md and `doc/design.md`, both checked into the repo):

1. **Stdlib-only Go compiler**: pure Go, standard library only, no cgo, no external modules. Enforcement check: `go list -m all | wc -l` == 1.
2. **Single-tier IR**: no separate high/mid/low IRs; lowering introduces new node kinds into the same suite (design.md). Node kinds need a stable, versioned `Kind` enumeration from day one; the pass manager walks kinds it knows and rejects kinds it doesn't.
3. **Mutable pointer-graph IR with intrusive use-lists** (CLAUDE.md decision (b)): this is the sanctioned exception to the user's global immutability coding rule — everything else (pass options, diagnostics, analysis results, flags) should still follow immutable patterns.
4. **Verifier pass runs after every transformation pass** (design.md); IR verifier must be cheap and callable between passes (Go compiler's `cmd/compile/internal/ssa` habit of a `check` pass, per CLAUDE.md).
5. **Unique integer IDs per IR node**, stable across mutations (for pass debugging, `.dot` dumps, verifier messages).
6. **Minimal necessary tests only** — explicitly no excessive test suites (design.md). *Tension noted:* the user's global rules demand 80% coverage and TDD; the project's checked-in CLAUDE.md overrides toward "only fundamental tests" for this repo. Planner should write the fundamental tests first (IR invariants, golden round-trip, determinism, legality/precondition rejection) and not pursue coverage theater.
7. **Package layout**: `runtime/` (not this phase), `pkg/` (public builder API), `internal/` (`ir`, `pass`, `opt`, `analysis`, `codegen` later, `diagnostic`, `visualization`, `log`, `flags`, `syntax` later), `cmd/` (not this phase). design.md also lists `util/`, `opt/`, `pass/` under `internal/`.
8. **Ambiguity protocol** (design.md): "For any uncertain/ambiguous problem, stop implementation and ask the user first, with comprehensive details."
9. **Small files** (user global coding-style rule, applies): 200–400 lines typical, 800 max — plan `internal/ir` as many small files (`node.go`, `use.go`, `mutate.go`, `verify.go`, `print.go`, `kind.go`, ...), not one monolith.
10. **Race-freedom-by-construction is the correctness model** (CLAUDE.md): the verifier checking structural invariants is the enforcement backbone — Phase 1's verifier framework is where Phases 3–4 hang their invariant checks.
11. **Codegen is read-only on the IR** (CG-01, later phase) — reinforces the read-only-view pattern for non-transforming consumers.
12. Reference scaffold `/home/ps/Workspace/cuda/project/scalfold` (read-only): naming/tier vocabulary consistency only for Phase 1.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Go toolchain | 1.26.2 (verified: `go version` on this machine) | Everything — the compiler is pure Go stdlib | Hard project constraint; toolchain already installed and matches CLAUDE.md pin |
| `strings`, `bytes` | stdlib (1.26.2) | Canonical printer / `.dot` emission buffers | CLAUDE.md stack table prescribes `strings.Builder` for emission |
| `fmt` | stdlib | Diagnostics, `.dot` text, pass-timing report | Prescribed by CLAUDE.md stack table |
| `flag` | stdlib | `--dump-after`/`--dump-graphviz`/`--pass-timing`/`--verify-each` and the golden `-update` flag | Stdlib `flag` is austere but sufficient (CLAUDE.md); golden `-update` pattern verified in stdlib's own `gofmt_test.go` |
| `log/slog` | stdlib (since 1.21) | Backend behind the `internal/log` facade | CLAUDE.md prescribes `slog` behind a facade; signatures verified via `go doc` |
| `slices`, `maps`, `sort` | stdlib (1.26.2) | Deterministic ordering, sorted emission, interning tables | CLAUDE.md stack table lists them for exactly this |
| `hash/fnv` | stdlib | Interning dedup keys if needed | CLAUDE.md stack table |
| `strconv` | stdlib | Canonical float/int constant rendering in the printer | `FormatFloat(f, fmt, prec, bitSize)` verified via `go doc` |
| `testing` | stdlib | All tests (golden harness, IR unit tests, determinism) | Stdlib-only by definition |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `errors`, `fmt.Errorf` | stdlib | Error plumbing in diagnostics | Everywhere errors cross package boundaries |
| `os`, `os/exec` | stdlib | Golden file I/O (`os.ReadFile`/`os.WriteFile`); `os/exec` only in Phase 2's testutil, NOT Phase 1 | Golden harness |
| `time` | stdlib | Pass-timing instrumentation durations | Never in emitted/golden output (see Pitfalls) |
| `go/scanner`, `go/token`, `go/ast` | stdlib (local source at `$GOROOT/src/go/...`) | **Reference only — never imported** (would drag Go-language semantics). Interface-family + marker-method shape verified in `$GOROOT/src/go/ast/ast.go` (`Node{Pos,End}`, `Expr` embeds `Node` + `exprNode()`) | Architecture reference for D-15 |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `slog` behind `internal/log` facade | Hand-rolled log levels | CLAUDE.md keeps hand-rolling as the fallback if slog's API fights compiler-style verbose-pass logging; facade makes swapping cheap. `slog.NewTextHandler` + `slog.Group` verified sufficient for per-pass/per-node structured records |
| Hand-rolled golden harness (~60 lines, per gofmt pattern) | External golden libs (e.g. `gotest.tools/golden`, `sebdah/goldie`) | All external modules — banned. The stdlib pattern is 60 lines and is what the Go project itself ships |
| `flag` | `pflag`-style POSIX flags | External. Stdlib `flag` fine for a v0 compiler CLI surface (Phase 5) |

**Installation:** none. Zero external Go modules by design; enforcement: `go list -m all | wc -l` == 1 [VERIFIED: CLAUDE.md installation section; `go version go1.26.2 linux/amd64` verified on machine].

**Version verification:** Go 1.26.2 confirmed installed (matches CLAUDE.md pin). All stdlib APIs cited below were verified against this exact toolchain via `go doc` or by reading `$GOROOT/src` — not against training memory.

## Architecture Patterns

### Recommended Package Structure (Phase 1 slice)

```
tcacomp/
├── go.mod                       # module tcacomp, go 1.26.2, zero requires
├── pkg/                         # D-07..D-10 public builder API (plan 01-04)
│   └── builder.go ...           # fluent chain, typed handles, expr DSL
└── internal/
    ├── ir/                      # plan 01-01: node.go, use.go, mutate.go,
    │                            #   kind.go, types.go, verify.go, print.go, dot.go
    ├── pass/                    # plan 01-02: manager.go, instrument.go,
    │                            #   legality.go, cache.go, context.go
    ├── analysis/                # plan 01-02: read-only skeleton (PASS-05)
    ├── flags/                   # plan 01-02: global compiler constants (PASS-04)
    ├── diagnostic/              # plan 01-03: severity, Bag, Loc types (TOOL-01)
    ├── log/                     # plan 01-03: facade over slog (TOOL-02)
    ├── visualization/           # plan 01-03: .dot emission (TOOL-03) — or ir/dot.go
    └── golden/                  # plan 01-03: generic golden harness (TOOL-05, D-14)
```

(internal/visualization vs ir/dot.go is a planner call; CLAUDE.md lists `internal/visualization` as the IR-visualization package, prefer that.)

### Pattern 1: IR Node Model — interface families + embedded base (D-15)

**What:** Every node struct embeds `*Base` (ID, Kind, Loc, use-list head, `LastTouchedBy`); family interfaces (`AbstractNode`, `LoopNode`, `TileOpNode`, `CollectiveNode`, `ScalarExprNode`) add marker methods; passes type-switch over families. This is exactly `go/ast`'s shape (`Node` interface; `Expr`/`Stmt` families; marker methods), verified in `$GOROOT/src/go/ast/ast.go`.

**When to use:** everywhere in `internal/ir`. The single-tier rule means loop/tile-op/collective kinds join the same node suite when Phases 3–4 lower them in — the family interfaces exist from day one even though Phase 1 only constructs abstract-family + scalar-expr nodes.

**Mapping note for IR-01/IR-02** (SSADef/SSAUse/SSAVar): the LLVM mapping is the cleanest substrate [CITED: llvm.org ProgrammersManual, verified this session]: there is *no distinction between an SSA variable and the operation that produces it* — the value-producing node IS the SSA value, the `Use` record is SSAUse, and the def-side object with its use-list is the SSAVar shared cell. Statements (IR-02) are ordered chains of value nodes in the single BB; a statement's RHS is either a `Use` (pipe-bound `%N` reference) or a value-producing node. Exact SSADef/SSAVar struct split is a planner call within this mapping [ASSUMED risk: low — design.md's "shared underlying pointer" is satisfied by `*Base` identity].

### Pattern 2: Intrusive Use Records + Choke-Point Mutation (D-16, D-17, IR-04) — research flag #2

**What:** One `Use` struct per operand slot, linked into the def's use list (next/prev), with back-pointers to def and user. All mutation flows through five choke-point functions taking a `*MutContext`:

```go
// Source: LLVM Programmer's Manual use-list section (verified) + llir/llvm
// design reference (named in CLAUDE.md), adapted to Go.

type Use struct {
    Val  Value      // the def this operand currently points to
    User Node       // back-pointer to the owning node (+ slot index)
    prev, next *Use // intrusive links in Val's use list
}

// Choke-point API — the ONLY sanctioned mutation path (IR-04).
// Every function stamps node.LastTouchedBy = ctx.Pass (D-17, structural PASS-06).
func (ctx *MutContext) SetDef(u *Use, newVal Value)            // redirect one operand
func (ctx *MutContext) ReplaceAllUsesWith(old, newVal Value)   // splice whole list, O(uses)
func (ctx *MutContext) Erase(n Node)                           // detach all operand uses,
                                                               // unlink from BB, tombstone ID
```

**When to use:** every pass mutation. Verified semantics from LLVM: `replaceAllUsesWith` "traverses the use list of a Value changing all Users" — splice, not re-point-and-rebuild.

**Guards the implementation must include** (from the use-list discipline):
- RAUW(old, old) is a no-op; never unlink the record you're about to visit — snapshot `next` before splicing (RAUW-during-iteration corrupts the walk otherwise).
- `Erase` on a node that still has users must refuse loudly (or poison + let the verifier catch it) — a def erased while `Use.Val` still points at it is a use-after-free waiting for Phases 3–4's more complex passes.
- Iteration APIs return *snapshots* or document splice-during-iteration semantics explicitly.

**Builder exemption:** construction (plan 01-04) does NOT go through `MutContext`; it performs D-10 eager checks and then runs the completed graph through the normal verifier pass. One authoritative verification path, fail-fast at the call site.

### Pattern 3: Pass Manager — flat pipeline + instrumentation callbacks (PASS-01, PASS-02)

**What:** Single-tier IR means a *flat* pipeline — MLIR's nested `OpPassManager` machinery does not apply (verified in the PassManagement doc) and should not be ported. Core:

```go
// Source: MLIR PassInfrastructure + PassInstrumentation docs (verified),
// narrowed to a flat single-BB world.

type Pass interface {
    ID() string            // stable name, e.g. "reduce-lower" (pipeline syntax + preconditions)
    Requires() []string    // PASS-01: pass IDs that must appear earlier in the pipeline
    Produces() Stage       // IR-07: legality window checked after this pass runs
    Preserved() []string   // PASS-03: analysis names kept valid; empty = invalidate all
    Run(*Context, *ir.Graph) error
}

type Instrumentation interface {   // PASS-02: PM callbacks, never inside passes
    BeforePass(Pass, *ir.Graph)
    AfterPass(Pass, *ir.Graph, err error)  // failure path carries the error
}
```

**Verified MLIR hook set to mirror** (from the PassInstrumentation doc): `runBeforePipeline` / `runAfterPipeline` / `runBeforePass` / `runAfterPass` / `runAfterPassFailed` / `runBeforeAnalysis` / `runAfterAnalysis`; instrumentations registered on the PM run "in a stack like fashion" (last `runBefore*` = first matching `runAfter*`). For this IR, Before/AfterPass + Before/AfterPipeline are sufficient; analysis hooks matter once the analysis cache grows in Phase 3+.

**Standard instrumentations** (each gated by `internal/flags`, each a PM callback — never code inside passes):
1. **VerifyEach** — runs the verifier after every pass. Disabled by `--verify-each=0`. Precedent verified: LLVM `opt -verify-each` "causes opt to add a verify pass after every pass". Run it as an AfterPass instrumentation, not a real pipeline pass (PASS-02's "not inside passes").
2. **IRDump** — `--dump-after=<pass-ids>|all` prints the canonical IR after the named passes; `--dump-graphviz` routes to `.dot`. Precedent: `-mlir-print-ir-after/-after-all` (verified); MLIR's `-after-change` variant needs an IR hash — note this exists but recommend skipping it in Phase 1 (D-03's minimal default + goldens make it low value).
3. **PassTiming** — `--pass-timing` accumulates `time.Since` per pass and prints a list report on pipeline end. Precedent verified: `-time-passes` "record the amount of time needed for each pass".

**Pass failure semantics** (verified from MLIR): a pass returning error aborts the whole pipeline run; the PM returns the error and the failing pass's name goes into the diagnostic.

### Pattern 4: Legality Windows — stage-centered table (IR-07, IR-03) — research flag #1, settled

**What:** Adapt MLIR's `ConversionTarget` legality-action model (verified: per-op `Legal` / `DynamicallyLegal(callback)` / `Illegal`; unknown ops handled separately; "Legal ... any combination of attributes, operands, types ... valid") from "per-conversion-pass target" to "per-pipeline-stage window":

```go
// Source: MLIR DialectConversion ConversionTarget model (verified), adapted.

type Stage int
const (   // mirrors the PM-enforced order in ROADMAP.md "Pipeline Staging"
    StageConstructed Stage = iota  // builder output: abstract + scalar-expr families
    StageLinearized                // effect-linearize done: no pipes
    StageParallelized              // annotations only
    StageIterLowered               // loop family appears; no raw iter nodes
    StageReduceLowered
    StageCacheLowered
    StageFinal                     // codegen-ready
)

type LegalityAction int
const ( Legal LegalityAction = iota; Illegal; Dynamic )

type LegalityWindow struct {
    Actions    map[ir.Kind]LegalityAction
    Predicates map[ir.Kind]func(ir.Node) error // consulted for Dynamic kinds
    // Unknown kinds: always rejected (IR-03) — the zero-value policy, matching
    // MLIR full-conversion mode ("only successful if all operations are
    // properly legalized to the given conversion target", verified).
}

// Central table (one place, not per-pass): internal/pass/legality.go
func WindowFor(s Stage) *LegalityWindow
```

The PM checks: after pass P runs with `P.Produces() == S`, walk every node; each node's `Kind` must be `Legal` in `WindowFor(S)` (or pass its `Dynamic` predicate). Failure → rich diagnostic: pass name, node ID, kind, stage, predicate message.

**Why this over per-pass declared windows** (the discretion alternative): IR-07's subject is the *pipeline stage*, not the pass; a stage outlives any single pass (e.g. `StageIterLowered` must hold from iter-lower through reduce-lower, cache-alloc, cache-lower); per-pass windows would force every pass to re-declare what the stage already knows and drift apart. Per-pass `Produces() Stage` (one word) is the only per-pass burden. Dynamic predicates cover family-level conditions that matter later (e.g. "after iter-lower: every `reduce` carries a `<@r>` annotation referencing a live loop var").

**Phase 1 reality check:** only `StageConstructed`'s window is exercised in Phase 1 (abstract + scalar-expr families all legal; no lowering passes exist yet). The table, `Stage` enum, and PM check must land now (IR-07 is a Phase 1 requirement); windows for later stages get *rows* as those phases land. [Confidence: MEDIUM — applied design judgment, explicitly flagged in ROADMAP as having no off-the-shelf precedent. The MLIR precedent underneath it is verified; the adaptation is the judgment call.]

### Pattern 5: Verifier Framework + Invariant Registry (PASS-06)

**What:** A `verify` framework inside `internal/ir` (or `internal/pass`): a registry of named invariant checkers, each `func(*Graph) []Violation`, where `Violation{NodeID, Invariant string, Msg string}` + the node's `LastTouchedBy` (free — D-17 stamps it). Transformation passes in later phases declare `Before() []string` / `After() []string` invariant names; the PM runs the corresponding checkers around the pass (via the VerifyEach instrumentation). Phase 1 ships the chain-consistency checkers (the IR-01 substrate): every `Use.Val` non-nil and points to a live def; every def's use-list is exactly the set of `Use`s pointing at it (both directions — the deliberately-corrupted-chain test from Success Criterion 2 corrupts one direction and requires the other direction's checker to catch it, reporting node ID + last-touched-by + invariant name).

**When to use:** after every pass (default on), and as the single authority after builder construction (D-10).

### Pattern 6: Canonical Printer + Determinism (TOOL-04, D-01..D-05, IR-05)

**What:** `strings.Builder` emitter; SSA numbers `%N` assigned by deterministic definition order, recomputed at print time from the BB's statement order (never cached across mutations); scalar expressions rendered bottom-up with parens decided by the precedence table in `doc/syntax.md` §7.6 (that table is the canonical rule source — reuse its levels, never invent new ones); float constants via `strconv.FormatFloat(v, 'g', -1, 32)` normalized so integral values keep a decimal point (`2` → `2.0f`, matching doc notation); zero `Loc` printed by default (D-03); stable `statement #N` headers (D-04). `.dot` emission reuses the same numbering pass, one `subgraph cluster_N` per statement (name must begin with `cluster` — verified syntactic convention from the DOT spec), edges def→user.

**When to use:** every golden test, every `--dump-after`, the Phase 5 parser-oracle comparison. The byte-identity property (D-02) is the product; the determinism rules in Pitfalls are how you keep it.

### Pattern 7: Golden Harness (TOOL-05, D-11..D-14)

**What:** One generic helper in `internal/golden`, modeled on the verified stdlib pattern (`cmd/gofmt/gofmt_test.go:18` — `var update = flag.Bool("update", false, "update .golden files")`; `go help packages` — `testdata/` directories are ignored by the go tool):

```go
package golden // internal/golden

var update = flag.Bool("update", false, "update golden files")

// Assert byte-compares got against path (creating under testdata/ on -update).
// One implementation serves IR text, .dot, and diagnostics (D-14).
func Assert(t *testing.T, path string, got []byte)
```

Package-level `flag.Bool` registration is safe across many importers: package initialization runs exactly once per program [CITED: go.dev/ref/spec §Program initialization — "only once" wording verified in the fetched spec]. `-update` writes with a trailing newline normalized; comparison is `bytes.Equal` (D-13); the failure report shows the first differing line with a few lines of context — a minimal hand-rolled locator, not a full diff (see Don't Hand-Roll).

### Pattern 8: Builder API — fluent typestate chain (FE-01, D-07..D-10)

**What:** The chain's stage ordering is enforced by Go method sets (typestate), not runtime checks alone: after `.Iter(...)` only `.Map(...)` exists; after `.Reduce(...)` only `.Map(...)`/`.Store(...)`; the chain value returned by `.Store(...)` is terminal. Illegal pipe orders become *compile-time* errors — the strongest form of D-10's eager checking, with runtime checks reserved for what types can't express (arity, extent compatibility). Typed handles (D-08): `Dim`, `Tensor`, `Expr` are distinct structs carrying their graph + type; `Tensor.At(axes...)` returns an operand handle with the axis binding baked in. Expr DSL (D-09) mirrors syntax.md §6.2 operator set 1:1; lambda bodies are Go funcs `func(l, r Expr) Expr`.

**When to use:** plan 01-04; the Phase 5 parser lowers to these same calls (FE-06), so the chain surface IS the frontend contract — keep it 1:1 with the doc notation.

### Anti-Patterns to Avoid

- **Letting passes touch node fields directly** — the D-15 embedded `Base` must expose no public mutators; if a pass needs a transformation the choke-point set doesn't provide, that's a design discussion, not a field write (design.md ambiguity protocol).
- **Caching SSA numbers across mutations** — numbering must be a pure function of current graph order (see Pitfall 6).
- **Iterating Go maps anywhere in emission/printing/verifier walks** — `maps.Keys` documents its order as "not specified ... not guaranteed" (verified via `go doc`) — this is the single biggest determinism killer; BB statement order + node ID are the only ordering sources.
- **Building legality as per-pass boilerplate** — central stage table (Pattern 4); per-pass duplication rots.
- **Instrumentations as pipeline passes** — PASS-02 forbids verify/dump/timing inside passes; they are PM callbacks so `--verify-each=0` can disable them without editing pass code.
- **Excessive test suites** — design.md explicitly bans them; the fundamental set is listed in Validation Architecture.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Float constant rendering | Custom float-to-string for the printer | `strconv.FormatFloat(v, 'g', -1, 32)` + doc-notation normalization | Shortest-round-trip formatting is subtle (tie-breaking, exponent rules); stdlib implements Ryu-style shortest exactly [VERIFIED: go doc strconv.FormatFloat] |
| Sorting/ordered emission | Custom sorts for deterministic order | `slices.SortFunc` (verified generic signature) over stable keys (node ID, def order) | `maps.Keys` order is explicitly unspecified [VERIFIED: go doc maps.Keys]; every emission path must sort from a stable key |
| Golden update flow | Bespoke per-package update mechanisms | One `internal/golden` package, gofmt pattern [VERIFIED: gofmt_test.go:18] | Package-level `flag.Bool` + `testdata/` convention is the Go-idiomatic shape and is already battle-tested inside the stdlib |
| Interning dedup | Pointer-chasing equality for types/dims | Intern tables keyed by `hash/fnv` (FNV-1a) content hashes [VERIFIED: go doc hash/fnv.New64a] | D-18 pointer-identity needs a deterministic interning step; FNV is the CLAUDE.md-prescribed hash |
| Structured compiler logging | Hand-rolled log levels | `log/slog` behind `internal/log` facade (`NewTextHandler`, `slog.Group` verified) | slog gives leveled, grouped, handler-swappable records; facade keeps pass code decoupled per CLAUDE.md |
| Diff display in golden failures | Full unified-diff implementation | First-differing-line report with context | A correct Myers diff is 200+ lines for marginal debugging value at this graph size; upgrade later only if pain is real |
| `.dot` emission | Graphviz generation library | Hand-rolled text emission (~80 lines) into `strings.Builder` | The DOT subset needed (digraph, nodes, edges, `subgraph cluster*`, labels) is tiny and verified from the spec; CLAUDE.md already prescribes text-only emission with `dot` as a dev-only renderer |

**Key insight:** in a stdlib-only project, "don't hand-roll" means "use the stdlib's hard-won primitives" (strconv, slices, slog, fnv, flag) rather than "import a library" — the failure modes these libraries prevent (float formatting, map-order nondeterminism, flag re-registration) are exactly Phase 1's determinism requirements.

## Common Pitfalls

### Pitfall 1: Map-iteration nondeterminism leaking into output
**What goes wrong:** golden tests pass locally, fail in CI, diffs show reordered lines with identical content.
**Why it happens:** Go randomizes map iteration per run; any `for k := range m` in a printer/walker/interner breaks byte-identity silently. `maps.Keys` documents this explicitly [VERIFIED: go doc].
**How to avoid:** emission order comes only from BB statement order, per-node operand order (fixed struct layout), and node IDs; sort anything map-sourced with `slices.SortFunc` by node ID before emitting. Code-review rule: no `range` over a map in `internal/ir`, `internal/pass`, `internal/visualization`.
**Warning signs:** golden diffs that reorder without changing content; tests that pass with `-count=1` but flake under `-count=10`.

### Pitfall 2: RAUW / use-list corruption during mutation
**What goes wrong:** use lists lose records or gain cycles; verifier reports inconsistent chains after a pass that "only did a simple replace".
**Why it happens:** splicing a `Use` out of list A into list B while iterating list A; RAUW(old, old); erasing a def that still has users.
**How to avoid:** choke-point functions implement the splice protocol once (snapshot-next-before-unlink; self-RAUW no-op; Erase refuses live users); passes never touch `Use` links directly; white-box corruption tests (Pattern 5) prove the verifier catches each corruption class.
**Warning signs:** "node has no def" diagnostics pointing at nodes the current pass never touched (stale links from an earlier corruption).

### Pitfall 3: Stale SSA numbering across mutations
**What goes wrong:** dumps/goldens show `%3` referring to two different nodes after a pass mutates the graph.
**Why it happens:** numbering cached on nodes at construction; a pass inserts/removes a def; cached numbers no longer match def order.
**How to avoid:** numbering is computed fresh in each print pass (definition-order walk of the BB); nothing mutates print state.
**Warning signs:** `--dump-after` output where use sites reference higher numbers than their defs.

### Pitfall 4: Golden-file hygiene failures
**What goes wrong:** tests fail on other machines or after unrelated edits; `-update` produces huge no-op diffs.
**Why it happens:** absolute paths or timestamps embedded in output; CRLF vs LF; missing/extra trailing newline; floats printed with locale- or platform-dependent formatting.
**How to avoid:** goldens contain only canonical printer output (D-03 minimal default — no locs, no times, no paths); harness normalizes trailing newline; printer formats floats via `strconv` only [VERIFIED: `FormatFloat` is locale-independent]; `.gitattributes` not needed if printer never emits `\r`.
**Warning signs:** golden diff containing a username, `/home/...`, or a date.

### Pitfall 5: Passes holding node pointers across pipeline boundaries
**What goes wrong:** an analysis or instrumentation holds `*Node` from before a pass; the pass erases it; later code dereferences freed intent.
**Why it happens:** IR is a mutable pointer graph; any pass may erase any node between two points.
**How to avoid:** analyses are computed fresh per pipeline position and invalidated by default (PASS-03); instrumentation callbacks get the graph, not retained node pointers; document that `*Node` is valid only within one pass invocation.
**Warning signs:** analysis results keyed by node pointer identity surviving a transform pass.

### Pitfall 6: Timing/diagnostic output contaminating golden streams
**What goes wrong:** `--pass-timing` or slog output lands inside a captured golden string.
**Why it happens:** everything written to the same `io.Writer`/stdout.
**How to avoid:** goldens capture only the printer's `strings.Builder` bytes; logs and timing reports go to stderr via slog's handler; harness never tees.
**Warning signs:** golden containing `elapsed=` or a log level string.

### Pitfall 7: `flag` re-registration panic in the golden harness
**What goes wrong:** `panic: flag redefined: update` when multiple test packages register the flag.
**Why it happens:** `flag.Bool` panics on duplicate registration; naive copies of the gofmt pattern register per-package.
**How to avoid:** exactly one registration in `internal/golden`'s package-level `var` — package initialization runs once per test binary regardless of how many test packages import it [CITED: go.dev/ref/spec §Program initialization, "only once" verified in fetched spec text].
**Warning signs:** panic at test binary startup, not in a test.

### Pitfall 8: Unknown-Kind handling silently skipping nodes
**What goes wrong:** a future pass introduces a new node kind; the verifier/legality/printer silently skip it; an invalid graph passes Phase 1's gates and explodes in Phase 3.
**Why it happens:** Go type-switch `default:` branches that no-op; walks keyed to today's kinds.
**How to avoid:** every type-switch and every walk has a `default: return error("unknown kind %d")` (IR-03 forward-compat rule from CLAUDE.md); legality table rejects unknown kinds structurally (Pattern 4); one test constructs a fake high-kind node and asserts rejection.
**Warning signs:** adding a kind compiles cleanly everywhere with zero test failures — that silence is the bug.

### Pitfall 9: Float-format drift in canonical constants
**What goes wrong:** `2.0f` prints as `2f` (or vice versa) after a Go upgrade or a printer refactor; every golden churns.
**Why it happens:** `'g'` shortest formatting drops the decimal point for integral values; someone "fixes" rendering without a pinned rule.
**How to avoid:** one `formatConst` function owning the rule (FormatFloat + "append `.0` if no `.`/`e`" + `f` suffix), pinned by goldens; MEDIUM confidence that shortest-round-trip digit choice is stable across Go versions — the goldens themselves are the tripwire [ASSUMED → mitigated by goldens].

## Code Examples

Verified patterns (sources cited per block) translated to this project's Go substrate.

### Choke-point mutation with intrusive Use records (IR-01, IR-04, D-16, D-17)

```go
// Source: LLVM Programmer's Manual, "The Use class" / "replaceAllUsesWith"
// (llvm.org/docs/ProgrammersManual.html, verified this session);
// llir/llvm named in CLAUDE.md as the definitive pure-Go design reference.

type Use struct {
    Val  Value          // def this operand points to
    User Node           // owning node (slot index derivable from User's operand layout)
    prev, next *Use     // intrusive links in Val's use-list
}

// ReplaceAllUsesWith splices every use of old onto new in O(uses),
// exactly mirroring LLVM: "traverses the use list of a Value changing
// all Users of the current value to refer to V instead".
func (ctx *MutContext) ReplaceAllUsesWith(old, new Value) {
    if old == new { return }                    // self-RAUW guard
    for u := old.firstUse; u != nil; {
        next := u.next                          // snapshot BEFORE splicing
        u.unlink()                              // out of old's list
        u.linkInto(new)                         // into new's list, preserving position intent
        u.Val = new
        u.User.Base().LastTouchedBy = ctx.Pass  // D-17 structural stamp
        u = next
    }
}
```

### Pass interface + PM run loop with instrumentation (PASS-01, PASS-02, IR-07)

```go
// Source: MLIR PassInfrastructure + PassInstrumentation docs (mlir.llvm.org,
// verified this session), narrowed to a flat single-BB pipeline.

func (pm *Manager) Run(g *ir.Graph) error {
    for _, pass := range pm.passes {
        if err := pm.checkPreconditions(pass); err != nil { // PASS-01
            return fmt.Errorf("pipeline order: %w", err)
        }
        for _, in := range pm.instrumentation { in.BeforePass(pass, g) }
        err := pass.Run(pm.ctx, g)
        for i := len(pm.instrumentation) - 1; i >= 0; i-- {  // stack-like, MLIR-verified
            pm.instrumentation[i].AfterPass(pass, g, err)
        }
        if err != nil { return fmt.Errorf("pass %s: %w", pass.ID(), err) } // abort, MLIR-verified
        if err := pm.checkLegality(pass.Produces(), g); err != nil {       // IR-07
            return err
        }
        pm.invalidateAnalyses(pass.Preserved())                             // PASS-03
    }
    return nil
}
```

### Golden harness core (TOOL-05, D-11..D-14)

```go
// Source: $GOROOT/src/cmd/gofmt/gofmt_test.go:18 (verified this session);
// "go help packages" testdata convention (verified this session).

package golden

var update = flag.Bool("update", false, "update golden files")

func Assert(t *testing.T, path string, got []byte) {
    t.Helper()
    got = bytes.TrimRight(got, "\n")
    got = append(got, '\n')
    if *update {
        if err := os.WriteFile(path, got, 0o644); err != nil { t.Fatal(err) }
        return
    }
    want, err := os.ReadFile(path)
    if err != nil { t.Fatalf("read golden %s: %v (run with -update)", path, err) }
    if !bytes.Equal(want, got) { // D-13 exact byte compare
        t.Fatalf("golden mismatch: %s\n%s", path, firstDiff(want, got))
    }
}
```

### Deterministic `.dot` cluster emission (TOOL-03, D-05)

```go
// Source: graphviz.org/doc/info/lang.html (verified this session):
// "If the name of the subgraph begins with cluster, Graphviz notes the
//  subgraph as a special cluster subgraph" — a syntactic convention.

func EmitDot(w io.Writer, g *ir.Graph) {
    fmt.Fprintf(w, "digraph %s {\n", g.Name)          // digraph grammar, verified
    for _, stmt := range g.Block.Statements {          // BB order: deterministic
        fmt.Fprintf(w, "  subgraph cluster_%d {\n", stmt.ID())
        for _, n := range stmt.Nodes {                 // operand order: struct layout
            fmt.Fprintf(w, "    n%d [label=%q];\n", n.ID(), dotLabel(n, numbering))
            for u := range n.Uses() {                  // def-use edges, explicit per D-05
                fmt.Fprintf(w, "    n%d -> n%d;\n", u.Val.ID(), n.ID())
            }
        }
        fmt.Fprintf(w, "  }\n")
    }
    fmt.Fprintf(w, "}\n")
}
```

### Builder fluent chain with typestate ordering (FE-01, D-07..D-10)

```go
// Chain stage types make illegal pipe orders compile-time errors (D-10):
// RootChain has only Map(); PostReduceChain has Map()/Store(); Stored is terminal.

stmt := b.Stmt().Iter(m, k, n).
    Map(pkg.At(lhs, m, n), pkg.At(rhs, n, k),
        func(l, r pkg.Expr) pkg.Expr { return b.Mul(l, r) }).As(m, k, n).
    Reduce(b.Float32(0.0),
        func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) }).
    Map(func(v pkg.Expr) pkg.Expr { return b.Mul(v, scale) }).As(m, k).
    Store(pkg.At(out, m, k))
graph, err := b.Build() // eager checks at every call; full verifier runs in Build via PM path
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| External parser generators / IR libs (goyacc, llir/llvm as a dependency) for Go compilers | Hand-written recursive descent + stdlib-only data structures | Project constraint (design.md) + LLVM/GCC both hand-written recursive descent today (per CLAUDE.md sources) | llir/llvm is reference-only; its *design* (use-lists, RAUW) is what we port |
| `golang.org/x/exp/slices` | `slices`/`maps` in stdlib (Go 1.21+) | Go 1.21 (2023) | Verified present in 1.26.2 via `go doc`; no version-sensitive stdlib features beyond this needed |
| Per-pass verifier calls inside pass bodies | PM instrumentation callbacks (`-verify-each` as instrumentation) | MLIR established the instrumentation model (verified docs) | PASS-02 is structural in our PM from day one — retrofitting costs a pipeline rewrite |
| Text IR parsed back for round-trip testing | Canonical printer + byte-compare goldens (D-06) | This project's decision (avoids building a second grammar before the real frontend) | Golden harness (Pattern 7) is the sole comparison oracle; Phase 5 parser compares against the same goldens |

**Deprecated/outdated:**
- Caching SSA numbers on nodes between mutations (LLVM never does this; numbering is a print-time function — see Pitfall 3).
- Per-package golden `-update` flags (the re-registration panic, Pitfall 7) — one harness package only.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `matmul_scale`'s oracle graph is the tca.md §5.1 / syntax.md §9.3 chain (map(l*r) as (m,k,n) → reduce<@n>(init=0.0f) → zero-iter map(*scale) as (m,k) → store(out[m][k])); the docs' printed TCA text doubles as the first golden expectation | Summary, Pattern 8, Validation | Low — both docs agree verbatim; planner should lift the golden text directly from doc/tca.md §5.1 |
| A2 | Shortest-round-trip float digit choice in `strconv.FormatFloat(-1)` is stable across future Go versions | Pattern 6, Pitfall 9 | Medium — mitigated: goldens tripwire any drift; if drift occurs, pin an explicit digit algorithm in `formatConst` |
| A3 | White-box (same-package) tests in `internal/ir` are acceptable to reach `Use` internals for the corrupted-chain test, given design.md's "only necessary tests" rule | Pattern 5, Validation | Low — alternative is an exported test-only corruption hook, strictly worse |
| A4 | One flat pipeline (no nested pass managers) suffices for all of v1 — nesting (per-statement sub-pipelines) arrives only if multi-kernel clusters need it, and can be added behind the same `Manager` API | Pattern 3 | Low — MLIR's nesting exists for region nesting our single-BB IR doesn't have (verified in docs) |
| A5 | `internal/visualization` hosts `.dot` emission (per CLAUDE.md package list) rather than `internal/ir/dot.go` | Package Structure | Cosmetic — planner may colocate if import cycles demand it |

**All other claims in this research are VERIFIED (toolchain `go doc`/`$GOROOT` reads) or CITED (fetched official docs: MLIR PassManagement, MLIR DialectConversion, LLVM ProgrammersManual, LLVM opt CommandGuide, Graphviz DOT lang spec, Go spec, gofmt_test.go).**

## Open Questions

1. **`--dump-after` filtering granularity.** TOOL-06 names the flag but not its value shape.
   - What we know: MLIR supports both `-mlir-print-ir-after-all` and `-mlir-print-ir-after=<list>` (verified).
   - What's unclear: whether Phase 1 wants boolean-all (simpler) or name-filtered (one flag, two uses).
   - Recommendation: implement name-filtered (`--dump-after=all` default-off) — trivially more code, matches the verified MLIR precedent, and `--dump-graphviz`/`--pass-timing` follow the same shape.

2. **Golden harness package name/location.**
   - What we know: must be one package with the single `flag.Bool` registration (Pitfall 7); D-14 says "one generic golden harness".
   - What's unclear: `internal/golden` vs inside Phase 2's `internal/testutil` (which CLAUDE.md reserves for the e2e driver).
   - Recommendation: `internal/golden` now; `testutil` composes it in Phase 2.

3. **Statement-ID policy under future split/merge passes (D-04).**
   - What we know: D-04 requires `statement #N` stable across passes so diffs track a statement; Phase 1 has no split/merge passes.
   - What's unclear: what happens in Phase 4 when loop-merge nests statements or effect-linearize rewrites chains.
   - Recommendation: Phase 1 assigns IDs at construction (builder) and preserves them under in-place mutation; document now that passes which split a statement derive child IDs as `#N.M` — decide the exact scheme in Phase 4 planning when the passes exist. Flagging so D-04's contract is written down before it bites.

4. **Where `Loc.PassOrigin` gets displayed.**
   - What we know: D-03 pins locs out of the default print; D-17 stamps `LastTouchedBy` on nodes; PASS-06 requires it in verifier diagnostics.
   - What's unclear: whether `--dump-after` verbose mode shows it inline (`loc("pass=reduce-lower")` style) or only verifier output carries it.
   - Recommendation: verifier diagnostics always carry it; printer verbose mode shows it — decide when wiring TOOL-06 flags in plan 01-03 (either is defensible; pick one and pin in goldens' verbose-mode tests).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Go toolchain | All of Phase 1 | ✓ | go1.26.2 linux/amd64 (verified `go version`) | — |
| `flag`, `slog`, `slices`, `maps`, `hash/fnv`, `strconv`, `testing` (stdlib) | pass flags, logging, interning, printer, golden harness | ✓ | Go 1.26.2 stdlib (signatures verified via `go doc`) | — |
| Graphviz `dot` renderer | TOOL-03 dev-time rendering of `.dot` dumps | ✗ not installed (verified `command -v dot`) | — | None needed for any Phase 1 gate — Phase 1 is emit-only (CLAUDE.md); rendering is a dev convenience. Optional `apt install graphviz` if the user wants to eyeball dumps |
| `go/ast`, `go/scanner` sources as *reference* | D-15 architecture reference | ✓ | `$GOROOT/src/go/ast/ast.go` read this session | — |

**Missing dependencies with no fallback:** none — Phase 1 is code/config-only plus optional rendering.

**Missing dependencies with fallback:** `dot` binary (fallback: textual dumps only; no behavioral impact).

## Validation Architecture

Test framework: Go `testing` (stdlib) — no config file needed; the only "config" is the `-update` flag flow (D-12).

| Property | Value |
|----------|-------|
| Framework | Go stdlib `testing` (Go 1.26.2) |
| Config file | none |
| Quick run command | `go test ./internal/ir/ ./internal/pass/ ./pkg/ -count=1` |
| Full suite command | `go test ./... -count=1` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| IR-01 | use-def/def-use chains consistent; trace uses; RAUW updates all refs; delete detaches | unit (white-box in `internal/ir`) | `go test ./internal/ir/ -run TestUseChain -count=1` | ❌ Wave 0 |
| IR-01/IR-04/PASS-06 | deliberately corrupted chain caught; diagnostic names node ID + last-touched-by + invariant | unit (white-box corruption) | `go test ./internal/ir/ -run TestVerifierCatchesCorruption -count=1` | ❌ Wave 0 |
| IR-02 | graph holds exactly one BB; statements hold chains; RHS is Use-or-Value | unit | `go test ./internal/ir/ -run TestSingleBlock -count=1` | ❌ Wave 0 |
| IR-03/Pass-IR-07 | unknown Kind rejected by PM legality; known kinds pass constructed-stage window | unit | `go test ./internal/pass/ -run TestLegalityRejectsUnknownKind -count=1` | ❌ Wave 0 |
| IR-05/TOOL-04/D-02 | canonical printer: two structurally identical graphs print byte-identical | unit | `go test ./internal/ir/ -run TestCanonicalPrint -count=1` | ❌ Wave 0 |
| PASS-01 | out-of-order pipeline (reduce-lower before iter-lower) rejected by preconditions | unit | `go test ./internal/pass/ -run TestPreconditionRejectsOrder -count=1` | ❌ Wave 0 |
| PASS-02/TOOL-06 | verify-after-every-pass runs as PM callback; `--verify-each=0` disables it; dump-after/timing flags fire | unit | `go test ./internal/pass/ -run TestInstrumentation -count=1` | ❌ Wave 0 |
| PASS-03 | transform pass without preservation whitelist invalidates cached analysis | unit | `go test ./internal/pass/ -run TestAnalysisInvalidation -count=1` | ❌ Wave 0 |
| FE-01 + TOOL-05 | `pkg/` builder constructs `matmul_scale`; printed output byte-matches golden lifted from doc/tca.md §5.1 | golden | `go test ./pkg/ -run TestMatmulScaleGolden -count=1` | ❌ Wave 0 |
| TOOL-03/D-05 | `.dot` emission mirrors text numbering; deterministic; cluster-per-statement | golden | `go test ./internal/visualization/ -run TestDotGolden -count=1` | ❌ Wave 0 |
| TOOL-01 | diagnostics carry severity + span + structured context | unit | `go test ./internal/diagnostic/ -count=1` | ❌ Wave 0 |
| VER-08 | determinism byte-compare: oracle graph printed twice in-process byte-identical; committed golden anchors cross-process identity | unit + golden | `go test ./pkg/ -run TestDeterminism -count=10` | ❌ Wave 0 |

(Exact test names are illustrative; the planner owns final names. Deliberately absent: coverage theater — design.md bans excessive suites.)

### Sampling Rate
- **Per task commit:** `go test ./internal/ir/ ./internal/pass/ ./pkg/ -count=1` (hermetic, seconds)
- **Per wave merge:** `go test ./... -count=1`
- **Phase gate:** full suite green + `go vet ./...` + `go list -m all | wc -l` == 1 (stdlib-only enforcement, CLAUDE.md) before `/gsd-verify-work`; golden CI mode = no `-update` flag (D-12)

### Wave 0 Gaps
- [ ] `go.mod` (module `tcacomp`, `go 1.26.2`, zero requires) — repo has no Go files at all yet
- [ ] Package skeletons: `internal/{ir,pass,analysis,flags,diagnostic,log,visualization,golden}`, `pkg/`
- [ ] `internal/golden` harness (Pattern 7) — needed by every other plan's tests
- [ ] First golden file `pkg/testdata/matmul_scale.ir` lifted from doc/tca.md §5.1 (planner: diff the lifted text against §9.3 for consistency first)
- [ ] `internal/flags` defaults for `VerifyEach/DumpAfter/DumpGraphviz/PassTiming` — needed by pass-manager tests

*(No existing test infrastructure to build on — greenfield.)*

## Security Domain

Phase 1 is compiler-internal infrastructure: no network surface, no authentication/authorization, no persistent storage, no secrets. Most ASVS categories do not apply; `security_enforcement` is not disabled in config, so the mapping below is stated explicitly rather than omitted.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth surface in Phase 1) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes — at the API boundary | Builder eager checks (D-10) + full verifier via PM path; graphs are data — malformed graphs must fail loudly, never construct partial state. This is the compiler analog of input validation and the user's global rule "validate at system boundaries" applies literally |
| V6 Cryptography | no | — (no crypto; `hash/fnv` is for interning dedup, NOT security — noted so nobody mistakes it for a security hash later) |

### Compiler-Specific Hygiene (not ASVS, but real for this phase)

| Pattern | Risk | Mitigation |
|---------|------|------------|
| Golden files / dumps containing absolute paths, hostnames, or usernames | Information leakage in committed artifacts + flaky cross-machine tests | Canonical printer emits no environment-derived strings (D-03); review rule for `testdata/` additions (Pitfall 4) |
| `panic` in library code (`internal/*`) | Compiler should surface errors with diagnostics, never crash the host process | Errors return through PM/diagnostic plumbing; `panic` reserved for "impossible" internal states per design.md's ask-first protocol |
| Dependency supply chain | Compilers are high-trust toolchain components | Zero external modules — enforced mechanically by `go list -m all | wc -l` == 1 |

## Sources

### Primary (HIGH confidence — fetched and read this session, 2026-10-15)
- `https://mlir.llvm.org/docs/PassManagement/` — PassInstrumentation hook set (runBefore/AfterPass, runBefore/AfterPipeline, analysis hooks), addInstrumentation stack semantics, PassTiming + IR-printing standard instrumentations, analysis preservation contract (invalidate-by-default, markAllAnalysesPreserved/markAnalysesPreserved), signalPassFailure abort semantics, OpPassManager nesting (confirmed NOT applicable to flat single-BB IR)
- `https://mlir.llvm.org/docs/DialectConversion/` — ConversionTarget legality actions (Legal / Dynamic(callback) / Illegal), unknown-op handling, markUnknownOpDynamicallyLegal, markOpRecursivelyLegal, partial/full/analysis conversion modes (full mode = "only successful if all operations are properly legalized" → basis for IR-03 unknown-kind rejection)
- `https://llvm.org/docs/ProgrammersManual.html` — Value/User/Use model: use-list (`use_begin`/`use_size`/`use_empty`/`use_back`), `replaceAllUsesWith` "traverses the use list ... changing all Users", def-use vs use-def chain terminology, "no distinction between an SSA variable and the operation that produces it", names debug-only (use identity/pointers)
- `https://llvm.org/docs/CommandGuide/opt.html` — `-verify-each` ("add a verify pass after every pass"), `-time-passes` ("record the amount of time needed for each pass"), `-print-before/-print-after` family semantics
- `https://graphviz.org/doc/info/lang.html` — DOT grammar (`[strict] (graph|digraph) [ID] { stmt_list }`), `subgraph cluster*` naming convention ("If the name of the subgraph begins with cluster, Graphviz notes the subgraph as a special cluster subgraph")

### Primary (HIGH confidence — verified against local Go 1.26.2 toolchain, not training memory)
- `go version` → go1.26.2 linux/amd64
- `go doc log/slog.New`, `log/slog.Group`, `log/slog.NewTextHandler`, `log/slog.Handler` — signatures confirmed
- `go doc slices.SortFunc`, `go doc maps.Keys` (order "not specified ... not guaranteed from one call to the next") — determinism pitfall confirmed from doc text
- `go doc strconv.FormatFloat`, `go doc hash/fnv.New64a` — signatures confirmed
- `go help packages` — "directories named 'testdata'" ignored by the go tool (D-11 convention)
- `$GOROOT/src/cmd/gofmt/gofmt_test.go:18` — `var update = flag.Bool("update", false, "update .golden files")` (golden-pattern precedent; sibling instances in crypto/tls, math/rand/v2, go/doc, cmd/compile internal tests)
- `$GOROOT/src/go/ast/ast.go` — `Node` interface (`Pos`/`End`), `Expr`/`Stmt` family interfaces with marker methods (D-15 reference shape)
- `https://go.dev/ref/spec#Package_initialization` — package initialization runs "only once" (single `flag.Bool` registration safety)
- `command -v dot` → absent (Graphviz renderer not installed)

### Secondary (MEDIUM confidence)
- `https://github.com/llir/llvm` — "Library for interacting with LLVM IR in pure Go" (org-level description verified on the fetched page; the detailed use-list/RAUW design specifics are from project CLAUDE.md decision (b), which names llir/llvm as the design reference — external-module ban makes it reference-only by construction)

### Tertiary (LOW confidence — none)
- No claim in this research rests on unverified single-source web material.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero external deps is a project constraint; every stdlib API cited was verified via `go doc` or `$GOROOT` source reads on the exact Go 1.26.2 toolchain that will build the project
- Architecture: HIGH for PM/instrumentation/analysis-preservation/golden patterns (verified MLIR/LLVM/std-lib precedents, directly translated); MEDIUM for the legality-window adaptation (explicit research flag; ConversionTarget substrate verified, stage-centered adaptation is applied judgment) and the SSADef/SSAVar-to-LLVM-Value mapping detail
- Pitfalls: HIGH for map-order, golden hygiene, unknown-kind, and use-list items (each anchored to a verified doc/toolchain fact); MEDIUM for float-format stability (tripwired by goldens)

**Research date:** 2026-10-15
**Valid until:** 2026-11-14 (stable domain — compiler-infrastructure patterns move slowly; the only drift risk is Go toolchain float-format behavior, covered by goldens)

---

*Phase: 01-ir-core-pass-infrastructure-tooling*
*Research produced: 2026-10-15*
