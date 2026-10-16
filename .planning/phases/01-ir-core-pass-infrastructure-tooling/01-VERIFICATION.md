---
phase: 01-ir-core-pass-infrastructure-tooling
verified: 2026-10-16T00:00:00Z
status: gaps_found
score: 22/23 must-haves verified
re_verification:
  previous_status: none
  previous_score: none
  gaps_closed: []
  gaps_remaining: []
  regressions: []
gaps:
  - truth: "Use-def and def-use chains stay consistent through every choke-point mutation (SetDef/AddUse/RemoveUse/RAUW/Erase)"
    status: failed
    reason: >
      AddUse and RemoveUse — two of the five sanctioned IR-04 choke points — corrupt
      the intrusive use-list. operandList stores Use records BY VALUE in a mutable
      slice (node.go:168 `ops []Use`); addSlot's append (node.go:178) reallocates on
      full-capacity nodes (a constructed binop is exactly len 2 cap 2, so the FIRST
      AddUse reallocates), stranding every *Use embedded in every def's use-list
      (CR-01). removeSlot's slices.Delete (node.go:185) shifts records under live
      *Use pointers, and RemoveUse (mutate.go:66-79) then nils u.Val/u.User AFTER
      the shift — destroying a live, correctly-wired operand (CR-02; empirical repro
      in 01-REVIEW.md). Verified present in code, not just claimed. Both choke points
      have ZERO call sites and ZERO test coverage today (IN-04a), so the suite is
      green and the defect is latent: it detonates on the first Phase 2/3 transform
      pass that grows or shrinks an operand list. Impact-wise this is pre-Phase-2
      debt (no consumer exists yet), but it fails the plan's must-have truth verbatim
      and undercuts the phase goal's "verified substrate" claim — the mutation API is
      the piece every later phase hangs off. It gaps THIS phase; it is not deferred
      (no later roadmap phase schedules the fix; Phases 3-4 lowering/DCE passes are
      the first call sites).
    artifacts:
      - path: "internal/ir/node.go"
        issue: "operandList.ops is []Use by value; addSlot append reallocates and moves records under live *Use pointers (lines 167-190)"
      - path: "internal/ir/mutate.go"
        issue: "AddUse (47-61) and RemoveUse (66-79) thread *Use pointers into the by-value slice; RemoveUse clears the record after slices.Delete has shifted it, niling the wrong (live) operand"
    missing:
      - "Store Use records as pointers: operandList.ops []*Use so records never move (01-REVIEW.md CR-01/CR-02 fix sketch)"
      - "In RemoveUse, unlink from the def and clear u.Val/u.User BEFORE dropping the slot"
      - "White-box round-trip tests: AddUse on a full-capacity binop keeps Verify clean; RemoveUse of each slot position of a select keeps Verify clean and surviving DefOf records intact (closes IN-04a coverage hole)"
---

# Phase 1: IR Core, Pass Infrastructure & Tooling Verification Report

**Phase Goal:** A verified, debuggable IR and pass-pipeline foundation with a programmatic graph-builder API — the substrate every later phase hangs off, with the correctness infrastructure (verifier, diagnostics, determinism, golden tests) structural from day one.
**Verified:** 2026-10-16
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

The phase goal is substantially achieved: the module foundation, IR core, pass manager, tooling backbone, builder API, doc-pinned oracle, and determinism infrastructure all exist, are substantive, wired, and tested — with one confirmed defect in the mutation choke-point API that fails a plan must-have verbatim. The correctness infrastructure (verifier, diagnostics, determinism, goldens) is genuinely structural, not decorative: every claim below was checked against running code, not SUMMARY prose.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1.1 | Use-def/def-use chains stay consistent through every choke-point mutation (SetDef/AddUse/RemoveUse/RAUW/Erase) | FAILED | RAUW/SetDef/Erase proven (TestUseChain, TestRauwGuards pass), but AddUse/RemoveUse corrupt the chain — CR-01/CR-02 confirmed in `internal/ir/node.go:168,178,185` and `internal/ir/mutate.go:47-79`; zero call sites today so the failure is latent. See Gaps. |
| 1.2 | A deliberately corrupted chain is caught by the verifier, naming node ID, last-touched-by pass, and invariant | VERIFIED | TestVerifierCatchesCorruption passes (full suite green); Violation struct + both-direction cross-check in verify.go |
| 1.3 | Every IR node carries a stable integer ID and a Loc; a graph holds exactly one basic block | VERIFIED | TestSingleBlock, TestNodeIdentity pass; Base{ID, Loc{Span, PassOrigin}} in node.go; AddStatement appends to the single Block |
| 1.4 | Graphs/nodes constructible through an exported surface usable from any package | VERIFIED | TestConstructSurface (external `ir_test` package) + pass/legality tests, visualization tests, and pkg/ builder all construct exclusively through NewGraph/NewNode/typed wrappers |
| 1.5 | Module builds and vets clean with exactly one module entry | VERIFIED | `go build ./...` + `go vet ./...` clean; `go list -m all \| wc -l` == 1; no go.sum; go.mod is exactly `module tcacomp` / `go 1.26.2` |
| 2.1 | PM runs an ordered flat pipeline, aborts with named-pass error on failure | VERIFIED | TestPreconditionRejectsOrder + manager.go "pass %s: %w" / "pipeline order" paths |
| 2.2 | Pipeline ordered against declared preconditions rejected before any pass runs | VERIFIED | Same test: [reduce-lower, iter-lower] rejected, [iter-lower, reduce-lower] accepted |
| 2.3 | After every pass, graph checked against produced stage's legality window; unknown Kinds always rejected | VERIFIED | TestLegalityRejectsUnknownKind (>= KindEnd fake kind + fail-closed unset stage); checkLegality called from Manager.Run after each pass |
| 2.4 | Verify-after-pass, IR dump, pass timing run as PM callbacks; --verify-each=0 disables verification | VERIFIED | TestInstrumentation: callback order (AfterPass reversed), VerifyEach gate, IRDump selection, sorted PassTiming |
| 2.5 | Pipeline verified once at end even with empty pass list; violations returned as error | VERIFIED | TestPipelineEndErrorSurfaces on NewManager(nil); VerifyEach implements PipelineEnd |
| 2.6 | Transform pass without preservation whitelist invalidates all cached analyses | VERIFIED | TestAnalysisInvalidation (direct + through Manager.Run) |
| 3.1 | Two structurally identical graphs print byte-identically | VERIFIED | TestCanonicalPrint (double-build + repeated-print stability) |
| 3.2 | Canonical matmul_scale print byte-matches the golden lifted from doc/tca.md §5.1 | VERIFIED | TestMatmulScaleGolden green; golden bytes inspected (LF, no trailing spaces, single trailing newline, §5.1 lines exact); file unchanged since 01-03 commit 0488ddd |
| 3.3 | Diagnostics carry severity, source span, structured context | VERIFIED | diagnostic.go Severity{Error,Warning,Remark}/Span/Context; 5 tests incl. String rendering |
| 3.4 | One generic golden harness serves all output kinds with -update flow | VERIFIED | golden.go Assert/Check; exactly one `flag.Bool("update")` in the repo; TestGoldenHarness |
| 3.5 | .dot emission mirrors text printer numbering with cluster per statement | VERIFIED | TestDotGolden + double-emission determinism; EmitDot reuses ir.ComputeNumbering; `subgraph cluster_` present |
| 4.1 | Fluent builder chain constructs matmul_scale; canonical print byte-matches committed golden | VERIFIED | pkg/oracle_test.go buildMatmulScale → b.Build() → ir.Print → golden.Assert; green |
| 4.2 | Illegal pipe orders are compile-time errors; arity/shape violations fail eagerly at call site | VERIFIED | 8 distinct rung types in chain.go (grep 8/8); TestBuilderTypestate compiles legal sequences; TestBuilderEagerChecks covers 9 negative cases; no `panic(` in pkg/ |
| 4.3 | Build() runs full verifier through the normal PM path; surfaces errors even with empty pass list | VERIFIED | builder.go contains NewManager+VerifyEach, zero `ir.Verify`; TestBuildVerifiesThroughPM positive AND negative (unwired operand) paths |
| 4.4 | Oracle prints byte-identically across repeated in-process builds and across processes | VERIFIED | TestDeterminism + TestDeterminismDot; `go test ./pkg/ -run TestDeterminism -count=10` green; committed golden is the cross-process anchor |
| 5.1 | --dump-after produces real IR text, not a placeholder header | VERIFIED | pkg/instrument.go TextEmitter = ir.Print; TestInstrumentationsRealDumpOutput asserts real content through Manager.Run |
| 5.2 | --dump-graphviz produces real digraph output, not a placeholder | VERIFIED | DotEmitter wraps visualization.EmitDot; same test asserts `digraph` + `subgraph cluster_` |
| 5.3 | Standard instrumentation set available as one constructor for Phase 5 CLI | VERIFIED | DefaultInstrumentations returns VerifyEach+IRDump+PassTiming; TestDefaultInstrumentationsComposition |

**Score:** 22/23 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| go.mod | module tcacomp, go 1.26.2, zero requires | VERIFIED | exact content; no go.sum |
| internal/ir/{kind,types,node,use,construct,mutate,verify,print,walk}.go | IR core per 01-01 | VERIFIED | all substantive (128-565 lines); KindVersion=1, KindEnd sentinel, FNV interner, Violation, Verify |
| internal/pass/{manager,legality,instrument,cache,context}.go | PM per 01-02 | VERIFIED | Pass interface, checkLegality, WindowFor fail-closed, Instrumentation/PipelineEnd, cache Invalidate |
| internal/flags/flags.go, internal/analysis/analysis.go | PASS-04/05 skeletons | VERIFIED | pinned defaults; read-only Analysis interface with MutContext rule in doc comment |
| internal/diagnostic/diagnostic.go, internal/log/log.go | TOOL-01/02 | VERIFIED | 3-severity model; slog facade, no stdlib `log` import, stderr default |
| internal/golden/golden.go, internal/visualization/dot.go | TOOL-05/03 | VERIFIED | single -update flag; EmitDot(w, g) error, text-only |
| pkg/{builder,chain,expr,instrument}.go | FE-01 builder + TOOL-06 wiring | VERIFIED | 269/551/332/45 lines; 8 rungs; 3+ expr DSL functions; DefaultInstrumentations |
| pkg/testdata/matmul_scale.ir | §5.1 oracle golden | VERIFIED | byte-exact, doc-pinned, unchanged since 01-03 |
| pkg/{oracle,determinism,builder,instrument}_test.go | flagship tests | VERIFIED | all present, all green |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| mutate.go | node.go LastTouchedBy | every choke point stamps ctx.Pass | WIRED (grep 2 matches) |
| verify.go | use.go use-lists | both chain directions walked + cross-checked | WIRED (firstUse walk) |
| construct.go | node.go Graph/Block | IDs only from g.NewID() | WIRED (4 NewID sites) |
| manager.go | legality.go | checkLegality after every pass | WIRED |
| instrument.go | ir.Verify | VerifyEach AfterPass + AfterPipeline | WIRED (5 ir.Verify sites) |
| manager.go | instrument.go PipelineEnd | AfterPipeline dispatched after last pass, error propagated | WIRED |
| print.go | syntax.md §7.6 precedence | lambda parens from pinned table | WIRED (prec usage) |
| dot.go | print.go numbering | shared ComputeNumbering | WIRED |
| golden.go | testdata/ convention | -update writes, CI byte-compares | WIRED |
| builder.go Build() | pass.Manager + VerifyEach | single authoritative verification path | WIRED; `ir.Verify` absent from pkg/ (grep 0) |
| oracle_test.go | pkg/testdata/matmul_scale.ir | golden.Assert on ir.Print bytes | WIRED + byte-match proven |
| instrument.go | pass.IRDump fields + visualization.EmitDot | real emitters wired | WIRED; "emitter wired" placeholder strings absent from pkg/ |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| TestMatmulScaleGolden | `ir.Print(graph)` bytes | buildMatmulScale → pkg builder → ir constructors → canonical printer | Yes — byte-matches doc-pinned golden (not a static echo: the golden was committed by 01-03 BEFORE the builder existed; 01-04 reconciled the printer to it) | FLOWING |
| TestDotGolden | EmitDot bytes | same graph, shared numbering pass | Yes — golden + determinism double-emission | FLOWING |
| DefaultInstrumentations | IRDump text/dot | ir.Print / EmitDot callbacks through Manager.Run on a stub pass | Yes — test asserts real `statement #N` header + stage text + digraph, placeholder line prefixes asserted absent | FLOWING |
| VerifyEach pipeline-end error | []ir.Violation | ir.Verify over the graph (incl. empty pass list) | Yes — negative path proven on an unwired-operand graph | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full phase suite | `go test ./... -count=1` | all 6 test packages ok | PASS |
| Vet | `go vet ./...` | clean | PASS |
| Stdlib-only | `go list -m all \| wc -l` | 1 | PASS |
| Determinism | `go test ./pkg/ -run TestDeterminism -count=10` | ok | PASS |
| Golden integrity | `cat -A pkg/testdata/matmul_scale.ir` + `xxd` tail | LF only, no trailing spaces, single trailing newline | PASS |
| No map iteration in emission | `grep -rn "range.*map\[" internal/{ir,pass,visualization}` | 0 matches | PASS |

### Requirements Coverage

All 22 phase requirement IDs are claimed across the five plans' frontmatter (01-01: IR-01..06, PASS-06; 01-02: IR-03, IR-07, PASS-01..06, TOOL-06; 01-03: IR-05, TOOL-01..05; 01-04: FE-01, TOOL-05, VER-08; 01-05: TOOL-06) and REQUIREMENTS.md maps exactly these 22 to Phase 1 — no orphans, no extras. Per-ID evidence:

| Requirement | Status | Evidence |
|-------------|--------|----------|
| IR-01 | SATISFIED | UsesOf/DefOf/RAUW/Erase + TestUseChain; trace-uses and replace-def proven. AddUse/RemoveUse defect is scored under IR-04 below. |
| IR-02 | SATISFIED | Graph→single Block→Statement; TestSingleBlock |
| IR-03 | SATISFIED | KindVersion=1, KindEnd, Family slots; legality rejects unknown kinds fail-closed (TestLegalityRejectsUnknownKind) |
| IR-04 | SATISFIED-WITH-GAP | All mutation flows through MutContext choke points (no public mutators on Base) — the discipline exists and is enforced; BUT 2 of 5 choke points (AddUse/RemoveUse) are defective (the gap). The sanctioned-path requirement holds; the path itself is broken at two points. |
| IR-05 | SATISFIED | per-graph monotonic IDs, tombstone on Erase, no reuse; determinism tests |
| IR-06 | SATISFIED | Loc{Span, PassOrigin} on every node; PassOrigin stamped from construction; Span zero by design until frontend |
| IR-07 | SATISFIED | stage-centered WindowFor table, PM-checked after every pass, fail-closed for unset stages |
| PASS-01..06 | SATISFIED | one row each: preconditions / callback instrumentation / invalidate-all cache / flags.F / read-only analysis skeleton / verifier diagnostics naming node+pass+invariant |
| TOOL-01..06 | SATISFIED | diagnostics / slog facade / EmitDot / canonical printer / golden harness / dump+timing+verify-each flags wired to real emitters |
| FE-01 | SATISFIED | pkg/ builder; matmul_scale oracle byte-matches parser target golden |
| VER-08 | SATISFIED | TestDeterminism -count=10 + TestDeterminismDot + cross-process golden anchor |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| internal/ir/node.go | 168, 177-190 | `ops []Use` by value; append realloc (CR-01) | Blocker (latent) | first AddUse on a full-capacity node strands all def use-list pointers; zero call sites today |
| internal/ir/mutate.go | 47-79 | RemoveUse clears record after slices.Delete shift (CR-02) | Blocker (latent) | removing a non-last slot nils a live, correctly-wired operand; zero call sites today |
| internal/ir/construct.go | 223-236 | wireOperands silently discards surplus operands (WR-01) | Warning | fail-open construction path; verifier-invisible; over-supply becomes silent miscompilation — contradicts the fail-closed design contract (IR-03 spirit) |
| internal/ir/print.go | 59-65 | FormatConst renders int-typed constants at float32 precision (WR-02) | Warning | distinct large i32/u32 constants print identically; no current consumer prints large ints, so goldens unaffected today |
| internal/ir/verify.go | 159-165 | checkDefUse message names the def twice (IN-01) | Info | human text only; structured NodeID correct |
| internal/ir/verify.go + internal/pass/instrument.go | 23 / 56 | two violation renderings disagree on quote style (IN-02) | Info | golden fragility risk if violation text is ever golden-pinned |
| internal/visualization/dot.go | 32 | graph name interpolated unquoted (IN-03) | Info | invalid DOT only for non-C-identifier names; hardening |
| internal/ir/use_test.go + internal/golden/golden.go | — | AddUse/RemoveUse untested (IN-04a); firstDiff dead branch (IN-04b) | Info | the coverage hole is exactly why CR-01/02 survived a green suite; folded into the gap's missing-work |
| internal/ir/print.go, pkg/chain.go | — | 565 / 551 lines | Info | exceeds the 200-400 "typical" band, under the 800 hard max |

### Code-Review Criticals: Gap vs. Pre-Phase-2 Debt — Explicit Ruling

Both criticals are **confirmed real in the delivered code** (not merely claimed): the by-value `ops []Use` slice, the `append` in `addSlot`, and the `slices.Delete` in `removeSlot` are all present verbatim, and `AddUse`/`RemoveUse` have zero call sites anywhere in the tree. The distinction:

- **As impact on shipped behavior, they are pre-Phase-2 debt.** Nothing in Phase 1 calls them; the flagship oracle, the PM, the printer, and the builder are all unaffected; the suite is green for real reasons.
- **As a phase-goal matter, they gap THIS phase and are scored as the single gap above.** The phase goal promises a *verified* substrate, and 01-01's must-have truth 1 is verbatim "chains stay consistent through **every** choke-point mutation" — the plan's own contract names AddUse/RemoveUse. A sanctioned mutation API whose first real use (any Phase 2/3 transform pass) corrupts the IR is a substrate defect discovered in Phase 1's deliverable, discovered by Phase 1's own code review, and cheapest to fix now (the review supplies the exact fix sketch and test shapes). No later roadmap phase schedules the fix; Phases 3-4's lowering/DCE passes are the first call sites. Fixing it in a gap-closure pass before Phase 2 keeps the "verifier catches corruption" promise honest instead of relying on post-hoc violation reports.

WR-01/WR-02 are **not** scored as goal gaps: they have no current behavioral impact (builder passes exact arity; no large integer constants are printed) and no failed must-have truth maps to them. They are recorded as warnings with a recommendation to fix alongside the gap closure (WR-01's fail-open construction path is the same class of defect as the criticals; WR-02 becomes live the moment the first large int constant prints).

### Human Verification Required

None. Every must-have is programmatically testable and was verified by running code; the only non-machine-checkable surface (visual .dot rendering) is explicitly dev-only per the research (no graphviz binary required for any gate).

### Gaps Summary

One grouped gap, two manifestations, single root cause: the intrusive `Use` records lack stable identity under operand-list growth/shrink. RAUW, SetDef, and Erase — the three choke points Phase 1 actually exercises — are proven correct; AddUse and RemoveUse corrupt the chains they exist to maintain. The fix is small and fully specified by the review (pointer storage + clear-before-drop + two round-trip white-box tests). All 22 requirement IDs are otherwise accounted for and evidenced; the phase's flagship outcomes (doc-pinned oracle byte-match, empty-pipeline verification, -count=10 determinism, stdlib-only module) are genuinely achieved.

---

_Verified: 2026-10-16_
_Verifier: Claude (gsd-verifier)_
