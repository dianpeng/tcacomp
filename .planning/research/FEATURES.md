# Feature Research

**Domain:** GPU kernel compilers / tensor DSLs (Triton, CUDA C++, Halide, TVM/MLIR, Taichi, scalfold)
**Researched:** 2026-10-15
**Confidence:** MEDIUM-HIGH (ecosystem facts verified against 2024-2025 sources; gap analysis is first-party, read directly from `doc/syntax.md` and `doc/tca.md`)

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist in any compiler/DSL whose selling point is performance with correctness. Missing these = the tool is untrustworthy, regardless of kernel speed.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Canonical **textual IR printer** | Every serious compiler (MLIR, Triton, Halide, Taichi) has one; it is the substrate for golden tests, bug reports, and pass dumps. `tca.md` §9 shows a textual notation but no printer is ever specified as a component. | LOW | Must be deterministic (stable node/axis ordering) so golden files diff cleanly. |
| **Golden IR regression tests** | LLVM FileCheck convention (`CHECK:`/`CHECK-NOT:`/regex captures) is the industry standard; Triton, TVM, MLIR all test passes this way. `syntax.md` §11.3 lists behavioral tests but no IR-snapshot mechanism. | LOW-MEDIUM | Golden files + an update script (LLVM's `update_test_checks.py` pattern); pattern matching, not exact diffs. |
| **Pass dump / trace flags** | Triton: `MLIR_ENABLE_DUMP=1` (IR before every pass), `MLIR_DUMP_PATH`, `TRITON_KERNEL_DUMP`. MLIR: `--mlir-print-ir-before/after=<pass>`, `--mlir-print-ir-after-change`, `--pass-timing`. PROJECT.md says "global flags, logging" but specifies nothing. | LOW | Needs: `--dump-after=<pass>`, `--dump-graphviz`, `--pass-timing`, `--verify-each/--no-verify`, dump directory. Build into pass manager from day one. |
| **Verify-after-every-pass toggle** | MLIR runs the verifier after each pass by default (`--verify-each`); this is how compiler bugs are caught at the pass that introduced them, not downstream. TCA plans per-pass verifiers (PROJECT.md) but no global toggle to run the pipeline without them (needed for debugging the verifier itself). | LOW | `--verify-each=0` escape hatch, MLIR convention. |
| **Numeric tolerance verification harness** | The ecosystem rule (2025 GPU-testing literature): sanitizer + numerical comparison against a *higher-precision* reference, over multiple shapes/dtypes/seeds — one-shape/one-seed testing caught 0/10 realistic kernel bugs in one study. TCA's core value IS numerical equivalence to a CPU reference, but neither doc specifies tolerances, reference precision (f64 reference for f32 kernels is the standard trick), or fuzzing dimensions. | MEDIUM | Specify: `rtol`/`atol` policy per element type, f64 CPU reference, N==0 + non-divisible extents + seed sweep. Harness lives outside the compiler but the flagship gate depends on it. |
| **compute-sanitizer stage in e2e tests** | Standard practice: racecheck/synccheck/initcheck/memcheck run as a dedicated pre-trust validation stage, because races and uninitialized reads silently corrupt both results and timings (scalfold's own methodology notes say: sanitize *before* trusting any timing). For a compiler whose correctness claim is race-freedom-by-construction, sanitizers are the *independent check* that the construction argument and codegen actually hold. PROJECT.md says sanitize "may be used for debugging" — too weak. | LOW | External tool; wrap e2e runs with small shapes under `compute-sanitizer --tool racecheck` / `synccheck` / `memcheck`. Make it a harness mode, not a manual step. |
| **Pinned FP policy for generated code** | Triton's biggest reproducibility scandal: `tl.dot` defaults to TF32 on NVIDIA and silently ignores framework-level IEEE switches (verified: 9/9 cases bit-matched TF32, 0/9 matched IEEE). Lesson: **never let a precision-reducing decision be silent.** `syntax.md` §9.1 says FMA contraction "follows the compiler's numerical policy" — but the policy is never pinned. nvcc defaults (`-fmad=true`, `-ftz=false`) then decide silently. | LOW | Pin in doc: default IEEE (no fast-math, no ftz), document whether fmad contraction is permitted (recommend: permitted, documented, since it is per-op IEEE-exact). |
| **Diagnostic severity channels** | MLIR has `MLIR_ENABLE_DIAGNOSTICS=warnings|remarks|stacktraces`; Halide and Clang distinguish errors/warnings/remarks. `syntax.md` §10 specifies only error messages (a strong table, to its credit) plus one uncoalesced-store performance warning in `tca.md` §2.7. | LOW | Add warning + remark severities to the diagnostic package; performance remarks ride the same channel. |
| **One-command end-to-end UX** | The flagship gate (`matmul_scale` correct on GPU vs CPU ref) requires a driver that: runs the Go compiler → nvcc-builds emitted `.cu` + runtime → runs on GPU → compares vs reference. Neither doc specifies this driver; the compiler "does not drive nvcc" (fine) but *someone* must orchestrate or the success gate is manual. | MEDIUM | A Go test driver (stdlib `os/exec` → cmake/ninja) + a documented recipe. Keep the compiler itself nvcc-free. |
| **Structured error for launch-time shape validation** | `syntax.md` §10 requires dynamic-equality failures to "produce a shape error" before launch. Ecosystem norm (Triton launch assertions, CUDA error strings): named error, expected vs actual extents, which parameter. | LOW | Runtime throws `tca::ShapeError{param, expected, actual}`; docs specify the contract but not the error surface. |

### Differentiators (Competitive Advantage)

Not expected, but they are where TCA can be better than Triton/Halide/Taichi rather than merely comparable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Race-freedom proven by construction + verifier-checked** | Triton, TVM, Halide, Taichi give you compute-sanitizer and hope. TCA's IR structurally excludes races (structural stores, reducer-only cross-lane flow, CTA-uniform collectives) and *verifies the invariants after every pass* (`tca.md` §5.7.8). Almost no production DSL states and machine-checks this argument. | HIGH (already designed) | The flagship trust story. Publish the invariant list as a doc section; the verifier checks are the enforcement. |
| **Run-to-run bitwise determinism for a fixed compiled kernel** | With no atomics, no data-dependent control flow, and fixed schedules, TCA kernels are deterministic by construction — Triton cannot promise this (atomics, autotuned split-K change reduction trees), PyTorch needs `use_deterministic_algorithms()` workarounds. State it explicitly in docs; it converts the no-control-flow restriction from a limitation into a guarantee. | LOW | Free given v0 design; just document. Distinct from cross-schedule reproducibility, which reassociation forbids (docs already honest about this). |
| **Once-per-logical-output seed semantics** | Nonzero seeds and non-identity combiners (softmax's `-inf_f32` max-reduce is identity-seeded, but `init = 1.0f` sum is not) are mishandled by reducers that assume identity partials. `tca.md` §5.10 mandates validity tags and once-per-output init application — a correctness edge most DSL runtimes get subtly wrong. | MEDIUM (already designed) | Enforce with the nonzero-seed tests `syntax.md` §11.3 already lists. |
| **IR-level reference interpreter (Go)** | Triton's `TRITON_INTERPRET=1` (NumPy CPU simulation) is beloved precisely because it gives an oracle with zero GPU involvement, steppable with pdb. TCA's IR is tiny (map/reduce/load/store/iter + typed scalar expressions, one basic block) — a Go interpreter is cheap and doubles as: nvcc-free CI oracle, documentation, fuzz target. No competitor in this niche has it. | MEDIUM | Consume the elaborator + shared expression registry (already required by `syntax.md` §11.2). Schedule after e2e works, not before. |
| **Source-span → generated CUDA line mapping** | 2025 work (Triton-San, LLVM Dev Mtg) shows debug-info preservation through MLIR passes is a live, unsolved pain. TCA generates the CUDA itself: emitting `#line` directives into the `.cu` (so nvcc errors and sanitizer reports map to recognizable IR constructs, and ideally back to frontend spans) is a one-passenger feature that pays off forever. Cheap now, painful to retrofit (requires loc fields on IR nodes from day one). | MEDIUM | Design decision to make at IR-core phase: every IR node carries a `Loc{frontend span, pass origin}`. |
| **Performance remarks from the verifier** | `tca.md` §2.7 rule 6 already mandates uncoalesced-store warnings. Shipping *any* perf diagnostics in v0 puts TCA ahead of most v1 DSLs; LLVM-style remarks (`-Rpass`) are the model. | LOW | Same channel as diagnostic severities (table stakes row above). |
| **Structural axis rebinding with extent validation** | `syntax.md` §4.1/§5.5: attaching different axis names to a named tensor's positional dims, validated. Triton has no axis algebra at all; Halide has full scheduling but no named-axis type checking. | MEDIUM (already designed) | Keep; it is a genuine usability wedge for the map/reduce model. |
| **Whole-warp-only scheduling, compiler-owned** | Triton exposes block/thread program ids and lets users hang themselves; scalfold exposes storage tiers explicitly. TCA's "whole warps, preset table, users never bind threads" (`tca.md` §5.7) is the most conservative GPU scheduling contract in the ecosystem — sellable as the safe-by-default option. | HIGH (already designed) | The `select(StatementCluster) -> Assignment` interface (`tca.md` §5.7.6) already isolates the future cost-model upgrade. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that are table stakes *elsewhere* but would damage TCA's v0 thesis, plus classic traps.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Autotuning / cost-model search** | Triton's `@triton.autotune` is its headline feature; users will ask for it immediately. | A search loop multiplies compile time, makes results non-reproducible across machines, and needs a working cost model before the compiler is even correct. v0's fixed preset table is a deliberate, defensible position. | Keep the `select()` interface (`tca.md` §5.7.6) so a solver drops in later without touching lowering passes. Add presets, not search. |
| **JIT / embedded compilation (in-process kernel compile+launch)** | Triton and Taichi feel magical because kernels compile inside the host program. | TCA's compiler is a separate Go binary emitting `.cu`; JIT would demand cgo or an embedded toolchain, breaking the "compiler does not drive nvcc" decision and the pure-Go constraint. | Offline compile + CMake runtime; the host launch stub is the integration point. scalfold already owns the embedded-C++ niche — do not compete with your own reference scaffold. |
| **Python bindings / embedding** | Every GPU DSL ships Python first. | Same coupling problem as JIT, plus a second language runtime to maintain. v0 users are C++ host programs. | C ABI host stubs; language bindings can come much later as thin FFI. |
| **Tensor-core intrinsics (mma/WMMA) or TF32 paths in v0** | "The GPU has tensor cores; use them." | Introduces exactly the silent-precision-cliff failure mode that makes Triton kernels irreproducible (verified above). FMA-based matmul on CUDA cores is IEEE-per-op and matches the determinism story. sm_86 supports mma, but the structural map+reduce model doesn't need it for v0 correctness. | FMA lowering now; an *explicit, opt-in* tensor-core lowering later, with its precision contract spelled out per-op. |
| **Any silent precision-reducing transform** | Fast-math, ftz, TF32 all buy speed. | One silent default destroys the reproducibility guarantee and the trust story; debugging "wrong at the 4th decimal" reports costs more than the speed ever earns. | Explicit is the only acceptable mode: every precision-affecting choice is a documented flag or an explicit cast/seed in source. |
| **User scheduling annotations** | Power users want `tcache`/`tune`-style pragmas (Halide's schedule DSL is the famous version). | Violates the compiler-owns-scheduling thesis; every annotation is a correctness surface (wrong SMEM hint = race) the structural model exists to eliminate. | The preset table + eventual solver. Users express *what*, never *where*. |
| **Device-side print/assert in kernels** | Triton has `tl.device_print`/`tl.device_assert`; invaluable in that model. | TCA kernels have no user control flow and no invalid points; a device printf ABI is runtime complexity for a problem the reference interpreter and IR dumps solve better in this model. | Interpreter + per-pass dumps + sanitizer stage. Revisit only if a real debugging dead-end appears. |
| **Multi-GPU, streams, CUDA graphs, cluster launch, cooperative grid sync** | Production inference wants all of these. | Each breaks the conservative single-kernel-cluster execution contract (`tca.md` §5.1) and the race-freedom argument in a different way. | Post-v0 execution-interface milestone, one at a time, each with its own invariant story. |
| **Grammar fuzzing / mutation testing infra** | LLVM-grade robustness programs have fuzzers. | v0 grammar is ~50 productions and hand-audited; fuzz infra is a project in itself. | Defer to v1.x; the parser's semicolon/brace recovery (`syntax.md` §11.2) plus the diagnostic table is adequate for v0. |
| **Full IEEE f64 support** | Scientific users ask. | Doubles the expression-registry, reducer-specialization, and runtime matrix; v0 deliberately excludes it (`syntax.md` §6.3). | Use f64 only in the *CPU reference harness* for tolerance headroom — free and standard practice. |

## Gap Analysis: What the Specs Are Missing

First-party read of `doc/syntax.md` + `doc/tca.md` + PROJECT.md against the landscape above. Status legend: **PRESENT** (spec covers it, at or above ecosystem bar), **IMPLIED** (necessary for stated goals but never specified), **MISSING** (table-stakes gap), **CONFLICT** (doc ambiguity to resolve before implementation).

| Item | Status | Evidence | Action |
|------|--------|----------|--------|
| Source spans in diagnostics | PRESENT | `syntax.md` §10: "identify the source span, the actual stage type/order, and the expected type/order" | None. Ahead of most v0 specs. |
| Required-diagnostics message table | PRESENT | `syntax.md` §10 gives 16 concrete messages | None. Treat as golden-test input. |
| IR verifier per transformation pass | PRESENT | PROJECT.md + `tca.md` §5.7.8 invariant list | None. The invariant list should become a published doc section (trust story). |
| Graphviz IR visualization | PRESENT | PROJECT.md requirements | None. Ensure it shares the deterministic printer's ordering. |
| Textual IR printer as a component | IMPLIED | `tca.md` §9 notation exists, but no printer/round-trip spec | Specify: deterministic ordering, comment/annotation syntax, golden-test format. Phase 1. |
| Pass dump/trace/verify flags | MISSING | "global flags, logging" only | Add flag set: `--dump-after`, `--dump-dir`, `--pass-timing`, `--verify-each`, `--dump-graphviz`. Phase 1, inside pass manager. |
| Tolerance/allclose verification policy | MISSING | Core value is numeric equality; no tolerances specified | Specify per-type rtol/atol + f64 CPU reference + multi-shape/seed matrix. Test-harness phase. |
| compute-sanitizer as required stage | MISSING (downgraded to "may") | PROJECT.md Context: "may be used for debugging" | Make racecheck/synccheck/memcheck harness modes run in CI; small shapes. Cheap, high trust value. |
| FMA contraction / FTZ / fast-math policy | CONFLICT | `syntax.md` §9.1 punts to "the compiler's numerical policy"; policy never stated | Pin before codegen phase: recommend IEEE default, fmad contraction explicitly permitted-and-documented, ftz off, no fast-math. One paragraph in tca.md §4 or syntax.md §6. |
| Run-to-run determinism statement | MISSING (free win) | Docs acknowledge cross-schedule bitwise drift (`syntax.md` §5.3) but never claim fixed-binary determinism | Add one sentence: "for a fixed compiled kernel, TCA execution is deterministic (no atomics, no data-dependent scheduling)." Trust story. |
| `#line` / loc tracking in generated CUDA | MISSING | Neither doc mentions debug info | Decide at IR-core phase: `Loc` field on IR nodes; emit `#line` in codegen. Retrofit cost is high. |
| CLI surface | MISSING | `cmd/` in layout; no flags spec | `--emit-ir`, `--emit-cuda`, `--dump-*`, `--preset`, `-o`; version flag. Phase with parser. |
| e2e driver orchestration | IMPLIED | Flagship gate requires compile→nvcc→run→compare; compiler won't drive nvcc | Go test driver via os/exec + a documented one-command recipe. Test-harness phase. |
| Numeric harness reference mechanism | IMPLIED | "CPU reference" without saying who computes it | f64 reference loop in the C++ harness (or Go interpreter later); document comparison policy. |
| Empty-extent semantics | PRESENT | `syntax.md` §5.3/§9.8/§11.3: empty reduction = seed, empty output = no invocations | None. Better specified than most ecosystem docs. |
| Warning/remark severity | MISSING | Error diagnostics + one perf warning; no severity model | Two extra severity levels in the diagnostic package. LOW. |

## Feature Dependencies

```
Textual IR printer ──requires──> deterministic node ordering (IR core)
Golden IR tests ──requires──> textual printer + pass-dump flags
Pass dump/trace flags ──requires──> pass manager with named passes (already planned)
`--verify-each` toggle ──requires──> verifier-as-pass registration (already planned)
e2e driver ──requires──> compiler CLI (emit .cu) + nvcc/cmake toolchain
compute-sanitizer stage ──requires──> e2e driver + small-shape test mode
Numerical harness ──requires──> e2e driver + runtime launch stubs (shape errors surface here)
`#line` mapping ──requires──> Loc on IR nodes ──MUST land in IR-core phase (retrofit = painful)
Reference interpreter ──requires──> elaborator + shared expression registry (already first)
Determinism guarantee ──requires──> pinned FP policy + fixed schedule (preset table, no atomics in v0)
```

### Dependency Notes

- **Golden IR tests require the textual printer:** without a canonical printer there is nothing to diff; this is why the printer belongs in the IR-core phase, not "when needed".
- **`#line` mapping requires Loc at IR-core phase:** adding location fields to every node after three phases of code is a cross-cutting rewrite; deciding "no locs" is also fine, but it must be a *decision*.
- **compute-sanitizer stage requires the e2e driver:** the driver is the integration point; sanitizer wraps driver runs. Build driver first.
- **Reference interpreter requires the elaborator only:** it does not require the parser or codegen; it can be built in parallel with codegen and becomes the cheapest possible oracle.
- **Determinism guarantee conflicts with future autotuning:** the day a cost-model search lands, per-machine schedules may differ; the promise must then be scoped to "fixed compiled artifact" (which remains true). Document the scope now.

## MVP Definition

### Launch With (v1)

Minimum for the flagship gate (`matmul_scale` correct end-to-end on A6000 vs CPU reference) to be trustworthy rather than anecdotal.

- [ ] **Textual IR printer** (deterministic ordering) — substrate for everything below
- [ ] **Pass dump/trace flags + `--verify-each` toggle** — per-pass visibility, MLIR/Triton convention
- [ ] **Golden IR regression tests** for simplification, merge, fusion, effect linearization, and each preset lowering — FileCheck-style patterns, auto-update script
- [ ] **Pinned numeric policy** — one doc paragraph: IEEE default, fmad documented, ftz off, no silent precision reduction
- [ ] **e2e driver** — compiler → nvcc build → GPU run → f64 CPU reference → allclose with per-type tolerances; multi-shape (N==0, non-divisible, tiny, realistic) × multi-seed matrix
- [ ] **compute-sanitizer harness mode** — racecheck/synccheck/memcheck over the e2e suite at small shapes
- [ ] **CLI surface** — `--emit-ir/--emit-cuda`, `--dump-*`, `--preset`, `-o`, `--version`

### Add After Validation (v1.x)

- [ ] **IR reference interpreter (Go)** — nvcc-free oracle, fuzz target, doc-by-execution; trigger: first debugging session where you wish you had it
- [ ] **`#line` emission in generated CUDA** — trigger: first sanitizer report you can't map back; requires the Loc decision made (even if negative) at IR-core time
- [ ] **Performance remarks channel** (`-Rpass` style) — trigger: first user asking "why is kernel X slow"
- [ ] **Diagnostic severity model (warnings/remarks)** — trigger: uncoalesced-store warnings outgrow the error channel
- [ ] **Grammar fuzzer** — trigger: parser survives its first real user

### Future Consideration (v2+)

- [ ] **Autotuning/cost-model solver behind `select()`** — only after presets are proven and reproducibility scope is documented
- [ ] **Tensor-core lowering, opt-in** — with an explicit per-op precision contract
- [ ] **Multi-GPU / streams / CUDA graphs** — each is its own execution-interface milestone with its own invariant argument
- [ ] **Language bindings (Python/Rust FFI)** — thin wrappers over the C ABI stubs
- [ ] **JIT mode** — only if offline compilation proves to be the adoption blocker

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Textual IR printer | HIGH | LOW | P1 |
| Golden IR tests | HIGH | LOW | P1 |
| Pass dump/trace flags | HIGH | LOW | P1 |
| `--verify-each` toggle | MEDIUM | LOW | P1 |
| Pinned numeric policy doc | HIGH | LOW | P1 |
| e2e driver + tolerances | HIGH | MEDIUM | P1 |
| compute-sanitizer harness mode | HIGH | LOW | P1 |
| CLI surface | MEDIUM | LOW | P1 |
| `#line`/Loc decision (even if "no") | HIGH | LOW (decision) / MEDIUM (impl) | P1 — decision at IR-core phase |
| Warning/remark severities | MEDIUM | LOW | P2 |
| IR reference interpreter | HIGH | MEDIUM | P2 |
| Perf remarks (`-Rpass`) | MEDIUM | LOW | P2 |
| Grammar fuzzer | LOW | MEDIUM | P3 |
| Autotuning | HIGH | HIGH | P3 (explicitly deferred, interface-ready) |
| Tensor-core lowering | MEDIUM | HIGH | P3 |
| JIT / Python bindings | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch (flagship gate trustworthiness)
- P2: Should have, add when possible
- P3: Nice to have / explicitly deferred anti-features for v0

## Competitor Feature Analysis

| Feature | Triton (2025) | Halide | Taichi | scalfold (reference) | Our Approach |
|---------|---------------|--------|--------|----------------------|--------------|
| IR dump per pass | `MLIR_ENABLE_DUMP`, `MLIR_DUMP_PATH`, kernel-name targeting | `HL_DEBUG_CODEGEN=1/2`, HTML lowered-stmt dumps | `ti.init(print_ir=True)` + 6 stage-specific flags | none (library, not compiler) | Match: named-pass dumps + Graphviz + timing |
| Verifier discipline | MLIR `--verify-each` (default on) | internal checks | partial | contract guards (always on, not NDEBUG-gated) | Exceed: per-pass verifiers + published invariant list |
| Race checking | user's job (compute-sanitizer docs) | user's job | user's job | methodology note: sanitize before timing | Exceed: structural exclusion + verifier + sanitizer CI stage |
| Reference/oracle mode | `TRITON_INTERPRET=1` NumPy CPU sim | `compile_jit` + tracing | CPU arch fallback | CPU reference loops in examples | Match then exceed: Go IR interpreter (no GPU, no nvcc) |
| Determinism story | none (TF32 defaults, atomics, autotune) | schedule-dependent | weak | n/a (single-writer views) | Exceed: fixed-binary bitwise determinism, documented |
| Numeric policy traps | TF32 default silently ignores framework switches (verified) | fast-math off by default | n/a | storage/compute precision boundary, one rounding per store | Pin policy; no silent precision reduction, ever |
| Debug info to GPU | line info on by default; Triton-San (Oct 2025) still working on precise mapping | none notable | limited | none | `#line` into generated `.cu` (cheap, ours-generated code) |
| Scheduling | user-exposed program ids + autotune | full user schedule DSL | compiler-decided mostly | explicit generators/storage by user | Compiler-owned presets only; whole warps; `select()` interface for a future solver |
| End-to-end UX | JIT in Python, cache dir, knobs | JIT + AOT compile | JIT | header-only library + CMake | Offline compiler → `.cu` + runtime + host stubs; driver orchestrates nvcc |

## Sources

- [Triton README — tips for hacking (env var flags)](https://github.com/triton-lang/triton/blob/main/README.md) — HIGH (official)
- [Triton Compiler Development Tips — Lei Zhang](https://www.lei.chat/posts/triton-compiler-development-tips/) — MEDIUM (verified against README)
- [MLIR Pass Management docs](https://mlir.llvm.org/docs/PassManagement/) — HIGH (official): print-before/after, verify-each, verify-diagnostics, pass-timing
- [MLIR Debugging docs](https://mlir.llvm.org/getting_started/Debugging/) — HIGH (official)
- [Debugging Triton — official docs](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html) — HIGH: `TRITON_INTERPRET=1` interpreter semantics
- [Triton troubleshooting/debugging reference (TensorRT-LLM)](https://github.com/nvidia/tensorrt-llm/blob/main/.claude/skills/kernel-triton-writing/references/troubleshooting.md) — MEDIUM
- [Halide tutorial: inspecting generated code](https://halide-lang.org/tutorials/tutorial_lesson_03_debugging_1.html) — HIGH (official): `HL_DEBUG_CODEGEN`, `compile_to_lowered_stmt`, `print_loop_nest`
- [Taichi debugging docs](https://docs.taichi-lang.org/docs/debugging) and [development tips](https://docs.taichi-lang.org/docs/development_tips) — HIGH (official): `print_ir=True` etc.
- [MLIR nondeterminism mitigation paper, arXiv 2609.25624](https://arxiv.org/html/2609.25624v1) — MEDIUM (preprint, but the TF32-default findings are consistent with Triton docs and issue trackers): TF32 default, framework switches don't propagate, IEEE fixes cross-arch reproducibility at 2.4-5.6x cost
- [Triton `tl.dot` docs](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) — HIGH: input_precision default tf32
- [NERSC compute-sanitizer guide](https://docs.nersc.gov/tools/debug/compute-sanitizer/) — HIGH (official-ish): tool suite (memcheck/racecheck/initcheck/synccheck)
- [GPU testing robustness study (single-shape/single-seed catches 0/10 bugs)](https://svac-llc.com/case-studies/gpu-race-conditions-low-precision.html) — LOW-MEDIUM (vendor case study; consistent with ecosystem practice)
- [LLVM Testing wiki (FileCheck conventions)](https://factory.com/open-source-wikis/llvm-project?page=how-to-contribute%2Ftesting.md) — HIGH
- [Triton-San, LLVM Dev Meeting Oct 2025](https://llvm.org/devmtg/2025-10/slides/technical_talks/lu.pdf) — MEDIUM: debug-info preservation through MLIR is an open problem
- scalfold README (`/home/ps/Workspace/cuda/project/scalfold/README.md`) — HIGH (first-party): methodology notes (sanitize before timing, cudaEvents, nsys/ncu)
- doc/syntax.md, doc/tca.md, PROJECT.md (this repo) — HIGH (first-party, authoritative for gap analysis)

---
*Feature research for: GPU kernel compiler / tensor DSL (TCA/tcacomp)*
*Researched: 2026-10-15*
