# Domain Pitfalls

**Domain:** GPU kernel compiler / DSL — pure-Go compiler emitting CUDA C++ + C++20 runtime, single-tier SSA IR, warp-based reductions, SMEM tile caches, correctness-first phasing (RTX A6000, sm_86, CUDA 12.2)
**Researched:** 2026-10-15
**Confidence:** HIGH for CUDA hardware/runtime semantics (verified against NVIDIA docs, programming guide, and multiple corroborating bug reports on identical hardware); MEDIUM-HIGH for compiler-construction and process pitfalls (verified against Go compiler SSA source docs and community practice, applied to this project's specific IR design)

---

## Critical Pitfalls

Mistakes that cause wrong results, deadlocks, or rewrites.

### P1: `__shfl_*_sync` mask/predicate bugs with out-of-range lanes

**What goes wrong:** A warp-shuffle reduction tree uses `__shfl_down_sync(0xffffffff, v, offset)` but some lanes have exited early, are behind a divergent branch, or hold garbage in out-of-range accumulator slots. On Volta+ (sm_86 included) this is undefined behavior: lanes outside the named mask return **undefined values**, and if any masked lane never reaches the intrinsic, the warp hangs. Alternatively the tree consumes the *identity* (0) of a lane that owns no elements, silently corrupting min/max-style reductions, or duplicates a lane's value into multiple tree slots.

**Why it happens:** Hand-written reduction examples on the internet uniformly assume `threadIdx.x < N` guards, which this project explicitly forbids (tca.md §5.7.8 invariant 5). The correct mental model — "all 32 lanes execute the shuffle; only the *contribution* is predicated" — is easy to violate during codegen when the `active` predicate from the predicated tail leaks into the collective.

**Consequences:** Silent numerical corruption (reads of undefined shuffle results often *look* like plausible floats) or intermittent deadlock that passes small tests and hangs on larger shapes.

**Prevention:**
- The runtime reducer owns the entire shuffle tree (tca.md §5.10); codegen never emits a shuffle. The reducer contract: every lane calls `Init`/`Reduce`/`Combine` unconditionally; lane *l* contributes its private accumulator only if `l` owns ≥1 valid element, otherwise contributes the declared identity **or** uses validity tags (tca.md §5.10 "Initialization semantics" — for non-identity initializers, validity tags are mandatory; an identity contribution of `0` is wrong for `init=5` or for `min`).
- Full mask `0xffffffff` is correct **only because** whole-warp groups guarantee all lanes reach the collective. The moment a future `(warp, lane)` mapping or sub-warp feature is added, masks must be recomputed from program logic (`__ballot_sync` before the branch), never `__activemask()`.
- Out-of-range shuffle *sources* within the tree are safe only when each lane pre-initializes its accumulator to the identity (wrap-around semantics of `__shfl_down_sync` return the lane's own value for out-of-range `lane+delta`); never let a lane shuffle a value it never initialized.
- Test with `compute-sanitizer --tool synccheck` plus a reducer fuzz test where every lane's element count varies 0..3 (strided tail coverage).

**Warning signs:** Reduction results that change when the extent is not a multiple of 32·group size; tests that hang only for certain grid sizes; sanitizer `Barrier error` / invalid mask reports.

**Phase:** 3 (parallelization + reduce lowering). The naive scalar reducer in phase 2 must already define the Init/Reduce/Combine contract so phase 3 only swaps the collective implementation.

---

### P2: `__syncthreads()` in divergent or non-uniform-trip-count code

**What goes wrong:** A cooperative tile load or cross-warp combine calls `__syncthreads()` inside a loop whose trip count differs across threads (classic `for (i = tid; i < n; i += blockDim.x)`), or behind a predicate that some threads don't take. Result: deadlock — or, worse on some driver/arch combos, silent completion with corrupted SMEM.

**Why it happens:** Pre-Volta CUDA synchronized at warp granularity, so `if (tid < n) { ...; __syncthreads(); }` *appeared* to work in old code samples. Volta's independent thread scheduling (sm_86) makes the barrier truly block-wide: it "will not succeed until reached by all non-exited threads in the block." This is exactly the bug class tca.md §5.12's predicated-tail codegen exists to prevent — but it only takes one hand-emitted early exit or one runtime helper with a data-dependent loop to reintroduce it.

**Consequences:** Kernel hangs (worst case: machine-wide GPU wedge requiring reset), or flaky corruption that depends on warp scheduling.

**Prevention:**
- Hard rule in codegen and runtime: **all loops containing a barrier have CTA-uniform trip counts; all predication is in the body.** tca.md §5.12 already mandates this shape (`for (base = 0; base < T; base += blockDim.x)` + `active` predicate). Enforce it with a runtime assertion in debug builds: record `T` on first entry and trap on divergence.
- Runtime tile helpers take the *tile extent*, never `threadIdx`-derived bounds, and compute iterations from `blockDim` themselves — codegen cannot accidentally pass a non-uniform bound.
- No early `return` anywhere in generated device code before all collectives complete (tca.md's no-user-control-flow model makes this enforceable: the verifier rejects any IR that could produce an exit path; there should be exactly one function exit, after the last collective).
- CI: `compute-sanitizer --tool synccheck` on every e2e test, not just debug runs.

**Warning signs:** Tests hang at specific extents (non-multiples of 256); hangs disappear when a `printf` is added (scheduling perturbation); sanitizer reports "invalid __syncthreads()".

**Phase:** 2 (runtime cooperative load/store + naive codegen loop shapes) — the uniform-trip-count discipline must be established before phase 3 multiplies the number of collectives.

---

### P3: Numerical comparison flakiness in e2e verification (bitwise vs tolerance)

**What goes wrong:** The e2e harness compares GPU output against the CPU reference with `memcmp` or an absolute epsilon. Phase 3 lands: the warp-shuffle tree sums in a different order than the sequential CPU loop, fp addition is non-associative, and every reduction test now "fails" — or worse, someone "fixes" the kernel to match the CPU order, destroying the parallelization. Conversely, a loose tolerance masks real bugs (a race that only perturbs the last ULP, a missing element in a strided tail).

**Why it happens:** The project ships both sides: nvcc-compiled GPU code and a g++ CPU reference. Two independent reordering sources are guaranteed: (1) parallel reduction reassociation (tree vs sequential), which tca.md §2.5 explicitly blesses ("results may differ bitwise between parallelization strategies"); (2) **FMA contraction** — nvcc contracts `a*b+c` into `fmaf` by default (`--fmad=true`), and g++ on the CPU side may or may not contract depending on flags, so even elementwise `matmul_scale` bodies differ in the last ULP.

**Consequences:** Either a false-failure crisis at the phase 2→3 boundary that erodes trust in the harness, or tolerance so loose it never catches a predicated-tail bug that drops one element out of 10,000.

**Prevention:**
- Define comparison policy **per operator class, in the harness, in phase 2** (before parallelization exists):
  - `max`/`min`/`bitwise` ops: bitwise equality is valid and should be demanded (they are exactly associative).
  - fp `+`/`*` chains: relative tolerance ~1e-5 for f32 with well-conditioned test inputs, with an explicit bound documented in the test (e.g., `N·ε·max|x|`). Never absolute epsilon alone.
  - Phase 2 (naive sequential codegen) *can* be bitwise against the CPU reference **if and only if** FMA contraction is controlled: compile the generated `.cu` with `--fmad=false` for the naive-codegen test lane (keep default `--fmad=true` for performance lanes), or accept tolerance from day one. Decide this in phase 2 and write it down — retrofitting tolerance policy mid-project is how flakiness gets in.
- Keep test inputs small and well-conditioned (lengths ≤ a few thousand, values in [0,1] or similar) so the non-associativity bound is tight enough to catch a dropped element (a dropped element is an O(1/N) relative error, far above 1e-5 for small N).
- Add one deliberate associativity canary: a reduction test with known pathological input (alternating huge/tiny magnitudes) documented as "expected to differ from CPU; check the error *bound*, not the value".
- Detect: any e2e failure that goes away when the tile size changes but the extent doesn't is associativity noise, not a bug; any failure that scales with extent is a real bug (dropped/garbage element).

**Phase:** 2 (harness design) — this is a process landmine planted in phase 2 that detonates in phase 3.

---

### P4: SSA use-def chain corruption during passes (the classic mutable-IR bug)

**What goes wrong:** A transformation pass replaces an operand or deletes a node and forgets to update one direction of the use-def/def-use chains. Subsequent passes (DCE, loop merge, hoisting) make decisions on stale use lists: dead code survives, live code is eliminated, or a hoisted load reads the wrong def. The IR verifier — which is supposed to catch this — may itself only check one direction.

**Why it happens:** design.md requires an explicitly *mutable* IR ("IR must be carefully designed to support optimization mutation"), which is in direct tension with the global immutability coding rule. If mutation is allowed ad hoc, every pass author re-implements chain surgery slightly differently. The Go compiler's own SSA backend exists as the reference for how Go handles this, and its key discipline is: **one choke-point mutation API plus a `verify` pass run after every pass** (`ssa/check` in the Go tree, enabled per-pass with `-d=ssa/check/on`).

**Consequences:** Silent miscompilation — garbage output with no crash. These are the hardest bugs to localize because the corruption happens in pass N but manifests in codegen of pass N+3.

**Prevention:**
- Single mutation API on `SSAVar` (design.md already names it): `SetDef`, `AddUse`/`RemoveUse`, `ReplaceAllUsesWith(old, new)`, `Delete` — the *only* functions that touch the shared underlying pointer. Passes never assign the SSAUse/SSADef fields directly. This is the project's sanctioned exception to the immutability rule: immutability everywhere else, disciplined mutation only through this API.
- A verifier pass run **after every transformation pass in debug builds** (not just "a verifier pass for every pass" as an artifact — wired into the pass manager so it cannot be skipped). It must check *both* directions: every use appears in its def's use-list; every def appears in each of its uses' arg lists; plus project invariants (single basic block, no pipes after effect linearization, `as` clauses present, store indexing rule of tca.md §5.7.8).
- Prefer stable integer IDs over Go pointers for node identity in maps/sets (pass fingerprints, CSE keys): pointers change meaning across mutation and defeat map-based analyses; Go slices reallocate on growth, invalidating pointer-based identity. If pointers are used for graph edges, never store them in pass-local maps keyed by address.
- Diagnostics: the verifier's failure message must print the offending node, its pass provenance (which pass last touched it — keep a last-touched-by field in debug builds), and the broken invariant. "IR verification failed" with a node dump is not a diagnostic.

**Warning signs:** A pass's output validates but codegen produces wrong addresses; DCE removing a node that still has uses (or failing to remove clearly dead chains); nondeterministic output across runs (map iteration order interacting with stale chains).

**Phase:** 1 (IR core + pass infrastructure). This is the single highest-leverage phase-1 decision: choke-point mutation API + mandatory post-pass verifier.

---

### P5: Dynamic SMEM >48 KB opt-in and silent launch failure

**What goes wrong:** The cache allocator (tca.md §5.8, 64 KB budget) emits a kernel using >48 KB of shared memory. The launch either fails with `invalid configuration argument` — reported **asynchronously** at the *next* CUDA call, miles from the cause — or the attribute call fails and is never checked. On the A6000 specifically: the opt-in ceiling is **99 KB (101,376 bytes)**, not the 164 KB of A100-class sm_80 parts; a cache plan sized against the wrong device silently overcommits.

**Why it happens:** >48 KB requires (1) `extern __shared__` dynamic SMEM (static `__shared__` arrays are capped at 48 KB — a compile-time trap: generated code that declares a static 64 KB buffer fails to compile, pushing the compiler toward dynamic SMEM everywhere), (2) `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` per kernel specialization, (3) passing the same byte count at `<<<grid, block, bytes>>>`. Each step has an independent failure mode, and `cudaFuncSetAttribute` returns an error that is almost never checked in example code (the sglang sm_120 crash is exactly this: unchecked `cudaFuncSetAttribute` failure followed by garbage).

**Consequences:** Silent wrong results (launch failed, output buffer never written, comparison reads uninitialized memory that sometimes passes), or crashes attributed to the wrong kernel.

**Prevention:**
- The runtime's launch stub owns the entire opt-in sequence: query `cudaDevAttrMaxSharedMemoryPerBlockOptin` once at init, clamp/validate the compiler-emitted SMEM plan against it at launch, call `cudaFuncSetAttribute` **per kernel specialization** (template instantiations each need their own call), check every return code, and fail the launch loudly with the kernel name, requested bytes, and device limit in the message.
- Generated code uses `extern __shared__` + a layout struct with offsets computed by the compiler (tile buffers + partials + padding in one allocation), never static `__shared__` — one allocation path, sized once, instead of N attribute-eligible static arrays.
- The 64 KB budget + runtime reserved space (reducer partials, padding slack) must fit in 101,376 B on the target device — encode the target-device constant in the launch validator, not just "≤ some large number".
- Harness: enable `CUDA_LAUNCH_BLOCKING=1` in e2e tests so async launch errors surface at the launch site with the right kernel name.

**Warning signs:** `cudaErrorInvalidValue`/`invalid configuration argument` at a memcpy or sync call following the kernel; kernel "runs" but output equals the pre-fill pattern.

**Phase:** 4 (fusion + caches, where the 64 KB budget is exercised) — but the launch-stub error-checking discipline belongs to phase 2's runtime.

---

### P6: cp.async completion-group handling in cooperative tile copies

**What goes wrong:** The runtime's cooperative `loadtile` issues `cp.async.cg/ca` copies and mishandles completion groups: `cp.async.commit_group` is called a different number of times across threads (so `cp.async.wait_group N` waits for different logical groups per thread); a `wait_group 0` flushes *all* outstanding groups including double-buffered stages from a previous pipeline iteration; or the code waits with `cp.async.wait_group` but omits `__syncthreads()` afterward, so thread A reads SMEM bytes that thread B's async copy has completed-but-not-made-visible... (actually wait_group guarantees the *calling thread's* copies are visible to itself; visibility to *other* threads of the CTA requires a barrier after the wait).

**Why it happens:** cp.async groups are **per-thread** state machines. The natural mental model "the CTA issued a copy, the CTA waits for it" is wrong — each thread has its own group queue. Uniformity of `commit_group`/`wait_group` call counts is therefore as much a collective invariant as `__syncthreads` (ties back to P2).

**Consequences:** Reading stale SMEM (wrong numerics, deterministic-looking), or pipeline deadlocks in a later double-buffering iteration.

**Prevention:**
- Contract (tca.md §5.11 already implies it): cooperative variants do **uniform commit counts** — every thread calls `commit_group` exactly once per tile stage, even if it issued zero `cp.async` instructions for that stage (commit of an empty group is legal and cheap).
- Use `cp.async`'s zero-fill operand (`cp.async.cg.shared.global [dst], [src], 16, src_size` with `src_size=0` for OOB) to implement predicated-tail loads *without* divergence — this neatly unifies the bounds rule (tca.md §4.3: invalid slots may hold unspecified padding, map invocation is suppressed) with uniform async-control-flow.
- After `cp.async.wait_group N`, always `__syncthreads()` before any thread reads bytes written by another thread's copy. Visibility-of-completion ≠ visibility-of-data-CTA-wide.
- Phase 4 only needs depth-1 pipelining (fill → wait → compute); design the group discipline now (uniform commits, one wait point, barrier after) so multi-stage double buffering is a runtime-internal change, never an IR change (tca.md §5.11 already promises this — hold that line).

**Warning signs:** Tile cache results correct for small tiles, wrong for tiles that hit the remainder guard; corruption only when a specific thread issued zero copies.

**Phase:** 4 (caches). Also phase 2 if cooperative load/store ships early — note sm_86 supports cp.async (sm_80+), it is the correct ceiling choice per PROJECT.md.

---

## Moderate Pitfalls

### P7: SMEM bank conflicts and padding math in the runtime tile layout

**What goes wrong:** The compiler emits tile shapes; the runtime declares SMEM buffers. If the tile's inner dimension is a multiple of 32 (f32) — e.g., a 32×32 f32 tile — column-wise access by strided threads is 32-way bank-conflicted, and the fix (padding `[32][33]`, swizzling `col ^ row`) lives in the runtime's layout, which the compiler's generated indexing must agree with. If the compiler computes addresses for an unpadded layout while the runtime pads (or vice versa), results are wrong *and* slow.

**Prevention:** The padded/swizzled layout is a runtime-owned accessor function (e.g., `tile.at(r, c)`) that codegen always calls — never open-coded index arithmetic in generated code. The compiler only passes the *logical* tile shape; padding is "extra space the variant may request" (already allowed by tca.md §5.11). Verify with `ncu` (bank-conflict metrics) once per preset, not per kernel. Note: padding costs SMEM and can cost occupancy — count padded sizes against the 64 KB budget, not logical sizes.

**Warning signs:** Matmul-preset kernels an order of magnitude below roofline on `ncu` with high `l1tex__data_bank_conflicts`.

**Phase:** 4.

---

### P8: Uncoalesced stores from `as`-order vs store-layout mismatch

**What goes wrong:** tca.md §2.7 rule 6 already specifies this, but it is worth elevating: when the last variable of `as` does not index the store's contiguous axis (transposed write), generated global stores are fully uncoalesced. The verifier emits a performance warning — the pitfall is that nothing *enforces* a response. Preset S2/S3 assign `p_c → Thread` assuming coalesced `p_c`-contiguous stores; a kernel whose store layout disagrees with `as` gets silently slow code from the preset table.

**Prevention:** Make the §2.7 rule-6 warning an input to preset selection (prefer assigning `CTA`/thread parallelism to the store-contiguous dimension) in a later milestone; in v0, surface the warning at compile time with the specific `as`/store mismatch named, and add one `ncu` coalescing check to the e2e suite for the flagship kernel. Distinct from correctness — never let a perf warning block codegen.

**Phase:** 3 (preset lowering); measurement in 4.

---

### P9: Zero-extent launches and empty tiles

**What goes wrong:** `grid = ceil(ext/T)` = 0 when a runtime extent is 0 → `invalid configuration argument` at launch, surfaced asynchronously. Related: an empty reduction (`N=0` reduced extent) must return the initializer (tca.md §4.3 promises this) — an implementation that unconditionally calls the combiner over zero elements usually gets this right by accident, but one that special-cases "no elements" by skipping `Combine` breaks the CTA-uniform collective invariant. S6's single-CTA full reduction with `M=0`/`N=0` is the degenerate case that hits every layer at once.

**Prevention:** The host launch stub guards every launch (`if (grid_size > 0)`), validates extents (already its job per PROJECT.md), and treats zero-extent outputs as correctly-shaped empty tensors with no launches. The reducer returns the initializer for zero valid elements **inside** the normal Init/Combine protocol (Init establishes it; Combine with no contributions yields it) — no special path. Add `N=0` and `N=1` and `N=255/256/257` (tile boundaries) to the e2e extent matrix in phase 3.

**Warning signs:** `invalid configuration argument` at a synchronize/memcpy after launch with a zero-size test case; S6 tests missing from the suite.

**Phase:** 2 (launch stub guards), 3 (reducer degenerate cases).

---

### P10: FMA contraction and f16/bf16 conversion policy in the lambda compiler

**What goes wrong:** Two layers: (1) nvcc's default `--fmad=true` contracts expression-tree `a*b+c` into FMAs, so the generated kernel is *not* bit-identical to a naive left-to-right evaluation of the typed scalar expression graph — fine for correctness-with-tolerance, fatal if any test assumed bitwise (ties to P3). (2) The v0 homogeneous-reducer rule means an f16 input reduced in f16 accumulates in f16: with a 10-bit mantissa and 5-bit exponent, long sums lose precision catastrophically (and overflow at 65504). The escape hatch — explicit `cast<f32>` before the reduce, cast back after (tca.md §4.2) — only works if casts survive codegen as real conversions and aren't "optimized away" as no-ops by a well-meaning simplification pass.

**Prevention:**
- Casts are IR nodes with explicit semantics; the simplifier may fold `cast<f32>(cast<f16>(x))` to a single conversion, but may **never** delete a widening/narrowing cast pair around a reduction. Add this to the verifier's invariant list and to the simplifier's regression tests.
- Document the accumulator-precision contract: the reducer's accumulator type = the combiner's parameter type (homogeneous rule), so precision policy is visible in the IR, not hidden in codegen. Compiler diagnostic: warn when a reduce over f16/bf16 has >256 elements without an f32 cast (mirrors tca.md's own `expf(cast<f32>(x))` example).
- Keep `--fmad` default for performance lanes; the *naive* phase-2 lane pins `--fmad=false` so phase 2 tests can be bitwise against the CPU reference (closes the associativity loop of P3 cleanly at the phase boundary).

**Warning signs:** f16 e2e tests passing at small extents and drifting at large ones; a simplifier test that "cleans up" redundant-looking casts.

**Phase:** 1 (simplifier + verifier invariants), 2 (cast codegen), 3 (reducer precision).

---

### P11: Go-specific determinism and representation traps

**What goes wrong:** (a) **Map iteration order is randomized in Go** — any pass that iterates a `map` while emitting code or numbering IR nodes produces nondeterministic output: generated `.cu` files diff between runs, golden-file tests flake, and debugging-by-diff dies. (b) **Slice reallocation invalidates pointers** stored in side tables (see P4). (c) `interface{}`-typed aux fields (the Go compiler's own `Value.Aux` pattern) defeat static checking — an opcode/aux type mismatch compiles fine and only fails in the verifier.

**Prevention:**
- Ordered structures everywhere output order matters: slices for node storage, explicit ID allocation, and a single `Sort` by stable ID before emission. Determinism test in phase 1: compile the same graph twice, byte-compare the output. This is one cheap test that pays for itself permanently.
- Treat the Go compiler SSA source (`cmd/compile/internal/ssa/README`, `rulegen.go` pattern) as the reference: opcode metadata in generated tables (`go generate`) instead of hand-maintained switch forests — this project has exactly the "many node kinds in one suite" shape that benefits.

**Phase:** 1.

---

### P12: Pass-order coupling and verifier coverage gaps

**What goes wrong:** tca.md fixes the pipeline order (canonicalization → merge/fusion → parallelization → iteration lowering → reduce lowering → cache lowering → codegen) with hard couplings: effect linearization must precede parallelization; iteration lowering must precede reduce lowering; the `reduce<@r>` annotation must be attached before tiling splits `@r`. Violating any coupling doesn't crash — it produces plausible-looking wrong lowering (e.g., Combine placed per-tile instead of around `@r_o`, the exact bug §5.10's placement rule exists to prevent).

**Prevention:** The pass manager declares dependencies explicitly (each pass names its required preconditions/analyses; the manager topologizes and refuses to run an out-of-order pipeline in debug builds). The per-pass verifier checks pass-specific invariants (§5.10's Init/Combine placement is checkable structurally: exactly one Init before `@r_o`, exactly one Combine after, `Reduce` inside `@r_i`). This is where design.md's "figure out invariants before/after each pass" pays off — the invariants must be written down per pass *in phase 1*, while the pipeline is small, or they never get written.

**Warning signs:** A lowering that works for untiled reduces and breaks when `T(R) > 1`; Combine appearing inside a loop nest in the Graphviz dump.

**Phase:** 1 (pass manager), 3 (the coupled passes).

---

## Minor Pitfalls

### P13: Diagnostic quality erosion
**What goes wrong:** Compiler diagnostics start rich ("store omits live variable @k at node %12, statement 1") and decay to `fmt.Errorf("invalid IR")` under deadline pressure. design.md demands rich verifier output; the pitfall is not building the diagnostic *infrastructure* (position tracking, node naming stable across passes, Graphviz dump hook per pass) in phase 1, when it's cheap.
**Prevention:** Diagnostic and visualization packages ship in phase 1 with the pass manager; every verifier message carries node ID, pass name, and a dump command to reproduce. Budget this as first-class phase-1 scope, not polish.

### P14: Gridviz/IR dump tooling drifting from the IR
**What goes wrong:** The Graphviz visualization reads IR internals directly; after a few passes add node kinds, the viz either breaks on new nodes or silently renders them wrong, and a debugging tool that lies is worse than none.
**Prevention:** Viz goes through the same node-dispatch table as the verifier (one visitor, all node kinds, default case renders an error box, never skips).

### P15: Over-testing against the 'fundamental tests only' rule — and its mirror image
**What goes wrong:** Two failure modes, opposite directions. (a) Golden-file/test-explosion creep: every pass author adds snapshot tests of full IR dumps; suites rot and each IR tweak produces 40 failing snapshots — this is the specific thing design.md forbids. (b) Under-testing the invariants: "only fundamental tests" gets read as "no IR tests," and use-def corruption (P4) ships.
**Prevention:** The fundamental set, concretely: (1) IR unit tests for the mutation API and each pass on 1-2 hand-built graphs targeting the *invariant*, not the snapshot; (2) the determinism byte-compare (P11); (3) e2e numerical suite with the extent matrix {0, 1, 255, 256, 257, non-multiple-of-tile} per kernel family; (4) synccheck/memcheck runs of the e2e suite. Anything beyond this needs justification at review. Verifier coverage substitutes for most IR-level test volume — that's the deal design.md is offering.

### P16: Premature optimization in the IR core
**What goes wrong:** Phase 1 builds hash-consing, interning pools, arena allocators, or fancy immutable-snapshot structures for IR nodes "because compilers need to be fast" — before there is a working codegen. The IR core balloons, the flagship gate slips, and the optimizations are tuned against a workload that doesn't exist yet.
**Prevention:** Phase 1 correctness criteria are structural (choke-point API, verifier, determinism, diagnostics), not performance criteria. Profiling-driven IR work is deferred until a phase-3/4 pass is demonstrably slow. The one acceptable early optimization: stable integer IDs (P11), which is a correctness decision, not a performance one.

### P17: Frontend parser before the elaborator earns its keep
**What goes wrong:** syntax.md §11.1 already orders elaborator/graph-builder before the parser; the pitfall is scope creep in phase 5 (full EBNF grammar, error recovery, pretty errors) delaying the milestone close. Also: name/shape resolution bugs that the IR-level builder cannot express (it never makes them) hide until the parser generates them — the graph-builder tests don't cover parser-only failure modes.
**Prevention:** Parser milestone ends at "all documented example kernels parse and produce IR identical to the hand-built graphs" — the hand-built graphs are the oracle, which is precisely why the elaborator-first order is correct. Resist error-recovery polish beyond one sensible message per failure class.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| 1: IR core | Use-def corruption via ad-hoc mutation (P4); nondeterministic output via map iteration (P11); diagnostic infra never built (P13); scope creep into perf work (P16) | Choke-point mutation API; mandatory post-pass verifier; determinism byte-test; ordered emission |
| 2: Naive codegen + runtime | Divergent-barrier shapes creeping into codegen (P2); launch guards missing for zero extents (P9); tolerance policy undefined until phase 3 forces it (P3); `--fmad` uncontrolled (P10); unchecked CUDA API returns (P5 discipline starts here) | Establish uniform-trip-count + predicated-body rule and collective contract now; pin `--fmad=false` in the naive test lane; define comparison policy in the harness |
| 3: Parallelization + reduce lowering | Shuffle mask/participation bugs (P1); Init/Combine placement vs tile loops (P12); reducer identity vs validity-tag semantics for non-zero inits (P1); per-lane empty strided ranges (P1/P9); uncoalesced stores from preset/as mismatch (P8) | Reducer owns all collectives; verifier checks placement structurally; extent matrix incl. 0/1/tile±1; synccheck in CI; `ncu` on flagship |
| 4: Fusion + caches | cp.async group discipline (P6); >48 KB opt-in + A6000 99 KB ceiling (P5); compiler/runtime layout disagreement on padding (P7); budget counted pre-padding (P7) | Uniform empty commits; zfill for tails; runtime-owned accessor; launch validator with device constants; padded sizes against budget |
| 5: Frontend | Parser-before-oracle testing (P17); test-suite creep (P15) | Hand-built IR as parser oracle; fundamental-tests-only rule enforced at review |

## Cross-Phase Invariants Worth Encoding Early

These recur across multiple phases; encode once in phase 1-2 infrastructure:

1. **Every collective is CTA-uniform** (P1/P2/P6) — one runtime contract, asserted in debug builds, covering shuffles, barriers, cp.async control flow, and reducer Init/Combine.
2. **Every CUDA API return is checked at the launch stub** (P5/P9) — one place, loud failures with kernel name + sizes.
3. **Every pass leaves the IR verifiable** (P4/P12) — the deal design.md proposes: verifier coverage in exchange for test restraint.
4. **Output is deterministic** (P11) — byte-stable generated code, tested from phase 1.

## Sources

- [Using CUDA Warp-Level Primitives (NVIDIA Developer Blog)](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) — mask discipline, no `__activemask()`, undefined values outside mask (HIGH)
- [What does mask mean in warp shuffle functions (NVIDIA Forums, Robert Crovella)](https://forums.developer.nvidia.com/t/what-does-mask-mean-in-warp-shuffle-functions-shfl-sync/67697) — "all non-exited threads named in mask must execute the same intrinsic" (HIGH)
- [Warp-wide reduction slowdown since CUDA 12.4 for sm_80 (NVIDIA Forums)](https://forums.developer.nvidia.com/t/warp-wide-reduction-operations-significant-slowdown-since-cuda-12-4-for-sm-80/294101) — runtime-mask penalty; target is CUDA 12.2 so not binding, flagged for toolchain upgrades (MEDIUM)
- [CUDA C++ Programming Guide (12.4)](https://docs.nvidia.com/cuda/archive/12.4.0/pdf/CUDA_C_Programming_Guide.pdf) — dynamic SMEM >48 KB opt-in requirement; Volta+ barrier semantics (HIGH)
- [sglang issue #29897: unchecked cudaFuncSetAttribute failure](https://github.com/sgl-project/sglang/issues/29897) — real-world unchecked-attribute crash pattern (HIGH, corroborated)
- [NVIDIA Forums: max shared memory per block/multiprocessor (sm_86 limits)](https://forums.developer.nvidia.com/t/question-about-max-shared-memory-in-block-and-multiprocessor/283345) — A6000/GA102 opt-in = 101,376 B (HIGH, hardware-verified by multiple reporters)
- [Running zero blocks in CUDA (Stack Overflow)](https://stackoverflow.com/questions/19022725/running-zero-blocks-in-cuda) — zero grid dim = invalid configuration argument (HIGH)
- [spconv issue #406: N=0 launch assert](https://github.com/traveller59/spconv/issues/406) — empty-tensor → zero-grid launch in the wild (HIGH, corroborated)
- [Stack Overflow: Can I use __syncthreads() after having dropped threads?](https://stackoverflow.com/questions/6666382/can-i-use-syncthreads-after-having-dropped-threads) — pre/post-Volta barrier semantics change (HIGH)
- [CUDA Programming Guide: Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html) — collective-uniform invocation contract (HIGH)
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — reduced-precision reduction hazards, fp16/bf16 accumulation (HIGH)
- [JAX issue #27496: bf16 gradient reduction](https://github.com/jax-ml/jax/issues/27496) — lossy accumulate-narrow chain; accumulate-wide fix (MEDIUM, corroborated)
- [Half-precision: __float2half vs __float2half_rn (Stack Overflow)](https://stackoverflow.com/questions/35198856/half-precision-difference-between-float2half-vs-float2half-rn) — conversion intrinsic equivalence (MEDIUM)
- [CUDA Math API: Half Precision Conversion](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html) — conversion family reference (HIGH)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) — padding example (140→199 GB/s) (HIGH)
- [CUDA shared memory bank conflicts (bias92)](https://bias92.github.io/en/posts/cuda-3-shared-memory/) — gcd(stride,32) conflict degree, padding mechanics (MEDIUM)
- [Introduction to the Go compiler's SSA backend](https://go.dev/src/cmd/compile/internal/ssa/README) — per-pass verify pattern, generated opcode tables (HIGH, reference implementation)
- [CUTLASS discussion #500: cp.async wait_group semantics](https://github.com/NVIDIA/cutlass/discussions/500) — multi-stage group counting discipline (MEDIUM)
- [TBIK / tree-based invariant kernels (arXiv 2001.05585 context + reduction-order analysis)](https://arxiv.org/pdf/2001.05585) — reduction order as a function of shape/config; determinism classes and tolerances (MEDIUM)
- [KrxGu kernel-skills: debug CUDA kernel correctness](https://github.com/KrxGu/kernel-skills/blob/master/skills/cuda/debug-cuda-kernel-correctness/SKILL.md) — tolerance guidance (~1e-5 rel f32), bitwise only for exactly-associative ops (MEDIUM)
