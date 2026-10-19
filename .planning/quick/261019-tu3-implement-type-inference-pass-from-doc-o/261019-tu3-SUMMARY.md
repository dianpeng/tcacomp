---
phase: quick-261019-tu3
plan: 01
subsystem: compiler-ir
tags: [type-inference, ir, ssa, pass-manager, typing-rules]

# Dependency graph
requires:
  - phase: quick-261019-ktm
    provides: Simplification passes (simplify-lower/inline/verify) whose output typeinfer types; Input/ImpactBy-only node suite the rules read
provides:
  - Computed-type cell on the IR (Base.ty) with the MutContext.SetType choke point (idempotent) and the TypeOf/IsErased read faces
  - ir.Builtin — the single pure-builtin signature registry (moved from pkg/expr.go)
  - The typeinfer pass (priority 105, between simplify-verify and gvn) typing every non-erased node of the 31-kind catalog per doc/optana/type_inf.md
  - Exported payload read faces for analysis passes (ConstTypeOf, CastTypeOf, CallNameOf, BinOpCodeOf, UnaryOpCodeOf, InputParamTypeOf, InputParamNameOf)
affects: [gvn, lowering passes, codegen, diagnostics, future cache-lowering (buffer typing), future frontend]

# Tech tracking
tech-stack:
  added: []  # stdlib-only pure Go, per project constraint
  patterns:
    - Compute-then-stamp pass discipline (memo fully computed before any SetType; a refused graph is never half-stamped)
    - Memoized iterative post-order typing over the operand DAG (explicit frame stack; no fixed point — justified by SSA/no-phis/no-back-edges)
    - Element-view typing for expression operands (a tensor in expression position denotes its per-point element)

key-files:
  created:
    - internal/ir/builtins.go
    - internal/ir/typecell_test.go
    - internal/ir/builtins_test.go
    - internal/opt/typeinfer/typeinfer.go
    - internal/opt/typeinfer/rules.go
    - internal/opt/typeinfer/typeinfer_test.go
  modified:
    - internal/ir/node.go
    - internal/ir/mutate.go
    - internal/ir/node_expr.go
    - internal/ir/node_var.go
    - pkg/expr.go
    - internal/opt/all/all.go
    - internal/opt/registry.go
    - doc/ir.md

key-decisions:
  - "One-pass memoized post-order replaces the spec's fixed-point question: the IR is SSA with no phis and no back edges, so the data-dependency graph is a DAG; operand cycles fail closed naming the node"
  - "Expression operands type at ELEMENT view (a tensor in expression position denotes its per-point element) — the strict rank-0-only reading would reject every post-inline builder body, e.g. matmul's `l * r` over lhs/rhs refs and `acc + now` over the piped stage"
  - "SetType is idempotent by design: an Equal type neither writes nor re-stamps, making typeinfer re-runnable with stable LastTouchedBy (machine-checked)"
  - "Closed untyped whitelist: memnew/memnewsmem/fragnew (allocations carry no dtype), as (an output-order clause), lambda (no function Type yet), and a 0-or-multi-value return (future tuple Type)"
  - "SMEM/frag loads fail closed until cache lowering binds buffer dtypes — the LIR rule exists per spec rule (4) 'HIR and LIR alike' and future passes satisfy it by typing their buffers"

patterns-established:
  - "Payload read faces: analysis passes read node payloads through exported total functions instead of asserting concrete node types"
  - "In-pass postcondition sweep (~10 lines) instead of a separate verify pass when there is no transformation to re-check"

requirements-completed: [TI-1, TI-2, TI-3, TI-4, TI-5, TI-6, TI-7, TI-8]

# Metrics
duration: 11min
completed: 2026-10-19
---

# Quick Task 261019-tu3: Type Inference Pass Summary

**Type inference (typeinfer, priority 105) stamps every non-erased node of the 31-kind catalog with a derived type on a new Base.ty cell — written through the idempotent MutContext.SetType choke point, computed by one memoized post-order pass, and readable by later passes via ir.TypeOf.**

## Performance

- **Duration:** ~11 min
- **Started:** 2026-10-19T13:47:13Z
- **Completed:** 2026-10-19T13:58:26Z
- **Tasks:** 2 (both TDD: RED + GREEN)
- **Files modified:** 14 (6 created, 8 modified)

## Commits

| Commit | Type | What |
|---|---|---|
| fb30c0f | test (RED) | type-cell/SetType/IsErased/ir.Builtin contracts pinned |
| cf13d9d | feat (GREEN) | IR type cell, SetType choke point, IsErased, ir.Builtin registry; pkg.Call delegates |
| 742f3db | test (RED) | typeinfer rules/idempotency/fail-closed/oracle-integration/registration pinned |
| 1a4beb0 | feat (GREEN) | typeinfer pass (105) + rules.go + ir read faces + all.go wiring + doc/ir.md sync |

## Accomplishments

### Task 1 — IR type cell, SetType choke point, IsErased, ir-level builtin registry

- `Base.ty` computed-type cell (nil = uninferred); convention-safe, never printed/serialized/verified — a computed type is pass state, not a chain invariant. Byte-exact goldens untouched (full suite green).
- `ir.TypeOf` / `ir.IsErased` nil-safe package read faces.
- `MutContext.SetType` — the eighth choke point: nil-node error like SetDef/Erase; Equal existing type = no write and no re-stamp (the idempotency primitive); header/doc comments updated seven → eight.
- `internal/ir/builtins.go`: the `(f32,...f32)->f32` table moved from pkg/expr.go as the single source of truth (`BuiltinSignature`, `Builtin(name)`), never iterated, no runtime mutation path.
- `pkg.Call` resolves through `ir.Builtin` — same messages, same f32 enforcement; pkg tests pass unchanged.

### Task 2 — the typeinfer pass

- `internal/opt/typeinfer/typeinfer.go`: `Pass` (ID `typeinfer`, Requires `["simplify-lower"]`, Produces StageConstructed, Preserved nil), self-registered at priority 105; gates mirror simplify (nil graph, materialized/CFG → error); compute phase = memoized iterative post-order over `OperandsOf` only (ImpactBy edges are effect order, never type inputs; lambda bodies ride the lambda node's operand slots); on-stack revisit = data-dependency cycle → fail-closed error naming the node; stamp phase through ONE `MutContext{Pass: "typeinfer"}` in allocation order; in-pass postcondition sweep over the closed untyped whitelist.
- `internal/opt/typeinfer/rules.go`: the closed 31-kind switch — const payload; param → target InputParam rule (RoleDim → u32 computed, others payload-or-error "input parameter carries no user type"); binop/unary/select/cast/call at element view with syntax.md §6.2 classes and ir.Builtin signatures; iter/forloop/dimvar/bdimvar/inductionvariable/tensordimof → u32; itercluster legacy passthrough (single input) vs 0/>1 inputs error; live iterclusterproject errors on sight; map → tensor< bodyElem, symbolic As-dims > with the param↔binding element-level reconcile; reduce → piped type minus last dim with the tca.md §4.2 homogeneous contract and optional-As rank/name validation; memload partial/full/over deref with scalar-u32 axes; memstore/storesmem/fragstore return the stored value's type; memloadsmem/fragload fail closed until buffers carry dtypes; keyedattribute unwraps; return types as its single value or stays nil.
- `internal/opt/all/all.go` blank import; `internal/opt/registry.go` priority-scale comment now carries the 105 anchor.
- `doc/ir.md` synced: node.go/mutate.go/node_expr.go/node_var.go rows, new builtins.go row, §5.3 eight-choke-point table with the SetType row, §4.1 paragraph on typeinfer (105).

### Verification

- `go build ./...` clean; `go vet ./...` clean.
- `go test ./...` — 14/14 packages green, including all pre-existing byte-exact golden tests (the feature is invisible to every existing golden).
- Machine-checked idempotency: two runs → identical TypeOf everywhere AND stable LastTouchedBy.
- Pipeline wiring: `opt.Lookup("typeinfer")` priority 105, Requires `[simplify-lower]`; `BuildPipeline` orders it between simplify-verify and gvn.
- Oracle integration: the matmul_scale shape through [simplify-lower, simplify-inline, simplify-verify, typeinfer] types root map tensor<f32,[m,k,n]>, reduce tensor<f32,[m,k]>, tail map tensor<f32,[m,k]>, store tensor<f32,[m,k]>, every non-erased node typed per the postcondition.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Plan inconsistency] Element-view typing for expression operands**
- **Found during:** Task 2 GREEN
- **Issue:** The plan's expr rule said "every operand must be a rank-0 *ir.NumericType (tensor operand -> error)", while the pinned integration behavior requires the post-inline matmul oracle to fully type. Post-inline lambda bodies route whole-tensor SSA values where the body means their elements (root body `mul(paramRef(lhs), paramRef(rhs))` — both tensor-typed refs; tail body `mul(reduce, paramRef(scale))` — a rank-2 stage value beside a scalar). The strict rule would reject every builder graph at the default pipeline's 105 slot.
- **Fix:** `scalarView` types an expression operand at element level: a rank-0 numeric as-is, a tensor its element type — the same convention the printer embodies (a paramRef to a tensor param IS the structural access) and simplify-inline produces. All the plan's scalar error cases still fire (f32+i32, LAnd non-bool, Mod floats, select arm mismatch, non-f32 builtin args).
- **Files modified:** internal/opt/typeinfer/rules.go (documented in the file header)
- **Commit:** 1a4beb0

**2. [Rule 3 - Blocking] Exported payload read faces in ir**
- **Found during:** Task 2 GREEN
- **Issue:** No exported read faces existed for const/cast payload types, call names, binop/unary opcodes, or InputParam payload type/debug name — the pass (outside package ir) cannot read them without asserting concrete node types.
- **Fix:** Added total exported faces: `ConstTypeOf`/`CastTypeOf`/`CallNameOf`/`BinOpCodeOf`/`UnaryOpCodeOf` (node_expr.go) and `InputParamTypeOf`/`InputParamNameOf` (node_var.go); doc/ir.md rows synced.
- **Files modified:** internal/ir/node_expr.go, internal/ir/node_var.go, doc/ir.md
- **Commit:** 1a4beb0

**3. [Rule 1 - Plan note inaccuracy] AllocatedNodes and tombstones**
- **Found during:** Task 1
- **Issue:** The plan's interface block said `AllocatedNodes()` "includes tombstones"; the implementation filters them (`Tombstones()` is the complement). The IsErased motivation was rewritten accordingly (it remains the per-node nil-safe face the pass uses for operand defs).
- **Fix:** typecell_test.go pins the real contract (erased node absent from AllocatedNodes, present in Tombstones); the pass skips erased operand defs through IsErased.
- **Files modified:** internal/ir/typecell_test.go
- **Commit:** cf13d9d

**4. [Rule 3 - Blocking] Oracle integration fixture built through ir constructors**
- **Found during:** Task 2 RED
- **Issue:** The plan asked for the oracle "through pkg.NewBuilder", but `pkg.Build()` materializes the graph and typeinfer (like every sea pass) refuses materialized graphs; pkg exposes no pre-Build graph accessor.
- **Fix:** The integration fixture hand-builds the exact oracle shape (same rung sequence, legacy wrapper clusters, chained store/return) through the exported ir constructors — the established pass-test pattern (simplify_test.go's docExample documents the same precedent). The legacy-wrapper clusters exercise the itercluster passthrough rule on the real oracle shape.
- **Files modified:** internal/opt/typeinfer/typeinfer_test.go
- **Commit:** 742f3db

## Known Stubs

None. (The memloadsmem/fragload fail-closed path is not a stub: it is the specified LIR contract until the future cache-lowering pass binds buffer types.)

## Threat Flags

None — no new security-relevant surface beyond the plan's threat model. T-ti-01..05 dispositions implemented as planned: iterative traversal (no recursion), compute-then-stamp, closed-kind fail-closed, node-ID-only diagnostics, compile-time builtin table.

## Self-Check: PASSED

- Files: internal/ir/builtins.go, internal/ir/typecell_test.go, internal/ir/builtins_test.go, internal/opt/typeinfer/typeinfer.go, internal/opt/typeinfer/rules.go, internal/opt/typeinfer/typeinfer_test.go — all present.
- Commits: fb30c0f, cf13d9d, 742f3db, 1a4beb0 — all in git log.
- Full test suite and vet green at HEAD (1a4beb0).
