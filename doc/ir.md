# TCA IR: the implementation-faithful reference (`internal/ir`)

This document describes the TCA IR **as implemented today** — the Go package
`internal/ir`, plus the pass-level scheduler in `internal/pass/scheduler.go`
and the construction surface in `pkg/`. Every claim here is verifiable against
the current source; where the design docs and the code disagree, the code is
right and the disagreement is called out explicitly (see §1.2).

Reader's map:

| Doc | Role |
|---|---|
| `doc/tca.md` | The algebra and pipeline specification: what the IR means (§2–§4), what the passes do to it (§5). |
| `doc/syntax.md` | The v0 frontend language the IR ultimately compiles. |
| **`doc/ir.md` (this file)** | The implementation-faithful IR spec: file organization, node catalog, SSA/effect/liveness machinery, scheduling, and worked conceptual-code → IR examples. |
| `doc/optana/induction_variable.md` | Formal spec of the IV/BIV value relations (`SameIV`, `Includes`, …). |
| `doc/optana/lower_itercluster.md` | The feature request the IterCluster→Iter lowering + Simplification passes implement (quick-261019-ktm); the Input-list restructure and the convention below are its amendments. |

## 1. Overview & how to read this doc

### 1.1 What the IR is

The TCA IR is a **single-tier, SSA, optimizer IR** (design.md's "IR must be
carefully designed to support optimization mutation" — one node suite from
which both the abstract kernel graph and the lowered forms are built; no
separate high/mid/low IRs). It is not an AST materialization: every reference
in the graph is an `SSAUse` record and every definition is an `SSADef`, so
passes rewrite graphs in place through a small choke-point API rather than
rebuilding trees (review `doc/review/phase1_1.md`, P1R-01).

> ### ⚠ REQUIREMENT — the Input/ImpactBy-only convention (quick-261019-ktm)
>
> A node's inputs live **ONLY** in its two dependency lists — nothing else:
>
> | List | Models | Storage |
> |---|---|---|
> | **Input** (operand slots) | **data dependencies** | the node's `operandList` |
> | **ImpactBy** | **side-effect dependencies** | a direct `[]SSAUse` field, on effect bearers only (`memload`/`memstore`/`return`) |
>
> **No node may hide state referencing SSA values / nodes / dimvars in record
> fields outside these two lists — every node kind, loops, lambdas, everything.**
> A field that *could* hold a reference is a violation the moment it is
> declared, even when zero-valued. Names, literal constants (e.g. an init
> value, an axis index), opcodes, and kinds are metadata, not dependencies —
> they stay. Binder declarations bind positionally from the Input list and
> never store back-pointers.
>
> **Why:** a hidden reference is invisible to `Walk`, the verifier, DCE, GVN,
> and the printers — the analysis machinery sees only the two lists, so any
> third storage is a correctness hole, not a style issue.
>
> **This is machine-checked** (§5.2): `Verify` runs a reflect sweep
> (`convention.go`) over every allocated node's struct fields and reports a
> `convention` violation for any field whose type can carry SSA/Node
> references outside the sanctioned spellings. Adding a node kind with a
> hidden reference fails verification in tests immediately.

Its place in the compiler:

```
pkg.Builder (fluent chain API, D-07..D-09)
      │  constructs via exported ir constructors (no MutContext — construction exemption)
      ▼
Level 1: sea of nodes  ── optimization passes mutate this ──►  Level 2: single-block CFG
(internal/ir Graph,            (internal/pass, via             (BasicBlock + Statement list,
 no CFG, no statements)          MutContext choke points)        produced only by Materialize
                                                              or re-ordered by pass.Scheduler)
                                                                      │
                                                                      ▼
                                              canonical printer (this doc §8.3), DOT emitter, future codegen
```

v0 shape facts (IR-02): exactly one basic block per kernel, no control flow,
no phis. The single-tier rule means lowering **introduces node kinds into the
same suite** (the LIR tier of the catalog, §4) rather than switching to a
different IR.

### 1.2 Known drift vs `doc/tca.md` (do not edit `doc/tca.md` here)

The canonical printer was re-spelled in quick-261018-bl0 (task 1): kind names
now render in **lowercase canonical spelling** — `memstore(...)`,
`memload(...)`, `memnew(...)`, `memloadsmem(...)`, `fragstore(...)`, … (D-6 as
implemented: `Kind.String()` is the one source of the spelling).

- **`doc/tca.md` §5.1 predates this sweep**: its example still shows
  `store(%output[@m][@k], %0)` where the printer — and the committed golden
  `pkg/testdata/matmul_scale.ir` — now emit `memstore(%output[@m][@k], %0)`.
  This is a **known drift**; syncing `doc/tca.md` §5.1 is a separate deferred
  doc-only task (flagged as a follow-up in quick-261018-bl0's summary).
- `doc/tca.md` §5.13's node-suite table likewise predates the v3
  re-organization (quick-261018-bl0): `FamilyTileOp` with `loadtile`/`storetile`
  is gone (replaced by the SMEM/FRAG families in `FamilyMemory`) and the file
  column still names the old monolithic `construct.go`/`tile.go` layout. Its
  `dimvar` operand list `[dim, init, step]` happens to match the implemented
  shape again since quick-261019-ktm (the dim binding moved from payload to
  operand 0, §4) — but the table still misses the six restructure kinds
  (`iterclusterproject`, `keyedattribute`, `as`, `lambda`, `tensordimof`,
  `inductionvariable`). Same deferred-sync category.

## 2. File organization

`internal/ir` is layered per the tinycudaopto skeleton re-organization
(quick-261018-bl0 task 1, requirement R4). The rule a new node kind follows:
**add the struct in its family's `node_*.go` file, add a row to
`catalog.go`, add a case to `Graph.allocNode` in `node.go`** — three places,
no enum surgery (append-only, IR-03).

| File | Owns | Why |
|---|---|---|
| `node.go` | The structural core: `Base` cell (ID, kind, `Loc`, `LastTouchedBy`, SSAVar, erased tombstone, surplus count, stmt back-pointer, pinned stmtID, computed-type cell `ty`), the `Node`/`Value`/`ScalarExprNode`/`nodeImpl` interfaces, `operandList` slot storage, the `Graph` root (name, params, `nodes` registry, `end`, `cfg`), `allocNode` (the one kind→struct dispatch), `wireOperands`/`wirePayloadUse`, `NewGraph`, `NewID`/`ReserveStatementID`/`PinStatementID`, `NewNode`, and the nil-safe type/tombstone read faces `TypeOf`/`IsErased`. | One place holds everything every node shares; the dispatch table is the single join point for new kinds (R4a). The `ty` cell is pass state (quick-261019-tu3): written only through `MutContext.SetType`, read via `TypeOf`, never printed/serialized/verified — a computed type is not a chain invariant. |
| `kind.go` | The `Kind` enum (`KindVersion = 5`), `Kind.String()` (lowercase canonical spelling, D-6), `Family` classification (D-15). | Enum + spelling + family only; all other per-kind metadata lives in `catalog.go` (R4e separation). |
| `catalog.go` | The dedicated listing file (R4e): one `kindEntry` per kind — tier (D-3), family, operand arity, one-line doc — plus `HIRKinds()`/`LIRKinds()`/`AllKinds()`/`Tier()`/`fixedArity()`. Rows are in enum order, so `catalog[k-1]` describes `k`. | Pinned invariants (catalog_test.go): `len(AllKinds()) == int(KindEnd)-1`; HIR kinds are exactly the kinds `< KindForLoop` (the tier boundary IS the enum boundary). |
| `doc.go` | Package design notes (the long header comment: optimizer shape, two-level IR, SSA chains, walking, lowered suite, interface-width contract). | Intent lives with the code it governs. |
| `node_lambda.go` | HIR lambda stages: `mapNode` (`ops = [value operands..., As?, lambda]`), `reduceNode` (`ops = [piped, KeyedAttribute, lambda]`), the `lambdaNode` expression node, and the **transient** `Lambda` builder record; read faces `LambdaOf`/`lambdaNodeParams`/`lambdaNodeBody`, `AsOf`, `ReduceInitOf`. | "Pipe" survives only as builder terminology (piped input = reduce operand 0). Since the convention revision the lambda itself is a **`KindLambda` node carried as the stage's trailing operand** — params and body are the lambda node's own operands, never hidden records (the `Lambda` struct exists only between the builder call and `NewMap`/`NewReduce`). |
| `node_expr.go` | HIR scalar-expression suite: `constNode`, `paramRefNode`, `binOpNode`, `unaryOpNode`, `selectNode`, `castNode`, `callNode`, plus the `BinOpCode`/`UnaryOpCode` tables mirroring `doc/syntax.md` §7.6 precedence and the payload read faces `ConstTypeOf`/`CastTypeOf`/`CallNameOf`/`BinOpCodeOf`/`UnaryOpCodeOf`. | Value-level leaves every lambda body is built from; lowering never rewrites them. The payload read faces (quick-261019-tu3) let analysis passes outside package ir read opcodes/types without asserting concrete node types. |
| `node_var.go` | `inputParamNode` and the `ParamRole` taxonomy (`RoleDim`, `RoleTensor`, `RoleScalar`, `RoleLambda`), `RoleOf`, `InputParamTypeOf`/`InputParamNameOf`. | Kernel params are graph-owned (`NewInputParam` appends to `g.params`); lambda params are graph-external by design (inlined at lowering). |
| `node_memory.go` | HIR GMEM family: `memLoadNode` (`ops = [tensor, axes...]`), `memStoreNode` (`ops = [value, tensor, axes...]`), `memNewNode`; the load/store `effectIns` ImpactBy payloads. | The structural tensor access rides **operand slots** since the convention revision (the old `access` payload record is gone) — a store's target components are data deps, its `effectIns` the side-effect deps; the two lists are the node's entire input surface. |
| `node_loop.go` | Loop-structure nodes: HIR `iterNode` (`ops = [dim, bound?, init, step]` — the bound slot exists only in the bounded form), `iterClusterNode` (`ops = [dims..., inputs...]` — dims LEAD the input list; `ClusterDims`/`ClusterInputs` derive the split), LIR `forLoopNode`. | Structure only — the LIR IV *value objects* live in `node_dimvar.go`. |
| `node_iterfmt.go` | The IterCluster-format and lowering bridge nodes (quick-261019-ktm): `iterClusterProjectNode`, `keyedAttributeNode`, `asNode`, `tensorDimOfNode`, `inductionVariableNode`. | The five kinds the Input-list restructure introduced — every one operand-only by construction (the convention's first citizens). |
| `node_dimvar.go` | LIR `dimVarNode` (`ops = [dim, init, step]`), `boundedDimVarNode` (`ops = [iv, max]`) and the IV/BIV relations: `SameIV`, `SameBIV`, `Includes` (`~>`), `BelongsTo` (`<~`), per `doc/optana/induction_variable.md`. | An induction variable is a value object: identity is the `(init, step)` tuple, never the name. |
| `node_lir_memory.go` | LIR memory family: `memLoadSMemNode`, `memStoreSMemNode`, `memNewSMemNode`, `fragLoadNode`, `fragStoreNode`, `fragNewNode`, the `StorageClass` metadata (`StorageGMEM`/`StorageSMem`/`StorageFrag`), `StorageClassOf`. | Every LIR memory node carries the storage class its constructor bound it to (X-4, mirroring tinycudaopto's `memory_class.go`); the buffer is an **operand slot**, the axes are bounded dimvars. |
| `node_end.go` | `returnNode` — the graph's ONE end node (R2): `Graph.end`, `End()`, `NewReturn`, `NewReturnChained`. | First `NewReturn` on a graph becomes `g.end`; later returns are ordinary nodes the verifier rejects (fail-closed multiple-end check). |
| `ssa.go` | The SSA handle layer: `SSAVar`, `SSADef`, `SSAUse`, the `ssaRefRecord` indirection, `UsesOf`/`DefOf`. | Sole pointee of the use-def/def-use relationship (directive 2); see §5. |
| `effect.go` | The side-effect model: `Effect` verdicts, `EffectOf`, `ImpactBy`, the construction `Env`, `NewMemStoreChained`. | The explicit per-tensor ordering edges of the statement-free sea; see §6. |
| `poset.go` | The dependency-poset read face: `DataDeps`, `EffectDeps`, `PosetDeps`, `SeaRoots`. | The primitive edge enumerations Level-1 analysis and the pass scheduler build on (R1); see §8.1. |
| `walk.go` | `Walk`/`Visitor`/`VisitorFunc`/`VisitResult` (Descend/SkipChildren/Stop). | The analysis/checking substrate (directive 3); level-aware roots; NOT the printer's order (D-5 resolution, §8.3). |
| `dce.go` | `BuildingDCE` and both level algorithms: `markReachable`/`buildingDCESea` (Level 1), `markDeadStatements`/`wipeDeadStatements` (Level 2), `isEffectKind`/`hasEffectNode`, `statementOwnership`. | Building debris is wiped before passes see the graph (P1R-08); see §7. |
| `schedule.go` | `Graph.Materialize()` — the Level-1 → Level-2 transition. | Deliberately minimal default scheduling (D-7); optimization scheduling is the pass's job (§8.2). |
| `cfg.go` | Level-2 control-flow shape: `CFG`, `BasicBlock`, `Statement`, the intrusive doubly-linked list (`Append`/`InsertBefore`/`InsertAfter`/`Remove`), `StmtKind` (`StmtAssign`/`StmtEffect`), `NewStatement`, `classifyStmt`. | O(1) removal by pointer (P1R-02); `NewStatement` is the factory passes use after scheduling. |
| `verify.go` | `Verify`, the `use-def`/`def-use` chain invariants, per-kind shape checks (incl. the convention-driven dim/bound and dims-lead checks), the multiple-end-nodes check, `Violation` diagnostics. | The cheap verifier callable after every pass (design.md); fail-closed everywhere. |
| `convention.go` | The Input/ImpactBy-only convention, machine-checked: `hiddenRefFieldType` (type classifier — what counts as a hidden reference) and `checkConvention` (the reflect sweep over every allocated node's struct fields). | Wired into `Verify`; see §1.1's requirement box and §5.2. A forbidden field trips a `convention` violation even when zero-valued — the check is shape-level, fail-closed, and nil types classify hidden. |
| `print.go` | The canonical deterministic printer: `Print`/`PrintOpts`, `Numbering`/`ComputeNumbering`, `StmtNodes`, `chainOf`, `stageText`/`value`, `Label`, `OperandsOf`/`LambdaBodies`/`LambdaParams`, `FormatConst`. | Byte-exact goldens (D-01..D-04, D-13); see §8.3. `LambdaBodies`/`LambdaParams` are DERIVED faces over the lambda node's operands — no stored records behind them. |
| `mutate.go` | `MutContext` and the eight choke points: `SetDef`, `AddUse`, `RemoveUse`, `ReplaceAllUsesWith`, `SetLambdaBody`, `ClearLambdaParams`, `SetType`, `Erase`. | The ONLY sanctioned mutation path outside construction (IR-04, D-17); see §5.3. The two lambda choke points are the sanctioned way to edit a stage's lambda through its node (rewire the body operand; drop dead param operand slots); `SetType` is the computed-type cell's write path (idempotent by design). |
| `types.go` | `ElemType` (syntax.md §3.3 mirror, f8 absent), `DimRef`, the `Type` interface and interned `NumericType`, `InternScalar`/`InternTensor`. | Extensible type interface (directive 1); interning makes interface `==` identity (D-18). |
| `builtins.go` | The pure scalar-builtin signature registry: `BuiltinSignature`, `Builtin(name)` over the fixed `(f32,...f32) -> f32` table (syntax.md §6.4). | Single source of truth shared by `pkg.Call` and the typeinfer pass (quick-261019-tu3); a plain map, never iterated, no runtime mutation path. |

The exported **construction surface** (`NewMap`, `NewReduce`, `NewMemLoad`,
`NewMemStore`, …) is split across the `node_*.go` files beside the node types
they build — the only way to create graphs/nodes from outside the package.
Constructors deliberately do **not** validate kinds or operands: kind
acceptance is the legality window's job (IR-03/IR-07) and operand wiring is
the verifier's job (PASS-06), so cross-package tests can construct
out-of-window or unwired graphs to prove rejection. Construction is exempt
from `MutContext` (RESEARCH Pattern 2) but stamps `PassOrigin` from birth
(D-17).

## 3. The two-level design

The IR has two forms (quick-261017-vdj, tinycudaopto skeleton).

**Level 1 — sea of nodes.** During construction and between passes the graph
is a statement-free sea: `cfg` is nil, `Block()`/`CFG()` return nil, and there
is no implicit "program order" to serialize memory operations. Dependencies
are explicit edges forming a **poset**:

- **data edges** — the operand-slot `SSAUse` records (`Input` data deps);
- **effect edges** — the `ImpactBy` payload on memory nodes and the end node,
  carrying the explicit total order of per-tensor side effects (§6).

**Level 2 — scheduled CFG.** Exactly one `BasicBlock` holding an intrusive
`Statement` list (IR-02), produced **only** by `Graph.Materialize()` after
construction and passes, before codegen — or re-ordered afterwards by
`pass.Scheduler` (§8.2). `Print`/`Label`/DOT refuse an unmaterialized graph
(codegen-adjacent surfaces schedule first); passes operate on the sea and
never need the block.

**Why two levels.** Passes mutate a sea — rewriting in place is cheapest when
nothing carries statement position. Statement order, by contrast, is an
observable, golden-pinned artifact, and choosing a *good* order is an
optimization decision: it belongs to a pass (or to the deliberately minimal,
correctness-preserving default), not to construction. Splitting the levels
also makes side-effect order impossible to get implicitly wrong: at Level 1
you cannot "accidentally" order two stores by list position, because there is
no list — only the explicit `ImpactBy` chains.

Statement identity is stable across passes (D-04): IDs come from the same
monotonic counter as node IDs (IR-05's single `NewID` source — statement
numbering can therefore never drift from node IDs), either pinned at
construction time (`ReserveStatementID` + `PinStatementID`, the builder's
"Scheme X" — required to keep committed goldens byte-identical) or drawn from
the counter at `Materialize` time for unpinned roots.

## 4. Node catalog

`KindVersion = 5` (IR-03). v3 was the deliberate **one-time** tinycudaopto
re-organization (quick-261018-bl0): HIR kinds group before LIR kinds (the
tier boundary is the enum boundary), the generic `Load`/`Store` and
`LoadTile`/`StoreTile` kinds are gone (R4d), and the GMEM + SMEM/FRAG
families take their places. v4 (quick-261019-ktm round 1) added the
Input-list restructure kinds to the HIR tier; v5 (the convention revision)
added `lambda` after `as` — one deliberate LIR-side renumber. From v3 onward
the rule is **append-only within a tier**: new kinds insert before `KindEnd`
in their tier and never renumber existing constants (v4/v5's HIR additions
sit before `KindForLoop`, keeping the tier boundary at the enum boundary).
`KindInvalid` (0) and `KindEnd` (34) are sentinels with no catalog row; the
pass manager's legality windows and the verifier reject both, and any kind
`>= KindEnd` (IR-03/IR-07).

All 33 kinds, exactly as listed in `catalog.go` (enum order; arity `-1` =
variadic, slots allocated from the operand list). **Every row is
Input/ImpactBy-only** (§1.1's requirement): the "Operands" column below IS
the node's complete input surface — the only payload storage left in the
suite is the `effectIns` ImpactBy list of the three effect bearers.

### HIR tier — the abstract kernel graph the builder produces (20 kinds)

| # | Kind (spelling) | Family | Arity | Operands | Semantics |
|---|---|---|---|---|---|
| 1 | `map` | Abstract | -1 | ops = `[value operands..., As?, lambda]` | Lambda map stage, element-wise over its value operands (piped input first on a piped map). The trailing `As` node is the output-axis order (`AsOf`, tca.md §2.7); the trailing `lambda` node closes the list. |
| 2 | `reduce` | Abstract | -1 | ops = `[piped, KeyedAttribute("init"), lambda]` | Lambda reduce stage over the producing map's last `as` axis (tca.md §4.2). The init rides operand 1 as a `keyedattribute` — `ReduceInitOf` unwraps its value operand; operand 0 is ALWAYS the piped input. |
| 3 | `memload` | Memory | -1 | ops = `[tensor, axes...]`; ImpactBy = `effectIns` | HIR GMEM load of `tensor[@m][@n]`; chains into the per-tensor effect order via ImpactBy. (The builder's tensor *reads* are `param` refs — §9.1; `memload` is the explicit form effect linearization materializes.) |
| 4 | `memstore` | Memory | -1 | ops = `[value, tensor, axes...]`; ImpactBy = `effectIns` | HIR GMEM structural store, the stage store. Returns the stored value (tca.md §2.5), which is what keeps fusion pipes intact. The target access rides operand slots 1+ (data deps). |
| 5 | `memnew` | Memory | 1 | ops = `[extent]` | HIR GMEM allocation; the created extent is a Value. |
| 6 | `iter` | Loop | -1 | ops = `[dim, bound?, init, step]` | Abstract induction variable `@m`, optionally bounded `@m/M` (tca.md §2.3) — the bound slot exists only in the bounded form. Shape check: operand 0 is a dim InputParam. |
| 7 | `itercluster` | Loop | -1 | ops = `[dims..., inputs...]` | Compound header `iter(@m, @k, @n)` = the Cartesian product of its dims (tca.md §2.4). The dim InputParams LEAD the input list; `ClusterDims`/`ClusterInputs` derive the split; shape check enforces dims-lead. |
| 8 | `const` | ScalarExpr | 0 | payload = interned type + value (metadata) | Typed scalar constant; floats print shortest-round-trip f32 with `.0` appended (`FormatConst`). |
| 9 | `param` | ScalarExpr | 1 | ops = `[the referenced InputParam]` | Scalar-expression parameter reference. A reference to a **tensor** param renders structurally (`lhs[@m][@n]` — the axes come from the tensor type's dims); scalar/lambda params render bare (`scale`, `v`). Shape check: operand 0 is an InputParam. |
| 10 | `binop` | ScalarExpr | 2 | ops = `[a, b]`; payload = opcode | Binary scalar op; the opcode table mirrors syntax.md §7.6 precedence. |
| 11 | `unaryop` | ScalarExpr | 1 | ops = `[x]`; payload = opcode | Unary scalar op (`+`, `-`, `!`, `~`). |
| 12 | `select` | ScalarExpr | 3 | ops = `[cond, then, else]` | Typed conditional `cond ? then : else`. |
| 13 | `cast` | ScalarExpr | 1 | ops = `[x]`; payload = target type | Explicit cast to a named scalar type; renders `cast<f32>(x)`. |
| 14 | `call` | ScalarExpr | -1 | ops = args; payload = builtin name | Registered **pure** builtin call. |
| 15 | `inputparam` | Abstract | 0 | payload = name (debug only), interned type (nil for dims), role | Graph-owned kernel/lambda input parameter. The node's `SSAVar` IS the parameter's def; every reference elsewhere is an `SSAUse`. |
| 16 | `return` | Abstract | -1 | ops = returned values; ImpactBy = `effectIns` | **The graph's single end node** (R2): the first `NewReturn` becomes `Graph.end`; carries return values AND every mem-param's final effect-chain state via ImpactBy (§6.3). The none Value — binds no SSA name. |
| 17 | `iterclusterproject` | Abstract | 1 | ops = `[cluster]`; payload = index (metadata) | **IterCluster-format bridge** (quick-261019-ktm): binds the cluster's input slot *i* to a Map's parameter *i* — `Map(..., IterClusterProject(cluster, 0), ...)` before lowering. |
| 18 | `keyedattribute` | Abstract | 1 | ops = `[value]`; payload = key string (metadata) | A named attribute riding an Input list — Reduce's `init`. The VALUE is an operand (a data dep); the name is metadata. |
| 19 | `as` | Abstract | -1 | ops = `[dim params in order]` | The as-order clause, carried as the LAST VALUE operand of its Map/Reduce (before the trailing lambda) — `AsOf(n)` scans the Input list, no stored pointer. Shape check: operands are dim InputParams; at most one, placed last. |
| 20 | `lambda` | Abstract | -1 | ops = `[param declarations..., body]` | The lambda expression, carried as the TRAILING operand of its Map/Reduce. The params (RoleLambda InputParams) bind positionally to the stage's Input-list value bindings; the body is the closing operand. The verifier permits only Map/Reduce to carry one. |

### LIR tier — the lowered suite introduced by optimization passes (13 kinds)

| # | Kind (spelling) | Family | Arity | Operands | Semantics |
|---|---|---|---|---|---|
| 21 | `forloop` | Loop | -1 | ops = `[iv(dimvar), bound, body, KeyedAttribute annotations...]` | Explicit lowered loop nest. Shape check: operand 0 must be a dimvar; trailing operands beyond slot 2 must be KeyedAttribute annotations (M1, quick-261021-jay — the tile-lower pass's soc/dosoc/loc materialization metadata). |
| 22 | `dimvar` | Abstract | -1 | ops = `[dim, init, step]` | Lowered induction variable — a value object; identity is the `(init, step)` tuple, never the name (doc/optana/induction_variable.md). Prints `@x(init=…, step=…)`. Shape check: operand 0 is a dim InputParam. |
| 23 | `bdimvar` | Abstract | 2 | ops = `[iv(dimvar), max]` | Bounded IV `((iv tuple), Max)`; prints `@x/M`. Shape check: operand 0 must be a dimvar. |
| 24 | `memloadsmem` | Memory | -1 | ops = `[src smem buffer, bivs...]`; class = `StorageSMem` | Lowered SMEM load. Shape check: src must be a `memnewsmem`. |
| 25 | `memstoresmem` | Memory | -1 | ops = `[value, dst smem buffer, bivs...]`; class = `StorageSMem` | Lowered SMEM store; returns the stored value (HIR contract, tca.md §2.5). |
| 26 | `memnewsmem` | Memory | 1 | ops = `[extent]`; class = `StorageSMem` | SMEM allocation. |
| 27 | `fragload` | Memory | -1 | ops = `[src frag buffer, bivs...]`; class = `StorageFrag` | Lowered register-fragment load. Shape check: src must be a `fragnew`. |
| 28 | `fragstore` | Memory | -1 | ops = `[value, dst frag buffer, bivs...]`; class = `StorageFrag` | Lowered fragment store; returns the stored value. |
| 29 | `fragnew` | Memory | 1 | ops = `[extent]`; class = `StorageFrag` | Fragment allocation. |
| 30 | `tensordimof` | Abstract | 2 | ops = `[tensor param, TU32 axis const]` | A tensor dimension's upper bound — the bound the lowering gives an `iter`. Shape checks: operand 0 a RoleTensor InputParam, operand 1 a const. |
| 31 | `inductionvariable` | Abstract | 1 | ops = `[iter]` | The u32 current-iteration value of an iter — what a lowered `MemLoad`'s axes become. Shape check: operand 0 is an iter. |
| 32 | `intrinsicgridmap` | Abstract | 1 | ops = `[DoSoC const u32]`; payload = field string | Launch-map query `IntrinsicGridMap(field, DoSoC)` (quick-261021-jay, §5 of the loop-tile-lower spec): the enclosing LNT's GridConcurrencyMap, field ∈ {index, dimension}, result u32. Shape checks: exactly one const-u32 operand, field vocabulary. Exists only between tile-lower and codegen. |
| 33 | `intrinsicthreadmap` | Abstract | 1 | ops = `[DoSoC const u32]`; payload = field string | The same query over the LNT's ThreadMap. Same shape contract as `intrinsicgridmap`. |

Notes:

- **Arity is enforced fail-closed**: fixed-arity kinds record operands supplied
  beyond the slot count as `Base.surplus`, which the verifier flags under the
  use-def invariant (WR-01). Variadic kinds wire every operand.
- **Shape checks** (verify.go, fail-closed): HIR GMEM load/store tensors bind
  a `RoleTensor` `inputparam`; SMEM buffers bind a `memnewsmem`; FRAG buffers
  bind a `fragnew`; `bdimvar` and `forloop` bind a `dimvar` at operand 0;
  `iter`/`dimvar` bind a dim InputParam at operand 0; `param` binds an
  InputParam at operand 0; `iterclusterproject` binds an itercluster;
  `inductionvariable` binds an iter; `tensordimof` binds a tensor param + a
  const axis; `as` binds dim InputParams, at most one, last value operand of
  its stage; `reduce`'s operand 1 is its `KeyedAttribute`; a map's value-operand
  count matches its lambda's declared parameter count while params remain.
- **The Input/ImpactBy-only convention sweep** (convention.go) runs on every
  `Verify` — see §5.2.
- The **families** (D-15): `FamilyScalarExpr` (8–14), `FamilyMemory` (3–5,
  24–29), `FamilyLoop` (6, 7, 21), `FamilyAbstract` (1, 2, 15–20, 22, 23,
  30–33 — the lowered dimvars/IVs/bounds and the launch-map queries are
  value-level objects, not
  loop-nest structure; the restructure kinds are value-level Input-list
  citizens), `FamilyCollective` (reserved, no members until reduce lowering
  lands).
- The pass manager rejects nodes with `Kind == KindInvalid` or
  `Kind >= KindEnd` (forward-compatibility within our own pipeline, IR-03).

### 4.1 The IterCluster format and Simplification (quick-261019-ktm)

The frontend-facing Map/Reduce shape above comes in two formats:

- **IterCluster format** (what a frontend builds): the map's inputs are
  `iterclusterproject` bridges into an `itercluster` whose inputs are the
  `memload`s over the raw dim params.
- **Iter format** (canonical): `iter(dim, tensordimof(tensor, axis))` →
  `inductionvariable(iter)` → `memload(tensor, ivs...)` → Map/Reduce — the
  loop is gone; everything is ordinary SSA data flow.

The **Simplification** passes (`internal/opt/simplify`, pipeline slots
102/103/104, between dce-verify and gvn) lower the first into the second and
finish the job: `simplify-lower` rewrites clusters into iter/IV form,
`simplify-inline` replaces every lambda-parameter reference with its
positional Input-list binding (a KeyedAttribute element contributes its
wrapped value; trailing As/lambda nodes bind nothing) and strips the dead
parameter declarations, and `simplify-verify` asserts no live projects remain
and every body is fully inlined. See `doc/optana/lower_itercluster.md` for
the source feature request.

Right after Simplification, the **type inference** pass (`internal/opt/
typeinfer`, pipeline slot 105, between simplify-verify and gvn —
quick-261019-tu3, `doc/optana/type_inf.md`) stamps every non-erased node's
computed type onto the `Base.ty` cell through `SetType`, readable by any
later pass via `ir.TypeOf`. One memoized post-order pass suffices (SSA, no
phis, no back edges — an operand cycle fails closed); the pass is
idempotent and re-runnable after future lowerings. The closed untyped set —
`memnew`/`memnewsmem`/`fragnew` (allocations carry no dtype), `as` (an
output-order clause, not a value), `lambda` (no function Type yet), and a
`return` with 0 or >1 values — deliberately stays nil.

## 5. The SSA handle layer (IR-01, D-16)

### 5.1 Handles over one pointee

Every node owns exactly one `SSAVar`, allocated 1:1 at node allocation
(`allocNode`, ID == node ID). The var stores the node's defining `Value` and
the list of ref records for uses pointing **at** this node:

```
SSAVar { id, node (back-pointer), define (Value), refs []*ssaRefRecord }
ssaRefRecord { def *SSAVar   // the def this use currently resolves to
               user Node     // back-pointer to the owning node
               killed bool } // set by RemoveUse: detached from both directions
```

`SSADef` and `SSAUse` are **value-type handles** over that shared pointee: a
def wraps the var, a use wraps one ref record. Handles are copied freely
across passes; because a use handle points at the *record* rather than
directly at the var, **rewiring the record's `def` pointer propagates the new
target to every copy of the handle** — the reverse-operation requirement of
directive 2. `UsesOf(def)` returns a snapshot of live uses in link order (the
slice is rebuilt per call, so callers may hold it across splice-during-
iteration mutations — RESEARCH Pitfall 2); `DefOf(u)` resolves a use to its
def node.

Invalid/zombie states are representable but checkable: zero handles, killed
records, and unwired records all read as dead through the nil-safe accessors,
and the verifier flags them structurally under the use-def invariant.

### 5.2 The Input/ImpactBy-only convention (and its enforcement)

Every reference in the graph is an `SSAUse`, and every `SSAUse` a node owns
lives in exactly one of its two dependency lists (§1.1's requirement):

- **Input (operand slots)** — the data dependencies, stored in the node's
  embedded `operandList` and wired by `wireOperands`. Since the convention
  revision this includes everything the pre-revision suite carried as
  payload: parameter refs (`param`), iteration dims/bounds (`iter`,
  `dimvar`), a cluster's dims, a store's target access (tensor + axes),
  map/reduce as-orders (the `as` node), reduce's init (the `keyedattribute`
  node's value), lambda params and lambda **bodies** (the `lambda` node's
  operands), IVs and tensor bounds.
- **ImpactBy** — the side-effect dependencies: a direct `effectIns []SSAUse`
  field on the three **effect bearers** (`memload`, `memstore`, `return`),
  wired by `wirePayloadUse` and enumerated via `payloadBearer`/`effectBearer`.
  This is the ONLY sanctioned payload spelling left in the suite.

The practical consequences passes rely on: the verifier's use-def/def-use
invariants and `Walk` cover **every** reference in both chain directions
with zero special cases; `DataDeps`/`PosetDeps` see store targets and
as-orders as ordinary data edges; DCE, GVN, the printers, and DOT see lambda
bodies and wrapper nodes (`as`, `keyedattribute`, `lambda`) as ordinary
graph nodes. GVN keys operands — the lambda included — **by identity**
(amendment D): two stages are one value only when they share operand NODES,
never by structural recursion through bodies.

**Machine-checked (D16).** `Verify` calls `checkConvention`
(`convention.go`): a reflect sweep over every allocated node's concrete
struct fields — skipping the embedded `*Base` machinery cell and the
embedded `operandList` (the Input list itself) — flagging any field whose
type can carry SSA/Node references. The classifier `hiddenRefFieldType` is
type-level and fail-closed: a direct `[]SSAUse` field is legal only on an
effect bearer; a nested record (e.g. the transient `Lambda` struct) is
hidden even there; a nil type classifies hidden. A violation reports as
invariant `convention`, "`field %q carries SSA/node references outside the
Input/ImpactBy lists`" — including for forged nodes no constructor can build
(the sweep reads `g.nodes`, not the root walk).

### 5.3 Mutation choke points (IR-04, D-17)

All chain edits outside construction flow through `MutContext` — eight
operations, each stamping `LastTouchedBy = ctx.Pass` on every node it touches
so PASS-06 diagnostics name the responsible pass structurally (with the one
deliberate exception: `SetType` stamps only when it actually writes):

| Choke point | Effect |
|---|---|
| `SetDef(u, newVal)` | Redirects one operand use: physically moves its ref record from the old def's var to the new one (nil unwires the slot). Every handle copy observes the new target. |
| `AddUse(user, val)` | Appends one operand slot and links it into val's ref list; returns the new handle. |
| `RemoveUse(u)` | Detaches u from its def's ref list and drops u's slot, **killing** the record so every copy reads dead. A killed record still occupying a slot is exactly the corruption class the verifier flags by node and slot. |
| `ReplaceAllUsesWith(old, new)` | RAUW, LLVM-style: relocates every live ref record of `old` into `new`'s var in one pass over a snapshot; self-RAUW is a no-op. |
| `SetLambdaBody(stage, newBody)` | Rewires a Map/Reduce's lambda node's trailing body operand (the sanctioned body edit — an identity-lambda body root cannot be reached by RAUW over its paramRefs). |
| `ClearLambdaParams(stage, keep)` | Drops the lambda node's dead parameter operand slots; `keep(p)` decides which declarations survive (a param with uses beyond its own declaration slot stays). |
| `SetType(n, ty)` | Writes n's computed-type cell (`Base.ty`, quick-261019-tu3). **Idempotent by design**: when n already carries an Equal type it neither writes nor re-stamps, so a re-run of the type-inference pass leaves `LastTouchedBy` stable. |
| `Erase(n)` | Removes n from the graph: kills **operand and payload records alike** (a DCE-erased node must not leave live-looking payload records in a param's ref list — the quick-261017-vdj Erase fix), unlinks n's statement from the block in O(1) through the stmt back-pointer, and tombstones the ID so no later allocation can reuse it (IR-05). Refuses while any live use remains. |

The var-level primitives (`replaceDefine`, `replaceAllReference`, `addRef`,
`removeRef`) are unexported — reachable only from the choke points and from
construction.

## 6. The effect model (ImpactBy + the one end node)

Between construction and Materialize there is no statement list, hence no
implicit program order. The total order of side effects is carried explicitly.

### 6.1 Effect verdicts (X-3)

`EffectOf` is deliberately coarse — free vs. full, delivered as a function to
keep the `Node` interface at exactly two methods (directive 4): a **GMEM
`memstore`** and the kernel's **`return`** are fully effectful; **every other
kind is free** — including the entire lowered SMEM/FRAG memory family, which
is in-program memory, not observable kernel I/O (R3), and is therefore
dead-code-eligible when unreferenced.

### 6.2 Per-tensor effect chains and the restrict-alias rule

A memory access records its per-tensor ordering edge in its **`effectIns`**
payload — the `ImpactBy` uses, exported through `ImpactBy(n)`. Chains
**include reads as well as writes** (tinycudaopto skeleton memory.go): the
env-aware constructors `NewMemStoreChained`/`NewMemLoadChained` wire
`ImpactBy = the tensor's current chain head` and then become the new head, so
two accesses to one tensor form an explicit second-ImpactBy-first edge while
distinct tensors stay isolated.

The chains are maintained by a construction-time **`Env`**: one isolated
chain per kernel tensor, keyed by the tensor's `RoleTensor` `inputparam` node
pointer (`map[*Base]Node` — identity is never a name, P1R-01). This works —
concurrently tracking isolated side-effect chains per symbol — **solely
because of the restrict-alias rule**: kernel tensors never alias; two distinct
tensor parameters never name the same memory. It is a documented construction
precondition; enforcement arrives with the frontend, exactly like the
skeleton. A nil `*Env` leaves accesses un-chained (identical to the plain
constructors).

### 6.3 The ONE end node (R2)

The graph has exactly one end node: the **first** `NewReturn` becomes
`Graph.end`. `NewReturnChained(origin, env, tensors, values...)` additionally
wires one `ImpactBy` edge per tensor in `tensors` pointing at that tensor's
**current chain head** (`Env.FinalStates`/`ChainHead` — heads only, never
intermediate links); a tensor with no recorded access contributes no edge. The
returned values stay ordinary operand slots.

So the single end node carries the kernel's whole observable behavior:

- **operands** — the return values (the builder passes the stores themselves,
  since a store returns the stored value, tca.md §2.5 — printing `%output`);
- **`ImpactBy` payload** — every mem-param's final chain state, so the whole
  per-tensor store chain stays live through the end node alone.

The verifier fails closed on a second reachable `KindReturn` (multiple
end nodes violation). `Graph.End()` is the exported read face; the end node is
the sea's second liveness root beside the `RoleTensor` params (§7).

## 7. DCE: two-root-set liveness (R3, P1R-08)

`BuildingDCE(g)` runs at Build time **before** the pass pipeline, so passes
never observe construction debris (an abandoned map before a store, a reduce
whose result is overwritten). It is level-aware.

**Level 1 (sea mark-sweep).** `markReachable` seeds from the **end node plus
ALL `g.params`** and follows exactly `Walk`'s child relation (operand slots —
which, since the Input/ImpactBy-only convention, already include lambda
nodes, their params and bodies, wrapper nodes, and store targets — plus the
`ImpactBy` payload uses). The seeding is deliberately
broader than the observable root set: the def-use verifier audits every
graph-owned param's ref list, so dim/scalar signature params must never be
swept even though they are not observable roots (the quick-261018-bl0
deviation — `SeaRoots()` still reports the narrower R3 set for analysis
consumers). The sweep walks `Graph.nodes` (the allocation registry) in
allocation-order fixpoint passes: a node that is unmarked **and** has zero
live uses is erased through `MutContext.Erase` — killing a dead user kills its
operand/payload records, freeing its defs for the next pass. A pass that
erases nothing ends the loop.

**The two-root set (what liveness means at Level 1).** `SeaRoots()` =
the `RoleTensor` params (declaration order — dims and scalars are not
observable roots), then the end node. Consequences:

- **Stores are never roots.** A store stays live through the end node's
  operand/`ImpactBy` paths; a store no end path reaches is unreachable
  building debris and is wiped like any other dead node.
- **SMEM/FRAG state is unobservable** unless it feeds the end node: the
  lowered stores are effect-free per `EffectOf` (§6.1), so unreferenced
  SMEM/FRAG computation is dead-code-eligible.

**Level 2 (statement fixpoint).** After Materialize, the review's original
three-step statement model runs unchanged for passes that operate post-
scheduling: seed live roots (`StmtEffect` statements or statements whose tree
`hasEffectNode`), propagate liveness backward along def-use edges through the
fixpoint (`markDeadStatements` — uses owned by no statement fail conservative),
then wipe dead statements' owned nodes in deterministic passes
(`wipeDeadStatements`; statement ownership = the first statement in list order
whose subtree reaches a node, so cross-statement defs are attributed to the
earlier statement and never wiped with the later dead consumer).

**Conservative discipline (both levels).** A node with live uses is skipped,
never force-erased; cyclic dead clusters where every member keeps one
live-looking use survive untouched (no corruption, residual deadness).
Builder-produced graphs share only statement roots across statements, so for
them the wipe is total.

## 8. Poset + scheduling

### 8.1 The poset read face (R1)

`poset.go` is the primitive edge enumeration of the sea, built for Level-1
analysis and consumed by the pass scheduler:

| Function | Returns |
|---|---|
| `DataDeps(n)` | The live operand-slot defs of n, in slot order — n's data "must come after" set. Since the convention this includes a store's target-access components (`[value, tensor, axes...]`) — they are data deps now. |
| `EffectDeps(n)` | The live `ImpactBy` payload defs of n — the explicit per-tensor ordering edges. Non-effectful nodes carry none. |
| `PosetDeps(n)` | The flat union: data edges (slot order) then effect edges — the single list a topological ordering consumes. |
| `g.SeaRoots()` | The Level-1 root set: `RoleTensor` params (declaration order) + the end node (§7). |

Edges point **use → def**. All enumerations skip killed/unwired records (the
handle model's checkable corruption classes stay checkable, never
traversable) and are deterministic (slot order, then payload order — never
map iteration, IR-05/RESEARCH Pitfall 1). One scheduling-level note: deps on
`inputparam` nodes impose no statement ordering (params are entry-available
leaves produced by no statement) — the pass scheduler skips them explicitly
when building statement edges, because shared tensor/dim params otherwise
manufacture false cross-statement cycles against the effect chains.

### 8.2 Scheduling: default `Materialize` (D-7) vs. `pass.Scheduler` (R5)

**`Graph.Materialize()` — the deliberately minimal default** (ir keeps only
this; optimization scheduling is the pass's job). v0 kernels have exactly one
basic block and no control flow (IR-02), so "scheduling" is: one statement per
**live effectful node** (`isEffectKind`: the GMEM/SMEM/FRAG store family and
the end node — the same set `EffectOf` calls full, plus the lowered stores)
reachable from the sea roots, in node **allocation order** (which is
construction order). Because reachability from the end node is the liveness
test, statement order equals the construction order of the side effects.
Each statement carries its `PinStatementID`-reserved ID or the next ID from
the graph's monotonic counter (D-04/IR-05; builder pinning — Scheme X — is
what keeps statement IDs dense across the chain nodes exactly as the
pre-scheduler flow produced them, pinned byte-for-byte by the committed
goldens). Refused on a nil graph, a second call, or an existing CFG. Shared
pure values do **not** become statements: statement boundaries exist only at
live effectful nodes, and everything reachable from a statement root renders
inside that root's tree.

**`pass.Scheduler` — the optimization pass** (quick-261018-bl0 task 3). An
ordinary `Pass` (PASS-01/IR-07): ID `"schedule"`, no requirements, produces
`StageConstructed`, preserves nothing. `Run` is three statement-level steps
built exclusively from the exported ir read face (`StmtNodes`, `OperandsOf`,
`PosetDeps`, `LambdaBodies`, `DataDeps`, `EffectOf`, `Base.Statement`) — the
pass package never reaches into ir internals:

1. **Stable topological sort.** A statement's predecessors are the statements
   owning the defs its tree depends on (data slots then ImpactBy payload,
   via `PosetDeps`). Statement ownership: a node rooting a statement
   (`Base.Statement() != nil`) is owned by exactly that statement — which is
   what makes pass-inserted statements (LICM hoists) win over the trees that
   still reference the hoisted node; any other node is owned by the first
   statement in block order whose `StmtNodes` tree contains it. Among ready
   statements the smallest list position wins, so statements with no ordering
   constraint keep their pre-existing relative order — on an already-ordered
   list the sort is the identity. Internal tree edges contribute nothing,
   except a genuine self-dependency, which is kept and makes the statement
   unschedulable. A cycle remainder **fails closed** with an error and leaves
   the block untouched.
2. **LICM** over lowered `forloop`s, innermost-first, candidates in
   `StmtNodes` order. A candidate must be: whitelisted
   (`{Const, InputParam, Param, BinOp, UnaryOp, Select, Cast, Call}` — the
   plan's safety set), **non-leaf** (it must bind at least one operand —
   hoisting bare const/param leaves would add no computation and pollute the
   schedule), effect-free, unpinned (its data-dependency closure touches no
   memory/loops/IV objects/effectful nodes; cycles count as pinned), not
   already a statement root (idempotence), and invariant with respect to
   **every** enclosing loop — neither varying nor an ancestor of a varying
   node. Hoisting splices a new statement through the same `NewStatement`
   factory Materialize uses, inserted **immediately before the earliest
   statement whose tree already contains the candidate** (consumers are met
   before producers in `StmtNodes` order, so producers land left of consumers
   and the ownership map stays acyclic for the closing topo).
3. **Second stable topo sort** to fold the hoisted statements into dependency
   order.

The apply step is collect + relink (`Remove` all, then `Append` in order);
statement IDs are stable (D-04) — only list order changes. The intrusive-list
back-pointer invariant: `Remove` clears the root's `stmt` back-pointer, and
`Append`/`InsertBefore`/`InsertAfter` restore it — so a reschedule round-trip
leaves every statement root's lookup intact (the quick-261018-bl0 symmetric
invariant fix).

### 8.3 Printer vs. Walk (D-5 resolution)

`Walk` (walk.go) is the **analysis/checking** substrate: iterative pre-order
DFS with a visited set (cyclic-safe), roots level-aware (Level 1: params in
declaration order + the end node; Level 2: params + the block's statements in
statement order, def then rhs), children in operand-slot order then payload
order then lambda bodies, with `Descend`/`SkipChildren`/`Stop` control. It
must **never** be used where emission order matters.

The canonical printer (print.go) does NOT walk that way. Its `StmtNodes` DFS
is an **emission-order function**: per statement, `chainOf` collects the stage
chain producer-first by following `pipeInput` backward (stopping at other
statements' roots, non-stage inputs, or cycles), then each stage's operand
subtrees depth-first, lambda bodies last. That order is golden-pinned (D-01..
D-04): `statement #N:` headers with stable IDs, stages joined by ` |`, 2-space
continuation indent, one blank line between statements. SSA `%N` numbers are
recomputed from scratch on every print by `ComputeNumbering` (definition
order over the materialized statement list — never cached), and a node that
roots a *different* `StmtAssign` statement renders as `%N` from other
statements' trees (the statement-boundary rule that makes LICM hoists print as
`%0 * 3.0`, never `(%0) * 3.0`). The DOT emitter shares `StmtNodes`,
`OperandsOf`, `Label` so `.dot` mirrors the text numbering (D-05).

## 9. Conceptual code → IR: worked examples

Every IR snippet in this section is either **verbatim from a committed
golden** or **derived from the printer rules of §8.3** (`chainOf` stage
collection, `stageText`/`value` rendering, `%0` pipe binding). Lines that are
illustrative-only — not derivable from print.go plus a golden — are marked
`# illustrative`.

### 9.1 The matmul_scale oracle (real golden, verbatim)

Conceptual kernel (doc/tca.md §5): `output[m][k] = ( Σ_n lhs[m][n] * rhs[n][k] ) * scale`.
Built through the public builder (`pkg/oracle_test.go: buildMatmulScale`):

```go
b := pkg.NewBuilder("matmul_scale")
m, k, n := b.Dim("m", 0), b.Dim("k", 0), b.Dim("n", 0)
lhs := b.Tensor("lhs", ir.TF32, m, n)
rhs := b.Tensor("rhs", ir.TF32, n, k)
out := b.Tensor("output", ir.TF32, m, k)
scale := b.ScalarParam("scale", ir.TF32)

root, _ := b.Stmt().Iter(m, k, n).
    Map([]pkg.Operand{pkg.At(lhs, m, n), pkg.At(rhs, n, k)},
        func(l, r pkg.Expr) pkg.Expr { return b.Mul(l, r) })
mid := root.As(m, k, n)
ar, _ := mid.Reduce(b.Float32(0.0),
    func(acc, v pkg.Expr) pkg.Expr { return b.Add(acc, v) })
am, _ := ar.Map(nil, func(v pkg.Expr) pkg.Expr { return b.Mul(v, scale) })
tail := am.As(m, k)
_, _ = tail.Store(out, m)
graph, _ := b.Build()
```

The committed golden `pkg/testdata/matmul_scale.ir`, quoted **verbatim**
(byte-for-byte, 8 lines):

```
statement #8:
  iter(@m, @k, @n) -> map(lhs[@m][@n], rhs[@n][@k], lambda(l, r): l * r) as (@m, @k, @n) |
  reduce(init=0.0, %0, lambda(acc, now): acc + now) |
  iter() -> map(%0, lambda(v): v * scale) as (@m, @k) |
  memstore(%output[@m][@k], %0)

statement #9:
  return %output
```

Node-by-node / stage-by-stage:

- **Parameters.** The seven kernel params are InputParam nodes 1–7 in
  declaration order: `@m, @k, @n, lhs, rhs, output, scale`. The tensor *reads*
  in the root map are `param` ref nodes (§4 row 9) — the builder never emits
  `memload` at this level; `memload` is the explicit form effect
  linearization (future pass) materializes. The refs render structurally:
  `lhs[@m][@n]` — the axes come from the tensor type's dims, not from stored
  names.
- **Statement IDs #8/#9 (Scheme X, D-04).** `Stmt()` reserves **two** IDs
  before any chain node exists — one pinned on the store, one on the end
  node — so statement IDs stay dense across the chain's 22 nodes (IDs 10–31)
  and the return node (32). Without the reservation the statements would
  print as ≥ #33, and the doc-pinned oracle would drift. Materialize consumes
  the pinned IDs; the golden pins them byte-for-byte.
- **Stage 1** — `iter(@m, @k, @n) -> map(...)` is one printed stage: the
  iteration cluster (node with the three dim payload uses) is transparent for
  piping, so it renders its body map inline. The map's operands are the two
  tensor refs; the lambda `lambda(l, r): l * r` is a `Lambda` payload record
  whose params `l, r` are graph-external `RoleLambda` InputParams and whose
  body is a `binop` node, canonically re-rendered (D-02). The mandatory
  `as (@m, @k, @n)` is the map's output-order payload (tca.md §2.7); `@n`
  last means `@n` is the reduced axis.
- **Stage 2** — `reduce(init=0.0, %0, lambda(acc, now): acc + now)`. The
  reduce's piped operand (slot 1) is the stage-1 cluster; because it is
  pipe-bound to the previous stage it renders as `%0`.
- **Stage 3** — `iter() -> map(%0, lambda(v): v * scale) as (@m, @k)` is the
  piped map after the reduce, wrapped in a **zero-arity** iteration cluster
  (`iter()`) — exactly the fused-tail form of tca.md §2.5. `scale` is a
  captured kernel scalar, a `param` ref rendering bare.
- **Stage 4** — `memstore(%output[@m][@k], %0)`. The store's target is the
  `access` payload (tensor use `%output` + axis uses `@m, @k`); its value
  operand is the piped stage-3 cluster, printing `%0`. The store is chained
  into `%output`'s per-tensor effect chain by `NewMemStoreChained` (§6.2) —
  here it is the first (and only) access, so its `ImpactBy` payload is empty.
- **Statement #9** — `return %output` is the ONE end node (R2). Its operand
  is the **store node itself** (the builder passes stores as return values,
  and a store returns the stored value — which is why it renders `%output`,
  tca.md §2.5). Its `ImpactBy` payload carries `%output`'s final chain state
  — the store — so the store stays live through the end node alone (R3: the
  store is reachable from a root, so building DCE never sweeps it).
- **`%0` everywhere.** The `%0` in stages 2, 3, and 4 is the **pipe
  variable**, stage-local per tca.md §2.6 — each is that stage's piped input
  bound to `%0`, not an SSA definition-order number. Definition-order `%N`
  numbering (§8.3) never appears in this golden because every operand renders
  by name or by pipe. (This is also why the reduce's input and the scale
  map's input can both print `%0`: two different stages, two different
  bindings of the same pipe notation.)
- The graph view of the same oracle is committed at
  `internal/visualization/testdata/matmul_scale.dot` (same numbering, D-05).

### 9.2 Teaching example: trivial scale kernel

Conceptual: `out[i] = in[i] * 2.0` with one dynamic dim — the smallest legal
chain (a root map must flow through nothing before a store in v0: Iter → Map
→ As → Store).

```go
b := pkg.NewBuilder("scale2")
m := b.Dim("m", 0)
in := b.Tensor("in", ir.TF32, m)
out := b.Tensor("out", ir.TF32, m)
root, _ := b.Stmt().Iter(m).
    Map([]pkg.Operand{pkg.At(in, m)},
        func(v pkg.Expr) pkg.Expr { return b.Mul(v, b.Float32(2.0)) })
_, _ = root.As(m).Store(out, m)
graph, _ := b.Build()
```

Derived IR (derivation below; every line follows from printer rules):

```
statement #4:
  iter(@m) -> map(in[@m], lambda(v): v * 2.0) as (@m) |
  memstore(%out[@m], %0)

statement #5:
  return %out
```

Derivation:

- **IDs.** Params `@m, in, out` are nodes 1–3. `Stmt()` reserves 4 (pinned on
  the store) and 5 (pinned on the end node) before the chain's nodes exist —
  so `statement #4` and `statement #5` regardless of the nine chain nodes
  that follow (paramRef `in`, lambda param `v`, paramRef `v`, const `2.0`,
  binop, map, itercluster, store, then the return at Build time).
- **Stage 1.** The map's single operand is a `param` ref to the `in` tensor;
  a non-scalar tensor ref renders from the type's dims → `in[@m]`. The
  lambda body `v * 2.0` is a `binop` (Mul) re-rendered with §7.6 precedence:
  both operands are atoms (prec 12 > Mul's 10), so no parens (D-02). A
  single-parameter map lambda is named `v` by the builder's name table
  (chain.go).
- **Stage 2.** The store's value operand is the piped stage-1 cluster;
  pipe-bound to the previous stage it renders `%0`. The target renders
  `%out[@m]` from the access payload.
- **Statement #5.** The end node returns the store (which returns the stored
  value) → `%out`.

### 9.3 Teaching example: two-store effect chains (ImpactBy made visible)

The text printer does **not** render `ImpactBy` edges: a store renders
through its own access rule (target + value only) and the end node renders
its value operands only. (This is why the one-end-node wiring of
quick-261018-bl0 task 2 needed **no** golden change — statement #9 above
stayed `return %output` in shape.) The effect order therefore has to be read
from the graph, not the print. This example states both.

Conceptual kernel — two independent outputs, one chain each:

```go
b := pkg.NewBuilder("two_out")
m := b.Dim("m", 0)
in := b.Tensor("in", ir.TF32, m)
out1 := b.Tensor("out1", ir.TF32, m)
out2 := b.Tensor("out2", ir.TF32, m)
mk := func() *pkg.Stored { // one scale-by-2 chain, stored by the caller
    r, _ := b.Stmt().Iter(m).
        Map([]pkg.Operand{pkg.At(in, m)},
            func(v pkg.Expr) pkg.Expr { return b.Mul(v, b.Float32(2.0)) })
    return r.As(m)
}
_, _ = mk().Store(out1, m)
_, _ = mk().Store(out2, m)
graph, _ := b.Build()
```

Derived IR (same derivation as §9.2, one more param and one more chain):

```
statement #5:
  iter(@m) -> map(in[@m], lambda(v): v * 2.0) as (@m) |
  memstore(%out1[@m], %0)

statement #15:
  iter(@m) -> map(in[@m], lambda(v): v * 2.0) as (@m) |
  memstore(%out2[@m], %0)

statement #6:
  return %out1, %out2
```

- **IDs.** Params are nodes 1–4; chain 1 reserves 5 (store) and 6 (end
  node); its nine nodes follow; chain 2 reserves **15** (store) and 16
  (unused — only the first output binding's end-node reservation is consumed,
  R2) before its nodes. Pinned IDs can therefore print **out of sorted
  order** (`#5, #15, #6`): list order is allocation order (§8.2), IDs are
  pinned (D-04).
- **The return.** `Build` emits ONE end node over all output bindings, in
  first-store order: values `store1, store2` (rendering `%out1, %out2`).

Now the part the print hides — the `ImpactBy` edge list for this graph
(edges point consumer → chain head, §6):

```
store1:  ImpactBy = {}            // first access to %out1 — chain head was nil
store2:  ImpactBy = {}            // first access to %out2
end:     ImpactBy = {store1, store2}   // one edge per output tensor's FINAL chain head (R2)
```

The per-tensor chains are **isolated** — the restrict-alias rule (§6.2) is
what makes two independent chains sound without any cross-ordering edge.

**Same tensor, second store.** If the second chain stored `out1` again
(`mk().Store(out1, m)`), the second store would chain after the first —
exactly the second-ImpactBy-first edge of §6.2:

```
store1:  ImpactBy = {}            // first access to %out1
store2:  ImpactBy = {store1}      // NewMemStoreChained wires env.ChainHead(out1)
end:     ImpactBy = {store2}      // final chain head ONLY — store1 stays live
                                  // transitively, via store2's ImpactBy path
                                  // (Level-1 DCE follows payload edges, §7)
```

(The builder's output binding also updates: the end node's operand for
`out1` becomes `store2` — last store wins — while `store1` remains live only
through the chain.)

### 9.4 Teaching example: LIR cache-placement sketch

What an HIR GMEM access lowered through an SMEM cache into a register
fragment looks like at the node level, once cache lowering (a **future**
phase) exists. No golden covers this: the node shapes and renderings below
are what `catalog.go` + `node_lir_memory.go` + `print.go` support today;
anything beyond is marked.

```
# illustrative — cache lowering is a FUTURE phase; nothing below is a golden.

alloc = memnewsmem(extent)                                  # SMEM buffer, StorageClass=smem
fill  = memstoresmem(%gmem_value, alloc, [@m_b, @k_b])     # ops = [value, dst buffer, biv axes]
tile  = memloadsmem(alloc, [@m_b, @k_b])                   # ops = [src buffer, biv axes]
faccr = fragnew(extent)                                     # register fragment, StorageClass=frag
acc   = fragstore(%tile_value, faccr, [@m_b, @k_b])         # returns the stored value (tca.md §2.5)
reuse = fragload(faccr, [@m_b, @k_b])

# the enclosing nest is the explicit lowered loop (a stage, §4 row 17):
for(@m_b(init=0.0, step=1.0)/M, ...)  body                  # illustrative — bound/body elided
#   └ forloop ops = [iv(dimvar), bound(bdimvar), body];  the biv axes are the
#     lowered form of the HIR access axes;  @m_b/M prints as bdimvar over the
#     normalized dimvar (doc/optana/induction_variable.md)
```

Reading the sketch against §4:

- The HIR **payload access** (tensor + axes uses on `memload`/`memstore`)
  has been promoted into **ordinary operand flow**: the SMEM/FRAG buffer is
  an operand slot, and the axes are **bounded dimvars** (`bdimvar` value
  objects, identity = the `(init, step)` tuple — never the name).
- Every LIR memory node carries its `StorageClass` (`smem`/`frag`), bound at
  construction (X-4); the verifier fails closed unless buffers are
  `memnewsmem`/`fragnew` defs.
- SMEM/FRAG stores are **effect-free** (§6.1): they are not statement roots
  and are dead-code-eligible unless they feed the end node — a cache that
  nothing observable consumes simply vanishes at Level-1 DCE.
- `# illustrative` lines: the `for(...)` rendering (a forloop stage prints
  `for(v in bound) body`, but no committed example pins the exact shape) and
  the elided bound/body.
