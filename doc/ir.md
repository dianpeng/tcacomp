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

## 1. Overview & how to read this doc

### 1.1 What the IR is

The TCA IR is a **single-tier, SSA, optimizer IR** (design.md's "IR must be
carefully designed to support optimization mutation" — one node suite from
which both the abstract kernel graph and the lowered forms are built; no
separate high/mid/low IRs). It is not an AST materialization: every reference
in the graph is an `SSAUse` record and every definition is an `SSADef`, so
passes rewrite graphs in place through a small choke-point API rather than
rebuilding trees (review `doc/review/phase1_1.md`, P1R-01).

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
  is gone (replaced by the SMEM/FRAG families in `FamilyMemory`), the file
  column still names the old monolithic `construct.go`/`tile.go` layout, and
  its `dimvar` operand list `[dim, init, step]` reflects the pre-layering
  shape — the implemented `dimVarNode` has operands `[init, step]` with the
  dim binding as a payload use (§4 below). Same deferred-sync category.

## 2. File organization

`internal/ir` is layered per the tinycudaopto skeleton re-organization
(quick-261018-bl0 task 1, requirement R4). The rule a new node kind follows:
**add the struct in its family's `node_*.go` file, add a row to
`catalog.go`, add a case to `Graph.allocNode` in `node.go`** — three places,
no enum surgery (append-only, IR-03).

| File | Owns | Why |
|---|---|---|
| `node.go` | The structural core: `Base` cell (ID, kind, `Loc`, `LastTouchedBy`, SSAVar, erased tombstone, surplus count, stmt back-pointer, pinned stmtID), the `Node`/`Value`/`ScalarExprNode`/`nodeImpl` interfaces, `operandList` slot storage, the `Graph` root (name, params, `nodes` registry, `end`, `cfg`), `allocNode` (the one kind→struct dispatch), `wireOperands`/`wirePayloadUse`, `NewGraph`, `NewID`/`ReserveStatementID`/`PinStatementID`, `NewNode`. | One place holds everything every node shares; the dispatch table is the single join point for new kinds (R4a). |
| `kind.go` | The `Kind` enum (`KindVersion = 3`), `Kind.String()` (lowercase canonical spelling, D-6), `Family` classification (D-15). | Enum + spelling + family only; all other per-kind metadata lives in `catalog.go` (R4e separation). |
| `catalog.go` | The dedicated listing file (R4e): one `kindEntry` per kind — tier (D-3), family, operand arity, one-line doc — plus `HIRKinds()`/`LIRKinds()`/`AllKinds()`/`Tier()`/`fixedArity()`. Rows are in enum order, so `catalog[k-1]` describes `k`. | Pinned invariants (catalog_test.go): `len(AllKinds()) == int(KindEnd)-1`; HIR kinds are exactly the kinds `< KindForLoop` (the tier boundary IS the enum boundary). |
| `doc.go` | Package design notes (the long header comment: optimizer shape, two-level IR, SSA chains, walking, lowered suite, interface-width contract). | Intent lives with the code it governs. |
| `node_lambda.go` | HIR lambda stages: `mapNode`, `reduceNode`, the `Lambda` payload record. | The piped input is an **ordinary operand** (map operand 0, reduce operand 1) — "pipe" survives only as builder terminology, so data-dependency edges cover piping with zero special cases. |
| `node_expr.go` | HIR scalar-expression suite: `constNode`, `paramRefNode`, `binOpNode`, `unaryOpNode`, `selectNode`, `castNode`, `callNode`, plus the `BinOpCode`/`UnaryOpCode` tables mirroring `doc/syntax.md` §7.6 precedence. | Value-level leaves every lambda body is built from; lowering never rewrites them. |
| `node_var.go` | `inputParamNode` and the `ParamRole` taxonomy (`RoleDim`, `RoleTensor`, `RoleScalar`, `RoleLambda`), `RoleOf`. | Kernel params are graph-owned (`NewInputParam` appends to `g.params`); lambda params are graph-external by design (inlined at lowering). |
| `node_memory.go` | HIR GMEM family: `memLoadNode`, `memStoreNode`, `memNewNode`, the `access` payload record (tensor + axes as SSAUses), and the load/store `effectIns` ImpactBy payloads. | The structural tensor access is a **payload** of InputParam uses at this level (P1R-01); lowering promotes the pieces into ordinary SSA operand flow (the LIR family). |
| `node_loop.go` | Loop-structure nodes: HIR `iterNode`, `iterClusterNode`; LIR `forLoopNode`. | Structure only — the LIR IV *value objects* live in `node_dimvar.go`. |
| `node_dimvar.go` | LIR `dimVarNode`, `boundedDimVarNode` and the IV/BIV relations: `SameIV`, `SameBIV`, `Includes` (`~>`), `BelongsTo` (`<~`), per `doc/optana/induction_variable.md`. | An induction variable is a value object: identity is the `(init, step)` tuple, never the name. |
| `node_lir_memory.go` | LIR memory family: `memLoadSMemNode`, `memStoreSMemNode`, `memNewSMemNode`, `fragLoadNode`, `fragStoreNode`, `fragNewNode`, the `StorageClass` metadata (`StorageGMEM`/`StorageSMem`/`StorageFrag`), `StorageClassOf`. | Every LIR memory node carries the storage class its constructor bound it to (X-4, mirroring tinycudaopto's `memory_class.go`); the buffer is an **operand slot**, the axes are bounded dimvars. |
| `node_end.go` | `returnNode` — the graph's ONE end node (R2): `Graph.end`, `End()`, `NewReturn`, `NewReturnChained`. | First `NewReturn` on a graph becomes `g.end`; later returns are ordinary nodes the verifier rejects (fail-closed multiple-end check). |
| `ssa.go` | The SSA handle layer: `SSAVar`, `SSADef`, `SSAUse`, the `ssaRefRecord` indirection, `UsesOf`/`DefOf`. | Sole pointee of the use-def/def-use relationship (directive 2); see §5. |
| `effect.go` | The side-effect model: `Effect` verdicts, `EffectOf`, `ImpactBy`, the construction `Env`, `NewMemStoreChained`. | The explicit per-tensor ordering edges of the statement-free sea; see §6. |
| `poset.go` | The dependency-poset read face: `DataDeps`, `EffectDeps`, `PosetDeps`, `SeaRoots`. | The primitive edge enumerations Level-1 analysis and the pass scheduler build on (R1); see §8.1. |
| `walk.go` | `Walk`/`Visitor`/`VisitorFunc`/`VisitResult` (Descend/SkipChildren/Stop). | The analysis/checking substrate (directive 3); level-aware roots; NOT the printer's order (D-5 resolution, §8.3). |
| `dce.go` | `BuildingDCE` and both level algorithms: `markReachable`/`buildingDCESea` (Level 1), `markDeadStatements`/`wipeDeadStatements` (Level 2), `isEffectKind`/`hasEffectNode`, `statementOwnership`. | Building debris is wiped before passes see the graph (P1R-08); see §7. |
| `schedule.go` | `Graph.Materialize()` — the Level-1 → Level-2 transition. | Deliberately minimal default scheduling (D-7); optimization scheduling is the pass's job (§8.2). |
| `cfg.go` | Level-2 control-flow shape: `CFG`, `BasicBlock`, `Statement`, the intrusive doubly-linked list (`Append`/`InsertBefore`/`InsertAfter`/`Remove`), `StmtKind` (`StmtAssign`/`StmtEffect`), `NewStatement`, `classifyStmt`. | O(1) removal by pointer (P1R-02); `NewStatement` is the factory passes use after scheduling. |
| `verify.go` | `Verify`, the `use-def`/`def-use` chain invariants, per-kind shape checks, the multiple-end-nodes check, `Violation` diagnostics. | The cheap verifier callable after every pass (design.md); fail-closed everywhere. |
| `print.go` | The canonical deterministic printer: `Print`/`PrintOpts`, `Numbering`/`ComputeNumbering`, `StmtNodes`, `chainOf`, `stageText`/`value`, `Label`, `OperandsOf`/`LambdaBodies`, `FormatConst`. | Byte-exact goldens (D-01..D-04, D-13); see §8.3. |
| `mutate.go` | `MutContext` and the five choke points: `SetDef`, `AddUse`, `RemoveUse`, `ReplaceAllUsesWith`, `Erase`. | The ONLY sanctioned mutation path outside construction (IR-04, D-17); see §5.3. |
| `types.go` | `ElemType` (syntax.md §3.3 mirror, f8 absent), `DimRef`, the `Type` interface and interned `NumericType`, `InternScalar`/`InternTensor`. | Extensible type interface (directive 1); interning makes interface `==` identity (D-18). |

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

`KindVersion = 3` (IR-03) is the deliberate **one-time** tinycudaopto
re-organization (quick-261018-bl0): HIR kinds now group before LIR kinds (the
tier boundary is the enum boundary — `KindReturn` is the last HIR kind,
`KindForLoop` the first LIR kind), the generic `Load`/`Store` and the
`LoadTile`/`StoreTile` kinds are gone (R4d), and the GMEM `MemLoad`/`MemStore`/`MemNew`
family plus the lowered `MemLoadSMem`/`MemStoreSMem`/`MemNewSMem` and
`FragLoad`/`FragStore`/`FragNew` families take their places. From v3 onward
the rule is **append-only**: new kinds insert before `KindEnd` within their
tier and never renumber existing constants. `KindInvalid` (0) and `KindEnd`
(26) are sentinels with no catalog row; the pass manager's legality windows
and the verifier reject both, and any kind `>= KindEnd` (IR-03/IR-07).

All 25 kinds, exactly as listed in `catalog.go` (enum order; arity `-1` =
variadic, slots allocated from the operand list):

### HIR tier — the abstract kernel graph the builder produces (16 kinds)

| # | Kind (spelling) | Family | Arity | Operands / payload | Semantics |
|---|---|---|---|---|---|
| 1 | `map` | Abstract | -1 | ops = operand values (piped input first on a piped map); payload = `Lambda` + `as`-order uses | Lambda map stage, element-wise over its operands; the mandatory `as` clause fixes output axis order (tca.md §2.7). |
| 2 | `reduce` | Abstract | 2 | ops = `[init, piped]`; payload = combiner `Lambda` | Lambda reduce stage over the producing map's last `as` axis; homogeneous reducer contract (tca.md §4.2). |
| 3 | `memload` | Memory | -1 | ops = `[tensor, axes...]`; payload = `effectIns` | HIR GMEM load of `tensor[@m][@n]`; chains into the per-tensor effect order via ImpactBy. (The builder's tensor *reads* are `param` refs — §9.1; `memload` is the explicit form effect linearization materializes.) |
| 4 | `memstore` | Memory | 1 | ops = `[value]`; payload = target `access` (tensor + axes) + `effectIns` | HIR GMEM structural store, the stage store. Returns the stored value (tca.md §2.5), which is what keeps fusion pipes intact. |
| 5 | `memnew` | Memory | 1 | ops = `[extent]` | HIR GMEM allocation; the created extent is a Value. |
| 6 | `iter` | Loop | 2 | ops = `[init, step]`; payload = dim use, bound use (unwired = unbounded) | Abstract induction variable `@m`, optionally bounded `@m/M` (tca.md §2.3). |
| 7 | `itercluster` | Loop | 1 | ops = `[body]`; payload = dim uses | Compound header `iter(@m, @k, @n)` = the Cartesian product of its dims (tca.md §2.4). |
| 8 | `const` | ScalarExpr | 0 | payload = interned type + value | Typed scalar constant; floats print shortest-round-trip f32 with `.0` appended (`FormatConst`). |
| 9 | `param` | ScalarExpr | 0 | payload = use of a lambda or scalar InputParam | Scalar-expression parameter reference. A reference to a **tensor** param renders structurally (`lhs[@m][@n]` — the axes come from the tensor type's dims); scalar/lambda params render bare (`scale`, `v`). |
| 10 | `binop` | ScalarExpr | 2 | ops = `[a, b]`; payload = opcode | Binary scalar op; the opcode table mirrors syntax.md §7.6 precedence. |
| 11 | `unaryop` | ScalarExpr | 1 | ops = `[x]`; payload = opcode | Unary scalar op (`+`, `-`, `!`, `~`). |
| 12 | `select` | ScalarExpr | 3 | ops = `[cond, then, else]` | Typed conditional `cond ? then : else`. |
| 13 | `cast` | ScalarExpr | 1 | ops = `[x]`; payload = target type | Explicit cast to a named scalar type; renders `cast<f32>(x)`. |
| 14 | `call` | ScalarExpr | -1 | ops = args; payload = builtin name | Registered **pure** builtin call. |
| 15 | `inputparam` | Abstract | 0 | payload = name (debug only), interned type (nil for dims), role | Graph-owned kernel/lambda input parameter. The node's `SSAVar` IS the parameter's def; every reference elsewhere is an `SSAUse`. |
| 16 | `return` | Abstract | -1 | ops = returned values; payload = `effectIns` | **The graph's single end node** (R2): the first `NewReturn` becomes `Graph.end`; carries return values AND every mem-param's final effect-chain state via ImpactBy (§6.3). The none Value — binds no SSA name. |

### LIR tier — the lowered suite introduced by optimization passes (9 kinds)

| # | Kind (spelling) | Family | Arity | Operands / payload | Semantics |
|---|---|---|---|---|---|
| 17 | `forloop` | Loop | 3 | ops = `[iv(dimvar), bound, body]` | Explicit lowered loop nest. Shape check: operand 0 must be a dimvar. |
| 18 | `dimvar` | Abstract | 2 | ops = `[init, step]`; payload = dim binding | Lowered induction variable — a value object; identity is the `(init, step)` tuple, never the name (doc/optana/induction_variable.md). Prints `@x(init=…, step=…)`. |
| 19 | `bdimvar` | Abstract | 2 | ops = `[iv(dimvar), max]` | Bounded IV `((iv tuple), Max)`; prints `@x/M`. Shape check: operand 0 must be a dimvar. |
| 20 | `memloadsmem` | Memory | -1 | ops = `[src smem buffer, bivs...]`; class = `StorageSMem` | Lowered SMEM load. Shape check: src must be a `memnewsmem`. |
| 21 | `memstoresmem` | Memory | -1 | ops = `[value, dst smem buffer, bivs...]`; class = `StorageSMem` | Lowered SMEM store; returns the stored value (HIR contract, tca.md §2.5). |
| 22 | `memnewsmem` | Memory | 1 | ops = `[extent]`; class = `StorageSMem` | SMEM allocation. |
| 23 | `fragload` | Memory | -1 | ops = `[src frag buffer, bivs...]`; class = `StorageFrag` | Lowered register-fragment load. Shape check: src must be a `fragnew`. |
| 24 | `fragstore` | Memory | -1 | ops = `[value, dst frag buffer, bivs...]`; class = `StorageFrag` | Lowered fragment store; returns the stored value. |
| 25 | `fragnew` | Memory | 1 | ops = `[extent]`; class = `StorageFrag` | Fragment allocation. |

Notes:

- **Arity is enforced fail-closed**: fixed-arity kinds record operands supplied
  beyond the slot count as `Base.surplus`, which the verifier flags under the
  use-def invariant (WR-01). Variadic kinds wire every operand.
- **Shape checks** (verify.go, fail-closed): HIR GMEM load/store tensors bind
  a `RoleTensor` `inputparam`; SMEM buffers bind a `memnewsmem`; FRAG buffers
  bind a `fragnew`; `bdimvar` and `forloop` bind a `dimvar` at operand 0.
- The **families** (D-15): `FamilyScalarExpr` (8–14), `FamilyMemory` (3–5,
  20–25 — gained the whole GMEM family plus SMEM/FRAG at v3, replacing the
  deleted `FamilyTileOp`), `FamilyLoop` (6, 7, 17), `FamilyAbstract`
  (1, 2, 15, 16, 18, 19 — the lowered dimvars are value-level IV objects, not
  loop-nest structure), `FamilyCollective` (reserved, no members until reduce
  lowering lands).
- The pass manager rejects nodes with `Kind == KindInvalid` or
  `Kind >= KindEnd` (forward-compatibility within our own pipeline, IR-03).

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

### 5.2 Payload uses are uses

Every reference in the graph is an `SSAUse` — operand slots **and** payload
uses alike: lambda params, iteration dims/bounds, map `as`-orders, store
target accesses, parameter refs, and `ImpactBy` effect inputs are all SSAUse
records owned by their referencing node, wired by the same `wirePayloadUse`
machinery as operand slots and enumerated via the `payloadBearer` interface.
The verifier's use-def/def-use invariants and `Walk` therefore cover payload
uses exactly like operand slots in **both** chain directions with zero special
checkers, and names survive only as `inputparam` debug payload — never
resolution or identity (P1R-01). Lambda bodies are the IR's one non-Use edge
kind, walked as an edge via `lambdaBearer`.

### 5.3 Mutation choke points (IR-04, D-17)

All chain edits outside construction flow through `MutContext` — five
operations, each stamping `LastTouchedBy = ctx.Pass` on every node it touches
so PASS-06 diagnostics name the responsible pass structurally:

| Choke point | Effect |
|---|---|
| `SetDef(u, newVal)` | Redirects one operand use: physically moves its ref record from the old def's var to the new one (nil unwires the slot). Every handle copy observes the new target. |
| `AddUse(user, val)` | Appends one operand slot and links it into val's ref list; returns the new handle. |
| `RemoveUse(u)` | Detaches u from its def's ref list and drops u's slot, **killing** the record so every copy reads dead. A killed record still occupying a slot is exactly the corruption class the verifier flags by node and slot. |
| `ReplaceAllUsesWith(old, new)` | RAUW, LLVM-style: relocates every live ref record of `old` into `new`'s var in one pass over a snapshot; self-RAUW is a no-op. |
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
ALL `g.params`** and follows exactly `Walk`'s child relation (operand slots,
payload uses including `ImpactBy`, lambda bodies). The seeding is deliberately
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
| `DataDeps(n)` | The live operand-slot defs of n, in slot order — n's data "must come after" set. |
| `EffectDeps(n)` | The live `ImpactBy` payload defs of n — the explicit per-tensor ordering edges. Non-effectful nodes carry none. |
| `PosetDeps(n)` | The flat union: data edges (slot order) then effect edges — the single list a topological ordering consumes. |
| `g.SeaRoots()` | The Level-1 root set: `RoleTensor` params (declaration order) + the end node (§7). |

Edges point **use → def**. Payload uses that are not ordering edges — store
targets, map `as`-orders, lambda params — are deliberately excluded: they name
parameters (graph roots) and contribute nothing to a def-poset. All
enumerations skip killed/unwired records (the handle model's checkable
corruption classes stay checkable, never traversable) and are deterministic
(slot order, then payload order — never map iteration, IR-05/RESEARCH Pitfall 1).

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
