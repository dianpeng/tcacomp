# TCA Frontend: Syntax Design (Implementation-Ready)

**Status:** implementation-ready design, produced by the design review of `doc/syntax.md`
(quick-261024, `.planning/quick/20261024-frontend-syntax-design-review/`).

**Verdict of the review:** the `doc/syntax.md` sample is directionally sound and lowers
near-mechanically onto the existing IR (`internal/ir`) and builder DSL (`pkg`), but it
had five gaps/holes and one IR-contract conflict. All were ruled by the author; the
rulings are normative here:

| # | Finding | Ruling |
|---|---------|--------|
| R1 | Shape names (`M, N, Q` in `shape=[M,N]`) are never declared | **Implicit from signature**: a shape identifier in a kernel signature is auto-declared as a symbolic dim, scoped to that kernel |
| R2 | No scalar parameter / capture mechanism (matmul_scale needs `scale`) | **Signature param + capture list**: scalars are kernel params; lambdas capture explicitly `[scale] (...)` |
| R3 | Map and reduce use different clause orders; `on(...)` means two different things | **Unified order** `verb (attrs)? on(args) (as (dims))? lambda-tail` for both verbs |
| R4 | `init CAN BE dimvar → InductionVariable` contradicts the homogeneous-reducer IR contract (a dimvar is not a scalar value) | **Deferred from v0**: reduce init is a typed scalar constant/expression; dimvar-init (scan-like seed) goes to the roadmap |
| R5 | Bridge placement (`internal/ir/builder.go`) would duplicate `pkg`'s validation | **Reuse `pkg` rungs**: the AST lowering drives the existing fluent rungs; validation stays single-sourced |

**Scope.** This document pins: the strict BNF grammar (§2), the AST design (§3), the
semantic checks owned by the syntax module (§4), the optional-type resolution algorithm
and its boundary with `internal/opt/typeinfer` (§5), the AST→IR construction roadmap
(§6), clarifications (§7), nice-to-have features (§8). It replaces the normative role
the old (deleted) `doc/syntax.md` §3.3/§4.1/§5–§7 played; several code comments cite
those old section numbers (e.g. `internal/ir/types.go`, `internal/ir/builtins.go`,
`internal/ir/kind.go`, `pkg/chain.go`, `pkg/expr.go`) — re-point them at this document's
sections during implementation (§6 Task 0).

---

## 1. Design overview

A C++-style surface language parsed by a hand-written recursive-descent parser
(precedence climbing for expressions), lowered to the single-tier SSA IR. The frontend
introduces no thread binding, no user tiling, no arbitrary memory access, no tensor-level
control flow. Four headline properties, from `doc/syntax.md`:

1. **Immutable locals** — parameters and `let` bindings are immutable; no in-place
   tensor mutation in v0.
2. **Optional typing, strongly typed** — annotations may be omitted where inference is
   total; where written they must match the inferred type exactly (no implicit
   promotion, ever).
3. **Pipe operator `|`** — chains tensor-computation stages inside one `let`.
4. **Annotations** — `@name`, `@runtime`, `@include` instrumentation consumed by the
   compiler.

Deliberately excluded from v0 (carried from the prior spec): `for`/`while`/`if` at
tensor level, recursion, pointers, mutable variables, indexed assignment, atomics, CUDA
thread identifiers, user-defined callable functions, preprocessing, templates,
namespaces, overloading. Scalar `?:`, `&&`, `||` are supported *inside* lambdas only.

Package layout (per `doc/syntax.md`, unchanged):

```
internal/syntax/lexer      // hand-rolled, go/scanner structure: scanner struct, next/peek, position tracking, error-handler callback
internal/syntax/ast        // Go interfaces + concrete structs per node, typed-switch visitor (go/ast pattern)
internal/syntax/parser     // recursive descent + precedence climbing
internal/syntax/semacheck  // scopes + the checks of §4, returns a TYPED AST (§5)
internal/syntax/build      // AST→IR lowering driver over pkg rungs (§6)
```

The lexer/parser never import `internal/ir`; only `semacheck` (types) and `build`
(IR) do. Module assembly (includes, kernel table) lives in `internal/syntax` (package
`syntax`) — it is a frontend concept, not a new IR kind (§6.4).

---

## 2. Grammar (strict BNF)

Notation: `'` literals are tokens; `[ X ]` optional; `{ X }` zero-or-more; `{ X }?`
one-or-more where marked; `(` … `)` grouping; `|` alternation; `(* … *)` comments.
**Trailing commas are always supported** in every comma-separated list (design decision
from the sample).

### 2.1 Lexical elements

```
identifier ::= [a-zA-Z_] [a-zA-Z0-9_]*
intLit     ::= digits                       (* type i32 *)
uintLit    ::= digits ('u' | 'U')           (* type u32 *)
floatLit   ::= digits '.' digits [exponent] ('f' | 'F')
             | digits exponent ('f' | 'F')  (* type f32; suffix REQUIRED *)
stringLit  ::= '"' { any-char-except-'"' } '"'
comment    ::= '//' to-end-of-line | '/*' … '*/'
```

Keywords (reserved): `kernel let return dimvar map on as reduce init true false`.
Annotation keys (`dtype`, `shape`, `cxx`, `cxx_path`) and attribute keys are contextual
identifiers, not reserved words. `inf_f32` is a predefined constant of type `f32`
(reserved identifier), usable without capture; `-inf_f32` is the max-reduction seed.

Element-type keywords (also reserved, they double as cast operators): `bool f32 f16
bf16 i8 i16 i32 u8 u16 u32`. `f8` is intentionally absent.

There are no f64, hexadecimal, octal, binary, character, or string literals in *scalar
expressions* (string literals appear only in annotations). Unary minus is an operator,
never part of a literal.

### 2.2 Tokens (punctuation)

```
@ ( ) [ ] { } < > , ; : = ? | & ^ ~ ! + - * / %
-> == != <= >= && || << >>
%0  %1  …        (* placeholder family — see §7.3; lexed as '%' + digits *)
```

### 2.3 Module

```
module          ::= { globalDecl }
globalDecl      ::= includeDecl | runtimeDecl | kernelDecl
includeDecl     ::= '@include' stringLit ';'
runtimeDecl     ::= '@runtime' '(' keyedConstList ')' ';'
kernelDecl      ::= { annotation } 'kernel' identifier '(' paramList ')' '->' type '{' stmtList '}'
annotation      ::= '@' identifier '(' keyedConstList ')'        (* no ';'; attaches to the next kernel *)
keyedConstList  ::= [ keyedConst { ',' keyedConst } [ ',' ] ]
keyedConst      ::= identifier '=' constValue
constValue      ::= intLit | uintLit | floatLit | stringLit | 'true' | 'false'
```

Attachment rule (from `doc/syntax.md`, made strict): the parser records each
`annotation` as pending; when the next declaration is a `kernel`, the pending set
attaches to it; a pending annotation followed by anything else is a **semantic error**
(`annotation @x does not precede a kernel`). `@include` and `@runtime` are global; they
take `';'` and never attach. At most one `@runtime` per module; `@include` cycles are
rejected by the module loader.

### 2.4 Parameters and types

```
paramList       ::= [ param { ',' param } [ ',' ] ]
param           ::= identifier ':' paramType
paramType       ::= tensorType | scalarType
type            ::= tensorType | scalarType                (* kernel return type *)

tensorType      ::= 'tensor' '<' 'dtype' '=' scalarType
                    [ ',' 'shape' '=' '[' shapeDimList ']' ] [ ',' ] '>'
shapeDimList    ::= [ shapeDim { ',' shapeDim } [ ',' ] ]
shapeDim        ::= uintLit | identifier                   (* identifier = implicit symbolic dim, R1 *)
scalarType      ::= 'bool' | 'f32' | 'f16' | 'bf16'
                  | 'i8' | 'i16' | 'i32' | 'u8' | 'u16' | 'u32'
```

Rules:
- A `tensor` type **must** name its `dtype`; `shape` may be omitted only in *local*
  annotations where it is inferable — in kernel parameter and return types `shape` is
  required (rank is always static).
- A `shapeDim` identifier declares the symbolic dim implicitly (R1); a `uintLit` is a
  compile-time constant extent (may be `0`). Shape arithmetic (`M*N`) is excluded.
- Fully reduced results use a bare scalar type (`f32`), never a rank-0 tensor.
- Scalar kernel parameters (`scale: f32`) are declared with `paramType ::= scalarType`;
  they are launch-uniform scalars (R2).

### 2.5 Statements

```
stmtList        ::= { stmt }
stmt            ::= dimvarDecl | letDecl | returnStmt
dimvarDecl      ::= 'dimvar' identifier [ ',' identifier { ',' identifier } [ ',' ] ] ';'
                  | 'dimvar' '(' identifierList ')' ';'
letDecl         ::= 'let' identifier [ ':' type ] '=' stageChain ';'
returnStmt      ::= 'return' identifier ';'
identifierList  ::= [ identifier { ',' identifier } [ ',' ] ]
```

- `dimvar` declares bounded logical iteration axes (the IR's `Iter` dims). Extents are
  not written here; a dimvar's extent is the extent of whatever shape dim it is bound
  to structurally (§2.6). A dimvar is **not a scalar value**: it may appear only in
  bindings and `as` lists, never in expressions (R4 ruling's flip side).
- Exactly one `return` per kernel, the last statement, naming a `let`-bound tensor.
- One namespace per kernel: parameter names, implicit dim names, dimvar names, and
  `let` names must all be distinct (no shadowing anywhere).

### 2.6 Stage chains (the pipe)

Unified clause order per ruling R3: **`verb (attrs)? on(args) (as (dims))? lambda-tail`**.

```
stageChain      ::= stage { '|' stage }
stage           ::= mapStage | reduceStage

mapStage        ::= 'map' 'on' '(' bindingList ')' [ 'as' '(' dimList ')' ]
                    lambda
reduceStage     ::= 'reduce' '(' keyedInitList ')' 'on' '(' paramSpecList ')'
                    [ 'as' '(' dimList ')' ] '->' scalarType ':' expr

bindingList     ::= [ binding { ',' binding } [ ',' ] ]
binding         ::= identifier { '[' identifier ']' }        (* tensor[dimvar][dimvar]… *)
keyedInitList   ::= [ 'init' '=' initExpr { ',' } ]
initExpr        ::= constexpr-ish scalar expr over constants/captures (§4.6)
dimList         ::= [ identifier { ',' identifier } [ ',' ] ]
lambda          ::= [ captureList ] '(' paramSpecList ')' [ '->' scalarType ] ':' expr
captureList     ::= '[' identifierList ']'
paramSpecList   ::= [ paramSpec { ',' paramSpec } [ ',' ] ]
paramSpec       ::= placeholder [ ':' scalarType ] | identifier ':' scalarType
placeholder     ::= '%' digits                                  (* v0: only %0 is legal *)
```

Stage-specific arity rules (semantic, §4):
- **Root map** (first stage of a chain): `on` holds ≥1 tensor bindings; lambda params
  correspond 1:1 to bindings in order. No captures of dimvars/tensors; captures must be
  kernel scalar params or const scalar lets (R2).
- **Piped map** (after `|`): `on` may be empty (pure elementwise) or hold bindings;
  lambda param 0 is the piped value — spelled `%0` (optionally `: scalarType`) or a
  named `identifier ':' scalarType` — followed by one param per binding.
- **Reduce**: `on` holds exactly two params: the accumulator (named, `identifier ':'`
  `scalarType`) and the piped element (`%0` or a name, optionally typed). The `->`
  return type is required for reduce (it pins the homogeneous reducer type T, §4.6);
  the `as` clause lists the surviving axes.

Both stages end with `-> scalarType ':' expr` — for map the params ride the lambda
prefix, for reduce they ride `on` (this is the unified tail; ruling R3).

### 2.7 Scalar expressions (precedence climbing)

Precedence table, tightest last-listed-first (climbing from the bottom up). This is the
**same operator set and typing as `pkg/expr.go` / the IR `BinOpCode`/`UnaryOpCode`
tables** (old spec §6.2/§7.6, normative again here):

```
expr        ::= ternary
ternary     ::= lor [ '?' expr ':' expr ]            (* C-style; lazy branches *)
lor         ::= land { '||' land }
land        ::= bitor { '&&' bitor }
bitor       ::= bitxor { '|' bitxor }                (* see pipe-escape rule §7.1 *)
bitxor      ::= bitand { '^' bitand }
bitand      ::= equality { '&' equality }
equality    ::= relational { ( '==' | '!=' ) relational }
relational  ::= additive { ( '<' | '<=' | '>' | '>=' ) additive }
additive    ::= multiplicative { ( '+' | '-' ) multiplicative }
multiplicative ::= unary { ( '*' | '/' | '%' ) unary }
unary       ::= ( '-' | '+' | '!' | '~' ) unary | postfix
postfix     ::= primary
primary     ::= literal | identifier | builtinCall | castExpr | '(' expr ')'
builtinCall ::= identifier '(' [ expr { ',' expr } [ ',' ] ] ')'   (* registry-resolved *)
castExpr    ::= scalarType '(' expr ')'              (* explicit cast only *)
literal     ::= intLit | uintLit | floatLit | 'true' | 'false' | 'inf_f32'
```

A lambda body is exactly **one** `expr` — no statements, no `return` keyword, no local
declarations inside lambdas (v0; the old spec's lambda-local `const` is dropped for
simplicity — name subexpressions via chained `let`s or builtins).

---

## 3. AST design

Go interfaces with concrete structs per node and a typed-switch visitor, exactly the
`go/ast` shape (project decision (a) in CLAUDE.md). All nodes carry a
`token.Position` (offset, line, column) from the lexer. AST is **untyped at parse
time**; `semacheck` returns the same tree with resolution/type cells filled (§5) —
no parallel typed-AST type hierarchy.

### 3.1 Node set

```go
// Node is every AST node's interface.
type Node interface{ Pos() token.Pos; astNode() }

// --- module ---
Module      { Includes []*Include; Runtime *RuntimeAnn; Kernels []*Kernel }
Include     { Path string; Resolved string }            // Resolved set by module loader
RuntimeAnn  { Args []*KeyedConst }
Annotation  { Name string; Args []*KeyedConst }
KeyedConst  { Key string; Value ConstValue }
ConstValue  { Kind ConstKind; Text string; Val any }    // Int|Uint|Float|String|Bool

// --- kernel ---
Kernel      { Anns []*Annotation; Name string; Params []*Param; Ret TypeRef; Body []Stmt }
Param       { Name string; Type TypeRef; Role ParamRole } // Role: TensorParam|ScalarParam (set by semacheck)

// --- types (TypeRef is an interface) ---
TensorType  { Elem ScalarType; Shape []ShapeDim }
ScalarType  { Elem ir.ElemType }                        // elem types map 1:1 onto ir.ElemType
ShapeDim    { Name string; Extent uint64 }              // Name set ⇔ dynamic (Extent==0)

// --- statements (Stmt is an interface) ---
DimVarDecl  { Names []string }
LetDecl     { Name string; Type TypeRef (nil = omitted); Chain *StageChain }
ReturnStmt  { Name string }

// --- stage chain ---
StageChain  { Stages []Stage }                          // Stage is an interface
MapStage    { Bindings []*Binding; As []string (nil = infer); Lambda *Lambda }
ReduceStage { Init Expr; OnParams []*ParamSpec; As []string (nil = infer); RetType ScalarType; Body Expr }
Binding     { Tensor string; Dims []string }
Lambda      { Captures []string; Params []*ParamSpec; RetType *ScalarType (nil = infer); Body Expr }

// --- parameter specs (stage params / pipe) ---
ParamSpec   { Pipe bool; Name string; Type *ScalarType (nil = infer) }  // Pipe ⇔ spelled %0

// --- expressions (Expr is an interface) ---
BasicLit    { Kind LitKind; Value string; Val any }     // Int|Uint|Float|Bool
NameRef     { Name string }                             // resolves: lambda param > capture > const scalar let
PipeRef     { }                                         // %0 — only reachable via its ParamSpec; never free
BinaryExpr  { Op BinOp; L, R Expr }                     // BinOp mirrors ir.BinOpCode 1:1
UnaryExpr   { Op UnaryOp; X Expr }                      // UnaryOp mirrors ir.UnaryOpCode
SelectExpr  { Cond, Then, Else Expr }
CallExpr    { Name string; Args []Expr }                // pure builtin registry only
CastExpr    { Target ir.ElemType; X Expr }
```

Visitor:

```go
type Visitor interface{ Visit(n Node) Visitor }   // go/ast shape
func Walk(v Visitor, n Node)
```

`Walk` dispatches children in source order (declaration order, binding order, operand
order) — deterministic, so every diagnostic printer and future formatter is stable.

### 3.2 Design notes

- **One interface per syntactic category** (`Stmt`, `Stage`, `Expr`, `TypeRef`) keeps
  the typed switches small; cross-category reuse (e.g. `ParamSpec` inside both
  `Lambda` and `ReduceStage.OnParams`) avoids duplicate node kinds.
- `ParamSpec.Pipe` is the *only* place `%0` exists; a free `PipeRef` outside a stage
  param list is impossible by construction (parser rejects `%` in `primary`), which
  removes a whole semacheck class.
- `As == nil` vs `As != nil` distinguishes omitted (infer) from explicit (check) — the
  two take different code paths in §5 and must stay distinguishable.
- No `ParenExpr`: precedence is resolved during parsing; parens never survive into the
  AST (matches how go/parser drops them into the tree only where needed — here: never).

---

## 4. Semantic checks (owned by `internal/syntax/semacheck`)

Errors are position-attributed (`internal/diagnostic`), one per violation, collected
but fail-fast at kernel granularity (a kernel with any error produces no typed AST).
The checks below mirror — and must stay in sync with — the eager checks in
`pkg/builder.go`/`pkg/chain.go` (single-sourced validation, ruling R5).

### 4.1 Module scope

1. Duplicate kernel names.
2. `@include` cycle detection (module loader); missing include file.
3. At most one `@runtime`; `cxx_path` present and a string.
4. Annotation registry: `@name` accepts exactly `{cxx}` (string); unknown annotation
   names or keys rejected.

### 4.2 Kernel scope (one namespace, no shadowing)

5. Duplicate names across params / implicit dims / dimvars / lets.
6. Implicit dims: collected from all signature shapes (params + return); a dim used in
   the return must appear in at least one parameter shape (defensive: launch
   validation owns the runtime check, but a return-only dim is rejected as
   unbindable).

### 4.3 Types

7. Tensor rank ≥ 1; static extents fit `uint64`; `dtype`/`shape` key kinds correct.
8. Exact-type matches everywhere a type is written: `let` annotation vs inferred;
   lambda param/return annotations vs inferred; reduce `-> T` vs homogeneous type.
   **No implicit conversions, ever** (int literals do not silently become f32, etc.).

### 4.4 Bindings and axes

9. Binding tensor is declared; subscript count == tensor rank (structural indexing:
   subscript *i* binds dim *i* positionally).
10. Every subscript is a declared dimvar (not a shape dim, not a scalar).
11. `as` clauses: root map — a permutation of the bound dimvars (explicit) or
    inference (§5); piped map — exactly the live axes in order; reduce — the live axes
    minus the reduced axis, in order.
12. Reduce requires ≥1 live axis; chained reduces reduce the new last axis.

### 4.5 Lambdas

13. Arity: root map lambda params == binding count; piped map == 1 + binding count;
    reduce == 2 (acc, elem).
14. Captures: each name is a kernel scalar param or a const scalar `let`; no tensor /
    dim / dimvar capture; every capture is a *used* name is only a warning (unused
    capture allowed, mirrors C++).
15. Body name resolution closes over lambda params + captures + `inf_f32` only —
    kernel-body names (tensors, dimvars, lets) are invisible inside lambdas.
16. A lambda body has no further lambdas (no nested stages), no assignment, no
    subscript, no member access (the grammar of §2.7 already enforces; semacheck
    double-checks only name resolution).

### 4.6 Scalar expression typing (the homogeneous rules, from `pkg/expr.go`)

17. Binary arithmetic (`+ - * /`): same arithmetic type both sides, result that type.
    `%`, shifts, bitwise: same *integer* type. Comparisons: same type, result `bool`.
    `&&`/`||`: `bool`. Unary `-`/`+`: arithmetic; `!`: bool; `~`: integer.
18. `?:`: cond `bool`, branches identical type. Branches are lazy (C++ conditional
    evaluation) — the IR `KindSelect` carries this; the frontend must not constant-fold
    a branch away (e.g. an unselected division-by-zero stays unselected).
19. Casts: explicit only, `scalarType(expr)`; admitted pairs come from the shared cast
    registry (arithmetic→arithmetic in v0; no bool, no identity casts) — **one
    registry shared with the IR**, never a frontend-local table.
20. Calls: name must be in `ir.Builtin` (the single registry `internal/ir/builtins.go`),
    arity exact, argument types exact per signature (f32 builtins take f32; `umin`/
    `umax` take u32).
21. Literals: decimal ints are `i32`, must fit; `u`-suffixed are `u32`; floats are
    `f32` (suffix required); `true`/`false` are `bool`. Unary minus is `Neg`, not a
    literal.
22. Reduce init (ruling R4): a constant or launch-uniform scalar expression over
    constants/captures, of type T (the pipe element type). **No dimvars in init in
    v0** — the parse shape permits `init = expr`, and semacheck rejects any expr
    containing a dimvar or a device-computed scalar with the deferred-feature
    diagnostic.

### 4.7 Statements

23. `let`: name unused at declaration; no reassignment (immutability — a second `let`
    with the same name is a duplicate-name error, §4.2 rule 5).
24. `let` annotation vs inferred chain-output type: exact match (rule 8).
25. `return`: names a `let`-bound tensor; its type == kernel return type; exactly one;
    last statement.
26. A `let` chain must begin with a root map and end with a stage whose live axes have
    nonzero extent product compatibility with the let's declared shape (the shape dims
    of the annotation are the extents of the as-axes — checked by extent compatibility:
    each as-axis's extent source must equal the annotation's dim).

### 4.8 Diagnostics quality bar

Every error names the node, the expectation, and the found value (the `pkg` error
strings are the style reference). Suggestion-quality (did-you-mean for misspelled
dimvars) is nice-to-have (§8), not v0.

---

## 5. Optional typing and the typeinfer boundary

**Two distinct layers — do not conflate:**

| Layer | Runs over | When | Fills |
|-------|-----------|------|-------|
| Frontend type resolution (this module, §5.1) | typed AST | parse → semacheck | omitted *source* annotations (lambda params/returns, `let`, `as`, `%0`) |
| `internal/opt/typeinfer` (`Pass` @105) | constructed IR sea | after simplify-lower | the IR's computed-type cell (`Base.ty`) for every node |

The IR pass is **untouched** by this frontend; it neither reads source annotations nor
resolves AST holes. Optional typing is fully resolved *before* IR construction, so the
builder always receives complete types. Written annotations are verified == inferred
at frontend resolution time (rule 8/24); nothing in the IR records that a source
annotation existed.

### 5.1 Frontend resolution algorithm (per kernel, in `let` order)

1. **Collect** implicit dims from signature shapes (name → `ShapeDim`).
2. **Root map**: operand element types come from the bound tensors' `dtype`; lambda
   param types = operand elem types (or the written annotation, checked equal); body
   type computed bottom-up (rules §4.6 — every node typed, no fixed point needed: the
   expression AST is a tree). Root map element type T = body type. Written `-> T`
   checked. `As`: explicit → permutation check; omitted → **first-binding-appearance
   order** of the bound dimvars (deterministic; this is the only inference rule for
   root maps, so a *transpose* must be written explicitly).
3. **Reduce**: T = pipe element type (from the previous stage). Acc/elem params typed T
   (written annotations checked); `-> T'` must equal T (homogeneous reducer,
   `tca.md §4.2`); `init` typed and checked == T; `As`: explicit → live-minus-last
   check; omitted → live axes minus the reduced last axis, in order.
4. **Piped map**: pipe param type = previous stage element type; binding params as in
   (2); body type = new element type; `As`: explicit → exact live order; omitted →
   live axes in order (no reordering possible in v0, mirroring `checkAsExact`).
5. **Let**: output type = `tensor<T, extents of the final as-axes>`; written
   annotation checked equal (extents: a bound axis's extent is its source shape dim —
   static extent or the same symbolic name).
6. **Return**: checked against the kernel return type.

Resolution is a single forward pass over each chain (live-axis list + element type
threaded through stages); chains are independent; failure anywhere fails the kernel.

### 5.2 Compatibility verdict with `internal/opt/typeinfer`

Compatible, with the division of labor above pinned. One consequence to note in code
review of the frontend: because lambda params are typed from *operand element types*,
the constructed IR is fully typed from birth, and typeinfer's rules for
`KindParam`/`KindLambdaInputParam` will simply re-derive the same types — a
belt-and-suspenders consistency the IR verifier already assumes.

---

## 6. Roadmap: AST → IR construction

### Task 0 — bookkeeping (half a day)

Re-point the dangling code-comment citations of the old `syntax.md` sections at this
document (`internal/ir/types.go` "§3.3", `internal/ir/builtins.go` "§6.4",
`internal/ir/kind.go` "§7.6", `pkg/chain.go`/`pkg/expr.go` "§4.1/§5.1/§5.2/§6.2").
No behavior change.

### Task 1 — lexer + tokens (`internal/syntax/lexer`)

Port of `go/scanner`'s structure: scanner struct over source, `next()`/`peek()` runes,
position tracking from the start, error-handler callback. Token set per §2.1–§2.2.
Numeric literal parsing via `strconv` with explicit base/size; fail-fast with
position-attributed errors. **Do not use `text/scanner`.**

### Task 2 — parser (`internal/syntax/parser`)

Recursive descent per §2; precedence climbing per §2.7 (each level a method, left-assoc
loops, right-assoc recursions). Produces the untyped AST of §3. The pipe-escape rule
(§7.1) lives here. Eager syntactic checks only; everything semantic is §4.

### Task 3 — module loader + semacheck (`internal/syntax`, `internal/syntax/semacheck`)

Include resolution (relative to the including file), kernel table, then §4 checks +
§5.1 resolution, producing the typed AST. Export a `Kernel` view that `build` consumes.

### Task 4 — lowering driver (`internal/syntax/build` over `pkg` rungs)

For each kernel, drive the existing fluent rungs (ruling R5 — validation stays
single-sourced in `pkg`):

| AST | IR / pkg rung |
|-----|----------------|
| `@name(cxx=…)` | `KindKernelMeta` node (graph-owned naming metadata) |
| implicit dims | `b.Dim(name, extent)` (0 = dynamic) |
| tensor params | `b.Tensor(name, elem, dims…)` |
| scalar params | `b.ScalarParam(name, elem)` |
| root map + bindings | `Stmt().Iter(bound dimvars…).Map(operands, lambda)` — lambda built from resolved typed AST exprs |
| root `as` | `.As(dims…)` (inferred or explicit — same rung) |
| reduce | `.Reduce(init, func(acc, v) …)` (homogeneity already checked in §4.6/§5.1) |
| piped map | `.Map(ops, fn)` on `AfterReduce`/`Tail` rungs |
| `let` local tensor | **extension**: `b.Local(name, type)` → `KindMemNew`-backed tensor handle (locals are conceptually GMEM, `doc/syntax.md`); final stage → `.Store(local, axes…)` |
| operand of later maps referencing a `let` | the local's `MemNew` node as the operand's tensor def (paramRef onto it) — small `pkg` relaxation: `checkOperand` today requires `RoleTensor` InputParams; admit the MemNew def |
| `return` | existing output-binding/end-node flow (`Build` emits the single `KindReturn`) |
| `@runtime(cxx_path)` | codegen emission config carried on the module object, not an IR node |

**New `pkg` surface needed** (keep it minimal): `Builder.Local`, and the
`checkOperand` relaxation for MemNew-backed locals. Everything else already exists and
is pinned by `pkg`'s own tests. Desugaring performed here (from `doc/syntax.md`):
pipes become sequential rungs; `%0` becomes lambda param 0 / the reduce `now` param;
annotations become KernelMeta/module config; `let` chains become statement chains.

### Task 5 — module object and entry point

`internal/syntax`: `type Module struct { Kernels []*Kernel; Runtime …; Source … }`,
`ParseModule(fset, path) (*Module, error)`, and `func (k *Kernel) Build() (*ir.Graph,
error)` wrapping the Task 4 driver + `pkg.Build` (which already runs BuildingDCE,
Materialize, and the VerifyEach gate). The CLI (`cmd/`) gains `tcacomp compile
<input.tca>` emitting CUDA C++ + runtime files per the existing codegen path.

### Task 6 — tests (fundamental only, per design.md)

- Lexer/parser round-trips on the §7.2 example programs (goldens).
- Semacheck: one positive and one negative test per §4 rule family (no exhaustive
  matrices).
- End-to-end: `matmul_scale` in the new syntax compiles and matches the CPU reference
  (reuse the `internal/e2e` harness) — this is the acceptance test of the whole
  frontend.

### Ordering rationale

Lexer → parser → semacheck → build matches the data dependency and lets each layer be
tested hermetically; `pkg` is untouched until Task 4; the e2e test lands only after
Task 5. Expected to slot into the GSD phase list as one phase with tasks 0–6.

---

## 7. Clarifications (normative answers to ambiguities in the sample)

### 7.1 Pipe vs bitwise-or token disambiguation

`|` is one token with two uses. Deterministic rule: inside the **top-level** expr of a
stage body, a `|` whose next token is the keyword `map` or `reduce` terminates the
stage (it is the pipe); everywhere else `|` is bitwise-or. Inside parentheses, `|` is
always bitwise-or (a pipe can never appear inside a paren group). This is LL(1) with
one token of lookahead at the `bitor` level and needs no backtracking; `a | b;` at
stage top level still parses as bitwise-or because `b` is not a stage keyword.

### 7.2 Annotation attachment

Exactly as §2.3: pending annotations attach to the next kernel; anything else is an
error. Annotations are not allowed inside kernel bodies in v0.

### 7.3 `%0` placeholder rules

- `%0` may appear **only** as a stage parameter spec (piped-map param 0, or the reduce
  element param), never in an expression `primary` (the grammar of §2.7 has no
  production for it — `PipeRef` exists in the AST only as the def-site of the param).
- `%0` may carry an optional `: scalarType`, checked against the pipe element type.
- A named param may substitute for `%0` anywhere (`v: f32`); position, not spelling,
  identifies the pipe slot (param 0 of a piped map; the **second** param of a reduce,
  after the accumulator).
- `%1`, `%2`, … are reserved and rejected in v0 (no multi-stage-back references).

### 7.4 `as` inference, restated

Omitted `as` is always derivable and deterministic: root map = first-binding-appearance
order; any later stage = the producer's live axes minus reduced axes, in order. An
omitted `as` can therefore never express a transpose or a broadcast reorder — those
require an explicit clause (and are then checked by rules 11/§5.1).

### 7.5 dimvar vs shape-dim identity

A shape dim (`M`) is an *extent* (runtime nonnegative integer supplied by the launch
interface). A dimvar (`a`) is an *axis* (a bounded iteration variable, extent inherited
from the shape dim it is structurally bound to). They live in one namespace (rule 5)
but are disjoint kinds: dimvars appear in bindings/`as`, shape dims appear in types
only. Two dimvars bound to the same shape dim are distinct axes (old spec §3.4 rule,
retained).

### 7.6 Scalar values at kernel-body level

Kernel-body scalar arithmetic is not allowed on device-computed scalars; scalar params
may be combined with constants only *inside* a lambda (captured) or in a reduce `init`.
Rationale (old spec §3.5, retained): such arithmetic must live in the structural graph.
v0 simply has no kernel-body scalar-expression statements outside `init`.

### 7.7 Numerical policy (pinned, from the prior spec — unchanged)

IEEE 754 per operation; no fast-math, no FTZ, no silent precision reduction (no TF32).
FMA contraction permitted by default, per operation only. Fixed compiled kernel is
deterministic (no atomics, no data-dependent scheduling); parallel reductions are
schedule-dependent across different compilations. `f16`/`bf16` are storage/conversion
types in v0: cast to `f32` for arithmetic/builtins, cast back for storage.

### 7.8 f16/bf16 note

Native low-precision arithmetic can be added later through the same typed-expression
registry (`ir.Builtin` + cast registry); the grammar does not change.

---

## 8. Nice-to-have / roadmap (explicitly NOT v0)

1. **init = dimvar / scan seeds (ruling R4)** — per-slice induction-seeded reductions;
   requires an IR-level design (dimvars are not scalar values today) before syntax.
2. **Sub-kernel invocation** — `doc/syntax.md` names it (inlined invocation); v0 raises
   a semacheck error. Reserve no syntax yet; design when the IR grows a call/inline
   story.
3. **Multi-output kernels** — the IR end node already carries multiple stores; extend
   `return (a, b);` when needed.
4. **Scalar-return kernels** — fully-reduced results returning a bare scalar.
5. **Lambda-local `const` declarations** and multi-statement lambda bodies.
6. **Shape arithmetic in types** (`shape=[M*N]`), zero-rank tensors.
7. **More literals**: hex ints, f64, char; **implicit-promotion policy** (with a
   pinned conversion lattice).
8. **More builtins / casts** — register through `ir.Builtin` and the shared cast
   registry without grammar changes.
9. **Broadcasting sugar** in bindings (omitted trailing subscripts).
10. **User scheduling annotations** (`@tile`, `@fuse` hints) consumed by opt passes.
11. **Diagnostics**: did-you-mean suggestions, multiple errors per kernel, recovery
    parsing.
12. **Formatter / pretty-printer** for the source language.
13. **Doc-comment extraction** (`///`) for kernel documentation.

---

## Appendix A — the matmul_scale reference program (new syntax, normative)

This program is the acceptance fixture for Task 6; it must parse, pass semacheck,
lower, compile, and match the CPU reference.

```
@include "tca-common.tca";

@runtime(
  cxx_path="runtime/tca_runtime.h",
);

@name(cxx="MatMulScale")
kernel matmul_scale(
  lhs: tensor<dtype=f32, shape=[M, N]>,
  rhs: tensor<dtype=f32, shape=[N, Q]>,
  scale: f32,
) -> tensor<dtype=f32, shape=[M, Q]>
{
  dimvar (a, b, c);

  let output = map
    on(lhs[a][b], rhs[b][c])
    as (a, b, c)
    (l: f32, r: f32) -> f32:
      l * r
  | reduce(init = 0.0f)
    on(acc: f32, %0: f32)
    as (a, c) -> f32:
      acc + %0
  | map
    on()
    [scale] (v: f32) -> f32:
      v * scale
  ;

  return output;
}
```

(The middle `as (a, c)` and the final `map on()` are shown written; both are omittable
per §7.4 — the acceptance fixture keeps them written to pin the explicit-clause path,
and a second fixture pins the fully-omitted inference path.)
