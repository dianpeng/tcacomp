# TCA Frontend: Syntax, Grammar, and Lowering

**Status:** proposed v0 frontend specification.

This document defines a C++-style language that lowers into TCA. It is a companion to the TCA IR design, not a replacement for it. The frontend introduces no thread binding, user-controlled tiling, arbitrary memory access, or explicit tensor-level control flow.

The grammar is written in **EBNF (Extended Backus–Naur Form)**.

## 1. Design Overview

### 1.1 C++-style, not a subset of standard C++

Use familiar declarations, template-like tensor types, semicolons, square-bracket indexing, trailing return types, C++ lambdas, and `static_cast`.

The TCA-specific extensions are:

- `kernel`: a logical tensor-computation entry point.
- `dim`: a nonnegative runtime shape parameter.
- `dimvar`: a bounded logical iteration axis.
- `map(...) as (...) apply lambda`: a structural elementwise computation.
- `reduce(init = ...) apply lambda`: reduction of the current last axis.
- `|`: composition of tensor-computation stages.

This language requires its own parser. A stock C++ compiler does not accept it; the backend emits ordinary CUDA C++.

**Naming convention:** frontend identifiers have no sigils. Frontend `m` becomes TCA `@m`; frontend local `output` becomes an SSA value such as `%output`. Thus the mandatory frontend `as (m, k, n)` corresponds exactly to IR `as (@m, @k, @n)`.

### 1.2 Preserve the TCA programming model

A user defines an iteration domain, supplies structurally indexed inputs, applies pure scalar lambdas, and optionally reduces dimensions. The compiler owns indexing, validity checks, scheduling, tiling, caching, collective participation, and unique store ownership.

Every map must have `as`, including a map after a reduction. A reduction removes exactly one axis: the last axis in the current order.

Named intermediates are immutable tensor values. Naming one does not require global-memory materialization or a separate CUDA launch; those are backend decisions subject to fusion legality.

### 1.3 Deliberately excluded from v0

No `for`, `while`, `if`, `switch`, recursion, pointers, references, mutable variables, indexed assignments, atomics, CUDA thread identifiers, explicit synchronization, user-defined callable functions, or arbitrary device-function calls.

Scalar `?:`, `&&`, and `||` are supported inside scalar expressions. They select scalar values; they do not add user-authored tensor-level control flow.

No preprocessing, templates, namespaces, classes, strings, or overloading of kernel names. Each kernel returns one tensor or scalar; multiple returned values can be a later extension.

## 2. Running Example

```cpp
kernel auto matmul_scale(
    dim M,
    dim N,
    dim K,
    tensor<f32, M, N> lhs,
    tensor<f32, N, K> rhs,
    f32 scale
) -> tensor<f32, M, K> {
    dimvar m = M, k = K, n = N;

    auto output =
        map(lhs[m][n], rhs[n][k]) as (m, k, n)
        apply [](f32 l, f32 r) -> f32 {
            return l * r;
        }
        | reduce(init = 0.0f)
        apply [](f32 acc, f32 value) -> f32 {
            return acc + value;
        }
        | map() as (m, k)
        apply [scale](f32 value) -> f32 {
            return value * scale;
        };

    return output;
}
```

The root map produces a logical value of shape `[M, K, N]`, not necessarily a materialized tensor. The reduction removes `n`, producing `[M, K]`. The final map scales that value without adding dimensions.

## 3. Declarations and Types

### 3.1 Kernel declarations

```cpp
kernel auto name(parameters) -> result_type {
    declarations;
    return result;
}
```

The trailing return type lets the result refer to shape parameters declared earlier in the signature, following a familiar C++ convention.

A kernel is a logical computation graph with a single entry block. It is not a promise of one physical CUDA launch. Dependencies that cannot legally fuse require an execution plan with separate launches or another supported synchronization mechanism.

Kernel declarations cannot call one another in v0. The host integration layer invokes them.

### 3.2 Shape parameters and tensor types

```cpp
dim M
tensor<f32, M, 128> input
```

`dim` is a shape type, not a synonym for `i32`. Its runtime value must be nonnegative and fit the runtime's supported indexing range. The launch interface supplies dimensions explicitly and validates the corresponding tensor shapes.

A tensor type has one scalar element type and one or more extents:

```cpp
tensor<f32, M, N>
tensor<f16, 64, 128>
```

Each extent is a nonnegative decimal integer literal or the name of a `dim` parameter. Shape arithmetic such as `M * N` is excluded from v0 types. Zero extents are valid.

Rank and element type are static; an extent may be dynamic. A dimension parameter must be declared before a parameter type that uses it. The trailing return type can reference all dimension parameters.

A fully reduced computation has a scalar type such as `f32`, rather than a frontend rank-zero `tensor<f32>` type.

For v0, input tensor views are dense row-major tensors. More general strides and layouts require an explicit extension to the tensor/launch contract, not a silent reinterpretation of this syntax.

### 3.3 Scalar types

```text
bool
f32  f16  bf16
i8   i16  i32
u8   u16  u32
```

`f8` is intentionally absent until the IR chooses concrete float8 formats.

Types must match exactly at lambda parameters, return expressions, scalar assignments, tensor assignments, and builtin calls. There are no implicit numeric conversions in v0. Use `static_cast<T>(value)` explicitly.

For the initial expression implementation, `f16` and `bf16` are storage/conversion types: cast to `f32` before arithmetic or builtin calls, then cast back if needed. Native low-precision arithmetic can be added through the same typed expression registry later.

### 3.4 Iteration-axis declarations

```cpp
dimvar m = M;
dimvar k = K, n = N;
```

A `dimvar` defines an axis with `init = 0`, `step = 1`, and the specified exclusive upper bound. For example, `m` represents all integers in `[0, M)`. The bound may be a `dim` parameter or a decimal integer literal; a literal defines a fixed-extent axis.

A `dimvar` is not a scalar value. It can appear only in structural tensor subscripts and `as` lists. It cannot be captured, passed to a builtin, cast, or referenced inside a lambda.

Axis identity is independent of extent. Two axes both bounded by `M` are still different axes.

### 3.5 Local declarations

```cpp
auto result = computation;
tensor<f32, M, K> result = computation;
const f32 coefficient = 2.0f;
```

All locals are immutable, whether or not `const` is written. Reassignment and shadowing are rejected. `auto` infers the entire scalar or tensor type, including rank and symbolic extents.

Kernel-body declarations may bind computation pipelines, scalar expressions over launch-uniform scalars, or aliases of existing values. Lambda-body declarations bind only scalar expressions.

A scalar produced by a tensor computation is a device-computed value. It may be returned, aliased, or used as an explicit map operand, but it may not participate in ordinary kernel-body scalar arithmetic. Such arithmetic must occur in a map lambda so the computation stays in TCA's structural graph.

## 4. Structural Indexing and Broadcasting

### 4.1 Tensor operands

```cpp
input[m][n]
bias[n]
weights[n][k]
```

These expressions are allowed only in a map's operand list. They describe the relation between tensor axes and the map's iteration axes; they are not scalar loads that a lambda can evaluate itself.

For a rank-R tensor, exactly R subscripts are required. Every subscript must be a declared `dimvar`, with no arithmetic, constants, repeated axes, or slicing in v0.

Therefore these forms are rejected:

```cpp
input[m + 1][n]  // Arbitrary index arithmetic.
input[0][n]      // Constant indexing is outside v0.
input[m][m]     // Diagonal/repeated-axis access is outside v0.
input[m]        // Wrong number of subscripts for a rank-2 tensor.
```

Each subscript axis must have the same extent as the corresponding tensor axis. Equal constants or the same shape symbol establish equality statically. Other required equalities become launch-time checks; they are never silently handled by truncation.

### 4.2 Broadcasting

An operand broadcasts across every map axis absent from its subscript list.

```cpp
map(input[m][n], bias[n]) as (m, n)
```

Here `bias[n]` broadcasts across `m`. A scalar operand broadcasts across every axis.

A dimension of extent 1 does not automatically broadcast against a differently bounded axis. Broadcasting is expressed by axis omission, not by a NumPy-style singleton rule.

### 4.3 What establishes the root domain?

For a root map, the bounded axes in `as` define its complete iteration domain. Every axis appearing in an operand must appear in `as` exactly once. Axes may appear in `as` without appearing in any operand, allowing explicit replication and constant fills.

```cpp
map(2.0f) as (m, n)
apply [](f32 value) -> f32 {
    return value;
}
```

This creates a tensor of shape `[M, N]` filled with `2.0f`. It lowers to `iter(@m/M) * iter(@n/N)`, even though its operand is scalar.

**v0 boundary:** a root map must have at least one explicit operand and at least one axis. Root `map()`, and root `map(...) as ()`, are rejected. This avoids assigning new semantics to the IR's currently special zero-iteration root-map case. Scalar computation without a tensor domain can be expressed directly as a scalar expression.

## 5. Maps, Pipelines, and Reductions

### 5.1 Root map

```cpp
map(operand0, operand1) as (axis0, axis1)
apply [](T0 x, T1 y) -> R {
    return expression;
}
```

Each explicit operand supplies one scalar lambda argument, in source order. The lambda returns exactly one scalar. The result is a tensor of that scalar type, with extents ordered by `as`.

The scalar lambda cannot return a tensor, tuple, pointer, or reference in v0.

### 5.2 Piped map

```cpp
producer
| map(extra_operand) as (remaining_axes)
apply [](T previous, U extra) -> R {
    return expression;
}
```

A piped map receives the previous stage's value as its **first** lambda argument. Explicit operands follow it in source order. There is no frontend `%0` placeholder.

Thus:

```cpp
producer
| map() as (m, n)
apply [](f32 previous) -> f32 {
    return previous * previous;
}
```

has one lambda argument, not zero.

A piped map must repeat the incoming axis order exactly. It cannot introduce an axis, permute axes, or resurrect a reduced axis in v0. Its explicit tensor operands may use any subset of the current axes; omitted axes broadcast.

A map after a complete reduction uses `as ()` and produces a scalar. This is a fused zero-iteration map, which has defined semantics in TCA.

To broadcast a reduced tensor into a larger domain or reorder a tensor, bind the producer to a name and start a new root map. That creates a graph dependency; it does not guarantee fusion.

### 5.3 Reduction

```cpp
producer
| reduce(init = 0.0f)
apply [](f32 acc, f32 value) -> f32 {
    return acc + value;
}
```

A reduction is only a pipeline stage. It takes no tensor subscript or explicit axis argument: it always removes the final axis of the current ordered value.

For example:

```text
(m, k, n) -> reduce -> (m, k) -> reduce -> (m)
```

The frontend records the removed axis explicitly in the IR as `reduce<@n>`, then `reduce<@k>`. This annotation remains stable after tiling.

For v0, the input element type, initializer type, both lambda parameter types, and lambda result type must all be the same scalar type T. Use a map with an explicit cast before reduction to change accumulator precision. Mixed-type reducers are deferred.

The combiner must be associative and commutative under the chosen numerical contract. The compiler can type-check the lambda but cannot generally prove these properties. A custom reduction lambda is the author's assertion that reordering and parallel grouping are permitted. Floating-point sums may differ bitwise across schedules.

The initializer is a seed applied **once per logical output**, not once per lane, warp, or tile. For values `v0 ... vn`, the semantics are a permitted regrouping of:

```text
combine(init, v0, ..., vn)
```

An empty reduction returns the initializer. A seed need not be an identity; runtime partial accumulators must preserve this distinction.

### 5.4 Collective results are not extra elements

After a parallel reducer broadcasts a result, its physical replicas still represent one logical value for that output coordinate. A following map, reduction, or store must not count replicas as additional elements or create duplicate stores.

The frontend operates exclusively on these logical values. It does not expose lane participation, ownership predicates, or scratch-memory layout.

### 5.5 Named values and output layout

A tensor binding records the producing pipeline's axis order as its logical dimension order. Newly materialized values use dense row-major layout in that order.

```cpp
auto transposed = map(input[m][n]) as (n, m) ...;
```

This produces shape `[N, M]` and, if materialized, stores at `transposed[n][m]`. It is a value-producing transpose, not a promise of a zero-copy view.

A later map may attach different axis names to a named tensor's positional dimensions, subject to extent validation. This is ordinary structural rebinding, not mutation of the producer's axis identities.

There is no user-written `store`. Tensor materialization and return lowering insert stores, preserving one owner per logical output address. Inputs are read-only, output storage is fresh, and in-place aliasing is excluded from v0.

## 6. Scalar Lambdas and Expressions

### 6.1 Lambda form

```cpp
[scale](f32 value) -> f32 {
    const f32 scaled = value * scale;
    return scaled > 0.0f ? scaled : 0.0f;
}
```

A lambda has an explicit capture list, typed parameters, an explicit scalar return type, zero or more immutable scalar declarations, and exactly one final `return`.

`[]` means no captures. Only explicit by-value captures are supported. `[=]`, `[&]`, reference captures, capture initializers, `mutable`, generic parameters, and omitted parameter/result types are not supported.

Captured values must be launch-uniform scalar parameters or scalar locals derived solely from such parameters and constants. Tensor values, shape parameters, axes, and device-computed scalars cannot be captured. Device-computed scalar dependencies instead appear as explicit map operands.

A capture becomes an explicit SSA operand of the TCA expression graph. It does not create a runtime C++ closure object.

### 6.2 Expressions and typing

Expressions support literals, scalar names, parentheses, unary operations, arithmetic, comparisons, Boolean operations, integer bitwise operations, scalar selection, explicit casts, and registered pure builtin calls.

There are no assignment expressions, increments, decrements, comma expressions, address operations, dereferences, subscripts, or member accesses inside a lambda.

Typing rules:

- Arithmetic operands have the same supported arithmetic type; the result has that type. There are no implicit integer promotions.
- `%`, shifts, and integer bitwise operators require integer operands. In v0 their operand types must match.
- Comparisons require compatible same-type operands and return `bool`.
- `!`, `&&`, and `||` require Boolean operands.
- `condition ? a : b` requires a Boolean condition and identically typed branches.
- A cast has the explicit target type and is accepted only if the shared IR cast registry supports that source/target pair.
- A call resolves to exactly one registered signature. Unknown calls and ambiguous matches are errors.

Numeric corner cases, supported cast pairs, and builtin behavior are defined by the shared frontend/IR expression registry. This frontend must not introduce a second, inconsistent numerical policy. In particular, it does not promise that out-of-range float-to-integer conversion is safe.

**Numerical policy (pinned for v0).** Scalar floating-point evaluation follows IEEE 754 semantics per operation: no fast-math, no flush-to-zero (`ftz`), and no silent precision reduction of any kind (in particular no TF32 or tensor-core down-conversion). FMA contraction (`a * b + c` contracting to a single rounding) is **permitted by default** and applied only per operation, never as part of a wider reassociation; a user who needs the exact unfused rounding can obtain it in the generated code (the compiler's e2e verification lane compiles with `--fmad=false`, and a per-kernel opt-out flag may be added later). For a fixed compiled kernel, execution is deterministic — no atomics and no data-dependent scheduling exist — so run-to-run results are bitwise identical; parallel reductions remain schedule-dependent across *different* compilations as stated in the IR spec.

`&&`, `||`, and `?:` retain C++-style conditional evaluation. An unselected expression is not semantically evaluated. The expression IR must preserve this behavior, for example with lazy select expressions; codegen must not eagerly evaluate a potentially invalid unselected division or conversion.

### 6.3 Literals

Decimal integer literals have type `i32`; a trailing `u` or `U` gives `u32`. Literals must fit their type. Unary minus is a separate operator.

Floating-point literals require an `f` or `F` suffix and have type `f32`:

```cpp
0.0f
1.25f
1e-5f
```

There are no `f64`, hexadecimal, octal, binary, character, or string literals in v0. Use explicit casts for smaller integer and floating types. `true` and `false` have type `bool`.

The expression environment provides a reserved predefined scalar constant `inf_f32`, the positive infinity value of type `f32`. It can be used without capture; `-inf_f32` is useful as a maximum-reduction seed.

### 6.4 Initial builtin registry

The initial registry can expose these CUDA-compatible names with exact signatures:

| Name | Signature |
|---|---|
| `expf`, `logf`, `sqrtf`, `rsqrtf` | `(f32) -> f32` |
| `fabsf`, `tanhf`, `sinf`, `cosf` | `(f32) -> f32` |
| `fminf`, `fmaxf` | `(f32, f32) -> f32` |
| `fmaf` | `(f32, f32, f32) -> f32` |

Additional CUDA scalar builtins can be registered without changing the grammar. This is not permission to call arbitrary CUDA functions. Memory access, synchronization, atomics, and thread-index builtins are not scalar-lambda calls.

## 7. Complete v0 EBNF

### 7.1 Notation

`=` defines a production; `,` means concatenation; `|` separates alternatives; `[ ... ]` is optional; `{ ... }` repeats zero or more times. Quoted strings are terminals. `(* ... *)` introduces a grammar comment. `? ... ?` denotes a lexical character class described in prose.

Quoted `"|"` is the literal pipe/bitwise-OR token, distinct from EBNF alternation.

The grammar defines the complete syntax. Name resolution, root-versus-piped map checks, typing, valid indices, and shape checks are semantic validation, specified in the other sections.

### 7.2 Tokens

The scanner uses longest-match tokenization. Whitespace and comments are discarded between tokens. Comment markers inside comments have no nesting behavior.

```ebnf
letter             = ? ASCII letter A-Z or a-z ? ;
digit              = "0" | "1" | "2" | "3" | "4"
                   | "5" | "6" | "7" | "8" | "9" ;
nonzero-digit      = "1" | "2" | "3" | "4" | "5"
                   | "6" | "7" | "8" | "9" ;
identifier-start   = letter | "_" ;
identifier-rest    = identifier-start | digit ;
identifier         = identifier-start, { identifier-rest } ;

decimal-digits     = digit, { digit } ;
decimal-integer    = "0" | nonzero-digit, { digit } ;
integer-literal    = decimal-integer, [ "u" | "U" ] ;
exponent           = ("e" | "E"), [ "+" | "-" ], decimal-digits ;
float-suffix       = "f" | "F" ;
float-literal      = (
                       decimal-digits, ".", [ decimal-digits ], [ exponent ]
                     | ".", decimal-digits, [ exponent ]
                     | decimal-digits, exponent
                    ), float-suffix ;
boolean-literal    = "true" | "false" ;
literal            = integer-literal | float-literal | boolean-literal ;

whitespace         = ? space, tab, carriage return, or line feed ? ;
line-comment       = "//", { ? any character except CR or LF ? } ;
block-comment      = "/*", ? shortest character sequence ending before the next */ ?, "*/" ;
trivia             = whitespace | line-comment | block-comment ;
```

A line comment ends at a line ending or end of file. An unterminated block comment is a lexical error. An identifier token cannot equal a reserved keyword. An otherwise contiguous malformed numeric token, such as `12xyz`, is a lexical error rather than an integer followed by an identifier.

Reserved keywords:

```text
kernel auto dim dimvar tensor const return
map as apply reduce init static_cast
bool f32 f16 bf16 i8 i16 i32 u8 u16 u32
true false
```

`inf_f32` and registered builtin names are predefined names that cannot be redeclared. Other unsupported C++ constructs do not acquire meaning simply because their spelling is familiar.

### 7.3 Translation unit, declarations, and types

```ebnf
translation-unit   = { kernel-declaration } ;

kernel-declaration = "kernel", "auto", identifier,
                     "(", [ kernel-parameters ], ")",
                     "->", value-type, kernel-body ;

kernel-parameters  = kernel-parameter, { ",", kernel-parameter } ;
kernel-parameter   = "dim", identifier
                   | value-type, identifier ;

value-type         = scalar-type | tensor-type ;
scalar-type        = "bool" | "f32" | "f16" | "bf16"
                   | "i8" | "i16" | "i32"
                   | "u8" | "u16" | "u32" ;
tensor-type        = "tensor", "<", scalar-type,
                     ",", extent, { ",", extent }, ">" ;
extent             = decimal-integer | identifier ;

kernel-body        = "{", { kernel-statement }, return-statement, "}" ;
kernel-statement   = axis-declaration | value-declaration ;

axis-declaration   = "dimvar", axis-binding, { ",", axis-binding }, ";" ;
axis-binding       = identifier, "=", extent ;

value-declaration  = [ "const" ], ("auto" | value-type), identifier,
                     "=", initializer, ";" ;
initializer        = pipeline | scalar-expression ;
return-statement   = "return", initializer, ";" ;
```

An identifier in an extent must resolve to a `dim` parameter. Scalar-expression syntax permits a bare identifier; in an initializer or kernel return, a bare tensor identifier is also accepted as an alias/value reference. Tensor names are otherwise rejected in scalar expressions.

### 7.4 Tensor-computation pipelines

```ebnf
pipeline           = map-stage, { "|", pipeline-stage } ;
pipeline-stage     = map-stage | reduce-stage ;

map-stage          = "map", "(", [ map-operands ], ")",
                     "as", "(", [ axis-list ], ")",
                     "apply", scalar-lambda ;

map-operands       = map-operand, { ",", map-operand } ;
map-operand        = indexed-operand | scalar-expression ;
indexed-operand    = identifier, subscript, { subscript } ;
subscript          = "[", identifier, "]" ;
axis-list          = identifier, { ",", identifier } ;

reduce-stage       = "reduce", "(", "init", "=", scalar-expression, ")",
                     "apply", scalar-lambda ;
```

The first map of a pipeline is a root map. Every subsequent map is piped. Empty operand and axis lists are accepted syntactically because they are valid for some piped maps; the root-map restrictions are checked afterward.

A pipeline cannot begin with an existing tensor name or a reduce. To reduce an existing tensor, start with an identity map over its elements.

### 7.5 Lambdas

```ebnf
scalar-lambda      = "[", [ capture-list ], "]",
                     "(", [ lambda-parameters ], ")",
                     "->", scalar-type, lambda-body ;

capture-list       = identifier, { ",", identifier } ;
lambda-parameters  = lambda-parameter, { ",", lambda-parameter } ;
lambda-parameter   = scalar-type, identifier ;

lambda-body        = "{", { scalar-declaration },
                     "return", scalar-expression, ";", "}" ;

scalar-declaration = [ "const" ], ("auto" | scalar-type), identifier,
                     "=", scalar-expression, ";" ;
```

A reduce lambda must have exactly two parameters. A map lambda's parameter count is determined by the root/piped argument rule. Captures do not count as positional parameters.

### 7.6 Scalar expressions and precedence

The following productions encode C++-style precedence. Binary operators at each repeated level associate left-to-right; the conditional operator associates right-to-left.

```ebnf
scalar-expression  = conditional-expression ;

conditional-expression
                   = logical-or-expression,
                     [ "?", scalar-expression, ":", conditional-expression ] ;

logical-or-expression
                   = logical-and-expression, { "||", logical-and-expression } ;
logical-and-expression
                   = bitwise-or-expression, { "&&", bitwise-or-expression } ;
bitwise-or-expression
                   = bitwise-xor-expression, { "|", bitwise-xor-expression } ;
bitwise-xor-expression
                   = bitwise-and-expression, { "^", bitwise-and-expression } ;
bitwise-and-expression
                   = equality-expression, { "&", equality-expression } ;

equality-expression
                   = relational-expression,
                     { ("==" | "!="), relational-expression } ;
relational-expression
                   = shift-expression,
                     { ("<" | "<=" | ">" | ">="), shift-expression } ;
shift-expression   = additive-expression,
                     { ("<<" | ">>"), additive-expression } ;
additive-expression
                   = multiplicative-expression,
                     { ("+" | "-"), multiplicative-expression } ;
multiplicative-expression
                   = unary-expression,
                     { ("*" | "/" | "%"), unary-expression } ;

unary-expression   = ("+" | "-" | "!" | "~"), unary-expression
                   | cast-expression
                   | primary-expression ;

cast-expression    = "static_cast", "<", scalar-type, ">",
                     "(", scalar-expression, ")" ;

primary-expression = literal
                   | identifier, [ "(", [ scalar-arguments ], ")" ]
                   | "(", scalar-expression, ")" ;

scalar-arguments   = scalar-expression, { ",", scalar-expression } ;
```

A call-form identifier must resolve to a registered builtin. A non-call identifier must resolve to an allowed scalar value or predefined scalar constant, subject to the bare-tensor-alias exception in Section 7.3.

The `|` token is not ambiguous between scalar and pipeline contexts: scalar bitwise OR occurs inside a scalar expression; pipeline composition occurs after a complete map/reduce stage, whose lambda ends with `}`.

## 8. AST-to-TCA Lowering Contract

### 8.1 Frontend objects

A minimal AST needs nodes for kernel declarations, shape/scalar/tensor parameters, axis declarations, immutable bindings, pipelines, map stages, reduce stages, indexed operands, scalar lambdas, and typed scalar expressions.

Each elaborated pipeline stage carries:

```text
value element type
ordered logical axes
extent of each axis
operand-to-axis access maps
scalar lambda graph and explicit captures
producer dependencies
```

Frontend names are resolved before graph construction. The AST does not need to know tile sizes, physical threads, reducer scratch layouts, or cache placement.

### 8.2 Core mapping

| Frontend | TCA |
|---|---|
| `dim M` | Runtime shape SSA parameter `%M` |
| `dimvar m = M` | Axis `@m(init=0, step=1)` bounded by `%M` at use |
| Root `as (m, n)` | `iter(@m/M) * iter(@n/N)` and map order `(@m, @n)` |
| `input[m][n]` | Structural operand `%input[@m][@n]`, later an explicit load |
| Root `map(a, b)` | Map with lambda operands `(a, b)` |
| Piped `map(a)` | `iter() -> map(%0, a, ...)` |
| `[scale]` | Explicit scalar dependency on `%scale` |
| `static_cast<f32>(x)` | Typed `cast<f32>(%x)` expression |
| `reduce(init = s)` | `reduce<last_axis>(init=s, %0, combiner)` |
| `auto x = pipeline` | Immutable SSA binding; inferred ordered tensor/scalar type |
| Tensor materialization | Structural `store` indexed in the value's axis order |
| `return x` | Logical return; ABI lowering publishes the output |

### 8.3 Stage-by-stage elaboration

For a root map:

1. Resolve the `as` axes and obtain their bounds.
2. Build the Cartesian-product iteration domain in that order for construction purposes; record `as` explicitly because Cartesian product itself is unordered.
3. Validate operand ranks, axis references, extent equalities, and broadcasting.
4. Type-check the lambda against the explicit scalar operand types.
5. Produce the result type and ordered axis list.

For a reduce, reject an empty axis list, type-check its initializer and combiner, record the last axis as the reduction target, and remove that axis from the result.

For a piped map, verify an unchanged axis list, prepend the incoming scalar to its operand list, and type-check the lambda.

For a named consumer, preserve the producer dependency and the consumer's structural access mapping. Do not declare it fused merely because both expressions appear in the same frontend kernel.

### 8.4 Notation used in the examples

The TCA listings below show pre-scheduling, pre-tiling IR. Bound symbols such as `M` stand for shape parameters. Scalar expression lambdas use compact `lambda(...)` notation.

For tensors, examples show an explicit conceptual materialization using `store`. These stores provide a clear reference lowering; the optimizer may eliminate intermediate materialization through legal fusion.

A scalar-returning chain is shown as an SSA scalar. The physical kernel ABI may still store it into a device output buffer with a unique owner.

The order remains:

```text
frontend AST -> typed TCA graph -> graph optimization and scheduling
             -> iteration/tile lowering -> reducer lowering
             -> runtime-call lowering -> CUDA C++
```

Frontend syntax does not alter the requirement that reducer placement happens after the final tile-loop structure exists.

## 9. Examples with TCA Mapping

### 9.1 Elementwise AXPY

**Frontend**

```cpp
kernel auto axpy(
    dim N,
    tensor<f32, N> x,
    tensor<f32, N> y,
    f32 alpha
) -> tensor<f32, N> {
    dimvar i = N;

    auto output =
        map(x[i], y[i]) as (i)
        apply [alpha](f32 xv, f32 yv) -> f32 {
            return alpha * xv + yv;
        };

    return output;
}
```

**TCA**

```text
iter(@i/N) ->
  map(%x[@i], %y[@i], lambda(xv, yv): %alpha * xv + yv) as (@i) |
  store(%output[@i], %0)

return %output
```

`alpha` is an explicit scalar dependency of the expression graph. Whether multiplication and addition contract into an FMA follows the pinned numerical policy of [Section 6.2](#62-expressions-and-typing): contraction is permitted per operation unless the kernel opts out; users can request a fused operation explicitly with `fmaf`.

### 9.2 Bias broadcasting and ReLU

**Frontend**

```cpp
kernel auto bias_relu(
    dim M,
    dim N,
    tensor<f32, M, N> x,
    tensor<f32, N> bias
) -> tensor<f32, M, N> {
    dimvar m = M, n = N;

    auto output =
        map(x[m][n], bias[n]) as (m, n)
        apply [](f32 value, f32 b) -> f32 {
            const f32 shifted = value + b;
            return shifted > 0.0f ? shifted : 0.0f;
        };

    return output;
}
```

**TCA**

```text
iter(@m/M) * iter(@n/N) ->
  map(%x[@m][@n], %bias[@n],
      lambda(value, b): select(value + b > 0.0f, value + b, 0.0f))
      as (@m, @n) |
  store(%output[@m][@n], %0)

return %output
```

`select` here denotes the typed conditional expression, not a user-callable builtin. The scalar local becomes an SSA definition; expression simplification can share the repeated `value + b`.

### 9.3 Matrix multiplication followed by scale

The full frontend appears in Section 2.

**TCA**

```text
iter(@m/M) * iter(@k/K) * iter(@n/N) ->
  map(%lhs[@m][@n], %rhs[@n][@k],
      lambda(l, r): l * r) as (@m, @k, @n) |
  reduce<@n>(init=0.0f, %0, lambda(acc, value): acc + value) |
  iter() ->
    map(%0, lambda(value): value * %scale) as (@m, @k) |
  store(%output[@m][@k], %0)

return %output
```

**Axis/type evolution**

```text
root map:   (m, k, n)    tensor<f32, M, K, N>
reduce:     (m, k)       tensor<f32, M, K>
scaled map: (m, k)       tensor<f32, M, K>
```

The logical product tensor need not be stored. The fixed `as` order gives the tiled nest `m_o, k_o, n_o, m_i, k_i, n_i`; reducer initialization and final combination surround `n_o` and maintain accumulator state for the surviving inner coordinates.

### 9.4 Low-precision input, f32 accumulation, low-precision output

**Frontend**

```cpp
kernel auto row_sum_f16(
    dim M,
    dim N,
    tensor<f16, M, N> input
) -> tensor<f16, M> {
    dimvar m = M, n = N;

    auto output =
        map(input[m][n]) as (m, n)
        apply [](f16 value) -> f32 {
            return static_cast<f32>(value);
        }
        | reduce(init = 0.0f)
        apply [](f32 acc, f32 value) -> f32 {
            return acc + value;
        }
        | map() as (m)
        apply [](f32 value) -> f16 {
            return static_cast<f16>(value);
        };

    return output;
}
```

**TCA**

```text
iter(@m/M) * iter(@n/N) ->
  map(%input[@m][@n], lambda(value): cast<f32>(value)) as (@m, @n) |
  reduce<@n>(init=0.0f, %0, lambda(acc, value): acc + value) |
  iter() ->
    map(%0, lambda(value): cast<f16>(value)) as (@m) |
  store(%output[@m], %0)

return %output
```

There is no implicit accumulator promotion. When `N == 0`, each existing row produces the seed `0.0f`, then converts to `f16`.

### 9.5 Two chained reductions and a scalar epilogue

**Frontend**

```cpp
kernel auto total_scaled(
    dim M,
    dim N,
    tensor<f32, M, N> input,
    f32 scale
) -> f32 {
    dimvar m = M, n = N;

    auto total =
        map(input[m][n]) as (m, n)
        apply [](f32 value) -> f32 {
            return value;
        }
        | reduce(init = 0.0f)
        apply [](f32 acc, f32 value) -> f32 {
            return acc + value;
        }
        | reduce(init = 0.0f)
        apply [](f32 acc, f32 value) -> f32 {
            return acc + value;
        }
        | map() as ()
        apply [scale](f32 value) -> f32 {
            return value * scale;
        };

    return total;
}
```

**TCA**

```text
%total =
  iter(@m/M) * iter(@n/N) ->
    map(%input[@m][@n], lambda(value): value) as (@m, @n) |
    reduce<@n>(init=0.0f, %0, lambda(acc, value): acc + value) |
    reduce<@m>(init=0.0f, %0, lambda(acc, value): acc + value) |
    iter() -> map(%0, lambda(value): value * %scale) as ()

return %total
```

The first reduce produces one value per `m`; the second consumes those logical values once each. Broadcast replicas from the first reduction must not become duplicate contributions. A backend that cannot yet lower a chained collective must split the computation or report unsupported scheduling, not silently miscompile it.

### 9.6 Transpose through output-axis order

**Frontend**

```cpp
kernel auto transpose(
    dim M,
    dim N,
    tensor<f32, M, N> input
) -> tensor<f32, N, M> {
    dimvar m = M, n = N;

    auto output =
        map(input[m][n]) as (n, m)
        apply [](f32 value) -> f32 {
            return value;
        };

    return output;
}
```

**TCA**

```text
iter(@n/N) * iter(@m/M) ->
  map(%input[@m][@n], lambda(value): value) as (@n, @m) |
  store(%output[@n][@m], %0)

return %output
```

Input indexing and output order are distinct. Identity-map elimination must retain the transpose access relation and the surviving order metadata; identity of the scalar lambda does not make the whole tensor operation an identity.

### 9.7 A piped map with an additional broadcast operand

**Frontend**

```cpp
kernel auto row_sum_plus_bias(
    dim M,
    dim N,
    tensor<f32, M, N> input,
    tensor<f32, M> bias
) -> tensor<f32, M> {
    dimvar m = M, n = N;

    auto output =
        map(input[m][n]) as (m, n)
        apply [](f32 value) -> f32 {
            return value;
        }
        | reduce(init = 0.0f)
        apply [](f32 acc, f32 value) -> f32 {
            return acc + value;
        }
        | map(bias[m]) as (m)
        apply [](f32 sum, f32 b) -> f32 {
            return sum + b;
        };

    return output;
}
```

**TCA**

```text
iter(@m/M) * iter(@n/N) ->
  map(%input[@m][@n], lambda(value): value) as (@m, @n) |
  reduce<@n>(init=0.0f, %0, lambda(acc, value): acc + value) |
  iter() ->
    map(%0, %bias[@m], lambda(sum, b): sum + b) as (@m) |
  store(%output[@m], %0)

return %output
```

The previous result is argument 0; `bias[m]` is argument 1. The frontend does not require explicit scalar captures for this tensor dependency.

### 9.8 Stable row-wise softmax with named intermediates

This example assumes finite input values. It illustrates graph dependencies and reduction-to-map broadcasting, not a promise that every stage can run in one CUDA launch.

**Frontend**

```cpp
kernel auto softmax(
    dim M,
    dim N,
    tensor<f32, M, N> input
) -> tensor<f32, M, N> {
    dimvar m = M, n = N;

    auto row_max =
        map(input[m][n]) as (m, n)
        apply [](f32 value) -> f32 {
            return value;
        }
        | reduce(init = -inf_f32)
        apply [](f32 acc, f32 value) -> f32 {
            return fmaxf(acc, value);
        };

    auto numerators =
        map(input[m][n], row_max[m]) as (m, n)
        apply [](f32 value, f32 maximum) -> f32 {
            return expf(value - maximum);
        };

    auto row_sum =
        map(numerators[m][n]) as (m, n)
        apply [](f32 value) -> f32 {
            return value;
        }
        | reduce(init = 0.0f)
        apply [](f32 acc, f32 value) -> f32 {
            return acc + value;
        };

    auto output =
        map(numerators[m][n], row_sum[m]) as (m, n)
        apply [](f32 value, f32 denominator) -> f32 {
            return value / denominator;
        };

    return output;
}
```

**Reference TCA lowering**

```text
// Statement 1: row maximum.
iter(@m/M) * iter(@n/N) ->
  map(%input[@m][@n], lambda(value): value) as (@m, @n) |
  reduce<@n>(init=-inf_f32, %0,
             lambda(acc, value): fmaxf(acc, value)) |
  store(%row_max[@m], %0)

// Statement 2: exponentials with a broadcast row maximum.
iter(@m/M) * iter(@n/N) ->
  map(%input[@m][@n], %row_max[@m],
      lambda(value, maximum): expf(value - maximum)) as (@m, @n) |
  store(%numerators[@m][@n], %0)

// Statement 3: row denominator.
iter(@m/M) * iter(@n/N) ->
  map(%numerators[@m][@n], lambda(value): value) as (@m, @n) |
  reduce<@n>(init=0.0f, %0, lambda(acc, value): acc + value) |
  store(%row_sum[@m], %0)

// Statement 4: normalization.
iter(@m/M) * iter(@n/N) ->
  map(%numerators[@m][@n], %row_sum[@m],
      lambda(value, denominator): value / denominator) as (@m, @n) |
  store(%output[@m][@n], %0)

return %output
```

A conservative executable lowering uses four dependency-ordered physical kernels with intermediate buffers. Legal fusion can eliminate some materializations, but a same-launch reload of another thread's unfinished output is not permitted.

When `N == 0`, the returned tensor is empty. The intermediate reductions have their seed values, but there are no output points at which to invoke the division lambda.

### 9.9 Scalar fill over an explicit domain

**Frontend**

```cpp
kernel auto fill(
    dim M,
    dim N,
    f32 value
) -> tensor<f32, M, N> {
    dimvar m = M, n = N;

    auto output =
        map(value) as (m, n)
        apply [](f32 v) -> f32 {
            return v;
        };

    return output;
}
```

**TCA**

```text
iter(@m/M) * iter(@n/N) ->
  map(%value, lambda(v): v) as (@m, @n) |
  store(%output[@m][@n], %0)

return %output
```

This demonstrates why root `as` may contain an axis absent from all operand subscripts. Removing the identity lambda must not remove the iteration domain or the stores.

## 10. Required Diagnostics

Diagnostics should identify the source span, the actual stage type/order, and the expected type/order where applicable.

| Invalid program condition | Example diagnostic |
|---|---|
| Map omits `as` | `map requires an explicit as(...) axis order` |
| Duplicate axis in `as` | `axis n appears more than once in the map domain` |
| Operand axis absent from `as` | `operand uses axis k, which is not in this map domain` |
| Unbound subscript name | `m is not a declared dimvar` |
| Wrong number of tensor subscripts | `input has rank 2 but this operand has 1 subscript` |
| Statically incompatible extents | `axis n has extent 64, but operand axis 1 has extent 128` |
| Root has an empty domain | `root maps require a nonempty domain; use a scalar expression` |
| Piped map reorders axes | `incoming order is (m, n), but map declares (n, m)` |
| Piped map adds an axis | `axis n is not live after the preceding reduction` |
| Reduction on a scalar | `reduce requires at least one remaining axis` |
| Map parameter-count mismatch | `piped map expects 2 parameters: previous value plus 1 operand` |
| Mixed reducer types | `reduction input is f16 but initializer is f32; insert a cast map` |
| Axis referenced inside lambda | `dimvar n is structural and is not visible as a scalar value` |
| Uncaptured outer scalar | `scale must appear in the lambda capture list` |
| Tensor capture | `capture input is a tensor; pass indexed values through map operands` |
| Unsupported call | `function atomicAdd is not a registered pure scalar builtin` |
| Return shape mismatch | `return has shape [N, M], expected [M, N]` |
| Assignment to an existing name | `TCA frontend values are immutable` |

Dynamic equalities are validated before launch. Failure must produce a shape error; the runtime must not execute a partially defined result by skipping logically required operands.

The frontend cannot validate associativity in general. Any diagnostic for a suspicious custom reduction should be presented as a warning or a documented unsupported operation, not as a claim of mathematical proof.

## 11. Implementation Plan

### 11.1 Implement the elaborator before the parser

Keep the first milestone aligned with direct IR construction. Build an AST/elaboration API that can construct the same graph as handwritten TCA graph code; feed it programmatically at first.

The parser should eventually populate that AST, not duplicate lowering or verifier logic.

### 11.2 Suggested components

1. **Lexer and parser:** implement this grammar, source locations, and useful recovery at semicolons and closing braces.
2. **Name and shape resolution:** distinguish dimensions, axes, tensors, scalar values, and builtin names; collect dynamic equality checks.
3. **Scalar expression checking:** share operator, cast, and builtin registries with the IR.
4. **Pipeline elaboration:** compute ordered axes, result types, capture dependencies, and explicit reduction targets.
5. **TCA graph builder:** create maps, reductions, structural accesses, immutable values, and returns.
6. **Reference execution/IR tests:** compare frontend-generated IR with direct graph construction before enabling fusion and scheduling optimizations.

### 11.3 Important tests

Test empty output dimensions separately from empty reduction dimensions; a row sum with `N == 0` must still produce one seed value per existing row.

Test nonzero reduction seeds to detect initialization once per physical participant instead of once per logical output.

Test chained reductions and post-reduction stores to detect duplicate contributions or stores from broadcast replicas.

Test transpose and fill after identity-map elimination to ensure axis metadata and access mappings survive simplification.

Test launch-time shape failures, explicit casts, capture resolution, piped-map argument order, and conditional expressions with an invalid unselected operation.

Test named intermediate dependencies both with fusion disabled and with each supported fusion enabled. Softmax is a useful graph test even before it becomes a performance target.

## 12. Explicit Decisions to Carry into the IR Specification

This proposal fixes several frontend choices rather than leaving them implicit:

- `as` is mandatory and uses ordinary frontend identifiers; IR sigils remain internal.
- Axes are explicitly bounded at declaration. Root `as` establishes the complete domain, including axes needed only for broadcasting or filling.
- Piped maps receive the previous value as argument 0, then explicit operands, and preserve the incoming axis order.
- v0 reducers are homogeneous typed combiners with a once-per-output seed.
- Named intermediates are values, not user-managed buffers or fusion directives.
- Fresh row-major outputs are stored in result-axis order; there are no arbitrary user stores or in-place writes.
- Scalar lambdas support explicit captures and casts, registered pure CUDA builtins, and conditional scalar expressions, but cannot inspect indices.
- Root scalar/empty-iteration maps remain excluded until their IR semantics are deliberately defined.

These choices are sufficient to implement the frontend independently of advanced scheduling. They do not resolve the separate backend questions of multidimensional whole-warp ownership, chained-collective scheduling, or multi-kernel execution planning; the frontend must preserve enough structure for those components to enforce correctness.
