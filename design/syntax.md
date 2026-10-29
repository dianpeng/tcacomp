# TCA Frontend: Syntax, Grammar, and Lowering

- **NOTES THIS IS A TOTALY NEW DESIGN OF FRONTEND**

# Design

## Syntax

- Sample

```
// global level annotation and instructions
@include "another-frontend.tca"; // including files
@runtime(
  cxx_path="a/b/c/tca.h" // <- path to tca runtime for cxx cuda output, notes this will change output include <path>
);

// The parser for annotation works like following
//
// 1] when it hits an annotation, it parse it with its grammar, all constants
// 2] it stores the annotation to a global field, waiting to be attached
//    ie which element to be attached with just parsed annotation
// 3] when it hits a kernel, it attaches the previously parsed annotation to it
// 4] otherwise, it is a global annotation

// Define a kernel
@name(cxx="MatMul") // annotation, ie instrumentation to be consumed by compiler, kernel level annotation
kernel matmul(
  lhs: tensor<dtype=f32, shape=[M, N]>, // dynamic shaped tensor definition
  rhs: tensor<dtype=f32, shape=[N, Q]>, // another one, notes trailing comma is always supported
) -> tensor<dtype=f32, shape=[M, Q]>    // output definition
{

  // define dimension varible
  dimvar a; // single dimensional variable
  dimvar (  // parallel definition of dimentional variable
    b,
    c,
  );

  // define local variable capture, ie all local variable are conceptually stored in GMEM
  let output  // output is **immutable, never allow inplace modification**

    : tensor<dtype=f32, shape=[M, Q]>  // OPTIONAL, compiler will try to use typeinfer to figure out
  =

    map // lambda keyword, ie map verb

    on( // keyword for what to map
      lhs[a][b], // binding, ie bind tensor's dimension to dimvar, implicit loop setup
      rhs[b][c], // binding, same, the binding value creates the **BoundedDimVar**
    )

    // OPTIONAL, the compiler will figure it out
    as (a, b, c) // output tensor shape

    // lambda expression, ie callback for element wise map
    // for map, the syntax is # of operands in on will result in # of parameters in lambda

    (
      l: f32, // strongly typed
      r: f32, // strongly typed
    ) -> f32: // return value typed
      l * r   // lambda can ONLY ALLOW A SINGLE EXPRESSION, directly write the expression

    | // <- pipe operator, chain the following operations

    reduce(       // lambda keyword, reduce, the '(' means reduce takes attributes, ie KeyedAttribute
      init= 0.0f, // keyed attribute, notes init CAN BE dimvar, will be translated into **InductionVariable of that loop**
    )

    // OPTIONAL, the compiler will figure it out
    as (a, c) // output tensor shape

    on( // similar to map, reduce's parameter
      current: f32, // the accumulated value, ALWAYS THE FIRST ONE
      %0: f32,      // placeholder, representing piped value
    ) -> f32:       // return value
      current + %0  // result

  ;  // expression is done, hence semicolon to close the expression


  // return the value
  return output;
}

```

- The above sample shows all the design decision for the syntax/frontend of the TCA
  - [1] immutable local
  - [2] optional typing but strongly typped
  - [3] pipe operator |
  - [4] annotation

- Design descision
  - [1] immutable
    - **FOR THIS VERSION**, tca does not inplace mutation of a tensor
      - 1) input parameters are immutable
      - 2) local variables are immutable
  - [2] sub kernel invocation is allowed
    - user can invoke another kernel, which effectively **inlined**
    - **NOTES:** for current version, it raise an runtime error, not supported yet

## Implementation

- All syntax code stays in its own module
  - internal/syntax/ast
  - internal/syntax/lexer
  - internal/syntax/parser
  - internal/syntax/semacheck

- The bridge between AST to IR is done by builder in ir
  - internal/ir/builder.go
  - [1] It builds a sea-of-nodes IR GRAPH from AST
  - [2] It desugar those syntax sugar
    - 1) pipe operator is gone
    - 2) placeholder parameter is wired correctly to specific expression
    - 3) annotation to be compiled to corresponding IRNODES attached to a specific kernel
    - 4) a top level module object is introduced in ir to represent a single syntax module
      - A sets of the kernel functions
      - Each one has its own compiled kernel
      - Module represents the full frontend syntax to be converted
        - Maybe multiple files kernel definitions
        - MULTIPLE kernel can be defined
