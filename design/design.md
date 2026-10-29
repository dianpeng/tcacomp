# TCA

- An optimizing compiler for writing CUDA kernel using extremely simple syntax

# Requirements

- Using golang to implement the compiler

- Compiler will compile user's input code and generate backend code
  - For now, it generates C++ cuda code

- Additionally, a runtime library must be created to be used by the generated C++ code
  - Reference scalfold DSL: /home/ps/Workspace/cuda/project/scalfold

- Others
  - using golang, pure Go (standard library only, no cgo / external dependencies)
  - using c++20
  - using CMake for runtime library
  - organize folder as
    - runtime/ <- c++ code
    - pkg/     <- public golang
    - internal/ <- internal golang module
    - cmd       <- binary/cli

- Modules
  - internal/syntax <- parser/lexer/ast input handling
  - internal/ir/    <- major TCA part, an SSA IR graph,
  - util/           <- general helper
  - opt/            <- optimization pass
  - pass/           <- pass manager, abstract factory, boilerplate
  - analysis/       <- analysis pass, ie collect information 
  - codegen/        <- code generation
  - diagnostic/     <- diagnostic information
  - visualization/  <- visualization of IR code, ie graphviz
  - log/            <- proper log infra, compiler needs LOTS of logs
  - flags/          <- global flag to tweak for compiler optimization passes
  - ...

# Frontend

- doc/syntax.md

# Backend

- doc/tca.md

# Requirements

- Writing well modualized code

- For implementation, ONLY add necessary suits of tests
  - DO not write excessive amount of tests
  - Making sure the fundamental work then that is fine

- Writing verification pass
  - (1) For each optimization pass or transformation pass
    - Figure out **invariants** before and after the pass
    - Writing verification pass to verify the IR graph to be cohesive accordingly

  - (2) Print rich information during verification
    - Eease the pain for debugging and troubleshooting

- For any uncertain/ambigious problem, stop implementation and asking me first
  - Provide comprehensive details for the stuff

- IR design
  - SSA style
    - Variable assignment => SSADef
    - Variable reference  => SSAUse
    - SSADef and SSAUse shared an underlying pointer which contains all relationship
      - SSAVar
      - Use-def chain and def-use chain
      - Can support
        - Trace all use and def
        - Change def value thus making all ref change accordingly
        - Delete

  - LHS and RHS
    - RHS can be SSAUse or Value(interface)
      - Value means anything creates a value that can be assigned to an SSADef

  - CFG style
    - CFG -> BB -> Statement
    - Notes we do not have control flow, so CFG just contains one BB
    - Each CFG corresponds to a single kernel/function
  - IR must be carefully designed to support optimization mutation
    - Many pass will change node along the way
  - IR must represent all different IR tier in a single suits
    - Notes we do not setup separate high/mid/low IR
    - lower pass will introduce new node in IR, but still the same setup

- Future Development
  - For parallisation/tile assignment
    - Introduce integer programming or ployhedra framework

  - Adding more in depth rule based optimization/rewrite
    - Recognize tensor core instructions
    - Recognize fast bulk load chance
    - Recognize pattern for copy instruction rewritten
