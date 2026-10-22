# Code Generation

- This is the final pass that takes finalized IR graph to produce backend code

- For now, we are planning to support 2 backends
  - [1] C++ cuda (this version)
  - [2] MLIR     (next version)

- Code Generation must reside in its own modules
  - internal/cg/cxx   => c++ cuda
  - internal/cg/mlir  => mlir version in the future

# CXX Code Generation

- The provided IR has following shape
  - 1) For IRNodes that has corresponding low level IRNode, already lowered
  - 2) Verb, map/reduce has been lowered into low level runtime primitive call
  - 3) SSAUse is not resolved
    - Which means the code generation part needs to generat local symbol on the fly


- CodeGen method
  - 1) Naturally visit all BB in the IR graph
  - 2) For each BB, based on its type generate the BB itself
  - 3) Recursively code generate all statement inside of it
  - All IRNodes naturally have very direct 1:1 mapping

- Runtime object
  - ALL runtime object stays within its own namespace tca
  - To use a runtime object, generate code that has qualified namespace
    - ie tca::XXX
