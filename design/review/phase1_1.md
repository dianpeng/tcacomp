# Review of Phase (1)


# IR section

- In short, IR is not composed correctly in phase #1
- Currently the IR looks more like an AST, instead of a SSA/CFG ir
  - Symbol is NOT resolved
  - All reference MUST be replaced with SSAUse
  - All definition MUST be replaced with SSADef
  - Symbol(string) is only used for debugging purpose
  - NO corresponding definition for dimvar/bounded dimvar
    - See /home/ps/Workspace/cuda/project/tcacomp/doc/optana/induction_varaible.md

- The IR graph is **representation after LOWERING from SYNTAX**
  - The IR graph has a building phase, when integrated from syntax AST
  - The AST will be translated into IR graph
    - During the translation, ie IR building
    - All symbols are resolved into SSA format
      - Since no control flow, no need to place PHI node
      - Also single pass is enoughu
    - The AST node to IR node translation will be done on the fly
    - We should see a fully resolved IR graph
    - The IR is designed for optimization, not buf syntax, we do not need symbol name
      which is useless for analysis of program

  - Because of the SSA Use/Def is resolved during building
    - U implicitly perform dead store elinimation
    - 1) For each statement clusters
      - It can have LHS, ie assignment of a new local variable
      - Or it is a purely side effect operations
        - For mutation of observable memory, ie inplace memory modification
      - If a statement's RHS is side effect
        - Add a side effect node, purely for its side effect in corresponding BB
    - 2) For SSADef that has no side-effect
      - NO one is referenced it
      - NO node reflect side effect in BB
      - It is DCEed automatically
```
map (...) store(%input[..][..], ...); // <- statement has side effect, ie last component
                                      //    is a store and memory is a known GMEM
map(...) output(%abc[..][...]); // <- if %abc is not referenced, it will be DCEed
                                // 1) add to symbol table; 2) no one referenced; 3) DCEed
map(...);  // <- dummy side effect operation, since it does not output to any known one,
           // no one is tracing it and dropped directly
```
    - Notes this DCE is partially done
      - After building is done, go through all SSADef, and **Mark Dead** for
        all NONE REFERENCE SSADef
      - We still need to run a dataflow run to fixpoint to propagate the **death mark**
        and finally wipe out everything

- Refactory
  - 1) Remove all symbol
    - Param is not a **name**, it MUST BE a SSAUse
    - For example, IR modeling must be like following

```

// (1) param is a SSADef points to a InputParam node
//     notes InputParam means a kernel level passed in param
kernel my(param: tensor<...>) -> ... {


  // (2) the lambda also reference its specified value as InputParam. Same as parameter
  //     above. Notes during the final lowering, the lambda will be direclty inlined.
  //     The lambda here is not anything like arbitary lambda, it must be a single
  //     expression body. Also we do not really have any control flow operation at all
  lambda (it: f32) ... { ... body ... }

  // (3) local variable, defined by SSADef
  //     This generats a SSADef along with its needed SSAVar, RHS is an Value
  let %local_variable = ...

  // (4) reference of variable
  //
  // The %lhs will be resolved into a **SSAUse** in IR graph
  map (%lhs[...])
}

```

  - 2) Several specialized node is missing
    - loadtile/storetile for cache optimization
    - lowered for node
    - Notes we ONLY use a single IR to construct everything
      - Tiered lowering
    - All nodes, high, middle, and low IR must be represented in a single IR module
      - Better organize them into submdoules to separate the concerns instead of putting
        them into a SINGLE file
    - How to represent

  - 3) The basic block, statement, data structure is inparropriate
    - 1) Statement is appeneded into list of BB(Block), this does not allow O(1) remove
      - Remove requires statement to find its position **from** the array
      - Use **intrusive pointer style linked list**
    - 2) Please in general review ur design
      - Make sure mutation for the GRAPH is doable
