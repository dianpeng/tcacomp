# Scheduling: Bring Sea-of-Nodes back to CFG

- This a combined optimization pass
  - Bring sea-of-nodes back into CFG + BB
  - Perform **LICM** on the fly

# Basic Block(BB) Overview

- High level BB that forms control flow structural
  - Not conventional Basic Block method
    - Each BB has no JUMPs inside; last instruction JUMP to another BB

- Our BB is more like a **syntax tree** to ease the final code generation
  - Our target code is not **low level instructions**, but high level target code
    - [1] CUDA C++ code
    - [2] LLVM MLIR (this is still high level code, since it allows structural syntax as well)
  - We do not need the strict typical BB/CFG in compiler backend

- AST style CFG
  - Overview
    - Our **CFG** is not really a normal BB, since it is defined **recursively**
      - We don't need the flatten BB style CFG, and that will be a hinder for us to generate code
  - Statements:
    - Statement is recursively defined, it can be a single real statement OR another Block

  - Kernel
    - Representing a single kernel
      - Modifier
      - Signature
      - Parameter
        - Notes, for any MemNew (GMEM temporary), it will have a corresponding external provided GMEM buffer
          - For GMEM, it is allocated by the kernel launcher
          - After compilation, we will know all GMEM allocation sites after compilation
            - Each ones type specifically
      - Body
        - Which is a **Group**

  - Block
    - Group
      - An arbitary groups of statements recursively
      - Mostly inserted to include runtime object initialization or LICM landing
        - All LNT will have a header block, Group, which can be used to schedule loop invariants
```
// Conceptually maps to

{
  // body of statements for the Group
}
```
    - Loop
      - In general
        - The loop structure contains 2 sections
          - 1) Header parts to include needed information for final loop materialization
            - induction variable init; upper bound; tripcount
            - extra information for CTALoop/ThrLoop to ensure it can test out of bound
          - 2) Loop Body
            - List of statements that can be translated into target instructions OR can be futhur lowered
      - (1) CTALoop
      - (2) ThrLoop
      - (3) SeqLoop

```
// conceptually, each Loop BB maps to following code

for (...) { // (1) section #1: header part of the BB
  ... // (2) body embodiment of statements ...
}
```

# Scheduling Overview

- 0) Perform loop nest forest analysis
- 1) For each loop nest tree, LNT, we perform an scheduling
  - Which resulted in a list of BBs
- 2) Finally, we merge all BBs together
  - Optionally, adding the GLUE codes for each BB
    - For example: runtime intrinsic object initialization

# Scheduling/LICM of a Single LNT

- For each LNT
  - (1) Recursively create a Loop BB corresponding to all nested iteration loops
    - Notes, mapping specific LIR loop node to corresponding Loop BB
    - The result generated here is called Loop BB Nest, ie LBBN
  - (2) For each LNT, ONE lexical Group block EMBEDDING the LBBN inside it
    - AMENDED 2026-10-23 (quick-261023-lex, author ruling): the original
      prepend/append sibling pair [prologue Group, LBBN, epilogue Group]
      is SUPERSEDED — a sibling prologue Group closes its lexical scope
      before the nest opens, stranding every declaration codegen places
      there (runtime objects, hoisted invariants) out of scope at their
      use sites. The LNT cluster is ONE lexical Group whose items are
      [head items (the old prologue region), the LBBN root as an embedded
      statement, after-nest items (the old epilogue region — no separate
      tail BB; after-nest placement is just an item position after the
      nest)]. All old semantics — SN depth requirements, effect ordering,
      the end-node landing — carry over positionally. Nest()/P-11 binding
      and the epilogue SN rule are unchanged (the Group carries Nest())
  - (3) Scheduling Numbering
    - (3.1) Scheduling Numbering Initialization
      - Give each IR node in sea-of-nodes GRAPH an attribute, scheduling number, denoted as SN
      - Visit the graph for each node
        - If it is an iteration node, set SN to be its **depth inside of loop nest tree**
          - outer most loop is 1, second nested loop is 2, so on so forth
        - Otherwise
          - set it to be 0 as initialization value
          - 0 means the **very first prepended Group BB before the loop BB cluster**
    - (3.2) Scheduling Numbering Propagation, a run to fixed point
      - 1) For each node, the meet operator is defined as
        - **maximum SN of all its Input and ImpactBy operands**
      - 2) Run until no node's SN has changed
        - Ie saturated for all nodes
      - Notes: an optimal method could be a BFS from all Iteration node
    - For each node's SN, it means where it has a direct mapping for BB that IR can map to
      - Notes within the BLOCK, the partial order is settle via data dependency and effect dependency

  - (4) Per BB Scheduling
    - For each rank of SN (ie max SN means nested most iteration SN)
      - For each IR nodes that fits into that rank, exclude iteration node (control flow node, already scheduled)
      - Finding the corresponding BB
        - 0 means Group before loop nest BB
        - 1 means first loop node's body
        - ...
      - Start to schedule this list of IR nodes in partial order defined by
        - Toplogical sort of all same rank IR nodes
          - 1) If node A ~data dependent~> on node B, schedule B first and then A
            - Learnt from Input
          - 2) If node A ~effect dependent~> on node B, schedule B first and then A
            - Learnt from ImpactBy

- After the above scheduling
  - 1) All nodes lands on the **most possible none loop nest block**
  - 2) Potentially, an expression that is loop invariants (LI) gets hoisted into BB 0
    - LI is determined by SN directly

# Scheduling the CFG

- 1) Perform loop nest forest analysis
- 2) Foreach LNT, performs Scheduling
- 3) Concatenate all output BB cluster to form the output kernel body
- 4) Based on other attributes of CFG to form the kernel other metadata
  - 4.1) Parameters
  - 4.2) # of GMEM allocation along with each its shapes
    - Will be allocated by kernel launcher
  - 4.3) Other useful metadata that needs to be carried forward
    - [1] For each LNT's runtime mapping information
      - Will be used for runtime object intialization/generation during final lowering
    - [2] Each loop's SoC information
      - Will be used for reducer lowering
        - Reducer will be lowered into different types, SEQ/WARP/GLOBAL

    the CFG) lives in the project's working copy of this document; this

    retract anything stated there

# Basic Block(BB) Overview — Kernel Signature: the Prototype block

- Kernel
  - Signature / Parameter
    - A `MemNew` (GMEM temporary) node and an `InputParam` node are
      **placeholders to known states, NOT statements**
      - An `InputParam` is a positional kernel parameter
      - A `MemNew` will have a corresponding externally provided GMEM
        buffer, allocated by the kernel launcher (see GMEM notes above)
    - Placeholders anchor at SN -1 into a **PrototypeBlock**
      - The kernel-signature block, kind `BlockPrototype`
      - Rendered FIRST in the compiled kernel — before every LNT's
        [prologue, Loop BB cluster, epilogue]
      - Its items are exactly the live `InputParam` + live `MemNew` nodes
      - `KindMemNewSMem` and `KindFragNew` are NOT placeholders
        - They are in-kernel, per-CTA declarations and stay normally
          scheduled at SN 0 (allocation carve-out unchanged)
      - Extent const values stay normally scheduled at SN 0 as well
        - They are values, not placeholders

# Scheduling the CFG — compiled-kernel output shape and the TOTAL SN mapping

- The compiled-kernel output shape is
  - 1) One PrototypeBlock FIRST (the signature scope)
  - 2) Then the concatenated per-LNT LEXICAL Groups (quick-261023-lex):
    each embeds its LBBN — [head items, the LBBN root, after-nest items]
- The SN → lexical-scope mapping is TOTAL
  - SN -1 = signature scope (the Prototype block)
  - SN 0 = kernel scope (the prologue/kernel Group)
  - SN r ≥ 1 = depth-r loop
- Anchoring of placeholders is FIXED during propagation, exactly like loop
  headers
  - A `MemNew` whose extent dep sits at SN 0 stays anchored at -1 — the
    extent dep never lifts it out of the signature scope
  - Because the propagation meet operator is a **maximum**, a -1 dep
    contributes nothing downstream — zero special conditions anywhere in
    the algorithm
