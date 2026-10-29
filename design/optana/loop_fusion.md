# Loop Optimization

- Our loop optimization is relative simple
  - Loop is representing in a extreamly compact format
  - Loop fusion needs finding out all loop nests
  - Loop nests is implicit

- This pass will run after loop been lowred into Iter node

## Loop Fusion via GVN

- We have something better :)

- Iter node is **side effect free node**
  - [1] The Iter node is itself side effect free, though it is a control flow node
    - We have very regular control flow, which allow us to represent control flow dependency via data dependency
  - [2] GVN will need to handle iter node's equality comparison, with nested awareness
    - (1) A Iter node A is equal to Iter node B, iff:
      - 1] The A and B node's BoundedDimVar are equal
        - ie DimVar and Bounds are both equal
      - 2] And All operands of Iter A and Iter B are Equal
        - Notes an Iter node will have input operands ONLY if it is nested inside of another one
        - By chasing the input operand list for deep equality of Iter node, we ensure that a loop nest will be GVNed only when all NODEs are equal

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!** !!!DEPRECATED!! **!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

## Loop Nests/Forrest analysis

- A analysis pass to figure out ALL loop nests in the code

- Loop nests is defined as following
  - Iter node will have its parental loop, if any, as its **second input** node

```

iter(@x) <- iter(@y) <- iter(@z) <- ...

```
  - Notes, loop nest forest is representing as separate tree/node structure other than IR node

  - We can find out ALL loop nest tree, ie the whole loop nest forest, by
    - 1) filtering out all iter node
    - 2) create an empty set with all intial iter node
    - 3) adding spanning tree edge by, iterating all edges in original IR that links iter(x) -> iter(y)
      - if a iter(x) -> iter(y), we add an edge in our empty set for corresponding 2 corresponding node
        - For straitforwardness, edge linked reversely, ie IR: iter(x) -> iter(y) => loop nest forest: iter(y) -> iter(x)
      - Remove x from the set, since it is reachable by y
      - Once we exhaust all edges, the set contains all root for loop nest tree

  - Notes
    - Iter IR node is normalized loop: init = 0, tripcount = 1

## Equality of 2 Iter nodes under loop optimization context

- 2 Iter nodes are equal only if
  - 1) DimVar is equal
  - 2) Max Bound is equal

- DimVar is equal defined by
  - init is equal
  - tripcount is equal
  - NOTES: for now all DimVar is normalized, hence it is always equal
    - But u have to write the code, since dimvar CAN CHANGE this assumption in the future

## Loop Fusion

- Fuses/merge 2 unrelated loop clusters into same loop clusters
- Conceptually like

```
for (auto i = 0; i < M; i++ ) ...body #1 ...;
for (auto i = 0; i < M; i++ ) ...body #2 ...;

-- merge/fused into one -->
for (auto i = 0; i < M; i++ ) ...body #1; body #2 ...;
```

- 2 loop nests, in loop nest forest set, can be merged when
  - When 2 trees's loop prefix are **equal**
    - Prefix equal is defined as
      - For each Iter node of 2 loop nest tree, iter node at corresponding depth are equal

- Merge 2 loops
  - Assume LOOP A and B are merged
    - There common prefix: is Prefix(A) and Prefix(B) and also Prefix(A) == Prefix(B)
  - We replace all Iter node for Prefix(A) with corresponding nodes in Prefix(B)
    - Which effectively making all Prefix(A)'s iter node **dead**
  - After replacement
    - All nodes in Prefix(A), ie Iter nodes, must have ZERO USEs, ready to be deleted
    - Delete those Iter nodes

- Load Store Forward and GEMEM remove
  - During the fusion, we will open opportunity to perform memory level optimization
    - load-store forward + load elimination
  - Example as following

```

// conceptual code

let cluster1 = map ...;            // <- this writes GMEM
let cluster2 = map (cluster1) ...; // <- this reads the GMEM

// the loop fusion will create IR like

MemStore(
  value, // local SSAUse
  gmem,  // SSAUse -> GMemInit
  InductionVariable(iterA),
  InductionVariable(iterB),
)

// the fused loop will have memory load like following right below this
%local = MemLoad(
  gmem,
  InductionVariable(iterA),
  InductionVariable(iterB),
)

// which essentially means
//
// - (1) mem_store(value, A, @m, @n)
// - (2) out = mem_load(A, @m, @n)
// 
// Obvisouly, the second load will result in value stored by first mem_store
// But in our setup, it is a little bit complicated
//
// [1] The load store forward, will just replace second MemLoad with **value**
//     notes, it just remove the MemLoad, since it is side effect free. AGAIN, we ONLY
//     remove MemLoad, not collapse MemStore + MemLoad, since the MemStore is a write
//     to side effect memory node
//
// [2] The first MemLoad is leaft there. And later on if the `gmem` does not have
//     any sane use, it will be gone, by a load elimination.
//
// The above 2 optimizations are all in side of the **load-store optimization**
//
// -------------------------------------------------------------------------------------

```

- Algorithm Soundness Check
  - All dependency for loop is expressed as data dependency, not control flow dependency
  - A node needs a loop X, ie Iter(X) because
    - InductionVariable   -> Iter(X)
  - The final control flow dependency will be determined by **scheduling**
    - Serialize sea-of-nodes into CFG's IR
  - Verification
    - Once we replace Prefix(A) with Prefix(B)
    - All iter nodes in Prefix(A) must have **ZERO USE**
