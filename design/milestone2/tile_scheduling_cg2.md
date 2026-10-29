# Lowering from Tile Scheduling

- Tile Scheduling
  - A tile scheduling involes a series of matrix which defines the loop nest trees for tiling
  - Vertically it mapps layer to a well defined series of NVDIA's cuda parallisation concept
  - For any GPU type
    - As long as it can support scope based parallisation
    - It is very easy to map to it

# Overview

- A tile scheduling
  - Breaks down an input polytope I with rank N into M layers of tiling
  - Each tile represents a scope of parallisation
  - It defines a **vertical layout of code structure**

- Operation Mapping
  - For operation, map OR reduce, can be seen as
  - A cooperative job that distributed to each specific tile, ie layers
    - This maps what a real cuda kernel work

  - For example

```

---------------------------------------------------------------------------------
tile scheduling
---------------------------------------------------------------------------------
          M     N     K     <map#1>      <reduce #2>    | <== high level operation
(0/CTA)  [ 4 ][ 4 ][ 1 ]     -                -         |
(1/WARP) [ 2 ][ 2 ][ 2 ]    mem_load    init + combine  |
(2/THR)  [ 4 ][ 4 ][ 2 ]    mem_load    init + combine  |
(3/SEQ)  [ 4 ][ 2 ][ 4 ]      map       init + reduce   |
---------------------------------------------------------------------------------

```

    - The above case shows a high level operation specified by user break down into parallized operation distributed
      - (1) Map
        - ALWAYS need an inner most map operation for element wise map
        - OPTIONALLY, layered mem_load for caching
      - (2) Reduce
        - RSR enforces that the layer that meet the RSR condition, will have a finalize
          - Specialized combine for final combination, using finalize to make it explicit
            - ie **last combine**
        - Intermediate layer with combine to reduce from different layers' SMEM/FRAG data
        - Inner most is reduce, ie element wise reduction, cannot avoid
        - Rules are
          - (0) reducer.init
            - boilerplat code for reducer setup
            - ALWAYS place **before the tile correspoding loop nests**
          - (1) reducer.combine optional
            - ONLY shows up when it needs to combine value from lower to upper
            - In short, when the inner most dimension does not SATISFY RSR
            - Add compensated init + combine in intermediate layers until RSR layer
          - (2) reducer.reduce: MUST
            - MUST show up in inner most layer
      - (3) Memory Caching
        - mechanically inserted as long as feel needed
        - subscript changes according to layer shape

    - Horizontally, it shows how final code generation will work
      - The lowering part works by following
        - 1) Using tile shape to generate each layer's loop nests
        - 2) For each high level operations, ordered from left to right
          - Insert IR nodes broken down by each high level operation based on requirements

    - Map Reduce Fusion
      - Notes fused operation and chained operation are different
        - Fused operation: fuse(op1, op2) => op3, which makes op1 and op2 gone, ie removed from IR, and IR only keep op3
          - treat as redundancy
        - Chained operation: map (...) | reduce (...) | map (...), these are normal operation and will be scheduled and lowered
      - We support multiple map reduce fusion, we split them into following situations
        - (1) map + map fusion
          - If second map to be unary, ie second one DOES NOT contain another input operand, directly fuse 2 lambdas
          - If second map has other input operand, all its map's operand expanded as **sequential loop as is**
        - (2) other fusion is not allowed
          - For correctness, they are not fusion in operator level, but treat as normal chained operation

    - Splitted Scheduling
      - For a chained multiple reductions, the dimension will keep changing
        - ie a map + reduce + reduce
          - Assume input has 3 dimension
          - The second reduce will work on 2 dimension input shape
      - The above lowering strategy works out naturally still
        - The observation is only on tile that mets RSR, the reduction result is settled
        - Which means the chained map OR reduce, will have to start consume data there
          - Which means it needs split the tiling at that point
          - Conceptually, it forms a loop nest tree with branches
            - 2 sub loop nest trees nested in a common parent

```

---------------------------------------------------------------------------------
tile scheduling
input: [128][64][16]
---------------------------------------------------------------------------------
          M     N     K     <map#1>      <reduce #2>    SPLIT      <reduce #3>
(0/CTA)  [ 4 ][ 4 ][ 1 ]       -              -           -       init + combine
(1/WARP) [ 2 ][ 2 ][ 2 ]    mem_load    init + combine  [2][2]    init + combine
(2/THR)  [ 4 ][ 4 ][ 2 ]       -        init + combine  [4][8]    init + combine
(3/SEQ)  [ 4 ][ 2 ][ 4 ]      map       init + reduce   [4][1]    init + reduce
---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
tile scheduling
input: [128][64][16]
---------------------------------------------------------------------------------
          M     N     K     <map#1>      <reduce #2>    SPLIT     <map #3>
(0/CTA)  [ 4 ][ 4 ][ 1 ]       -              -           -          -
(1/WARP) [ 2 ][ 2 ][ 2 ]    mem_load    init + combine  [2][2]       -
(2/THR)  [ 4 ][ 4 ][ 2 ]       -        init + combine  [4][8]       -
(3/SEQ)  [ 4 ][ 2 ][ 4 ]      map       init + reduce   [4][1]      map
---------------------------------------------------------------------------------

```
      - Notes the SPLIT operation happens at 1'th tiled layer
        - Because RSR of previous reduce #2 settle thered
        - Which means the second REDUCE (reduce #3) must split a new groups layer at that point
          - Since that WARP stores **partial settled result**
          - Conceptually we have 2 child parallel loop nest trees

```

for {
  for {
    for {
      // <- (0/CTA) loop nests

      smem reduce_result[16][8]; // notes here stores a setteled reduce result

      // (1) (1/2/3) tiled loop for operation <map#1> and <reduce#2>
      for {
        ...
      }

      // (2) SPLITTED new groups of tile forked here, this is for <reduce#3>
      //     notes these new loop nest trees corresponds to the tile shape produced by SPLIT
      for {
        ...
      }

      // common shared parental loop nests
    }
  }
}
```
