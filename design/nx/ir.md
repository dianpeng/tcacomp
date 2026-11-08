# NX IR

- A new IR design for NX

# Design

## Node

- Each node in graph is a single operation
  - An operation is recursive, which means it can carry a lambda
  - A lambda is a single root of a sub-graph
  - An operation contains following property
    - 1) Input (Sea-of-nodes style)
      - Data dependency input
      - Side effect dependency input
      - Control flow dependency input
    - 2) A single output node, ie as return
      - Notes, we only allow single return, multiple return value using tuple type
    - 3) A type
    - 4) A SideEffect
      - Representing the side effect of the node
    - 5) Property
      - Representing properties of node itself
        - For example: is it a commutative node
    - 6) Body
      - ie lambda. Notes this field must be used selectively, ie for nodes can take it
    - 7) Name
    - 8) Attribute
      - Unlike TCA v1.0, we have a dedicated attribute field

- Input lists
  - 1) All data dependency parameters are expressed as SSA inside of Input list
    - It means the operations requires to use the given list of parameters/SSA
  - 2) All **explicit control flow** dependencies are expressed as Control flow list
    - It means the node MUST schedule **inside of the node**
    - For now the IR graph ONLY has control flow node to represent loop like structure
      - Parallel node
      - Loop node
  - 3) Side effect dependency
    - [1] MIR memory node
      - All memory mutation is SSAified
        - 1) All memory node are injective in terms of its **index/subscript**
        - 2) All memory node are mutually restricted
          - One memory node never alias the other
        - 3) All memory node are **scoped**
          - ie control flow dependent on a specific parallel/loop(control flow) node
      - Memory dependency is directly modeled as data dependencies
    - [2] LIR memory node
      - Invole manual memory node mutation in loop

```

// (1) Frontend

kernel matmul(
  lhs: tensor<shape=[M, K]>,
  rhs: tensor<shape=[K, N]>,
) -> tesnor<shape=[M, N]> {
  dimvar {
    m,
    k,
    n,
  };

  return map 
           on(lhs[m][k], rhs[k][n])
           apply(l: f32, r: f32) -> f32: l * r as (m, n, k) |

         reduce(init = 0.0f)
           on()
           apply(prev: f32, now: f32) -> f32: prev + now;
}

// (2) HIR

$hir.within(m, k, n) () { // <- polytope space
  %map_value  = $hir.map on(m, k, n): tensor<shape=[m, n, k]>;
  %map_reduce = $hir.reduce(init=0.0, axis=2) (%map_value): tensor<shape=[m, n]>;
}

// (3) MIR
//
// reduce.max(element.add) => mma

$hir.within(m, k, n) () {
  (1) map + reduce
  (2) intrin.matmul
}

// ->

GRID([m0, k0, n0]) OPTIONAL
      ^
      |
      +
CTA ([m1, k1, n1])                | (1) reduce  |              (2) reduce
      ^
      |
      +
WARP([m2, k2, n2])                | (1) reduce  | <-- (2) mma
      ^
      |
      +
THR ([m3, k3, n3]) <-- (1) map    | (1) reduce  |

// (4) LIR

Above scheduling turns into sea-of-nodes and then lower to CFG based scheduling

```

- Type Constraints and Memory/Buffer model
  - Type of tensor is modeled as following
    - (1) Tensor is modeled as type
      - [1] shape = [ints]: [1, 2, 3]
      - [2] flatten: int; weather this tensor can be derive a flatten 1D array
        - 1) 0: true
        - 2) 1: unknown
        - 3) 2: false
      - [3] storage major: row vs column
      - [4] class: storage class
        - For example: GMEM/SMEM/TMEM/FRAG

    - (2) Certain groups of tensor types can be converted automatically via builtin rewrite rules
      - The rewriting rules are mostly just inserting buffer(SMEM) and populate it to corresponding needed one

    - (3) Type is one of the major way to express constarints
      - ie type constraint


  - Memory/Buffer operation
    - (1) Buffer is created with new buffer ir node with specific
      - 1) storage major
      - 2) class
      - 3) shape
      - By default, new buffer returned memory is always flatten = 0
        - ie always flattenable
      - Notes: input parameter is also another source of input buffer
        - GMEM buffer

    - (2) To access a buffer
      - 1) One must define the stride transformation, ie layout
      - 2) A specialized operation View will be needed, ie return a accessor
      - 3) The view object returned always allow a pure mapping

```

// (1) A GMEM access with following pattern

- 1) GMEM of shape: [10][20][30]
  - [30][20][10]
- 2) GMEM access  : [i + 1][j + 2][k + 3] (0 <= i < 10; 0 <= j < 20; 0 <= k < 30)
- 3) The transformed view node is like

%mem_view = MemView(
  %some_memory_view,    // original multi-dimensional GMEM view
  [off1, off2, off3],   // offset
  [w0, w1, w2],         // extents of stride per dimension
  mapping {             // subscript mapping, must be affine/injective
    0: (i) => i + 1,
    1: (j) => j + 2,
    2: (k) => k + 3,
  },
  [0, 1, 2], // if need access permutation, this is where it goes
);

// now memory view can be accessed via natural coordinate of GMEM

// (2) GMEM loading
//     typically we need to load a tile of data from GMEM into SMEM with GMEM coalescense aware
//     method


%xx = reduce.thread([m3, k3], [k3, n3]);
reduce.warp(%xx, memory, WARP)
reduce.cta (memory, CTA)


```

input: [128][128][64]
-----------------------------------------------------------------------
cta:   [2]  [2]  [2]   <- [2][2][32][32][2]
warp:  [2]  [2]  [8]   <- [2][2][32][32][8]
thr:   [32] [32] [4]
