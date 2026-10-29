> **SUPERSEDED** — 2026-10-28
>
> Superseded by doc/tile_scheduling.md (canonical tile scheduling design doc).
> The formal content of this file corresponds to its §2 (Definitions), §3 (The
> Tiling Operation), §4 (Multi-Layer Factorization), §5 (Scheduling: SoC
> Assignment and LoC), and §6 (Mechanical Loop-Nest Code Generation).
> Retained for history. Do not modify — all changes should go to the canonical
> document.

---

# Tile Scheduling Formulation

- A formalized process to perform **opinioned tile scheduling** for cuda program
  - 1) Based on layout algebra
  - 2) Automatic parallisation
  - 3) Mechanically translated code generation
  - 4) Automatic memory cache

# Background

- Cuda system heirachy view
  - Execution unit heirachy
    - (1) Thread Grid
    - (2) CTA
    - (3) Warp Group (selective hardware)
    - (4) Warp
    - (5) Thread
    - (6) Sequential, ie within in a thread
  - Memory unit heirachy (excluding opaque L1/L2 cache)
    - Fragment/register -> FRAG
    - Shared Memory     -> SMEM
    - Tensor Memory     -> TMEM
    - Global Memory     -> GMEM

- Layout Structure
  - We follow cute's layout algebra to define the term
  - 0) 2 Types of algebra structure is used
    - Tuple(X) flatten ordered element lists subset of X
    - HTuple(X) a recursive verison of Tuple(X)
      - Allow **denormalized** elements, ie tuple of tuples
    - 2 operators are defined
      - ~ => conguruence, ie htuple structure matches
      - weak ~ => subset of htuple structure matches
    - For our discussion, we ONLY use Tuple, instead of HTuple directly
      - HTuple are flattened

  - 1) A shape is a HTuple(X), where X is typically Z
    - Defines compatibility between 2 shapes
      - In short, their HTuple format is conguruent and also norm of each HTuples are equal
  - 2) A coordinates set is defined a HTuple(Z) that is related to a shape S
    - Natural coordinate set is coordinates that has HTuple(Z) congurence against S
      - ie of exactly same shape of S structure defined
      - (0, 1) => (M, N)
  - 3) A stride is a HTuple(Z), that performs **mapping** that turns a coorindate compatible to a shape S into a scalar
    - The scalar is interpreted as **offset**

- Tile Extension
  - 1) We use Shape to define a **bounded iteration space**, ie a polytope
    - A Shape S defines a bounded iteration spaces
    - Rank(S): # of iteration axis, ie # of induction variable
    - Like layout algebra's shape, it has its own coordinate sets, ie the iteration space
  - 2) natural coordinates set defines all points in polytope S
  - 3) Tile is an operation to perform **paritioni of polytope against all axis**
    - [1] tile(S, I) = O
      - S(input shape): ready to be tiled shape S, ie the polytope
      - I(tiling shape): tiling size input, which is HTuple(Z) congurent against S
      - O(subspace shape): tiled subspace, which is HTuple(Z) congurent against S
        - O = {O[i]: O[i] = S[i] / I[i]; 0 <= i < Rank(S)}
      - For example:
```
S = [100][200][50]

tile(
  S,                  // <- input shape
  I: [2][2][10]       // <- tile_shape
) => [50][100][5]     // <- subspace shape, ie yet another polytope

```

    - [2] Coordinates and Offset
      - naturally, the coordinates set O is called subspace coordinates
      - there will be a corresponding coorindates set from bijective mapping between S and O for each subspace coordinates
        - The coordinates corresponding to a subspace coordinate is denoted as parental space coordinate

      - an I defines PI(I[i]) # of subspaces
        - each subspace has a index with shape HTuple, congurent against S
        - For example, first subspace is [0][0][0]
      - each subspace O has its corresponding offset can be used to transform from or to parental space cordinate
      - Offset w.r.t Tile
        - Given a Tile convert S with I
        - It also yields an Offset HTuple per subspace
          - For a subpsace with index {index[i]; 0 <= i < Rank(s)}
          - Its corresponding Offset will be {Offset[i]: index[i] * O[i]}
      - To translate coordinates in subspace to parental space
        - {parental[i]: Offset[i] + subspace_coordinate[i]}
      - To translate a parental coordinates to subspace
        - 1) If u know the parental coordinates MUST BELONG to a subspace coordinates
          - {subspace[i]: parental_coordinate[i] - Offset.j[i]; j means specific subspaces' Offset}
        - 2) Otherwise, use modular algorithm
          - subspace coordinate: {subspace[i]: parental_coordinate[i] % O[i]}
          - subspace index: {index[i]: floor(parental_coordinate[i] / O[i])}
          - Notes: subspace index + subspace coordinate decides a specific subspaces' coordinates

# Tile Scheduling

## Overview

- The tile scheduling is used to define a parallisation shape on 2 levels
  - 1) An m repeated steps of tiling operation end up with m different shapes
    - The tiling happened recursively as following
    - For the k'th layer tiling, its shape after tiling becomes the input tile_shape
  - 2) For each tiled layer, 0 ... m-1, each one will be assigned with a specific cuda execution unit
  - 3) Within each tiled layer, it has Rank(S) # of dimension
    - A subset of Rank(S) dimension, for each will be assigned with a level of concurrency
      - ie # of conurrency allowed

## Scheduling Process

- Vocabulary
  - Space of Currency, denoted as SoC
    - 1) CTA, ie mapping tile into a specific CTA
    - 2) WARP GROUP, optional
    - 3) WARP, ie mapping sub-tile in CTA's tile into WARP
    - 4) THREAD, ie mapping sub-tile in WARP's tile into THREAD
    - 5) SEQ, ie sequential scheduling

  - Level of Concurrency, denoted as LoC
    - Defines # of allowed parallism for each SoC
    - For example:
      - LoC(CTA),    # of totall CTAs
      - LoC(WARP),   # of totally WARPs per CTA
      - LoC(THREAD), # of totally threads per WARP
      - LoC(SEQ),    1 always
  - Maximum Level of Concurrency, dentoed as MaxLoC
    - Defines allowed upper bound of scheduling
    - For example
      - MaxLoC(CTA), 2^31 - 1
      - MaxLoc(WARP GROUP), 32 * 7
      - MaxLoC(WARP), 32 warps
      - MaxLoC(Thread), 32 threads per warp

- For a iteration space polytope
  - Denoted as input shape S
  - Tiling it M times, ends up with TS sets
    - ||TS|| = M
    - For element belongs to TS, it is a congurent HTuple against S
    - It represents a specific tiling shape
    - `For 0 <= i < Rank(S): PI(TS[m][i]; 0 <= m < M) == S[i]`
    - Naturally
      - 1) for TS we have each TS's subspace's Offset set, denoted as TSOffset
        - Offset is an other set must be indexed against subspace index
        - TSOffset[i][j][k]...
      - 2) for TS we also have TS's stride set, denoted as TSStride
      - Notes: 
        - TSOffset[m] means return m'th tile's Offset set
        - TSSTride[m] means return m'th tile's stride, ie its **returned O's shape**

  - Parallisation Assignment
    - Form a SoC priority list includes CTA/WARP GROUP/WARP/THREAD
      - WARP GROUP can be excluded, if not supported
      - PRIORITY high to low: CTA -> WARP GROUP -> WARP -> THREAD
    - From m0 to mM-1 different tiles
      - Notes
        - Denote all m tile's corresponding assigned SoC set as TileSoC
          - TileSoC[m] returns m'th tile's SoC assignment

      - Pop first element and assign SoC to it
      - For each dimension of the tile, perform LoC assignment
        - Choose a subset of tile's dimension, for each dimension assign a LoC
          - Notes, for SEQ SoC, its LoC is always 1
        - Denotes subset as P, and its corresponding assigned LoC to be PLoC[i], 0 <= i < ||P|| <= Rank(S)
        - For LoC assignment, we have following constraints:
          - (0) quantity constraints: 0 < PI(PLoC[i]) <= TS[m-1][i]
            - ie tiling shape's extent to each dimension must be less than input shape for that specific m'th tile
          - (1) max constraints: PI(PLoC[i]) <= MaxLoC(TileSoC[m])

          - (2) alignment constraints:
            - This is used on demand, for example if the assignment requires multipler of 32 
              - ie per warp, instead slicing the warp into smaller one
            - We need to make sure that PLoC[i] % 32 == 0 as constraints
          - (3) taboo constraints:
            - It is used to represent that certain dimension bailout from given SoC
            - For example, if k'th dimension bailouts, the LoC of k'th dimension must be 1

  - An scheduling result interpretation
    - Assume we schedule a polytope, rank = 3: `[128][128][128]` into 4 layer tiles
    - The scheduling will make us to have
      - 0) 4 elements vector to represent each tile's SoC
      - 1) 4 * 3 LoC   matrix
      - 2) 4 * 3 shape matrix
        - for m layer tile, the corresponding row m in shape is derived by
          - tile(shape[m-1], LoC[m])

```

// (1) SoC Vector
[CTA][WARP][THR][SEQ]

// (2) LoC matrix
[4][4][2]    // SoC(CTA), max constraints: 4 * 4 * 2 = 32 < MaxLoC
[2][2][4]    // WARP, max constraints: 2 * 2 * 4 = 16 < 32 (32 warps most for CTA)
[16][2][1]   // THR,  max constraints: 16 * 2    = 32 (each WARP has 32 threads)
[1][1][1]    // SEQ, all are 1s

// (3) tile shape matrix
//
// Every column in the matrix will be multiplied to input shape S's corresponding dimension

[4][4][2]
[2][2][4]
[16][2][1]
[1][8][16]  // <- notes, the difference, the last one is different, since it does not perform tile

// (4) stride matrix

[128][128][128]
-----------------------
[32][32][64] // <- CTA
[16][16][16] // <- WARP
[1][8][16]   // <- PER WARP/thread
[1][1][1]

```

  - Code Generation
    - The tile shape matrix contains all we need to generate loop nest trees
      - 1) Each row of tile shape matrix corresponding to a **loop nests**
        - Denoted as a loop nest cluster, ie LNC
        - Each LNC has Rank(S) # of nested loops
        - Each loop's bounds are shape's correpsonding extent to the dimension
        - Tripcount is 1, normalized; init is 0
      - 2) The SoC of the row decides
        - None sequential SoC represents parallisation
        - Loop specialization, ie for CTA/WARP/THREAD level, strides are different
        - Offset calculation is different
