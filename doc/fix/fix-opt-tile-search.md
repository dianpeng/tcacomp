# Improving the loop tiling schedule search algorithm

- Current status
  - Very naive scheduling algorithm
  - The code structure is not a strictly mapping from the algorithm

- Define a formalized and unambigious general purpose searching algorithm
  - Finding out our parallisation scheduling/assignment

# Search Algorithm

- The following design doc describe a general purpose of scheduling assignment

## Input
- A loop nests, ie N nested iterations
  - We will choose a sub-set of the N iterations, for now we choose up to M **inner most consequetive loops**
    - By default, set **M = 4**
      - Notes for now it is 5

## Problem Formulation

- For each iterations, assign an **Schedule Assignment** in allowed scheduling space
  - A Schedule Assignment is
    - (1) SoC: space of concurrency
      - Sequential
      - CTA
      - Thread/WARP
    - (2) LoC: level of concurrency
      - Sequential: = 1 always
      - CTA: **# of CTAs per allocated dimension**
        - Notes: we use to use a percentage which is invalid
        - And also the kernel launcher's gridDim.x parameter must be dynamically derived accordingly
          - Later on described

      - Thread/WARP: **# of WARPs per allocated dimension**
        - Notes: we use to use a percentage which is invalid
        - And also the kernel launcher's gridDim.x parameter must be dynamically derived accordingly
          - Later on described
        - 1 means 1 WARP, ie 32 threads

    - (3) TileSize: corresponding dimension's size of stride responsible for a SoC
      - For example tile_size = 32 for a CTA means:
        - each CTA will be responsible for managing a consequtive 32 elements for that dimension
      - TileSize decides the inner tile-inside-loop's iteration space's upper bound
      - It also implicity defines the SMEM/FRAG cache size
        - SRAM, ie SMEM or FRAG, based cache are always per tile
        - Tile size defines the size of the SMEM size locally

  - A Schedule Assignment is **Structural/Heirachy**
    - Denote:
      - Tile Between Loop as TBL
      - Tile Inside Loop as TIL
    - (1) CTA
      - TBL: CTA level concurrency
        - (1) A consequtive tile_size iteration space is mapped to a specific CTA
          - Each CTA is mapped to cta_index * <tile_size>
          - [0, <tile_size>] [<tile_size>, 2 * <tile_size>] ... etc
      - TIL: Thread level concurrency
        - Structurally, the tile inside loop is partitioned by a partition of thread space regarding that CTA
    - (2) Thr
      - TBL: **Thread Level Concurrency** within the assigned CTA
      - TIL: **Sequential loop**
    - (3) Sequential
      - TBL: Sequential
      - TIL: Sequential

    - In short
      - A CTA assignment, implicitly uses 2 parallisation SoC
        - CTA for loop between
        - THR for loop inside
      - A THREAD assignment, implicitly just use ONE SoC
        - THR for loop between
        - SEQ for loop inside
      - A Sequential assignment, both TBL and TIL are sequential

  - SRAM caching
    - As a side effect, the SRAM caching parameter will be locked down

- Constraints
  - For a specific proposal scheduling
    - A list of iterations with its own **Schedule Assignment**
    - NOTES the list of iteration is **AFTER TILING**
      - ie if input is 3 nests, now it is 6
  - (1) maximum allowed parallisation mapping: MAPM
    - SUM all iteration's # of CTA and THR/WARP
      - Denote total CTA as total_cta
      - Denote total THR as total_thr
    - total_cta <= MaxPartition.cta (maximum allowed cta partitaion)
    - total_thr <= MaxPartition.thr (maximum allowed thread partition)
    - By default
      - MaxPartition.cta = 3
      - MaxParittion.thr = 3
      - Ie similar to cuda's (x/y/z) mapping

  - (2) maximum allowed LoC: MALoC
    - **Mul** all iteration's # total LoC
      - For all iterations that is categorized as CTA assignment SoC: Mul all LoC
        - Denoted as total_cta_loc
      - For all iterations that is categorized as THR assignment SoC: Mul all LoC
        - Denoted as total_thr_loc
    - total_cta_loc <= MaxLoC.cta
    - total_cta_thr <= MaxLoC.thr
    - By default
      - MaxLoC.thr = 32 (ie 32 WARPs maximum allowed per CTA)
      - MaxLoC.cta = 2^31 - 1 (ie maximum allowed value in gridDim.x)

  - (3) maximum allowed SRAM: MASRAM
    - ie # of total bytes allowed for caching
    - For each iteration's CACHE SITE
      - For storage class to be SEQUENTIAL, sum all its byte size
        - denoted as total_frag
      - For storage class to be THR, sum all its byte size
        - denoted as total_smem
    - Ensure total_frag + total_smem <= MaxSRAM
    - By default
      - MaxSRAM = 64kb

  - (4) cuda mapping: CuMAP
    - For all iteration schedule assignment, at least one must be CTA
      - This is easy to understand, since if no iteration uses a CTA assignment, cuda will not work

  - (5) implicitly, the # of loop iterations constriants total # of SoC

  - For kernel launcher configuration
    - (1) We are always using 1 * 1 allocation, ie gridDim.x and blockDim.x are relevant
    - (2) For gridDim.x parameter
      - **maximum of all LNT in kernel's total_cta_loc**
    - (3) For blockDim.x parameter
      - **maximum of all LNT in kernel's total_cta_thr**

- Objective Function
  - **NOTES, we use a new and well defined objective functions**
  - NOTES about concurrency
    - The allocation will uses a portion like LoC, and by MaLoC constraints, all concurrency will be exhausted regardlessly
  - In short, it is composed of 2 parts, with a weighted sum
    - [1] **maximize SMEM/FRAG reuse**
    - [2] **granularity of concurrency**
      - Since we will partition all given threads totally anway
      - We want to optimize the granularity of partition
        - Ie fine grained partition always lead to better utilization
        - We are given irregular unknown tensor as input, fine grained parition will have less chance to waste CPUs
          - Ie thread quick return because shape mismatch/OOB
        - ie threads (1024) vs thread(32) * thread(32)
        - the latter 32 * 32 will have high chance to waste less than a single 1024
          - Consider a 28 * 1024 tensor, the first one will ONLY have 28 concurrency at most, even given 1024
            - The latter will have 28 * 32 concurrency, which is way better than first one
      - Propotional to the SM utilization

  - The objective function can be defined as
    - (1) **maximize SMEM/FRAG reuse**, normalize to [0.0, 1.0]
      - for each group resulted in findINS
        - each element in a certain set will be overlapped with others for M dimension
          - M >= 2 as hard constraints for findINS filter
        - For a SMEM/FRAG site, its reuse bytes are defined by
          - Multiplication of all overlaped dimension's tile_size
          - Denoted as reuse_size
      - reuse_score = **SUMATION of reuse_size of all sites** / MaxSRAM
    - (2) **granularity of concurrency**, ie GoC, normalized [0.0, 1.0]
      - goc_max = max(MaxPartition.cta + MaxParition.thr, 2 * # of iterations)
        - ie coarsely counting allowed total maximum loc, at most 3 + 3 will be allowed
      - goc_score = (total_cta_loc + total_cta_thr) / goc_max
    - (3) objective_function = alpha * reuse_score + (1.0 - alpha) * goc_score
      - by default alpha = 0.5

- Feasible Area
  - Each iteration can be assigned one of the 3 possible SoC
  - Each iteration can have one of the tile_size in a preset list
    - Assuming it contains 32/64/128 (totally 3 different sets)
  - Each iteration can be assigned with specific types of LoC
    - For CTA LoC pre list: (1, 5, 10)
    - For THR LoC pre list: (1, 4, 8)
    - All are 3 elements
    - Notes due to the exhaust rule, the last CTA/THR LoC can be simply assigned with -1
      - use up whatever left in runtime
      - effectively limit the searching space
  - For now we clamp only 5 most nested loops
  - The total searching space is
    - (3 * 3 * 3)  ^ 4 = 531441
    - The searching space is pretty managable
  - Notes, SRAM assignment will use a backtracking algorithm to find feasible one
    - tile_size decides **if the SRAM is populated, what the SRAM costs it will incur**
    - to decide weather a potential SRAM site is populated, we use a simple backtrace algorithm to find
    - notes to find **potential set of SRAM, use existed findINS algorithm**
      - already implemented in overlap.go

## Searching Algorithm

- Overview
  - A exhausted search that optimize the objective function
  - Perform backtrace ONLY WHEN **MASRAM** constraints violation
    - Like Chattin-Briggs RA allocation
    - For all assignment combination, always try to fullfill ALL SMEM requirements in candidative set
    - If success, we are done to evaluation
    - Otherwise
      - Try to pop one SMEM/FRAG cache out of the cache site and recalcuate constraints
      - Until we satisfy the given constraints
  
- Algorithm
  - Permutate one instance in **Feasible Area**
  - Check its constraints
    - Using SRAM Backtrace algorithm to make the candidate satisfy MASRAM constraints rule
  - If constriants NOT satisfied, continue to next one
  - Otherwise evaluation
    - calculate the score based on objective function
    - update the maximum one as candidative assignment
  - Return the optimal one searching by the algorithm


# Implementation

- Current implementation is really messy
  - The algorithm lacks a clear separation of concernces
    - Iterator that exhausts feasiable area
    - Constraints checking
    - Objective function score obtaining
    - Evaluation and updates

- Desired implementation
  - Very well organized code
  - [1] All configuration parameter as pass input with sane default
    - Configurable and may use hyper-parameter optimizer to customize them in the future
  - [2] Implementation by module
    - (1) Iterator for feasiable area management
    - (2) Constraints checking and backtracing for adjustment
      - Notes the backtracing of SRAM will finalize the SMEM part
    - (3) Score evaluation
      - Obtaining the objective function score based on the proposal scheduling
    - (4) Updating if current scheduling is better than the recoreded best one
    - (5) Finally, return the best one we have searched
