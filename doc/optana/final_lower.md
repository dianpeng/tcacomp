# Final Lower: The last pass before code generation

- This pass happened **after the scheduling**
  - Code is in our AST style BB/CFG setup

- Its main goal is to
  - 1) Lower reducer if any
  - 2) Lower map
    - Very trivial, just inline the Body of Map and remove it entirely
    - It is not very useful during code generation
  - 3) Lower runtime object initialization/insertion

## Fix or Ensure Lambda IR Node, ie lambdaNode, to follow design convention

- The lambda represents a **stateless function**, with a single expression as body
  - Its Input operand is
    - 1) LAST one is SSAUse -> Body expression
    - 2) All others in Input list is **Parameter Definition**
      - Notes NOT reference to Parameters, since it cannot because it is a stateless function
      - But it needs to contain information of **How many parameters are defined**
        - Each parameter's **type**
    - 3) The lambda's type is decided by its **Body Expression**'s type
      - ie Type of Input()[-1]
  - Make sure the lambdaNode follows the above requirements

## Reducer Lower

- Fix of existed pipeline
  - [1] Remove inline for **Reduce node's Lambda**
    - Keep the Reduce's body as a separated lambda and we will need to generate it
    - Notes, its body is represented as SSAUse -> <lambdaNode>
  - [2] Notes, MAP's body inline is still kept and useful

- Overview of Reducer
  - 1) Performs reduction against the **last dimension of the producing Map's as order** — NOT the last axis of the input tensor (doc/tca.md §2.7 rule 3: "a downstream reduce reduces the last variable of `as`")
  - 2) BINDS statically to a dimension, which binds to a iteration
  - 3) Iteration can be executed in different space of concurrency
    - Based on parallisation assignment, we can schedule an iteration for CTA/WARP(thread)/SEQUENTIAL

- We have 3 specialized reducer runtime implementation

### Overview Format

- Runtime Reducer Object Interface

```

template <typename T>
class Reducer {
 public:
  // (1) Called before the reduction loop
  template <typename Lambda>
  void Init(
    T init = <user' provided code's init value in KeyedAttribute>,
    Lambda lambda // notes generated lambda instance for reduction
  );

  // (2) Called **for each step within the reduction loop**
  void Reduce(T iteration_value);

  // (3) Called **after the loops are done**, used mostly for GMEM synchronization
  T Combine();

 public:
  // Other helpers, if useful
  // (1) Return user provided initialized value
  T init_value() const;

  // (2) Return currently reduced value
  T current_value() const;

 private:
  // Conceptually member fields
  const T init_;
  T current_;
};

```

- Notes, regardless weather it is implemented as Seq/WARP/GLOBAL
  - The exposed Reducer interface is in above shape
  - The Reducer provides a consistent way for the lowered code to work with
    - Except the **constructor is different**, since specialized instance must be created

### Reducer Lower Integration

- Before Lowering, Reducer will be like
  - The Reduce lambda body is **NOT INLINED**
    - Notes this is on purpose, otherwise we cannot give a consistent surface for different specialized reducer
    - Its body is inside of an lambda
  - Notes, the Map's body is inlined, since Map does not need specialization

```

%previous_value = <... can be Map or Reduce ...>; // <- input for Reduce

Reduce(
  %previous_value,
  KeyedAttribute(name='init', value=<xxx>),
  As(...),
  /** body lambda **/, // this is not INLINED, and is SSAUse -> LambdaNode
)

// Notes the body's lambda expression contains an expression will use special operands
//
// [1] <ReducePrevious> IR node, means previously reduced value passed in
// [2] <...IR node for current...>, to be reduced value

```

- During the lowering
  - 0) Identify the reduced dimension @r — MUST run before Map Lower
    - The Reduce node does NOT carry @r directly, derive it from the dataflow
      - Input #0 (SSAUse -> piped value) resolves to the producing Map
      - The producing Map's trailing As operand (ie `ir.AsOf`) fixes the as order
      - @r = the LAST dim of that As order
    - Fail closed: no Map/As chain found is a hard diagnostic, never guess
    - Notes, this is exactly why Map Lower must run AFTER Reducer Lower

  - 1) Resolve the bracket loop X — the OUTERMOST loop iterating over @r
    - Fetch the per-dim tiling provenance from the preserved "tiling-record"
      (see "Tiling Provenance Record" below)
    - Tiled path (T > 1): @r is split into tile-between + tile-inside loops
      - X = the tile-BETWEEN loop
      - The Reduce node's SN-bound home loop is the tile-INSIDE loop
      - Init/Combine must NOT bracket the tile-inside loop: otherwise every tile
        gets its own Init/Combine pair and only the LAST tile's partial survives
        (per-tile partials). doc/tca.md §5.10 law: Init/Combine are never placed
        inside the tiling loops of @r — one reduction spans ALL tiles
    - Untiled path (T == 1 or dim never tiled): X = the single loop, which is
      also the Reduce's home loop
    - Non-reduced loops interleaved between X and the Reduce home (eg the @m_i
      walk) do NOT extend the reduction space — they stay INSIDE the bracket

  - 2) Reduce require insert code at 4 different points
    - 2.1) Create an object at the first Group of the owning Loop BB Cluster
      - Instantiate a specialized Reducer instance, provide a SSAName
      - Specialization follows X's ASTLoopMeta.Soc (internal/ir/astcfg.go)
        - Soc 0 (seq) -> RuntimeSequentialReducer
        - Soc 2 (thr) -> RuntimeWarpReducer
        - Soc 1 (cta) -> RuntimeGlobalReducer
    - 2.2)
      - insert reducer.Init(...) immediately BEFORE loop X, in X's parental
        block body
      - when X is the cluster-outermost loop, Init lands in the owning LNT's
        prologue Group
    - 2.3)
      - within the Reduce's SN-bound home loop (tile-inside on the tiled path)
      - The node <ReducePrevious> IR node maps to reducer.current_value() call
      - call reducer.Reduce(...) with input of its first operand
      - Notes, when non-reduced loops run between X and the Reduce, the reducer
        state is a TILE of accumulators indexed by those loop vars, and Reduce
        carries the slot index, ie Reduce(@m_i, %current_value)
        (doc/tca.md §5.10 point 3)
    - 2.4)
      - insert reducer.Combine() immediately AFTER loop X, in X's parental
        block body
        - ie one Combine spans ALL tiles of @r, no per-tile partials
      - Consumers of Combine()'s result (the fused map/store tail) are placed
        after Combine, inside a RE-ITERATION of the non-reduced inner loops
        (doc/tca.md §5.10 point 4)

### Tiling Provenance Record (consumed, not produced, by this pass)

- The tile-between/tile-inside relationship is NOT instrumented onto the loop
  IRNode
  - ir deliberately knows NOTHING about opt/tiling (astcfg.go layering rule)
  - Instead the relationship is published as DATA by tile-lower and fetched
    here — same mechanism as the existing "tiling-record" side table
    (published via the analysis cache, survives scheduling through the
    scheduler's Preserved["tiling-record"] whitelist)

- Tile-lower publishes one entry per rewritten dim, keyed by the DIM node
  (a RoleDim InputParam — stable across every later pass, and the identity
  step 0's @r derivation naturally yields):

```
per-dim provenance entry {
  Between ir.Node // the tile-between forloop (== the single loop when T==1)
  Inside  ir.Node // the tile-inside forloop (nil when T==1)
  Soc     int     // the effective SoC (mirrors ASTLoopMeta)
  Tile    int     // the effective tile size (1 = untiled)
}
```

- Fail closed: a Reduce whose @r has NO provenance entry is a hard diagnostic
  (the dim was never part of a rewritten LNT)
- Notes, tile-lower already holds every field of this entry in its dimRewrite
  struct at rewrite time (internal/opt/tiling/lower_rewrite.go), so
  publication is pure plumbing on the existing record

- Runtime Object Node
  - 1) Add a new expression node called RuntimeCallMethod
    - calling a method on a runtime IR node
    - input #0: another Runtime node
    - input #1: name of methiod
    - input #2-K: parameters for the method
  - 2) 3 Runtime reducer specialization node
    - RuntimeSequentialReducer
    - RuntimeWarpReducer
    - RuntimeGlobalReducer
  - 3) A Reduce IR node's operand is as following
    - Input #0: SSAUse -> current loop/iteration's value
    - Input #1: KeyedAttribute
    - Input #2: SSAUse -> <Lambda> node
  - 3) To call a method, the IR node is generated as following

```

%reducer = RuntimeGlobalReducer();

RuntimeCallMethod(%reducer, "Init", <Lambda>) // call reducer.Init()

%previous = ...; // value of map

// assuming reduce is a simple add reduction
%reduce_current = RuntimeCallMethod(%reducer, "current_value");

%current_value = <Input #1 of Reduce IR Node>; // Points to reduction value

// notes the Lambda node must already been materialized somewhere during scheduling
// just use SSAUse for lambda to invoke it, ie rewritten as something like, ExprCall
// to call an lambda

// This node effectively codegen (C++) as
//
// reduce_result = lambda(reduce_current, previous)

// call runtime reducer's Reduce against locally reduced value. Notes since lambda is
// provided to Reducer object, internally it will call the lambda to do the job to
// perform specialized reduction
RuntimeCallMethod(%reducer, "Reduce", %current_value);

%reduced_final_value =
  RuntimeCallMethod(%reducer, "Combine"); // The SSAUse -> Reduce node is replaced with
                                          // Combine which returns the final reduced value
                                          // for all call site regardless thread/CTA/SEQ

```

## Map Lower

- Directly eliminate Map with
  - Remove Map's IRNODE with **Body Expression's** SSAUse
  - Map is an idenetity IRNode that just return value of evaluation of its Body Expression
  - Notes MAP must be inlined before, hence it DOES NOT HAVE Lambda at all

## Runtime Object Lower

### Initial Runtime Object Construction

- 1 runtime object to obtain all **cuda kernel related information**
  - inserted at very first of the kernel
  - can be used to obtain
    - total # of CTA
    - total # of threads per CTA

  - **PREPEND to the starting block**

```

// Conceptual code
__global__ void CompiledKernel(...) {
  KernelInfo kernel_info;
}

// IR code, insert the code into the **very first Group block in kernel**
%global = RuntimeConstruct("KernelInfo");

// To obtain total # of cta
%total_number_of_cta = RuntimeMethodCall(%global, "total_cta");
%total_number_of_thread_per_cta = RuntimeMethodCall(%global, "total_thread_per_cta");


```

### Per LNT's Mapping Object Construction

- For each LNT, it will be dynamically mapped to launched kernel's thread
  - It needs 2 special runtime objects for mapping
  - **PREPEND to the starting block of each Loop BB Cluster**

```

// Conceptual code
__global__ void CompiledKernel(...) {
  KernelInfo kernel_info; // always inserted

  // ONE LNT
  {
    GridMapping<
      3, // # of dimension
      2, // first dimension 20%
      5, // second dimension 50%
      -1 // third takes the rest
    > grid_mapping(kernel_info);

    ThreadMapping<
      2, // # of dimension
      2, // first dimension 2/8 = 25%
      -1 // second takes the rest
    > thread_mapping(kernel_info);
  }
}

// IR code, insert the code into the **very first Group block in kernel**
//
// NOTES we do not use RuntimeConstruct, but use RuntimeGridMapping/ThreadMapping IR NODE

%grid_mapping = RuntimeGridMapping(
  %kernel_info, // SSAUse to kernel info
  3,            // # of dimension
  2,            // 20%
  5,            // 50%
  -1            // REST
);

%thread_mapping = RuntimeThreadMapping(
  %kernel_info, // SSAUse to kernel info
  2,            // # of dimension
  2,            // 20%
  -1            // REST
);

```
