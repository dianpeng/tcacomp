# Final Lower: The last pass before code generation

- This pass happened **after the scheduling**
  - Code is in our AST style BB/CFG setup

- Naming convention: the runtime-object surface is named **Runtime** — the old
  "Intrinsic" vocabulary (KindIntrinsicGridMap/ThreadMap) is renamed to it.
  Runtime objects are OPAQUE to the user and to the IR: the lowered runtime
  nodes are a **mechanical literal representation** with ZERO assumption about
  the underlying codegen — the backend CG chooses the best target construct

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
    - 3) The lambdaNode itself is **UNTYPED** — KindLambda deliberately
      carries no type (the typeinfer pass's existing decision)
      - The types that matter are the per-**parameter definition** type inputs
      - The reducer's element type T is checked against them at Reduce
        lowering (doc/tca.md §4.2 homogeneous rule), NOT derived from the
        lambda node
  - The lambda is **STATELESS — no upvalue capture**: the body references
    ONLY its own parameter definitions and constants
    - This is what makes hoisting it to a standalone function trivially
      sound: MLIR -> private func.func at module scope + symbol-ref operand;
      CUDA C++ -> file-scope `__device__ static inline` function, passed as
      the runtime template's Lambda parameter
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

template <typename T, typename Lambda>
class Reducer {
 public:
  // (0) CONSTRUCTION carries ALL the state: the user's init value and the
  // combiner lambda are passed in HERE — never to Init
  Reducer(T init, Lambda lambda);

  // (1) Called before the reduction bracket — argument-FREE, re-initializes
  // from the construction-time init_/lambda_ (matches doc/tca.md §5.10)
  void Init();

  // (2) Called **for each step within the reduction loop**
  void Reduce(T iteration_value);

  // (3) Called **after the loops are done**, used mostly for GMEM synchronization
  T Combine();

 public:
  // Other helpers, if useful — RUNTIME-INTERNAL ONLY: the lowering NEVER
  // emits these as IR attribute reads (the combiner application happens
  // inside the runtime; see Reducer Lower Integration 2.3)
  // (1) Return user provided initialized value
  T init_value() const;

  // (2) Return currently reduced value
  T current_value() const;

 private:
  // Conceptually member fields — current_ is the runtime's OWN state (the
  // "prev" of the combiner application); the compiler never reads or writes
  // it through the IR
  const T init_;
  Lambda lambda_;
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
    - Zero-between-dims degenerate (the reduced dim is the LNT's only dim,
      2026-10-24 tile-disable reframing): home (the tile-inside loop) is a
      DIRECT child of X, so X's parent is trivially the innermost
      output-element scope — the flat bracket applies (Init immediately
      before X, Combine immediately after, both in X's parent). There is
      NO untiled path: every dim is tiled (T >= 2; a TileSize below 2 is
      rejected at Params validation, record verification, and the
      tile-lower rewrite backstop).
    - Non-reduced loops interleaved between X and the Reduce home (eg the @m_i
      walk) do NOT extend the reduction space — they stay INSIDE the bracket

  - 2) Reduce require insert code at 4 different points
    - 2.1) Create the object at the first Group of the owning Loop BB Cluster
      - Instantiate the specialized Reducer CONSTRUCTION node, provide a SSAName
        - operands: (%init_value, %combiner_lambda) — the init value unwrapped
          from the Reduce's KeyedAttribute("init"), the combiner lambda as a
          SSAUse to the Reduce's trailing lambda node
        - ie RuntimeWarpReducer(%init, %lambda): construction carries ALL the
          state, the runtime owns the combiner from here on
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
      - call reducer.Reduce(...) with the Reduce node's **Input #0** — the
        fused/piped SSA value straight from the producing Map
      - The (prev, cur) combiner invocation is **HIDDEN from the compiler**:
        the lambda was handed over at construction, and it is the runtime
        Reducer's own implementation that applies it — prev is the runtime's
        internal member field (current_), cur is the value passed into Reduce
      - ie <ReducePrevious> inside the combiner lambda body IS the lambda's
        first parameter, satisfied by the runtime at application time; the
        compiler lowers NOTHING for it
      - Notes, when non-reduced loops run between X and the Reduce, the reducer
        state is a TILE of accumulators indexed by those loop vars, and Reduce
        carries the slot index, ie Reduce(@m_i, %piped_value)
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
  Between ir.Node // the tile-between forloop
  Inside  ir.Node // the tile-inside forloop (always populated — every dim
                  // is tiled, T >= 2; nil is record corruption)
  Soc     int     // the effective SoC (mirrors ASTLoopMeta)
  Tile    int     // the effective tile size (>= 2 always)
}
```

- Fail closed: a Reduce whose @r has NO provenance entry is a hard diagnostic
  (the dim was never part of a rewritten LNT)
- Notes, tile-lower already holds every field of this entry in its dimRewrite
  struct at rewrite time (internal/opt/tiling/lower_rewrite.go), so
  publication is pure plumbing on the existing record

- Runtime Object Nodes
  - 1) Every runtime object gets its own DEDICATED construction node (no
    generic RuntimeConstruct) — arity and operand types are verifier-pinnable:
    - RuntimeKernelInfo()
    - RuntimeGridMapping(%kernel_info, <dims + shapes ...>)
    - RuntimeThreadMapping(%kernel_info, <dims + shapes ...>)
    - RuntimeSequentialReducer(%init_value, %lambda)
    - RuntimeWarpReducer(%init_value, %lambda)
    - RuntimeGlobalReducer(%init_value, %lambda)
  - 2) Exactly TWO generic nodes, both **mechanical literal representation** —
    the translator assumes the method/attribute exists; the backend CG picks
    the best target-language construct (an attribute read may become a field
    access, an accessor call, or be folded away — the IR does NOT care):
    - RuntimeCallMethod: a METHOD call on a runtime object (stateful, ordered)
      - input #0: the runtime object node
      - input #1..K: the method arguments
      - string PAYLOAD (never an operand — the intrinsicNode.field precedent,
        internal/ir/node_intrinsic.go): the method name, CLOSED vocabulary
        {Init, Reduce, Combine}
      - EFFECT: ordered side effect — never pure, never GVN-CSE'd
    - RuntimeGetAttribute: an ATTRIBUTE read on a runtime object
      - input #0: the runtime object node
      - input #1..K: attribute arguments (eg the map dimension for
        index/dimension reads)
      - string PAYLOAD: the attribute name, CLOSED vocabulary
        {total_cta, total_thread_per_cta, index, dimension}
      - EFFECT: pure read on immutable objects (KernelInfo, mappings) —
        GVN-sound, same effect model as today's launch-map queries
      - Notes, current_value/init_value are runtime-side C++ helpers ONLY —
        the lowering NEVER emits them (see Reducer Lower Integration 2.3)
  - 3) Kind registration follows the six-companion checklist (the
    node_intrinsic.go precedent): node struct + catalog row, allocNode case,
    verify shape case, typeinfer rules, legality-window note
    (final-lower-onward, LIR tier), contentHash/contentEqual
    - Family assignment (internal/ir/kind.go):
      - FamilyRuntime (NEW): the KernelInfo/GridMapping/ThreadMapping
        constructions + RuntimeCallMethod + RuntimeGetAttribute
      - FamilyCollective (its first members — the reserved slot): the three
        RuntimeXXXReducer constructions
  - 4) A Reduce IR node's operand layout (source of truth: NewReduce + the
    D1 As-before-lambda contract, internal/ir/node_lambda.go):
    - Input #0: SSAUse -> the piped/fused value (the value to be reduced)
    - Input #1: KeyedAttribute("init", ...)
    - Input #2: optional trailing As (rides BEFORE the lambda)
    - LAST: SSAUse -> the combiner lambda node
  - 5) Lowered shape example (simple add reduction, warp-specialized):

```

// construction carries ALL state: init value + combiner lambda
%reducer = RuntimeWarpReducer(%init, %lambda);

// before loop X — argument-free
RuntimeCallMethod(%reducer, "Init");

// inside the tile-inside loop, per iteration — argument is the Reduce node's
// Input #0, the fused/piped value
RuntimeCallMethod(%reducer, "Reduce", %piped_value);

// the lambda(prev, cur) application happens INSIDE the runtime — prev is its
// member field (current_), cur is %piped_value; the compiler emits NO
// lambda-call node at all

// right after loop X
%reduced_final_value =
  RuntimeCallMethod(%reducer, "Combine"); // the SSAUse -> Reduce node is replaced
                                          // with this value for ALL call sites,
                                          // regardless of SEQ/WARP/GLOBAL

```

## Map Lower

- Directly eliminate Map with
  - Remove Map's IRNODE with **Body Expression's** SSAUse
  - Map is an idenetity IRNode that just return value of evaluation of its Body Expression
  - Notes MAP must be inlined before — inline strips the lambda PARAMETERS,
    but the lambda node carrying the body SURVIVES inline (simplify/inline.go)
- Debris disposal, per eliminated Map, in order:
  - 1) RAUW: replace the Map node with its Body Expression's SSAUse
  - 2) EXPLICITLY erase the trailing As operand (positional metadata, no side
    effects — never left dangling)
  - 3) the now-orphaned lambda node is left to the standard mark-sweep DCE
    (an unreachable value node after the RAUW gets swept)
- Ordering: Map Lower runs AFTER Reducer Lower — step 0 derives @r from the
  producing Map's As, so every Reduce must be lowered before its producing
  Map is eliminated

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
%kernel_info = RuntimeKernelInfo();

// Attribute reads — RuntimeGetAttribute, NOT method calls (no codegen
// convention leaks into the IR)
%total_number_of_cta = RuntimeGetAttribute(%kernel_info, "total_cta");
%total_number_of_thread_per_cta = RuntimeGetAttribute(%kernel_info, "total_thread_per_cta");


```

- Notes, the HOST-side launch configuration (gridDim/blockDim) is NOT an IR
  node: under the fix4 factorization model (doc/fix/fix4.md §5.1, T5) it
  DERIVES from the emitted loops' soc/dosoc/loc annotations — gridDim = max
  over LNTs of the CTA LoC products, blockDim = (max warp-K product) × 32 —
  and is consumed by the host launch wrapper + the runtime's launch-validation
  object (doc/runtime_objects.md §1). The IR carries it only implicitly via
  the mapping objects' absolute shape constants and the loop annotations

### Per LNT's Mapping Object Construction

- For each LNT, it will be dynamically mapped to launched kernel's thread
  - It needs 2 special runtime objects for mapping
  - **PREPEND to the owning LNT's prologue Group** (the first Group of that
    LNT's Loop BB Cluster — NOT the kernel's first Group; one pair per LNT)

```

// Conceptual code — the fix4 factorization model (doc/fix/fix4.md §3.2/
// §4.1): the constructor carries an EXPLICIT ndim followed by the per-dim
// ABSOLUTE shapes — LoC_d (CTA counts) for GridMapping, K_d (warp counts)
// for ThreadMapping. No portions, no -1 remainder, no share arithmetic:
// the launch IS what you factorize. Multi-dim schedules nest one slot per
// parallel dim (GridMap = CTA dims only, ThreadMap = WARP dims only — the
// CTA dim's coupled inside walk rides the no-arg flat rank and takes NO
// thread slot). A zero-WARP LNT constructs the ZERO-EXTENT ThreadMapping
// (ndim 0 — one launched warp; the flat query/lane_id/is_first_in_cta/
// ThrIterator still read the object).
__global__ void CompiledKernel(...) {
  KernelInfo kernel_info; // always inserted

  // ONE LNT: 2 CTA dims (LoC 4 and 16 -> 64 CTAs) + 2 WARP dims
  // (K 2 and 8 -> 16 warps = 512 threads)
  {
    GridMapping grid_mapping(kernel_info, 2u, 4u, 16u);

    ThreadMapping thread_mapping(kernel_info, 2u, 2u, 8u);
  }
}

// IR code, insert into the **owning LNT's prologue Group**
//
// NOTES dedicated construction nodes, NOT a generic RuntimeConstruct

%grid_mapping = RuntimeGridMapping(
  %kernel_info, // SSAUse to kernel info
  2,            // # of dimensions (explicit)
  4, 16         // the CTA dims' ABSOLUTE LoC (CTA counts)
);

%thread_mapping = RuntimeThreadMapping(
  %kernel_info, // SSAUse to kernel info
  2,            // # of dimensions (explicit; 0 for a zero-WARP LNT)
  2, 8          // the WARP dims' ABSOLUTE warp counts K_d
);

// The launcher derives the launch from the loop annotations and bakes
// literals (max over LNTs of the CTA-LoC products / (warp-K products) x 32):

tca::rt::ValidateLaunch(dim3(64u, 1u, 1u), dim3(512u, 1u, 1u), smemBytes, maxCTA);
CompiledKernel<<<dim3(64u, 1u, 1u), dim3(512u, 1u, 1u), smemBytes, stream>>>(...);

```

- Launch-map query REWRITE (the query -> object binding):
  - the pre-final-lower query nodes (the old KindIntrinsicGridMap/ThreadMap,
    renamed to the Runtime vocabulary) are rewritten into ATTRIBUTE reads on
    the enclosing LNT's mapping object
    - resolution: the query's SN -> block -> owning nest (each LNT has
      exactly ONE GridMapping and ONE ThreadMapping)
    - ie the query ("index", @d) becomes
      RuntimeGetAttribute(%grid_mapping, "index", @d)
    - the ZERO-OPERAND forms rewrite to ZERO-ARG attribute reads: the flat
      ("index") ruling of 2026-10-23 and the lane query ("lane_id", fix4
      §3.1's lane-chunk walks) become .index() / .lane_id() on the
      ThreadMapping
  - this makes the binding EXPLICIT in SSA for both backends — MLIR has no
    lexical runtime scope to recover it from
  - after the rewrite the standalone query kinds vanish from the graph (they
    are pre-final-lower material only)
