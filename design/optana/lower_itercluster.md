# Lowering the IterCluster to Iters

- The frontend will initially build map/reduce cluster into IR nodes as following
  - (1) The ir suits misses a IterClusterProjection node
    - Used to bridge link between **Map's parameters to IterCluster's input operands**
    - For example:
      - Map -- 0 --> IterClusterProject(0) --> IterCluster
         |                                       ^
         |                                       |
         +---- 1 --> IterClusterProject(1) ------+

      - The above IR graph is the highest version of map
        - IterClusterProject(0/1/...) basically means IterCluster's first operand will be mapped to Map's first param
      - This format is called **IterCluster format**

  - (2) The IterCluster format is not cananoical, will be lowered almost immediately in Simplifaction to
    - Iter(x) -> InductionVariable -> MemLoad -> Map/Reduce
    - For now IR also lacks InductionVariable
      - Takes Iter as input and presumebly return its induction variable
      - Iter represents the whole loop
      - InductionVariable(Iter) represents a specific iteration's value of induction variable
        - Of type u32
        - Participate in normal expression
    - This format is called Iter format


- Lowering from IterCluster format to Iter format

```

// input conceptual syntax
kernel myfoo(lhs: ..., rhs: ...) -> ... {
  dimvar a;
  dimvar b;
  dimvar c;
  let some_value = map(
    lhs[a][b],
    rhs[a][c],
  ) ...;
}

// (1) IterCluster format

// Referenced by Map indirectly via IterClusterProject(0)
nodeIterCluster = IterCluster(
  Input (
    MemLoad(
      lhs, /** SSAUse -> Param **/,
      a,   /** SSAUse -> DimVar **/,
      b,   /** SSAUse- > DimVar **/,
    ),
    MemLoad(
      rhs, /** SSAUse -> Param **/,
      a,   /** SSAUse -> DimVar **/,
      c,   /** SSAUse -> DimVar **/,
    ),
  ),
  EffectList (
    lhs, /** SSAUse -> Param **/,
    rhs, /** SSAUse -> Param **/,
  ),
  ...
);

Map(
  Input(
    lambdaParam1 = IterClusterProject(nodeIterCluster, 0),
    lambdaParam2 = IterClusterProject(nodeIterCluster, 1),
  ),
  ImpactBy(...),

  // Map has an output which points to
  As(
    a, /** SSAUse -> DimVar **/
    b, /** SSAUse -> DimVar **/
    c, /** SSAUse -> DimVar **/
  ),

  Lambda(
    // Lambda body is represented as a real separate function, will be inlined directly
    // in Simplification pass. The expression access inside of lambda body for input
    // parameter will be like
    // SSAUse -> Param (lambda's positional parameter, ie 0/1/2/3)
    //
    // And the lambda's input parameter list is in **Map/Reduce node's** Input operand list
    //
    // Notes: there is a buggy definition here, the Lambda struct MUST NOT contain params[]SSAUse
    // For all ir node, its input/output must be only part of the Input and ImpactBy list
    // these 2 are known to rest the world, any hidden reference will be invalid due to
    // they will not be part of the analysis/optimization pass.
    //
    //
    // The inline will be linke, replace all lambda' body's SSAUse -> Param with the
    // corresponding index/position in Map/Reduce node's input list's corresponding
    // value. The returned value does not change, since to reference a mapped value or
    // reduced value will always be like SSAUse -> Map/Reduce, body is hidden from
    // outside of the world, this is what we want now. Notes, we do not perform
    // constant folding or expression level optimization for now, so current setup is
    // fine.
    //

    ... Body ..., /** each input parameter is also a SSAUse -> Param **/
  )
);

// (2) IterCluster format lowered into Iter format
//     The loop is gone, but totally becomes data dependency IR graph.
//
// The above IR graph will be lowred as

iterA = Iter(
  a /** SSAUse -> DimVar **/,
  TensorDimOf(lhs, 0), /** New node TensorDimOf, which extract tensor's specific dimension' value, ie upper bound **/,
); // type u32


iterB = Iter(
  b /** SSAUse -> DimVar **/,
  TensorDimOf(lhs, 1), /** New node TensorDimOf, which extract tensor's specific dimension' value, ie upper bound **/,
); // type u32

iterC = Iter(
  c /** SSAUse -> DimVar **/,
  TensorDimOf(rhs, 1), /** New node TensorDimOf, which extract tensor's specific dimension' value, ie upper bound **/,
); // type u32

iterAIV = InductionVariable(iterA) // iterA's induction variable, type u32
iterBIV = InductionVariable(iterB) // iterB's induction variable, type u32
iterCIV = InductionVariable(iterC) // iterC's induction variable, type u32

inputLHS = MemLoad(
  lhs, /** SSAUse -> Param **/
  iterAIV, /** 0'th dimension index **/,
  iterBIV, /** 1'th dimension index **/,
) // type is inferenced based on lhs(tensor) shape

inputRHS = MemLoad(
  rhs, /** SSAUse -> Param **/
  iterAIV, /** 0'th dimension index **/,
  iterCIV, /** 1'th dimension index **/,
)

// Map will be lowered as following
map = Map(
  Input(
    inputLHS,
    outputRHS,
  ),
  ImpactBy(...),

  // not changed from previous Map
  As(
    a, /** SSAUse -> DimVar **/
    b, /** SSAUse -> DimVar **/
    c, /** SSAUse -> DimVar **/
  ),
  Lambda(
    // body will be lowered:
    // old reference of lambda parameter is via SSAUse -> Parama (lambda, 0) -> ...
    // now becomes inputLHS
    // Notes, the map's input operand list changes, the positional parameter for inline
    // must also be changed. We separate these as 2 sub-passes in Simplification
    //
    // 1) does the high level rewrite
    // 2) inline lambda
    ...,
  ),
)

```

- Reduce
  - Reduce has extra initialize value, ie as attribute
```
reduce(init=0.0f) <lambda body>...
```

  - We need to also **model reduce's attributes into IR graph** using input list
  - Add a new IR node called, KeyedAttribute
  - Reduce IR node will be like
  - Notes, the Reduce will reduce last dimension of input
    - We will have a analysis pass, TypeInference to consolidate and reconsile the type
    - Aftewards, all node will be typped

```
Reduce(
  Input(
    param, // For reduce, only one parameter as input, it is a SSAUse -> <input>
           // notes the parameter ALWAYS shows up in first/0 index in InputList for Reduce
           // Additionally, pipe is not existed in IR anymore, the builder of IR will
           // desugar it, the param will always points to another Reduce or Map
    KeyedAttribute(name='init', value=<some value>),
  ),
  Effect(...),
  As(...),
  Lambda(
    // Similar as MAP
    ...
  ),
)

```

# Overall

- Few nodes to be added
  - IterClusterProject
  - InductionVariable
  - TensorDimOf
  - KeyedAttribute

- Few nodes to be modified
  - Reduce
  - Map

- One pass needs to be added
  - Simplification
    - Which is a small lowering pass
    - Contains 2 separate passes
    - Lower IterCluster into Iter node
    - Perform inline for all Map/Reduce operations
