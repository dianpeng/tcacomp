# Implement Type Inference (TI)

- Notes the TI works on all nodes
  - Regardless weather it is LIR or HIR
  - Run multiple times during optimization pass
    - For example, after lowering, run it again
  - Must be idempotent: can be executed multiple-times

- Type inference algorithm
  - A data flow algorithm
  - Mostly just one pass
    - But u have to check weather we really can use one pass or setup a conservative run to fixed point loop
  - Meet operator defined for each node that needs operand from else where
    - (1) Expr node CAN ONLY work on scalar type
    - (2) Loop related node
      - Iter -> u32
      - IterCluster -> untyped, lowered before TI pass
      - IterClusterProject -> untyped, lowered before TI pass
      - ForLoop -> u32
        - Same as Iter, but just for lowering part
      - InductionVariable -> u32
      - DimVar -> u32
      - TensorDimOf -> u32
    - (3) Lambda
      - Map    -> return tensor type indicated by **As** field
      - Reduce -> Remove the last dimension of previously returned type
        - User can also provide As as output for Reduce
        - If provided, validate As against the derived one from the rules to be compatible
    - (4) Memory
      - In general memory node can return scalar/tensor(ie sliced view)
      - MemLoad -> derive based on operand
        - input#0: SSAUse -> Memory Node
        - input#1...K: index into different dimension
        - The returned value type is
          - Type of input #0
          - Based on # of indexing to decide returned type
            - If # of indexing provided is not fully deref to the final value, write out a tensor type with unprovided dimension as shape
            - Otherwise return the tensor's dtype
        - For example
```

mem1 = MemLoad(tensor:<dtype=f32, shape=[M, K, N]>, <index node>); // type will be: tensor<dtype=f32, shape=[K, N]>
mem1 = MemLoad(
  tensor:<dtype=f32, shape=[M, K, N]>,
  <index node>,
  <index node>,
  <index node>,
); // type will be: f32

```
      - MemStore -> return **value been stored's type as type returned**
        - MemStore will return stored value
        - It does support multi-dim array store, hence it can return tensor as well
        - input#0: SSAUse -> value to be stored
        - input#1: SSAUse -> memory region/node
        - input#2 ... K: Index for memory node

      - Notes for all memory node, same, regardless weather it is HIR or LIR

    - (5) Other nodes
      - Param:
        - input parameter is typed by user, otherwise return error
      - Return:
        - expression of type returned will be return's type
