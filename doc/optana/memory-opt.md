# Optimize redundant memory operations

## Background

- Memory model
  - Each SSA points to a unique memory node is treated as
    - 1) ONLY name to that specific memory node
    - 2) All access to same memory node is treated equally
      - NO alias analysis against the subscript for now
  - Memory's side effect order is guaranteed by ImpactBy list


- Memory store representation in IR
  - Initially, for all local declared tensor type
    - All stored in GMEM
    - All will use MemStore to write them out
    - MemStore will be chained after the Map/Reduce lambda
      - The fused chain's very last one is essentially a MemStore
    - MemStore takes
      - input#0: input value
      - input#1: dest memory node, via SSAUse
      - input#2 ... k: dimension where the value is supposed to be copied to
      - MemStore returns value of input#0
  - Example

```

// User written frontend syntax conceptually

kernel MatMul(
  lhs: tensor<dtype=f32, shape=[M, K]>,
  rhs: tensor<dtype=f32, shape=[K, N]>,
  scale: f32,
) -> tensor<dtype=f32, shape=[M, N]> {
  dimvar @m;
  dimvar @n;
  dimvar @k;

  // here we have a local storage, this is automatically stored in GMEM regardlessly
  let matmul_result: tensor<dtype=f32, shape=[M, N]> =
    map                  on(lhs[@m][@k], rhs[@k][@n]) (l: f32, r: f32) -> f32: l * r |
    reduce(init = 0.0f)                               (init, %0) -> f32: init + %0
  ;

  return map on(matmul_result[@m][@n]) (value: f32) -> value * scale;
}

// During the IR construction the matmul_result will be constructed as following

MemStore(
  Map,
  matmul_result, /** SSAUse -> MemInit (allocate global memory) **/
  m's induction variable, /** SSAUse -> InductionVaraible **/,
  n's induction variable, /** SSAUse -> InductionVaraible **/,
  k's induction variable, /** SSAUse -> InductionVaraible **/,
)

// notes for **modify an existed local variable**, the MemStore is exactly the same,
// ie still pointed to MemInit for its dest(input #1) arguments. Except its ImpactBy
// contains recent most MemStore as side effect chain dependency.

```

## Optimization

- Store load forward
  - forward the value stored with very next load at exactly same position
  - Candidative operations
    - For all MemLoad: A
      - visit its side effect chain until it hit a recent most MemStore: B
      - checking (A, B) validity for optimization
      - If A and B are pointed to the same position in same memory node
        - Same position is defined as
          - For MemLoad and MemStore, its access path is represented as
            - (<memory node>, <dim0>, <dim1>, <dim2>, ...)
          - 2 access paths are same iff
            - 1) same memory node, this can directly compare on the SSAUse pointed to value node
              - compare pointer address sort of
            - 2) same subscript
              - Same length of subscript for both access path
              - For a specific dimension, they are same ONLY when they are ValueEqual returns true
              - Notes, this pass runs after GVN, hence mostly 2 dims are same when pointed to same ir NODE
        - Forward first Store's value parameter to replace the second MemLoad

- Store store elimination
  - 2 stores to same memory node without any reading in between causes redundancy
  - **RUN it after DCE**
  - Backwards travesal the graph
    - For a MemStore node
      - Walk its fffect chain

      - Finding a MemStore regarding exactly same node 
        - Notes this means no interleaved read, since otherwise we should see MemLoad
        - Effect chain maintains total order

      - Remove the previous MemStore node entirely

- Dead store elimination
  - For a MemStore, if the no other MemLoad happened after MemStore in **effect chain**
    - Can simply just check SSAVar has use or not
      - SSAVar is for the MemStore this specific one
      - The use of MemStore will ONLY BE INSIDE OF effect chain list
      - Memory usage is directly on the original memory node, not on last modification
      - Memory is not SSAify
    - It is subjected to be removed, since no one read the value of that store

  - Notes the dead store can arise after loop merge/fusion operation
    - Ie we chain as much as separate map-reduce cluster into a single one, which will leave intermediate GMEM become dead

- This pass will be run multiple times
  - For example after DCE and after loop fusion/merge optimization

- Please also check the algorithm is sound or not
  - It should be sound since we have strict alias assumption
    - All memory operation happened on same memory node is serialized automatically
  - The only concerns are the node that touches memory are correctly represented and also follow the effect rule

## Review verdict (implemented corrections)

The algorithm review (quick task 261020-ejf, `.planning/quick/261020-ejf-*/261020-ejf-REVIEW.md` — the full record) found the three optimizations SOUND under the memory model's preconditions (restrict-alias rule, chained reads, single-Return observability), with the corrections below implemented in `internal/opt/memopt` exactly as stated. The prose above stays as originally written for auditability; where it disagrees with this section, THIS section is what the pass implements.

- **C2a — dead-store rule (corrects "no MemLoad after ⇒ remove").** The prose rule alone is unsound (it would kill output-tensor stores at *different* positions, which the caller observes). The sound operational rule is the doc's own SSAVar test: erase a store iff it has ZERO live uses — no chain successor, not a Return ImpactBy target, no data uses — with data uses first RAUW'd to the store's value operand (a store returns input#0, so the rewire is value-preserving). Swept to a fixpoint so tail cascades (S1→S2 both unread, region unwired — the post-loop-fusion intermediate-GMEM scenario) fully die.
- **C2b — store-store elimination needs the same PROVEN access path.** "Same memory node" is not enough: the rule fires only on same tensor operand node + same axes arity + pairwise `ValueEqual` axes (identity-keyed, post-GVN). The backwards walk STOPS at the first load (it may read that position); walking PAST an unequal-position store is sound under the restrict-alias rule. Forwarding (store→load) follows the most-recent-store discipline: the FIRST store reached behind the load, proven same path, only then forward.
- **C2c — region keys and "MemInit".** The doc's "MemInit" is `KindMemNew`; the store/load tensor operand is whatever node the region is keyed by (RoleTensor InputParam today, MemNew for locals). Regions key by tensor operand NODE identity — exactly what `Env` does — never by kind, never by name.
- **C1 — chain-totality eligibility guard (load-bearing for soundness).** The walks trust the chain. Per region (grouped by tensor operand node over the live `KindMemLoad`/`KindMemStore` set), the pass verifies ONE LINEAR chain: each access ≤1 ImpactBy edge to a predecessor in the set, ≤1 effect-user successor in the set ∪ {Return}, the whole set connected as a single path. A region that fails (e.g. a store built through un-chained `NewMemStore`) is legal-but-unoptimized — SKIPPED silently: no error, no mutation.
- **C3 — chain splicing needs the `MutContext.RemoveImpactBy` choke point.** Removing an access from a chain re-points its successor's ImpactBy edge at the removed node's predecessor via plain `SetDef`; when the removed node is the chain HEAD, the successor's payload edge is DROPPED entirely through `RemoveImpactBy` (kill record + drop the handle from the bearer's `effectIns` — `RemoveUse` cannot do this; the killed record would linger and the verifier flags it). This is the chain-splice half DCE's hinge-6 conservatism documented as missing.
- **C4 — scope.** GMEM HIR only (`KindMemLoad`/`KindMemStore`; the LIR SMEM/FRAG families carry no effect chains and are out of scope by construction). Level-1 sea only (nil/materialized gates). Pipeline slot: priority 112, immediately after `gvn`/`gvn-verify` (110/111) — identity-keyed axes make `ValueEqual` maximally effective post-GVN — with the sibling `memopt-verify` at 113. The pass is erase/rewire-only: no new nodes, no new kinds.

**Post-conditions (INV-A..G), asserted by `memopt-verify` and the property tests:** SSA integrity (`ir.Verify` clean, no dangling use records); chain preservation (survivors form one linear chain, a subsequence of the original order; Return's ImpactBy names every wired region's surviving tail); memory-state preservation (every surviving load still resolves to the same most-recent proven-same-position store, or was forwarded against exactly it); observability preservation (no Return-wired store is ever erased; Return data operands keep the same VALUES via RAUW-to-value); erase/rewire-only with every mutation riding one `MutContext{Pass: "memopt"}`; idempotence (a second run changes nothing); determinism (allocation-order regions, chain-order accesses, no map-iteration-influenced decisions).

**Expectation note:** in today's default builder graphs the rules rarely fire (tensor reads are paramRefs, not GMEM loads). The pass is the enabler for IterCluster-format graphs, the frontend's local-tensor model, and future loop-fusion runs — the stated reason this doc exists.
