# Review verdict: doc/optana/memory-opt.md vs the source code

**Reviewer context:** read against `internal/ir/{effect,node_memory,ssa,mutate,verify,dce,node_end,walk,value,node}.go`,
`internal/opt/{manager,registry}.go`, `internal/opt/{dce,gvn,simplify}/*`, `internal/analysis/sideeffect/sideeffect.go`,
`pkg/{builder,chain}.go`, `internal/opt/dce/dce_test.go`.

## VERDICT: SOUND — safe to implement, with 4 binding conditions

The three optimizations (store→load forwarding, store-store elimination, dead-store
elimination) are sound **under the existing memory model's preconditions**, and those
preconditions are exactly the ones the IR already documents. Two prose rules in the doc
are imprecise and MUST be implemented in corrected form (C2a/C2b below), and one
mechanical detail is missing from the doc entirely (C3: chain splicing needs a new
`MutContext` choke point).

### Why it is sound (the argument, checked against the code)

The IR's memory model (`internal/ir/effect.go`) is a sea-of-nodes with **explicit
per-region total order** carried by `ImpactBy` effect chains:

- One chain per memory region; region identity = the tensor operand NODE
  (`Env.heads` is keyed by `tensor.BasePtr()` — never a name).
- Chains include READS (`NewMemLoadChained` chains a load exactly like a store), so
  two accesses to one region always form a second-ImpactBy-first edge.
- Restrict-alias rule: kernel tensors never alias (documented construction precondition).
- Observability: only what the single Return (end node, R2) observes — its returned
  values (data operands) plus each output tensor's final chain head (ImpactBy payload,
  wired by `NewReturnChained` from `pkg.Builder.Build`).

Given that model, each rule was checked including the **unknown-aliasing** hazard
(two structurally-unequal subscript nodes may still denote the same runtime
coordinate — ValueEqual is identity-keyed, not value-computing):

1. **Store→load forwarding** — forward load L from store B only when B is the FIRST
   store reached walking L's chain backwards AND access paths are proven equal
   (same tensor node + same arity + pairwise `ValueEqual` axes). If an
   unequal-but-runtime-aliasing store sits between B and L, IT is the first store
   reached → no forward. Loads between B and L never write. So the forwarded value
   is always the semantically-latest write at L's position. SOUND.
2. **Store-store elimination** — eliminate earlier store S0 when a later store S has
   the SAME PROVEN access path and the walk S→…→S0 crosses only stores (any load
   stops the walk: it may read that position). Even under unknown aliasing: any load
   after S whose coordinate aliases S0's must also alias S (S0 ≡ S proven), and S is
   later, so it reads S — S0 is never the latest write at any observed coordinate.
   SOUND. Walking PAST unequal-position stores is sound; stopping at the first load
   is mandatory.
3. **Dead-store elimination** — a store with ZERO live SSA uses is unobservable: no
   chain successor can read it (no successor), the region's final state is not
   returned (not Return-wired), and its value uses are preserved by RAUW to its value
   operand (a store returns input#0). SOUND **only under chain totality** (below).

### Binding conditions (the pass MUST implement these)

- **C1 — chain totality/linearity guard (load-bearing for soundness).** The walk
  trusts the chain. A store created un-chained (`NewMemStore` direct) is invisible to
  the walk and would break rule 3 (and the doc's own note admits this: "the only
  concerns are the node that touches memory are correctly represented"). The pass
  must derive, per region, the access set (graph scan over live `KindMemLoad`/
  `KindMemStore` grouped by tensor operand node) and verify it forms ONE linear chain
  (each access ≤1 ImpactBy edge to a predecessor in the set; ≤1 effect-user successor
  in the set ∪ {Return}; the whole set connected as a single path). A region that
  fails = legal-but-unoptimized (effect.go documents un-chained construction as
  legal) → SKIP the region, never error, never touch it.
- **C2 — doc corrections (implement the corrected rules, not the prose):**
  - C2a: "no MemLoad after ⇒ remove" is UNSOUND taken alone (it would kill
    output-tensor stores at *different* positions, which the caller observes). The
    sound operational rule is the doc's own SSAVar test: erase a store iff it has
    ZERO live uses (no chain successor, not a Return ImpactBy target, no data uses;
    data uses are first RAUW'd to the stored value). Sweep to a fixpoint so
    tail-cascades (S1→S2, both unread, region unwired) fully die — the doc's
    post-loop-fusion intermediate-GMEM scenario.
  - C2b: store-store elimination requires the SAME ACCESS PATH (tensor + proven-equal
    subscript), not merely "same memory node". Walk stops at the first load.
  - C2c: the doc's "MemInit" is `KindMemNew`; the store/load tensor operand is
    whatever node the region is keyed by (RoleTensor InputParam today, MemNew for
    locals per the doc's model). Key regions by tensor operand node identity
    (exactly what `Env` does) — never by kind, never by name.
- **C3 — chain splicing needs a new IR choke point.** Removing an access from the
  middle of a chain must re-point its successor's ImpactBy edge at the removed node's
  predecessor. When the removed node is the chain HEAD, the successor's payload edge
  must be DROPPED entirely. `MutContext.RemoveUse` cannot do this: it kills the
  record but cannot remove the handle from the bearer's `effectIns` slice — the
  killed record stays in the slice and `Verify` flags it ("holds a killed use
  record"). DCE's own hinge-6 test documents this gap ("removing it would need chain
  splicing … which v0's DCE deliberately does not do"). Add
  `MutContext.RemoveImpactBy(u SSAUse) error` in `internal/ir/mutate.go` (drop the
  handle from the user's effectIns by record identity, kill the record, stamp) —
  following the existing choke-point conventions (IR-04). Splice via plain
  `SetDef(succEdge, pred)` when a predecessor exists.
- **C4 — scope.** GMEM HIR only (`KindMemLoad`/`KindMemStore`). LIR SMEM/FRAG
  families carry no effect chains (`node_lir_memory.go` has no effectIns) and are
  out of scope. Level-1 sea only (level gates like dce/gvn). Pipeline slot: after
  `gvn` (doc: "runs after GVN"; identity-keyed axes make ValueEqual most effective
  post-GVN). The pass erase/rewire-only — no new nodes, no new kinds.

Note for expectations: in TODAY's default pipeline graphs (builder path) tensor reads
are paramRefs, not MemLoads, and simplify-lower rewrites load axes to
InductionVariables while store axes stay dim params — so the rules rarely fire on
current builder output. The pass becomes load-bearing for IterCluster-format graphs,
the frontend's local-tensor model, and future loop-fusion runs (the doc's stated
reason). This is fine — the pass's own tests construct chain-carrying graphs directly
via the ir constructors (same as dce_test.go).

## Invariants (pre/post) — the verification and property-test substrate

**Maintained by the pass (post-conditions, all assertable):**

- **INV-A SSA integrity:** `ir.Verify(g)` clean after the pass; no use record
  dangles at an erased node; ref lists match owning slots (both chain directions).
- **INV-B chain preservation:** per region, surviving accesses still form ONE linear
  chain; each access's ≤1 effect-user is its chain successor or the Return; the
  survivor order is the original chain order minus removed accesses (a subsequence).
  Splices only ever shortcut removed middle links; Return's ImpactBy keeps naming the
  surviving chain tail of every wired region.
- **INV-C memory-state preservation (the correctness theorem):** for every
  Return-wired region, the final memory state is unchanged: every surviving load
  still resolves to the same most-recent proven-same-position store as before the
  pass (or was itself forwarded against exactly that store, RAUW'd to its value);
  the last write before Return at every position is unchanged.
- **INV-D observability preservation:** no Return-wired store is ever erased
  (forwarding erases loads only; store-store erases strictly-earlier stores;
  dead-store erases zero-use stores — a Return-wired store always has its payload
  use). Return's data operands keep the same VALUES (RAUW-to-value for erased defs).
- **INV-E erase/rewire-only:** node count monotonically decreases; no new nodes or
  kinds (StageConstructed legality re-checks green); mutations ride one
  `MutContext{Pass: "memopt"}` so LastTouchedBy is structural.
- **INV-F idempotence:** a second run changes nothing (also the memopt-verify
  fixpoint assertion).
- **INV-G determinism:** AllocatedNodes order; per-region first-access order; chain
  order. No map-iteration-influenced decisions.

**Assumed on entry (pre-conditions, guarded per C1):** chain linearity/completeness
per region; verified graph (level gates: nil / materialized / CFG-carrying →
deterministic error, untouched graph).

## memopt-verify (sibling verification pass, per repo convention)

Priority 113, Requires ["memopt"], mutates nothing. Re-derives chains independently
(shared derivation code inside the package, mirroring gvn's shared `lambdaBodyRoots`)
and asserts: (1) no dangling uses of/erased defs in memory nodes; (2) FIXPOINT: no
forwardable load remains, no eliminable store pair remains, no zero-use store
remains — over ELIGIBLE (C1-passing) regions only, so legally-unoptimizable regions
do not fail it; (3) effect users of every access ∈ accesses ∪ {Return}.

## Property tests (mirroring dce_test.go's fixture+assert style)

1. Gates: nil graph; materialized graph refused untouched.
2. Forwarding fires: S(p,v)→L(p): L erased, users RAUW'd to v, successor spliced to
   S, Return data operand updated when it returned the load, `ir.Verify` green.
3. Forwarding blocked: different position; intervening unequal store (most-recent
   store discipline); no predecessor store at all.
4. Two loads same position both forwarded; middle load chain-splice (hinge-6 shape).
5. Store-store: same position no read → earlier erased; blocked by interleaved load;
   different positions kept; walk-past-unequal-store (S0(p)→S1(p′)→S2(p): S0 dies,
   S1 stays); 3-store cascade.
6. Dead stores: unwired S1→S2 tail cascade both erased; data-use store RAUW'd to its
   value; Return-wired tail NEVER erased.
7. C1 guard: two chains on one region (one access built un-chained) → region skipped,
   nothing erased, no error.
8. Head-removal splice via the new `RemoveImpactBy` choke point (store-store where
   the earlier store is the chain head) — the case C3 exists for.
9. Idempotence; Manager interop (`[dce, gvn, memopt, memopt-verify]` + VerifyEach);
   registry/BuildPipeline includes memopt after the `all` blank import.
10. memopt-verify: green after the pass on every fixture; flags a planted
    forwardable load; Requires-order rejection; level gates.

## Implementation shape (convention-conformant)

- `internal/opt/memopt/memopt.go` — `Pass{}`, ID "memopt", Priority 112,
  Requires ["gvn"], Produces StageConstructed, Preserved nil. Run: level gates →
  derive chains (C1) → fixpoint { forward loads → eliminate shadowed stores →
  sweep zero-use stores } over eligible regions, all mutations through one
  MutContext.
- `internal/opt/memopt/verify.go` — `VerifyPass{}`, ID "memopt-verify", Priority
  113, Requires ["memopt"].
- `internal/ir/mutate.go` — add `RemoveImpactBy` choke point (C3).
- `internal/opt/all/all.go` — blank import; `internal/opt/registry.go` — extend the
  priority-scale comment (112/113 anchors).
- `doc/optana/memory-opt.md` — append the review-verdict section (corrections
  C2a/C2b, conditions, invariants) so the algorithm doc and implementation agree.
- Tests in `internal/opt/memopt/memopt_test.go` (+ a small mutate-level test for the
  new choke point in `internal/ir`).
