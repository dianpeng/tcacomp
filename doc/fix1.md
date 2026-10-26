# fix1 — remove loop body operands: invalid code per the sea-of-nodes design

**Status:** RULED (2026-10-24, author ruling — supersedes the F9 request; see §4)
**Scope:** `internal/ir` (ForLoop layout, LoopSpec, verifier loop-scope), `internal/analysis/loopforest`, `internal/opt/tiling` (rewrite + §8 checks), `internal/opt/gvn`, `internal/opt/schedule` — no generated-code semantics change
**First surfaced by:** quick-261024-k2i — the first real frontend program through the default pipeline (`matmul_scale`) fails tile-lower §8; diagnosis exposed the body-operand model as a design violation, not a localized bug.

---

## 0. The IR's placement model (author clarification, 2026-10-24 — normative)

A loop node is a **control-flow node with no body**. The sea-of-nodes never
materializes control-flow containment as an operand:

- **Membership is implicit:** a node is inside a loop's iteration because it
  *uses an InductionVariable* that references the loop. No statement list, no
  body operand, no nesting edge to an expression.
- **Ordering is dependency-only:** data dependencies and side-effect (chain)
  dependencies carry the entire partial order. The scheduler never reads a
  loop's "body statement" — it derives all placement from those edges (the
  261022-b2m unified walk: SN ≤ cap → place at SN; SN > cap → contributing
  nest's epilogue).
- **NEVER a direct link from a control-flow node to an expression.** Loop-to-
  loop structure may be explicit (HIR iters already carry a parent-iter
  operand — loop-to-loop, permitted) or derived from IV references; a loop
  pointing at an expression (the LIR body operand) is invalid code.
- **Liveness is ordinary:** a fill is a normal memory load — it must have a
  memory read consuming it, otherwise it is dead code and memory
  optimization removes it (memopt's domain, not DCE's). No special
  containment rules keep anything alive.

## 1. The defect that exposed it

### 1.1 Symptom

```
tile-lower: fill for site 61 is not reachable from its fill-position loop 146's
body operand (the F8 splice must place the fill inside the mandated host, R1)
```

### 1.2 Trigger shape

A kernel combining, in one LNT: rank ≥ 2, a **reduction of the innermost
axis** (store axes omit the reduced dim), and a **promoted load site** whose
readers sit on the reduce side of the chain. No prior golden combined these
(the no-reduce matmul golden always elected the *store* as body root; the
reduce goldens have no fills).

### 1.3 Mechanics (verified on the failing frontend run)

`bodyRootOf` elects the innermost body operand among MemLoad/MemStore ranked
by deepest referenced iter. With the reduced dim innermost, the store ranks
shallow and **a load wins** (rhs load, ID tie-break). The §8 PLACEMENT check
then requires the fill reachable *backward from the host's body operand* —
but the fill's structural tie runs through the reader chain, which consists
of **users** of the elected load. A backward walk from a load can never reach
its own readers: the check fails on a correctly-spliced graph.

### 1.4 The graph was never wrong

Probe of the failing graph in the IR's own vocabulary (operand-closure IV
references):

```
fill source load  (memload#155): references iv77 (loop 146 = the HOST's IV) ✓
fill store        (fragstore#158): iv77 (host) + iv65 (enclosing) + iv120 (round base) ✓
buffer access     (fragload#159): iv127 (loop 142 = element loop inside host's nest) ✓
                                     chain-predecessor edge: access → fill ✓
final store       (memstore#38): references all six IVs → SN > cap → epilogue ✓
```

Every placement fact the design requires is present: the fill holds its epoch
membership via the host's IV, the access holds its membership via the element
loop's IV, and the fill→access order rides the chain edge the F8 splice
wires. The body-operand check was asking a question the IR does not define.

## 2. What must NOT change

| # | Ruling | Consequence |
|---|--------|-------------|
| C1 | **R-A bracket law** (261024-bz9 HI-01): Init before the K-between loop; Reduce inside the tile-inside walk; Combine after; store after the whole bracket | The store keeps reaching the epilogue via SN > cap; nothing pulls it into a per-round position. |
| C2 | **F8/R1 fill placement** (261021-o61): fills sit at variant-mandated positions, re-executing per fill-key epoch | Placement is emergent from the fill's coordinates (the promoted load's own axes); the structural assertion is the chain edge (§4), not containment. |
| C3 | **One edge relation** (261022-b2m): propagation, containment cap, and ownership all derive from data + side-effect edges; def = ForLoop node → after whole loop; def = IV → iteration membership | Loopforest linking respects this: IV *data-dep* references nest; completion (effect) edges order, never nest. |
| C4 | **Sea-of-nodes loop model** (§0): no body statements, membership via IV back-references, no loop→expression links | Body operands are removed outright — including every consumer built on them (LFA BodyRoot, verifier loop-scope closure, GVN's body-root exemption, buildNest chaining). |
| C5 | Goldens byte-identical for unaffected shapes | No-reduce and no-promoted-site shapes emit unchanged code. |

## 3. The ruled fix (2026-10-24)

**Remove all loops' body operands — they are invalid code.** The relationship
between an IR node and its loop is via the operand/side-effect chain only;
never a direct link from a control-flow node to an expression. The §8 fill
check reduces to the chain fact: **the fill is a chain predecessor of every
buffer access of the site**, and the fill is a normal memory load that must
have a consuming memory read — otherwise memory optimization removes it
(author correction: memopt, not DCE).

Previously considered alternatives, for the record:

- ~~Option A (recast §8 checks around the body model)~~ — superseded by the
  ruling: the body model itself is deleted, not accommodated.
- ~~Option B (elect KindReduce as body root)~~ — moot; props up the invalid model.
- **Option C (forward user-edge walk)** — REJECTED: turns the check into a
  connectivity test that cannot distinguish inside from adjacent.
- **Option D (closure-extended store depth)** — REJECTED, tried and reverted
  in 261024-k2i: puts the post-reduce store inside the reduced-dim strip
  walk, violating R-A. Recorded so it is never retried.

## 4. The ruling (recorded verbatim in substance)

> Remove the all loops' body operand, they are invalid code. The relationship
> between an IR node for its loop is via operand/side effect chain. NEVER
> direct link from control flow node to expression. As with checking, the
> fill is a chain predecessor of every buffer access of the site. The fill
> expression is a normal memory load, which must have a memory read to
> consume, otherwise it should be DCEed.

Derived consequences (defaults; flagged for objection):

- **D-a (ForLoop layout):** `NewForLoop` operands become `[iv, bound, attrs...]`;
  `KindForLoop`'s body slot is deleted from node_loop.go, kind.go's layout
  comment, catalog.go's FamilyLoop row, and print.go.
- **D-b (loopforest):** LIR loop-to-loop nesting derives from **IV data-dep
  references**: loop L_in is nested in L_out iff L_out's IV dimvar is in
  L_in's data-dep operand closure (round bases already carry enclosing IVs —
  probe-verified: the fill store carries iv120/iv65 alongside its own epoch's
  IVs). Effect edges are NEVER linking input (C3: completion edges order,
  never nest). `LoopSpec.BodyRoot` is deleted; the interface keeps `IsLoop` +
  `ParentEdge` (HIR iters keep their parent-iter operand — loop-to-loop).
- **D-c (verifier loop-scope):** `checkLoopScope`'s operand-closure
  containment test is deleted with the body model. Its leakage class
  (reading an induction value not defined at the use's position) is recast
  in sea terms: a use of L's IV *is* membership in L, so the residual
  incoherence is a node that uses L's IV yet is ordered after L's completion
  (a completion edge to L in its chain). If that form proves vacuous in
  practice, the invariant is deleted rather than kept as a tautology — the
  tiling §8 SCOPE check (coordinates reference only enclosing loops) already
  guards the placement-level hazard.
- **D-d (GVN):** `lambdaBodyRoots` and its body-root exemption are deleted;
  with no body operands there is nothing to exempt.
- **D-e (§8 checks):** PLACEMENT-by-reachability and body-subtree LAYERING
  are deleted. ORDERING becomes the structural check: every live buffer
  access of a site has the fill among its chain predecessors (ImpactBy).
  LAYERING is recast: a cooperative fill's operand-closure references no
  inside-loop IV. SCOPE and UNIFORMITY are unchanged (already IV-based).
  EPOCH is no longer a separate check — the fill's coordinates are the
  promoted load's own axes by construction, so epoch membership is inherent.
- **D-f (fill liveness):** a fill whose buffer has no live reads is dead
  code, and its removal belongs to **memopt**, not DCE (author correction,
  2026-10-24: "Not DCE, we have memory optimization which should remove
  it"). A buffer store whose buffer has zero live loads is dead together
  with its source load; this is the same class as memopt's dead-load
  elimination. Wire the fill case into memopt (cf. the open memopt
  dead-load gap, opt-pipeline-audit-rulings — likely the same fix site);
  DCE stays untouched.
- **D-g (rewrite):** `bodyRootOf` is deleted; `buildNest` creates loops with
  `[iv, bound, attrs]` and no body chaining. The P6 inside-set
  (`lw.inside`, today `reachableFrom(body)`) is re-derived as: nodes whose
  data-dep closure references any of the LNT's loops' IVs — the same
  relation loopforest now uses (one membership relation everywhere).

## 5. Implementation plan

Ordered so the tree stays green at each step (each step = one commit):

1. **ir: ForLoop layout** — drop the body slot (node_loop.go NewForLoop
   signature, kind.go comment, catalog.go row, print.go). Update every
   constructor call site (tiling's buildNest is the only producer). Verifier
   loop-scope (D-c) and astcfg compile-time layout expectations updated in
   the same commit. Goldens containing printed forloops regenerate
   mechanically (operand-count deltas only).
2. **analysis/loopforest** — replace BodyRoot-containment linking with
   IV-data-dep derivation (D-b); delete `LoopSpec.BodyRoot`. Loopforest's
   own unit tests re-pin to the new forest derivation.
3. **opt/tiling** — delete `bodyRootOf`, re-derive `lw.inside` (D-g), recast
   `checkFillPlacement` to ORDERING + LAYERING-IV (D-e). This commit flips
   the frontend e2e trigger from fail to pass at tile-lower.
4. **opt/gvn** — delete `lambdaBodyRoots` and the exemption (D-d).
5. **opt/schedule** — replace local body/reach derivations with the
   loopforest IV-membership relation; fail-closed gates unchanged.
6. **memopt fill liveness** (D-f) — unconsumed buffer stores (fills with no
   live buffer reads) are removed by memopt's dead-load/store elimination,
   together with their source loads; DCE is untouched.
7. **Tests (fundamental only, per design.md):**
   1. *Positive:* the frontend trigger fixture passes the recast §8 suite;
      loopforest unit tests pin IV-derived nesting (inner loop under outer).
   2. *Negative pins (non-vacuity):* (a) a fill wired without the chain edge
      to an access fails ORDERING; (b) a cooperative fill carrying an
      inside-IV coordinate fails LAYERING; (c) an unconsumed fill is removed
      by memopt.
   3. *Golden regression:* `matmul_store.cu`, `reduce_warp.cu` byte-identical
      (C5).
   4. *E2E gate:* `TCACOMP_E2E=1 go test ./internal/e2e/ -run Frontend` —
      both Appendix A variants full-pipeline, CPU-reference match within
      reduceTol, sanitizers clean. **Contingency:** downstream walls
      (schedule/final-lower on the reduce shape, cf. 261024-bz9 deferred
      rank-3 items) are new findings with their own rulings; this fix is not
      judged by them.

### 5.1 Downstream verification (no code change expected)

Per C3 the scheduler already places the trigger chain correctly once the §8
gate stops rejecting it: access and per-round compute inside the element
loops (SN from IV/dep closures), `KindReduce` inside the tile-inside walk,
piped map and store at the reduced nest's epilogue (SN > cap). Pin with one
schedule-level assertion over the trigger fixture (store's SchedBlock =
epilogue rank) and the existing bracket scanner
(`internal/cg/cxx/bracket.go`) once the shape reaches codegen.

## 6. Risks / open points

- **`reachableFrom` semantics shift:** its remaining callers must be audited
  so no walk silently changes meaning when body operands vanish (the P6
  re-derivation in D-g is the intended replacement, not a side effect).
- **astcfg Loop blocks:** `NewASTLoop` builds Loop blocks from live ForLoop
  nodes; the block's *body items* come from the schedule's SchedBlock
  assignment, not body operands — verify this holds once the slot is gone
  (expected: yes; the slot fed forest derivation, not AST population).
- **Multi-reduce chains** (two reduces, builder root-as parking): the recast
  checks are per-fill and dim-agnostic; no new exposure expected.
- **Cross-CTA reductions** out of scope (v0 single-CTA, doc/tca.md §5.10).
