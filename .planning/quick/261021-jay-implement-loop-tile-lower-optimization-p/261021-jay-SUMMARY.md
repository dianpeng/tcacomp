---
phase: quick-261021-jay
plan: 01
type: summary
status: complete
---

# Quick Task 261021-jay — Summary

## Objective

Implement the loop-tile-lower optimization pass — the lowering sequel to the
loop-tiling decide pass — per the normative spec `doc/optana/loop_tile_lower.md`
(§6bis mechanical procedure), consuming the decide pass's tiling-record side
table and performing the IR node surgery: annotated tiled ForLoop nests,
IntrinsicGridMap/IntrinsicThreadMap queries, and SMEM/FRAG cache insertion.

## What was built

**Task 1 — PHASE 0, IR prerequisites** (commit `49fa94f`):
- `KindIntrinsicGridMap`/`KindIntrinsicThreadMap` (LIR tier, FamilyAbstract,
  append-only before `KindEnd`, no KindVersion bump) with all six §5
  companions: node struct + catalog rows (`internal/ir/node_intrinsic.go`,
  `catalog.go`), allocNode cases (`node.go`), verify shape case
  (`verify.go`: one const-u32 DoSoC operand, field vocabulary
  index|dimension), typeinfer u32 rules (`rules.go`), legality-window note
  in the catalog row doc (P9), contentHash/contentEqual field mixing
  (`value.go`).
- `umin`/`umax` u32 builtins (`builtins.go`, P2).
- M1 variadic annotated `NewForLoop` (soc/dosoc/loc KeyedAttribute trailing
  inputs, verifier-enforced, print-rendered).
- `KeyedAttributeOf` exported read face (`node_iterfmt.go`).
- `doc/ir.md` updated (33 kinds, KindEnd 34, forloop arity -1).

**Tasks 2+3 — the pass** (commits `5ef7bbe`, `4da1ea2`):
- `internal/opt/tiling/lower.go` — `tile-lower`@116, Requires `["tiling"]`,
  Preserved EMPTY (record dies with the pass), fail-closed record fetch,
  PHASE 1 derivation, PHASE 5 epilogue (the §8 self-check sweep as pass
  exit conditions, P7): retired-iters, loop annotations, fill layering,
  no stale loads, no barriers, FRAG fully-unroll.
- `internal/opt/tiling/lower_rewrite.go` — PHASE 2 DoSoC numbering (§2,
  per-LNT restart, coupled-warp counting) + PHASE 3 loop rewrite: the
  SEQ/CTA/WARP templates (T==1 single loops; T>1 between+inside with
  materialized `umin(T, extent − offset)` bounds), M1 annotations,
  inside-out nest construction, deepest-effectful-op body placement, IV
  RAUW + MutContext-only retirement, P6 effect-chain re-pointing.
- `internal/opt/tiling/lower_cache.go` — budget resolution (halve →
  drop-at-1, pre-rewrite overrides) + PHASE 4 site surgery: per-site
  buffers with bound tile dtypes, tile-shaped partial-deref fill sources
  (covered dims loader-enumerated, round bases on the fill store,
  loadtile variant annotation), chained buffer accesses, P6 chain
  re-pointing to the fill source, body-root re-pointing.
- `internal/ir/node_lir_memory.go` — `NewMemLoadSMemChained`/
  `NewFragLoadChained` (env-chained construction precedent).
- `internal/opt/typeinfer/rules.go` — the allocation rule returns a
  cache-lowering-bound dtype (the buffer-typing contract the MemLoadSMem
  rule's own message names).
- `internal/opt/registry.go` — the 116 anchor line.

## Verification

- Full-repo gates green throughout: `go build ./... && go vet ./... &&
  go test ./...`.
- Pinned tests (`lower_test.go`): fail-closed gates + registry; §2 numbering
  (distinct coupled/independent thread dims, per-LNT restart); SEQ/CTA/WARP
  templates incl. strides, bounds, rank intrinsics; T==1 single loops;
  annotation vocabulary; graph health (verify-zero + P6 liveness);
  determinism (byte-identical renders); P8 typeinfer re-run; matmul
  end-to-end with real data flow (access rewrites, buffer extents, loadtile
  variants, loads-only); the §6bis worked check; budget halving/drop;
  individually-injected §8 self-check violations; and the decide →
  tiling-verify → tile-lower composition over the real search record.

## Deviations from the plan (all documented at the code sites)

1. **P8 buffer typing**: the pass DOES bind buffer tile dtypes via
   `MutContext.SetType` — the typeinfer `MemLoadSMem` rule demands a typed
   buffer and its own error message names cache lowering as the binder;
   the allocation rule now returns a bound cell. Every OTHER inserted node
   stays untyped per P8 (pipeline re-run infers them).
2. **Fill source shape** (P5 reading): covered dims contribute NO axis to
   the fill's source deref (the loader enumerates the tile interior —
   "the compiler emits no loop for the fill"); the covered round bases ride
   the fill STORE's biv operands. Compound axes mixing covered and
   above-tie iters substitute round bases. This makes the fill the P5
   tile-shaped partial deref for pure sites (e.g. lhs `lhs[m]`).
3. **Fill layering check scope**: narrowed to the covered dims' own
   inside-loop coordinates — the spec's worked example has the lhs fill's
   m coordinate legitimately reference the enclosing M index expression
   (which contains M's inside IV); the reachable §4 violation class is a
   fill referencing a covered dim's in-tile coordinate.
4. **Drop-at-1 sweep**: shared covered iters can drive a non-picked site
   to tile 1, so the drop sweep covers every promoted SMEM site after each
   halving round (facts order, deterministic).
5. **Site-fact derivation timing**: the INS chase runs at PHASE 1
   (pre-rewrite) — it walks InductionVariable nodes PHASE 3 retires; the
   surgery itself still runs after the loop rewrite per §6bis.

## Must-haves truth audit

All six plan must_haves hold: no live KindIter/InductionVariable of a
rewritten LNT survives; per-LNT DoSoC bindings restart counters and give
sequential dims no dimension; every promoted MemLoad is erased with uses
RAUW'd to accesses over a buffer declared at the fill point; every
materialized bound is a u32-representable umin call; `ir.Verify` is clean
and a typeinfer re-run succeeds; the pass fails closed without the record.

## Next steps

- Phase 02 (naive-codegen-runtime-e2e) consumes the lowered form; codegen
  maps the soc/dosoc/loc annotations to the cta_loop/thr_loop/seq_loop
  materializations (§4) and the intrinsics to the per-LNT map queries.
