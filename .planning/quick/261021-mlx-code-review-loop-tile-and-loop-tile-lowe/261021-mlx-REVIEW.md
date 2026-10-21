---
status: complete
reviewed: 2026-10-21T17:30:00Z
depth: deep
files_reviewed: 36
files_reviewed_list:
  - internal/opt/tiling/tiling.go
  - internal/opt/tiling/search.go
  - internal/opt/tiling/overlap.go
  - internal/opt/tiling/record.go
  - internal/opt/tiling/verify.go
  - internal/opt/tiling/lower.go
  - internal/opt/tiling/lower_rewrite.go
  - internal/opt/tiling/lower_cache.go
  - internal/opt/tiling/tiling_test.go
  - internal/opt/tiling/overlap_test.go
  - internal/opt/tiling/verify_test.go
  - internal/opt/tiling/lower_test.go
  - internal/ir/node_intrinsic.go
  - internal/ir/node_lir_memory.go
  - internal/ir/node_loop.go
  - internal/ir/node_dimvar.go
  - internal/ir/node_memory.go
  - internal/ir/node.go
  - internal/ir/kind.go
  - internal/ir/catalog.go
  - internal/ir/builtins.go
  - internal/ir/value.go
  - internal/ir/verify.go
  - internal/ir/print.go
  - internal/ir/effect.go
  - internal/ir/mutate.go
  - internal/ir/walk.go
  - internal/ir/schedule.go
  - internal/ir/dce.go
  - internal/opt/typeinfer/rules.go
  - internal/opt/registry.go
  - internal/opt/manager.go
  - internal/opt/cache.go
  - internal/opt/all/all.go
  - doc/optana/loop_tile_lower.md
  - doc/optana/loop_tiling.md
findings:
  critical: 2
  high: 6
  medium: 5
  low: 7
  total: 20
---

# Code Review: loop-tiling decide pass + loop-tile-lower pass (261021-mlx)

**Scope:** `internal/opt/tiling/` (both halves), the IR prerequisites landed with the lower half
(`internal/ir/node_intrinsic.go`, `node_lir_memory.go`, `node_loop.go`, `kind.go`, `builtins.go`,
`value.go`, `verify.go`, `print.go`, `catalog.go`), the typeinfer rules, the registry/manager
wiring, and the tests. Specs: `doc/optana/loop_tile_lower.md` (§4, §5, §6, §6bis, §7, §7bis, §8),
`doc/optana/loop_tiling.md`, `doc/opt.md` ordering.

**Method:** specs read first, then code walked in execution order (record fetch → DoSoC →
rewrite → cache insertion → fallback → epilogue). One full matmul hand-trace (M=1024, N=2048,
K=512; M=CTA/100/T32, N=WARP/8/T32, K=SEQ/T32; lhs tie=iterN covering {K}, rhs tie=nil covering
{N,K}) — strides, offsets, tile bounds, buffer extents and reuse counts recomputed by hand and
cross-checked against the code and the pinned tests. `go build ./...`, `go vet`, and
`go test ./internal/opt/tiling/ ./internal/ir/ ./internal/opt/typeinfer/` all pass (green
baseline; no GPU e2e was run, none exists).

## Summary table

| ID  | Sev | Area | One-line summary | Tested? |
|-----|-----|------|------------------|---------|
| CR-01 | CRITICAL | lower_rewrite | u32 wraparound breaks `tile_max` on tail rounds → inside loops run out-of-range ranks → OOB accesses | untested claim (no tail-round test) |
| CR-02 | CRITICAL | lower_cache | Fill is never structurally placed at the §6bis fill point — executes once, not per covered round; reads loop dimvars from outside their loops | tests pin operands only, not placement |
| HI-01 | HIGH | lower_cache/verify | Fill's baked above-tie coordinate contains the dim's tile-INSIDE dimvar — undefined at the mandated fill position; layering check blind to it | test pins the broken shape as expected |
| HI-02 | HIGH | lower_cache | `bodyRoot == load` path puts the fill inside the innermost tile-inside loop (§4 layering violation) and the buffer access outside the nest | untested path |
| HI-03 | HIGH | lower_cache/decide | WARP-covered SMEM site (the doc's own `rhs`) lowers incoherently: per-thread fill base + cooperative LoadTileCta + one T×T shared buffer | test pins variant flag only |
| HI-04 | HIGH | record/search/lower | `WarpPortion == PortionWarp` (legal Params) + a CTA dim ⇒ `coupledThreads == 0` ⇒ zero-stride ForLoop (infinite loop) | untested |
| HI-05 | HIGH | lower_rewrite | Branched LNTs (sibling iters) silently rewritten into a linear nest — siblings become nested, wrong iteration space | untested |
| HI-06 | HIGH | search vs verify/lower | SMEM budget scope mismatch: decide gates per-LNT; tiling-verify + resolveBudget gate kernel-wide | untested (no multi-LNT SMEM fixture) |
| ME-01 | MEDIUM | lower_cache | Budget fallback can strand a promoted site at a covered tile of 1 → `lowerSite` errors ("selection never fails" violated) | drop path tested only for the nested case |
| ME-02 | MEDIUM | lower_cache | §6's per-thread FRAG byte budget is not implemented anywhere | untested claim |
| ME-03 | MEDIUM | overlap/lower | §7's cross-LNT UNION evaluation unimplemented (single-tree ownership, foreign iters silently dropped); cross-LNT rewrite leaks foreign dimvars | untested |
| ME-04 | MEDIUM | lower.go §8 checks | `checkNoBarriers` vacuous (`annotated` always true); fill layering check is data-only, never structural | self-check tests inject synthetic states |
| ME-05 | MEDIUM | tests | "matmul e2e" is IR-level only — no numerical/GPU e2e, no Materialize/print round-trip of the lowered graph | — |
| LO-01 | LOW | node_lir_memory/print | `loadtile` KeyedAttribute rides the store's biv operand list; prints as a bracket subscript; no verifier pin on store trailing operands | partially |
| LO-02 | LOW | node_intrinsic/value | §5's "GVN-CSE of the intrinsics is desirable" is unsound across LNTs (same DoSoC+field = different per-LNT maps) | value-numbering tests are single-LNT |
| LO-03 | LOW | lower.go | tile-lower does not require tiling-verify; corrupted per-iter decisions reach the rewrite unvalidated | untested |
| LO-04 | LOW | legality | Legality windows nominal for the new LIR kinds (all Legal at StageConstructed) | — |
| LO-05 | LOW | lower_rewrite | Extent typing assumed u32; non-u32 BIV maxes emit ill-typed `umin` (caught only by the later typeinfer re-run) | acknowledged in test comment |
| LO-06 | LOW | search/lower | Coupled warp hard-pinned to `AutoWarpPortion`=4 even when no independent warp dim exists — half the warp space idle | untested |
| LO-07 | LOW | lower_cache | Post-RAUW chain re-point loop is dead code (RAUW already moved every live use); comment claims it fires | untested |

Baseline: `go build ./...` clean, `go vet` clean, all three packages' tests pass.

## Hand-traced matmul arithmetic (grounding for the findings)

M=1024 (CTA, LoC=100, T=32), N=2048 (WARP, LoC=8 warps, T=32), K=512 (SEQ, T=32),
`warpsPerPortion = (512/32)/8 = 2`, coupled threads `(8−4)·2·32 = 256`.

- Nest order (buildNest): `M.between(1024, step 3200) → N.between(2048, step 8192) → K.between(512, step 32) → M.inside(step 256, soc=2) → N.inside(step 1, soc=0) → K.inside(step 1, soc=0) → store`. Matches "all betweens in dim order, all insides inside them" — verified by `TestNumbering`.
- M round 0: ranks 0..31 own offsets 0..992, `tile_max = umin(32, 1024−offset)` exact; ranks 32..99 have `offset ∈ [1024, 3168]` — this is the CR-01 tail.
- N: ranks 0..63 valid (offsets 0..2016); ranks 64..255 wrap — CR-01 tail.
- Buffers: lhs `4·32 = 128` B covering {K}; rhs `4·32·32 = 4096` B covering {N,K} — matches `TestLowerMatmulEndToEnd`.
- Reuse (decide objective): lhs reuse dims {N} → `ceil(2048/32)=64`; rhs {M} → `ceil(1024/32)=32`; traffic = `128·63 + 4096·31` — matches `TestTrafficArithmeticAndOptimality`'s independent enumeration.
- Coverage of full rounds is exact and disjoint (rank×T chunking, element-stride coupled walk) — the parallel decomposition itself is arithmetically sound.

## Critical Issues

### CR-01: u32 wraparound defeats the tail-round whole-body no-op (`tile_max` becomes T, not ≤ 0)

**File:** `internal/opt/tiling/lower_rewrite.go:231-234` (tileMax), `:303-306` (CTA offset), `:324-327` (WARP offset); types pinned by `internal/ir/builtins.go:31` and `internal/opt/typeinfer/rules.go:170-190`.

**Evidence:**

```go
func (lr *lowerRun) tileMax(tile int, extent, offset ir.Node) ir.Node {
    sub := lr.g.NewBinOp(lowerOrigin, ir.BinOpSub, extent, offset)
    return lr.g.NewCall(lowerOrigin, "umin", lr.u32c(float64(tile)), sub)
}
```

`extent`, `offset` and the whole chain (`rank` is a u32 intrinsic, `T` a u32 const, dimvars u32)
are u32, so `extent − offset` is **unsigned** arithmetic. `doc/optana/loop_tiling.md:697-700`
explicitly relies on signed behavior: "on a tail round an out-of-range CTA rank gives
a_tile_max <= 0: the WHOLE body becomes a no-op via the loop materialization".

**Why it is wrong:** on the traced matmul, M's between round 0 lets ranks 33..99 compute
`offset` up to 3168 > 1024. At runtime `(u32)(1024 − 3168) = 4294965152`, so
`tile_max = umin(32, 4294965152) = 32` — positive. The inside loop therefore *runs* for
out-of-range ranks (the §4 whole-body skip never triggers — the thr_loop's only bound is this
`tile_max`), and the body evaluates `index = offset + a_i + thread_rank ∈ [3168, 3424)` against
an M extent of 1024: out-of-bounds GMEM/SMEM accesses and OOB tile coordinates. The same
wrap occurs for every WARP dim (thread ranks 64..255 at N=2048, T=32) whenever the extent is
not a multiple of `LoC·T` (CTA) or `nThreads·T` (WARP) — i.e., for essentially every real
shape, since `LoC·T = 3200` at defaults. The spec's "compiler emits no guard and no branch"
ruling is only sound if `tile_max ≤ 0` is observable; in u32 it is not.

**Fix:** make the skip signal representable without a branch, e.g. (a) emit
`tile_max = umin(T, extent − offset)` computed with a saturating subtract (a new u32
`usub_sat` builtin / `umax(0, extent−offset)` shape — still branch-free), or (b) plumb the
raw extent into the inside-loop annotations so the runtime materialization guards the *global*
index against the extent rather than against the wrapped tile bound. Also add a pinned test
with `extent` not a multiple of `LoC·T` asserting the inside bound expression lowers to the
saturating form.

### CR-02: PHASE 4 fill is never placed at the fill point — it executes once, not per covered round

**File:** `internal/opt/tiling/lower_cache.go:188-405` (`lowerSite` — no loop wiring), `:362-378`
(the only placement path, guarded by `lw.bodyRoot == sf.Load`); `internal/ir/schedule.go:39-60`
+ `internal/ir/dce.go:133-139` (Materialize semantics); spec `doc/optana/loop_tile_lower.md`
§6bis PHASE 4 ("FILL POINT: inside the innermost covered dim's between-round body, immediately
before the tile-inside loops", M3).

**Evidence:** `lowerSite` builds `src`, `fillStore`, `buffer`, `access` as free-floating nodes.
The only structural connection to a loop is:

```go
for _, lw := range lr.rewrites {
    if lw.bodyRoot == nil || lw.bodyRoot.BasePtr() != sf.Load.BasePtr() {
        continue   // matmul: bodyRoot is the STORE (bodyRootOf picks deepest+latest), never the loads
    }
    ...SetDef(u, fillStore)...
}
```

For the flagship worked example (matmul), `bodyRootOf` (`lower_rewrite.go:376-431`) picks the
out-store (deepest referenced level = K, latest allocation), so for **both** promoted loads the
guard is false and the fill is attached to nothing: it is ordered only through the buffer effect
chain (`access.effectIns → fillStore`, `lower_cache.go:350-357`). Loop membership in this IR is
"inside the ForLoop body operand subtree"; the fill is in no body subtree.

**Why it is wrong:** once statements exist, a `MemStoreSMem` fill is `isEffectKind` and becomes
its **own top-level statement** (`schedule.go:52-57`), i.e. it executes **once**, before/after
the loops per the topo sort — never per k-round. Its covered round-base operand
(`K.betweenIV`) reads a loop dimvar from **outside** that loop (the verifier has no scoping
invariant, so this passes `ir.Verify` — which is exactly why the tests are green). Every
k-round after the first then reads a stale buffer. This voids the pass's core deliverable for
the project's own headline kernel: the "fill per k_o round" property the spec and
`TestMatmulWorkedCheck` describe exists only in the fill's *operand shapes*, not in the graph
structure.

**Fix:** at PHASE 4, splice the fill into the nest at the mandated point: re-point the
innermost covered dim's between-loop body chain so the round body is
`fill → (old body root of that position)`, i.e. make the fill the body-prefix (a chain node the
tile-inside loops' outermost loop orders after via its ImpactBy edge), and add a §8 self-check
that every fill is data-reachable **from** its covered dim's between loop body operand (the
structural dual of `checkFillLayering`).

## High Issues

### HI-01: Fill's above-tie coordinate bakes the dim's tile-INSIDE dimvar — undefined at the fill point; the layering check is blind to it

**File:** `internal/opt/tiling/lower_cache.go:281-307` (axis bake), `internal/opt/tiling/lower.go:294-337`
(`checkFillLayering`); spec §6bis PHASE 4 + §8.

**Evidence:** the lhs fill's m coordinate is the post-RAUW m axis = `M.indexExpr =
((cta_rank·T + M.betweenIV) + M.insideIV) + thread_rank` (`lower_rewrite.go:312-314`) — it
contains `M.insideIV`, the dimvar of the **M tile-inside loop**. The mandated fill position is
"immediately before the tile-inside loops", i.e. *outside* `M.inside`. `checkFillLayering`
only flags references to **covered** dims' inside IVs (`coveredInside` set, lower.go:308-313)
and its comment claims "a fill's coordinates may reference the full index expressions of dims
OUTER than the fill point" — but `M.inside` is *inner* than that point in the canonical
betweens-then-insides nest.

**Why it is wrong:** there is no position in the generated nest where the fill's coordinates
are all defined and the fill re-executes per k-round only: the fill needs the *current m
element* (inside `M.inside`) while the spec's fill point precedes `M.inside`. The spec's own
worked check ("the lhs fill's m coordinate IS the enclosing m index expression", §6bis) and its
fill-point rule are mutually inconsistent, and the implementation inherits the inconsistency:
`TestMatmulWorkedCheck` (lower_test.go:781-817) pins the m-coordinate-shape but nothing checks
that the coordinate is in scope where the fill sits (it isn't — see CR-02).

**Fix:** decide the intended scoping and encode it structurally: either move the fill inside
`M.inside` (fill per (m-element, k-round), reuse = the N sweep only — matching the tie=iterN
model) or make the above-tie coordinate the round base + rank (still per-thread-defined) — and
extend `checkFillLayering` to assert every dimvar reachable from the fill's source axes is
defined by a loop that encloses the fill's structural position (CR-02's fix).

### HI-02: `bodyRoot == promoted load` corner places the fill INSIDE the innermost tile-inside loop and the access OUTSIDE the nest

**File:** `internal/opt/tiling/lower_cache.go:362-378`; `lower_rewrite.go:344-369` (buildNest).

**Evidence:** when the erased load *was* the LNT's innermost body operand, the re-point makes
`host := innermost.inside`'s body operand the **fillStore** — the fill becomes the body of the
innermost tile-inside (bailing element) loop. §4's layering constraint is explicit: "A
cooperative tile op emitted inside a thr_loop body is a spec violation"; §6bis/M3 place the
fill at the between level. Simultaneously the RAUW (`lower_cache.go:383-385`) moves the load's
*value* users onto the buffer `access`, which now lives outside the nest while reading
`covered` dims' `insideIV` coordinates — use-outside-def-scope again.

**Why it is wrong:** this path produces exactly the structure §4 forbids (cooperative fill
inside the element loop — the cooperative loader can no longer be uniformly called) and/or
reads loop-private coordinates from outside. The path is **never exercised by any test** (all
fixtures' bodyRoot is a store), so it is unvalidated code that fires the moment a kernel's
innermost effectful op referencing the LNT's iters is a load (e.g. post-fusion shapes memopt
produces).

**Fix:** delete the special case and use the single structural fill-point splice from CR-02's
fix for all sites; add a fixture whose bodyRoot is a promoted load.

### HI-03: WARP-covered SMEM site lowers incoherently (per-thread fill base × cooperative LoadTileCta × one T×T shared buffer)

**File:** `internal/opt/tiling/lower_rewrite.go:316-332` (WARP `roundBase = offset` where
`offset = thread_rank·T + betweenIV`), `lower_cache.go:249-264` (buffer extent = product of
covered tile sizes), `:324-330` (variant rule); spec §6 variant table + §6bis PHASE 4.

**Evidence (traced matmul `rhs`, which the real judge classifies SMEM with covered {N=SoCWARP, K}):**
buffer extent = `4·T_n·T_k = 4096` B; fill variant = LoadTileCta (any covered dim group-parallel);
the fill's N round base = `N.offset = thread_rank·32 + round` — **per-thread**. The access reads
`buffer[N.insideIV, K.insideIV]` where `N.insideIV ∈ [0,32)` for every thread.

**Why it is wrong:** a cooperative `LoadTileCta` requires CTA-uniform source coordinates
(§4/§7bis uniform-call rule); `thread_rank·T + round` differs per thread, so each participating
thread computes a different source region into the *same* 32×32 SMEM buffer (racy, incoherent),
and 256 threads each expecting "their own" 32-chunk read the same 32 slots. Under the
alternative reading (fill base = the round base alone), the loader would cover the group's
`nThreads·T` window but the buffer is sized `T` — underallocated by 8×. Either way the
doc's own worked example (rhs tie nil covering [N][K] with N WARP-assigned) cannot execute as
lowered. `TestLowerMatmulEndToEnd` pins only the variant *flag* (1 vs 2), not coherence.

**Fix:** for covered dims with SoCWARP, either (a) size the SMEM buffer by the group's round
coverage (`LoC·32·T` for that dim) and use the thread-uniform round base, with the loader
enumerating the group window, or (b) demote such sites to per-thread FRAG/`LoadTileThread`
(the coherent per-thread-window reading the access code already assumes). Update the judge or
document the ruling change in the spec.

### HI-04: `WarpPortion == PortionWarp` with a CTA dim yields a zero-stride (infinite) inside loop

**File:** `internal/opt/tiling/record.go:269-271` (Validate allows `[0, PortionWarp]`),
`internal/opt/tiling/search.go:148-156` (independent warp takes the full 8 portions),
`internal/opt/tiling/lower_rewrite.go:204-215,307-309`:

```go
return (PortionWarp - warpPortion) * quantum * 32   // warpPortion=8 ⇒ 0
...
d.insideIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(float64(coupled)))  // step 0
```

**Why it is wrong:** an explicit `WarpPortion = 8` (legal per `Params.Validate`) plus any CTA
dim leaves **zero** warp portions for the CTA dim's coupled inside walk. Nothing in decide,
tiling-verify (its warp check `verify.go:174-179` only bounds the *independent* LoC), or
tile-lower rejects this; the emitted ForLoop has step 0 and a `loc=0` annotation — an infinite
loop / divide-by-zero trip count in any downstream materialization, and no verifier invariant
forbids a zero step. The record's own comment (`record.go:183-188`) asserts "the coupled
tile-inside walk implicitly consumes the remaining 4, >= 1" — the `>= 1` is never enforced.

**Fix:** in `Params.Validate` (or `evalSchedule`), reject `WarpPortion` configurations that
leave the coupled walk under 1 portion when the LNT carries a CTA dim (`WarpPortion` must be
≤ `PortionWarp − 1` in that case); defensively error in `generateDimExprs` when
`coupled < 1`, and add an ir-verifier invariant that a ForLoop dimvar step is a positive const.

### HI-05: Branched LNTs are silently rewritten as a linear nest (sibling loops become nested)

**File:** `internal/opt/tiling/lower_rewrite.go:93-129` (`rewriteLNT`), `:344-369` (`buildNest`
nests `dims` linearly), `internal/opt/tiling/overlap.go:364-378` (`treeItersOf` DFS preorder);
single-body design at `:376-431` (`bodyRootOf`).

**Why it is wrong:** `loopforest.Tree.Children` is a slice — sibling iters under one parent are
a first-class part of the model (the searcher even plans for them: "Branched siblings stay
sequential", search.go:10-12). `buildNest` takes `treeItersOf` (DFS preorder: parent, child1,
child1's subtree, child2, …) and nests **all** of them into one chain, so sibling loops B1, B2
under A become `A.between{ B1.between{ B2.between{ …body… }}}` — the body executes over the
B1×B2 cross product instead of B1-then-B2, and B2's original body ops (not reachable from the
single `bodyRoot`) end up outside the nest while their IVs were RAUW'd to index expressions of
loops they are not inside. No check anywhere requires the tree to be a chain, and no test
covers a branched LNT.

**Fix:** fail closed in `rewriteLNT` when any tree node has >1 child ("branched LNT not
supported by the v0 rewrite"), or implement per-branch sub-nests. Add a branched fixture test
pinning the chosen behavior.

### HI-06: SMEM budget scope is inconsistent — decide gates per-LNT, tiling-verify and resolveBudget gate kernel-wide

**File:** `internal/opt/tiling/search.go:168-183` (per-LNT: `for _, sf := range lf.CS … if smem > p.MaxCache`),
`internal/opt/tiling/verify.go:199-242` (`smem := 0; for _, lf := range all { … smem += … }`),
`internal/opt/tiling/lower_cache.go:76-88` (kernel-wide `smemSum`).

**Why it is wrong:** with two LNTs each deciding ≤ `MaxCache` of SMEM sites (perfectly legal
per the decide gate — each LNT is searched in isolation), the kernel-wide sum can exceed
`MaxCache`. Then tiling-verify (115) **errors** on a record the decide pass just published
(`recomputed SMEM footprint > MaxCache`), aborting the pipeline on a legal graph; if verify
were skipped, tile-lower's `resolveBudget` would silently halve tiles the record chose.
Whichever scope is intended (§6's "Σ SMEM bytes ≤ Max.cache" reads kernel-wide), the two
halves must agree; today they do not.

**Fix:** pick one scope. Kernel-wide: make `evalSchedule`'s feasibility accumulate across
LNTs (thread the running sum through `searchSchedule`'s caller) or re-gate after all LNTs are
decided. Per-LNT: change `verify.go` and `resolveBudget` to gate per-LNT. Add a two-LNT
SMEM-carrying fixture pinning the chosen semantics.

## Medium Issues

### ME-01: Budget fallback can strand a promoted site at a covered tile of 1 → pass errors ("selection never fails" violated)

**File:** `internal/opt/tiling/lower_cache.go:110-118,133-151` (drop only when **all** covered
tiles are 1) vs `:233-237` (`lowerSite` errors when **any** covered dim has tile 1) and the
judge gate (`search.go:221-228`).

**Why it is wrong:** halving only touches the *picked* site's covered iters. With non-nested
covered sets (possible on branched LNTs, e.g. sites {C}, {B2}, {C,B2} under root A with
children B1→C and B2), the pick's halving can drive one shared iter of a *different* site to 1
while that site's other covered dim stays >1. The all-one drop sweep leaves it promoted, the
sum drops under budget, the loop exits, and `lowerSite` then fails closed — the pass errors on
input the fallback was supposed to make feasible. §6: "Fallback (selection never fails)".
Also note the code comment quotes the rule as "when **a tile size** reaches 1 drop that site"
while implementing the all-one variant.

**Fix:** drop a site when **any** covered iter reaches tile 1 (matching the judge's promotion
gate and the quoted rule); add a fallback test with non-nested covered sets.

### ME-02: §6's per-thread FRAG byte budget is unimplemented

**File:** `internal/opt/tiling/lower_cache.go:59-154` (only `ClassSMEM` sites enter
`smemSum`); spec §6 "FRAG sites additionally require … a per-thread FRAG byte budget
(× threads/CTA) beside Max.cache" and "the covered tile dimensions fully unroll".

**Why it is wrong:** nothing bounds FRAG register pressure: a FRAG site's bytes never enter
any budget, so the fallback never halves or drops FRAG sites, and unbounded register arrays
(which §5.8 of doc/tca.md says spill to local memory, "strictly worse than the GMEM load it
replaced") can be emitted. The "fully unroll" half is proxied by the covered-sequential
re-assert (`checkFragSites`), which is reasonable; the byte budget half is simply absent.

**Fix:** add `MaxFragPerThread` (or reuse `MaxCache` scaled by threads) to the budget
resolution: `Σ frag_bytes ≤ budget`, participating in the same halve/drop loop.

### ME-03: §7 cross-LNT rules unimplemented — no union evaluation, silent foreign-iter drop, rewrite leakage, no explicit non-candidate declaration

**File:** `internal/opt/tiling/overlap.go:17-23` (self-declared "LNT ownership"
interpretation), `:186-216` (ownership pass), `:287-295` (`AxesIters` filtered by `memberOf` —
foreign iters silently discarded); `internal/opt/tiling/lower_rewrite.go:419-421` (the `claimed`
guard only catches a node chosen as *best* twice).

**Why it is wrong:** §7 mandates "a load whose INS spans multiple LNTs is evaluated against the
UNION of the referenced iters' assignments" and "Never-bucketed loads are declared
non-candidates (explicit, not silent)". The implementation assigns such a load to one tree and
drops the foreign iters from `AxesIters` entirely — the judge never sees them (no fail-closed:
a mixed axis is judged as if it referenced only owning-tree iters). Worse for the lower half:
when the owning LNT is rewritten, the load's foreign-iter references were RAUW'd to the
*other* LNT's new loop dimvars, so a node inside LNT1's nest references LNT2's loop variables —
scope-ill-formed output, unchecked by any invariant. Loads with empty INS are skipped without
any "declared non-candidate" record.

**Fix:** at minimum, fail closed: detect loads whose INS spans ≥2 trees and either implement
the union judge or mark them non-candidates **in the record** (explicit entry) and refuse to
rewrite bodies referencing foreign iters. Long term, implement §7's union semantics.

### ME-04: §8 self-checks are weaker than claimed — `checkNoBarriers` vacuous, fill layering data-only

**File:** `internal/opt/tiling/lower.go:364-371` (`checkNoBarriers` tests `fill.annotated`,
which is hard-coded `true` at `lower_cache.go:402` — can never fail), `:302-337`
(`checkFillLayering` inspects only the fill source's axis subtree, never the fill's structural
position — which is undefined anyway per CR-02).

**Why it is wrong:** §8's exit conditions "Every fill sits OUTSIDE bailing element loops" and
"ZERO barriers emitted" are listed as pass exit conditions, but the implemented checks cannot
fail on any graph this pass produces: the barrier check verifies a constant, and the layering
check would pass even for a fill wired as an inside-loop body (HI-02). `TestSelfCheckInjections`
proves the checks fire only on synthetic hand-built `lowerRun` states, not on pass output.

**Fix:** make `checkNoBarriers` actually scan the emitted fills' operand lists for a live
`loadtile` attribute (parse it back from the node), and make the layering check structural:
assert each fill is reachable from its covered between-loop's body operand and not from any
inside-loop's body operand.

### ME-05: Test-honesty: the "matmul e2e" pin is IR-level only; no execution, no materialize/print round-trip

**File:** `internal/opt/tiling/lower_test.go:710-774` (`TestLowerMatmulEndToEnd`), `:1007-1028`
(`TestDecideLowerComposition`); commit 4da1ea2's message ("matmul e2e").

**Why it matters:** the pinned "e2e" asserts node shapes: loads erased, buffers with extents
128/4096, two fills with variant flags, verifier + typeinfer clean. It never materializes,
prints, or executes the graph (there is no codegen or runtime in-tree yet), and it asserts
nothing about fill *position* (CR-02), tail rounds (CR-01), or the WARP-site coherence (HI-03) —
the three ways this structure is currently wrong. The commit message's "e2e" label overstates
what is pinned. `print.go` rendering of a lowered graph (e.g. the `loadtile` attribute inside a
store's bracket list, LO-01) is also untested — no golden covers a tile-lowered print.

**Fix:** rename/re-scope the claim (e.g. "matmul lowering composition"), and add: (1) a
Materialize+print round-trip test of a lowered graph, (2) a tail-round structural test once
CR-01 is fixed, (3) a fill-position assertion once CR-02 is fixed.

## Low Issues

### LO-01: `loadtile` KeyedAttribute rides the fill store's biv operand list

**File:** `internal/opt/tiling/lower_cache.go:335-341`; `internal/ir/node_lir_memory.go:140-148`
(no shape pin on store trailing operands, unlike ForLoop's M1 pin at `verify.go:245-254`);
`internal/ir/print.go:610-621` (`bufferAccess` renders every trailing operand as `[biv]` — the
attribute prints as a bracket subscript); `lower.go:275-292` (`loopAnnotations` starts at
operand index 3, correct for ForLoop, works for the store only because every fill has ≥1 round
base).

**Fix:** either give MemStoreSMem/FragStore their own trailing-attribute convention with a
verifier pin (mirroring M1) or carry the variant as node payload; make `loopAnnotations`
start at the kind-appropriate index.

### LO-02: §5's "GVN-CSE of the intrinsics is sound and desirable" is wrong across LNTs

**File:** `internal/ir/node_intrinsic.go:16-19`; `internal/ir/value.go:272-277,322-324`.

Two LNTs querying `IntrinsicGridMap(0, "index")` compare ValueEqual (same kind, same field,
same operand identity) once their DoSoC consts are CSE'd — but per §2/§7bis they bind to
**different per-LNT map instances** with different dimensionalities. Latent today only because
no GVN pass runs after tile-lower and each `lr.u32c` allocates a distinct const. Fix: mix an
LNT discriminator into the node (or forbid CSE of these kinds) and correct the spec claim.

### LO-03: tile-lower consumes per-iter decisions without revalidating them

**File:** `internal/opt/tiling/lower.go:133-144` — decisions are resolved fail-closed for
*presence* only; the feasibility re-checks (CTA LoC == MaxCTA, warp portion multiples, v0
policy) live exclusively in tiling-verify, which tile-lower does not `Require`. A corrupted
record (e.g. CTA LoC 50) flows straight into strides/annotations. Fix: require `tiling-verify`
or re-run the cheap per-iter feasibility checks in PHASE 1.

### LO-04: Legality window for the new kinds is nominal

**File:** `internal/ir/catalog.go:68-69`; `internal/opt/legality.go:76-91` — StageConstructed
marks every catalog kind Legal, so the §5 companion-5 "exists only between this pass and
codegen" window is a doc note, not a check (self-acknowledged "revision point"). Revisit when
stage windows land.

### LO-05: u32 extent typing is assumed, not checked

**File:** `internal/opt/tiling/lower_rewrite.go:188-197,231-234`. `extentOfBiv` accepts any
operand; with an f32-typed BIV max (exactly what the decide-half fixture `newChain` builds),
the emitted `sub`/`umin` are ill-typed and only the pipeline's later typeinfer re-run catches
it (`newChain32`'s comment admits the assumption). Fix: check `ir.TypeOf(extent)` is u32 (or
emit a cast) and fail closed with a named error.

### LO-06: Coupled warp under-exhaustion when no independent warp dim exists

**File:** `internal/opt/tiling/search.go:148-156`, `lower_rewrite.go:204-215`. With
`WarpPortion` auto and a CTA dim but no independent warp dim, the coupled walk still gets only
`AutoWarpPortion`=4 portions (256 of 512 threads); "warp-assigned dims' portions … SUM to
PORTION_WARP" is not achieved and half the warp space idles. Consistent between halves, so not
a correctness bug — a searcher-policy gap worth a ruling.

### LO-07: Dead code — post-RAUW chain re-point loop can never fire

**File:** `internal/opt/tiling/lower_cache.go:386-390`:

```go
if err := lr.mctx.ReplaceAllUsesWith(sf.Load, access); err != nil { ... }
for _, u := range ir.UsesOf(sf.Load) {   // RAUW already moved every live use ⇒ always empty
    if err := lr.mctx.SetDef(u, src); err != nil { ... }
}
```

`ReplaceAllUsesWith` moves **all** live ref records (including the load's incoming ImpactBy
chain edges — they now point at `access`, not `src`), so the loop body is unreachable and the
comment's "the fill source takes the load's chain position" never happens (successors on the
same tensor chain after the buffer read instead of the fill's GMEM read — order-safe only by
transitivity through the buffer chain). Delete the loop or re-point those edges *before* the
RAUW.

## Verified by test (strengths — these claims are pinned and pass)

- Decide half: D2 no-mutation (`TestRunMutatesNothing`), record publication + name, determinism/
  idempotence, innermost-K window defaults, portion completion (CTA exhausts MaxCTA; auto warp =
  4×quantum; explicit override), v0 one-CTA/one-WARP policy filter, promotion judge cases
  (CTA-at-last blocks; last-two-seq → FRAG; warp → SMEM; covered tile 1 blocks), single-term
  traffic objective with an independent full-enumeration optimality cross-check, tie/covered
  derivation on the doc's worked example (lhs tie iterN covering {K}; rhs nil covering {N,K}),
  tiling-verify's planted-corruption classes, record survival across both passes' invalidation
  (`Preserved`), registry order 114/115 and tile-lower@116 requiring tiling.
- Lower half: fail-closed record gates (missing/wrong-typed/nil), DoSoC numbering with per-LNT
  restart and distinct coupled/independent ThreadMap dims, all three canonical T>1 forms'
  strides (3200/256/8192/32/1), `umin`-shaped inside bounds, T==1 single-loop skip with direct
  IV mapping, M1 annotations (soc/dosoc/loc, no dosoc when seq), RAUW retirement of every
  Iter/IV (zero survivors), verifier cleanliness + end-node reachability (P6), pass determinism
  (byte-identical renders), typeinfer re-run over all products, budget fallback no-op/halving/
  drop-at-1 (nested-coverage case), §6 loads-only scope, IR prerequisites (intrinsic shape
  checks incl. const-u32 DoSoC and field vocabulary, value-numbering of the field payload,
  u32 umin/umax registry entries, ForLoop M1 verifier pin, print cases for every new kind).
- Pipeline wiring matches doc/opt.md: tiling 114 → tiling-verify 115 → tile-lower 116 →
  schedule 900; `opt/all` imports `internal/opt/tiling` (all three passes register from one
  import).

## Untested claims (asserted by comments/spec, not pinned)

- Tail-round behavior of ANY parallel dim (no fixture with extent not a multiple of the round
  stride) — CR-01.
- Fill structural position / per-round refill — CR-02, HI-01, HI-02.
- WARP-covered SMEM site coherence — HI-03.
- `WarpPortion = PortionWarp` path — HI-04.
- Branched LNTs — HI-05.
- Multi-LNT SMEM accounting — HI-06.
- Cross-LNT loads — ME-03.
- Any execution of lowered code (no codegen/runtime exists in-tree yet).

## Overall sanity verdict

**Decide half (tiling@114 / tiling-verify@115): fundamentally sound.** The overlap analysis,
tie/covered derivation, portion-grammar completion, feasibility gates, single-term objective
and record plumbing faithfully implement loop_tiling.md's final model and are honestly tested
(including an independent optimality cross-check). The real defects are boundary-scope issues:
per-LNT vs kernel-wide SMEM budget (HI-06), unimplemented §7 cross-LNT union semantics (ME-03),
and the un-enforced `coupled >= 1` portion floor (HI-04, shared with the lower half).

**Lower half (tile-lower@116): the loop-rewrite machinery (PHASE 2-3) is solid; the cache
insertion (PHASE 4) is not shippable.** DoSoC numbering, the three canonical loop forms,
T==1 skip, annotations, and the RAUW/retire choke-point discipline match the templates and are
well pinned. But the pass's headline deliverable — promoted loads lowering into per-round
refilled SMEM/FRAG caches — is structurally unimplemented (CR-02: fills float outside every
loop), its coordinate arithmetic is scope-inconsistent (HI-01/HI-02), its flagship WARP-site
case is incoherent (HI-03), and the tail-guard arithmetic every parallel dim depends on
wraps in u32 (CR-01, OOB accesses for essentially every real extent). Combined with the
silent mis-lowering of branched nests (HI-05), tile-lower currently produces verifier-clean
graphs that cannot execute correctly once codegen lands. The green test suite passes because
it pins node shapes, not semantics. Both CRITICALs and HI-01..HI-05 should be fixed (or the
affected inputs rejected fail-closed) before this pass feeds anything downstream.

---

_Reviewer: Claude (gsd-code-reviewer, deep pass)_
_Method: spec-first read, execution-order walk, one full decide→lower matmul hand-trace; claims verified by reading code, not comments. `go build ./...`, `go vet`, `go test` (tiling/ir/typeinfer): green._
