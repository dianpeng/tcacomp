---
phase: quick-261101-kel
plan: 01
type: execute
wave: 1
depends_on: []
tags: [codegen, fix7, slice-1a, streaming-register-block, frag-chunking, gpu-validation]
files_modified:
  - internal/opt/tiling/lower_cache.go
  - internal/opt/tiling/lower_rewrite.go
  - internal/cg/cxx/fragchunk_test.go
  - internal/cg/cxx/testdata/fragchunk_stream.cu
  - internal/cg/cxx/testdata/fragchunk_const_extent.cu
  - internal/e2e/fix7stream_test.go
  - design/fix/fix7.md
  - .planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/reuse-audit.txt
  - .planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/before-bench.txt
  - .planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/after-bench.txt
autonomous: true
requirements: [QUICK-261101-KEL]

must_haves:
  truths:
    - "A promoted FRAG site whose raw per-thread window exceeds its priced block footprint emits a Frag declaration sized to the CHUNK (true const extents, budget-clamped to j.bytes) — never the raw covered window; the 256 KB/thread local-memory decl (Frag<float, 2u, 256u, 256u>) is gone from every suite emitter"
    - "For a chunk-clamped covered dim, the tile-between round advances by the chunk (SeqIterator step = chunk, replacing step = T), the inside walk clamps to the chunk via the existing umin chain (clamp target changed from nominal T to the chunk), and the LoadTileThread fill stays at the tile-between level — one refill per round — with the round bases (c8n D-fix6-03 anchors) advancing per chunk automatically"
    - "A covered COMPILE-TIME-CONST dim sizes its chunk by TRUE extent (e4 -> 4, not the nominal T=256) — pathology 2 dead by construction; the const extent is read from the BIV max operand's const value / the canonical e<extent> dim-param spelling (the c8n identity source), NEVER from ir.TypeOf (nil computed cell post-typeinfer)"
    - "judge.go, search.go, record.go pricing, the search budget gate (search.go:231), and the schedule pass are UNTOUCHED — the chunk computation READS siteWindowBytes (the ONE shared pricing source) and changes only the lower's window-factor realization + the cxx decl that prints the same buffer node"
    - "Every v62/c8n pin stays green UNWEAKENED: TestFix5T2OneTierCollapseEmission, the fragwindow suite, TestFillCoordReconstruction (all modes), the r3 pin family; all pre-existing goldens byte-stable except the new pinned stream goldens"
    - "RESTORATION GATE: row_sum < 459 ms and softmax < 59.3 ms (pre-merge numbers); IMPROVEMENT GATE: dup_expr/dup_const/axpy3/scale_copy strictly better than 787/804/1351/1397 ms — suite 14/14 at --jobs 1 after scripts/build.sh rebuild, canary 0, maxdiff within suite.toml tolerances"
    - "Memcheck is clean at FULL suite dims on every schedule-changed kernel INCLUDING softmax (the 256 KB/thread local array must disappear; the c8n reduced-dims protocol is retired — a lingering full-dims OOM is a hard failure, not a workaround); row_sum re-run manually against the cached fixture binary without the orchestrator 300 s cap (c8n precedent)"
    - "The §3.4 priced-vs-actual reuse audit (LoadTileThread fill-site count per nest path on the scale_copy/row_sum emitters vs the judge's priced reuse) is recorded in reuse-audit.txt; any actual < priced discrepancy is recorded as a placement-gap follow-up — never patched silently, never fixed by a judge tweak"
  artifacts:
    - path: "internal/cg/cxx/fragchunk_test.go"
      provides: "the §3.5 pins 1+2: crafted-record fill fixtures (fillcoord_test.go idiom) pinning chunked Frag decl dims + stream-loop structure + per-round window anchors (byte goldens)"
      min_lines: 80
    - path: "internal/cg/cxx/testdata/fragchunk_stream.cu"
      provides: "byte golden: the pure-dynamic budget-clamp case (chunked decl, chunk-stepped between rounds, per-chunk anchors)"
    - path: "internal/cg/cxx/testdata/fragchunk_const_extent.cu"
      provides: "byte golden: the const-true-extent case (e4 chunks to 4, not nominal 256) — pathology 2's regression test"
    - path: "internal/e2e/fix7stream_test.go"
      provides: "pin 3: scale_copy + row_sum emitted through the real optimizer — chunked Frag, no 256 KB decl, tensor-dim coordinate order (c8n semantics), LoadTileThread site counts per nest path"
    - path: "internal/opt/tiling/lower_cache.go"
      provides: "the chunk-dim computation in lowerSite: chunk = covered extents (true const extents) clamped to the site's priced block footprint via siteWindowBytes; the FRAG buffer's InternTensor dims ARE the chunk (one source of truth for the cxx decl)"
    - path: "internal/opt/tiling/lower_rewrite.go"
      provides: "the emission realization: between IV step = chunk for chunk-clamped covered dims; tileMax umin clamp target = chunk (schedule tile d.tile untouched — GATE 3 still checks the record tile)"
    - path: ".planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/reuse-audit.txt"
      provides: "the §3.4 priced-vs-actual reuse record per suite emitter nest path"
    - path: ".planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/after-bench.txt"
      provides: "the honest --jobs 1 --bench-iters 20 after-table (before = c8n's 2026-10-31 after-bench.txt numbers)"
  key_links:
    - from: "internal/opt/tiling/lower_cache.go lowerSite (buffer dims, ~lines 394-420)"
      to: "internal/cg/cxx/walk.go:130-133 (Frag decl)"
      via: "the buffer node's InternTensor type dims — walk.go prints them verbatim, so the decl needs NO cxx change; the buffer type is the ONE source of truth (§3.2: no divergence)"
      pattern: "Frag<.*%du, %s>"
    - from: "internal/opt/tiling/lower_rewrite.go generateDimExprs (~line 394: d.betweenIV = NewDimVar(dim, 0, t); ~line 338-341: tileMax = umin(T_d, rem))"
      to: "the chunk per covered dimRewrite"
      via: "the emitted between step and the inside-walk clamp target both read the chunk (captured on the dimRewrite by lowerSite), NOT d.tile, for chunk-clamped FRAG dims; SMEM-class and under-budget FRAG dims keep step/clamp = d.tile (byte-stable)"
      pattern: "SeqIterator\\(%s, %s, %s\\)"
    - from: "internal/opt/tiling/judge.go siteWindowBytes (line 251)"
      to: "the chunk computation"
      via: "READ as the budget (j.bytes = min(raw, MaxFragPerThread)) — the ONE shared pricing source; the chunk clamp loop targets sizeof(chunk) <= j.bytes; the judge/search/record files themselves are never modified"
      pattern: "siteWindowBytes\\("
    - from: "internal/cg/cxx/memory.go stmtFill (~line 207 LoadTileThread)"
      to: "runtime/tca/loader.h detail::load_tile_rows (~line 128)"
      via: "the fill enumerates the BUFFER TYPE dims (now the chunk) anchored at the round bases riding d.betweenIV — the loader's existing umin(chunk, extent-base) tail clamp is the runtime contract (§3.2: NO runtime change, NO new loader machinery)"
      pattern: "LoadTileThread\\("
---

<objective>
Land fix7 §3 slice 1a: streaming register-block lowering — the emitted code realizes the priced block. A FRAG buffer's peak live bytes become its chunk; the covered window streams through the chunk in tile-between-round-sized pieces; the existing tile-between rounds ARE the chunk loops.

Purpose: today codegen sizes a promoted FRAG buffer to the RAW covered window (`Frag<float, 2u, 256u, 256u>` = 256 KB/thread, dynamically indexed -> nvcc spills the whole array to local memory/DRAM). The v62 pricing already pays only min(raw, MaxFragPerThread) per site; 1a makes the allocation, the round structure, and the inside clamps honor that contract. This is the fix6 §3 lowering half and the prerequisite for the restoration of row_sum/softmax (currently +331%/+343% vs pre-merge) and for slices 1b/1c.

Output: chunk-dim computation in tile-lower (one source of truth = the buffer type), chunk-stepped between rounds + chunk-clamped inside walks, the three §3.5 RED pins in order (unit fill fixture -> const-extent chunking -> e2e emission on scale_copy + row_sum), the §3.4 reuse audit, and the full §9 GPU verification with the §3.6 gates.

HARD SCOPE GUARD (§3.3, non-negotiable): NO judge/search/pricing/record changes (search.go:231 budget gate DO NOT CHANGE); NO schedule-pass changes (a §3.4-confirmed fill placement gap is a SEPARATE follow-up task); NO SMEM/cp.async/swizzle (§6); NO float4 effect-chain work (§5); NO bracket amortization (§4); NO runtime changes (loader.h/view.h stay untouched — the loader's tail clamp is the existing contract).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@design/fix/fix7.md
@CLAUDE.md
@.planning/STATE.md
@.planning/quick/261031-c8n-fix-d-fix6-03-stmtfill-const-dim-coordin/261031-c8n-SUMMARY.md
@.planning/quick/261030-v62-implement-fix6-3-minimal-slice-honest-fr/261030-v62-SUMMARY.md
@internal/opt/tiling/lower_cache.go
@internal/opt/tiling/lower_rewrite.go
@internal/opt/tiling/judge.go
@internal/cg/cxx/walk.go
@internal/cg/cxx/fillcoord_test.go

<!-- Load-bearing gotchas (from project memory + the spec): -->
<!-- 1. Post-typeinfer nodes have a NIL computed type cell: ir.TypeOf is nil for
     tile-lower-produced nodes. Const extents come from ConstTypeOf / the BIV
     max operand's const payload / the dim param's canonical `e<extent>`
     spelling (build.go dimKey — the c8n identity source). NEVER TypeOf. -->
<!-- 2. bin/tcacomp is a STALE gitignored prebuilt: ALWAYS scripts/build.sh
     before any GPU run. -->
<!-- 3. Benchmarks MUST use --jobs 1 (parallel fixtures contend on the A6000,
     4-6x timing inflation). Benchmark numbers at --jobs > 1 are invalid. -->
<!-- 4. The overlap admission filter rejects a SINGLETON co-traversal set: a
     promoted site needs TWO co-traversed loads (the fillcoord fixture idiom)
     unless a real suite shape is used. -->
<!-- 5. Pin discipline: never weaken a drifted pin without re-derivation
     recorded in the pin's comment. -->

<interfaces>
<!-- Exact contracts the executor uses — extracted from the codebase. -->

From internal/opt/tiling/judge.go:
    // siteWindowBytes is the ONE shared pricing source (fix6 §3): FRAG sites
    // price the register-block footprint min(raw, MaxFragPerThread).
    func siteWindowBytes(sf SiteFacts, iters map[ir.Node]IterDecision, class CacheClass, p Params) int
    func siteWindowBytesRaw(sf SiteFacts, iters map[ir.Node]IterDecision, class CacheClass) int

From internal/opt/tiling/lower_cache.go lowerSite (~lines 379-420) — the change site:
    // Buffer dims are built HERE today as RAW factors (SEQ -> d.tile, WARP ->
    // ceil(d.tile/32) under C1b; SMEM WARP -> d.loc*d.tile), then:
    lr.mctx.SetType(buffer, ir.InternTensor(elem, dims)) // dims []ir.DimRef
    // The header comment at lines 65-67 ("the allocation itself stays the full
    // window") is the pre-1a doctrine THIS PLAN SUPERSEDES for ClassFRAG.
    // siteFill records {siteID, load, src, store, buffer, covered, host, variant}
    // — a per-covered-dim chunk field rides the dimRewrite (lower_rewrite.go)
    // and/or siteFill; the fill's roundBases already ride d.betweenIV.

From internal/opt/tiling/lower_rewrite.go:
    // tileMax — the inside-walk clamp target (the §3.2 "clamp target changes
    // from nominal T to the chunk"):
    func (lr *lowerRun) tileMax(tile int, extent, offset ir.Node) ir.Node // umin(tile, extent - umin(offset, extent))
    // generateDimExprs (~line 394) — the between IV (the §3.2 round step):
    d.betweenIV = lr.g.NewDimVar(lowerOrigin, dim, lr.u32c(0), lr.u32c(t)) // t = float64(d.tile) — the step becomes the chunk for clamped FRAG dims
    // GATE 3 (~line 387) rejects d.tile < 2 — it checks the SCHEDULE tile; the
    // chunk is a lowering realization and may be 1 (a step-1 between loop is
    // legal; the 2026-10-24 ruling forbids tile-1 SCHEDULES, not chunks).
    // extentOfBiv (~line 299) already reads the iter's static extent from the
    // BIV max operand with a ConstTypeOf fallback — the natural const source.
    // dimOfIter + ir.InputParamNameOf give the canonical "e<extent>" spelling
    // (the c8n identity source in internal/cg/cxx/memory.go).

From internal/cg/cxx/walk.go (~lines 121-134) — NO CHANGE NEEDED:
    // The Frag decl prints the buffer type's dims verbatim:
    write(fmt.Sprintf("tca::rt::Frag<%s, %du, %s> %s;", ts, uint(len(dims)), strings.Join(ds, ", "), local))
    // => tile-lower setting the buffer's InternTensor dims to the chunk IS the
    // decl fix (one source of truth, §3.2).

From internal/cg/cxx/fillcoord_test.go (the pin idiom):
    // fillCoordGraph(t, name, mode) builds a frontend-faithful rank-3 graph
    // (mixed mode: ir.DimRef{Extent:4} empty-Name const position, dim param
    // named "e4"); a crafted VERIFIED tiling record (R=CTA, C=SEQ, e4=SEQ,
    // both co-traversed loads promoted FRAG — C1b forces FRAG) is stamped and
    // the real pipeline (tiling.Lower -> finallower -> typeinfer -> schedule
    // as in this file's emit helper) runs to Emit; assertions are a targeted
    // regex PLUS a byte golden under testdata/. COPY THIS IDIOM.

From internal/opt/tiling/record.go:
    Params.MaxFragPerThread = 256 (line 460, the default F6 budget);
    Params.Validate rejects MaxFragPerThread < 1 (lines 528-529).

Suite/verification driver:
    scripts/build.sh                       # rebuild bin/tcacomp FIRST (stale prebuilt gotcha)
    python3.13 testing/orchestrator.py --jobs 1                      # suite correctness (TCACOMP binary from bin/)
    python3.13 testing/orchestrator.py --jobs 1 --sanitizer memcheck --filter <kernel>
    python3.13 testing/orchestrator.py --jobs 1 --bench-iters 20     # benches (single-job only)
    go build ./... && go vet ./... && go test ./... ; go list -m all | wc -l   # hermetic (== 1)
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: RED pin 1 (unit fill fixture: chunked decl + stream structure + anchors) then GREEN the chunk machinery for the dynamic budget clamp</name>
  <files>internal/cg/cxx/fragchunk_test.go, internal/cg/cxx/testdata/fragchunk_stream.cu, internal/opt/tiling/lower_cache.go, internal/opt/tiling/lower_rewrite.go</files>
  <behavior>
    - Pin 1a (decl): a crafted-record FRAG site whose raw per-thread window exceeds its priced block footprint emits tca::rt::Frag with dims = the CHUNK, not the raw window. Fixture: all-dynamic covered dims so the budget clamp is isolated (no const dims) — e.g. covered [C, K] both SEQ with T_C = T_K = 64, float32, MaxFragPerThread = 256 (64 elements): raw = 4096 elements = 16 KB, priced j.bytes = 256. With the policy below the chunk is [1, 64] (outermost covered dim halved to fit: 64->32->16->8->4->2->1, innermost keeps 64) = 256 B exactly. Assert the decl line `tca::rt::Frag<float, 2u, 1u, 64u>` via targeted regex AND byte golden testdata/fragchunk_stream.cu.
    - Pin 1b (stream structure): the clamped dim's tile-between loop advances by the chunk (SeqIterator step 1u for the [1,64] case — the rounds ARE the chunk loops per §3.2's ruling intent), NOT step 64u; the innermost covered dim's inside walk clamps via the existing umin chain with the chunk as the clamp target.
    - Pin 1c (anchors): the per-round window anchors (the fill's TensorWindow coordinate expressions riding the between dimvars, c8n D-fix6-03 semantics) advance with the chunk-stepped between IVs — assert the anchor expression references the clamped dim's between iterator (same idiom as fillcoord_mixed.cu's `TensorWindow(x_v, v9, C_it1.Index(), e4_it4.Index())` assertion).
    - RED evidence: at HEAD the fixture FAILS — the decl prints the raw window (`Frag<float, 2u, 64u, 64u>`) and the between step is 64u. Record the exact failing output in the task dir before implementing.
  </behavior>
  <action>
    RED: create internal/cg/cxx/fragchunk_test.go following the fillcoord_test.go idiom EXACTLY (frontend-faithful DimRefs, two co-traversed loads so the overlap admission filter admits the site, crafted VERIFIED record, real pipeline to Emit, targeted regex + byte golden). Fixture dims: two covered dynamic SEQ dims, T = 64 each, float32, Params.MaxFragPerThread = 256. Record the RED failure, then commit the pin.

    GREEN — the chunk machinery, FRAG class only (SMEM path untouched, byte-stable):

    1. internal/opt/tiling/lower_cache.go lowerSite: for class == ClassFRAG, compute the chunk dims BEFORE building the buffer's dims. Per covered dim in the buffer's (tensor) order, the chunk base = the current raw factor (SEQ -> d.tile; WARP under C1b -> ceil(d.tile/32)) reduced to the dim's TRUE compile-time-const extent when the iter's extent is a const (Task 2 wires the const source; in THIS task handle the dynamic path: chunk base = raw factor). The budget = siteWindowBytes(sf, the covered dims' effective decisions map, ClassFRAG, p.Params) — CALL the shared function, never reimplement the pricing (judge.go/search.go/record.go stay untouched — verify with git diff at the end of the task). CHUNK POLICY (pin it in the code comment AND the pin's re-derivation comment): while sizeof(chunk)*elemBytes > budget, halve the OUTERMOST covered dim whose chunk > 1 (floor 1). Fail closed with a named error if every chunk reaches 1 and it still does not fit (unreachable for elemBytes <= budget; never a silent clamp). Set the buffer's InternTensor dims to the chunk dims — this IS the decl fix via walk.go:130-133 (do NOT touch internal/cg/cxx for the decl). Record the chosen chunk on each covered dimRewrite (new field, e.g. chunk int; zero/absent = use d.tile) and on the siteFill for §8-check/debug visibility. Update the lower_cache.go header comment (lines 65-67) to the 1a doctrine: for ClassFRAG the allocation IS the priced streaming chunk; SMEM windows stay full-size.

    2. internal/opt/tiling/lower_rewrite.go generateDimExprs: where the between IV step is built (~line 394, t := float64(d.tile)) and tileMax's clamp const is supplied (~line 338-341), use the dim's chunk (when recorded and < d.tile) for ClassFRAG covered dims; everything else keeps d.tile. GATE 3 (d.tile < 2) stays as-is — it guards the SCHEDULE tile, and the record tile is unchanged (this is a lowering realization, not a schedule change; §3.2/§3.3). The fill stays at the tile-between level: do NOT move fillStore or its roundBases — the bases ride d.betweenIV, which now advances per chunk (anchors correct by construction; pin 1c proves it).

    3. Do NOT touch: judge.go, search.go (the search.go:231 budget gate especially), record.go pricing, the schedule pass, internal/cg/cxx/{walk,memory}.go, runtime/* (loader.h/view.h need no changes per §3.2 — load_tile_rows already tail-clamps umin(chunk, extent-base) and zero-fill-pads). Do NOT weaken any existing pin. A chunk of 1 rendering a between-loop step of 1u is cosmetic-only for loops.go's comment classification (line 88) — emission correctness does not ride on it; leave loops.go alone.

    Run the full hermetic suite; expect every pre-existing golden byte-stable (under-budget FRAG sites chunk to their raw window; SMEM untouched).
  </action>
  <verify>
    <automated>go test ./internal/cg/cxx/ -run 'TestFragChunkStream|TestFillCoordReconstruction' -count=1 -v &amp;&amp; go test ./internal/opt/tiling/ -count=1 &amp;&amp; go build ./... &amp;&amp; go vet ./... &amp;&amp; test "$(go list -m all | wc -l)" = "1" &amp;&amp; git diff --name-only origin/dev...HEAD -- internal/opt/tiling/judge.go internal/opt/tiling/search.go internal/opt/tiling/record.go | grep -q . &amp;&amp; exit 1 || exit 0</automated>
  </verify>
  <done>Pin 1 RED recorded then GREEN: the byte golden shows the chunked decl (Frag dims = [1, 64], 256 B — not the 16 KB raw window), the between loop steps by the chunk, anchors reference the chunk-stepped iterator; go build/vet/test green across the repo; go list -m all | wc -l == 1; git diff shows judge.go/search.go/record.go untouched; every pre-existing test green unweakened.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: RED pin 2 (const-extent chunking: true extent 4, not nominal T=256) then GREEN the const-extent source</name>
  <files>internal/cg/cxx/fragchunk_test.go, internal/cg/cxx/testdata/fragchunk_const_extent.cu, internal/opt/tiling/lower_cache.go</files>
  <behavior>
    - Pin 2 (pathology 2's regression test): a covered COMPILE-TIME-CONST dim sizes its chunk by TRUE extent. Fixture: the suite's exact shape — covered [C, e4] with e4 a frontend const (ir.DimRef{Extent:4}, empty Name; dim InputParam named "e4" — build.go dimKey spelling), C dynamic, T_C = 64, T_e4 = 256 NOMINAL (the search's nominal tile on a const dim — the pathology), float32, MaxFragPerThread = 256. Expected chunk: e4 = min(256, true 4) = 4; C halved to fit: 64->32->16 (16*4*4 B = 256 B = j.bytes exactly) => chunk [16, 4]. Assert decl `tca::rt::Frag<float, 2u, 16u, 4u>` (regex + byte golden testdata/fragchunk_const_extent.cu), e4's between step 4u, inside clamp target 4 — NOT 256 anywhere in the decl.
    - RED evidence: at the Task 1 state the fixture FAILS — the const dim chunks by nominal T (decl `... 64u, 256u`-class) because Task 1 wired only the dynamic path. Record the exact failing output before implementing.
    - Also pinned: the const extent is read WITHOUT ir.TypeOf (nil computed cell post-typeinfer — the load-bearing memory gotcha); the pin comment records the source precedence.
  </behavior>
  <action>
    RED: add the const-extent subtest + fixture to fragchunk_test.go (same idiom; the const position's DimRef is ir.DimRef{Extent:4} with empty Name and its dim InputParam is "e4", exactly fillCoordGraph's mixed mode). Record the RED failure, commit the pin.

    GREEN: in lowerSite's chunk computation (lower_cache.go), reduce a covered dim's chunk base to its TRUE extent when the iter's extent is compile-time const. Source precedence, in order: (1) the BIV max operand's const value — the LO-05 path extentOfBiv already reads (ConstValueOf / the const payload type); (2) the dim param's canonical `e<extent>` name spelling via dimOfIter + ir.InputParamNameOf (the c8n identity source). NEVER ir.TypeOf on lower-produced nodes (nil computed cell — the gotcha is real and load-bearing). Apply the reduction BEFORE the budget-halving loop (const dims are already their true size; halving then clamps dynamic outer dims first). Re-derive in the pin comment: e4 chunk = 4 (true), C 64->16 by the halving policy = 256 B.

    Then run the ENTIRE v62/c8n pin set and confirm green UNWEAKENED: TestFix5T2OneTierCollapseEmission (internal/e2e/fix5t2_one_tier_test.go), the fragwindow suite, TestSearchRechoiceCollapse, TestR3SearchRechoiceConstK, TestR3FamilyBGuard, TestFillCoordReconstruction (all modes), TestBudgetFallback. If a pin drifts because its fixture now legitimately emits a chunk, do NOT weaken it — re-derive the expectation from the policy and record the re-derivation in the pin's comment (the discipline rule); if the drift signals a mechanism bug, fix the mechanism.
  </action>
  <verify>
    <automated>go test ./internal/cg/cxx/ -run 'TestFragChunkStream' -count=1 -v &amp;&amp; go test ./internal/e2e/ -run 'TestFix5T2OneTierCollapseEmission' -count=1 -v &amp;&amp; go test ./... -count=1 &amp;&amp; go vet ./... &amp;&amp; test "$(go list -m all | wc -l)" = "1"</automated>
  </verify>
  <done>Pin 2 RED recorded then GREEN: the const-extent fixture emits Frag dims [16, 4] with e4's true extent (no 256 anywhere in the decl), full repo test suite green with every v62/c8n pin unweakened, module count 1, all pre-existing goldens byte-stable except the two new pinned stream goldens.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: RED pin 3 (e2e emission on scale_copy + row_sum) then §3.4 reuse audit and the §9 GPU protocol with §3.6 gates</name>
  <files>internal/e2e/fix7stream_test.go, design/fix/fix7.md, .planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/reuse-audit.txt, .planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/before-bench.txt, .planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/after-bench.txt</files>
  <behavior>
    - Pin 3: scale_copy and row_sum emitted through the REAL optimizer (no hand-stamped record — the fix5t2_one_tier_test.go e2e idiom) declare chunked Frag buffers — grep-pin that `Frag<float, 2u, 256u, 256u>` appears NOWHERE in either emitted .cu — with coordinate lists in TENSOR DIM ORDER (c8n semantics unchanged: scale_copy `TensorWindow(z_v, vNN, C_itM.Index(), e4_itK.Index())`; row_sum const-middle order preserved), and the test reports the LoadTileThread fill-site count per nest path (the §3.4 data).
    - RED evidence: at the Task 2 state the 256 KB decl still appears in at least one emitter (or the fill-site structure differs from the priced model) — record what the pre-pin emission actually was.
    - Expected GREEN: no new mechanism — Tasks 1-2 own the machinery. If the pin instead exposes the §3.4 placement gap (fill not at the between level / priced reuse != actual refills), RECORD it — do NOT move fills and do NOT touch the judge (§3.3: separate follow-up task).
  </behavior>
  <action>
    RED: create internal/e2e/fix7stream_test.go emitting scale_copy and row_sum via the real pipeline; assert the chunked decl (targeted regex + absence of the 256 KB decl), tensor-dim coordinate order, and emit the per-nest-path LoadTileThread site counts. Record the RED state, commit.

    GREEN: confirm the pin passes with the Task 1-2 machinery. Then, in order:

    1. §3.4 REUSE AUDIT — write reuse-audit.txt in the task dir: for every schedule-changed suite emitter, count the LoadTileThread fill sites per nest path and the actual refill count per window, and compare against the judge's priced reuse (j.reuse — derive the priced number from the site's record/decision facts via the tiling package's own APIs in the audit test, or record the priced reuse printed by the emission comments/verbose path). Any actual < priced discrepancy is recorded as the placement-gap follow-up (§3.4) — attributed, never patched silently, never fixed by a judge tweak. This audit is a §3.6 deliverable, not optional.

    2. §9 VERIFICATION PROTOCOL (n63/qwo, exact order):
       a. scripts/build.sh FIRST (the gitignored bin/tcacomp prebuilt is stale by design — GPU runs against a stale binary are void).
       b. Hermetic: go build ./... && go vet ./... && go test ./... all exit 0; go list -m all | wc -l == 1; git status/diff shows only this plan's files_modified (judge/search/record/schedule/runtime untouched).
       c. Schedule-change detection: sha256-diff the emitted .cu set against the uai pinned baseline manifests (the c8n SUMMARY's method). Expected: the six const-shaped kernels + layernorm/transpose change (chunked emission); the matmul family (gemm, mlp, mm_small, mm_mid, mm_splitk, qk_score) BIT-IDENTICAL. Record the actual diff set honestly.
       d. Correctness: python3.13 testing/orchestrator.py --jobs 1 -> suite 14/14, canary 0, maxdiff within suite.toml tolerances.
       e. Memcheck on EVERY schedule-changed kernel at FULL suite dims (hard gate): python3.13 testing/orchestrator.py --jobs 1 --sanitizer memcheck --filter <kernel> per kernel. SOFTMAX AT FULL DIMS MUST PASS — the 256 KB/thread local array must disappear; the c8n reduced-dims protocol is retired as a c8n-era consequence (§3.6): a lingering full-dims OOM is a HARD failure signal, not a protocol workaround. row_sum: if the orchestrator's 300 s subprocess cap times out under instrumentation, re-run manually against the cached fixture binary without the cap (c8n precedent) and record both runs.
       f. Benchmarks: python3.13 testing/orchestrator.py --jobs 1 --bench-iters 20 ONLY (parallel fixtures contend 4-6x; --jobs > 1 numbers are invalid). Stage before-bench.txt (copy c8n's 2026-10-31 after-bench.txt with its tree-identity note as the BEFORE reference) and after-bench.txt (the new run, which must also re-pass the suite) in the task dir; build the honest before/after table in the SUMMARY.

    3. GATES (§3.6 + §9.6, recorded in the SUMMARY — violations recorded honestly per the c8n precedent, never hidden):
       - Restoration: row_sum < 459 ms AND softmax < 59.3 ms.
       - Improvement: dup_expr < 787 ms, dup_const < 804 ms, axpy3 < 1351 ms, scale_copy < 1397 ms (strictly better).
       - Global: no suite kernel >5% slower than its best-known-good; matmul family bit-identical.
       - The >=10x elementwise gate vs the 2026-10-29 pins: RECORD the number honestly — it is NOT a 1a gate (§10: it becomes measurable after slice 1b).
       If a gate fails and the §3.4 audit attributes it to the fill-placement gap, state "1a mechanism landed, gates NOT met, placement follow-up required" verbatim in the SUMMARY and fix7.md — do not patch the schedule pass or the judge to force the gate.

    4. Update design/fix/fix7.md: a short §3 status note (pins landed, audit result, gate verdicts, honest numbers). Write the task SUMMARY.md (RED/GREEN commits, hermetic evidence, sha256 diff set, suite/memcheck verdicts, reuse audit, before/after table, gate verdicts).
  </action>
  <verify>
    <automated>go test ./internal/e2e/ -run 'TestFix7StreamEmission' -count=1 -v &amp;&amp; go test ./... -count=1 &amp;&amp; go vet ./... &amp;&amp; test "$(go list -m all | wc -l)" = "1" &amp;&amp; test -s .planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/reuse-audit.txt</automated>
  </verify>
  <done>Pin 3 RED recorded then GREEN; reuse-audit.txt records priced-vs-actual per nest path; scripts/build.sh rebuilt binary passes suite 14/14 at --jobs 1 with canary 0; memcheck clean at FULL dims on every schedule-changed kernel including softmax; the honest --jobs 1 --bench-iters 20 before/after table is staged; the restoration gate (row_sum < 459 ms, softmax < 59.3 ms) and improvement gate (elementwise four strictly better than 787/804/1351/1397 ms) verdicts are recorded; fix7.md §3 status + SUMMARY.md written.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none new) | Compiler-only change: no new external input surface, no network, no new dependencies, no runtime interface change. The emitted CUDA C++ is the product; its correctness is gated by the GPU suite + memcheck below. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-kel-01 | Tampering | internal/cg/cxx/testdata/fragchunk_*.cu goldens | mitigate | Byte goldens only ever updated via the repo golden helper with the re-derivation comment rule (policy + arithmetic recorded in the pin comment); RED must be recorded before any GREEN golden update; never auto-accept a drifted golden |
| T-kel-02 | Tampering | internal/opt/tiling/judge.go, search.go, record.go (out of scope) | mitigate | git diff gate in Task 1 verify: any modification to the pricing files fails the task; the chunk computation READS siteWindowBytes instead of reimplementing it |
| T-kel-SC | Tampering | Go module supply chain | accept+mitigate | No new modules (stdlib only); `go list -m all \| wc -l == 1` is a hard gate in every task's verify |
</threat_model>

<verification>
Overall (after Task 3):
- go build ./... && go vet ./... exit 0; go test ./... green; go list -m all | wc -l == 1.
- git diff vs dev touches ONLY this plan's files_modified (pricing/schedule/runtime files untouched).
- Pins in §3.5 order, all RED-recorded then GREEN: fragchunk_stream.cu (decl+structure+anchors), fragchunk_const_extent.cu (true extent 4), fix7stream e2e (scale_copy+row_sum, no 256 KB decl, tensor-dim order).
- v62/c8n pins green unweakened: TestFix5T2OneTierCollapseEmission, fragwindow suite, TestFillCoordReconstruction, r3 family, TestSearchRechoiceCollapse, TestBudgetFallback.
- scripts/build.sh rebuilt binary: suite 14/14 at --jobs 1, canary 0, maxdiff within suite.toml tolerances; memcheck clean FULL dims on every schedule-changed kernel including softmax.
- reuse-audit.txt records priced-vs-actual reuse per nest path; discrepancies recorded, never patched.
- after-bench.txt staged; restoration + improvement gate verdicts recorded honestly; >=10x number recorded (not a 1a gate).
</verification>

<success_criteria>
- The emitted code realizes the priced block for every promoted FRAG site: decl = chunk (true const extents, budget-clamped via the ONE shared siteWindowBytes), between rounds step by the chunk, inside walks clamp to the chunk, one LoadTileThread fill per round — pathologies 1 and 2 dead by construction.
- The §3.6 restoration gate (row_sum < 459 ms, softmax < 59.3 ms) and improvement gate (elementwise four strictly better than 787/804/1351/1397 ms) are met and recorded, OR any failure is attributed honestly (e.g. the §3.4 placement gap) with the follow-up recorded in the SUMMARY and fix7.md.
- Hermetic + GPU protocol fully green per §9; the honest --jobs 1 before/after table is in the task dir; the matmul family is bit-identical.
</success_criteria>

<output>
Create `.planning/quick/261101-kel-fix7-3-slice-1a-streaming-register-block/261101-kel-SUMMARY.md` when done, with reuse-audit.txt, before-bench.txt, after-bench.txt staged alongside it.
</output>
