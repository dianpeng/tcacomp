---
phase: quick-261024-ger
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - internal/opt/tiling/record.go
  - internal/opt/tiling/search.go
  - internal/opt/tiling/verify.go
  - internal/opt/tiling/lower_rewrite.go
  - internal/opt/tiling/lower.go
  - internal/opt/tiling/lower_cache.go
  - internal/opt/finallower/reducer.go
  - internal/cg/cxx/bracket.go
  - internal/opt/tiling/tiling_test.go
  - internal/opt/tiling/verify_test.go
  - internal/opt/tiling/lower_test.go
  - internal/opt/finallower/finallower_test.go
  - internal/cg/cxx/emit_test.go
  - internal/e2e/reduce3_test.go
  - doc/optana/loop_tiling.md
  - doc/optana/loop_tile_lower.md
  - doc/optana/final_lower.md
  - doc/tca.md
  - doc/runtime_objects.md
autonomous: true
requirements: [QUICK-261024-GER]
must_haves:
  truths:
    - "No tile==1 / single-form branch remains anywhere in compiler source or tests (grep-clean)"
    - "Params.Validate rejects every TileSizes entry < 2 (gate 1)"
    - "tiling-verify rejects any record whose iter carries TileSize < 2 (gate 2)"
    - "tile-lower fails closed on any effective tile < 2, even on a hand-stamped Verified record (gate 3, the backstop)"
    - "The searcher sweeps only Params.TileSizes (all >= 2); an LNT with no feasible tile >= 2 schedule returns an ERROR, never a t=1 fallback"
    - "The retargeted minimal-legal-tile e2e ({CTA,100,2}/{WARP,8,2}) passes numerically with memcheck and racecheck clean"
    - "Both committed goldens (matmul_store.cu, reduce_warp.cu) regenerate byte-identical"
    - "No normative doc still documents a tile-1 disable form; doc/rt.md §4.6 carries the trailing-barrier rider"
  artifacts:
    - path: internal/opt/tiling/search.go
      provides: "tileCandidates without the injected 1; uniform tile applied to every LNT dim; no-tile>=2 => error"
    - path: internal/opt/tiling/lower_rewrite.go
      provides: "mechanical between/inside rewrite for every dim; tile<2 backstop; single* fields and branches deleted"
    - path: internal/opt/finallower/reducer.go
      provides: "home = provenance.Inside unconditionally; zero-between degenerate re-frame; warpSingleIVs machinery deleted"
  key_links:
    - from: internal/opt/tiling/search.go (tileCandidates)
      to: internal/opt/tiling/record.go (IterDecision.TileSize)
      via: "evalSchedule decision map"
      pattern: "TileSize: tile"
    - from: internal/opt/tiling/record.go (Iters)
      to: internal/opt/tiling/verify.go (record gate)
      via: "d.TileSize < 2 rejection"
      pattern: "TileSize %d .*>= 2"
    - from: internal/opt/tiling/lower_rewrite.go (generateDimExprs)
      to: internal/opt/tiling/record.go (DimProvenance)
      via: "Inside always populated for every dim"
      pattern: "Inside:"
    - from: internal/opt/finallower/reducer.go (home resolution)
      to: internal/opt/tiling (Provenance.Dims[].Inside)
      via: "home = prov.Inside, fail-closed on nil"
      pattern: "prov\\.Inside"
    - from: internal/cg/cxx/bracket.go (ScanBracketLaw)
      to: internal/cg/cxx/testdata/*.cu
      via: "tightened law (X carries exactly Y) holds over unchanged golden bytes"
      pattern: "golden"
---

<objective>
Remove the tile-disable (T==1 single-form) path per the author's ruling (2026-10-24): the compiler NEVER generates a non-tiling schedule, enforced fail-closed at every boundary by never allowing tile size 1. The whole special single-loop form in tile-lower is deleted — every dim takes the mechanical between/inside path — together with the dead machinery it spawned in finallower (the HI-02 consumer-move framing, the WARP single-form mirror gate, warpSingleIVs) and the bracket-audit's T==1 acceptance. The searcher loses its t=1 sweep point and its tile-1 fallback (no feasible tile >= 2 is an ERROR). The reducet1 e2e is re-targeted to the minimal legal tile so sweep-endpoint coverage survives at the NEW boundary. Both goldens are T>1 shapes and MUST stay byte-stable. Rider: doc/rt.md §4.6's WarpReducer trailing-barrier amendment.

Purpose: the T==1 single form was the source of four defects (the SEQ-single identity-indexExpr wall, HI-02's scoping bug, the dead betweenBlocks branch, the missing T==1 WARP mirror gate); deletion replaces the whole degenerate-case family with one enforcement rule.
Output: gate-complete compiler + retargeted/rejection-pinned tests + amended docs.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@./CLAUDE.md
@.planning/STATE.md
@internal/opt/tiling/lower_rewrite.go
@internal/opt/tiling/search.go
@internal/opt/tiling/record.go
@internal/opt/tiling/verify.go
@internal/opt/tiling/lower_cache.go
@internal/opt/finallower/reducer.go
@internal/cg/cxx/bracket.go

<enforcement_chain>
The four gates, in pipeline order (all must fail closed on tile < 2):
1. Params.Validate (record.go) — every TileSizes entry >= 2.
2. tiling-verify (verify.go) — every record IterDecision.TileSize >= 2.
3. tile-lower backstop (lower_rewrite.go generateDimExprs) — every effective tile (record decision OR resolveBudget override) >= 2.
4. The searcher (search.go) — tileCandidates returns only Params.TileSizes; searchSchedule returns an ERROR when no tile >= 2 schedule is feasible (the tile-1 fallback is deleted).
</enforcement_chain>

<design_rulings>
Rulings the executor must NOT re-litigate (decided during planning):

R1. Non-window sequential iters (search.go evalSchedule's default `IterDecision{SoC: SoCSequential, LoC: 1, TileSize: 1}`) now take the SWEPT UNIFORM TILE (`TileSize: tile`), not a new constant. The uniform knob applies to every LNT dim; a sequential dim tiled at T is still exhaustive (between steps T, inside bounded by the saturating tileMax). No current fixture has an LNT deeper than WindowK=5, so goldens/e2e are untouched by this.

R2. resolveBudget's halving (lower_cache.go) never writes an override below 2: when halving the picked site's covered dim would compute next < 2, the SITE IS DROPPED to GMEM (added to `dropped`) instead of recording a tile-1 override. The `anyTileOne` sweep's `<= 1` predicate stays as belt-and-braces (it can no longer fire; update its comment to say the halving floor owns the drop).

R3. finallower's `len(betweenBlks) == 0` branch SURVIVES but is re-framed: post-removal `prov.Inside` is ALWAYS populated, so `home != X` always; the empty-between case is the ZERO-BETWEEN-DIMS DEGENERATE of R-A (the reduced dim is the LNT's only dim — X's parent is trivially the innermost output-element scope). Its nil-subs `moveCombineConsumers` call stays (the lexical-scope law is general). What dies is the T==1 framing and the `home := x` fallback.

R4. `warpSingleIVs` + `refuseThreadPartitionedConsumers` (gate (b) in moveCombineConsumers) are DELETED: they key on `prov.Tile == 1`, which is now unconstructible. Gates (a), (c), (d) remain.

R5. doc/review/*.md mentions of T==1 are HISTORICAL review records — do NOT amend them. Amend only the normative docs listed in Task 1.

R6. The e2e/emit fixture identifiers (`redt1`, `ReduceT1`, `reducet1_e2e`, runner strings) KEEP their names (bin-name stability, minimal diff); their comments are rewritten to "the minimal-legal-tile endpoint ({CTA,100,2}/{WARP,8,2})".
</design_rulings>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Delete the single form, enforce tile >= 2 at every gate, remove dead machinery, amend docs</name>
  <files>internal/opt/tiling/record.go, internal/opt/tiling/search.go, internal/opt/tiling/verify.go, internal/opt/tiling/lower_rewrite.go, internal/opt/tiling/lower.go, internal/opt/tiling/lower_cache.go, internal/opt/finallower/reducer.go, internal/cg/cxx/bracket.go, doc/optana/loop_tiling.md, doc/optana/loop_tile_lower.md, doc/optana/final_lower.md, doc/tca.md, doc/runtime_objects.md, doc/rt.md</files>
  <action>
  Work gate-by-gate, then dead machinery, then docs. The package will not compile between sub-steps; finish all Go edits before building.

  A. tiling package — enforcement chain:
  - record.go `Params.Validate`: the TileSizes loop currently rejects `v < 1`; tighten to reject `v < 2` with a message naming the ruling ("tile size 1 is not a legal schedule — the compiler never generates a non-tiling schedule (2026-10-24 ruling); every entry must be >= 2"). Update the `TileSizes` field comment (drop "the uniform sweep also includes t=1 (tiling disabled)") and the Validate doc comment accordingly. Delete the `ReasonTileOne` constant (record.go ~147) and its doc comment.
  - search.go `tileCandidates` (~485): drop the injected `1` — sweep only `p.TileSizes`, ascending deduped; rewrite the comment and the file-header bullet (~13-19) that documents "{1} ∪ TileSizes (t=1 = tiling disabled)" and the `|TileSizes|+1` bound (now `3^K * |TileSizes|`).
  - search.go `searchSchedule` (!found fallback, ~147-165): DELETE the tile-1 fallback (the `fb` SoC vector + `evalSchedule(lf, window, fb, 1, ...)`) and replace with an error return: an LNT with no feasible tile >= 2 schedule under the kernel-wide budgets is a configuration/schedule ERROR, never an un-tiled fallback. Mention the deleted t=1 guarantee in the error text (e.g. "no feasible schedule at any tile >= 2 under the kernel-wide budgets — tiling is never disabled (2026-10-24 ruling)").
  - search.go `evalSchedule` (~181): the non-window sequential default becomes `IterDecision{SoC: SoCSequential, LoC: 1, TileSize: tile}` (design ruling R1) with a comment noting every LNT dim carries the uniform tile.
  - search.go `judgeSite` (~321-326): tighten the fail-closed check from `d.TileSize < 1` to `d.TileSize < 2` (still `ReasonUnknownIter`); DELETE the `d.TileSize == 1` / `ReasonTileOne` gate (~324-326) and the doc bullet at ~291-293 that describes it.
  - verify.go (~190): the record gate becomes `d.LoC < 1 || d.TileSize < 2` with a message stating LoC >= 1 and TileSize >= 2 (the compiler never generates a non-tiling schedule); update the file-header bullet at ~85-87.

  B. tiling package — delete the single form (lower_rewrite.go, lower.go, lower_cache.go):
  - lower_rewrite.go `dimRewrite` struct (~37-67): delete the fields `single`, `singleIV`, `singleIVNode`, `singleBound`, `singleAttrs`; strip the "when T==1"/"single loop" qualifiers from the `between`/`inside`/`betweenIV`/`insideIV` comments.
  - lower_rewrite.go `generateDimExprs` (~385-453): at the top of the per-dim loop add the fail-closed backstop — `d.tile < 2` returns an error naming gate 3 (Params.Validate and tiling-verify are gates 1-2; a hand-stamped Verified record or a bad override dies HERE, never as a non-tiling rewrite). Then DELETE the entire `case d.tile == 1:` branch (~395-453), including the SoCSequential/SoCTA/SoCWARP single-form constructions and the T==1 WARP single-form mirror gate (~419-447). The T>1 mirror gate in the SoCWARP T>1 case (~534-538) STAYS. Rewrite `saturatingRemainder`'s comment (~350-355): its only caller is now `tileMax` (drop the single-form mirror-gate sentence).
  - lower_rewrite.go `buildNest` (~556-581): delete both `d.single` branches — the inside-loop `continue` (~561-563) and the single-loop construction (~571-573); every dim builds between + inside. Update the function comment ("ALL between/single loops" -> "ALL between loops").
  - lower_rewrite.go `recordProvenance` (~171-199): `Inside: d.inside` comment drops the "nil at T==1" note; the CTA re-record condition `d.soc == SoCTA && !d.single` becomes `d.soc == SoCTA`.
  - lower.go `checkFillPlacement` (~361-373): delete the `d.singleIV` ivLoop entry (~370-372).
  - lower_cache.go `resolveBudget` (~186-192): per R2 — when `next := dec(it).TileSize / 2` would be < 2 for a covered iter of the picked site, mark the picked site `dropped` (continue the outer halving loop) instead of recording the override; rewrite the function doc comment (the F4 floor: a site that cannot fit at tile >= 2 is dropped, never demoted to a non-tiling schedule). The `anyTileOne` helper's comment updates to note the halving floor owns the drop (predicate `<= 1` may stay).
  - lower_cache.go lowerSite covered-dim guard (~325-331): `if d.single || d.insideIV == nil` becomes `if d.insideIV == nil` (kept as the corruption guard; update the comment).
  - lower_cache.go `fillHost` (~612-615): the FRAG fill-key path returns `keyDim.inside` with a fail-closed nil check (post-removal every dim has an inside loop; the `keyDim.between` fallback and its "T==1 fill-key dim" comment are deleted). The `hostDim.between` SMEM comment "T > 1 precondition" may stay.
  - lower_cache.go `uniformBaseSubs` (~629-654) and the fillHost doc (~574-576): strike the "T==1 single IV" alternatives from comments (SEQ dims ride their between-round dimvar, full stop).

  C. finallower/reducer.go — dead machinery:
  - Home resolution (~117-120): `home` is `prov.Inside` unconditionally, fail-closed on nil/erased with an error stating every dim is tiled (T >= 2) so a missing Inside is record corruption. Delete the `home := x` fallback.
  - The `len(betweenBlks) == 0` branch (~235-259): KEEP the construction (per R3) but re-comment it as the ZERO-BETWEEN-DIMS DEGENERATE (the reduced dim is the LNT's only dim; X's parent is trivially the innermost output-element scope; one bracket per output element). Keep the nil-subs `moveCombineConsumers` call; rewrite the HI-02 comment (~253-258) to the general lexical-scope law (no T==1 framing).
  - File-header comment: rewrite bullets (1) (~12-15: "empty at T==1" -> "empty when the reduced dim is the LNT's only dim") and the (2.2) T==1 clause (~40-45) to the degenerate framing.
  - `moveCombineConsumers`: DELETE gate (b) — the `refuseThreadPartitionedConsumers` call (~560-562) — plus the functions `refuseThreadPartitionedConsumers` (~614-641) and `warpSingleIVs` (~643-660) (they key on prov.Tile == 1, unconstructible now). Rewrite the doc comment (~499-523): drop "HI-02's T==1 flat twin", renumber the surviving gates (reduced-dim poison, between-IV rebuild when subs != nil — the degenerate passes nil, coordinates already defined at enclosing scopes — and the P-12 enclosure assertion).
  - `betweenBlocks` doc comment (~799-805): keep the empty-vs-nil distinction guarantee, reworded for the degenerate (drop "T==1 flat-bracket"/"HI-02 latent bug" phrasing — one sentence may credit the historical bug as the reason the non-nil guarantee exists).
  - Straggler comments: ~206 ("the T==1 shape nests the producer deeper") and ~492 ("shared gates with the T==1 flat bracket") — reword to the degenerate.

  D. internal/cg/cxx/bracket.go — the mechanical bracket audit:
  - Delete the T==1 accepted form: X (the reduced dim's tile-between loop) ALWAYS carries exactly one directly nested loop (Y, the tile-inside walk) and Reduce ALWAYS sits inside Y at the governed depth — the "X carries 0 nested loops / Reduce directly inside X" acceptance (~127-158, the `want exactly Y (1) or the T==1 single form (0)` message and the T==1 depth error at ~151) is tightened to exactly-Y. Rewrite the header comment (~7-11). The committed goldens are T>1 shapes: ScanBracketLaw must still pass over their UNCHANGED bytes (this is the byte-stability proof).

  E. Docs (normative only; doc/review/*.md is historical — leave it):
  - doc/optana/loop_tiling.md: strike the "if tile_size == 1 ... cache is NOT promoted" note (~431; tile 1 no longer exists — the F4 halving floor at 2 is the only demotion) and the "tile_size == 1 ... loop tiling is disabled" note in the Rewrite section (~599-600).
  - doc/optana/loop_tile_lower.md: §6bis PHASE 3 — delete the "TileSize == 1 -> NO tiling for that dim" rule (~364-365); every dim becomes between-round + inside loops. Reword the F4 drop-rule text (~315) to the floor-2 drop (a covered dim that cannot halve below 2 drops the site to GMEM). Sweep the file for remaining single-form language.
  - doc/optana/final_lower.md: amend the "Untiled path (T == 1 or dim never tiled)" bullet (~152) and the DimProvenance field comments (~213-216, "the single loop when T==1"/"nil at T==1"/"(1 = untiled)").
  - doc/tca.md §5.10: replace the bracket-law T==1 clause (end of point 2, ~726: "The T==1 form is the same law over the single K loop...") with the zero-between degenerate sentence (empty between set: X's parent IS the innermost output-element scope; bracket [Init, X{Y{...Reduce...}}, Combine, consumers] in X's parent). Strike the "When @r is untiled (T = 1)..." bullet (~755) and replace with one sentence: there is no untiled form — tile sizes are always >= 2, and a TileSize below 2 is rejected at Params validation, record verification, and the tile-lower backstop.
  - doc/runtime_objects.md: strike the T==1 rows of the per-dim object tables (~396-401: SEQ/CTA/WARP single forms), the bare `threadmap(d,"index")` WARP-single-loop row (~245), and reword the F4 "tile 1" mention (~554).
  - RIDER doc/rt.md §4.6: VERIFY (do not blindly rewrite) that the WarpReducer Combine bullet documents the full protocol — `__syncthreads()` FIRST, the all-threads ascending slot fold, then a TRAILING `__syncthreads()` before return — with the racecheck-dirty rationale (slot WAR: straggler fold reads vs the next bracket's Init/Reduce writes). As of planning this text is ALREADY PRESENT (§4.6 point 3, "the T-bz9-05 contingency"); if the text is present, make no edit; if any element is missing, amend minimally. The runtime code (runtime/tca/reducer.h, Combine ~144-161) already implements it and is NOT touched.
  </action>
  <verify>
    <automated>cd /home/ps/Workspace/cuda/project/tcacomp && go build ./... && grep -rn "ReasonTileOne\|singleIV\|d\.single\b\|tile == 1\|TileSize: 1\b" internal/opt/tiling/ internal/opt/finallower/reducer.go internal/cg/cxx/bracket.go | grep -v _test.go | wc -l | grep -x 0</automated>
  </verify>
  <done>Package builds; zero non-test references to the single form or ReasonTileOne remain in the three touched packages; all four enforcement gates present (Validate, verify, backstop, searcher-error); docs amended; rt.md rider confirmed present.</done>
</task>

<task type="auto">
  <name>Task 2: Re-target T==1 fixtures to tile 2, pin the new rejections at every gate</name>
  <files>internal/opt/tiling/tiling_test.go, internal/opt/tiling/verify_test.go, internal/opt/tiling/lower_test.go, internal/opt/finallower/finallower_test.go, internal/cg/cxx/emit_test.go, internal/e2e/reduce3_test.go</files>
  <action>
  Tests are re-targeted, not deleted, unless they pin deleted machinery — then they are replaced by rejection pins at the new boundary.

  A. internal/opt/tiling/tiling_test.go:
  - Validate: add a pin that `Params` with any TileSizes entry 1 (or 0) is rejected and the legal set {2,...} passes (extend the existing Validate test if present).
  - Judge: the "covered dim at tile 1 blocks promotion" pin (~418-424) — tile 1 is now INVALID input, not a refusal reason: re-pin that a covered decision at TileSize 1 fails closed (ReasonUnknownIter) and DELETE the ReasonTileOne expectations (the constant is gone).
  - `enumerateBest` (~704-730): the helper mirrors the searcher's sweep including `tiles := []int{1}` (~711) — drop the injected 1 to mirror the new tileCandidates; update its doc comment ("F4 tile-1 gate" bullet).
  - Searcher: add a pin that `searchSchedule` returns an ERROR (not a fallback schedule) when no tile >= 2 vector is feasible — e.g. budget floors (`smemFloor`/`fragFloor`) already at the caps leave every candidate infeasible; assert the error mentions the never-disabled ruling.

  B. internal/opt/tiling/lower_test.go:
  - DELETE `TestRewriteTileOne` (~598-663) and `TestRewriteWarpSingleMirrorGate` (~664-772) — they pin deleted machinery.
  - ADD `TestRewriteTileOneRejected`: a crafted record (Verified-stamped, bypassing tiling-verify) with any dim at TileSize 1 makes `LowerPass.Run` fail with the gate-3 backstop error — the direct pin that a hand-stamped non-tiling record dies at the rewrite.
  - ADD (or extend) a minimal-legal-tile shape pin: the 2-dim chain at `{CTA,100,2}/{SEQ,1,2}` rewrites to FOUR loops (two per dim, the mechanical between/inside nest in dim order) — the sweep-endpoint coverage at the new boundary, unit-level.
  - Re-target surviving fixtures that carry `TileSize: 1` decisions (~227, and any others found by grep) to 2, updating loop-count/nest-shape assertions (one loop per dim becomes two) and the file-header comment (~6, "T==1 and T>1" -> "T >= 2"). The `// T==1: the single-loop form` fixture (~883) and the "K untiled: single loop" comment (~903) go the same way.

  C. internal/opt/tiling/verify_test.go:
  - ADD a pin: a record with any iter at TileSize 1 (all else valid) is rejected by the verify pass with the tile >= 2 message.

  D. internal/opt/finallower/finallower_test.go:
  - Re-target every fixture decision carrying `TileSize: 1` (~198, ~230, ~264, ~458, ~1013) to `TileSize: 2` and update the downstream shape expectations to the between/inside forms (SEQ dims at tile 2 are legal mechanical shapes). Where a re-targeted test becomes a duplicate of an existing T>1 test, delete it instead.
  - `TestReducerBracketT1` (~1381-1432): re-target to the minimal-legal-tile R3 shape — decisions `{CTA,100,2}/{WARP,8,2}` — and re-pin the bracket law for ONE between dim (the fresh element loop takes X's position; Init is the first item of the element loop; X{home{Reduce}} follows; Combine immediately after X's block; the moved consumers follow Combine). The flat-bracket-at-home==X shape is gone; note in the test comment that the zero-between degenerate (single-dim LNT) keeps the flat form per the general law.
  - DELETE the WARP-single refusal probe (~1434+): `warpSingleIVs` is gone; the shape is unconstructible. Update the file-header R-A note (~1197-1207) that describes the T==1 sweep endpoint and the SEQ-single unbuildable variant (the wall is moot — the form is deleted).

  E. internal/cg/cxx/emit_test.go + internal/e2e/reduce3_test.go:
  - `reduceT1Graph` (emit_test.go ~211-244) and `buildReduceT1Fixture` (reduce3_test.go ~264-272): decisions `{CTA,100,1}/{WARP,8,1}` -> `{CTA,100,2}/{WARP,8,2}`. Keep all identifiers (redt1/ReduceT1/reducet1_e2e/runner strings) per R6; rewrite the comments to "the minimal-legal-tile endpoint ({CTA,100,2}/{WARP,8,2})" — the sweep-endpoint coverage now pins the NEW boundary (smallest generatable tile), exercising the 1-element-loop R3 bracket and the T>1 WARP mirror gate at minimal tiles. Strike the "not buildable today / SEQ T==1" note (~214-217, ~1205-1207) — the single form no longer exists.
  - reduce3_test.go file header (~6-12): Shape B description updated accordingly.
  </action>
  <verify>
    <automated>cd /home/ps/Workspace/cuda/project/tcacomp && go test ./internal/opt/... ./internal/cg/... && grep -rn "TileSize: 1\b\|tile 1\b" internal/opt/ internal/cg/ internal/e2e/ --include="*_test.go" | grep -v "rejected\|refuses\|Reject\|invalid\|fail" | wc -l | grep -x 0</automated>
  </verify>
  <done>All hermetic unit tests pass; the only remaining tile-1 mentions in tests are the rejection pins themselves; the minimal-legal-tile shape is pinned at unit and emit level.</done>
</task>

<task type="auto">
  <name>Task 3: Full gates — hermetic suite, goldens byte-stability, CPU runtime layer, GPU e2e + sanitizers</name>
  <files>(no source edits — verification only; a diff in a golden or a sanitizer failure loops back into Tasks 1-2)</files>
  <action>
  Run the complete gate ladder in order; each must be green before the next. Any golden byte diff is EITHER a bug (fix in Task 1/2 scope) OR must be explained in writing in the SUMMARY (the constraint allows a written explanation, but the default expectation is byte-identity — the goldens' decisions are all T>1 and the searcher's argmax is unchanged by removing a strictly-worse t=1 candidate).

  1. Hermetic repo gate: `go build ./... && go vet ./... && go test ./...` — every package, no skips beyond the TCACOMP_E2E gates.
  2. Golden byte-stability: `go test ./internal/cg/cxx/ -run 'Golden|ScanBracketLaw' -v` — TestGoldenMatmulStore and TestGoldenReduceWarp regenerate byte-identical against internal/cg/cxx/testdata/matmul_store.cu and reduce_warp.cu, and ScanBracketLaw proves the TIGHTENED law (X carries exactly Y) over the unchanged golden bytes. Do NOT rewrite golden files to force a pass; a forced diff means the removal leaked into the T>1 path — fix the leak.
  3. CPU runtime layer: rebuild and run the existing CPU test binary (cmake -S runtime -B runtime/tests/build -G Ninja, build, run the tcacomp_cpu_tests executable / ctest from that build dir). No runtime source was touched; this gate proves the layer still builds and passes.
  4. GPU e2e on the A6000: `TCACOMP_E2E=1 go test ./internal/e2e/ -v` — the full suite including the retargeted minimal-legal-tile fixture (reducet1 at {CTA,100,2}/{WARP,8,2}): numerical canary 0 / tolerance pass, and the in-suite compute-sanitizer runs (memcheck + racecheck, already wired into e2e_test.go) report 0 errors — racecheck cleanliness is the decided gate for the 2026-10-24 WarpReducer trailing-barrier protocol.
  5. Straggler sweep (grep discipline close-out): `grep -rn "T==1\|single-form\|singleIV\|d\.single\b\|ReasonTileOne\|TileSize: 1\b\|tile == 1\|tile_size == 1" internal/ cmd/ doc/optana/ doc/tca.md doc/runtime_objects.md doc/rt.md` — every remaining hit is either a rejection-pin test comment, a historical doc/review/ record (excluded from the grep paths above), or a struck-doc remnant to clean. Doc-review historical records are exempt by design ruling R5.
  </action>
  <verify>
    <automated>cd /home/ps/Workspace/cuda/project/tcacomp && go test ./... && go test ./internal/cg/cxx/ -run 'Golden|ScanBracketLaw'</automated>
  </verify>
  <done>Hermetic suite green; both goldens byte-identical (or a written explanation in the SUMMARY); CPU runtime layer green; TCACOMP_E2E=1 suite fully green on the A6000 including the retargeted minimal-tile fixture with memcheck/racecheck clean; straggler sweep clean.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| crafted-record -> tile-lower | Test fixtures and any future API caller can hand-stamp `Verified: true` records bypassing tiling-verify |
| Params -> searcher | External Params (CLI/driver) supply TileSizes |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-ger-01 | Tampering | lower_rewrite.go generateDimExprs | mitigate | tile < 2 fail-closed backstop AFTER the verify stamp — a crafted record dies at the rewrite, never as a silently untiled (semantically divergent) schedule |
| T-ger-02 | Tampering | record.go Params.Validate | mitigate | TileSizes entries < 2 rejected at the earliest boundary |
| T-ger-03 | Repudiation | search.go searchSchedule | mitigate | the no-feasible-tile ERROR is a loud diagnostic, not a silent fallback — determinism/auditability of generated schedules preserved |
| T-ger-04 | DoS | search.go tileCandidates | accept | sweep space shrinks (t=1 point removed); no new enumeration cost |
</threat_model>

<verification>
Task-level automated gates are in each task. Phase-level: the four enforcement gates reject tile < 2 (unit-pinned at Validate, verify, backstop, searcher); goldens byte-identical; full GPU e2e + sanitizers green.
</verification>

<success_criteria>
- The compiler cannot generate, verify, or lower a non-tiling (tile 1) schedule at ANY of the four boundaries — each pinned by a rejection test.
- No single-form code, field, doc rule, or test fixture survives (grep-clean modulo rejection pins and historical review docs).
- reducet1 e2e green at {CTA,100,2}/{WARP,8,2} with memcheck/racecheck clean; reduce3 e2e unchanged and green.
- matmul_store.cu and reduce_warp.cu byte-identical; ScanBracketLaw passes the tightened exactly-Y law.
- Hermetic `go test ./...` and the CPU runtime layer green.
- doc/rt.md §4.6 rider present (barrier-first + trailing barrier + racecheck rationale).
</success_criteria>

<output>
Create `.planning/quick/261024-ger-remove-the-tile-disable-path-per-author-/261024-ger-SUMMARY.md` when done (what was deleted, the four gate pins, golden disposition, e2e/sanitizer results, any written golden-diff explanation if applicable).
</output>
