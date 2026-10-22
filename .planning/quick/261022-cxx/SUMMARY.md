---
status: complete
completed: 2026-10-22
type: quick
---

# SUMMARY — implement internal/cg/cxx (CUDA C++ backend)

## What was done

Implemented the codegen backend per doc/cg.md as one coherent pass
(committed d97c86c, 15 files, ~2.3k lines incl. goldens): `internal/cg`
(backend-neutral Options/Result) + `internal/cg/cxx` (9 source files +
tests) emitting the full trio — `<name>.cu` (kernel + extern "C" launcher)
and `CMakeLists.txt`. All five planned tasks landed in the single pass
(the emission pieces interlock; commits stayed atomic per doc/code).

Verified by: golden .cu over the FULL default pipeline matmul (searched
tiling: lhs SMEM promotion, 6-loop CTA→Seq→Thr×2→Seq×2 chain, cooperative
fill, 64KB dynamic-smem pool + opt-in), golden over the tiled warp-reduce
chain (WarpReducer bracket, D7 slot Get in the re-iteration, hoisted
combiner lambda), determinism twins, fail-closed gates. Full repo suite +
vet + gofmt green.

## Implementation discoveries (folded into doc/cg.md §11)

- Iterator ctors carry (init, step) literals + bound expr; Has() is the
  CTA-uniform round walk — guard semantics corrected to tca.md §5.12
  "predicated, not exited" (views/loaders own validity, never branches)
- Loader coordinate merge convention pinned (axes/bivs by tensor dim
  order; FRAG mixed-with-skips fails closed)
- Scheduler anchors lambda subtrees + keyedattributes as items — walk
  skips them (lambda closure marking; annotations emit nothing)
- Dead scheduler-placed locals + duplicate post-GVN attribute reads:
  acceptable v0, pinned

## KNOWN pre-existing defect surfaced (NOT fixed here — needs its own task)

The schedule pass places the cooperative fill AFTER the tile-inside loops
in the between-round body (matmul golden pins it): the emitted C++ reads
the SMEM buffer before the round's fill. The sea's ImpactBy chain is
correct; the AST item placement doesn't realize it. Emitter renders item
order by contract (§1.1) — fixing belongs to the schedule pass and will
churn the matmul golden.

## Follow-ups

1. Schedule-pass fill-placement fix (the defect above)
2. C++ runtime library per doc/cg.md §8 (THE next task)
3. GMEM coalescing + SMEM swizzle + SMEM liveness RA (descriptor/offset-only)
