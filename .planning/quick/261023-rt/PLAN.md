---
status: complete
created: 2026-10-23
type: quick
---

# Final runtime design doc (doc/rt.md) + reducer scratch gap fix

## Done (commit a5f6e64)

1. [x] Gap fix: reducer constructions emit the slot TILE EXTENT as their
       template arg (<T, Slots>, e.g. WarpReducer<float, 32u>) — prescan
       takes the product of slot dims' tile annotations (NOT the dim count,
       which was the first wrong attempt: 1 vs 32); cg.md D11
2. [x] Loader window→axis contract pinned (trailing axes; D12) — the second
   gap found while writing the doc
3. [x] doc/rt.md written: build shape, principles, conformance table,
       per-type device surface + behavioral contracts, host surface, SMEM
       accounting, test strategy, implementation order, future list
