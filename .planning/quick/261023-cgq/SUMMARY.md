---
status: complete
completed: 2026-10-23
type: quick
---

# SUMMARY — codegen review fixes (quick-261023-cgq)

Commit 319b420. All three review buckets landed:

[1] Bugs: dead const locals skipped (referenced-closure precompute —
    dimvar/annotation operands sealed since their values emit as literals);
    duplicate post-GVN attribute reads folded (emitter text-keyed CSE);
    stale tca_kernel_ symbol in cudaFuncSetAttribute fixed; outer Body
    brace pair flattened.
[2] Nice-to-haves: semantic local names (<dim>_it, <tensor>_tile, kinfo,
    grid_map, reducer — collision-guarded); __restrict__ (structural
    alias rule claimed); __launch_bounds__(512).
[3] Directives: Google-style PascalCase function symbols with KindKernelMeta
    (KindVersion 6) as the frontend naming channel (Prototype-anchored,
    DCE-rooted, validated identifiers, PascalCase fallback); section
    comments synthesized from existing IR facts + the new tile annotation
    (none|sections|verbose, default sections; verbose = PassOrigin stamps).

The reduce golden now reads end-to-end: LNT header shows the tiling
decision per dim, loops are labeled tile-between vs inside-walk, the fill
and reduction bracket carry their runtime contracts.
