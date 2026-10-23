---
status: complete
created: 2026-10-23
type: quick
---

# Codegen review fixes: Google naming, KernelMeta, section comments

Author review of the generated C++: [1] bugs (dead const locals, duplicate
attribute reads, non-Google function names, cosmetic brace noise), [2]
nice-to-haves (semantic local names, restrict, launch_bounds), [3]
directives (Google-style naming; IR instrumentation for kernel/launcher
names; comment instrumentation for reviewers).

## Rulings confirmed (D9/D10)

- <Name> / <Name>Launch / <Kernel>Combine<i> PascalCase symbols
- KindKernelMeta Prototype meta node carries the names
- Section comments (option-gated none|sections|verbose)

## Done (commit 319b420)

1. [x] KindKernelMeta (KindVersion 6) — six companions, Prototype anchor,
       DCE root, print form; schedule anchoring; identifier validation gate
2. [x] Google naming + semantic locals (dim/tensor-derived, collision-guard)
3. [x] tile annotation on ForLoops (tile-lower) + section comments
4. [x] B1 dead-pure-item skip (referenced closure); B2 text-keyed CSE;
       B4 outer-brace flatten; __restrict__ + __launch_bounds__
5. [x] Goldens regenerated; KernelMeta tests; full repo green
