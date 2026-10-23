---
status: complete
created: 2026-10-23
type: quick
---

# One lexical Group per LNT (codegen scoping fix)

## Description

Author diagnosis of the generated-C++ scoping bug: the per-LNT
[prologue Group, LBBN root, epilogue Group] sibling triple makes the
prologue's C++ scope close before the nest opens, stranding declarations.
Fix in the SCHEDULE pass: one lexical Group per LNT embedding the nest
([head, nest, after-nest]); delete the tail BB.

## Steps (all done)

1. [x] scheduleCluster → one lexical Group; no separate prologue/tail blocks
2. [x] final-lower: blockIndex stamps the group subtree; reducer
       construction inserts at head-end (before the embedded nest)
3. [x] Order net → item-position paths (exposed + fixed the latent
       rank-r-appended-after-sub-block inversion; fills now precede readers)
4. [x] Test shape pins updated; cxx goldens regenerated; specs amended
       (schedule.md, cg.md §1.1/§11 known-gap note retired); full repo green
