---
status: complete
completed: 2026-10-23
type: quick
---

# SUMMARY — one lexical Group per LNT

Commit 7839356. The generated kernels now compile: declarations
(GridMapping/ThreadMapping/KernelInfo/reducers/hoisted invariants) share
one C++ scope with their uses — the nest is a statement inside the LNT's
lexical Group; the tail BB is deleted (after-nest = item position).

Two bonus resolutions from the stricter item-position order net:
- placeRank now INSERTS rank-r statements before their block's sub-block
  (they appended after it — deeper loops' bounds executed after the loops
  consuming them; masked by block-granular checking)
- The fill-after-reads defect (pinned as a known gap in 261022-cxx) is
  FIXED: fills sit at the top of their host body, before every reader.

Consumers updated: final-lower (blockIndex subtree stamping; reducer
construction at head-end), schedule tests (rootBlockOf/afterRoot/headNodes
helpers), cxx goldens regenerated, schedule.md + cg.md amended.
