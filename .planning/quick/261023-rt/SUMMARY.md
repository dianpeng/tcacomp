---
status: complete
completed: 2026-10-23
type: quick
---

# SUMMARY — runtime design doc + scratch ruling (quick-261023-rt)

Commit a5f6e64. doc/rt.md is the library task's single input: every
signature quoted from the goldens, every behavioral contract cited to its
ruling, the SMEM accounting table (dynamic pool + static reducer scratch),
and the test strategy. The scratch gap is closed at the emitter: <T, Slots>
carries the slot TILE EXTENT (product of slot-dim tiles), runtime sizes
acc[Slots] + static stage from it; identity-init full-warp-tree
simplification + replicated-local-combine staging protocol pinned in §4.6.
A second latent gap (loader window→axis mapping) surfaced and was pinned as
the trailing-axes contract (D12).
