---
status: complete
completed: 2026-10-23
type: quick
---

# SUMMARY — Tensor-contract launcher (quick-261023-ten)

Commit eef64f06. The launcher now speaks tca::Tensor<T, Rank> (device
memory + shape + layout in one object): read-only inputs as const Tensor&,
stored outputs as Tensor&; every kernel dim is derived from its first
tensor occurrence and cross-validated with named TCA_REQUIRE messages —
the shape contract is enforced at the boundary. Kernel hidden in the
anonymous namespace (launcher = the only exported symbol); indentation of
the kernel body fixed (one level in). Runtime §8 surface gains Tensor +
TCA_REQUIRE (the library task implements them); §9 rewritten around the
protocol.
