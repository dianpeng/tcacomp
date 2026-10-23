---
status: complete
created: 2026-10-23
type: quick
---

# Tensor-contract launcher protocol + anon-ns kernel + indent fix

Author review: the launcher had no mutually understood contract — loose
(ptr, dim, dim, ...) lists. Landed (commit eef64f06):

1. [x] tca::Tensor<T, Rank> as the user-facing launcher type; dims DERIVED
       from tensors + TCA_REQUIRE cross-validation (named per dim)
2. [x] Kernel in the anonymous namespace; the launcher is the only export
       (extern "C" dropped — the Tensor signature is C++)
3. [x] Indentation: body statements emit one level into the kernel; the
       pool comment routed through the kernel-top builder
4. [x] cg.md §8 (Tensor/TCA_REQUIRE) + §9 rewritten; goldens; full repo green
