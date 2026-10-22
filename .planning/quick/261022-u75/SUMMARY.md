---
status: complete
completed: 2026-10-22
type: quick
---

# SUMMARY — complete doc/cg.md (CUDA C++ codegen design)

## What was done

Expanded the 31-line seed doc/cg.md into the complete CUDA C++ backend
codegen design (11 sections), grounded in a full review of the
post-final-lower codebase (two parallel explorations: IR node inventory +
pipeline entry; scalfold runtime reference) and the existing spec set
(tca.md §5.10-13, final_lower.md, runtime_objects.md, loop_tile_lower.md).

Key sections: input contract (ASTKernel walk + node whitelist + metadata
beyond the IR), emitted artifact trio, the memory shim model, emitter
architecture (internal/cg/cxx package layout, deterministic on-the-fly SSA
naming), type/opcode mapping, the per-node emission table (every surviving
kind), loop materialization via runtime iterators, the full pinned
`tca::rt` runtime API surface (§8 — the contract the follow-up library task
implements), host launcher + generated CMake, testing, follow-ups.

## Author rulings incorporated (2026-10-22)

- D1 artifact = full trio (.cu with kernel + extern "C" launcher, CMakeLists)
- D2 namespace = tca::rt
- D3 GMEM access = runtime TensorView wrappers (never raw pointers)
- D4 SMEM = kernel-top hoist
- D6 (mid-turn message) ALL memory through C++ runtime wrappers with
  compiler-emitted layout descriptors; GMEM coalescing + SMEM swizzle are
  the next optimization targets after codegen — descriptor-only changes

## Rulings derived and FLAGGED back (documented in doc/cg.md §0)

- D5: SMEM must be DYNAMIC `extern __shared__` + opt-in (MaxCache 64KB
  budget > the 48KB static ceiling; worked example's lhs window is exactly
  64KB) — preserves D4's kernel-top placement intent.
- D7: slot-IV reducers materialize Combine() once into a slot-addressable
  local; re-iteration uses `.Get(slot ivs...)` in Reduce slot-arg order.

## Findings reported

- No CLI / no internal/syntax frontend exists; codegen ships as a Go API
  with golden .cu tests (documented §1.4, §10).
- pkg.Builder.Build() materializes and the schedule gate refuses
  materialized graphs — fixtures must use internal/ir constructors (§1.4).
- Launch config lives in tiling Params, not the IR (§1.3).

## Follow-up (next task)

Implement the C++ runtime library per doc/cg.md §8 + doc/runtime_objects.md.
Then: GMEM coalescing + SMEM swizzle passes (descriptor changes only).
