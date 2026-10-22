---
status: complete
created: 2026-10-22
type: quick
---

# Implement internal/cg/cxx — the CUDA C++ backend

## Description

Implement the codegen backend per doc/cg.md (completed in quick-261022-u75):
`internal/cg` (Options/Result) + `internal/cg/cxx` emitting the full trio
(`<name>.cu` + `CMakeLists.txt`). Golden-`.cu` verification only — the C++
runtime library is the follow-up task.

## Tasks

1. [x] internal/cg: Options/Result; cxx: gates + naming/types + walk +
       scalar exprs + loops (iterators) + GMEM views — matmul store-chain
       golden (full default pipeline fixture)
2. [x] SMEM pool + views + fills (loader calls, axes/biv merge rule) + frag
       + chained accesses (part of the same golden)
3. [x] Runtime objects: KernelInfo/mappings/reducers + CallMethod (D7 slot
       rule) + GetAttribute + lambda hoist — reduce-chain golden (crafted
       tiling record fixture)
4. [x] Host launcher + generated CMakeLists goldens; determinism test;
       fail-closed gate tests
5. [x] doc/cg.md §7/§8 amendments discovered during implementation
       (iterator ctor args incl. init/step; loader arg convention;
       Has()-uniform semantics per tca.md §5.12); STATE.md row

## Constraints

- Pure Go stdlib; deterministic output (no unsorted map iteration)
- Follow ir read faces only (OperandsOf, *Of accessors) — no concrete-type
  assertions outside package ir
- Only necessary tests (golden + determinism + gates)
