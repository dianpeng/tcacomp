---
status: complete
---

# Quick Task 20261024-frontend-syntax-design-review

Design review of the new TCA frontend design (`doc/syntax.md`, a total rewrite) with a
ready-to-implement design doc as the deliverable.

## Findings (all ruled by the author before writing)

- F1 shape names (`M, N, Q`) never declared → R1 implicit-from-signature
- F2 no scalar param/capture mechanism (matmul_scale needs `scale`) → R2 signature param + capture list
- F3 map/reduce clause-order inconsistency, `on(...)` overloaded → R3 unified `verb (attrs)? on(args) (as)? lambda-tail`
- F4 `init=dimvar → InductionVariable` conflicts with homogeneous-reducer IR contract → R4 deferred from v0 (roadmap)
- F5 `internal/ir/builder.go` bridge would duplicate `pkg` validation → R5 lower by driving `pkg` rungs
- F6 optional typing resolved at frontend build time; `internal/opt/typeinfer` boundary pinned (it types IR nodes post-simplification, never fills AST holes)
- F7 old (deleted) syntax.md section anchors still cited by code comments (`ir/types.go §3.3`, `ir/builtins.go §6.4`, `ir/kind.go §7.6`, `pkg/chain.go`, `pkg/expr.go`) → Task 0 re-points them
- F8 `%0` placeholder, pipe-vs-bitwise-or `|`, as-inference, dimvar-vs-dim namespaces clarified normatively

## Deliverable

- `doc/syntax-design.md` — implementation-ready: strict BNF (§2), AST design (§3),
  semacheck rule catalog (§4), optional-type resolution + typeinfer boundary (§5),
  AST→IR construction roadmap Tasks 0–6 (§6), normative clarifications (§7),
  nice-to-have roadmap (§8), matmul_scale acceptance fixture (Appendix A).

## Plan

1. Read doc/syntax.md + old spec from git + internal/ir + pkg + internal/opt/typeinfer — done
2. Surface findings, get rulings — done (4 AskUserQuestion rulings, all recommended option)
3. Write doc/syntax-design.md — done
4. SUMMARY.md + STATE.md + commit — this commit
