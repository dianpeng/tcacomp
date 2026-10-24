---
status: complete
date: 2026-10-24
---

# Summary — frontend-syntax-design-review

Reviewed the new frontend design in `doc/syntax.md` against the live IR
(`internal/ir`), the builder DSL (`pkg`), and the type inference pass
(`internal/opt/typeinfer`). Verdict: directionally sound, lowers near-mechanically
onto existing `pkg` rungs, but had one grammar hole (undeclared shape names), two
cohesion gaps (stage clause order, `%0`/pipe rules unspecified), a missing scalar-param
story, one IR-contract conflict (`init=dimvar`), and a bridge-placement duplication
risk. All eight findings ruled by the author (R1–R5 + clarifications); produced
`doc/syntax-design.md` as the implementation-ready spec: strict BNF grammar, Go AST
design, semantic-check catalog, frontend type-resolution algorithm with the typeinfer
boundary pinned, AST→IR roadmap (Tasks 0–6, lowering drives `pkg` rungs with one small
`Local` extension), and the matmul_scale acceptance fixture.
