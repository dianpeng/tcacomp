// Package ir implements the TCA single-tier SSA IR.
//
// Design notes (Phase 01.1 refactor — directives 1-4):
//
//   - Types: Type is an interface with exactly one method, Equal(Type) bool.
//     NumericType is the concrete numeric component (unexported elem/dims);
//     interns are shared *NumericType pointers, so interface == is identity
//     for numerics and Equal is the cross-component comparison point for
//     future tuple/struct types. Consumers needing elem/dims assert:
//     nt, ok := t.(*NumericType); a nil or non-numeric type renders as "?"
//     and is left for the verifier to reject — the printer's numeric paths
//     are never touched by future components.
//
//   - Values: Value is a real interface (Node + IsNone). Every concrete node
//     embeds *Base, which promotes IsNone() == false; returnNode overrides it
//     to true — the statement root that binds no SSA name ("a none Value is
//     a Value but just nothing"). AddStatement/AddStatementAs stay Node-typed
//     on purpose: all exported constructors return Node, which never
//     statically satisfies Value, so a Value-typed entry point would break
//     six out-of-package callers (pkg/chain.go, pkg/builder.go,
//     pkg/oracle_test.go, pass/manager_test.go x2, pass/instrument_test.go)
//     and irx_test.go with no compensating benefit. The Value/NoneValue
//     classification lives in exactly one place: Statement.Value(), whose
//     rhs assertion is total for in-repo nodes (no panic path). Do not
//     re-litigate this without migrating all of those callers first.
//
//   - SSA chains: each node owns one SSAVar (allocated 1:1 at node
//     allocation, ID == node ID). SSADef and SSAUse are value-type HANDLES
//     over that shared pointee; operand slots hold SSAUse handles pointing
//     at ref records {def var, user node, killed}, so rewiring a record
//     propagates through every copy of the handle. Killed and zero handles
//     are representable but checkable: the verifier flags them under the
//     use-def invariant, naming the slot. All chain edits flow through the
//     five MutContext choke points (SetDef/AddUse/RemoveUse/
//     ReplaceAllUsesWith/Erase), each stamping LastTouchedBy; the var-level
//     primitives (replaceDefine/replaceAllReference/record kill) are
//     unexported, reachable only via those choke points.
//
//   - Walking: Walk(g, Visitor) is the analysis/checking substrate —
//     iterative pre-order DFS with a visited set (cyclic-safe), roots in BB
//     statement order (def then rhs), children in operand-slot order then
//     lambda bodies. VisitResult gives Descend/SkipChildren/Stop control
//     (SkipChildren prunes exactly the skipped node's subtree; a node
//     reachable by another edge is still visited). The canonical PRINTER and
//     the dot emitter do NOT walk this way: their StmtNodes DFS is an
//     emission-order function (chain stages producer-first via pipeInput),
//     golden-pinned — Walk must never be used where emission order matters.
//
// Interface-width contract (directive 4): Node has exactly 2 methods
// (BasePtr, KindOf); Value adds exactly IsNone; Type adds exactly Equal;
// Visitor adds exactly Visit. New capabilities arrive as new interfaces or
// new Type implementations, never as widened ones.
package ir
