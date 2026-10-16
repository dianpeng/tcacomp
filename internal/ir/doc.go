// Package ir implements the TCA single-tier SSA IR.
//
// Design notes (Phase 01.1 refactor — directives 1-4; Phase 1 review reshape
// — P1R-01/02/03/06/07):
//
//   - Optimizer shape: the IR is an optimizer IR, not an AST materialization
//     (doc/review/phase1_1.md). Every reference in the graph is an SSAUse —
//     operand slots AND payload uses alike (lambda params, iteration
//     dims/bounds, map as-orders, store target accesses, parameter refs,
//     all SSAUse records owned by their referencing node) — and every
//     definition is an SSADef. Kernel and lambda parameters are InputParam
//     NODES (graph-owned for kernel params); names are debug payload only
//     and are never consulted for resolution or identity. Construction
//     resolves everything on the fly: a freshly built graph has zero
//     name-keyed references (pkg.Builder's declaration-time symbol tables
//     are the on-the-fly resolution the review mandates).
//
//   - CFG/statements: Graph → CFG (v0: exactly one *BasicBlock) → Statement
//     intrusive doubly-linked list (head/tail on the block, prev/next on the
//     statement). Removal of a statement given its pointer is O(1) — no
//     position search, no slice reindex. Statements classify as StmtAssign
//     (binds a new local: def wired to the RHS value) or StmtEffect (pure
//     side-effect op: store to observable memory, kernel return).
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
//     out-of-package callers with no compensating benefit. The
//     Value/NoneValue classification lives in exactly one place:
//     Statement.Value(), whose rhs assertion is total for in-repo nodes (no
//     panic path). Do not re-litigate this without migrating all of those
//     callers first.
//
//   - SSA chains: each node owns one SSAVar (allocated 1:1 at node
//     allocation, ID == node ID). SSADef and SSAUse are value-type HANDLES
//     over that shared pointee; operand slots hold SSAUse handles pointing
//     at ref records {def var, user node, killed}, so rewiring a record
//     propagates through every copy of the handle. Payload uses share the
//     same record mechanics (wirePayloadUse) and are enumerated via the
//     payloadBearer interface so the verifier's use-def/def-use invariants
//     and Walk cover them exactly like operand slots. Killed and zero
//     handles are representable but checkable: the verifier flags them under
//     the use-def invariant, naming the use. All chain edits flow through
//     the five MutContext choke points (SetDef/AddUse/RemoveUse/
//     ReplaceAllUsesWith/Erase), each stamping LastTouchedBy; the var-level
//     primitives (replaceDefine/replaceAllReference/record kill) are
//     unexported, reachable only via those choke points.
//
//   - Walking: Walk(g, Visitor) is the analysis/checking substrate —
//     iterative pre-order DFS with a visited set (cyclic-safe), roots in
//     graph-param order then BB statement order (def then rhs), children in
//     operand-slot order, then payload uses, then lambda bodies.
//     VisitResult gives Descend/SkipChildren/Stop control (SkipChildren
//     prunes exactly the skipped node's subtree; a node reachable by another
//     edge is still visited). The canonical PRINTER and the dot emitter do
//     NOT walk this way: their StmtNodes DFS is an emission-order function
//     (chain stages producer-first via pipeInput), golden-pinned — Walk must
//     never be used where emission order matters.
//
//   - Lowered suite (review P1R-04): dimvar/bounded-dimvar (value-object
//     induction variables, (init, step) tuples with EQ/NE and the
//     Include/Belong relations of doc/optana/induction_variable.md),
//     loadtile/storetile (FamilyTileOp; the tensor is an OPERAND slot, the
//     axes bounded dimvars), and the explicit for-loop (FamilyLoop; ops =
//     [iv, bound, body]). Kinds are append-only with a versioned Kind
//     enumeration (IR-03; KindVersion bumps on additions); the verifier's
//     shape checks fail closed on operand kinds (bdimvar binds a dimvar,
//     tiles bind a RoleTensor InputParam, for-loops bind a dimvar).
//
// Interface-width contract (directive 4): Node has exactly 2 methods
// (BasePtr, KindOf); Value adds exactly IsNone; Type adds exactly Equal;
// Visitor adds exactly Visit. New capabilities arrive as new interfaces or
// new Type implementations, never as widened ones.
package ir
