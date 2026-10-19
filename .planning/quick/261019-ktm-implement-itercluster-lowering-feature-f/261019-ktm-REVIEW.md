---
phase: 261019-ktm-implement-itercluster-lowering-feature-f
reviewed: 2026-10-19T13:13:27Z
depth: standard
files_reviewed: 44
files_reviewed_list:
  - internal/ir/catalog.go
  - internal/ir/cfg.go
  - internal/ir/convention.go
  - internal/ir/dce.go
  - internal/ir/doc.go
  - internal/ir/effect.go
  - internal/ir/kind.go
  - internal/ir/mark.go
  - internal/ir/mutate.go
  - internal/ir/node_dimvar.go
  - internal/ir/node_end.go
  - internal/ir/node_expr.go
  - internal/ir/node.go
  - internal/ir/node_iterfmt.go
  - internal/ir/node_lambda.go
  - internal/ir/node_lir_memory.go
  - internal/ir/node_loop.go
  - internal/ir/node_memory.go
  - internal/ir/node_var.go
  - internal/ir/poset.go
  - internal/ir/print.go
  - internal/ir/schedule.go
  - internal/ir/ssa.go
  - internal/ir/types.go
  - internal/ir/value.go
  - internal/ir/verify.go
  - internal/ir/walk.go
  - internal/opt/all/all.go
  - internal/opt/cache.go
  - internal/opt/context.go
  - internal/opt/dce/dce.go
  - internal/opt/dce/verify.go
  - internal/opt/doc.go
  - internal/opt/factory.go
  - internal/opt/gvn/gvn.go
  - internal/opt/gvn/verify.go
  - internal/opt/instrument.go
  - internal/opt/legality.go
  - internal/opt/manager.go
  - internal/opt/registry.go
  - internal/opt/schedule/scheduler.go
  - internal/opt/simplify/inline.go
  - internal/opt/simplify/lower.go
  - internal/opt/simplify/verify.go
findings:
  critical: 0
  warning: 4
  info: 11
  total: 15
status: issues_found
---

# Quick 261019-ktm: Code Review Report

**Reviewed:** 2026-10-19T13:13:27Z
**Depth:** standard
**Files Reviewed:** 44 (all non-test .go under `internal/ir` and `internal/opt`)
**Status:** issues_found

## Summary

The restructure is architecturally sound: the SSA record model (ssa.go/mutate.go) handles the lambda-node storage correctly — RAUW, SetDef, RemoveUse, and Erase all traverse operand slots and the ImpactBy payload uniformly, and I traced the use-record lifetime through ClearLambdaParams, project erasure, and GVN dedup without finding a dangling or double-counted record. The simplify passes are fail-closed on every malformed shape they can name, the scheduler's edge construction is deterministic in outcome (the `preds` map iteration feeds only a length count; `succs` and the position-based Kahn pick are ordered), and LICM's invariance check covers every enclosing loop. `go vet` is clean.

What the tests do not catch falls into four warnings:

1. The convention sweep (convention.go) fails open on exactly the interface types concrete nodes satisfy (Value, ScalarExprNode, payloadBearer) — the hidden-reference bug class it exists to machine-check can walk past it.
2. The verifier's new map input/param count check and simplify-inline's param stripping contradict each other: survivors of ClearLambdaParams trip the check on legal pass output, and unbound positions are laundered into a clean final verify.
3. simplify-lower trusts the cluster's dims-lead positional split without re-validating it, and then erases the evidence.
4. A def-use diagnostic prints the def's ID twice, mis-naming the offending user.

No critical issues (no security surface, no crashes on sanctioned input, no SSA bookkeeping corruption found).

## Critical Issues

None.

## Warnings

### WR-01: Convention sweep fails open on node-satisfiable interface fields

**File:** `internal/ir/convention.go:55-61`
**Issue:** The interface case of `hiddenRefFieldType` classifies a field of interface type `t` as legal iff the `Node` interface is NOT assignable to `t`:

```go
case reflect.Interface:
    return reflect.TypeOf((*Node)(nil)).Elem().Implements(t)
```

`Node` does not satisfy `Value` (it lacks `IsNone`/`ValueHash`/`ValueEqual`), `ScalarExprNode`, `payloadBearer`, or `effectBearer` — yet every concrete IR node satisfies `Value`, the whole scalar suite satisfies `ScalarExprNode`, and the memory/end nodes satisfy the bearer faces. A struct field of type `Value` or `ScalarExprNode` (or, inside package ir, `payloadBearer`) can therefore hold a live IR node reference and still pass `checkConvention`. This is a fail-open hole in the one machine check that backs the Input/ImpactBy-only convention — the exact "hidden SSA/Node reference in a struct field" bug class the sweep was written to make impossible. No current node exploits it, so the suite is clean today; the guarantee is what is broken.
**Fix:** Probe whether any concrete node type implements `t`, not whether `Node` does. Since all concrete nodes share the same embedded surface, probing two representatives covers the suite:

```go
case reflect.Interface:
    nodeFace := reflect.TypeOf((*Node)(nil)).Elem()
    if nodeFace.Implements(t) {
        return true
    }
    // Interfaces concrete nodes satisfy (Value, ScalarExprNode, the bearer
    // faces) can hide references just as well as Node itself.
    probe := []reflect.Type{
        reflect.TypeOf((*paramRefNode)(nil)).Elem(),  // plain node surface
        reflect.TypeOf((*memLoadNode)(nil)).Elem(),   // + payload/effect faces
    }
    for _, p := range probe {
        if p.Implements(t) {
            return true
        }
    }
    return false
```

(If a future kind adds a novel method surface, extend the probe set beside the allocNode dispatch.)

### WR-02: Map input/param count check and simplify-inline param stripping contradict each other

**File:** `internal/ir/verify.go:277-290` (with `internal/ir/mutate.go:168-192`, `internal/opt/simplify/inline.go:103-104,121-132`)
**Issue:** Two directions of the same broken contract.

(a) *False positive on legal pass output.* `checkNodeShape` fires `map input count must equal lambda parameter count` whenever `lamParams > 0 && len(valueOps) != lamParams`. But `simplify-inline` strips param declarations selectively: `ClearLambdaParams`'s `keep` predicate (inline.go:121-128) deliberately keeps any param with a live use beyond its own declaration slot — inline.go's own comment blesses this ("an acc-style survivor stays declared"). A survivor arises whenever a RoleLambda InputParam is referenced directly (not through a paramRef) by anything else — e.g. used as another stage's input operand, which the public constructor API allows. After stripping the other params, the lambda declares k params while the map still carries N value inputs (inline never touches the input list), so `k != N` and `ir.Verify` rejects a graph the sanctioned pass sequence produced. Selective stripping also silently breaks the positional contract `Input[i] ↔ param i` even when the counts happen to match.

(b) *Fail-open on malformed input.* inline.go:103-104 skips unbound positions (`i >= len(bindings) || bindings[i] == nil`) with `// unbound position: the verifier owns the mismatch`. But the verifier's ownership evaporates once inline strips all bound params: `lamParams == 0` makes the count check vacuous "by design" for fully-inlined maps, so a 3-inputs/2-params map is flagged only by the non-fatal per-pass rendering (VerifyEach.AfterPass logs but cannot abort), and the final authoritative Verify passes clean. Inline launders the evidence the comment says the verifier owns.
**Fix:** Make the two sides consistent — the cheapest sound option is to keep positions aligned: when `ClearLambdaParams` drops a param declaration, also drop the stage's corresponding value-input slot (RemoveUse on the stage's operand), so both counts shrink together and the count check stays meaningful for survivors. Independently, inline should fail closed on `lamParams != len(bindings)` (return an error naming the stage) instead of `continue`, so unbound positions can never be laundered.

### WR-03: simplify-lower trusts the dims-lead positional split without re-validating it, then erases the evidence

**File:** `internal/opt/simplify/lower.go:300-303` (with `internal/ir/node_loop.go:63-83`)
**Issue:** Sweep 2b resolves a project via `ir.ClusterInputs(cl)[idx]`, whose split point comes from `ClusterDims`' scan-until-first-non-dim. Sweep 1's coverage checks instead use `memLoadInputs`, which collects MemLoad operands by kind anywhere in the list. On a malformed cluster whose dims do not lead (a shape `ir.Verify` flags — but per-pass verification is non-fatal, so the pass still runs), the two disagree: `ClusterInputs` then includes the leading dim params in the input list, `idx` indexes the wrong slot, and the pass silently rewrites stage inputs to the wrong defs. Sweep 3 subsequently erases the malformed cluster, destroying the only node the verifier would have flagged — the final Verify is clean and the wrong binding survives to codegen. Today's builders always lead dims (`NewIterCluster` appends dims first), so this needs a hand-forged or future-frontend-bug graph; it is nonetheless a fail-open path in the pass's fail-closed story.
**Fix:** Validate the shape the pass is about to consume positionally: in Sweep 1, before any mutation, check that every leading operand of the cluster is a RoleDim param and reject with an error naming the cluster (mirror of verify.go's `cluster dims must lead the input list`), or derive the input list with the same membership semantics Sweep 1 used (drop operands that are RoleDim params / counted dims, not a positional split).

### WR-04: def-use under-link diagnostic prints the def's ID twice, mis-naming the offending user

**File:** `internal/ir/verify.go:445-450`
**Issue:**

```go
*vs = append(*vs, nodeViolation(r.user, invDefUse,
    fmt.Sprintf("operand use of node %d is missing from node %d's use-list",
        nb.id, nb.id)))
```

Both format arguments are `nb.id` (the def). The Violation header is stamped with `r.user`, but the message text claims the def's own use-list is missing a use *of the def* — actively sending debugging at the wrong node. This is the under-link branch (a walked operand/payload record absent from the def's ref list), where the interesting identity is the user that owns the record. The wording "operand use" is also stale now that the same loop covers payload records.
**Fix:**

```go
fmt.Sprintf("use owned by node %s is missing from node %d's use-list",
    nodeIDStr(r.user), nb.id)
```

## Info

### IN-01: gvn's body-root exclusion rests on a pre-revision raw-pointer claim that is now false

**File:** `internal/opt/gvn/gvn.go:78-84`
**Issue:** The Run doc says "Body roots are the ONE non-SSAUse edge (Lambda.Body is a raw pointer — RAUW cannot rewire it) ... deduping it would strand a dangling lam.Body on the erased twin's bearer." Since the convention rework, a body is the trailing operand slot of the KindLambda node; `SetLambdaBody`'s own doc (mutate.go:126-132) states RAUW can now reach body uses. The exclusion in `lambdaBodyRoots` is therefore vestigial (sound, but it forfeits legal lambda-body CSE) and its rationale is a lie.
**Fix:** Update the comment to say the exclusion is kept as a conservative remnant (or remove the exclusion now that RAUW covers the body slot — the gvn post-RAUW zero-uses assert already guards the transition).

### IN-02: inline's SetLambdaBody branch is dead code; its comments describe the removed raw-pointer edge

**File:** `internal/opt/simplify/inline.go:196-207` (also 41-42)
**Issue:** `ReplaceAllUsesWith(ref, binding)` on the line above moves the lambda's body-slot record onto the binding, so the subsequent `ir.LambdaBodies(stage)` read resolves to `binding` and `body.BasePtr() == ref.BasePtr()` is never true (reachable only in the degenerate `binding == ref` case). The comments ("the one edge RAUW cannot reach", "the Body pointer would dangle at a tombstone") describe the pre-revision model.
**Fix:** Delete the branch and the stale comment lines, or keep it as an explicit belt-and-braces assert with an updated justification.

### IN-03: SetLambdaBody is a choke point with no reachable caller

**File:** `internal/ir/mutate.go:125-158`
**Issue:** With IN-02, the only call site (inline.go:201) is unreachable, making the whole exported choke point dead API surface post-restructure.
**Fix:** Remove it, or note in its doc that it is currently retained only as the sanctioned body-rewrite primitive for future passes (its doc's own framing "kept as its own choke point for discoverability" should then say who will use it).

### IN-04: simplify-verify resolves paramRef targets through a reverse scan justified by a stale comment

**File:** `internal/opt/simplify/verify.go:108-110,153-165`
**Issue:** Both comments say "the ref's payload use is not exported" — false since the convention: the target is operand 0, directly readable as `ir.OperandsOf(n)[0]`. `refTargetDeclared` instead scans every declared param's use list per ref (params × uses), which works but is the pre-revision shape.
**Fix:** Resolve directly: `target := operandAt-equivalent via ir.OperandsOf(n); if len(target) > 0 && declared[target[0].BasePtr()] { ... }`, and delete both stale comments.

### IN-05: poset.go header describes payload bearers that no longer exist; SeaRoots is unconsumed and its root set is duplicated

**File:** `internal/ir/poset.go:10-13,60-74`
**Issue:** The header claims "Payload uses that are NOT ordering edges (store targets, map as-orders, lambda params) are deliberately excluded" — post-convention there are no non-ordering payload uses, and store targets/as-orders/lambda params now ride `DataDeps` (operand slots), i.e. they are included in the def-poset the file describes. Separately, `SeaRoots` has no production caller (tests only), and dce.go's `markReachable` re-implements its root set inline.
**Fix:** Rewrite the header for the ImpactBy-only world; either have `markReachable` consume `SeaRoots`/a shared helper or drop the note that ties them together.

### IN-06: replaceDefine is dead code

**File:** `internal/ir/ssa.go:67-69`
**Issue:** Unexported, zero callers anywhere (including tests), while its doc claims it is "reachable only from the MutContext choke points" — none of the five choke points call it.
**Fix:** Delete it (the doc comment in ssa.go's header listing the var-level primitives should drop the mention).

### IN-07: type intern table is an unsynchronized package global

**File:** `internal/ir/types.go:112-114,144-154`
**Issue:** `internTable` is a mutable global map with no mutex, while registry.go justifies its own mutex with "tests may register later". Any future concurrent graph construction (parallel test builders, a parallel pass driver) turns InternTensor into a data race/panic. Single-threaded today, so latent.
**Fix:** Guard with a `sync.Mutex` (lookup-only fast path is not worth more), or document the single-threaded construction invariant at the interning entry points.

### IN-08: print.go exceeds the project's file-size cap

**File:** `internal/ir/print.go:1-958`
**Issue:** 958 lines against the CLAUDE.md coding-style cap of 800 (200-400 typical). The file mixes four concerns: constant spelling, numbering/operand read faces, the chain/stage renderer, and expression precedence.
**Fix:** Split the exported read faces (ComputeNumbering/OperandsOf/StmtNodes/Label) into a numbering.go and keep the renderer in print.go.

### IN-09: LICM's reposition can silently drop a statement if the insertion target is nil

**File:** `internal/opt/schedule/scheduler.go:283-286` (with `internal/ir/cfg.go:67-69`)
**Issue:** `bb.Remove(hoisted)` unlinks before `bb.InsertBefore(hoisted, firstStatementContaining(bb, n))`; `InsertBefore` returns silently when `at == nil`. If the "never nil for a candidate taken from a statement tree" invariant ever broke, the hoisted statement would vanish from the block with no error — silent wrong codegen rather than a loud failure. Currently unreachable (candidates are always in the snapshot statement's tree and nothing unlinks it).
**Fix:** Nil-check the target and either fall back to `bb.Append(hoisted)` or return an error — one line either way.

### IN-10: duplicated first-wins ownership walk in dce.go

**File:** `internal/ir/dce.go:235-266` and `internal/ir/dce.go:284-314`
**Issue:** `statementOwnership` and the inline owner/ownedBy construction in `wipeDeadStatements` are the same first-wins DFS implemented twice; `markDeadStatements` runs one and `wipeDeadStatements` re-runs the other over the same block.
**Fix:** Extract one `ownershipWithNodes(g) (owner map[Node]*Statement, ownedBy map[*Statement][]Node)` and let both consumers share it.

### IN-11: payloadBearer and effectBearer are now co-extensive

**File:** `internal/ir/node.go:242-253`, `internal/ir/effect.go:128-133`
**Issue:** Since the convention rework, the only payload left is the ImpactBy list, so the two interfaces have identical implementation sets (memLoad/memStore/return). The split (enumeration face vs read-copy face) still earns its keep at the call sites (verifier/Walk/DCE enumerate; poset/effect read), but the doc strings still describe the pre-revision plurality ("store targets, map as-orders, lambda params" as payload examples). This is a design-note, not a defect.
**Fix:** Either merge them (one interface, two accessors) or update the doc strings to say the extension sets are equal by convention and must stay equal until a second payload kind exists.

---

_Reviewed: 2026-10-19T13:13:27Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
