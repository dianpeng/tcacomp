---
phase: 01-ir-core-pass-infrastructure-tooling
fixed_at: 2026-10-16T00:00:00Z
review_path: .planning/phases/01-ir-core-pass-infrastructure-tooling/01-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-10-16
**Source review:** .planning/phases/01-ir-core-pass-infrastructure-tooling/01-REVIEW.md
**Iteration:** 1
**Fix scope:** critical_warning (CR-01, CR-02, WR-01, WR-02; Info findings IN-01..IN-04 out of scope)

**Summary:**
- Findings in scope: 4
- Fixed: 4
- Skipped: 0

Verification: `go build ./...`, `go vet ./internal/ir/`, and `go test ./...` all green after
each commit, including the committed golden fixtures. WR-01 and WR-02 change observable
behavior (a new verifier invariant and printer output spelling) and are flagged
"fixed: requires human verification" per the logic-fix policy; CR-01/CR-02 are pinned by
new white-box tests reproducing the reviewer's empirical failure modes.

## Fixed Issues

### CR-01: `AddUse` corrupts every use-list when `append` reallocates the ops slice

**Files modified:** `internal/ir/node.go`, `internal/ir/use.go`, `internal/ir/use_test.go`,
`internal/ir/construct_test.go`, `internal/ir/verify_test.go`
**Commit:** 287bd62
**Applied fix:** Root-cause repair exactly as prescribed: `operandList.ops` is now
`[]*Use` — one heap-allocated, address-stable `*Use` per slot — so `addSlot`'s append
moves pointers, never the records that def use-lists and public APIs hold. `removeSlot`
compares and drops pointers. The `Use` doc comment now documents the stable-identity
storage model. Two white-box tests that took `&bin.ops[i]` (invalid under the new
representation) were updated to the pointer form. Added `TestAddUseKeepsChainsConsistent`:
appends a third operand to a len-2/cap-2 binop (forces reallocation) and grows a call
node 8 slots past capacity, asserting `UsesOf` integrity and a clean `Verify` throughout.

### CR-02: `RemoveUse` of a non-last slot shifts records and then nils out the WRONG use

**Files modified:** `internal/ir/mutate.go`, `internal/ir/use_test.go`
**Commit:** 6c41756
**Applied fix:** Applied the reviewer's prescribed reorder on top of the CR-01
pointer-valued slots: `RemoveUse` now clears `u.Val, u.User` BEFORE `impl.removeSlot(u)`,
so no ordering can nil out a surviving record. Added `TestRemoveUseEachSlotPosition`:
removes each of the three slots of a fully-wired select (fresh graph per position),
asserting slot count, full detachment of the removed record, intact `DefOf` wiring and
def use-lists for survivors, and a clean `Verify` — the middle-slot case is the one that
previously destroyed the live record.

### WR-01: `wireOperands` silently discards operands beyond a fixed-arity kind's slot count

**Files modified:** `internal/ir/node.go`, `internal/ir/construct.go`, `internal/ir/verify.go`,
`internal/ir/verify_test.go`
**Commit:** 38bdef7
**Status:** fixed: requires human verification
**Applied fix:** Chose the reviewer's primary option (surplus recording + verifier
invariant) over the `NewNode` signature change, keeping the public construction surface
stable and preserving cross-package rejection tests' ability to construct out-of-window
graphs. `wireOperands` now returns the surplus count (counting every overflow operand,
not just the first); `NewNode` records it in a new `Base.surplus` field for fixed-arity
kinds (variadic kinds and the raw fallback wire everything, surplus always 0); the
verifier's `walkCheckUseDef` flags `node constructed with N surplus operands (arity M)`
under `invUseDef`. Added `TestVerifierFlagsSurplusOperands` (3-operand binop flagged;
exact-arity sibling clean). Human check suggested: the invariant's message wording and
the decision to reject via verifier rather than at construction.

### WR-02: `FormatConst` renders large integer-typed constants at float32 precision

**Files modified:** `internal/ir/print.go`, `internal/ir/print_test.go`
**Commit:** 2836d02
**Status:** fixed: requires human verification
**Applied fix:** Applied the reviewer's typed-dispatch design: new `constText(*constNode)`
dispatches on the node's interned element type — `TI8/TI16/TI32` render via
`strconv.FormatInt`, `TU8/TU16/TU32` via `strconv.FormatUint`, `TBool` as `true`/`false`,
and the float family keeps `FormatConst`'s shortest-round-trip f32 spelling (nil type
falls back to the float spelling so the printer stays deterministic on malformed nodes;
the verifier owns their rejection). The printer's `value()` routes const nodes through
`constText`; `FormatConst` remains exported as the float-family spelling owner. Added
`TestConstTextTyped` pinning `Int32(2147483647)` vs `Int32(2147483600)` as distinct
text, `UInt32(4294967295)`, `2^24+1`, bools, and an end-to-end print through a return
stage. Existing goldens pass unchanged (all pinned constants are f32). Human check
suggested: integer-typed constants now print as bare integers (no `.0`) in printer and
DOT output — confirm that spelling is the desired canonical form.

## Skipped Issues

None.

---

_Fixed: 2026-10-16_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
