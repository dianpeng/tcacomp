---
phase: 01-ir-core-pass-infrastructure-tooling
reviewed: 2026-10-16T00:00:00Z
depth: standard
files_reviewed: 36
files_reviewed_list:
  - go.mod
  - internal/analysis/analysis.go
  - internal/diagnostic/diagnostic.go
  - internal/diagnostic/diagnostic_test.go
  - internal/flags/flags.go
  - internal/golden/golden.go
  - internal/ir/construct.go
  - internal/ir/construct_test.go
  - internal/ir/irx_test.go
  - internal/ir/kind.go
  - internal/ir/mutate.go
  - internal/ir/node.go
  - internal/ir/print.go
  - internal/ir/print_test.go
  - internal/ir/types.go
  - internal/ir/use.go
  - internal/ir/use_test.go
  - internal/ir/verify.go
  - internal/ir/verify_test.go
  - internal/ir/walk.go
  - internal/log/log.go
  - internal/pass/cache.go
  - internal/pass/cache_test.go
  - internal/pass/context.go
  - internal/pass/instrument.go
  - internal/pass/instrument_test.go
  - internal/pass/legality.go
  - internal/pass/legality_test.go
  - internal/pass/manager.go
  - internal/pass/manager_test.go
  - internal/visualization/dot.go
  - internal/visualization/dot_test.go
  - pkg/builder.go
  - pkg/builder_test.go
  - pkg/chain.go
  - pkg/determinism_test.go
  - pkg/doc.go
  - pkg/expr.go
  - pkg/instrument.go
  - pkg/instrument_test.go
  - pkg/oracle_test.go
findings:
  critical: 2
  warning: 2
  info: 4
  total: 8
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-10-16
**Depth:** standard
**Files Reviewed:** 36
**Status:** issues_found

## Summary

Reviewed the Phase-1 IR core, pass infrastructure, printer, DOT emitter, and
public builder at standard depth. Baseline `go build`, `go vet`, and the full
test suite are green; `go.mod` has zero external requires (constraint held);
no hardcoded secrets, injection surfaces, or unsafe constructs exist anywhere
in scope.

The headline concern is a **single root-cause defect in the core data
structure** that produces two independently confirmed chain-corruption bugs:
`operandList` stores `Use` records **by value** in a mutable Go slice while
the intrusive use-list, the public `UsesOf` API, and the MutContext choke
points all hand out and thread `*Use` pointers **into that slice**. Any
`append` that reallocates (CR-01) or any `slices.Delete` shift (CR-02) moves
the records and strands every pointer embedded in every def's use-list —
use-after-free in Go. Both choke points (`AddUse`, `RemoveUse`) currently
have **zero call sites and zero test coverage**, so the defect is latent, not
active; it detonates the moment the first Phase-2/3 transform pass calls
either API. Empirical repro (scratch white-box test, since removed) is
described under each finding; the verifier does catch the resulting damage,
but only as post-hoc violation reports, and CR-02 additionally nils out a
*live, correctly-wired* operand.

The two warnings are printer-side fidelity issues: `wireOperands` silently
discards surplus operands at construction (fail-open, verifier-invisible),
and `FormatConst` renders integer-typed constants at float32 precision so
distinct large i32/u32 constants print identically.

Determinism review is otherwise clean: no map iteration feeds emission
(interner map is lookup-only, timing report is sorted, diagnostics in
insertion order), ID allocation is per-graph monotonic with no reuse, and the
legality windows genuinely fail closed (absence-from-Actions rejects, unset
stages get empty windows — both pinned by tests).

## Critical Issues

### CR-01: `AddUse` corrupts every use-list when `append` reallocates the ops slice

**File:** `internal/ir/mutate.go:47-61`, root cause `internal/ir/node.go:167-180`
**Issue:** `operandList.ops` is `[]Use` — records by value. `linkIntoDef`
splices `&l.ops[i]` (a pointer into the backing array) into the def's
intrusive use-list, and `UsesOf`/`DefOf`/`MutContext.SetDef` all expose and
consume those `*Use` pointers. `addSlot` does `l.ops = append(l.ops, Use{})`:
when `len == cap`, append allocates a new backing array and **moves every
existing Use record**, leaving every `*Use` embedded in every def's use-list
(and every record's own prev/next) pointing at the abandoned old array, which
the GC then frees. This is not theoretical: a binop's ops slice is exactly at
capacity after construction (len 2, cap 2), so the *first* `AddUse` on any
two-operand node reallocates.

Confirmed empirically (scratch test, since removed):

```
sum := g.NewBinOp(...)            // ops len 2 cap 2
ctx.AddUse(sum, c)                // append -> realloc; a's and b's records move
Verify(g) // 4 def-use violations:
//   "use-list of node 1 contains a use that no operand slot owns"
//   "operand use of node 1 is missing from node 1's use-list" (x2 for nodes 1, 2)
```

The same hazard applies to variadic nodes (`call`, `map`) whenever a pass
grows them past capacity. Note construction itself (`allocNode` +
`wireOperands`) is safe only because all `addSlot` calls complete before any
linking — the bug is exclusive to the MutContext-era mutation path, i.e.
exactly the path every future pass uses.

**Fix:** Give Use records stable identity — store pointers in the slice:

```go
// node.go
type operandList struct {
    ops []*Use // one *Use per operand; records never move
}

func (l *operandList) addSlot() *Use {
    u := &Use{}
    l.ops = append(l.ops, u) // moves pointers, not records
    return u
}

func (l *operandList) forEachOperand(fn func(*Use)) {
    for _, u := range l.ops {
        fn(u)
    }
}
```

`removeSlot` then drops a pointer (no shift of records) but still must
re-validate links; see CR-02. Add a white-box test that calls `AddUse` on a
full-capacity node and asserts `Verify` stays clean.

### CR-02: `RemoveUse` of a non-last slot shifts records and then nils out the WRONG use

**File:** `internal/ir/mutate.go:66-79`, root cause `internal/ir/node.go:182-190`
**Issue:** `removeSlot` locates `u` at index `i` and calls
`slices.Delete(l.ops, i, i+1)`. That copies `ops[i+1:]` down one position —
so the record that *was* at `ops[i+1]` now lives at address `ops[i]`, which
is exactly the pointer `u` still holds. Every pointer into the shifted region
dangles: the moved records' prev/next links and the defs' use-list entries
still reference the old addresses. Then, worse, `RemoveUse`'s final statement
`u.Val, u.User = nil, nil` executes **after** the shift, so it nils the def
and user of the *moved, live* record — a correctly-wired operand is
destroyed.

Confirmed empirically (scratch test, since removed), on a fully-wired select
`ops = [cond=a, then=b, else=c]`:

```
ctx.RemoveUse(&sel.ops[1]) // remove the MIDDLE slot (def b)
Verify(g) // 1 use-def violation:
//   "operand slot 1 has nil def (unwired operand)"   <- slot 1 was c, fully wired!
```

Reproducing requires only a three-operand node (select) — not an exotic
shape. Zero call sites exist today, but `RemoveUse` is a sanctioned
choke-point that every DCE pass will call.

**Fix:** Same root-cause fix as CR-01 (`ops []*Use`: `slices.Delete` then
only moves pointers; records and their links stay put), **plus** nil out the
record before removal or not at all:

```go
func (ctx *MutContext) RemoveUse(u *Use) {
    if u == nil {
        return
    }
    user := u.User
    if u.Val != nil {
        unlinkFromDef(u.Val.BasePtr(), u)
    }
    u.Val, u.User = nil, nil // clear BEFORE the slot disappears
    if impl, ok := user.(nodeImpl); ok {
        impl.removeSlot(u) // with []*Use this drops a pointer; address stays valid
    }
    ctx.stamp(user)
}
```

Add a white-box test removing each slot position of a select and asserting
`Verify` stays clean and the surviving operands' `DefOf` records are intact.

## Warnings

### WR-01: `wireOperands` silently discards operands beyond a fixed-arity kind's slot count

**File:** `internal/ir/construct.go:223-236`
**Issue:** `NewNode(KindBinOp, origin, a, b, c)` wires `a` and `b` and
**silently drops `c`** (`if i >= len(slots) { return }`). The file comment
frames this as deliberate ("operands beyond a fixed-arity kind's slot count
are ignored"), but unlike the other deliberate non-validations — which leave
detectable residue the verifier flags (nil def, unknown kind) — a dropped
operand leaves **no trace**: the verifier sees a well-formed graph missing a
third of its intended input. A pass or builder bug that over-supplies an
operand becomes silent miscompilation instead of a caught error. This is a
fail-open construction path in a design whose stated contract is fail-closed
(IR-03, legality windows).

**Fix:** Record the surplus so rejection is possible. Cheapest option that
fits the existing architecture: make `wireOperands` return the surplus count
and have `NewNode` record it on `Base` (e.g. `wiredOps, surplus int` fields),
then add a verifier invariant "node constructed with N surplus operands
(arity M)" under `invUseDef`. Alternatively change `NewNode` to return an
error when `len(operands) > fixedArity(k)` for fixed-arity kinds — acceptable
because construction is internal to the module and every in-tree caller
already passes the exact arity. At minimum, panic-free loud failure at
construction time; never silent drop.

### WR-02: `FormatConst` renders large integer-typed constants at float32 precision — distinct constants print identically

**File:** `internal/ir/print.go:59-65`
**Issue:** `FormatConst` formats every constant with
`strconv.FormatFloat(v, 'g', -1, 32)` regardless of the node's type. For
f32 that is the pinned shortest-round-trip behavior, but `constNode` also
carries i32/u32/bool types (created via `Builder.Int32`, `Builder.UInt32`,
`Builder.Bool`), and float32 mantissa has 24 bits:

```
Int32(2147483647)  -> "2.1474836e+09"
Int32(2147483600)  -> "2.1474836e+09"   // same text, different value
UInt32(4294967295) -> "4.2949673e+09"
16777217           -> "1.6777216e+07"   // 2^24+1 rounds to 2^24
```

Two distinct IR constants render byte-identically, breaking the printer's
round-trip fidelity (D-01/D-02: the print must faithfully represent the
graph), and any downstream text consumer — including a golden that bakes in
the lossy spelling — sees the wrong value. Today the builder only feeds
small constants (0.0, 1.0), so goldens pass; the first `Int32` shape
parameter or large seed will print lossy.

**Fix:** Type-dispatch the rendering; the node already carries `typ`:

```go
// print.go — call site: case *constNode: return r.constText(t)
func (r *renderer) constText(t *constNode) string {
    switch t.typ.Elem() {
    case TI8, TI16, TI32:
        return strconv.FormatInt(int64(t.val), 10)
    case TU8, TU16, TU32:
        return strconv.FormatUint(uint64(t.val), 10)
    case TBool:
        if t.val != 0 { return "true" }
        return "false"
    default: // f16/bf16/f32
        s := strconv.FormatFloat(t.val, 'g', -1, 32)
        if !strings.ContainsAny(s, ".eE") { s += ".0" }
        return s
    }
}
```

Keep `FormatConst` exported for f32 goldens, but route const rendering
through the typed path; pin `Int32(2147483647)` in `TestFormatConst`.

## Info

### IN-01: `checkDefUse` "missing from use-list" message names the def node twice

**File:** `internal/ir/verify.go:159-165`
**Issue:** `fmt.Sprintf("operand use of node %d is missing from node %d's use-list", nb.id, nb.id)` — both slots get `nb.id` (the def). The message should name the *user* node's ID in the first slot (`u.User.BasePtr().id`), like the sibling "over-link" messages already do via `nodeIDStr(u.User)`. The structured `Violation.NodeID` field is correct (it uses `u.User`), so only the human text lies. Invariant-by-invariant diagnostics are the PASS-06 debugging substrate; a message that names the wrong node sends the reader to the wrong node.
**Fix:** `fmt.Sprintf("operand use of node %d is missing from node %d's use-list", userID(u.User), nb.id)` — guard nil `u.User` as `nodeIDStr` does.

### IN-02: Two "canonical" renderings of the same violation disagree on quote style

**File:** `internal/ir/verify.go:23`, `internal/pass/instrument.go:56`
**Issue:** `Violation.String()` renders `last touched by '<pass>'` (single quotes); `VerifyEach.render` renders `last touched by "<pass>"` (double quotes, `%q`). Both are documented as canonical forms of the same diagnostic (the instrument.go doc comment even pins the `"..."` spelling). Golden tests that match on violation text will be fragile against whichever path produced them.
**Fix:** Have `VerifyEach.render` delegate to `violation.String()` (single rendering, single source of truth); or update `Violation.String` to `%q`. One form, used everywhere.

### IN-03: `EmitDot` interpolates the graph name unquoted into the digraph header

**File:** `internal/visualization/dot.go:32`
**Issue:** `fmt.Fprintf(bw, "digraph %s {\n", g.Name())` emits the raw name. `NewBuilder`/`NewGraph` accept arbitrary strings, so a name containing spaces, quotes, or a Graphviz keyword (`strict`, `graph`) produces syntactically invalid DOT from an otherwise valid graph. Labels are safely `%q`-quoted; the header is not.
**Fix:** Quote it: `fmt.Fprintf(bw, "digraph %q {\n", g.Name())` — Go `%q` escaping is compatible with DOT quoted IDs for all names this compiler generates (C-identifier-style today, so this is hardening, not a live failure).

### IN-04: The two untested choke-point mutations are exactly where CR-01/CR-02 live — and `firstDiff` has a dead branch

**File:** `internal/ir/use_test.go`, `internal/golden/golden.go:82-84`
**Issue:** (a) `use_test.go` exercises RAUW, Erase, and SetDef, but `AddUse` and `RemoveUse` have no direct coverage anywhere in the tree — which is why the CR-01/CR-02 chain corruption survived a green suite. The success criteria for this phase center on the choke-point API being trustworthy; add round-trip tests for both (grow a call node past capacity; remove each slot position of a select) asserting `Verify` stays clean. (b) `golden.go`'s `firstDiff` first branch prints `at == n && len(wl) == len(gl)` with `len(wl), len(gl)` — two equal numbers, and the branch is unreachable whenever the byte-compare said "differ" (equal split lines with equal lengths imply equal bytes). Harmless dead text; delete the branch or report the byte lengths.

---

_Reviewed: 2026-10-16_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
