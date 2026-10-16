package ir

import (
	"errors"
	"fmt"
)

// MutContext identifies the running pass at every choke-point mutation
// (IR-04, D-17). The IR stamps LastTouchedBy = ctx.Pass on every node a
// mutation touches, making PASS-06's last-touched-by diagnostics structural
// rather than advisory. Construction (construct.go) is exempt; every other
// mutation of the chains flows through these five functions. The operations
// take and return SSAUse/SSADef value-type handles: copies are free and
// mutations propagate through the shared ref records (directive 2).
type MutContext struct {
	Pass string
}

// stamp records ctx.Pass as the last pass to touch n.
func (ctx *MutContext) stamp(n Node) {
	if n != nil {
		n.BasePtr().lastTouchedBy = ctx.Pass
	}
}

// SetDef redirects one operand use to a new def, physically moving its ref
// record from the old def's var into the new one's. A nil newVal unwires the
// slot (the verifier's "nil def" invariant flags it). The record — not the
// handle — moves, so every copy of the SSAUse observes the new target.
func (ctx *MutContext) SetDef(u SSAUse, newVal Node) error {
	if u.rec == nil {
		return errors.New("ir: SetDef on nil use")
	}
	oldVar := u.rec.def
	var newVar *SSAVar
	if newVal != nil {
		newVar = newVal.BasePtr().ssa
	}
	// Even a same-var "move" re-stamps (D-17): the old implementation stamped
	// unconditionally, and passes rely on SetDef for the last-touched-by mark.
	if oldVar != nil {
		oldVar.removeRef(u.rec)
	}
	u.rec.def = newVar
	if newVar != nil {
		newVar.refs = append(newVar.refs, u.rec)
	}
	ctx.stamp(u.rec.user)
	if oldVar != nil {
		ctx.stamp(oldVar.node)
	}
	ctx.stamp(newVal)
	return nil
}

// AddUse appends one operand slot to user and links it into val's ref list,
// returning the new SSAUse handle. The zero handle is returned if user is
// not an IR node. A nil val leaves the slot unwired.
func (ctx *MutContext) AddUse(user Node, val Node) SSAUse {
	impl, ok := user.(nodeImpl)
	if !ok {
		return SSAUse{}
	}
	u := impl.addSlot()
	if val != nil && val.BasePtr().ssa != nil {
		rec := val.BasePtr().ssa.addRef(user)
		u.rec = rec
		impl.setSlot(lenOf(impl)-1, u) // write the wired handle into the new slot
	}
	ctx.stamp(user)
	ctx.stamp(val)
	return u
}

// lenOf returns the node's operand-slot count.
func lenOf(n nodeImpl) int {
	count := 0
	n.forEachOperand(func(SSAUse) { count++ })
	return count
}

// RemoveUse detaches u from its def's ref list and drops u's slot from the
// user's operand storage, killing the record so every copy of the handle
// reads as dead (IsKilled). The record's def/user back-pointers are left
// intact: a killed record still occupying a slot is exactly the corruption
// class the verifier's use-def check flags by node and slot.
func (ctx *MutContext) RemoveUse(u SSAUse) {
	if u.rec == nil {
		return
	}
	user := u.rec.user
	if def := u.rec.def; def != nil {
		def.removeRef(u.rec)
	}
	u.rec.killed = true
	if impl, ok := user.(nodeImpl); ok {
		impl.removeSlot(u)
	}
	ctx.stamp(user)
}

// ReplaceAllUsesWith moves every live use of old onto newVal in O(uses),
// mirroring LLVM's replaceAllUsesWith: each ref record is rewired into
// newVal's var in one pass over a snapshot, so the walk survives its own
// mutation (RESEARCH Pitfall 2) and every handle copy sees the new target.
// Self-RAUW is a no-op.
func (ctx *MutContext) ReplaceAllUsesWith(old, newVal Node) error {
	if old == nil || newVal == nil {
		return errors.New("ir: ReplaceAllUsesWith on nil node")
	}
	if old == newVal {
		return nil
	}
	ov, nv := old.BasePtr().ssa, newVal.BasePtr().ssa
	moved := ov.liveRefs() // snapshot BEFORE the move: exactly the users RAUW touches
	ov.replaceAllReference(nv)
	for _, r := range moved {
		ctx.stamp(r.user)
	}
	ctx.stamp(old)
	ctx.stamp(newVal)
	return nil
}

// Erase removes n from the graph: it detaches all of n's operand uses from
// their defs' ref lists (killing the records), drops any statement holding n
// as its def or rhs from the single block, and tombstones n's ID so no later
// allocation can reuse it (IR-05) and the verifier flags any lingering use
// of it. Erase refuses while any live use of n remains — erasing a def with
// live users would strand dangling operand handles.
func (ctx *MutContext) Erase(n Node) error {
	if n == nil {
		return errors.New("ir: Erase on nil node")
	}
	if uses := UsesOf(n); len(uses) > 0 {
		return fmt.Errorf("ir: cannot erase node %d: %d use(s) still point at it", n.BasePtr().id, len(uses))
	}
	if impl, ok := n.(nodeImpl); ok {
		impl.forEachOperand(func(u SSAUse) {
			if u.rec == nil {
				return
			}
			if def := u.rec.def; def != nil {
				def.removeRef(u.rec)
			}
			u.rec.killed = true
		})
	}
	if g := n.BasePtr().graph; g != nil {
		// O(1) unlink through the statement's intrusive list links (review
		// P1R-02): the node's stmt back-pointer names its statement directly,
		// no scan of the block's statements.
		if s := n.BasePtr().stmt; s != nil {
			g.cfg.entry.Remove(s)
			n.BasePtr().stmt = nil
		}
	}
	n.BasePtr().erased = true
	ctx.stamp(n)
	return nil
}
