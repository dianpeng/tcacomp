package ir

import (
	"errors"
	"fmt"
)

// MutContext identifies the running pass at every choke-point mutation
// (IR-04, D-17). The IR stamps LastTouchedBy = ctx.Pass on every node a
// mutation touches, making PASS-06's last-touched-by diagnostics structural
// rather than advisory. Construction (construct.go) is exempt; every other
// mutation of the chains flows through these five functions.
type MutContext struct {
	Pass string
}

// stamp records ctx.Pass as the last pass to touch n.
func (ctx *MutContext) stamp(n Node) {
	if n != nil {
		n.BasePtr().lastTouchedBy = ctx.Pass
	}
}

// SetDef redirects one operand use to a new def, unlinking it from the old
// def's use-list and splicing it into the new one. A nil newVal unwires the
// slot (the verifier's "nil def" invariant flags it).
func (ctx *MutContext) SetDef(u *Use, newVal Node) error {
	if u == nil {
		return errors.New("ir: SetDef on nil use")
	}
	old := u.Val
	if old != nil {
		unlinkFromDef(old.BasePtr(), u)
	}
	u.Val = newVal
	if newVal != nil {
		linkIntoDef(newVal.BasePtr(), u)
	}
	ctx.stamp(u.User)
	ctx.stamp(old)
	ctx.stamp(newVal)
	return nil
}

// AddUse appends one operand slot to user and links it into val's use-list,
// returning the new Use. It returns nil if user is not an IR node.
func (ctx *MutContext) AddUse(user Node, val Node) *Use {
	impl, ok := user.(nodeImpl)
	if !ok {
		return nil
	}
	u := impl.addSlot()
	u.Val = val
	u.User = user
	if val != nil {
		linkIntoDef(val.BasePtr(), u)
	}
	ctx.stamp(user)
	ctx.stamp(val)
	return u
}

// RemoveUse detaches u from its def's use-list and drops u's slot from the
// user's operand storage. The use is fully removed from both directions of
// the chain.
func (ctx *MutContext) RemoveUse(u *Use) {
	if u == nil {
		return
	}
	user := u.User
	if u.Val != nil {
		unlinkFromDef(u.Val.BasePtr(), u)
	}
	if impl, ok := user.(nodeImpl); ok {
		impl.removeSlot(u)
	}
	ctx.stamp(user)
	u.Val, u.User = nil, nil
}

// ReplaceAllUsesWith splices every use of old onto newVal in O(uses),
// mirroring LLVM's replaceAllUsesWith: each use record is unlinked from
// old's list and linked into newVal's, in one pass. Self-RAUW is a no-op;
// each use's next link is snapshotted BEFORE splicing so the walk survives
// its own mutation (RESEARCH Pitfall 2).
func (ctx *MutContext) ReplaceAllUsesWith(old, newVal Node) error {
	if old == nil || newVal == nil {
		return errors.New("ir: ReplaceAllUsesWith on nil node")
	}
	if old == newVal {
		return nil
	}
	ob, nb := old.BasePtr(), newVal.BasePtr()
	for u := ob.firstUse; u != nil; {
		next := u.next // snapshot BEFORE unlink clears it
		unlinkFromDef(ob, u)
		u.Val = newVal
		linkIntoDef(nb, u)
		ctx.stamp(u.User)
		u = next
	}
	ctx.stamp(old)
	ctx.stamp(newVal)
	return nil
}

// Erase removes n from the graph: it detaches all of n's operand uses from
// their defs' use-lists, drops any statement holding n as its def or rhs
// from the single block, and tombstones n's ID so no later allocation can
// reuse it (IR-05) and the verifier flags any lingering use of it. Erase
// refuses while any use of n remains — erasing a def with live users would
// strand dangling operand pointers.
func (ctx *MutContext) Erase(n Node) error {
	if n == nil {
		return errors.New("ir: Erase on nil node")
	}
	if uses := UsesOf(n); len(uses) > 0 {
		return fmt.Errorf("ir: cannot erase node %d: %d use(s) still point at it", n.BasePtr().id, len(uses))
	}
	if impl, ok := n.(nodeImpl); ok {
		impl.forEachOperand(func(u *Use) {
			if u.Val != nil {
				unlinkFromDef(u.Val.BasePtr(), u)
			}
			u.Val, u.User = nil, nil
		})
	}
	if g := n.BasePtr().graph; g != nil {
		kept := g.block.statements[:0]
		for _, s := range g.block.statements {
			if s.rhs == n || s.def == n {
				continue
			}
			kept = append(kept, s)
		}
		for i := len(kept); i < len(g.block.statements); i++ {
			g.block.statements[i] = nil
		}
		g.block.statements = kept
	}
	n.BasePtr().erased = true
	ctx.stamp(n)
	return nil
}
