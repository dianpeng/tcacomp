package ir

// Use is one operand slot: a single SSA use record in the intrusive
// use-list of the def it points at (IR-01, D-16). The record is heap-allocated
// per slot and referenced BY POINTER from the user node's operand storage
// (operandList.ops), so its address is stable across slot growth and removal;
// its prev/next links thread it into the def's list, and its back-pointers
// name both ends of the edge.
type Use struct {
	Val        Node // the def this operand currently points to (nil if unwired)
	User       Node // back-pointer to the owning node
	prev, next *Use // intrusive links in Val's use-list
}

// UsesOf returns a snapshot slice of the uses pointing at def, in list
// (link) order — most recently linked first, per the prepend discipline.
// The slice is rebuilt on every call, so callers may hold it across
// splice-during-iteration mutations (RESEARCH Pitfall 2).
func UsesOf(def Node) []*Use {
	if def == nil {
		return nil
	}
	var out []*Use
	for u := def.BasePtr().firstUse; u != nil; u = u.next {
		out = append(out, u)
	}
	return out
}

// DefOf returns the def the use points at.
func DefOf(u *Use) Node { return u.Val }

// linkIntoDef splices u (with u.Val already set) into def's intrusive
// use-list at the head. Used identically by construction and by the
// choke-point mutations.
func linkIntoDef(def *Base, u *Use) {
	u.prev = nil
	u.next = def.firstUse
	if def.firstUse != nil {
		def.firstUse.prev = u
	}
	def.firstUse = u
}

// unlinkFromDef removes u from def's intrusive use-list and clears u's
// links. u.Val is left untouched; callers decide whether to retarget or
// clear it.
func unlinkFromDef(def *Base, u *Use) {
	if u.prev != nil {
		u.prev.next = u.next
	} else {
		def.firstUse = u.next
	}
	if u.next != nil {
		u.next.prev = u.prev
	}
	u.prev, u.next = nil, nil
}
