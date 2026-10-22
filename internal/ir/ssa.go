package ir

// This file is the SSA handle layer (directive 2): SSAVar is the sole
// pointee that stores the use-def/def-use relationship; SSADef and SSAUse are
// value-type handles over it (seabird mechanics adapted — pattern only, no
// code copied). Handles are copied freely across passes; because a use handle
// points at a ref RECORD rather than directly at the var, rewiring the
// record's def pointer propagates the new target to EVERY copy of the handle.
// Invalid/zombie states (zero handles, killed records) are representable but
// checkable — the verifier flags them structurally under the use-def
// invariant, and every accessor here is nil-safe (T-1.1-03).
//
// All chain edits flow through the MutContext choke points in mutate.go
// (IR-04); the var-level primitives below (replaceDefine, replaceAllReference,
// addRef, removeRef) are unexported and reachable only from there and from
// construction (construct.go's construction exemption).

// SSAVar is the per-node SSA cell: one per node, allocated at construction
// (ID == node ID, IR-05's single NewID source), holding the node's defining
// Value and the list of ref records for uses pointing AT this node.
type SSAVar struct {
	id     uint64
	node   Node            // back-pointer to the owning node (nil-safe on zero var)
	define Value           // the value this var defines (nil-safe on zero var)
	refs   []*ssaRefRecord // uses pointing at this var, in link (append) order
}

// ID returns the var's stable ID (equal to its node's ID). Nil-safe.
func (v *SSAVar) ID() uint64 {
	if v == nil {
		return 0
	}
	return v.id
}

// Node returns the var's owning node, or nil on a zero var. Nil-safe.
func (v *SSAVar) Node() Node {
	if v == nil {
		return nil
	}
	return v.node
}

// Define returns the value the var defines, or nil on a zero var. Nil-safe.
func (v *SSAVar) Define() Value {
	if v == nil {
		return nil
	}
	return v.define
}

// RefSize returns the number of LIVE (non-killed) ref records the var holds.
// Nil-safe.
func (v *SSAVar) RefSize() int {
	if v == nil {
		return 0
	}
	n := 0
	for _, r := range v.refs {
		if r != nil && !r.killed {
			n++
		}
	}
	return n
}

// replaceDefine swaps the var's defining value. Unexported: reachable only
// from the MutContext choke points (IR-04).
func (v *SSAVar) replaceDefine(d Value) { v.define = d }

// liveRefs returns a snapshot slice of the var's live ref records, in link
// order. The slice is rebuilt on every call, so callers may hold it across
// rewiring mutations (RESEARCH Pitfall 2).
func (v *SSAVar) liveRefs() []*ssaRefRecord {
	var out []*ssaRefRecord
	for _, r := range v.refs {
		if r != nil && !r.killed {
			out = append(out, r)
		}
	}
	return out
}

// addRef appends one live ref record naming user to the var's list and
// returns it. Unexported: construction and MutContext only.
func (v *SSAVar) addRef(user Node) *ssaRefRecord {
	r := &ssaRefRecord{def: v, user: user}
	v.refs = append(v.refs, r)
	return r
}

// removeRef drops r from the var's list by record identity. Unexported.
func (v *SSAVar) removeRef(r *ssaRefRecord) {
	for i, s := range v.refs {
		if s == r {
			v.refs = append(v.refs[:i], v.refs[i+1:]...)
			return
		}
	}
}

// replaceAllReference relocates every live ref record of v into nv's list in
// one pass (RAUW's whole-chain move). The records' def pointers are rewired,
// so every SSAUse handle copy sees the new target. Unexported: reachable only
// from the MutContext choke points (IR-04).
func (v *SSAVar) replaceAllReference(nv *SSAVar) {
	if nv == nil {
		return
	}
	kept := v.refs[:0] // live records move out; killed records stay listed-but-inert
	for _, r := range v.refs {
		if r == nil || r.killed {
			kept = append(kept, r)
			continue
		}
		r.def = nv
		nv.refs = append(nv.refs, r)
	}
	v.refs = kept
}

// ssaRefRecord is the pointer-to-pointer indirection at the heart of the
// handle model: an operand slot holds an SSAUse handle that POINTS AT this
// record, and the record points AT the def's SSAVar. Rewiring (SetDef, RAUW)
// changes rec.def; every copy of the handle — each operand slot holds one —
// resolves through the same record and so sees the new target immediately.
type ssaRefRecord struct {
	def    *SSAVar // the def var this use currently points at (nil if unwired)
	user   Node    // back-pointer to the owning node
	killed bool    // set by RemoveUse: the record is detached from both directions
}

// SSADef is the value-type def handle: a wrapper over the node's SSAVar. The
// zero value is the unwired/invalid def (a statement's def before a pass
// wires it). Copies are free; all accessors are nil-safe.
type SSADef struct {
	ssa *SSAVar
}

// SSA returns the handle's var, or nil on a zero def. Nil-safe.
func (d SSADef) SSA() *SSAVar { return d.ssa }

// IsInvalid reports whether the handle is the zero (unwired) def; it replaces
// the old "def == nil" checks.
func (d SSADef) IsInvalid() bool { return d.ssa == nil }

// Define returns the value the def names, or nil on a zero def. Nil-safe.
func (d SSADef) Define() Value {
	if d.ssa == nil {
		return nil
	}
	return d.ssa.define
}

// SSAUse is the value-type use handle: a wrapper over one ref record. Copies
// share the record, so a mutation through any copy propagates to all of them
// (directive 2's reverse-operation requirement). The zero value is an unwired
// slot. All accessors are nil-safe.
type SSAUse struct {
	rec *ssaRefRecord
}

// SSA returns the def var this use currently resolves to, or nil for a zero
// handle, a killed record, or an unwired (nil-target) record. Nil-safe.
func (u SSAUse) SSA() *SSAVar {
	if u.rec == nil || u.rec.killed {
		return nil
	}
	return u.rec.def
}

// IsKilled reports whether the handle is unusable: the zero handle, a killed
// record, or a nil target all read as killed.
func (u SSAUse) IsKilled() bool {
	return u.rec == nil || u.rec.killed || u.rec.def == nil
}

// Def returns the def node this use points at, or nil for a zero/killed/
// unwired handle. Nil-safe.
func (u SSAUse) Def() Node {
	if u.rec == nil || u.rec.killed {
		return nil
	}
	if u.rec.def == nil {
		return nil
	}
	return u.rec.def.node
}

// UsesOf returns a snapshot slice of the uses pointing at def, in link
// (append) order. The slice is rebuilt on every call, so callers may hold it
// across splice-during-iteration mutations (RESEARCH Pitfall 2). A nil def
// yields nil.
func UsesOf(def Node) []SSAUse {
	if def == nil {
		return nil
	}
	v := def.BasePtr().ssa
	if v == nil {
		return nil
	}
	var out []SSAUse
	for _, r := range v.liveRefs() {
		out = append(out, SSAUse{rec: r})
	}
	return out
}

// DefOf returns the def node the use points at. Nil-safe: a zero or killed
// use yields nil.
func DefOf(u SSAUse) Node { return u.Def() }
