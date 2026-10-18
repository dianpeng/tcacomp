package ir

// mark.go is the transient node-held mark state for mark-sweep passes
// (quick-261018-l36 requirement 2) plus the sweep-enumeration face over the
// graph's allocation registry.
//
// D-15 EXCEPTION (stated here because it is the one piece of per-node state
// that is NOT IR content): the mark is transient ANALYSIS state owned by the
// mark-sweep driver (internal/opt/dce), never part of the graph's meaning.
// It is never printed, never serialized, never verified — print.go, dot.go,
// and verify.go do not read it. The driver resets marks before AND after
// every run (ResetMarks on entry and on defer), so stale state cannot leak
// across passes or protect dead nodes in a later sweep.

// MarkAlive records n as live for the current mark-sweep run. The mark has
// exactly two states — not-marked (false) and alive (true); there is no
// third "dead" state, because unmarked IS dead.
func (b *Base) MarkAlive() { b.mark = true }

// Alive reports the transient mark-sweep mark: false = not marked, true =
// alive for the current run. See the file comment for the D-15 exception.
func (b *Base) Alive() bool { return b.mark }

// ResetMarks clears the mark on every node the graph has ever allocated,
// including params and the end node. It is nil-safe. Mark-sweep drivers call
// it before marking (no stale alive marks) and after sweeping (no marks leak
// out of the pass).
func (g *Graph) ResetMarks() {
	if g == nil {
		return
	}
	for _, n := range g.nodes {
		n.BasePtr().mark = false
	}
}

// AllocatedNodes returns every non-erased node of the graph in allocation
// order — the sweep registry mark-sweep passes iterate. It mirrors the
// internal Level-1 registry (g.nodes) for mark-sweep passes, filtering
// erased tombstones; the returned slice is fresh, so callers may reorder or
// retain it freely.
func (g *Graph) AllocatedNodes() []Node {
	if g == nil {
		return nil
	}
	out := make([]Node, 0, len(g.nodes))
	for _, n := range g.nodes {
		if n == nil || n.BasePtr().erased {
			continue
		}
		out = append(out, n)
	}
	return out
}

// Tombstones returns every ERASED node of the graph in allocation order —
// the complement of AllocatedNodes over the same registry. Mark-sweep
// verification (internal/opt/dce's dce-verify pass) iterates it to assert
// that no live use record still points at a swept node: Erase kills the
// records it processes and refuses live-used nodes, so any record the
// tombstone's var still lists is a bookkeeping corruption worth naming.
func (g *Graph) Tombstones() []Node {
	if g == nil {
		return nil
	}
	out := make([]Node, 0)
	for _, n := range g.nodes {
		if n == nil || !n.BasePtr().erased {
			continue
		}
		out = append(out, n)
	}
	return out
}

// Erased reports whether the node was removed from the graph by a
// MutContext.Erase. Erased nodes keep their identity (Base cell, ID,
// SSA var) so verification can name them; only membership in the live
// registry is gone.
func (b *Base) Erased() bool { return b.erased }

// User returns the node holding this use slot, or nil for a zero/killed
// handle or a record with no user. It is the use-side mirror of Def — the
// def-use direction verification walks to name an offending edge (user →
// def) in diagnostics. It lives here rather than beside Def in ssa.go
// because only mark-sweep verification consumes it today; ssa.go keeps the
// minimal construction face.
func (u SSAUse) User() Node {
	if u.rec == nil || u.rec.killed || u.rec.user == nil {
		return nil
	}
	return u.rec.user
}
