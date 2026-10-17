package ir

import "errors"

// This file is the Level-1 → Level-2 transition of the two-level IR
// (quick-261017-vdj, tinycudaopto skeleton node.go): Materialize schedules
// the statement-free sea of nodes into the single-block CFG that codegen and
// the canonical printer consume.
//
// The scheduler is deliberately minimal — v0 kernels have exactly one basic
// block and no control flow (IR-02), so "scheduling" is: one statement per
// LIVE effect root, in effect-root registration order (which is allocation
// order — construction order), each carrying its PinStatementID-reserved ID
// or the next ID from the graph's monotonic counter. Because every store,
// tile store, and return registered itself in Graph.effectRoots at allocation
// time, statement order is exactly the construction order of the side
// effects — the same order the pre-scheduler flow appended statements, which
// the committed canonical goldens pin byte-for-byte.
//
// Shared pure values (a def used by several statements) do NOT become
// statements of their own: statement boundaries exist only at effect roots,
// and everything reachable from a root renders inside that root's statement
// tree (the printer's chain walk handles shared subtrees). Passes that need
// assign statements insert them into the materialized block through the
// ordinary list primitives.

// Materialize converts g from its Level-1 sea-of-nodes form into the Level-2
// scheduled CFG: it creates the single basic block and appends one statement
// per live effect root (skipped when nil or tombstoned by DCE/Erase), in
// effect-root order. Statement IDs come from each root's pinned ID
// (PinStatementID) or, for unpinned roots, from the graph's monotonic counter
// in emission order — the single counter keeps statement IDs dense with node
// IDs (IR-05/D-04). It returns an error when called twice, on a nil graph, or
// when a cfg already exists; the graph is usable at Level 1 regardless.
func (g *Graph) Materialize() error {
	if g == nil {
		return errors.New("ir: Materialize on nil graph")
	}
	if g.materialized {
		return errors.New("ir: graph already materialized")
	}
	if g.cfg != nil {
		return errors.New("ir: graph already has a CFG")
	}
	bb := &BasicBlock{graph: g}
	g.cfg = &CFG{entry: bb}
	for _, root := range g.effectRoots {
		if root == nil || root.BasePtr().erased {
			continue // DCE or a pass erased the effect: it schedules nothing
		}
		id := root.BasePtr().stmtID
		if id == 0 {
			id = g.NewID()
		}
		s := &Statement{id: id, kind: classifyStmt(root), rhs: root}
		root.BasePtr().stmt = s
		if s.kind == StmtAssign {
			s.def = SSADef{ssa: root.BasePtr().ssa}
		}
		bb.Append(s)
	}
	g.materialized = true
	return nil
}
