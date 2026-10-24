package ir

import "errors"

// This file is the Level-1 → Level-2 transition of the two-level IR
// (quick-261017-vdj, tinycudaopto skeleton node.go): Materialize schedules
// the statement-free sea of nodes into the single-block CFG that codegen and
// the canonical printer consume.
//
// The scheduler is deliberately minimal (D-7: optimization scheduling is the
// pass.Scheduler's job, task 3 — ir keeps only this default) — v0 kernels
// have exactly one basic block and no control flow (IR-02), so "scheduling"
// is: one statement per LIVE effectful node (the memory-store family and the
// end node, dce.go's isEffectKind) reachable from the sea roots (R2/R3 — the
// end node's operand and ImpactBy paths), in node ALLOCATION order (which is
// construction order), each carrying its PinStatementID-reserved ID or the
// next ID from the graph's monotonic counter. Because reachability from the
// end node is the liveness test, statement order is exactly the construction
// order of the side effects — the same order the pre-scheduler flow appended
// statements, which the committed canonical goldens pin byte-for-byte.
//
// Shared pure values (a def used by several statements) do NOT become
// statements of their own: statement boundaries exist only at live effectful
// nodes, and everything reachable from a statement root renders inside that
// root's statement tree (the printer's chain walk handles shared subtrees).
// Passes that need assign statements insert them into the materialized block
// through NewStatement (cfg.go) and the ordinary list primitives.

// Materialize converts g from its Level-1 sea-of-nodes form into the Level-2
// scheduled CFG: it creates the single basic block and appends one statement
// per live effectful node reachable from the sea roots (skipped when
// tombstoned by DCE/Erase or unreachable — e.g. a store no end node
// references), in allocation order. Statement IDs come from each root's
// pinned ID (PinStatementID) or, for unpinned roots, from the graph's
// monotonic counter in emission order — the single counter keeps statement
// IDs dense with node IDs (IR-05/D-04). It returns an error when called
// twice, on a nil graph, or when a cfg already exists; the graph is usable
// at Level 1 regardless.
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
	live := markReachable(g)
	for _, n := range g.nodes {
		if n == nil || n.BasePtr().erased || !live[n] || !isEffectKind(n.KindOf()) {
			continue // DCE/pass-erased or unreachable from the sea roots: schedules nothing
		}
		bb.Append(g.NewStatement(n))
	}
	g.materialized = true
	return nil
}

// Dematerialize reverts g from its Level-2 scheduled CFG form back to the
// Level-1 sea-of-nodes form: the CFG (and its statements) is dropped and the
// materialized flag is cleared, so a full pass-manager pipeline — including
// the scheduling/tiling passes, which refuse materialized graphs (their
// gates expect the Level-1 sea) — can run over the graph. Frontends that
// build through pkg.Build (which Materializes as its final construction
// step) use this to hand the verified graph to opt.BuildPipeline before
// codegen (quick-261024-k2i). It returns an error when the graph is not
// materialized; Dematerialize/Run-pipeline/Materialize cycles are not
// supported (passes mutate the sea in place).
func (g *Graph) Dematerialize() error {
	if g == nil {
		return errors.New("ir: Dematerialize on nil graph")
	}
	if !g.materialized {
		return errors.New("ir: graph is not materialized")
	}
	g.cfg = nil
	g.materialized = false
	return nil
}
