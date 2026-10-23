package finallower

// objects.go is final-lower's RUNTIME OBJECT LOWER (LR-11/LR-12,
// doc/optana/final_lower.md "Runtime Object Lower" as amended):
//
//   - exactly ONE RuntimeKernelInfo, prepended as the FIRST item of the
//     kernel Body Group (the Prototype block is signature scope —
//     untouched), stamped SN 0;
//   - per LNT (one pair, ALWAYS): RuntimeGridMapping + RuntimeThreadMapping
//     prepended to the OWNING LNT's prologue Group — never the kernel's
//     first Group — grid before thread, both stamped SN 0/prologue, their
//     portion vectors from the provenance record's per-LNT MappingPlan
//     (P-3's encoding: kernel_info operand, TI32 ndim + portions);
//   - every pre-final-lower launch-map query (RuntimeGridQuery/
//     RuntimeThreadQuery) rewritten into RuntimeGetAttribute(%<owning
//     LNT's mapping object>, field, dosoc): the binding becomes EXPLICIT
//     SSA for both backends (MLIR has no lexical runtime scope to recover
//     it from); after the rewrite the query kinds vanish from the graph
//     (LR-6/LR-12).
import (
	"fmt"

	"tcacomp/internal/ir"
)

// insertRuntimeObjects builds the kernel information object and every LNT's
// mapping pair (LR-11).
func (lr *lowering) insertRuntimeObjects() error {
	// The single KernelInfo: item 0 of the kernel Body Group. Re-running
	// the pass is impossible (one schedule per graph), but a duplicated
	// construction in the incoming graph is still a named error.
	for _, n := range lr.g.AllocatedNodes() {
		if n != nil && !n.BasePtr().Erased() && n.KindOf() == ir.KindRuntimeKernelInfo {
			return fmt.Errorf("final-lower: graph already carries a RuntimeKernelInfo (node %d)", n.BasePtr().ID())
		}
	}
	ki := lr.g.NewRuntimeKernelInfo(lowerOrigin)
	if err := lr.ast.Body.InsertItemAt(0, ir.ASTItem{Node: ki}); err != nil {
		return fmt.Errorf("final-lower: prepending the KernelInfo to the kernel Body Group: %w", err)
	}
	ki.BasePtr().SetSched(0, lr.ast.Body)

	// Per-LNT mapping pairs: prepended to the OWNING prologue Group. The
	// grid mapping goes FIRST (grid before thread); prepending thread then
	// grid leaves [grid, thread, ...existing].
	for _, lnt := range lr.ast.LNTs {
		prologue, err := lr.prologueOf(lnt)
		if err != nil {
			return err
		}
		plan, ok := lr.prov.LNTs[lnt.Root]
		if !ok {
			return fmt.Errorf("final-lower: LNT root node %d carries no mapping plan in the tiling provenance",
				lnt.Root.BasePtr().ID())
		}
		thread, err := lr.prependMapping(ir.KindRuntimeThreadMapping, prologue, ki, plan.ThreadPortions)
		if err != nil {
			return err
		}
		grid, err := lr.prependMapping(ir.KindRuntimeGridMapping, prologue, ki, plan.GridPortions)
		if err != nil {
			return err
		}
		lr.gridOf[lnt] = grid
		lr.threadOf[lnt] = thread
	}
	return nil
}

// prologueOf resolves the LNT's lexical Group (quick-261023-lex: the
// schedule pass's per-LNT cluster is ONE Group embedding the nest — the
// head region is the prologue): the kernel Body item whose Block is a
// Group with Nest() == lnt. A missing group is a named
// error.
func (lr *lowering) prologueOf(lnt *ir.ASTLNT) (*ir.ASTBlock, error) {
	for _, it := range lr.ast.Body.Items() {
		if it.Block == nil || it.Block.Kind() != ir.BlockGroup {
			continue
		}
		if it.Block.Nest() == lnt {
			return it.Block, nil
		}
	}
	return nil, fmt.Errorf("final-lower: LNT root node %d has no prologue Group in the kernel Body",
		lnt.Root.BasePtr().ID())
}

// prependMapping builds one mapping construction and prepends it to the
// prologue Group, stamped SN 0/prologue (P-10).
func (lr *lowering) prependMapping(kind ir.Kind, prologue *ir.ASTBlock, ki ir.Node, portions []int) (ir.Node, error) {
	var obj ir.Node
	switch kind {
	case ir.KindRuntimeGridMapping:
		obj = lr.g.NewRuntimeGridMapping(lowerOrigin, ki, portions)
	case ir.KindRuntimeThreadMapping:
		obj = lr.g.NewRuntimeThreadMapping(lowerOrigin, ki, portions)
	default:
		return nil, fmt.Errorf("final-lower: unknown mapping kind %s", kind)
	}
	if err := prologue.InsertItemAt(0, ir.ASTItem{Node: obj}); err != nil {
		return nil, fmt.Errorf("final-lower: prepending a %s to the LNT prologue: %w", kind, err)
	}
	obj.BasePtr().SetSched(0, prologue)
	return obj, nil
}

// rewriteQueries rewrites every live launch-map query into an explicit
// attribute read on the owning LNT's mapping object (LR-12): grid queries →
// the GridMapping, thread queries → the ThreadMapping; the getattr carries
// the same field payload, the resolved mapping object as operand 0, and the
// query's DoSoC const as its single arg; every query use is RAUW'd onto the
// getattr; the query node is erased; the query's AST item is replaced IN
// PLACE with the getattr item, stamped with the query's own SN in the same
// block (P-10).
func (lr *lowering) rewriteQueries() error {
	for _, q := range lr.g.AllocatedNodes() {
		if q == nil || q.BasePtr() == nil || q.BasePtr().Erased() {
			continue
		}
		var kind ir.Kind
		switch q.KindOf() {
		case ir.KindRuntimeGridQuery:
			kind = ir.KindRuntimeGridQuery
		case ir.KindRuntimeThreadQuery:
			kind = ir.KindRuntimeThreadQuery
		default:
			continue
		}
		field := ir.RuntimeAttributeOf(q)
		dosoc := ir.OperandsOf(q)
		if len(dosoc) != 1 || dosoc[0] == nil {
			return fmt.Errorf("final-lower: query node %d carries a malformed DoSoC operand", q.BasePtr().ID())
		}

		// Resolution: the query's SchedBlock → owning nest → that LNT's
		// construction (a query outside every LNT cluster cannot be bound).
		blk := q.BasePtr().SchedBlock()
		lnt, err := lr.lntOf(blk, fmt.Sprintf("query node %d", q.BasePtr().ID()))
		if err != nil {
			return err
		}
		var obj ir.Node
		if kind == ir.KindRuntimeGridQuery {
			obj = lr.gridOf[lnt]
		} else {
			obj = lr.threadOf[lnt]
		}
		if obj == nil {
			return fmt.Errorf("final-lower: LNT root node %d has no %s object (insertRuntimeObjects did not run?)",
				lnt.Root.BasePtr().ID(), kind)
		}

		// The getattr: same field payload, the mapping object, the query's
		// DoSoC const as the single arg (LR-12: "DoSoC const preserved,
		// field payload preserved").
		getattr := lr.g.NewRuntimeGetAttribute(lowerOrigin, field, obj, dosoc[0])

		if err := lr.mctx.ReplaceAllUsesWith(q, getattr); err != nil {
			return fmt.Errorf("final-lower: RAUW of query node %d: %w", q.BasePtr().ID(), err)
		}
		if err := lr.mctx.Erase(q); err != nil {
			return fmt.Errorf("final-lower: erasing query node %d: %w", q.BasePtr().ID(), err)
		}

		// Replace the query's AST item IN PLACE with the getattr, stamped
		// with the query's own SN in the same block.
		if qBlk, idx := lr.findItem(q); qBlk != nil && idx >= 0 {
			removed := qBlk.RemoveItemsIf(func(it ir.ASTItem) bool {
				return it.Node != nil && it.Node.BasePtr() == q.BasePtr()
			})
			if removed != 1 {
				return fmt.Errorf("final-lower: query node %d's AST item removal count = %d, want 1",
					q.BasePtr().ID(), removed)
			}
			if err := qBlk.InsertItemAt(idx, ir.ASTItem{Node: getattr}); err != nil {
				return fmt.Errorf("final-lower: re-inserting query node %d's replacement: %w", q.BasePtr().ID(), err)
			}
			getattr.BasePtr().SetSched(q.BasePtr().SchedSN(), qBlk)
		} else {
			// A query with no AST item: it sits purely inside an
			// expression (never an item itself). The pass still rewires its
			// USES; stamp the getattr with the query's SN/block linkage so
			// codegen sees a consistent scope.
			getattr.BasePtr().SetSched(q.BasePtr().SchedSN(), blk)
		}
	}
	return nil
}
