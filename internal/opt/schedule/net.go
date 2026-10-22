package schedule

// net.go is the post-placement cross-block ordering net (quick-261022-b2m,
// locked decision 2 + LO-03): the EMPIRICAL GUARD of the uniform-relation
// placement model — the net that would have caught CR-01/CR-02 (the clamp
// scheduled fills and unenclosed consumers BEFORE their in-loop
// dependencies, silently). It runs after the per-root placement loops and
// BEFORE the kernel metadata/AttachAST, so an error leaves the AST
// unattached — the existing never-a-partial-schedule discipline.
//
// Two checks, both deterministic by construction (allocation order and
// placement order only, never map iteration):
//
//   - LEFTOVER: every live node landed in SOME block (the old net, subsumed
//     — a leftover names the sorted offending IDs).
//   - ORDER: for every live node and every live PosetDeps def, the def's
//     block must not execute AFTER the user's block in the kernel's linear
//     block order. Equal indices are LEGAL (same block — intra-block order
//     is topoRank's contract). The first violation in the deterministic
//     scan order fails closed naming both node IDs and kinds.

import (
	"fmt"
	"sort"

	"tcacomp/internal/ir"
)

// verifyBlockOrder flattens the kernel's blocks into a linear execution
// index, then checks that every live node landed in a block (leftover) and
// that no live node's block executes before any of its live PosetDeps'
// blocks (cross-block order). A nil kernel fails closed. The block index is
// assigned at DFS ENTRY over k.Body — items in placement order, Block items
// recursed at their position — so a cluster's [prologue, LBBN, epilogue]
// order and the per-root concatenation give exactly the kernel's execution
// order.
func verifyBlockOrder(k *ir.ASTKernel, nodes []ir.Node, live map[ir.Node]bool) error {
	if k == nil || k.Body == nil {
		return fmt.Errorf("schedule: no kernel body to verify (nil AST)")
	}
	blockIndex := make(map[*ir.ASTBlock]int)
	parent := make(map[*ir.ASTBlock]*ir.ASTBlock)
	var flatten func(b *ir.ASTBlock, p *ir.ASTBlock)
	flatten = func(b *ir.ASTBlock, p *ir.ASTBlock) {
		blockIndex[b] = len(blockIndex)
		parent[b] = p
		for _, it := range b.Items() {
			if it.Block != nil {
				flatten(it.Block, b)
			}
		}
	}
	flatten(k.Body, nil)

	// blockContains reports whether desc is b itself or lives in b's item
	// subtree (the parent chain built during flatten).
	blockContains := func(b, desc *ir.ASTBlock) bool {
		for cur := desc; cur != nil; cur = parent[cur] {
			if cur == b {
				return true
			}
		}
		return false
	}

	// LEFTOVER (subsumes the old net, same message shape): a live node with
	// no landing block means the placement missed it — never a silently
	// partial AST.
	var leftover []uint64
	for _, n := range nodes {
		if live[n] && n.BasePtr().SchedBlock() == nil {
			leftover = append(leftover, n.BasePtr().ID())
		}
	}
	if len(leftover) > 0 {
		sort.Slice(leftover, func(i, j int) bool { return leftover[i] < leftover[j] })
		return fmt.Errorf("schedule: %d live nodes landed in no block (ids: %v)", len(leftover), leftover)
	}

	// ORDER: blockIndex(d) > blockIndex(n) is a cross-block inversion — the
	// user's block executes before a block its own dependency lives in. A
	// live node whose SchedBlock is not part of the kernel AST also fails
	// closed (a foreign block's default index 0 could otherwise mask real
	// inversions). First violation in the deterministic scan order wins.
	for _, n := range nodes {
		if !live[n] {
			continue
		}
		ni, ok := blockIndex[n.BasePtr().SchedBlock()]
		if !ok {
			return fmt.Errorf("schedule: live node %d (%s) landed in a block outside the kernel AST",
				n.BasePtr().ID(), n.KindOf())
		}
		for _, d := range ir.PosetDeps(n) {
			if d == nil || !live[d] {
				continue
			}
			di, ok := blockIndex[d.BasePtr().SchedBlock()]
			if !ok {
				return fmt.Errorf("schedule: live node %d (%s) landed in a block outside the kernel AST",
					d.BasePtr().ID(), d.KindOf())
			}
			// LOOP-BODY CONTAINMENT EDGE: a ForLoop's body operand IS the
			// nested loop (use→def), so the parent loop node "depends on"
			// its child — but that edge denotes CONTAINMENT (the body runs
			// INSIDE the loop block), not before-use ordering. The child's
			// block naturally carries a higher linear index. Only the loop
			// nodes get the carve-out: a STATEMENT whose dep sits in a
			// deeper block of its own subtree cannot exist (SN propagation
			// would have raised its SN), so statements stay fully guarded.
			if n.KindOf() == ir.KindForLoop && blockContains(n.BasePtr().SchedBlock(), d.BasePtr().SchedBlock()) {
				continue
			}
			if di > ni {
				return fmt.Errorf("schedule: node %d (%s) is scheduled before its dependency node %d (%s) — cross-block order violated",
					n.BasePtr().ID(), n.KindOf(), d.BasePtr().ID(), d.KindOf())
			}
		}
	}
	return nil
}
