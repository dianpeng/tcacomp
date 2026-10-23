package schedule

// net.go is the post-placement cross-block ordering net (quick-261022-b2m,
// locked decision 2 + LO-03; positions model quick-261023-lex): the
// EMPIRICAL GUARD of the uniform-relation placement model — the net that
// would have caught CR-01/CR-02 (the clamp scheduled fills and unenclosed
// consumers BEFORE their in-loop dependencies, silently). It runs after the
// per-root placement loops and BEFORE the kernel metadata/AttachAST, so an
// error leaves the AST unattached — the existing never-a-partial-schedule
// discipline.
//
// Two checks, both deterministic by construction (allocation order and
// placement order only, never map iteration):
//
//   - LEFTOVER: every live node landed in SOME block (the old net, subsumed
//     — a leftover names the sorted offending IDs).
//   - ORDER: for every live node and every live PosetDeps def, the def must
//     not execute AFTER the user. Execution order is compared by ITEM
//     POSITION PATHS, not block entry indices: since quick-261023-lex one
//     lexical Group holds items BOTH before and after its embedded loop
//     nest, so a block's DFS-entry index alone cannot order its own items
//     against the nest's contents. A node's position is the item-index path
//     from the kernel Body down to the node's own item; a node stamped into
//     a block without being an item there (loop headers, the forloop node
//     itself) executes at block ENTRY — the path plus a virtual −1 that
//     precedes every body item. The first violation in the deterministic
//     scan order fails closed naming both node IDs and kinds.

import (
	"fmt"
	"sort"

	"tcacomp/internal/ir"
)

// verifyBlockOrder checks that every live node landed in a block (leftover)
// and that no live node executes before any of its live PosetDeps' (cross-
// block order, by item-position paths). A nil kernel fails closed.
func verifyBlockOrder(k *ir.ASTKernel, nodes []ir.Node, live map[ir.Node]bool) error {
	if k == nil || k.Prototype == nil || k.Body == nil {
		return fmt.Errorf("schedule: kernel missing its Prototype or Body (nil AST)")
	}

	// Positions: blockPath maps each block to its item-index path from the
	// kernel root (Prototype = [-1], Body = []); nodePos maps each Node
	// item to its block's path plus its item index; a block's non-item
	// stamped nodes (loop headers) are not in nodePos and fall back to the
	// block's path plus −1 (block entry). parent drives containment.
	blockPath := make(map[*ir.ASTBlock][]int)
	parent := make(map[*ir.ASTBlock]*ir.ASTBlock)
	nodePos := make(map[ir.Node][]int)
	var flatten func(b *ir.ASTBlock, p *ir.ASTBlock, path []int)
	flatten = func(b *ir.ASTBlock, p *ir.ASTBlock, path []int) {
		blockPath[b] = path
		parent[b] = p
		for i, it := range b.Items() {
			if it.Node != nil {
				pos := make([]int, 0, len(path)+1)
				pos = append(pos, path...)
				nodePos[it.Node] = append(pos, i)
				continue
			}
			if it.Block != nil {
				child := make([]int, 0, len(path)+1)
				child = append(child, path...)
				flatten(it.Block, b, append(child, i))
			}
		}
	}
	// PROTOTYPE FIRST (quick-261022-e2u, D-2): the signature block renders
	// before the body (path [-1] sorts below every Body position), so every
	// -1 dep is trivially legal.
	flatten(k.Prototype, nil, []int{-1})
	flatten(k.Body, nil, []int{})

	// posOf returns n's item-position path (see the file comment).
	posOf := func(n ir.Node) ([]int, bool) {
		if pos, ok := nodePos[n]; ok {
			return pos, true
		}
		path, ok := blockPath[n.BasePtr().SchedBlock()]
		if !ok {
			return nil, false
		}
		pos := make([]int, 0, len(path)+1)
		pos = append(pos, path...)
		return append(pos, -1), true // block entry: before every body item
	}

	// before reports whether position a executes strictly before b
	// (lexicographic; a strict-prefix pair cannot occur — an item is either
	// a Node or a Block, so two positions diverge at or before the shorter
	// length, and equal positions mean the same block, which the caller
	// handles separately).
	before := func(a, b []int) bool {
		for i := 0; i < len(a) && i < len(b); i++ {
			if a[i] != b[i] {
				return a[i] < b[i]
			}
		}
		return len(a) < len(b)
	}

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

	// ORDER: pos(d) executing strictly after pos(n) is a cross-block
	// inversion. Same-block pairs are INTRA-block order — topoRank's
	// contract, not this net's. First violation in the deterministic scan
	// order wins.
	for _, n := range nodes {
		if !live[n] {
			continue
		}
		nPos, ok := posOf(n)
		if !ok {
			return fmt.Errorf("schedule: live node %d (%s) landed in a block outside the kernel AST",
				n.BasePtr().ID(), n.KindOf())
		}
		for _, d := range ir.PosetDeps(n) {
			if d == nil || !live[d] {
				continue
			}
			if d.BasePtr().SchedBlock() == n.BasePtr().SchedBlock() {
				continue // same block: intra-block order is topoRank's contract
			}
			dPos, ok := posOf(d)
			if !ok {
				return fmt.Errorf("schedule: live node %d (%s) landed in a block outside the kernel AST",
					d.BasePtr().ID(), d.KindOf())
			}
			// LOOP-BODY CONTAINMENT EDGE: a ForLoop's body operand IS the
			// nested loop (use→def), so the parent loop node "depends on"
			// its child — but that edge denotes CONTAINMENT (the body runs
			// INSIDE the loop block), not before-use ordering. Only the
			// loop nodes get the carve-out: a STATEMENT whose dep sits in a
			// deeper block of its own subtree cannot exist (SN propagation
			// would have raised its SN), so statements stay fully guarded.
			if n.KindOf() == ir.KindForLoop && blockContains(n.BasePtr().SchedBlock(), d.BasePtr().SchedBlock()) {
				continue
			}
			// SIGNATURE-EDGE CARVE-OUT (quick-261022-e2u, D-4): a prototype
			// placeholder (InputParam/MemNew, SN -1) may reference scheduled
			// values — a MemNew's extent const stays at SN 0 in a lexical
			// head — but a placeholder is DECLARATION, not execution:
			// nothing executes inside the signature scope, so its position
			// asserts nothing about runtime order. Prototype-landed
			// DEPENDENTS (scheduled users of a placeholder) stay guarded.
			if n.BasePtr().SchedSN() == ir.PrototypeSN && blockContains(k.Prototype, n.BasePtr().SchedBlock()) {
				continue
			}
			if before(nPos, dPos) {
				return fmt.Errorf("schedule: node %d (%s) is scheduled before its dependency node %d (%s) — cross-block order violated",
					n.BasePtr().ID(), n.KindOf(), d.BasePtr().ID(), d.KindOf())
			}
		}
	}
	return nil
}
