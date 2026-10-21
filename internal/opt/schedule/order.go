package schedule

// order.go is the per-rank stable topological scheduling of spec step 1.4:
// the same-rank IR nodes of one block are scheduled in the partial order
// defined by data dependency (Input operand slots) and effect dependency
// (ImpactBy payload) — ir.PosetDeps already orders data edges first, effect
// edges second. Kahn's algorithm with an ALLOCATION-order tiebreak among
// ready nodes (smallest AllocatedNodes index wins) keeps the placement
// fully deterministic; a dependency cycle remainder (including a genuine
// self-dependency) fails closed naming the stuck node IDs.

import (
	"fmt"
	"sort"

	"tcacomp/internal/ir"
)

// topoRank schedules nodes — one rank's collection for one block — in
// dependency order: for each node, every live poset def that is also in
// the collection contributes a B-before-A edge (data slots then ImpactBy
// payload, per ir.PosetDeps), self-edges kept so a genuine self-dependency
// is unschedulable. Among ready nodes the smallest allocation index wins.
// A cycle remainder fails closed with the spec's message shape and leaves
// the caller to abandon the whole schedule — the graph's AST is never
// partially emitted.
func topoRank(rank int, nodes []ir.Node, allocIndex map[ir.Node]int) ([]ir.Node, error) {
	if len(nodes) == 0 {
		return nil, nil
	}
	inSet := make(map[ir.Node]bool, len(nodes))
	for _, n := range nodes {
		inSet[n] = true
	}
	indeg := make(map[ir.Node]int, len(nodes))
	succs := make(map[ir.Node][]ir.Node, len(nodes))
	edgeSeen := make(map[ir.Node]map[ir.Node]bool, len(nodes))
	for _, n := range nodes {
		for _, d := range ir.PosetDeps(n) {
			if d == nil || !inSet[d] {
				continue // cross-rank or foreign dep: another block's ordering problem
			}
			if edgeSeen[d] == nil {
				edgeSeen[d] = make(map[ir.Node]bool)
			}
			if edgeSeen[d][n] {
				continue // parallel operand slots referencing the same def: one edge
			}
			edgeSeen[d][n] = true
			indeg[n]++
			succs[d] = append(succs[d], n)
		}
	}

	var ready []ir.Node
	for _, n := range nodes {
		if indeg[n] == 0 {
			ready = append(ready, n)
		}
	}
	ordered := make([]ir.Node, 0, len(nodes))
	for len(ready) > 0 {
		// Allocation-order tiebreak: the ready node with the smallest
		// AllocatedNodes index wins (deterministic, map-iteration-free).
		best := 0
		for i := 1; i < len(ready); i++ {
			if allocIndex[ready[i]] < allocIndex[ready[best]] {
				best = i
			}
		}
		n := ready[best]
		ready = append(ready[:best], ready[best+1:]...)
		ordered = append(ordered, n)
		for _, o := range succs[n] {
			indeg[o]--
			if indeg[o] == 0 {
				ready = append(ready, o)
			}
		}
	}

	if len(ordered) != len(nodes) {
		emitted := make(map[ir.Node]bool, len(ordered))
		for _, n := range ordered {
			emitted[n] = true
		}
		var stuck []uint64
		for _, n := range nodes {
			if !emitted[n] {
				stuck = append(stuck, n.BasePtr().ID())
			}
		}
		sort.Slice(stuck, func(i, j int) bool { return stuck[i] < stuck[j] })
		return nil, fmt.Errorf("schedule: dependency cycle among rank-%d nodes (stuck: %v)", rank, stuck)
	}
	return ordered, nil
}
