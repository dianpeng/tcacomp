package finallower

// maplower.go is final-lower's MAP LOWER (LR-9, doc/optana/final_lower.md
// "Map Lower" as amended): a post-inline Map is a pure identity projector
// onto its lambda's body expression — eliminated with full debris disposal.
//
// Per eliminated Map, in allocation order:
//
//	1) RAUW the Map node onto its body expression node (lambdaNodeBody of
//	   LambdaOf(map)); a nil body or a non-lambda trailing operand is a
//	   named shape error (the graph is post-simplify — the shape is fixed).
//	2) EXPLICITLY erase the trailing As operand (positional metadata, no
//	   side effects — never left dangling).
//	3) Dispose of the orphaned lambda subtree per P-7: iteratively erase
//	   use-free nodes among the lambda node + its operand closure until a
//	   pass erases nothing (the mark-sweep discipline applied LOCALLY).
//	   NEVER force-erase a used node, and NEVER run a whole-graph
//	   BuildingDCE sweep: the Init/Reduce call statements final-lower emits
//	   have no users, and a roots-based sea sweep would erase them
//	   (reachability roots do not know runtime-call statements).
//
// Ordering (LR-2): Map Lower runs AFTER Reducer Lower — the @r derivation
// reads Reduce.op0 → producing Map → AsOf, so every Reduce is lowered
// before its producing Map disappears.
import (
	"fmt"

	"tcacomp/internal/ir"
)

// lowerMaps eliminates every live KindMap (LR-9).
func (lr *lowering) lowerMaps() error {
	for _, m := range lr.g.AllocatedNodes() {
		if m == nil || m.BasePtr() == nil || m.BasePtr().Erased() || m.KindOf() != ir.KindMap {
			continue
		}
		if err := lr.eliminateMap(m); err != nil {
			return err
		}
	}
	return nil
}

// eliminateMap performs one Map's elimination + debris disposal.
func (lr *lowering) eliminateMap(m ir.Node) error {
	lam := ir.LambdaOf(m)
	if lam == nil {
		return fmt.Errorf("final-lower: map node %d carries no trailing lambda node (post-simplify shape violation)",
			m.BasePtr().ID())
	}
	body := ir.LambdaBodies(m)
	if len(body) != 1 || body[0] == nil {
		return fmt.Errorf("final-lower: map node %d carries no lambda body expression", m.BasePtr().ID())
	}

	// 1) RAUW the Map onto its body expression, then erase the Map (its
	// remaining operand records — the As and the lambda — die with it).
	if err := lr.mctx.ReplaceAllUsesWith(m, body[0]); err != nil {
		return fmt.Errorf("final-lower: RAUW of map node %d onto its body: %w", m.BasePtr().ID(), err)
	}

	// Capture the debris candidates BEFORE erasing the Map: the trailing As
	// operand and the lambda node.
	as := ir.AsOf(m)
	candidates := []ir.Node{lam}
	if as != nil {
		candidates = append(candidates, as)
	}

	if err := lr.mctx.Erase(m); err != nil {
		return fmt.Errorf("final-lower: erasing map node %d: %w", m.BasePtr().ID(), err)
	}

	// 2) EXPLICITLY erase the trailing As (LR-9 step 2): the As is
	// positional metadata whose only user was the Map — but if some foreign
	// use survived, Erase's live-use refusal is the safety net (never
	// force-erase).
	if as != nil && !as.BasePtr().Erased() {
		if err := lr.mctx.Erase(as); err != nil {
			return fmt.Errorf("final-lower: erasing the map's trailing As node %d: %w", as.BasePtr().ID(), err)
		}
	}

	// 3) The conservative local sweep of the orphaned lambda subtree (P-7).
	return lr.sweepDebris(candidates)
}

// sweepDebris iteratively erases use-free nodes among candidates and their
// operand closure until a full pass erases nothing. A node still holding
// live uses is NEVER force-erased (Erase's refusal is the backstop) — it
// simply stays; a fixpoint with survivors is legitimate (the body
// expression is the RAUW target and stays alive through its new users).
func (lr *lowering) sweepDebris(candidates []ir.Node) error {
	// The closure over operand edges, seeded with the candidates.
	closure := make(map[ir.Node]bool)
	var add func(n ir.Node)
	add = func(n ir.Node) {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || closure[n] {
			return
		}
		closure[n] = true
		for _, op := range ir.OperandsOf(n) {
			add(op)
		}
	}
	for _, c := range candidates {
		add(c)
	}
	for changed := true; changed; {
		changed = false
		for n := range closure {
			if n.BasePtr().Erased() || len(ir.UsesOf(n)) > 0 {
				continue
			}
			if err := lr.mctx.Erase(n); err != nil {
				return fmt.Errorf("final-lower: debris sweep erasing node %d (%s): %w",
					n.BasePtr().ID(), n.KindOf(), err)
			}
			changed = true
		}
	}
	return nil
}
