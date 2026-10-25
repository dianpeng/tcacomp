package finallower

// maplower.go is final-lower's MAP LOWER (LR-9, doc/optana/final_lower.md
// "Map Lower" as amended by fix2 D-b/R-A): a post-inline Map is a pure
// identity projector onto its lambda's body expression — eliminated by the
// verb's three prescribed steps, and NOTHING else:
//
//	1) RAUW the Map node onto its body expression node (lambdaNodeBody of
//	   LambdaOf(map)); a nil body or a non-lambda trailing operand is a
//	   named shape error (the graph is post-simplify — the shape is fixed).
//	2) erase the Map node (its remaining operand records die with it).
//	3) EXPLICITLY erase the trailing As operand (positional metadata, no
//	   side effects — never left dangling; Erase's live-use refusal is the
//	   backstop against a foreign use).
//
// The orphaned lambda subtree is NOT this pass's business (fix2 D-b/R-A:
// lower is a separate pass that does transformation, DCE is beyond its
// job): the lower never drops statements — the pipeline's tail DCE owns
// the debris. This deletes the historical "full debris disposal" sweep,
// which asked a liveness question inside a lower (R-A's prohibition).
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

// eliminateMap performs one Map's elimination — exactly the verb (fix2
// D-b): RAUW onto the body, erase the Map, erase the trailing As. Nothing
// else: no debris sweeps, no orphan collection (the tail DCE's domain).
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

	// 1) RAUW the Map onto its body expression.
	if err := lr.mctx.ReplaceAllUsesWith(m, body[0]); err != nil {
		return fmt.Errorf("final-lower: RAUW of map node %d onto its body: %w", m.BasePtr().ID(), err)
	}

	// Capture the trailing As BEFORE erasing the Map: AsOf reads the Map's
	// operand records, which Erase kills.
	as := ir.AsOf(m)

	// 2) Erase the Map (its remaining operand records — the As and the
	// lambda — die with it).
	if err := lr.mctx.Erase(m); err != nil {
		return fmt.Errorf("final-lower: erasing map node %d: %w", m.BasePtr().ID(), err)
	}

	// 3) Erase the trailing As (LR-9 step 3): positional metadata whose only
	// user was the Map — but if some foreign use survived, Erase's live-use
	// refusal is the safety net (never force-erase).
	if as != nil && !as.BasePtr().Erased() {
		if err := lr.mctx.Erase(as); err != nil {
			return fmt.Errorf("final-lower: erasing the map's trailing As node %d: %w", as.BasePtr().ID(), err)
		}
	}
	return nil
}
