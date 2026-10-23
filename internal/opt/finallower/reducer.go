package finallower

// reducer.go is final-lower's REDUCER LOWER (LR-1..LR-3, LR-7, LR-8, LR-13,
// LR-14; doc/optana/final_lower.md "Reducer Lower Integration" as amended —
// the review 261022-g6i FL-1 fix): per live KindReduce, in allocation order,
//
//	(0)  derive @r — Reduce.op0 → (recursing through Reduce chains) →
//	     producing Map → ir.AsOf → the LAST dim operand of the As order; the
//	     dim MUST carry a provenance entry (fail-closed). NO Map/As chain is
//	     a HARD diagnostic, never a guess. Runs BEFORE Map Lower (LR-2),
//	     which rewires Reduce.op0 away.
//	(1)  resolve X (provenance.Between — the OUTERMOST loop over the reduced
//	     dim; the single loop at T==1), home (provenance.Inside, else X),
//	     and the between-set: the Loop blocks strictly nested between X's
//	     block and home's block on the parent chain (empty at T==1).
//	(2.1) homogeneity (LR-8): the piped value's ELEMENT type must equal the
//	     init attribute's type; allocate the specialized construction
//	     (LR-7: Soc 0→Sequential, 2→Warp, 1→Global) over (%init, %lambda),
//	     appended after the existing content of the OWNING LNT's prologue
//	     Group (the owner = the LNT of home's block), stamped SN 0.
//	(2.2) RuntimeCallMethod(%reducer, "Init") — argument-free — inserted
//	     immediately BEFORE X's item in X's PARENTAL block (the prologue
//	     Group when X is its cluster's outermost loop).
//	(2.3) RuntimeCallMethod(%reducer, "Reduce", betweenIVs..., %piped)
//	     inserted inside the HOME loop, immediately after the piped value's
//	     producing item. NOTE on the anchor: the piped Map is a pure
//	     projection the scheduler lands in the nest EPILOGUE (SN == the
//	     reduced loop's depth — astcfg.go's documented duality) while the
//	     VALUE it projects (the Map's lambda body) is the home loop's
//	     in-body producer; the call therefore anchors at the body-chased
//	     node's item, which must BE an item of the home block (fail-closed)
//	     — the only reading consistent with LR-1's law ("the Reduce call
//	     stays INSIDE the home loop") on the scheduler's actual landing.
//	     The combiner application itself is RUNTIME-INTERNAL (LR-3): the
//	     compiler emits NO lambda-call node and never reads
//	     current_value/init_value — prev is the runtime's member field, the
//	     lambda's formals are the carriers.
//	(2.4) RuntimeCallMethod(%reducer, "Combine") — RAUW EVERY former use of
//	     the Reduce onto it, erase the Reduce + its KeyedAttribute
//	     explicitly (the lambda stays live through the construction's
//	     operand), insert Combine immediately AFTER X's item, then build
//	     the fused-consumer RE-ITERATION (LR-14/P-12): fresh sequential
//	     ForLoops over the between dims, and the qualifying owning-LNT
//	     epilogue items MOVED into the innermost re-iteration block.
//
// One reduction spans ALL tiles of @r: Init/Combine bracket X, NEVER the
// tile-inside loop — no per-tile partials (doc/tca.md §5.10). Accumulator
// tiles (LR-13): when non-reduced loops run between X and home, the Reduce
// call carries the between-loop IVs as slot indices; Init/Combine stay
// argument-free and cover the whole tile.
//
// Do NOT run a whole-graph DCE sweep here (P-7): the Init/Reduce call
// statements have no users, and a roots-based sea sweep would erase them —
// reachability roots do not know runtime-call statements.
import (
	"fmt"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt/tiling"
)

// lowerReducers lowers every live KindReduce (allocation order).
func (lr *lowering) lowerReducers() error {
	for _, red := range lr.g.AllocatedNodes() {
		if red == nil || red.BasePtr() == nil || red.BasePtr().Erased() || red.KindOf() != ir.KindReduce {
			continue
		}
		if err := lr.lowerReduce(red); err != nil {
			return err
		}
	}
	return nil
}

// lowerReduce performs one Reduce's full lowering (steps 0-2.4).
func (lr *lowering) lowerReduce(red ir.Node) error {
	// (0) @r derivation (LR-2), fail-closed.
	dim, producingMap, piped, err := lr.deriveReducedDim(red)
	if err != nil {
		return err
	}
	prov, ok := lr.prov.Dims[dim]
	if !ok {
		return fmt.Errorf("final-lower: reduce node %d: reduced dim node %d carries no tiling provenance entry (the dim was never part of a rewritten LNT)",
			red.BasePtr().ID(), dim.BasePtr().ID())
	}

	// (1) X / home / the between chain.
	x := prov.Between
	if x == nil || x.BasePtr().Erased() {
		return fmt.Errorf("final-lower: reduce node %d: the provenance Between loop of dim %d is dead",
			red.BasePtr().ID(), dim.BasePtr().ID())
	}
	xBlk := x.BasePtr().SchedBlock()
	if xBlk == nil {
		return fmt.Errorf("final-lower: reduce node %d: the bracket loop X (node %d) carries no SchedBlock linkage",
			red.BasePtr().ID(), x.BasePtr().ID())
	}
	home := x
	if prov.Inside != nil && !prov.Inside.BasePtr().Erased() {
		home = prov.Inside
	}
	homeBlk := home.BasePtr().SchedBlock()
	if homeBlk == nil {
		return fmt.Errorf("final-lower: reduce node %d: the home loop (node %d) carries no SchedBlock linkage",
			red.BasePtr().ID(), home.BasePtr().ID())
	}
	betweenBlks := betweenBlocks(xBlk, homeBlk)
	if betweenBlks == nil {
		return fmt.Errorf("final-lower: reduce node %d: the home block is not nested inside X's block (corrupted schedule?)",
			red.BasePtr().ID())
	}
	ownerLNT, err := lr.lntOf(homeBlk, fmt.Sprintf("reduce node %d's home block", red.BasePtr().ID()))
	if err != nil {
		return err
	}

	// (2.1) Homogeneity (LR-8) + the specialized construction (LR-7).
	init := ir.ReduceInitOf(red)
	if init == nil {
		return fmt.Errorf("final-lower: reduce node %d carries no init KeyedAttribute", red.BasePtr().ID())
	}
	pipedTy, initTy := ir.TypeOf(piped), ir.TypeOf(init)
	if pipedTy == nil || initTy == nil {
		return fmt.Errorf("final-lower: reduce node %d: the piped value or the init carries no computed type (typeinfer must run before final-lower)",
			red.BasePtr().ID())
	}
	if !elementViewEqual(pipedTy, initTy) {
		return fmt.Errorf("final-lower: reduce node %d: piped element type %v violates the homogeneous reducer contract against init type %v (doc/tca.md §4.2)",
			red.BasePtr().ID(), pipedTy, initTy)
	}
	lambda := ir.LambdaOf(red)
	if lambda == nil {
		return fmt.Errorf("final-lower: reduce node %d carries no combiner lambda", red.BasePtr().ID())
	}
	construction, err := lr.newReducerConstruction(prov.Soc, init, lambda)
	if err != nil {
		return fmt.Errorf("final-lower: reduce node %d: %w", red.BasePtr().ID(), err)
	}
	prologue, err := lr.prologueOf(ownerLNT)
	if err != nil {
		return err
	}
	// Quick-261023-lex: the LNT cluster is ONE lexical Group embedding the
	// nest, so "appended to the prologue" is the END OF THE HEAD REGION —
	// immediately BEFORE the embedded root-loop Block item — never the end
	// of the Group (that would place the construction after the nest, out
	// of scope for every Init/Reduce/Combine call).
	headEnd := len(prologue.Items())
	for i, it := range prologue.Items() {
		if it.Block != nil {
			headEnd = i
			break
		}
	}
	if err := prologue.InsertItemAt(headEnd, ir.ASTItem{Node: construction}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: inserting the reducer construction: %w", red.BasePtr().ID(), err)
	}
	construction.BasePtr().SetSched(0, prologue)

	// (2.2) Init: immediately BEFORE X's item, in X's parental block — X
	// appears as a BLOCK item (its own Loop block), so the parent is the
	// block holding X's SchedBlock as an item (the prologue Group when X is
	// its cluster's outermost loop).
	parentBlk, xIdx := lr.findBlockItem(xBlk)
	if parentBlk == nil || xIdx < 0 {
		return fmt.Errorf("final-lower: reduce node %d: the bracket loop X (node %d) is not an AST item anywhere",
			red.BasePtr().ID(), x.BasePtr().ID())
	}
	initCall := lr.g.NewRuntimeCallMethod(lowerOrigin, "Init", construction)
	if err := parentBlk.InsertItemAt(xIdx, ir.ASTItem{Node: initCall}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: inserting Init before X: %w", red.BasePtr().ID(), err)
	}
	scope := lr.scopeDepth(parentBlk)
	initCall.BasePtr().SetSched(scope, parentBlk)

	// (2.3) The Reduce call: inside the HOME loop, after the piped value's
	// producing item (see the file comment's anchor note). The projection
	// chase applies only on the DIRECT piped-Map shape; a reduce-after-reduce
	// chain keeps the piped node itself as the anchor (its landing is
	// asserted against the home block either way).
	anchor := piped
	if producingMap != nil && piped != nil && producingMap.BasePtr() == piped.BasePtr() {
		if bodies := ir.LambdaBodies(producingMap); len(bodies) == 1 && bodies[0] != nil {
			anchor = bodies[0]
		}
	}
	anchorBlk, anchorIdx := lr.findItem(anchor)
	if anchorBlk == nil || anchorIdx < 0 {
		return fmt.Errorf("final-lower: reduce node %d: the piped value's producer (node %d) is not an AST item of the home loop",
			red.BasePtr().ID(), anchor.BasePtr().ID())
	}
	// Inside the home loop = the home block itself or any descendant (the
	// T==1 shape nests the producer deeper than the single k loop).
	if anchorBlk != homeBlk && findSubBlock(homeBlk, anchorBlk) == nil {
		return fmt.Errorf("final-lower: reduce node %d: the piped value's producer (node %d) sits outside the home loop (node %d) — the Reduce call must stay inside the home loop (LR-1)",
			red.BasePtr().ID(), anchor.BasePtr().ID(), home.BasePtr().ID())
	}
	args := make([]ir.Node, 0, len(betweenBlks)+1)
	for _, b := range betweenBlks {
		args = append(args, ir.OperandsOf(b.Loop())[0]) // the between-loop IV, outer→inner (LR-13)
	}
	args = append(args, piped)
	reduceCall := lr.g.NewRuntimeCallMethod(lowerOrigin, "Reduce", construction, args...)
	if err := anchorBlk.InsertItemAt(anchorIdx+1, ir.ASTItem{Node: reduceCall}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: inserting the Reduce call: %w", red.BasePtr().ID(), err)
	}
	reduceCall.BasePtr().SetSched(lr.scopeDepth(homeBlk), homeBlk)

	// (2.4) Combine: RAUW every Reduce use onto it, erase the Reduce + its
	// KeyedAttribute, insert immediately AFTER X's item. X moved to xIdx+1
	// when Init landed at xIdx, so Combine takes xIdx+2 and the re-iteration
	// chain xIdx+3.
	combineCall := lr.g.NewRuntimeCallMethod(lowerOrigin, "Combine", construction)
	if err := lr.mctx.ReplaceAllUsesWith(red, combineCall); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: RAUW onto the Combine call: %w", red.BasePtr().ID(), err)
	}
	ops := ir.OperandsOf(red)
	if err := lr.mctx.Erase(red); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: erasing the Reduce: %w", red.BasePtr().ID(), err)
	}
	if len(ops) > 1 && ops[1] != nil && !ops[1].BasePtr().Erased() && len(ir.UsesOf(ops[1])) == 0 {
		if err := lr.mctx.Erase(ops[1]); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: erasing its KeyedAttribute: %w", red.BasePtr().ID(), err)
		}
	}
	if err := parentBlk.InsertItemAt(xIdx+2, ir.ASTItem{Node: combineCall}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: inserting Combine after X: %w", red.BasePtr().ID(), err)
	}
	combineCall.BasePtr().SetSched(scope, parentBlk)

	// The fused-consumer re-iteration (LR-14/P-12).
	return lr.buildReiteration(red, dim, xIdx+3, parentBlk, betweenBlks, ownerLNT, combineCall)
}

// deriveReducedDim implements LR-2 step 0: Reduce Input #0 → (recursing
// through reduce-after-reduce chains, tca.md L93) → the producing Map → its
// As → the LAST dim operand. Returns the dim node, the producing Map, and
// the piped value node. A chain ending in no Map/As is a hard error (never
// a guess).
func (lr *lowering) deriveReducedDim(red ir.Node) (dim, producingMap, piped ir.Node, err error) {
	// The call's %piped is ALWAYS this Reduce's own Input #0; the chase
	// variable only walks the producing chain for @r.
	piped = redOp0(red)
	cur := red
	for {
		p := redOp0(cur)
		if p == nil {
			break
		}
		if p.KindOf() == ir.KindReduce {
			cur = p // reduce-after-reduce: recurse through the chain
			continue
		}
		if p.KindOf() == ir.KindMap {
			as := ir.AsOf(p)
			ops := ir.OperandsOf(as)
			if as != nil && len(ops) > 0 && ops[len(ops)-1] != nil {
				return ops[len(ops)-1], p, piped, nil
			}
		}
		break
	}
	return nil, nil, piped, fmt.Errorf("final-lower: reduce node %d: no producing Map/As chain derivable for the reduced dim (the derivation is a hard diagnostic, never a guess)", red.BasePtr().ID())
}

// betweenBlocks returns the Loop blocks strictly nested between xBlk and
// homeBlk on the parent chain (outer→inner), or nil when homeBlk is not a
// descendant of xBlk. homeBlk itself is the terminus — never a between
// block (it is the reduced dim's own home, not a slot-index space, and
// re-iterating over it would re-walk the reduced dimension).
func betweenBlocks(xBlk, homeBlk *ir.ASTBlock) []*ir.ASTBlock {
	var path []*ir.ASTBlock
	var search func(blk *ir.ASTBlock) bool
	search = func(blk *ir.ASTBlock) bool {
		if blk == homeBlk {
			return true
		}
		for _, it := range blk.Items() {
			if it.Block == nil {
				continue
			}
			if it.Block == homeBlk {
				return true // terminus reached — home is not part of the path
			}
			path = append(path, it.Block)
			if search(it.Block) {
				return true
			}
			path = path[:len(path)-1]
		}
		return false
	}
	if !search(xBlk) {
		return nil
	}
	return append([]*ir.ASTBlock(nil), path...)
}

// newReducerConstruction allocates the Soc-specialized construction (LR-7).
func (lr *lowering) newReducerConstruction(soc tiling.SoCType, init, lambda ir.Node) (ir.Node, error) {
	switch soc {
	case tiling.SoCSequential:
		return lr.g.NewRuntimeSequentialReducer(lowerOrigin, init, lambda), nil
	case tiling.SoCWARP:
		return lr.g.NewRuntimeWarpReducer(lowerOrigin, init, lambda), nil
	case tiling.SoCTA:
		// LR-7's rule (cta → Global); v0-unreachable by the scheduling
		// policy (no CTA-parallel reduced dims) — registered scaffolding.
		return lr.g.NewRuntimeGlobalReducer(lowerOrigin, init, lambda), nil
	default:
		return nil, fmt.Errorf("unknown SoC %d for the reducer specialization", int(soc))
	}
}

// buildReiteration implements LR-14/P-12: fresh sequential ForLoops over the
// between dims (same dim InputParam, same bound expression — its schedule
// position must enclose the insertion point, fail-closed), nested as AST
// Loop blocks right after Combine, with the qualifying owning-LNT epilogue
// items (in order, the KindReturn end node excluded) moved into the
// innermost block. Within the moved subtrees each between loop's ORIGINAL
// dimvar is rewired onto the re-iteration loop's dimvar (scoped RAUW); a
// moved consumer referencing a REDUCED-dim dimvar is a hard diagnostic. An
// empty between-set (T==1) leaves the consumers after Combine as-is; no
// qualifying items means no re-iteration blocks (never empty loops).
func (lr *lowering) buildReiteration(red, dim ir.Node, insertAt int, parentBlk *ir.ASTBlock, betweenBlks []*ir.ASTBlock, ownerLNT *ir.ASTLNT, combine ir.Node) error {
	if len(betweenBlks) == 0 {
		return nil // T==1: consumers simply stay after Combine
	}

	// The qualifying epilogue items: the owning LNT's epilogue Group's
	// items (in order, the end node excluded) that CONSUME the Combine
	// value (doc/tca.md §5.10 point 4: "operations that consume %valred
	// are placed after Combine, inside a re-iteration of the non-reduced
	// inner dimensions"). The between-IV reference alone must NOT
	// qualify: the piped Map's projection lambda (anchored in the epilogue
	// by the scheduler) references the between loops' IVs through its
	// lowered coordinates, yet it is a PRODUCER for the nest — the Reduce
	// call inside the home loop pipes it — and Map Lower (running after
	// this pass, LR-2) disposes it where it sits.
	epilogue, err := lr.epilogueOf(ownerLNT)
	if err != nil {
		return err
	}
	betweenIVs := make([]ir.Node, len(betweenBlks))
	for i, b := range betweenBlks {
		betweenIVs[i] = ir.OperandsOf(b.Loop())[0]
	}
	qualifies := func(n ir.Node) bool {
		if n == nil || n.KindOf() == ir.KindReturn {
			return false
		}
		return operandClosure(n)[combine]
	}
	var moved []ir.Node
	for _, it := range epilogue.Items() {
		if it.Node != nil && qualifies(it.Node) {
			moved = append(moved, it.Node)
		}
	}
	if len(moved) == 0 {
		return nil
	}
	// Reduced-dim dimvars are poison in moved consumers: their values are
	// not defined at the re-iteration position (the reduced loops are DONE).
	reducedVars := lr.reducedDimVars(dim)
	for _, m := range moved {
		if containsAny(operandClosure(m), reducedVars) {
			return fmt.Errorf("final-lower: moved consumer node %d (%s) references a reduced-dim dimvar — its value is not defined after the reduction bracket (doc/tca.md §5.10 point 4)",
				m.BasePtr().ID(), m.KindOf())
		}
	}

	// Build the fresh re-iteration chain innermost-first: each loop's body
	// operand is wired at construction (the first moved item anchors the
	// innermost body — the sea-level reachability anchor; the AST blocks
	// carry the real statement order).
	freshIVs := make([]ir.Node, len(betweenBlks))
	innermost := moved[0]
	var innerBlk *ir.ASTBlock
	for i := len(betweenBlks) - 1; i >= 0; i-- {
		loop := betweenBlks[i].Loop()
		iv := ir.OperandsOf(loop)[0]
		dimParam := ir.OperandsOf(iv)[0]
		maxExpr := ir.OperandsOf(ir.OperandsOf(loop)[1])[1]
		if err := lr.assertBoundEncloses(maxExpr, parentBlk); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: %w", red.BasePtr().ID(), err)
		}
		freshIV := lr.g.NewDimVar(lowerOrigin, dimParam, lr.u32c(0), lr.u32c(1))
		freshWrapper := lr.g.NewBoundedDimVar(lowerOrigin, freshIV, maxExpr)
		freshLoop := lr.g.NewForLoop(lowerOrigin, freshIV, freshWrapper, innermost,
			lr.g.NewKeyedAttribute(lowerOrigin, "soc", lr.u32c(0)),
			lr.g.NewKeyedAttribute(lowerOrigin, "loc", lr.u32c(1)))
		blk, err := ir.NewASTLoop(freshLoop)
		if err != nil {
			return fmt.Errorf("final-lower: reduce node %d: building a re-iteration loop block: %w", red.BasePtr().ID(), err)
		}
		depth := lr.scopeDepth(parentBlk) + 1
		freshLoop.BasePtr().SetSched(depth, blk)
		freshIV.BasePtr().SetSched(depth, blk)
		freshWrapper.BasePtr().SetSched(depth, blk)
		if innerBlk != nil {
			if err := blk.InsertItemAt(0, ir.ASTItem{Block: innerBlk}); err != nil {
				return fmt.Errorf("final-lower: reduce node %d: nesting the re-iteration chain: %w", red.BasePtr().ID(), err)
			}
		}
		innerBlk, innermost = blk, freshLoop
		freshIVs[i] = freshIV
	}

	// Re-iteration coordinates: within the moved consumers, each between
	// loop's ORIGINAL dimvar becomes the fresh re-iteration dimvar. A bare
	// scoped RAUW is UNSOUND here — the consumers' coordinate expressions
	// (the lowered index expressions) are SHARED with the in-nest loads,
	// and rewiring a shared node's dimvar use would corrupt the loads.
	// The moved items instead get REBUILT operand subtrees with the
	// substitution applied (the tile-lower rebuildAxis precedent: clone
	// the expression spine, share the untouched leaves) — the semantic
	// equivalent of P-12's "consumer references to those loops' dimvars
	// are RAUW'd", made sound on shared expressions.
	subs := make(map[ir.Node]ir.Node, len(betweenIVs))
	for i, oldIV := range betweenIVs {
		subs[oldIV] = freshIVs[i]
	}
	for _, m := range moved {
		if err := lr.rebuildOperands(m, subs); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: rebuilding a moved consumer's between-dim coordinates: %w", red.BasePtr().ID(), err)
		}
	}

	// Move the qualifying items out of the epilogue, in order, into the
	// innermost re-iteration block; nest the chain right after Combine.
	idx := 0
	epilogue.RemoveItemsIf(func(it ir.ASTItem) bool {
		if it.Node != nil && idx < len(moved) && it.Node.BasePtr() == moved[idx].BasePtr() {
			idx++
			return true
		}
		return false
	})
	for _, m := range moved {
		innerBlk.AppendItem(ir.ASTItem{Node: m})
	}
	if err := parentBlk.InsertItemAt(insertAt, ir.ASTItem{Block: innerBlk}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: inserting the re-iteration chain: %w", red.BasePtr().ID(), err)
	}
	return nil
}

// closureHitsAny reports whether n's operand closure contains any key of
// subs.
func closureHitsAny(n ir.Node, subs map[ir.Node]ir.Node) bool {
	cl := operandClosure(n)
	for k := range subs {
		if cl[k] {
			return true
		}
	}
	return false
}

// rebuildWithSubs clones n's expression spine replacing every mapped node
// with its substitution, sharing every untouched subtree (the leaves stay
// the original defs — the tile-lower rebuildAxis discipline). Only the
// scalar-expression kinds that can appear in a coordinate position clone;
// anything else returns as-is (an unmapped leaf).
func (lr *lowering) rebuildWithSubs(n ir.Node, subs map[ir.Node]ir.Node) ir.Node {
	if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
		return n
	}
	if sub, ok := subs[n]; ok {
		return sub
	}
	ops := ir.OperandsOf(n)
	newOps := make([]ir.Node, len(ops))
	copy(newOps, ops)
	changed := false
	for i, op := range ops {
		r := lr.rebuildWithSubs(op, subs)
		if r != op {
			newOps[i] = r
			changed = true
		}
	}
	if !changed {
		return n
	}
	// Clone the node with rebuilt operands (construction-exempt — the same
	// freedom the constructors exercise; only NEW nodes result).
	switch n.KindOf() {
	case ir.KindBinOp:
		op, _ := ir.BinOpCodeOf(n)
		return lr.g.NewBinOp(lowerOrigin, op, newOps[0], newOps[1])
	case ir.KindUnaryOp:
		op, _ := ir.UnaryOpCodeOf(n)
		return lr.g.NewUnaryOp(lowerOrigin, op, newOps[0])
	case ir.KindSelect:
		return lr.g.NewSelect(lowerOrigin, newOps[0], newOps[1], newOps[2])
	case ir.KindCast:
		return lr.g.NewCast(lowerOrigin, ir.CastTypeOf(n), newOps[0])
	case ir.KindCall:
		return lr.g.NewCall(lowerOrigin, ir.CallNameOf(n), newOps...)
	default:
		return n // leaves (consts, params, dimvars, runtime reads) clone as themselves
	}
}

// rebuildOperands rewrites user's direct operand slots whose operand-edge
// closure references a substitution key: each distinct OLD operand def gets
// ONE rebuilt spine (rebuildWithSubs), and every use record of that old def
// held by user rewires onto it through the SetDef choke point. Slots over
// the same old def therefore all read the same rebuilt expression — the
// correct sharing for a coordinate referenced twice.
func (lr *lowering) rebuildOperands(user ir.Node, subs map[ir.Node]ir.Node) error {
	rebuilt := make(map[ir.Node]ir.Node)
	for _, old := range ir.OperandsOf(user) {
		if old == nil || !closureHitsAny(old, subs) {
			continue
		}
		if _, done := rebuilt[old]; done {
			continue
		}
		fresh := lr.rebuildWithSubs(old, subs)
		rebuilt[old] = fresh
		if fresh == old {
			continue
		}
		for _, u := range ir.UsesOf(old) { // snapshot: survives its own rewiring
			if usr := u.User(); usr != nil && usr.BasePtr() == user.BasePtr() {
				if err := lr.mctx.SetDef(u, fresh); err != nil {
					return fmt.Errorf("node %d (%s) slot rewiring: %w", user.BasePtr().ID(), user.KindOf(), err)
				}
			}
		}
	}
	return nil
}

// epilogueOf resolves the LNT's epilogue Group (the LAST Group item whose
// Nest() == lnt — the cluster layout is [prologue, LBBN, epilogue]).
func (lr *lowering) epilogueOf(lnt *ir.ASTLNT) (*ir.ASTBlock, error) {
	var found *ir.ASTBlock
	for _, it := range lr.ast.Body.Items() {
		if it.Block != nil && it.Block.Kind() == ir.BlockGroup && it.Block.Nest() == lnt {
			found = it.Block
		}
	}
	if found == nil {
		return nil, fmt.Errorf("LNT root node %d has no epilogue Group in the kernel Body", lnt.Root.BasePtr().ID())
	}
	return found, nil
}

// reducedDimVars collects the reduced dim's loop dimvars (the Between and
// Inside loops' IVs from the provenance) — the poison set for moved
// consumers.
func (lr *lowering) reducedDimVars(dim ir.Node) map[ir.Node]bool {
	out := make(map[ir.Node]bool)
	prov, ok := lr.prov.Dims[dim]
	if !ok {
		return out
	}
	for _, loop := range []ir.Node{prov.Between, prov.Inside} {
		if loop == nil || loop.BasePtr().Erased() {
			continue
		}
		if iv := ir.OperandsOf(loop)[0]; iv != nil {
			out[iv] = true
		}
	}
	return out
}

// assertBoundEncloses fail-closes when the bound expression DEPENDS ON a
// DimVar (the IR's scoping carrier) whose loop block does not enclose (is
// not an ancestor-or-self of) the insertion block: such an IV is genuinely
// not live at the re-iteration position (P-12). Computed sub-expressions
// (binops, consts — even the bound node itself) are pure SSA values the
// scheduler stamps at their innermost USE site (here: inside X); SSA value
// semantics keep them available after X closes, and the plan pins reusing
// the m tile-inside loop's very same bound node — so only the IV leaves
// are scope-checked.
func (lr *lowering) assertBoundEncloses(bound ir.Node, insertion *ir.ASTBlock) error {
	for n := range operandClosure(bound) {
		if n.KindOf() != ir.KindDimVar {
			continue
		}
		bBlk := n.BasePtr().SchedBlock()
		if bBlk == nil || bBlk == insertion || findSubBlock(bBlk, insertion) != nil {
			continue // unscheduled, or the insertion sits inside the IV's loop block
		}
		return fmt.Errorf("the re-iteration bound depends on the IV node %d, whose loop does not enclose the insertion position — the bound is not defined at the re-iteration position", n.BasePtr().ID())
	}
	return nil
}

// findSubBlock reports whether needle is blk or nested inside it.
func findSubBlock(blk, needle *ir.ASTBlock) *ir.ASTBlock {
	if blk == needle {
		return blk
	}
	for _, it := range blk.Items() {
		if it.Block != nil {
			if found := findSubBlock(it.Block, needle); found != nil {
				return found
			}
		}
	}
	return nil
}

// operandClosure collects n's transitive operand-edge closure (self
// included, live nodes only).
func operandClosure(n ir.Node) map[ir.Node]bool {
	set := make(map[ir.Node]bool)
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || x.BasePtr().Erased() || set[x] {
			return
		}
		set[x] = true
		for _, d := range ir.OperandsOf(x) {
			visit(d)
		}
	}
	visit(n)
	return set
}

// containsAny reports whether any key of needles is in set.
func containsAny(set map[ir.Node]bool, needles map[ir.Node]bool) bool {
	for n := range needles {
		if set[n] {
			return true
		}
	}
	return false
}

// redOp0 returns the Reduce's Input #0 (the piped value), nil-safe.
func redOp0(red ir.Node) ir.Node {
	ops := ir.OperandsOf(red)
	if len(ops) == 0 || ops[0] == nil {
		return nil
	}
	return ops[0]
}

// u32c allocates a u32 const node.
func (lr *lowering) u32c(v float64) ir.Node {
	return lr.g.NewConst(lowerOrigin, ir.InternScalar(ir.TU32), v)
}

// elementViewEqual compares t against want at element view (a tensor
// contributes its element type) — the typeinfer scalarView rule's mirror.
func elementViewEqual(t, want ir.Type) bool {
	nt, ok := t.(*ir.NumericType)
	if !ok {
		return false
	}
	if nt.IsScalar() {
		return t.Equal(want)
	}
	return ir.InternScalar(nt.Elem()).Equal(want)
}
