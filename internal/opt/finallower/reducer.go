package finallower

// reducer.go is final-lower's REDUCER LOWER (LR-1..LR-3, LR-7, LR-8 as
// amended by the R1/R2/R3 ruling of 2026-10-23): per live KindReduce, in
// allocation order,
//
//	(0)  derive @r — Reduce.op0 → (recursing through Reduce chains) →
//	     producing Map → ir.AsOf → the LAST dim operand of the As order; the
//	     dim MUST carry a provenance entry (fail-closed). NO Map/As chain is
//	     a HARD diagnostic, never a guess. Runs BEFORE Map Lower (LR-2),
//	     which rewires Reduce.op0 away.
//	(1)  resolve X (provenance.Between — the OUTERMOST loop over the reduced
//	     dim), home (provenance.Inside, else X), and the between chain: the
//	     Loop blocks strictly nested between X's block and home's block on
//	     the parent chain (empty at T==1).
//	(2.1) homogeneity (LR-8) + the specialized construction (LR-7: Soc
//	     0→Sequential over (init, lambda); 2→Warp / 1→Global over (init,
//	     lambda, ThreadMapping, staging) per R1/R2): the WARP/GLOBAL
//	     constructions additionally allocate a staging KindMemNewSMem — one
//	     #warps × sizeof(T) row in the compiler's dynamic-SMEM pool at a
//	     disjoint D8 offset — inserted with the construction at the head
//	     end of the OWNING LNT's prologue Group (before the embedded root
//	     loop), stamped SN 0. The ThreadMapping operand resolves through
//	     this pass's own per-LNT objects (Run() inserts them FIRST).
//	(2.2) R3 — per-output-element brackets. The tiled shape (T>1):
//	     the between loops DISSOLVE as per-thread walks; a FRESH UNIFORM
//	     sequential element loop chain (one loop per between dim, init 0,
//	     step 1, the between loop's own clamped bound) replaces X at X's
//	     position in X's parental block, and X (with home inside) moves
//	     INTO the innermost element loop. Per element: Init (first item) →
//	     the reduced-dim walks with Reduce(value) (slot arguments are GONE
//	     — R1's scalar protocol) → Combine right after X. Every reference
//	     to a dissolved between IV — inside the moved walk and inside the
//	     moved epilogue consumers — is REBUILT onto the fresh element IV
//	     (cloned spines, shared leaves; the tile-lower rebuildAxis
//	     discipline). The uniform element loop keeps Combine CTA-uniform
//	     (its __syncthreads is legally reached by every thread), and the
//	     per-element epilogue stores become single-owner stores (the
//	     emitter gates them on is_first_in_cta(), R2/R3).
//	     The T==1 shape keeps the flat bracket (R-A's T==1 law): Init
//	     immediately before X, Combine immediately after, both in X's
//	     parental block — one output element per bracket, no element loops
//	     — and the qualifying epilogue consumers move in after Combine so
//	     the emitted Combine local and its consumers share one C++ scope
//	     (HI-02).
//	(2.3) RuntimeCallMethod(%reducer, "Reduce", %piped) — VALUE ONLY —
//	     inserted inside the HOME loop, immediately after the piped value's
//	     producing item (the scheduler lands the piped Map's projection in
//	     the nest EPILOGUE while the value it projects is the home loop's
//	     in-body producer; the call anchors at the body-chased node's
//	     item, which must BE an item of the home block — fail-closed).
//	     The combiner application is RUNTIME-INTERNAL (LR-3): the compiler
//	     emits no lambda-call node and never reads current_value/init.
//	(2.4) RuntimeCallMethod(%reducer, "Combine") — RAUW EVERY former use of
//	     the Reduce onto it, erase the Reduce + its KeyedAttribute, then
//	     move the qualifying owning-LNT epilogue consumers into the
//	     innermost element loop after Combine (rebuilding their between-dim
//	     coordinates onto the fresh element IVs). Combine returns T (R1) —
//	     the consumers read the scalar directly; the D7 slot reads are
//	     gone.
//
// One reduction spans ALL tiles of @r per bracket: Init → the reduced-dim
// walks → Combine (doc/tca.md §5.10). The bracket's ITERATION SPACE is the
// output element (the non-reduced between dims) — R3.
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
	// A fresh rebuild session per reduce: the between-IV substitution clones
	// are shared across every rebuilt user of this reduce (the walk AND the
	// moved consumers) — see the rebuildMemo field note.
	lr.rebuildMemo = make(map[ir.Node]ir.Node)
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

	// (2.1) Homogeneity (LR-8) + the specialized construction (LR-7 + R1/R2).
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
	construction, err := lr.newReducerConstruction(prov.Soc, init, lambda, initTy, ownerLNT)
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
	// The staging allocation item (if any) precedes the construction: the
	// construction's operand references it, and the reader sees the pool row
	// before its consumer.
	if st := lr.stagingOf(construction); st != nil {
		if err := prologue.InsertItemAt(headEnd, ir.ASTItem{Node: st}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: inserting the reducer staging row: %w", red.BasePtr().ID(), err)
		}
		st.BasePtr().SetSched(0, prologue)
		headEnd++
	}
	if err := prologue.InsertItemAt(headEnd, ir.ASTItem{Node: construction}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: inserting the reducer construction: %w", red.BasePtr().ID(), err)
	}
	construction.BasePtr().SetSched(0, prologue)

	// (2.3) The Reduce call's anchor: the piped value's producing item (see
	// the file comment). The projection chase applies only on the DIRECT
	// piped-Map shape; a reduce-after-reduce chain keeps the piped node
	// itself as the anchor (its landing is asserted against the home block
	// either way).
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
	// R1: Reduce(value) — the slot dimvars are gone; the single argument is
	// the piped value (Map Lower later resolves it at the load).
	reduceCall := lr.g.NewRuntimeCallMethod(lowerOrigin, "Reduce", construction, piped)
	if err := anchorBlk.InsertItemAt(anchorIdx+1, ir.ASTItem{Node: reduceCall}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: inserting the Reduce call: %w", red.BasePtr().ID(), err)
	}
	reduceCall.BasePtr().SetSched(lr.scopeDepth(homeBlk), homeBlk)

	// (2.4) Combine: RAUW every Reduce use onto it, erase the Reduce + its
	// KeyedAttribute, then place the call per R3.
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

	if len(betweenBlks) == 0 {
		// T==1: R-A's flat bracket — Init immediately before X's block,
		// Combine immediately after it, both in X's parental block (one
		// output element per bracket; X is the reduced dim's single loop).
		parentBlk, xIdx := lr.findBlockItem(xBlk)
		if parentBlk == nil || xIdx < 0 {
			return fmt.Errorf("final-lower: reduce node %d: the bracket loop X (node %d) is not an AST item anywhere",
				red.BasePtr().ID(), x.BasePtr().ID())
		}
		initCall := lr.g.NewRuntimeCallMethod(lowerOrigin, "Init", construction)
		if err := parentBlk.InsertItemAt(xIdx, ir.ASTItem{Node: initCall}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: inserting Init before X: %w", red.BasePtr().ID(), err)
		}
		initCall.BasePtr().SetSched(lr.scopeDepth(parentBlk), parentBlk)
		if err := parentBlk.InsertItemAt(xIdx+2, ir.ASTItem{Node: combineCall}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: inserting Combine after X: %w", red.BasePtr().ID(), err)
		}
		combineCall.BasePtr().SetSched(lr.scopeDepth(parentBlk), parentBlk)
		// HI-02 (R-A's consumer half at T==1): the epilogue consumers of the
		// Combine value move in with it — the emitted `auto vN =
		// reducer.Combine();` local and the consuming store must share ONE
		// lexical C++ scope (the flat bracket's parent block), else the
		// store reads a local declared inside a loop that already closed.
		return lr.moveCombineConsumers(red, dim, ownerLNT, combineCall, parentBlk, lr.scopeDepth(parentBlk), nil)
	}

	// (2.2) R3 — the per-output-element restructure.
	return lr.buildElementLoop(red, dim, x, xBlk, homeBlk, betweenBlks, ownerLNT, construction, combineCall, anchor)
}

// buildElementLoop implements R3 under the R-A bracket law (2026-10-24
// ruling): the between loops dissolve as per-thread walks; a fresh UNIFORM
// sequential element loop chain (one per between dim, same dim + clamped
// bound) takes X's position in X's parental block; X (with home inside, the
// between wrappers stripped) moves into the INNERMOST element loop — X's
// parent under R-A — carrying the bracket [Init immediately before X →
// X{home{...Reduce}} → Combine immediately after → the moved consumers]:
// one bracket per INNERMOST output-element tuple, never per X-round. Every
// between-IV reference in the moved walk and the moved epilogue consumers
// is rebuilt onto the fresh element IVs (cloned spines, shared leaves). A
// consumer referencing a reduced-dim dimvar is a hard diagnostic (its value
// is not defined per element).
func (lr *lowering) buildElementLoop(red, dim, x ir.Node, xBlk, homeBlk *ir.ASTBlock,
	betweenBlks []*ir.ASTBlock, ownerLNT *ir.ASTLNT,
	construction, combineCall, anchor ir.Node) error {

	parentBlk, xIdx := lr.findBlockItem(xBlk)
	if parentBlk == nil || xIdx < 0 {
		return fmt.Errorf("final-lower: reduce node %d: the bracket loop X (node %d) is not an AST item anywhere",
			red.BasePtr().ID(), x.BasePtr().ID())
	}

	// The between IVs (outer→inner) and their bounds — read BEFORE the
	// blocks dissolve; the fresh element loops reuse the very same clamped
	// bound expressions (scheduler-stamped pure SSA values stay available
	// at the element-loop position, which X's parental block encloses).
	betweenIVs := make([]ir.Node, len(betweenBlks))
	boundMaxs := make([]ir.Node, len(betweenBlks))
	for i, b := range betweenBlks {
		bops := ir.OperandsOf(b.Loop())
		if len(bops) < 2 || bops[0] == nil {
			return fmt.Errorf("final-lower: reduce node %d: between loop %d carries no (iv, bound) header",
				red.BasePtr().ID(), b.Loop().BasePtr().ID())
		}
		betweenIVs[i] = bops[0]
		bb := ir.OperandsOf(bops[1])
		if len(bb) != 2 || bb[1] == nil {
			return fmt.Errorf("final-lower: reduce node %d: between loop %d bound is not a (iv, max) bdimvar",
				red.BasePtr().ID(), b.Loop().BasePtr().ID())
		}
		boundMaxs[i] = bb[1]
		if err := lr.assertBoundEncloses(bb[1], parentBlk); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: the element loop over dim of loop %d: %w",
				red.BasePtr().ID(), b.Loop().BasePtr().ID(), err)
		}
	}

	// Build the fresh uniform element chain innermost-first (each loop's
	// body operand is wired at construction — the piped producer anchors the
	// innermost body, the sea-level reachability anchor; the AST blocks
	// carry the real statement order). Uniform: init 0, step 1, soc 0
	// (sequential), loc 1 — every thread walks every element, which is what
	// keeps Combine's collectives uniformly reached.
	//
	// R-A (HI-01, 2026-10-24 ruling): X's parent must be the INNERMOST
	// output-element loop — the construction runs innermost-first, so the
	// FIRST block built (i == len-1) is the innermost (elemBlk) while the
	// running chain head ends as the OUTERMOST (outermostBlk, the block
	// inserted into X's old parental position at the end). At
	// len(betweenBlks)==1 elemBlk == outermostBlk — the committed
	// 1-between-dim golden shape is bit-identical.
	freshIVs := make([]ir.Node, len(betweenBlks))
	innermost := anchor
	var outermostBlk, elemBlk *ir.ASTBlock
	for i := len(betweenBlks) - 1; i >= 0; i-- {
		dimParam := ir.OperandsOf(betweenIVs[i])[0]
		freshIV := lr.g.NewDimVar(lowerOrigin, dimParam, lr.u32c(0), lr.u32c(1))
		freshWrapper := lr.g.NewBoundedDimVar(lowerOrigin, freshIV, boundMaxs[i])
		freshLoop := lr.g.NewForLoop(lowerOrigin, freshIV, freshWrapper, innermost,
			lr.g.NewKeyedAttribute(lowerOrigin, "soc", lr.u32c(0)),
			lr.g.NewKeyedAttribute(lowerOrigin, "loc", lr.u32c(1)))
		blk, err := ir.NewASTLoop(freshLoop)
		if err != nil {
			return fmt.Errorf("final-lower: reduce node %d: building an element loop block: %w", red.BasePtr().ID(), err)
		}
		depth := lr.scopeDepth(parentBlk) + 1 + i
		freshLoop.BasePtr().SetSched(depth, blk)
		freshIV.BasePtr().SetSched(depth, blk)
		freshWrapper.BasePtr().SetSched(depth, blk)
		if outermostBlk != nil {
			if err := blk.InsertItemAt(0, ir.ASTItem{Block: outermostBlk}); err != nil {
				return fmt.Errorf("final-lower: reduce node %d: nesting the element chain: %w", red.BasePtr().ID(), err)
			}
		}
		if i == len(betweenBlks)-1 {
			elemBlk = blk // the FIRST built is the INNERMOST element loop
		}
		outermostBlk, innermost = blk, freshLoop
		freshIVs[i] = freshIV
	}
	// elemDepth is the INNERMOST element loop's depth (one level per between
	// dim below X's original parental scope) — the depth the bracket items
	// are stamped at.
	elemDepth := lr.scopeDepth(parentBlk) + len(betweenBlks)

	// Init: the FIRST item of the innermost element loop — the per-element
	// bracket opens here, immediately before X's block (R-A: X is appended
	// right after, nothing sits between).
	initCall := lr.g.NewRuntimeCallMethod(lowerOrigin, "Init", construction)
	if err := elemBlk.InsertItemAt(0, ir.ASTItem{Node: initCall}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: inserting Init into the element loop: %w", red.BasePtr().ID(), err)
	}
	initCall.BasePtr().SetSched(elemDepth, elemBlk)

	// Dissolve the between wrappers: home's block unhooks from the chain
	// and re-attaches directly inside X's block at the outermost between
	// block's old position; the between blocks (their loop wrappers and
	// their pure items) are discarded. A between block carrying a SECOND
	// Block item is a corrupted schedule — fail closed.
	outerBetween := betweenBlks[0]
	xItems := xBlk.Items()
	detachAt := -1
	for i, it := range xItems {
		if it.Block == outerBetween {
			detachAt = i
			break
		}
	}
	if detachAt < 0 {
		return fmt.Errorf("final-lower: reduce node %d: the outermost between block is not an item of X's block (corrupted schedule?)", red.BasePtr().ID())
	}
	for _, b := range betweenBlks {
		nBlocks := 0
		for _, it := range b.Items() {
			if it.Block != nil {
				nBlocks++
			}
		}
		if nBlocks != 1 {
			return fmt.Errorf("final-lower: reduce node %d: between block (loop %d) carries %d child blocks, want exactly the chain child",
				red.BasePtr().ID(), b.Loop().BasePtr().ID(), nBlocks)
		}
	}
	xBlk.RemoveItemsIf(func(it ir.ASTItem) bool { return it.Block == outerBetween })
	if err := xBlk.InsertItemAt(detachAt, ir.ASTItem{Block: homeBlk}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: re-homing the home block inside X: %w", red.BasePtr().ID(), err)
	}

	// Move X's block (now [clamps..., home]) out of its parental block and
	// into the innermost element loop, right after Init (R-A: elemBlk is X's
	// parent — the innermost output-element loop).
	parentBlk.RemoveItemsIf(func(it ir.ASTItem) bool { return it.Block == xBlk })
	elemBlk.AppendItem(ir.ASTItem{Block: xBlk})

	// Combine: right after X's block, still inside the innermost element
	// loop — the per-element bracket closes; every thread leaves with the
	// element's fully combined value (R1: the call returns T).
	elemBlk.AppendItem(ir.ASTItem{Node: combineCall})
	combineCall.BasePtr().SetSched(elemDepth, elemBlk)

	// The between-IV -> fresh-IV substitution the consumer rebuild and the
	// walk rebuild below share.
	subs := make(map[ir.Node]ir.Node, len(betweenIVs))
	for i, oldIV := range betweenIVs {
		subs[oldIV] = freshIVs[i]
	}

	// Rebuild the moved WALK's between-IV references (every item of X's
	// block and home's subtree — the loads' non-reduced coordinates) onto
	// the fresh element IVs.
	var rebuildWalk func(blk *ir.ASTBlock) error
	rebuildWalk = func(blk *ir.ASTBlock) error {
		for _, it := range blk.Items() {
			if it.Node != nil {
				if err := lr.rebuildOperands(it.Node, subs); err != nil {
					return fmt.Errorf("final-lower: reduce node %d: rebuilding the walk's between-dim coordinates: %w", red.BasePtr().ID(), err)
				}
				continue
			}
			if it.Block != nil {
				if err := rebuildWalk(it.Block); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if err := rebuildWalk(xBlk); err != nil {
		return err
	}

	// The orphan guard + disposal. First: X's loop BODY operand pointed at
	// the outermost between loop (the scheduler's chain wiring) — rewire it
	// onto home's loop, X's new direct child, so the dissolved loop's last
	// legitimate use dies. The erasure itself is DEFERRED to
	// finalizeDissolutions (after Map Lower): the piped Map's projection
	// machinery — the producing Map, its As, and its epilogue-anchored
	// lambda — still references the walk's coordinate expressions until
	// Map Lower disposes them, so the between dimvar's use list only
	// reaches zero there.
	homeLoop := homeBlk.Loop()
	rewired := false
	for _, u := range ir.UsesOf(betweenBlks[0].Loop()) {
		if usr := u.User(); usr != nil && usr.BasePtr() == x.BasePtr() {
			if err := lr.mctx.SetDef(u, homeLoop); err != nil {
				return fmt.Errorf("final-lower: reduce node %d: rewiring X's body operand onto home: %w", red.BasePtr().ID(), err)
			}
			rewired = true
			break
		}
	}
	if !rewired && len(ir.UsesOf(betweenBlks[0].Loop())) > 0 {
		u := ir.UsesOf(betweenBlks[0].Loop())[0]
		return fmt.Errorf("final-lower: reduce node %d: node %d (%s) still references the dissolved between loop %d after the element-loop restructure",
			red.BasePtr().ID(), u.User().BasePtr().ID(), u.User().KindOf(), betweenBlks[0].Loop().BasePtr().ID())
	}
	for i := range betweenBlks {
		loop := betweenBlks[i].Loop()
		// Capture the loop's operand closure BEFORE any erasure — the
		// finalize sweep's candidate set (wrapper, annotations, and their
		// coordinate debris).
		closure := operandClosure(loop)
		lr.dissolved = append(lr.dissolved, dissolveJob{loop: loop, iv: betweenIVs[i], red: red, closure: closure})
	}

	// Insert the element chain at X's old position (the OUTERMOST element
	// loop takes X's slot) — the restructure is complete: [element loops
	// outer→inner [ ... [ Init, X [ home [ ..., Reduce ] ], Combine, moved
	// consumers... ] ...]] with the bracket at the INNERMOST level (R-A).
	// This runs BEFORE the consumer move so the enclosure assertion sees
	// elemBlk's final ancestor chain (the detached subtree would defeat the
	// ancestor-or-self test).
	if err := parentBlk.InsertItemAt(xIdx, ir.ASTItem{Block: outermostBlk}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: inserting the element loop chain: %w", red.BasePtr().ID(), err)
	}

	// The qualifying epilogue consumers, relocated after Combine inside the
	// innermost element loop (shared gates with the T==1 flat bracket).
	if err := lr.moveCombineConsumers(red, dim, ownerLNT, combineCall, elemBlk, elemDepth, subs); err != nil {
		return err
	}
	return nil
}

// moveCombineConsumers implements R3's consumer half (and HI-02's T==1 flat
// twin, 2026-10-24): the owning LNT's epilogue items whose operand closure
// reaches combineCall relocate into dst, inserted IMMEDIATELY AFTER the
// Combine call's item — the Combine local and its consumers must share one
// lexical C++ scope (the emitter materializes the call as a local at its
// item position; a consumer outside that scope reads a dead local). Shared
// gates, in order:
//
//	(a) reduced-dim poison — a consumer referencing the reduced dim's loop
//	    dimvars would read an iteration variable of the WRONG element's
//	    strip (doc/tca.md §5.10 point 4);
//	(b) thread-partitioned refusal — a consumer referencing the single IV
//	    of a WARP-SoC dim (a thread-partitioned walk) would leave a
//	    coverage hole under the is_first_in_cta store gate (single-owner
//	    stores hold only over CTA-replicated element walks) — a defensive
//	    fail-closed for an unruled v0 shape, unreachable when the reduced
//	    dim is WARP (the one-WARP-dim policy leaves no second WARP dim);
//	(c) the between-IV coordinate rebuild onto the fresh element IVs when
//	    subs != nil (the T==1 flat bracket passes nil — its coordinates are
//	    defined at scopes enclosing the bracket already);
//	(d) the generalized enclosure assertion (assertBoundEncloses over
//	    whole closures, P-12): every DimVar in a moved consumer's closure
//	    must be defined at a block enclosing dst — a coordinate not live at
//	    the bracket position is a named error, never an emitted out-of-scope
//	    reference.
func (lr *lowering) moveCombineConsumers(red, dim ir.Node, ownerLNT *ir.ASTLNT,
	combineCall ir.Node, dst *ir.ASTBlock, depth int, subs map[ir.Node]ir.Node) error {
	epilogue, err := lr.epilogueOf(ownerLNT)
	if err != nil {
		return err
	}
	// The qualifying epilogue consumers: the Group's items (in order, the
	// end node excluded) that CONSUME the Combine value. The between-IV
	// reference alone must NOT qualify (the piped Map's projection lambda is
	// a PRODUCER for the nest; Map Lower disposes it where it sits).
	qualifies := func(n ir.Node) bool {
		if n == nil || n.KindOf() == ir.KindReturn {
			return false
		}
		return operandClosure(n)[combineCall]
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
	// (a) Reduced-dim dimvars are poison in moved consumers: their walks
	// re-execute per element inside the bracket — the consumer would read
	// an iteration variable of the WRONG element's strip.
	reducedVars := lr.reducedDimVars(dim)
	for _, m := range moved {
		if containsAny(operandClosure(m), reducedVars) {
			return fmt.Errorf("final-lower: moved consumer node %d (%s) references a reduced-dim dimvar — its value is not defined at the element-loop position (doc/tca.md §5.10 point 4)",
				m.BasePtr().ID(), m.KindOf())
		}
	}
	// (b) The thread-partitioned-consumer refusal.
	if err := lr.refuseThreadPartitionedConsumers(moved); err != nil {
		return err
	}
	// (c) The moved consumers' between-dim coordinates rebuild onto the
	// fresh element IVs (shared expression spines are cloned; the in-nest
	// walk's rebuild below shares this subs map and the discipline of one
	// clone per shared def per USER).
	if subs != nil {
		for _, m := range moved {
			if err := lr.rebuildOperands(m, subs); err != nil {
				return fmt.Errorf("final-lower: reduce node %d: rebuilding a moved consumer's between-dim coordinates: %w", red.BasePtr().ID(), err)
			}
		}
	}
	// (d) The generalized enclosure assertion: every DimVar still in a
	// moved consumer's closure must be defined at dst or an enclosing
	// block.
	for _, m := range moved {
		for n := range operandClosure(m) {
			if n.KindOf() != ir.KindDimVar {
				continue
			}
			bBlk := n.BasePtr().SchedBlock()
			if bBlk == nil || bBlk == dst || findSubBlock(bBlk, dst) != nil {
				continue // unscheduled, or the bracket sits inside the IV's loop
			}
			return fmt.Errorf("final-lower: reduce node %d: moved consumer node %d (%s) references dimvar node %d whose loop does not enclose the bracket block — the coordinate is not defined at the bracket position (P-12)",
				red.BasePtr().ID(), m.BasePtr().ID(), m.KindOf(), n.BasePtr().ID())
		}
	}
	// Relocate: detach from the epilogue, then insert immediately after the
	// Combine call's item in dst.
	idx := 0
	epilogue.RemoveItemsIf(func(it ir.ASTItem) bool {
		if it.Node != nil && idx < len(moved) && it.Node.BasePtr() == moved[idx].BasePtr() {
			idx++
			return true
		}
		return false
	})
	after, ok := itemIndexOf(dst, combineCall)
	if !ok {
		return fmt.Errorf("final-lower: reduce node %d: the Combine call is not an item of the bracket block (corrupted bracket?)",
			red.BasePtr().ID())
	}
	for i, m := range moved {
		if err := dst.InsertItemAt(after+1+i, ir.ASTItem{Node: m}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: inserting a moved epilogue consumer: %w", red.BasePtr().ID(), err)
		}
		m.BasePtr().SetSched(depth, dst)
	}
	return nil
}

// refuseThreadPartitionedConsumers fail-closes when a moved consumer's
// operand closure references the single IV of a WARP-SoC dim: that single
// loop is a THREAD-PARTITIONED walk (each element of the dim is owned by
// exactly one thread of the dim's warp group), so the is_first_in_cta store
// gate would leave the non-owner threads' partitioned elements unwritten —
// a coverage hole, never a silent pass. Defensive for an unruled adjacent
// v0 shape: unreachable when the reduced dim is WARP (the one-WARP-dim
// policy), and the reduced dim's own single IV is caught earlier by the
// reduced-dim poison check.
func (lr *lowering) refuseThreadPartitionedConsumers(moved []ir.Node) error {
	singles := lr.warpSingleIVs()
	for _, m := range moved {
		cl := operandClosure(m)
		var worst ir.Node
		for iv := range singles {
			if cl[iv] && (worst == nil || iv.BasePtr().ID() < worst.BasePtr().ID()) {
				worst = iv // deterministic diagnostic: the lowest-ID offender
			}
		}
		if worst == nil {
			continue
		}
		wdim := singles[worst]
		return fmt.Errorf("final-lower: moved consumer node %d (%s) references the single IV of WARP dim %q (node %d) — a thread-partitioned walk: the is_first_in_cta store gate yields single-owner stores only over CTA-replicated element walks, so a WARP-partitioned non-reduced output dim would leave a coverage hole (defensive fail-closed for an unruled v0 shape)",
			m.BasePtr().ID(), m.KindOf(), ir.InputParamNameOf(wdim), wdim.BasePtr().ID())
	}
	return nil
}

// warpSingleIVs maps every WARP-SoC dim's single-loop IV to its dim param —
// the thread-partitioned walks of the provenance (Tile == 1, no inside
// split; Between IS the single loop).
func (lr *lowering) warpSingleIVs() map[ir.Node]ir.Node {
	out := make(map[ir.Node]ir.Node)
	for dim, prov := range lr.prov.Dims {
		if prov.Soc != tiling.SoCWARP || prov.Tile != 1 || prov.Between == nil {
			continue
		}
		ops := ir.OperandsOf(prov.Between)
		if len(ops) == 0 || ops[0] == nil || ops[0].KindOf() != ir.KindDimVar {
			continue
		}
		out[ops[0]] = dim
	}
	return out
}

// itemIndexOf reports node's item index within blk (node identity).
func itemIndexOf(blk *ir.ASTBlock, node ir.Node) (int, bool) {
	for i, it := range blk.Items() {
		if it.Node != nil && it.Node.BasePtr() == node.BasePtr() {
			return i, true
		}
	}
	return -1, false
}

// stagingOf returns the construction's staging MemNewSMem operand (R1: the
// warp/global specializations carry one; the sequential specialization
// carries none).
func (lr *lowering) stagingOf(construction ir.Node) ir.Node {
	switch construction.KindOf() {
	case ir.KindRuntimeWarpReducer, ir.KindRuntimeGlobalReducer:
		ops := ir.OperandsOf(construction)
		if len(ops) == 4 && ops[3] != nil {
			return ops[3]
		}
	}
	return nil
}

// newReducerConstruction allocates the Soc-specialized construction (LR-7)
// plus, for warp/global, the R1 staging row: one #warps × sizeof(T) bytes in
// the compiler's dynamic-SMEM pool (extent in bytes — the pool accounting
// unit; D8 disjointness comes from the emitter's prescan offset assignment).
func (lr *lowering) newReducerConstruction(soc tiling.SoCType, init, lambda ir.Node, initTy ir.Type, ownerLNT *ir.ASTLNT) (ir.Node, error) {
	if soc == tiling.SoCSequential {
		return lr.g.NewRuntimeSequentialReducer(lowerOrigin, init, lambda), nil
	}
	// R1/R2: warp and global carry (init, lambda, ThreadMapping, staging).
	threadMap, ok := lr.threadOf[ownerLNT]
	if !ok || threadMap == nil {
		return nil, fmt.Errorf("no RuntimeThreadMapping for the owning LNT (insertRuntimeObjects must run before the reducer lower)")
	}
	if lr.prov.MaxThreadsPerCTA <= 0 || lr.prov.MaxThreadsPerCTA%32 != 0 {
		return nil, fmt.Errorf("the provenance launch total %d is not a positive multiple of 32 (the staging row needs the CTA's warp count)", lr.prov.MaxThreadsPerCTA)
	}
	scalar, ok := initTy.(*ir.NumericType)
	if !ok || scalar == nil || scalar.Rank() != 0 {
		return nil, fmt.Errorf("the init type %v is not scalar (the staging row sizes by the element type)", initTy)
	}
	bytes, err := elemBytes(scalar.Elem())
	if err != nil {
		return nil, err
	}
	warps := lr.prov.MaxThreadsPerCTA / 32
	staging := lr.g.NewMemNewSMem(lowerOrigin, lr.u32c(float64(warps*bytes)))
	switch soc {
	case tiling.SoCWARP:
		return lr.g.NewRuntimeWarpReducer(lowerOrigin, init, lambda, threadMap, staging), nil
	case tiling.SoCTA:
		// LR-7's rule (cta → Global); v0-unreachable by the scheduling
		// policy (no CTA-parallel reduced dims) — registered scaffolding.
		return lr.g.NewRuntimeGlobalReducer(lowerOrigin, init, lambda, threadMap, staging), nil
	default:
		return nil, fmt.Errorf("unknown SoC %d for the reducer specialization", int(soc))
	}
}

// elemBytes maps a scalar element type to its C++ size in bytes — the
// staging row's sizing unit (R1).
func elemBytes(e ir.ElemType) (int, error) {
	switch e {
	case ir.TBool, ir.TI8, ir.TU8:
		return 1, nil
	case ir.TI16, ir.TU16, ir.TF16, ir.TBF16:
		return 2, nil
	case ir.TI32, ir.TU32, ir.TF32:
		return 4, nil
	default:
		return 0, fmt.Errorf("no pinned byte size for element type %v (the staging row cannot size)", e)
	}
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
// block (it is the reduced dim's own home; re-walking over it would
// re-walk the reduced dimension).
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
	// Always a NON-nil slice on success: the EMPTY between set (home == X,
	// the T==1 flat-bracket shape) must stay distinguishable from the
	// not-found nil — `append(nil-slice)` with zero elements would return
	// nil and make the T==1 branch forever unreachable (the HI-02 latent
	// bug: every T==1 reduce died here as a "corrupted schedule").
	out := make([]*ir.ASTBlock, 0, len(path))
	return append(out, path...)
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
// anything else returns as-is (an unmapped leaf). Clones are MEMOIZED per
// original def within the current reduce's rebuild session: every rebuilt
// user of a shared spine reads the SAME clone — the loop-scope invariant's
// operand-closure containment then holds through the in-nest users (the
// load), and per-user clones would mint store-private spines outside every
// defining loop's closure.
func (lr *lowering) rebuildWithSubs(n ir.Node, subs map[ir.Node]ir.Node) ir.Node {
	if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
		return n
	}
	if sub, ok := subs[n]; ok {
		return sub
	}
	if lr.rebuildMemo != nil {
		if clone, ok := lr.rebuildMemo[n]; ok {
			return clone
		}
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
	var clone ir.Node
	switch n.KindOf() {
	case ir.KindBinOp:
		op, _ := ir.BinOpCodeOf(n)
		clone = lr.g.NewBinOp(lowerOrigin, op, newOps[0], newOps[1])
	case ir.KindUnaryOp:
		op, _ := ir.UnaryOpCodeOf(n)
		clone = lr.g.NewUnaryOp(lowerOrigin, op, newOps[0])
	case ir.KindSelect:
		clone = lr.g.NewSelect(lowerOrigin, newOps[0], newOps[1], newOps[2])
	case ir.KindCast:
		clone = lr.g.NewCast(lowerOrigin, ir.CastTypeOf(n), newOps[0])
	case ir.KindCall:
		clone = lr.g.NewCall(lowerOrigin, ir.CallNameOf(n), newOps...)
	default:
		return n // leaves (consts, params, dimvars, runtime reads) clone as themselves
	}
	if lr.rebuildMemo != nil {
		lr.rebuildMemo[n] = clone
	}
	return clone
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
// not live at the insertion position (P-12). Computed sub-expressions
// (binops, consts — even the bound node itself) are pure SSA values the
// scheduler stamps at their innermost USE site; SSA value semantics keep
// them available after their minting loops close — so only the IV leaves
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
		return fmt.Errorf("the element-loop bound depends on the IV node %d, whose loop does not enclose the insertion position — the bound is not defined at the element-loop position", n.BasePtr().ID())
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
