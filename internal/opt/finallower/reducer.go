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
//	     dim), home (provenance.Inside — always populated, every dim is
//	     tiled T >= 2), and the between chain: the Loop blocks strictly
//	     nested between X's block and home's block on the parent chain
//	     (empty when the reduced dim is the LNT's only dim — the
//	     zero-between-dims degenerate).
//	(2.1) homogeneity (LR-8) + the specialized construction (LR-7: Soc
//	     0→Sequential over (init, lambda); 2→Warp / 1→Global over (init,
//	     lambda, ThreadMapping, staging) per R1/R2): the WARP/GLOBAL
//	     constructions additionally allocate a staging SMEM-class MemNew — one
//	     #warps × sizeof(T) row in the compiler's dynamic-SMEM pool at a
//	     disjoint D8 offset — inserted with the construction at the head
//	     end of the OWNING LNT's prologue Group (before the embedded root
//	     loop), stamped SN 0. The ThreadMapping operand resolves through
//	     this pass's own per-LNT objects (Run() inserts them FIRST).
//	(2.2) R3 — per-output-element brackets. The tiled shape (every dim
//	     T >= 2):
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
//	     The ZERO-BETWEEN-DIMS DEGENERATE (the reduced dim is the LNT's
//	     only dim; home is a direct child of X, so X's parent is trivially
//	     the innermost output-element scope) keeps the flat bracket: Init
//	     immediately before X, Combine immediately after, both in X's
//	     parental block — one output element per bracket, no element loops
//	     — and the qualifying epilogue consumers move in after Combine so
//	     the emitted Combine local and its consumers share one C++ scope
//	     (the lexical-scope law).
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
	// The home loop is prov.Inside ALWAYS: every dim is tiled (T >= 2, the
	// 2026-10-24 tile-disable ruling), so Inside is populated on every
	// provenance entry — a nil or erased Inside is record corruption, fail
	// closed (never fall back to X).
	home := prov.Inside
	if home == nil || home.BasePtr().Erased() {
		return fmt.Errorf("final-lower: reduce node %d: the provenance Inside loop of dim %d is dead — every dim is tiled (T >= 2), so a missing Inside is record corruption",
			red.BasePtr().ID(), dim.BasePtr().ID())
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
	// zero-between degenerate nests the producer deeper than home itself).
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
		// ZERO-BETWEEN-DIMS DEGENERATE of R-A (the 2026-10-24 tile-disable
		// reframing): the reduced dim is the LNT's ONLY dim — home (the
		// tile-inside walk) is a DIRECT child of X (the tile-between
		// loop), so X's parent is trivially the innermost output-element
		// scope and the flat bracket applies: Init immediately before X's
		// block, Combine immediately after it, both in X's parental block
		// (one output element per bracket).
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
		// The lexical-scope law (the historical HI-02 bug is why it exists):
		// the epilogue consumers of the Combine value move in with it — the
		// emitted `auto vN = reducer.Combine();` local and the consuming
		// store must share ONE lexical C++ scope (the flat bracket's parent
		// block), else the store reads a local declared inside a loop that
		// already closed.
		return lr.moveCombineConsumers(red, dim, ownerLNT, combineCall, parentBlk, lr.scopeDepth(parentBlk), nil)
	}

	// (2.2) R3 — the per-output-element restructure. fix7 §4 slice 1b
	// (quick-261101-mt9): on the qualified shape (Q0-Q5, see
	// amortizedQualifies) the per-element bracket chain is replaced by the
	// amortized reshape — ONE bracket per chunk — and buildElementLoop
	// never runs; on ANY miss the R3 restructure below runs UNCHANGED.
	if ok, err := lr.tryAmortizedElementLoop(red, dim, x, xBlk, homeBlk, betweenBlks, ownerLNT,
		construction, combineCall, reduceCall, piped, anchorBlk, anchorIdx, init, lambda, initTy); ok || err != nil {
		return err
	}
	return lr.buildElementLoop(red, dim, x, xBlk, homeBlk, betweenBlks, ownerLNT, construction, combineCall)
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
	construction, combineCall ir.Node) error {

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

	// Build the fresh element chain innermost-first (the AST blocks carry
	// the real statement order — since quick-261024-sdk removed the body
	// slot there is NO loop-to-loop operand to wire; the loops are chained
	// solely as Block items).
	//
	// The fresh walks' HEADERS are class-dependent (quick-261029-v9k, the
	// fix5 R2 one-tier amendment): MULTI-TIER (Warp/Global) builds UNIFORM
	// walks — init 0, step 1, soc 0, loc 1 — because the CTA-collective
	// Combine requires EVERY thread to walk EVERY element (the originals
	// may be per-thread-specialized, e.g. the coupled CTA walk; the
	// uniform rebuild flattens them, which is load-bearing there). The
	// ONE-TIER SeqReducer's accumulator is PER-THREAD: each thread owns
	// exactly the elements its OWN walks visit, so the fresh walks CLONE
	// the original headers (init/step operands reused, soc/dosoc/loc/tile
	// attributes copied) — the R3 collapse's WARP-covered element dim
	// carries its lane-chunk init lane x ceil(T/32), and resetting it to 0
	// (the uniform path) broke the chunk-local FRAG coordinate and the
	// store coordinate, GPU-proven as memcheck Invalid __local__ reads on
	// row_sum/softmax.
	//
	// R-A (HI-01, 2026-10-24 ruling): X's parent must be the INNERMOST
	// output-element loop — the construction runs innermost-first, so the
	// FIRST block built (i == len-1) is the innermost (elemBlk) while the
	// running chain head ends as the OUTERMOST (outermostBlk, the block
	// inserted into X's old parental position at the end). At
	// len(betweenBlks)==1 elemBlk == outermostBlk — the committed
	// 1-between-dim golden shape is bit-identical.
	oneTier := construction.KindOf() == ir.KindRuntimeSequentialReducer
	freshIVs := make([]ir.Node, len(betweenBlks))
	freshBlks := make([]*ir.ASTBlock, len(betweenBlks))
	var outermostBlk, elemBlk *ir.ASTBlock
	for i := len(betweenBlks) - 1; i >= 0; i-- {
		dimParam := ir.OperandsOf(betweenIVs[i])[0]
		var freshIV, freshWrapper ir.Node
		var attrs []ir.Node
		if oneTier {
			ivOps := ir.OperandsOf(betweenIVs[i])
			if len(ivOps) != 3 {
				return fmt.Errorf("final-lower: reduce node %d: between IV %d is not a (dim, init, step) dimvar (corrupted rewrite?)",
					red.BasePtr().ID(), betweenIVs[i].BasePtr().ID())
			}
			freshIV = lr.g.NewDimVar(lowerOrigin, dimParam, ivOps[1], ivOps[2])
			// Clone the original loop's annotation set (soc/dosoc/loc/tile)
			// — the attribute NODES are cloned (never shared with the
			// discarded original loop) to keep single ownership.
			lops := ir.OperandsOf(betweenBlks[i].Loop())
			if len(lops) < 2 {
				return fmt.Errorf("final-lower: reduce node %d: between loop %d carries no header (corrupted schedule?)",
					red.BasePtr().ID(), betweenBlks[i].Loop().BasePtr().ID())
			}
			for _, op := range lops[2:] {
				k, v := ir.KeyedAttributeOf(op)
				if k == "" {
					return fmt.Errorf("final-lower: reduce node %d: between loop %d carries a non-attribute header operand %d (corrupted schedule?)",
						red.BasePtr().ID(), betweenBlks[i].Loop().BasePtr().ID(), op.BasePtr().ID())
				}
				attrs = append(attrs, lr.g.NewKeyedAttribute(lowerOrigin, k, v))
			}
		} else {
			freshIV = lr.g.NewDimVar(lowerOrigin, dimParam, lr.u32c(0), lr.u32c(1))
			attrs = []ir.Node{
				lr.g.NewKeyedAttribute(lowerOrigin, "soc", lr.u32c(0)),
				lr.g.NewKeyedAttribute(lowerOrigin, "loc", lr.u32c(1)),
			}
		}
		freshWrapper = lr.g.NewBoundedDimVar(lowerOrigin, freshIV, boundMaxs[i])
		freshLoop := lr.g.NewForLoop(lowerOrigin, freshIV, freshWrapper, attrs...)
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
		outermostBlk = blk
		freshIVs[i] = freshIV
		freshBlks[i] = blk
	}
	// elemDepth is the INNERMOST element loop's depth (one level per between
	// dim below X's original parental scope) — the depth the bracket items
	// are stamped at.
	elemDepth := lr.scopeDepth(parentBlk) + len(betweenBlks)

	// D-j (fix3, R-I — completes fix2's blocked D-c): the between blocks'
	// Node items are RE-PLACED by OPERAND SCOPE, never discarded. Capture
	// them BEFORE the dissolve unlinks the blocks. Classification: an item
	// whose operand closure references X's OWN IV (a SURVIVING loop that
	// moves inside the element loops — per-round effectful items, the
	// frontend trigger's fill) re-places INSIDE X, immediately before home
	// (the innermost block dominated by that loop, still preceding its
	// memory successors — the per-round epoch count is preserved); every
	// other item (whose foreign coordinates are all between-IV-derived —
	// the source loads, the coordinate binops) re-places into ITS OWN
	// fresh element loop's item list, in order. fix2 D-c's block-relative
	// "re-place into L's item list, in order" is superseded by this scope
	// rule. The rebuild of every re-placed item runs AFTER the walk's and
	// the moved consumers' rebuilds (the pinned order — an earlier rebuild
	// emits an extra local and drifts the golden; fix2 SUMMARY Deviation 4).
	xIV := ir.OperandsOf(x)[0]
	betweenItems := make([][]ir.Node, len(betweenBlks))
	var xScoped []ir.Node
	for i, b := range betweenBlks {
		for _, it := range b.Items() {
			if it.Node == nil {
				continue
			}
			if operandClosure(it.Node)[xIV] {
				xScoped = append(xScoped, it.Node)
				continue
			}
			betweenItems[i] = append(betweenItems[i], it.Node)
		}
	}
	// The scope-scoped (X-class) items re-place inside X AFTER the walk
	// rebuild below (so the walk never covers them); the fresh-loop items
	// insert now, in order at their block's head, ahead of the nested
	// chain block / Init.
	var rePlaced []ir.Node
	for i, items := range betweenItems {
		blk := freshBlks[i]
		for j, n := range items {
			if err := blk.InsertItemAt(j, ir.ASTItem{Node: n}); err != nil {
				return fmt.Errorf("final-lower: reduce node %d: re-placing a between item into its element loop: %w", red.BasePtr().ID(), err)
			}
			n.BasePtr().SetSched(lr.scopeDepth(parentBlk)+1+i, blk)
			rePlaced = append(rePlaced, n)
		}
	}

	// Init: the FIRST item of the innermost element loop — the per-element
	// bracket opens here, immediately before X's block (R-A: X is appended
	// right after, nothing sits between).
	initCall := lr.g.NewRuntimeCallMethod(lowerOrigin, "Init", construction)
	if err := elemBlk.InsertItemAt(len(betweenItems[len(betweenItems)-1]), ir.ASTItem{Node: initCall}); err != nil {
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

	// D-j, the scope-scoped half: items whose operand closure references
	// X's own IV re-place INSIDE X, immediately BEFORE home — the innermost
	// block dominated by that surviving loop, still preceding the memory
	// successors (each read of the item's buffer must sit inside home's
	// subtree; anything else is an uncovered shape and fails closed, R-D).
	// Inserted AFTER the walk rebuild so the walk never covers them; the
	// re-placed items' own rebuild runs after the consumers' (below).
	if len(xScoped) > 0 {
		homeIdx := -1
		for i, it := range xBlk.Items() {
			if it.Block == homeBlk {
				homeIdx = i
				break
			}
		}
		if homeIdx < 0 {
			return fmt.Errorf("final-lower: reduce node %d: the home block is not an item of X's block after the restructure (corrupted bracket?)", red.BasePtr().ID())
		}
		xDepth := lr.scopeDepth(xBlk)
		for j, n := range xScoped {
			if err := xBlk.InsertItemAt(homeIdx+j, ir.ASTItem{Node: n}); err != nil {
				return fmt.Errorf("final-lower: reduce node %d: re-placing a scope-constrained item inside X: %w", red.BasePtr().ID(), err)
			}
			n.BasePtr().SetSched(xDepth, xBlk)
			rePlaced = append(rePlaced, n)
			// A re-placed FILL additionally keeps its per-round epoch
			// contract only if every buffer read follows it: assert the
			// reads sit inside home's subtree.
			if sc, ok := ir.StorageClassOf(n); ok && sc != ir.StorageGMEM && n.KindOf() == ir.KindMemStore {
				if err := lr.assertFillReadsInside(red, n, homeBlk); err != nil {
					return err
				}
			}
		}
	}

	// The orphan guard + disposal. Since quick-261024-sdk removed the body
	// slot, X carries no body operand; the IR-level chain wiring is
	// tile-lower's canonical-chain "nest" attribute — every non-root chain
	// loop holds one pointing at its canonical parent. The restructure
	// dissolves the between loops, so EVERY referrer — always such a "nest"
	// attribute — is retargeted onto the nearest surviving canonical
	// ancestor (walking the dissolved chain upward through the "nest"
	// attrs): the post-fix1 form of the old X-body→home rewire. Retargeting
	// uniformly — dissolved bearers included — leaves no dissolved loop
	// referencing another, so finalizeDissolutions can erase the jobs in
	// any order (each dissolved bearer's own "nest" attr is one of its
	// header nodes and dies with it, now pointing at a survivor). Anything
	// not a "nest" attribute still referencing a dissolved loop is a corrupted
	// restructure — fail closed. The erasure itself is DEFERRED to
	// finalizeDissolutions (after Map Lower): the piped Map's projection
	// machinery — the producing Map, its As, and its epilogue-anchored
	// lambda — still references the walk's coordinate expressions until
	// Map Lower disposes them, so the between dimvar's use list only
	// reaches zero there.
	dissolving := make(map[ir.Node]bool, len(betweenBlks))
	for _, b := range betweenBlks {
		dissolving[b.Loop()] = true
	}
	// nestParentOf reads a loop's canonical parent through its own "nest"
	// attribute operand (nil when the loop is the chain root).
	nestParentOf := func(loop ir.Node) ir.Node {
		for _, op := range ir.OperandsOf(loop) {
			if key, val := ir.KeyedAttributeOf(op); key == "nest" && val != nil {
				return val
			}
		}
		return nil
	}
	type retarget struct {
		use    ir.SSAUse
		target ir.Node
	}
	var retargets []retarget
	for _, b := range betweenBlks {
		loop := b.Loop()
		for _, u := range ir.UsesOf(loop) { // collected before any SetDef mutates use lists
			usr := u.User()
			if usr == nil || usr.BasePtr() == nil {
				continue
			}
			if key, _ := ir.KeyedAttributeOf(usr); key != "nest" {
				return fmt.Errorf("final-lower: reduce node %d: node %d (%s) still references the dissolved between loop %d after the element-loop restructure",
					red.BasePtr().ID(), usr.BasePtr().ID(), usr.KindOf(), loop.BasePtr().ID())
			}
			// Walk the canonical ancestor chain (each loop's own "nest" attr)
			// past the dissolved region; the attr retargets onto the first
			// surviving ancestor. The walk always advances at least one hop —
			// `loop` itself is dissolving, so it can never be the target.
			target := loop
			for {
				parent := nestParentOf(target)
				if parent == nil || parent.BasePtr() == nil || parent.BasePtr().Erased() {
					return fmt.Errorf("final-lower: reduce node %d: the dissolved between loop %d has no surviving canonical ancestor to retarget its nest reference onto",
						red.BasePtr().ID(), loop.BasePtr().ID())
				}
				target = parent
				if !dissolving[parent] {
					break
				}
			}
			retargets = append(retargets, retarget{use: u, target: target})
		}
	}
	for _, rt := range retargets {
		if err := lr.mctx.SetDef(rt.use, rt.target); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: retargeting a dissolved between loop's nest reference: %w", red.BasePtr().ID(), err)
		}
	}
	for i := range betweenBlks {
		loop := betweenBlks[i].Loop()
		// The erasures are DEFERRED to finalizeDissolutions (after Map
		// Lower): the piped Map's projection machinery — the producing
		// Map, its As, and its epilogue-anchored lambda — still references
		// the walk's coordinate expressions until Map Lower rewires those
		// operands away (fix2 D-a: only the header nodes themselves are
		// ever erased, never a swept closure).
		lr.dissolved = append(lr.dissolved, dissolveJob{loop: loop, red: red})
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
	// innermost element loop (shared gates with the flat bracket of the
	// zero-between degenerate).
	if err := lr.moveCombineConsumers(red, dim, ownerLNT, combineCall, elemBlk, elemDepth, subs); err != nil {
		return err
	}

	// D-j's PINNED rebuild order: the re-placed items' between-IV rebuild
	// runs LAST — after the walk's and the moved consumers' rebuilds (fix2
	// SUMMARY Deviation 4: an earlier rebuild of a re-placed pure item makes
	// the consumers keep referencing it, emitting an extra local and
	// drifting the golden; rebuilt last, a pure re-placed item whose
	// consumers read rebuilt clones ends user-free and the emitter's
	// unreferenced-pure-item skip elides it).
	for _, n := range rePlaced {
		if err := lr.rebuildOperands(n, subs); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: rebuilding a re-placed item's between-dim coordinates: %w", red.BasePtr().ID(), err)
		}
	}
	return nil
}

// assertFillReadsInside fail-closes the D-j scope rule's memory-successor
// clause (R-D, fix3 §5): a fill re-placed inside X must still precede every
// read of its buffer — each live tiered load on the buffer must sit inside
// home's subtree (strictly after the fill's position). A read that would
// land outside X (e.g. re-placed into an element loop ABOVE the bracket)
// is an uncovered shape: the lower never improvises a placement for it.
func (lr *lowering) assertFillReadsInside(red, fill ir.Node, homeBlk *ir.ASTBlock) error {
	buf := ir.OperandsOf(fill)[1]
	if buf == nil {
		return nil
	}
	for _, n := range lr.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemLoad {
			continue
		}
		if sc, _ := ir.StorageClassOf(n); sc == ir.StorageGMEM {
			continue
		}
		ops := ir.OperandsOf(n)
		if len(ops) == 0 || ops[0] == nil || ops[0].BasePtr() != buf.BasePtr() {
			continue
		}
		blk := n.BasePtr().SchedBlock()
		if blk == nil || findSubBlock(homeBlk, blk) == nil {
			return fmt.Errorf("final-lower: reduce node %d: the re-placed fill (node %d) has a buffer read (node %d) outside the home loop — the scope-derived position would not precede it (R-D: uncovered shape, needs a ruling)",
				red.BasePtr().ID(), fill.BasePtr().ID(), n.BasePtr().ID())
		}
	}
	return nil
}

// moveCombineConsumers implements R3's consumer half (and the flat-bracket
// twin of the zero-between degenerate, 2026-10-24): the owning LNT's
// epilogue items whose operand closure reaches combineCall relocate into
// dst, inserted IMMEDIATELY AFTER the Combine call's item — the Combine
// local and its consumers must share one lexical C++ scope (the emitter
// materializes the call as a local at its item position; a consumer
// outside that scope reads a dead local). Shared gates, in order:
//
//	(a) reduced-dim poison — a consumer referencing the reduced dim's loop
//	    dimvars would read an iteration variable of the WRONG element's
//	    strip (doc/tca.md §5.10 point 4);
//	(b) the between-IV coordinate rebuild onto the fresh element IVs when
//	    subs != nil (the zero-between degenerate passes nil — its
//	    coordinates are defined at scopes enclosing the bracket already);
//	(c) the generalized enclosure assertion (assertBoundEncloses over
//	    whole closures, P-12): every DimVar in a moved consumer's closure
//	    must be defined at a block enclosing dst — a coordinate not live at
//	    the bracket position is a named error, never an emitted out-of-scope
//	    reference.
func (lr *lowering) moveCombineConsumers(red, dim ir.Node, ownerLNT *ir.ASTLNT,
	combineCall ir.Node, dst *ir.ASTBlock, depth int, subs map[ir.Node]ir.Node) error {
	moved, epilogue, err := lr.qualifyCombineConsumers(red, dim, ownerLNT, combineCall)
	if err != nil || len(moved) == 0 {
		return err
	}
	// (b) The moved consumers' between-dim coordinates rebuild onto the
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
	// (c) The generalized enclosure assertion.
	if err := lr.assertConsumerEnclosure(red, moved, dst); err != nil {
		return err
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

// qualifyCombineConsumers implements the shared collect + gate (a) half of
// the consumer move (the per-element brackets and the fix7 §4 amortized
// bracket both consume the Combine value the same way): the owning LNT's
// epilogue items (in order, the end node excluded) whose operand closure
// reaches combineCall, poison-checked against the reduced dim's loop
// dimvars (a coordinate referencing them would read an iteration variable
// of the WRONG element's strip). Returns the qualifying nodes and their
// epilogue block. The between-IV reference alone must NOT qualify (the
// piped Map's projection lambda is a PRODUCER for the nest; Map Lower
// disposes it where it sits).
func (lr *lowering) qualifyCombineConsumers(red, dim ir.Node, ownerLNT *ir.ASTLNT,
	combineCall ir.Node) ([]ir.Node, *ir.ASTBlock, error) {
	epilogue, err := lr.epilogueOf(ownerLNT)
	if err != nil {
		return nil, nil, err
	}
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
		return nil, epilogue, nil
	}
	// (a) Reduced-dim dimvars are poison in moved consumers: their walks
	// re-execute per element inside the bracket — the consumer would read
	// an iteration variable of the WRONG element's strip.
	reducedVars := lr.reducedDimVars(dim)
	for _, m := range moved {
		if containsAny(operandClosure(m), reducedVars) {
			return nil, nil, fmt.Errorf("final-lower: moved consumer node %d (%s) references a reduced-dim dimvar — its value is not defined at the element-loop position (doc/tca.md §5.10 point 4)",
				m.BasePtr().ID(), m.KindOf())
		}
	}
	return moved, epilogue, nil
}

// assertConsumerEnclosure is gate (c) of the consumer move (P-12): every
// DimVar in every moved consumer's closure must be defined at a block
// enclosing dst — a coordinate not live at the bracket position is a named
// error, never an emitted out-of-scope reference.
func (lr *lowering) assertConsumerEnclosure(red ir.Node, moved []ir.Node, dst *ir.ASTBlock) error {
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
	return nil
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
	// Always a NON-nil slice on success: the EMPTY between set (the
	// zero-between-dims degenerate — the reduced dim is the LNT's only
	// dim, home a direct child of X) must stay distinguishable from the
	// not-found nil — `append(nil-slice)` with zero elements would return
	// nil and make the flat-bracket branch forever unreachable (the
	// historical HI-02 latent bug, the reason this non-nil guarantee
	// exists: every flat-bracket reduce died here as a "corrupted
	// schedule").
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

// ---------------------------------------------------------------------------
// fix7 §4 slice 1b — the amortized bracket (quick-261101-mt9)
// ---------------------------------------------------------------------------
//
// The §4 chunk-fold DECISION (recorded in design/fix/fix7.md §4 and the
// 261101-mt9 plan): codegen emits an EXPLICIT chunk-wide accumulator — a
// FRAG-class MemNew block spanning X's FULL walk — with the combiner body
// inlined per element. NO runtime SeqReducer chunk-fold entry, NO
// per-round partials: the block accumulates across ALL of X's chunk rounds
// by construction, so there is nothing to fold. Rationale: (1) the
// per-chunk accumulator is inherently a chunk-wide ARRAY (one accumulator
// per output element of the chunk) — a runtime SeqReducer holds ONE scalar
// acc_, so a runtime entry would need a NEW template-sized block-reducer
// class (new runtime surface against the minimal-runtime doctrine; R1
// struck template sizing from the reducer protocol) while codegen would
// still declare/emit the block; (2) the combiner body is a pure homogeneous
// scalar expression (LR-8-checked at construction) — map lambdas are
// already inlined the same way by Map Lower (RAUW onto the lambda body
// expression, maplower.go), so cloning it over (acc-slot load, piped)
// reuses the verified rebuildWithSubs machinery; (3) runtime zero-diff —
// runtime/tca/reducer.h stays untouched (layernorm/transpose keep their
// SeqReducer — they fail the qualification below).
//
// LR-3 AMENDMENT (shape-scoped): LR-3's "the compiler emits no lambda-call
// node and never reads current_value/init" is amended FOR THIS SHAPE ONLY
// by fix7 §4's explicit authorization of the codegen-accumulator branch:
// the amortized path mints no lambda-call node (the body is INLINED, the
// same mechanism Map Lower uses), but it DOES read the combiner's
// current_value binding (lambda param 0 <- the acc-slot load) — never the
// init (the init walk stores the CONSTRUCTION'S OWN init operand). Every
// other reducer shape keeps LR-3 verbatim.
//
// Where the reshape lives (verified empirically at the RED commit — the
// plan's flat-branch premise did not match the emitted defect): the suite's
// const-innermost reduces carry a NON-EMPTY between chain over the reduced
// dim's provenance — X (the e4 tile-between loop) directly contains the
// thread-specialized R coupled-inside walk, which contains the uniform SEQ
// C inside walk, which contains home (between = 2). The per-element bracket
// the §4 audit measured is buildElementLoop's R3 product (one bracket per
// INNERMOST output element), so the amortization probe sits at the R3
// call site, BEFORE the restructure, and subsumes it: the outer between
// loops (thread-specialized trips) SURVIVE (unlike R3, which dissolves
// them), the innermost uniform SEQ between loop becomes the element walk E
// and moves INSIDE X, and fresh init/store walks cloning E's header drain
// the acc block around X. Because the between loops keep their identity,
// NO coordinate rebuild runs in the walk — only the moved consumers
// rebuild E's IV onto the store-walk IV.
//
// The fill-placement-gap verdict (recorded in the plan): INSEPARABLE — the
// fill's gap-closing position IS its current X-scoped derived position
// (fill position = dominated-by-operand-scope LICM, the 2026-10-25
// ruling); closing the §3.4 gap REQUIRES inverting the nesting (element
// walk inside X), which is this reshape. The D-j scope discipline is
// preserved: the fill stays an item of X's block, the reads stay in home
// inside the element loop inside X — the fill precedes every read of its
// buffer within each X round by construction (the assertFillReadsInside
// epoch contract holds structurally; a stranded fill coordinate fails Q3
// closed). The schedule pass is untouched.

// amortizedPlan is the resolved qualification context for one reduce's
// amortized bracket — every field read in the qualification probe,
// consumed by the mutating apply step.
type amortizedPlan struct {
	eBlk                       *ir.ASTBlock // the element walk E (innermost between block)
	eIV, eBoundMax             ir.Node      // E's header
	eDimParam, eInit, eStep    ir.Node
	eAttrs                     []ir.Node
	grandBlk                   *ir.ASTBlock // B[0]'s enclosing block (xIdx = X's item in it)
	xIdx                       int
	xIV                        ir.Node // X's own IV (the D-j scope key)
	chainHead                  *ir.ASTBlock // B[1] (nil when len(betweenBlks) == 1)
	chainIdx                   int          // B[1]'s item index inside B[0]
	chunk                      float64      // the acc block's element count (E's clamp const)
	chunkDim                   string       // E's dim name (the acc buffer's debug dim)
	elem                       ir.ElemType
	body, param0Ref, param1Ref ir.Node // the combiner body and its two param refs
}

// tryAmortizedElementLoop evaluates the fix7 §4 qualification (Q0-Q5) and,
// on success, replaces the per-element bracket chain with the amortized
// reshape. Returns handled=true when the reshape ran (err carries its
// failure), handled=false when ANY qualification check missed — the caller
// then runs buildElementLoop unchanged. The probe is READ-ONLY: nothing is
// mutated before every check has passed.
func (lr *lowering) tryAmortizedElementLoop(red, dim, x ir.Node, xBlk, homeBlk *ir.ASTBlock,
	betweenBlks []*ir.ASTBlock, ownerLNT *ir.ASTLNT, construction, combineCall, reduceCall, piped ir.Node,
	anchorBlk *ir.ASTBlock, anchorIdx int, init, lambda ir.Node, initTy ir.Type) (bool, error) {
	plan, ok := lr.amortizedQualifies(red, dim, x, xBlk, homeBlk, betweenBlks, ownerLNT, construction, combineCall, lambda, initTy)
	if !ok {
		return false, nil
	}
	return true, lr.applyAmortizedElementLoop(red, dim, ownerLNT, construction, combineCall,
		reduceCall, piped, anchorBlk, anchorIdx, init, lambda, xBlk, homeBlk, betweenBlks, plan)
}

// amortizedQualifies is the read-only Q0-Q5 probe for the R3 (non-empty
// between chain) shape. ANY miss returns ok=false (fail closed to today's
// per-element brackets — never a guess). Checks, in order:
//
//	Q1: the construction is a one-tier SeqReducer (Warp/Global multi-tier
//	    CTA-collective sync protocols never qualify).
//	Q4: the between chain's INNERMOST block E (home's direct parent) is a
//	    uniform SEQUENTIAL walk, and every OUTER between block is
//	    thread-specialized (BlockThrLoop — per-trip brackets, as today);
//	    each between body holds exactly its chain child plus re-placeable
//	    node items (a second nested Block disqualifies). For the single-
//	    between degenerate the loop above X's parental block must likewise
//	    be non-sequential (the two-uniform-SEQ-walk element chunk —
//	    layernorm/transpose's class — keeps today's shape).
//	Q4d: E's static chunk is > 1 element — a 1-element amortization saves
//	    zero ceremony (init walk + store walk ≡ Init + Combine) and marks
//	    the element dim not chunk-covered (layernorm/transpose — the suite's
//	    Q4-failing pair — are exactly this class: E's clamp target is 1).
//	Q0/Q3: every tiered fill item directly in X is FRAG-class (an SMEM
//	    cooperative fill disqualifies) and its window coordinate closure
//	    references NEITHER E's IV (the inversion's safety condition: the
//	    fill rides at X level) — a stranded coordinate fails closed.
//	Q5: at least one Combine-value consumer exists in the owning LNT's
//	    epilogue, none references a reduced-dim dimvar, and none is a
//	    reduce-after-reduce chain (the Combine feeding another reduce).
//	The combiner lambda must bind exactly (current_value, now) with both
//	params referenced in the body (the acc/piped substitution sites), and
//	E's clamped bound must expose the static chunk const that sizes the
//	acc block.
func (lr *lowering) amortizedQualifies(red, dim, x ir.Node, xBlk, homeBlk *ir.ASTBlock,
	betweenBlks []*ir.ASTBlock, ownerLNT *ir.ASTLNT, construction, combineCall, lambda ir.Node,
	initTy ir.Type) (*amortizedPlan, bool) {
	// Q1: one-tier only.
	if construction.KindOf() != ir.KindRuntimeSequentialReducer {
		return nil, false
	}
	n := len(betweenBlks)
	// Q4a: the innermost between block E is a uniform sequential walk with
	// a well-formed (dim, init, step)/(iv, max) header; every outer between
	// block is thread-specialized; each between body holds exactly its
	// chain child plus node items.
	for i, b := range betweenBlks {
		want := ir.BlockThrLoop
		if i == n-1 {
			want = ir.BlockSeqLoop
		}
		if b.Kind() != want {
			return nil, false
		}
		nBlocks := 0
		for _, it := range b.Items() {
			if it.Block != nil {
				nBlocks++
			}
		}
		if nBlocks != 1 {
			return nil, false
		}
	}
	eBlk := betweenBlks[n-1]
	eLoop := eBlk.Loop()
	if eLoop == nil {
		return nil, false
	}
	eOps := ir.OperandsOf(eLoop)
	if len(eOps) < 2 || eOps[0] == nil || eOps[1] == nil {
		return nil, false
	}
	eIV := eOps[0]
	ivOps := ir.OperandsOf(eIV)
	if len(ivOps) != 3 || ivOps[0] == nil {
		return nil, false
	}
	bOps := ir.OperandsOf(eOps[1])
	if len(bOps) != 2 || bOps[0] == nil || bOps[0].BasePtr() != eIV.BasePtr() {
		return nil, false
	}
	eBoundMax := bOps[1]
	// Q4b: the single-between degenerate — the loop above X's parental
	// block must be thread-specialized (a second uniform SEQ walk above
	// the element scope disqualifies, the layernorm/transpose class).
	if n == 1 {
		if grandBlk, _ := lr.findBlockItem(xBlk); grandBlk != nil && grandBlk.Kind() == ir.BlockSeqLoop {
			return nil, false
		}
	}
	// Q0/Q3: the tiered fills directly in X.
	for _, it := range xBlk.Items() {
		if it.Node == nil {
			continue
		}
		sc, isMem := ir.StorageClassOf(it.Node)
		if !isMem || it.Node.KindOf() != ir.KindMemStore || sc == ir.StorageGMEM {
			continue
		}
		if sc == ir.StorageSMem {
			return nil, false // Q0: a cooperative SMEM fill disqualifies
		}
		if operandClosure(it.Node)[eIV] {
			return nil, false // Q3: a fill window coordinate rides the element IV
		}
	}
	// Q4d + the acc block's static chunk: E's clamped bound max — either a
	// const itself (the unclamped degenerate) or the 1a clamp chain's const
	// operand (umin(chunkConst, remaining)). A chunk of ONE saves zero
	// ceremony (the init/store walks degenerate to the Init/Combine pair)
	// and marks the element dim not chunk-covered — keep today's shape.
	chunk, ok := staticChunkOf(eBoundMax)
	if !ok || chunk <= 1 {
		return nil, false
	}
	nt, ok := initTy.(*ir.NumericType)
	if !ok || nt == nil || nt.Rank() != 0 {
		return nil, false
	}
	// The combiner lambda: exactly (current_value, now), both referenced.
	lamOps := ir.OperandsOf(lambda)
	if len(lamOps) != 3 {
		return nil, false
	}
	body := lamOps[2]
	if body == nil {
		return nil, false
	}
	var paramRefs [2]ir.Node
	refs := 0
	for nd := range operandClosure(body) {
		if nd.KindOf() != ir.KindParam {
			continue
		}
		pOps := ir.OperandsOf(nd)
		if len(pOps) != 1 || pOps[0] == nil {
			continue
		}
		for i := 0; i < 2; i++ {
			if pOps[0].BasePtr() == lamOps[i].BasePtr() {
				if paramRefs[i] != nil {
					return nil, false // a param bound twice is corruption
				}
				paramRefs[i] = nd
				refs++
			}
		}
	}
	if refs != 2 || paramRefs[0] == nil || paramRefs[1] == nil {
		return nil, false
	}
	// Q5: consumers exist, poison-free, no reduce-after-reduce chain.
	moved, _, err := lr.qualifyCombineConsumers(red, dim, ownerLNT, combineCall)
	if err != nil || len(moved) == 0 {
		return nil, false
	}
	for _, m := range moved {
		for nd := range operandClosure(m) {
			if nd.KindOf() == ir.KindReduce {
				return nil, false // the Combine feeding another reduce — chain partials are out of scope
			}
		}
	}
	// E's annotation set (the oneTier header-clone idiom of buildElementLoop).
	var attrs []ir.Node
	for _, op := range eOps[2:] {
		k, _ := ir.KeyedAttributeOf(op)
		if k == "" {
			return nil, false
		}
		attrs = append(attrs, op)
	}
	grandBlk, xIdx := lr.findBlockItem(xBlk)
	if grandBlk == nil || xIdx < 0 {
		return nil, false
	}
	plan := &amortizedPlan{
		eBlk: eBlk, eIV: eIV, eBoundMax: eBoundMax,
		eDimParam: ivOps[0], eInit: ivOps[1], eStep: ivOps[2],
		eAttrs:   attrs,
		grandBlk: grandBlk, xIdx: xIdx, xIV: ir.OperandsOf(x)[0],
		chunk: chunk, chunkDim: ir.InputParamNameOf(ivOps[0]),
		elem: nt.Elem(), body: body, param0Ref: paramRefs[0], param1Ref: paramRefs[1],
	}
	if n > 1 {
		// The chain child that moves into X: B[1], nested inside B[0].
		chainIdx := -1
		for i, it := range betweenBlks[0].Items() {
			if it.Block == betweenBlks[1] {
				chainIdx = i
				break
			}
		}
		if chainIdx < 0 {
			return nil, false
		}
		plan.chainHead = betweenBlks[1]
		plan.chainIdx = chainIdx
	}
	return plan, true
}

// staticChunkOf extracts the static chunk const from the element walk's
// clamped bound max: the bound is either the const itself (the unclamped
// degenerate) or the 1a clamp chain's umin(chunkConst, remaining) — the
// const operand IS the chunk the acc block sizes to. Anything else (both
// operands const, no const at all) is ambiguous — fail closed.
func staticChunkOf(boundMax ir.Node) (float64, bool) {
	if v, ok := ir.ConstValueOf(boundMax); ok {
		return v, true
	}
	if boundMax.KindOf() != ir.KindCall || ir.CallNameOf(boundMax) != "umin" {
		return 0, false
	}
	ops := ir.OperandsOf(boundMax)
	var chunk float64
	nConst := 0
	for _, op := range ops {
		if v, ok := ir.ConstValueOf(op); ok {
			chunk, nConst = v, nConst+1
		}
	}
	if nConst != 1 {
		return 0, false
	}
	return chunk, true
}

// applyAmortizedElementLoop performs the reshape on the qualified shape,
// subsuming buildElementLoop. Assumes the plan was built by
// amortizedQualifies against an unmutated tree. Steps (see the file
// comment for the design rationale):
//
// (a)  the acc buffer — a FRAG-class MemNew with an InternTensor type
//
//	stamping the element chunk (the 1a lower_cache idiom; walk.go
//	prints the decl verbatim — the P8 no-self-typing exception, same
//	as tile-lower's buffer binding), inserted into the prologue head;
//
// (b)  the init/store walks — FRESH loops cloning E's header (dimParam,
//
//	init/step operands reused, annotation NODES cloned — the oneTier
//	idiom of buildElementLoop);
//
// (c)  the acc update replaces the Reduce call item inside home: the
//
//	value is a rebuildWithSubs clone of the combiner body with
//	{param0 <- acc load at E's IV, param1 <- the piped node}, stored
//	to the acc at E's IV; the Reduce call is then explicitly erased
//	(P-7: no whole-graph DCE sweep);
//
// (d)  the re-parent: the outer between block B[0] takes X's old position
//
//	in its enclosing block; X (carrying the fills) nests inside B[0]
//	between the init and store walks; the inner chain B[1..] (E with
//	home inside) moves into X immediately after the fills — the fill
//	precedes every read of its buffer within each X round by
//	construction (the assertFillReadsInside epoch contract); for the
//	single-between degenerate E itself moves into X and the walks take
//	X's old position;
//
// (e)  every Combine-value consumer rewires its value operand onto a
//
//	fresh acc load at the STORE-walk IV (SetDef through the choke
//	point), rebuilds its E-IV coordinates onto the store-walk IV, and
//	relocates into the store walk after the shared gates;
//
// (f)  the init walk stores the CONSTRUCTION'S OWN init operand (never a
//
//	hardcoded zero) to the acc at the init-walk IV;
//
// (g)  the Init/Reduce/Combine calls and the SeqReducer construction are
//
//	explicitly erased (the orphaned combiner lambda stays in the sea —
//	the maplower precedent, NOT this pass's business; the emitter
//	emits no symbol because registerLambda runs for surviving
//	constructions only);
//
// (h)  every minted node is Sched-stamped at its block (the post-lower
//
//	verifier's dominance check reads the stamps).
func (lr *lowering) applyAmortizedElementLoop(red, dim ir.Node, ownerLNT *ir.ASTLNT,
	construction, combineCall, reduceCall, piped ir.Node,
	anchorBlk *ir.ASTBlock, anchorIdx int, init, lambda ir.Node,
	xBlk, homeBlk *ir.ASTBlock, betweenBlks []*ir.ASTBlock, p *amortizedPlan) error {
	id := red.BasePtr().ID()

	// (a) The acc buffer: chunk-wide FRAG block, prologue-declared.
	eb, err := elemBytes(p.elem)
	if err != nil {
		return fmt.Errorf("final-lower: reduce node %d: amortized bracket: %w", id, err)
	}
	prologue, err := lr.prologueOf(ownerLNT)
	if err != nil {
		return err
	}
	acc := lr.g.NewFragNew(lowerOrigin, lr.u32c(p.chunk*float64(eb)))
	if err := lr.mctx.SetType(acc, ir.InternTensor(p.elem,
		[]ir.DimRef{{Name: p.chunkDim, Extent: uint64(p.chunk)}})); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: amortized bracket: binding the acc block type: %w", id, err)
	}
	conIdx, ok := itemIndexOf(prologue, construction)
	if !ok {
		return fmt.Errorf("final-lower: reduce node %d: the reducer construction is not a prologue item (corrupted head?)", id)
	}
	if err := prologue.InsertItemAt(conIdx, ir.ASTItem{Node: acc}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: inserting the acc block: %w", id, err)
	}
	acc.BasePtr().SetSched(0, prologue)

	// (b) The init/store walks: fresh loops cloning E's header.
	var hostBlk *ir.ASTBlock
	if p.chainHead != nil {
		hostBlk = betweenBlks[0] // the thread-specialized trips own the bracket
	} else {
		hostBlk = p.grandBlk // the single-between degenerate: X's old scope
	}
	hostDepth := lr.scopeDepth(hostBlk)
	freshLoop := func() (ir.Node, *ir.ASTBlock, error) {
		freshIV := lr.g.NewDimVar(lowerOrigin, p.eDimParam, p.eInit, p.eStep)
		var attrs []ir.Node
		for _, op := range p.eAttrs {
			k, v := ir.KeyedAttributeOf(op)
			attrs = append(attrs, lr.g.NewKeyedAttribute(lowerOrigin, k, v))
		}
		freshBIV := lr.g.NewBoundedDimVar(lowerOrigin, freshIV, p.eBoundMax)
		fresh := lr.g.NewForLoop(lowerOrigin, freshIV, freshBIV, attrs...)
		blk, err := ir.NewASTLoop(fresh)
		if err != nil {
			return nil, nil, err
		}
		fresh.BasePtr().SetSched(hostDepth+1, blk)
		freshIV.BasePtr().SetSched(hostDepth+1, blk)
		freshBIV.BasePtr().SetSched(hostDepth+1, blk)
		return freshIV, blk, nil
	}
	initIV, initBlk, err := freshLoop()
	if err != nil {
		return fmt.Errorf("final-lower: reduce node %d: amortized bracket: building the init walk: %w", id, err)
	}
	storeIV, storeBlk, err := freshLoop()
	if err != nil {
		return fmt.Errorf("final-lower: reduce node %d: amortized bracket: building the store walk: %w", id, err)
	}

	// (f) The init walk body: the construction's OWN init operand (never a
	// hardcoded zero) stored per acc slot.
	initStore := lr.g.NewFragStore(lowerOrigin, init, acc, initIV)
	initBlk.AppendItem(ir.ASTItem{Node: initStore})
	initStore.BasePtr().SetSched(hostDepth+1, initBlk)

	// (c) The acc update replaces the Reduce call item inside home.
	accLoadE := lr.g.NewFragLoad(lowerOrigin, acc, p.eIV)
	value := lr.rebuildWithSubs(p.body, map[ir.Node]ir.Node{
		p.param0Ref: accLoadE,
		p.param1Ref: piped,
	})
	accStore := lr.g.NewFragStore(lowerOrigin, value, acc, p.eIV)
	anchorBlk.RemoveItemsIf(func(it ir.ASTItem) bool {
		return it.Node != nil && it.Node.BasePtr() == reduceCall.BasePtr()
	})
	// anchorIdx+1: the Reduce call's own slot (immediately after the piped
	// producer's item) — the acc update reads the producer's local.
	if err := anchorBlk.InsertItemAt(anchorIdx+1, ir.ASTItem{Node: accStore}); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: inserting the acc update: %w", id, err)
	}
	accStore.BasePtr().SetSched(lr.scopeDepth(anchorBlk), anchorBlk)
	if err := lr.mctx.Erase(reduceCall); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: erasing the Reduce call: %w", id, err)
	}

	// (d) The re-parent.
	if p.chainHead != nil {
		// B[0] takes X's old position; X nests inside B[0] between the
		// walks; the inner chain (E with home inside) moves into X right
		// after the fills.
		b0 := betweenBlks[0]
		p.grandBlk.RemoveItemsIf(func(it ir.ASTItem) bool { return it.Block == xBlk })
		xTail := -1
		for i, it := range xBlk.Items() {
			if it.Block == b0 {
				xTail = i
				break
			}
		}
		if xTail < 0 {
			return fmt.Errorf("final-lower: reduce node %d: the outer between block is not an item of X (corrupted bracket?)", id)
		}
		xBlk.RemoveItemsIf(func(it ir.ASTItem) bool { return it.Block == b0 })
		if err := p.grandBlk.InsertItemAt(p.xIdx, ir.ASTItem{Block: b0}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: placing the trip loop: %w", id, err)
		}
		b0.RemoveItemsIf(func(it ir.ASTItem) bool { return it.Block == p.chainHead })
		// D-j (the buildElementLoop discipline this reshape subsumes): B[0]'s
		// node items whose operand closure references X's OWN IV (the
		// fill, its source load, the per-round clamps — everything the
		// scheduler landed in the trip walk) re-place INSIDE X, ahead of
		// the element chain — the fill stays an item of X's block and
		// precedes every read of its buffer within each X round. The
		// remaining items (B[0]-scoped, e.g. the trip coordinate binop)
		// stay where they are.
		var xScoped []ir.Node
		for _, it := range b0.Items() {
			if it.Node != nil && operandClosure(it.Node)[p.xIV] {
				xScoped = append(xScoped, it.Node)
			}
		}
		if len(xScoped) > 0 {
			b0.RemoveItemsIf(func(it ir.ASTItem) bool {
				if it.Node == nil {
					return false
				}
				for _, n := range xScoped {
					if it.Node.BasePtr() == n.BasePtr() {
						return true
					}
				}
				return false
			})
			for _, n := range xScoped {
				if err := xBlk.InsertItemAt(xTail, ir.ASTItem{Node: n}); err != nil {
					return fmt.Errorf("final-lower: reduce node %d: re-placing an X-scoped item: %w", id, err)
				}
				xTail++
				n.BasePtr().SetSched(lr.scopeDepth(xBlk), xBlk)
			}
		}
		if err := xBlk.InsertItemAt(xTail, ir.ASTItem{Block: p.chainHead}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: nesting the element chain inside X: %w", id, err)
		}
		// The init/store/X trio takes the chain child's old position — the
		// index clamps because the xScoped extraction above may have shrunk
		// B[0] (the remaining items stay ahead).
		hostIdx := p.chainIdx
		if hostIdx > len(b0.Items()) {
			hostIdx = len(b0.Items())
		}
		if err := b0.InsertItemAt(hostIdx, ir.ASTItem{Block: storeBlk}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: inserting the store walk: %w", id, err)
		}
		if err := b0.InsertItemAt(hostIdx, ir.ASTItem{Block: xBlk}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: inserting X into the trip loop: %w", id, err)
		}
		if err := b0.InsertItemAt(hostIdx, ir.ASTItem{Block: initBlk}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: inserting the init walk: %w", id, err)
		}
	} else {
		// Single-between degenerate: E moves into X; the walks take X's
		// old position in its enclosing block.
		xTail := -1
		for i, it := range xBlk.Items() {
			if it.Block == p.eBlk {
				xTail = i
				break
			}
		}
		if xTail < 0 {
			return fmt.Errorf("final-lower: reduce node %d: the element walk is not an item of X (corrupted bracket?)", id)
		}
		xBlk.RemoveItemsIf(func(it ir.ASTItem) bool { return it.Block == p.eBlk })
		p.grandBlk.RemoveItemsIf(func(it ir.ASTItem) bool { return it.Block == xBlk })
		if err := xBlk.InsertItemAt(xTail, ir.ASTItem{Block: p.eBlk}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: nesting the element walk inside X: %w", id, err)
		}
		if err := p.grandBlk.InsertItemAt(p.xIdx, ir.ASTItem{Block: storeBlk}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: inserting the store walk: %w", id, err)
		}
		if err := p.grandBlk.InsertItemAt(p.xIdx, ir.ASTItem{Block: xBlk}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: inserting X at the chunk level: %w", id, err)
		}
		if err := p.grandBlk.InsertItemAt(p.xIdx, ir.ASTItem{Block: initBlk}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: inserting the init walk: %w", id, err)
		}
	}

	// (e) The Combine-value consumers: value onto the acc load at the
	// store-walk IV, coordinates rebuilt onto it, relocated into the store
	// walk after the shared gates.
	moved, epilogue, err := lr.qualifyCombineConsumers(red, dim, ownerLNT, combineCall)
	if err != nil {
		return err
	}
	accLoadS := lr.g.NewFragLoad(lowerOrigin, acc, storeIV)
	for _, m := range moved {
		for _, u := range append([]ir.SSAUse(nil), ir.UsesOf(combineCall)...) {
			if usr := u.User(); usr != nil && usr.BasePtr() == m.BasePtr() {
				if err := lr.mctx.SetDef(u, accLoadS); err != nil {
					return fmt.Errorf("final-lower: reduce node %d: rewiring a consumer onto the acc load: %w", id, err)
				}
			}
		}
		if err := lr.rebuildOperands(m, map[ir.Node]ir.Node{p.eIV: storeIV}); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: rebuilding a consumer onto the store-walk IV: %w", id, err)
		}
	}
	// The store walk must be reachable from the kernel body BEFORE the
	// enclosure gate runs (the gate reads the AST ancestry — a detached
	// block would defeat it, the buildElementLoop ordering note).
	if err := lr.assertConsumerEnclosure(red, moved, storeBlk); err != nil {
		return err
	}
	idx := 0
	epilogue.RemoveItemsIf(func(it ir.ASTItem) bool {
		if it.Node != nil && idx < len(moved) && it.Node.BasePtr() == moved[idx].BasePtr() {
			idx++
			return true
		}
		return false
	})
	for _, m := range moved {
		storeBlk.AppendItem(ir.ASTItem{Node: m})
		m.BasePtr().SetSched(hostDepth+1, storeBlk)
	}

	// (g) The ceremony erasures: the Combine call and the SeqReducer
	// construction (its prologue item first). The orphaned combiner lambda
	// stays in the sea per the maplower precedent — the emitter emits no
	// symbol for it (registerLambda runs for surviving constructions only)
	// — BUT its anchored items die HERE: the scheduler anchored the lambda's
	// params/body as prologue items, which a surviving construction shields
	// from emission via the emitter's markClosure; with the construction
	// erased the shield is gone, so the closure's items are removed
	// explicitly (P-7 spirit: verb-death only, no sea sweep — the orphaned
	// nodes themselves are the tail DCE's domain).
	prologue.RemoveItemsIf(func(it ir.ASTItem) bool {
		return it.Node != nil && it.Node.BasePtr() == construction.BasePtr()
	})
	if err := lr.mctx.Erase(combineCall); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: erasing the Combine call: %w", id, err)
	}
	if err := lr.mctx.Erase(construction); err != nil {
		return fmt.Errorf("final-lower: reduce node %d: erasing the reducer construction: %w", id, err)
	}
	if lambda != nil && !lambda.BasePtr().Erased() {
		orphan := operandClosure(lambda)
		var strip func(blk *ir.ASTBlock)
		strip = func(blk *ir.ASTBlock) {
			blk.RemoveItemsIf(func(it ir.ASTItem) bool {
				return it.Node != nil && orphan[it.Node]
			})
			for _, it := range blk.Items() {
				if it.Block != nil {
					strip(it.Block)
				}
			}
		}
		strip(lr.ast.Body)
	}
	return nil
}
