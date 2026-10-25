// Package finallower implements the FINAL LOWERING pass
// (quick-261022-l1o, doc/optana/final_lower.md as amended — design review
// 261022-g6i verdict GO): the last pass before code generation. It runs
// AFTER scheduling and consumes the attached AST-style CFG plus the tiling
// provenance record preserved through the cache. Three things are lowered:
//
//   - REDUCERS (reducer.go): every live KindReduce becomes a specialized
//     runtime-object construction + the bracket-law Init/Reduce/Combine
//     RuntimeCallMethod statements (LR-1..LR-3, LR-7, LR-13, LR-14) — one
//     reduction spanning ALL tiles of the reduced dim, never per-tile
//     partials (doc/tca.md §5.10).
//   - MAPS (maplower.go): pure identity projectors eliminated by the
//     three-step verb — RAUW onto the body, erase the Map, erase the As
//     (LR-9 as amended by fix2 D-b: NO debris sweeps; the orphaned lambda
//     subtree is the tail DCE's) — ordered AFTER Reducer Lower, whose @r
//     derivation needs the producing Maps alive (LR-2).
//   - RUNTIME OBJECTS (objects.go): exactly one RuntimeKernelInfo as the
//     FIRST item of the kernel Body Group, one GridMapping+ThreadMapping
//     pair prepended to every LNT's prologue Group (LR-11), and every
//     pre-final-lower launch-map query rewritten into an explicit
//     RuntimeGetAttribute on the owning LNT's mapping object (LR-12) —
//     after the rewrite the query kinds vanish from every graph.
//
// Codegen consumes only this pass's output. Launch configuration
// (gridDim/blockDim) is deliberately NOT an IR node (LR-15): it derives
// from the tiling Params through the kernel metadata into the host launch
// wrapper + the runtime's launch validation.
package finallower

import (
	"errors"
	"fmt"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
	"tcacomp/internal/opt/tiling"
)

// init self-registers the pass: Priority 950 — immediately after schedule
// (900), the doc/opt.md pipeline's final slot before codegen. Requires
// "schedule": the pass consumes the attached AST CFG and the per-node
// SchedSN/SchedBlock linkage only the scheduler produces.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "final-lower",
		Priority: 950,
		Requires: []string{"schedule"},
		Factory:  func() opt.Pass { return Pass{} },
	})
}

// lowerOrigin is the PassOrigin stamp every node this pass constructs
// carries from birth (construction is exempt from MutContext but records
// PassOrigin, D-17).
const lowerOrigin = "final-lower"

// Pass is the final lowering pass. Stateless: every Run builds a fresh
// lowering state over the graph's attached AST.
type Pass struct{}

// ID returns the pipeline name "final-lower".
func (Pass) ID() string { return "final-lower" }

// Requires pins schedule: the AST CFG, the SN→block linkage, and the
// preserved tiling provenance are all scheduler-era products.
func (Pass) Requires() []string { return []string{"schedule"} }

// Produces: builder-output stage — same as every current pass. The
// catalog-iteration rule marks every AllKinds() kind Legal at
// StageConstructed; the StageFinal window is deliberately empty/fail-closed
// (legality.go's WindowFor), so StageConstructed is the only usable row and
// the manager's post-run window re-check stays clean.
func (Pass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — no pass runs after final-lower; every cached
// analysis dies here by the invalidate-all default.
func (Pass) Preserved() []string { return nil }

// Run lowers a scheduled kernel to codegen-ready IR:
//
//	gates (nil graph / no AST / provenance fetch fail-closed)
//	  → insertRuntimeObjects (objects.go, LR-11 — FIRST among the lowers:
//	     the R1/R2 reducer constructions consume the per-LNT
//	     RuntimeThreadMapping nodes this step creates)
//	  → lowerReducers (reducer.go, LR-1..LR-3/LR-7/LR-8 + R3 — before Map
//	     Lower: the @r derivation needs the producing Maps alive, LR-2)
//	  → lowerMaps (maplower.go, LR-9)
//	  → rewriteQueries (objects.go, LR-12)
//	  → the erased-item sweep (P-6's RemoveItemsIf over every block)
//	  → ir.Verify
//
// Every refusal is deterministic and named; the graph is left consistent
// (choke-point discipline throughout — MutContext{Pass: "final-lower"}).
func (Pass) Run(ctx *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("final-lower: nil graph")
	}
	ast := g.AST()
	if ast == nil {
		return errors.New("final-lower: no AST CFG attached (schedule must run first)")
	}
	prov, err := fetchProvenance(ctx)
	if err != nil {
		return err
	}

	lr := &lowering{
		g:        g,
		ast:      ast,
		prov:     prov,
		mctx:     &ir.MutContext{Pass: lowerOrigin},
		gridOf:   make(map[*ir.ASTLNT]ir.Node),
		threadOf: make(map[*ir.ASTLNT]ir.Node),
	}

	// Runtime objects (LR-11) — FIRST: the reducer constructions reference
	// the per-LNT ThreadMapping objects (R1/R2), so they must exist before
	// Reducer Lower runs. The query rewrite still runs LAST (after Map
	// Lower disposed the Maps that anchored query-using projections).
	if err := lr.insertRuntimeObjects(); err != nil {
		return err
	}

	// Reducer Lower (before Map Lower: the @r derivation reads Reduce.op0 →
	// producing Map → AsOf, so every Reduce must be lowered before Map
	// Lower rewires those operands away, LR-2).
	if err := lr.lowerReducers(); err != nil {
		return err
	}

	// Map Lower (LR-9).
	if err := lr.lowerMaps(); err != nil {
		return err
	}

	// The dissolved between-loop disposal (R3, fix2 D-a): erase the
	// dissolved loop headers and their OWN header nodes — verb-death only;
	// a surviving AST ITEM still referencing a dissolved header node is a
	// named hard error, sea-only references are the tail DCE's garbage.
	if err := lr.finalizeDissolutions(); err != nil {
		return err
	}

	if err := lr.rewriteQueries(); err != nil {
		return err
	}

	// (Literal canonicalization EXILED — fix3 D-o, R-A/R-B: a lower never
	// optimizes. GVN predates the passes that mint most const atoms, so
	// the pipeline's TAIL now owns a minimal const dedup ahead of its dce
	// instance — internal/pipeline's constDedupPass.)

	// The erased-item sweep: no AST item anywhere may reference an erased
	// node (Map bodies, retired queries, Reduce nodes).
	// The query rewrite replaced its own items in place; everything else
	// lands here.
	lr.sweepErasedItems()

	// The chain-consistency gate: the pass's products must verify clean.
	if vs := ir.Verify(g); len(vs) != 0 {
		return fmt.Errorf("final-lower: verifier reported %d violation(s) after the lowering: %s",
			len(vs), vs[0])
	}
	return nil
}

// fetchProvenance resolves the tiling provenance record fail-closed (P-5):
// a nil context/cache, a MISSING entry, and a WRONG-TYPED entry are all
// named errors — the reducer bracket loops and the per-LNT mapping portions
// are unreadable without it, and guessing is never acceptable.
func fetchProvenance(ctx *opt.Context) (*tiling.Provenance, error) {
	if ctx == nil || ctx.Cache == nil {
		return nil, errors.New("final-lower: no analysis cache (the tiling provenance is fetched through the cache)")
	}
	got := ctx.Cache.Get(tiling.ProvenanceName)
	if got == nil {
		return nil, fmt.Errorf("final-lower: no tiling provenance under %q (did tile-lower run?)", tiling.ProvenanceName)
	}
	prov, ok := got.(*tiling.Provenance)
	if !ok {
		return nil, fmt.Errorf("final-lower: cache entry under %q is %T, want *tiling.Provenance", tiling.ProvenanceName, got)
	}
	return prov, nil
}

// lowering carries one Run's shared state: the graph, its attached AST, the
// provenance, the choke-point context, the block→LNT index (P-11, built
// lazily — objects.go and reducer.go share it), the per-LNT mapping
// objects insertRuntimeObjects created (the query rewrite's resolution
// targets, LR-12), and the deferred between-loop dissolutions (R3 —
// finalized after Map Lower).
type lowering struct {
	g         *ir.Graph
	ast       *ir.ASTKernel
	prov      *tiling.Provenance
	mctx      *ir.MutContext
	byLNT     map[*ir.ASTBlock]*ir.ASTLNT // every block of an LNT's cluster → its LNT
	lntsIx    bool                        // byLNT built
	gridOf    map[*ir.ASTLNT]ir.Node      // per-LNT RuntimeGridMapping
	threadOf  map[*ir.ASTLNT]ir.Node      // per-LNT RuntimeThreadMapping
	dissolved []dissolveJob               // R3: between loops dissolved by the element-loop restructure
	// rebuildMemo is the CURRENT reduce's between-IV rebuild session (reset
	// per lowerReduce): original def → rebuilt clone. One clone per shared
	// def per SESSION — the in-nest walk (the load) and the moved epilogue
	// consumers (the store) read the SAME rebuilt spine, keeping the spine
	// inside the defining loops' operand closures (the loop-scope
	// invariant's containment notion); per-user clones would mint a
	// store-private spine outside every closure.
	rebuildMemo map[ir.Node]ir.Node
}

// dissolveJob is one dissolved between loop awaiting disposal (R3): its
// loop node and the originating Reduce for diagnostics.
type dissolveJob struct {
	loop ir.Node
	red  ir.Node
}

// finalizeDissolutions disposes the dissolved between loops after Map
// Lower. fix2 D-a (R-A: verb-death only): the ONLY erasures here are the
// ones the element-loop restructure's rule prescribes — each dissolved loop
// header and its OWN header nodes (the KeyedAttributes, the BoundedDimVar
// wrapper, the IV). The dead coordinate spines those erasures orphan stay
// in the sea: they are the tail DCE's input, never this pass's business.
// Erase's live-use refusal is the backstop: a header node still holding
// uses is LEFT IN PLACE unless a surviving AST ITEM references it — such an
// item reads a dimvar whose loop no longer exists in the AST (the emitter
// would have no iterator for it), a named hard error. Sea-only references
// are garbage DCE owns (fix2 §4.8 negative pin (a)'s premise: the lower
// never crashes on garbage it no longer makes).
func (lr *lowering) finalizeDissolutions() error {
	for _, job := range lr.dissolved {
		// The loop's own header nodes, captured before the loop itself dies.
		// ops = [iv, bound, attrs...]; erased in REVERSE so each erasure
		// finds no live user among the set itself (the attrs are used only
		// by the loop, the wrapper by the loop, the IV by loop + wrapper).
		// The dim PARAM and the bound-max const are NOT candidates: the
		// fresh element loop shares them and they are never erased here.
		ops := append([]ir.Node(nil), ir.OperandsOf(job.loop)...)
		if err := lr.mctx.Erase(job.loop); err != nil {
			return fmt.Errorf("final-lower: reduce node %d: erasing a dissolved between loop: %w", job.red.BasePtr().ID(), err)
		}
		for i := len(ops) - 1; i >= 0; i-- {
			n := ops[i]
			if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() {
				continue
			}
			if err := lr.mctx.Erase(n); err != nil {
				if item := lr.itemReferencing(n); item != nil {
					return fmt.Errorf("final-lower: reduce node %d: AST item node %d (%s) still references the dissolved between header node %d of loop %d after Map Lower (its value is not defined at that position)",
						job.red.BasePtr().ID(), item.BasePtr().ID(), item.KindOf(), n.BasePtr().ID(), job.loop.BasePtr().ID())
				}
				// Sea-only uses remain: dead coordinate debris the tail DCE
				// owns (fix2 D-a) — leave the node in the sea.
				continue
			}
		}
	}
	lr.dissolved = nil
	return nil
}

// itemReferencing returns the first AST item node whose operand closure
// references n (or that IS n), or nil — the surviving-item probe behind
// finalizeDissolutions' narrowed hard error (fix2 D-a: only an AST ITEM
// referencing a dissolved header is fatal; sea-only references are
// garbage DCE owns).
func (lr *lowering) itemReferencing(n ir.Node) ir.Node {
	var search func(blk *ir.ASTBlock) ir.Node
	search = func(blk *ir.ASTBlock) ir.Node {
		for _, it := range blk.Items() {
			if it.Node != nil {
				if it.Node.BasePtr() == n.BasePtr() || operandClosure(it.Node)[n] {
					return it.Node
				}
				continue
			}
			if it.Block != nil {
				if found := search(it.Block); found != nil {
					return found
				}
			}
		}
		return nil
	}
	if lr.ast.Body == nil {
		return nil
	}
	return search(lr.ast.Body)
}

// blockIndex builds the block→LNT index per P-11: for each LNT, its
// LEXICAL Group (the kernel Body item whose Block.Nest() == lnt) and the
// Group's whole subtree — the embedded LBBN loop blocks included
// (quick-261023-lex: the nest is an item inside the group, not a sibling).
// The kernel Body Group itself and the Prototype belong to NO LNT.
func (lr *lowering) blockIndex() map[*ir.ASTBlock]*ir.ASTLNT {
	if lr.lntsIx {
		return lr.byLNT
	}
	lr.byLNT = make(map[*ir.ASTBlock]*ir.ASTLNT)
	for _, lnt := range lr.ast.LNTs {
		for _, it := range lr.ast.Body.Items() {
			if it.Block == nil || it.Block.Kind() != ir.BlockGroup || it.Block.Nest() != lnt {
				continue
			}
			var stamp func(b *ir.ASTBlock)
			stamp = func(b *ir.ASTBlock) {
				lr.byLNT[b] = lnt
				for _, sub := range b.Items() {
					if sub.Block != nil {
						stamp(sub.Block)
					}
				}
			}
			stamp(it.Block)
		}
	}
	lr.lntsIx = true
	return lr.byLNT
}

// lntOf resolves the owning LNT of a block, fail-closed: a block that
// belongs to NO LNT (the kernel Body Group, the Prototype, or a corrupted
// linkage) cannot bind a query or a reducer and is a hard diagnostic (P-11).
func (lr *lowering) lntOf(blk *ir.ASTBlock, what string) (*ir.ASTLNT, error) {
	if blk == nil {
		return nil, fmt.Errorf("final-lower: %s carries no SchedBlock linkage (unscheduled node?)", what)
	}
	lnt, ok := lr.blockIndex()[blk]
	if !ok || lnt == nil {
		return nil, fmt.Errorf("final-lower: %s resolves to a block outside every LNT cluster (a query or reducer outside every cluster cannot be bound)", what)
	}
	return lnt, nil
}

// findItem locates node's AST item: the containing block and the item's
// index, or (nil, -1) when no block of the kernel holds the node as an
// item. Recursive over the Body's block items (the Prototype never holds
// statements).
func (lr *lowering) findItem(n ir.Node) (*ir.ASTBlock, int) {
	var search func(blk *ir.ASTBlock) (*ir.ASTBlock, int)
	search = func(blk *ir.ASTBlock) (*ir.ASTBlock, int) {
		for i, it := range blk.Items() {
			if it.Node != nil && it.Node.BasePtr() == n.BasePtr() {
				return blk, i
			}
			if it.Block != nil {
				if b, idx := search(it.Block); b != nil {
					return b, idx
				}
			}
		}
		return nil, -1
	}
	return search(lr.ast.Body)
}

// findBlockItem locates blk's PARENT: the block holding blk as a Block item
// , plus that item's index, or (nil, -1). A loop's parental block — the
// bracket insertion point — is found through the loop's own SchedBlock.
func (lr *lowering) findBlockItem(blk *ir.ASTBlock) (*ir.ASTBlock, int) {
	var search func(parent *ir.ASTBlock) (*ir.ASTBlock, int)
	search = func(parent *ir.ASTBlock) (*ir.ASTBlock, int) {
		for i, it := range parent.Items() {
			if it.Block == blk {
				return parent, i
			}
			if it.Block != nil {
				if b, idx := search(it.Block); b != nil {
					return b, idx
				}
			}
		}
		return nil, -1
	}
	if blk == nil || blk == lr.ast.Body {
		return nil, -1
	}
	return search(lr.ast.Body)
}

// scopeDepth returns the SN scope of a block per the TOTAL mapping
// (quick-261022-e2u): kernel Body Group → 0; an LNT prologue Group → 0; a
// Loop block → its nest depth r (1-based, the owning LNT's Loops index+1).
// Inserted nodes are stamped SetSched(scopeDepth, block) so SchedSN/
// SchedBlock stay complementary for codegen and diagnostics (P-10).
func (lr *lowering) scopeDepth(blk *ir.ASTBlock) int {
	if blk == nil {
		return 0
	}
	if blk.Kind() == ir.BlockGroup {
		return 0
	}
	lnt, ok := lr.blockIndex()[blk]
	if !ok || lnt == nil || blk.Loop() == nil {
		return 0
	}
	for i, meta := range lnt.Loops {
		if meta.Loop != nil && meta.Loop.BasePtr() == blk.Loop().BasePtr() {
			return i + 1
		}
	}
	return 0
}

// sweepErasedItems removes every AST item whose Node is nil or tombstoned
// (P-6's RemoveItemsIf): the eliminated Maps and their As/lambda debris
// were scheduled as ordinary statement items, and their items must not
// linger once the nodes die. The behavioral pin ("no AST item anywhere
// references an erased node") is the POSTcondition — the query rewrite
// performs its own in-place item replacement, everything else lands here.
func (lr *lowering) sweepErasedItems() {
	var sweep func(blk *ir.ASTBlock)
	sweep = func(blk *ir.ASTBlock) {
		blk.RemoveItemsIf(func(it ir.ASTItem) bool {
			return it.Node != nil && it.Node.BasePtr().Erased()
		})
		for _, it := range blk.Items() {
			if it.Block != nil {
				sweep(it.Block)
			}
		}
	}
	sweep(lr.ast.Body)
}
