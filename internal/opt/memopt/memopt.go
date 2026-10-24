// Package memopt implements the pipeline's memory-optimization pass
// (quick-261020-ejf, doc/optana/memory-opt.md as corrected by the algorithm
// review 261020-ejf-REVIEW.md): store-to-load forwarding, store-store
// elimination, and dead-store elimination over the per-tensor GMEM effect
// chains — plus its dedicated fixpoint verifier (verify.go).
//
// The review's verdict is SOUND under the memory model's preconditions
// (restrict-alias rule, chained reads, single-Return observability), with
// four binding conditions this pass implements exactly:
//
//   - C1 chain totality: a region is optimized only when its whole access
//     set forms ONE linear ImpactBy chain. A region that fails (e.g. an
//     un-chained NewMemStore accession) is legal-but-unoptimized — SKIPPED
//     silently, never an error, never touched.
//   - C2 corrected rules: (a) a store dies iff it has ZERO live SSA uses —
//     data uses first RAUW'd to its value operand (a store returns
//     input#0), swept to a fixpoint; (b) store-store elimination needs the
//     same PROVEN access path (tensor node identity + pairwise ValueEqual
//     axes) and the walk stops at the first load; (c) regions key by the
//     tensor operand NODE (what Env does — never a kind, never a name).
//   - C3 chain splicing: re-pointing a successor's edge at the removed
//     node's predecessor is a plain SetDef; dropping the edge of a removed
//     chain head rides the new MutContext.RemoveImpactBy choke point.
//   - C4 scope: GMEM HIR only (KindMemLoad/KindMemStore; the LIR SMEM/FRAG
//     families carry no effect chains and are out of scope by
//     construction), Level-1 sea only, slotted after gvn (identity-keyed
//     axes make ValueEqual maximally effective post-GVN), erase/rewire
//     only — no new nodes, no new kinds.
//
// quick-261024-sdk (fix1.md D-f) adds ONE LIR-scoped rule beside the
// region rules — FILL LIVENESS: an unconsumed buffer store (a
// KindFragStore/KindMemStoreSMem whose buffer has no live buffer load)
// is dead code removed together with its unread source GMEM load. The
// author's correction is explicit: this is memory optimization's domain,
// never DCE's — the fill's source load is chain-reachable (so DCE keeps
// it) but produces a value nobody reads, which is exactly the
// ordering-only dead-load class the 2026-10-20 audit named as memopt's
// open gap.
//
// Expectation note (review): today's builder graphs rarely fire the rules —
// tensor reads are paramRefs, not GMEM loads — so the pass's own tests
// construct chain-carrying graphs directly through the ir constructors. The
// pass becomes load-bearing for loop-fusion intermediates and the
// frontend's local-tensor model (the doc's stated enabler).
package memopt

import (
	"errors"
	"fmt"

	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the pass in the global registry: Priority 112 —
// memory optimization after the CSE window (gvn/gvn-verify at 110/111), so
// the proven-path comparison runs on identity-keyed axes (C4). The registry
// scale in internal/opt/registry.go documents the anchors.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "memopt",
		Priority: 112,
		Factory:  func() opt.Pass { return Pass{} },
	})
}

// Pass is the Level-1 (sea-of-nodes) memory optimizer. It is stateless:
// every Run re-derives the eligible regions from the graph, so re-runs are
// idempotent (INV-F).
type Pass struct{}

// ID returns the pipeline name "memopt".
func (Pass) ID() string { return "memopt" }

// Requires: gvn must have run earlier — post-GVN, equal axes are the SAME
// SSA node, so the proven-path check (pairwise ValueEqual) is maximally
// effective (C4: the doc's "runs after GVN").
func (Pass) Requires() []string { return []string{"gvn"} }

// Produces: builder-output stage — the pass erases and rewires only, it
// introduces no new kinds, so the constructed legality window re-checks
// the graph.
func (Pass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — every analysis is invalidated.
func (Pass) Preserved() []string { return nil }

// Run optimizes the GMEM effect chains over the Level-1 sea:
//
//   - LEVEL GATE first (the dce/gvn refusal pattern): a nil graph and a
//     materialized/CFG-carrying graph are deterministic errors evaluated
//     before any mutation, so a refused graph is untouched.
//
//   - DERIVE the eligible regions (C1, deriveRegions below — the shared
//     helper verify.go re-runs for its independent re-derivation): group
//     the live GMEM accesses by tensor operand node identity, keep only
//     the regions whose accesses form ONE linear ImpactBy chain.
//
//   - FIXPOINT (progress-flag rounds bounded by the live-access count + 1,
//     the dce sweep pattern; every mutation erases exactly one node, so
//     the bound is belt-and-braces) over the eligible regions, all
//     mutations through the one MutContext{Pass: "memopt"} (INV-E):
//     forward loads (rule a), eliminate shadowed stores (rule b), sweep
//     zero-use stores (rule c), and the LIR-scoped fill-liveness sweep
//     (rule d, D-f) over every live buffer store. Determinism (INV-G):
//     regions in first-access allocation order, accesses in chain order —
//     maps group, they never steer iteration.
//
// The side-effect analysis is computed once on entry: only stores and
// loads are ever erased, never the pure axes nodes the proven-path
// comparison consults, so the verdicts the rules rely on stay valid
// through the fixpoint (unknown nodes would read effectful and simply
// block a rule — fail-closed, never unsound).
func (Pass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("memopt: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("memopt: graph already materialized (memopt runs on the Level-1 sea)")
	}
	se := sideeffect.Analyze(g)
	mctx := &ir.MutContext{Pass: "memopt"}

	total := 0
	for _, n := range g.AllocatedNodes() {
		if isGMEMAccess(n.KindOf()) || isBufferStore(n.KindOf()) {
			total++
		}
	}
	for round := 0; round <= total+1; round++ {
		progress := false
		for _, r := range deriveRegions(g) {
			// (a) FORWARD LOADS: a load whose first-reached store behind it
			// holds the same proven path is erased and its users see the
			// stored value.
			for _, a := range r.accesses {
				if a.KindOf() != ir.KindMemLoad {
					continue
				}
				fired, err := forwardLoad(mctx, se, a)
				if err != nil {
					return err
				}
				progress = progress || fired
			}
			// (b) STORE-STORE ELIMINATION: an earlier store shadowed by a
			// later same-path store (only stores between them) is erased.
			for _, a := range r.accesses {
				if a.KindOf() != ir.KindMemStore || a.BasePtr().Erased() {
					continue
				}
				fired, err := eliminateShadowedStores(mctx, se, a)
				if err != nil {
					return err
				}
				progress = progress || fired
			}
			// (c) DEAD-STORE SWEEP: a store with no effect-user successor
			// is unobservable once its data uses see its value operand.
			for _, a := range r.accesses {
				if a.KindOf() != ir.KindMemStore || a.BasePtr().Erased() {
					continue
				}
				fired, err := sweepDeadStore(mctx, r, g.End(), a)
				if err != nil {
					return err
				}
				progress = progress || fired
			}
		}
		// (d) FILL LIVENESS (D-f, LIR-scoped — outside the regions): a
		// buffer store whose buffer has no live reads is unconsumed dead
		// code, removed together with its unread source load.
		fired, err := sweepUnconsumedBufferStores(mctx, g)
		if err != nil {
			return err
		}
		progress = progress || fired
		if !progress {
			return nil
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Region derivation (C1) — shared with verify.go.
// ---------------------------------------------------------------------------

// region is one tensor's eligible GMEM access chain (C1): the tensor
// operand node the accesses are keyed by (C2c), and the accesses in CHAIN
// ORDER — head first, tail last.
type region struct {
	tensor   *ir.Base
	accesses []ir.Node
}

// deriveRegions groups g's live GMEM accesses by their tensor operand NODE
// identity (load operand 0, store operand 1 — exactly the Env's keying,
// never a kind and never a name, C2c) and keeps only the regions whose
// accesses form ONE linear ImpactBy chain (C1). Regions return in
// first-access allocation order (INV-G); a region that fails linearity is
// legal-but-unoptimized construction (effect.go documents un-chained
// accession) and is SKIPPED — no error, no mutation.
func deriveRegions(g *ir.Graph) []region {
	var order []*ir.Base
	groups := make(map[*ir.Base][]ir.Node)
	for _, n := range g.AllocatedNodes() {
		if !isGMEMAccess(n.KindOf()) {
			continue
		}
		t := tensorOf(n)
		if t == nil {
			continue
		}
		key := t.BasePtr()
		if _, seen := groups[key]; !seen {
			order = append(order, key)
		}
		groups[key] = append(groups[key], n)
	}

	var out []region
	for _, key := range order {
		chain, ok := linearChain(g, groups[key])
		if !ok {
			continue // C1: skip, never error
		}
		out = append(out, region{tensor: key, accesses: chain})
	}
	return out
}

// linearChain verifies that accesses forms ONE linear ImpactBy chain and
// returns it in chain order (head first). Linearity (C1):
//   - every access holds at most one ImpactBy edge, whose def is itself in
//     the set (chains never leave the region);
//   - exactly one access has no predecessor (the head);
//   - every access has at most one effect-user successor in the set ∪
//     {End} — an interior node the Return also observes (a post-wiring
//     chain append) carries two orderings and voids the region;
//   - walking successors from the head covers every access exactly once
//     (a single path, no forks, no detached second chain).
//
// The maps are lookup structures only; the returned order comes from the
// deterministic head→tail walk.
func linearChain(g *ir.Graph, accesses []ir.Node) ([]ir.Node, bool) {
	inSet := make(map[*ir.Base]bool, len(accesses))
	for _, a := range accesses {
		inSet[a.BasePtr()] = true
	}

	var head ir.Node
	heads := 0
	for _, a := range accesses {
		imp := ir.ImpactBy(a)
		if len(imp) > 1 {
			return nil, false // forked ordering constraints
		}
		if len(imp) == 0 {
			head = a
			heads++
		} else if d := imp[0].Def(); d == nil || !inSet[d.BasePtr()] {
			return nil, false // chains outside the region: not total
		}
	}
	if heads != 1 {
		return nil, false // no unique head: several chains or a cycle
	}

	// successor view: access -> the one access (or the Return) whose
	// ImpactBy points at it. A second claimant on either side voids the
	// region.
	succ := make(map[*ir.Base]ir.Node, len(accesses))
	for _, a := range accesses {
		for _, e := range ir.ImpactBy(a) {
			d := e.Def()
			if d == nil {
				continue
			}
			if _, taken := succ[d.BasePtr()]; taken {
				return nil, false
			}
			succ[d.BasePtr()] = a
		}
	}
	end := g.End()
	for _, e := range ir.ImpactBy(end) {
		d := e.Def()
		if d == nil || !inSet[d.BasePtr()] {
			continue // another region's tail, or a stale edge
		}
		if _, taken := succ[d.BasePtr()]; taken {
			return nil, false // interior node also observed by the Return
		}
		succ[d.BasePtr()] = end
	}

	// Single path: walk from the head, at most one successor per step.
	chain := make([]ir.Node, 0, len(accesses))
	cur := head
	for i := 0; i <= len(accesses); i++ {
		chain = append(chain, cur)
		next, ok := succ[cur.BasePtr()]
		if !ok || next == end {
			break // the tail (observed by the Return or unwired)
		}
		cur = next
	}
	if len(chain) != len(accesses) {
		return nil, false // disconnected or cyclic residue
	}
	return chain, true
}

// ---------------------------------------------------------------------------
// The three rules (C2a/C2b). All mutations ride the caller's one MutContext.
// ---------------------------------------------------------------------------

// forwardLoad implements rule (a): load l whose FIRST-reached store behind
// it (walking ImpactBy backwards, passing through loads — the most-recent-
// store discipline) holds the same proven path is forwarded: its users are
// RAUW'd onto the store's value operand (Return data operands included),
// its successor edges are spliced to its own predecessor (SetDef) or
// dropped when l was chain-first (RemoveImpactBy — C3), and l is erased.
func forwardLoad(mctx *ir.MutContext, se sideeffect.SideEffects, l ir.Node) (bool, error) {
	b := firstStoreBehind(l)
	if b == nil || !sameProvenPath(se, l, b) {
		return false, nil
	}
	value := storeValue(b)
	if value == nil {
		return false, fmt.Errorf("memopt: store node %d (%s) carries no value operand",
			b.BasePtr().ID(), b.KindOf())
	}
	// Capture the ordering context BEFORE the RAUW moves records: l's own
	// predecessor (nil = chain head) and the successor edges that must be
	// spliced past l (at most one in an eligible region: the next access's
	// ImpactBy record, or the Return's when l is the region tail). Edge
	// identity is RECORD identity: comparing SSAUse handles (== compares
	// the shared ref record) keeps the Return's DATA use of the same load
	// out of the splice set — only the ordering edge moves.
	var pred ir.Node
	if edges := ir.ImpactBy(l); len(edges) == 1 {
		pred = edges[0].Def()
	}
	var succEdges []ir.SSAUse
	for _, u := range ir.UsesOf(l) {
		user := u.User()
		if user == nil {
			continue
		}
		for _, e := range ir.ImpactBy(user) {
			if e == u { // this use record IS one of user's ImpactBy edges
				succEdges = append(succEdges, u)
			}
		}
	}
	if len(succEdges) > 1 {
		return false, fmt.Errorf("memopt: load node %d (%s) has %d effect successors in an eligible region (at most one)",
			l.BasePtr().ID(), l.KindOf(), len(succEdges))
	}
	if err := mctx.ReplaceAllUsesWith(l, value); err != nil {
		return false, fmt.Errorf("memopt: forward load %d: %w", l.BasePtr().ID(), err)
	}
	for _, e := range succEdges {
		if pred != nil {
			if err := mctx.SetDef(e, pred); err != nil {
				return false, fmt.Errorf("memopt: splice successor of load %d: %w", l.BasePtr().ID(), err)
			}
		} else if err := mctx.RemoveImpactBy(e); err != nil {
			return false, fmt.Errorf("memopt: drop head edge of load %d: %w", l.BasePtr().ID(), err)
		}
	}
	if err := mctx.Erase(l); err != nil {
		return false, fmt.Errorf("memopt: erase forwarded load %d: %w", l.BasePtr().ID(), err)
	}
	return true, nil
}

// eliminateShadowedStores implements rule (b): walking store s's ImpactBy
// backwards, the walk STOPS at the first load (it may read that position)
// and at each earlier store s0 reached fires only when the two hold the
// SAME PROVEN access path (C2b): s0's data uses are RAUW'd onto its value
// operand, the just-traversed edge is spliced to s0's predecessor (SetDef,
// or RemoveImpactBy when s0 was chain-first — C3), s0 is erased, and the
// walk continues from s0's predecessor so same-position cascades die in
// one walk. An unequal-position store is walked PAST — sound under the
// restrict-alias rule: any later load whose coordinate aliases s0's must
// also alias s's, and s is later (the review's argument 2).
func eliminateShadowedStores(mctx *ir.MutContext, se sideeffect.SideEffects, origin ir.Node) (bool, error) {
	progress := false
	cur := origin // the walk cursor; origin stays the comparison anchor
	for {
		edges := ir.ImpactBy(cur)
		if len(edges) != 1 {
			return progress, nil // chain head reached, or a forked region
		}
		s0 := edges[0].Def()
		if s0 == nil || s0.BasePtr().Erased() {
			return progress, nil
		}
		if s0.KindOf() == ir.KindMemLoad {
			return progress, nil // a read may observe the position
		}
		if !sameProvenPath(se, origin, s0) {
			cur = s0 // walk PAST the unequal-position store
			continue
		}
		var pred ir.Node
		if p := ir.ImpactBy(s0); len(p) == 1 {
			pred = p[0].Def()
		}
		value := storeValue(s0)
		if value == nil {
			return progress, fmt.Errorf("memopt: store node %d (%s) carries no value operand",
				s0.BasePtr().ID(), s0.KindOf())
		}
		if err := mctx.ReplaceAllUsesWith(s0, value); err != nil {
			return progress, fmt.Errorf("memopt: eliminate shadowed store %d: %w", s0.BasePtr().ID(), err)
		}
		if pred != nil {
			if err := mctx.SetDef(edges[0], pred); err != nil {
				return progress, fmt.Errorf("memopt: splice successor of store %d: %w", s0.BasePtr().ID(), err)
			}
		} else if err := mctx.RemoveImpactBy(edges[0]); err != nil {
			return progress, fmt.Errorf("memopt: drop head edge of store %d: %w", s0.BasePtr().ID(), err)
		}
		if err := mctx.Erase(s0); err != nil {
			return progress, fmt.Errorf("memopt: erase shadowed store %d: %w", s0.BasePtr().ID(), err)
		}
		progress = true
		// s's ImpactBy edge now resolves at s0's predecessor: the loop
		// re-reads it and the cascade continues.
	}
}

// sweepDeadStore implements rule (c) — the corrected dead-store rule
// (C2a): a store with NO effect-user successor (no payload use from the
// region's accesses and none from the Return) is unobservable, so its data
// uses are RAUW'd onto its value operand (a store returns input#0 — the
// rewire is value-preserving) and it is erased. A Return-wired store
// always keeps its payload use and is NEVER erased (INV-D). The outer
// fixpoint sweeps tail cascades (S1→S2 both unread) to completion.
func sweepDeadStore(mctx *ir.MutContext, r region, end ir.Node, s ir.Node) (bool, error) {
	if hasEffectSuccessor(r, end, s) {
		return false, nil
	}
	value := storeValue(s)
	if value == nil {
		return false, fmt.Errorf("memopt: store node %d (%s) carries no value operand",
			s.BasePtr().ID(), s.KindOf())
	}
	if err := mctx.ReplaceAllUsesWith(s, value); err != nil {
		return false, fmt.Errorf("memopt: sweep dead store %d: %w", s.BasePtr().ID(), err)
	}
	if err := mctx.Erase(s); err != nil {
		return false, fmt.Errorf("memopt: erase dead store %d: %w", s.BasePtr().ID(), err)
	}
	return true, nil
}

// ---------------------------------------------------------------------------
// Rule (d) — fill liveness (D-f, quick-261024-sdk). LIR-scoped, outside the
// regions: the fill families carry no per-tensor GMEM chains of their own
// (the fill store holds no ImpactBy; its readers chain TO it), so the rule
// keys on buffer-operand identity and the live-readership predicate alone.
// ---------------------------------------------------------------------------

// sweepUnconsumedBufferStores implements rule (d) — fill liveness (D-f,
// fix1.md): "a buffer store whose buffer has zero live loads is dead
// together with its source load". An LIR buffer store (KindFragStore /
// KindMemStoreSMem — tile-lower's fill spill and its SMEM-window sibling)
// whose buffer operand has NO live buffer load is unconsumed: the store is
// erased and, when the stored value is a GMEM load nobody reads, the load
// dies with it (ordering-only dead-load removal — its chain successors
// splice past it, C3). DCE stays untouched; this is memory-optimization
// domain per the author correction (§0: "a fill is a normal memory load —
// it must have a memory read consuming it, otherwise it is dead code").
func sweepUnconsumedBufferStores(mctx *ir.MutContext, g *ir.Graph) (bool, error) {
	progress := false
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || !isBufferStore(n.KindOf()) {
			continue
		}
		ops := ir.OperandsOf(n)
		if len(ops) < 2 || ops[1] == nil {
			continue
		}
		if bufferHasLiveReads(g, ops[1]) {
			continue
		}
		fired, err := removeUnconsumedFill(mctx, n)
		if err != nil {
			return progress, err
		}
		progress = progress || fired
	}
	return progress, nil
}

// bufferHasLiveReads reports whether any live buffer load reads buffer —
// the liveness predicate of rule (d) ("zero live loads", D-f). The load
// families are the only readers a buffer can have: a buffer name is
// consumed through its loads, never through a raw operand (construction
// data edges terminate at the store).
func bufferHasLiveReads(g *ir.Graph, buffer ir.Node) bool {
	bb := buffer.BasePtr()
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || !isBufferLoad(n.KindOf()) {
			continue
		}
		if ops := ir.OperandsOf(n); len(ops) > 0 && ops[0] != nil && ops[0].BasePtr() == bb {
			return true
		}
	}
	return false
}

// removeUnconsumedFill erases the buffer store s and, when its value is an
// unread GMEM load, the load with it (D-f: "dead together with its source
// load"). The store is value-RAUW'd first (the store families return
// input#0 — the sweepDeadStore discipline) and erased; the source load —
// its data uses gone with the store — splices its chain successors past
// itself and erases. A load that still has a data use is left alone
// (fail-closed: not this rule's case).
func removeUnconsumedFill(mctx *ir.MutContext, s ir.Node) (bool, error) {
	value := storeValue(s)
	if value == nil {
		return false, fmt.Errorf("memopt: buffer store node %d (%s) carries no value operand",
			s.BasePtr().ID(), s.KindOf())
	}
	if err := mctx.ReplaceAllUsesWith(s, value); err != nil {
		return false, fmt.Errorf("memopt: RAUW unconsumed buffer store %d: %w", s.BasePtr().ID(), err)
	}
	if err := mctx.Erase(s); err != nil {
		return false, fmt.Errorf("memopt: erase unconsumed buffer store %d: %w", s.BasePtr().ID(), err)
	}
	if value.KindOf() == ir.KindMemLoad && !hasDataUse(value) {
		if err := splicePast(mctx, value); err != nil {
			return true, err
		}
	}
	return true, nil
}

// splicePast removes an ordering-only node n (no data uses — the caller
// checks): its at-most-one effect successor re-points at n's own
// predecessor (SetDef), or drops the edge when n was chain-first
// (RemoveImpactBy) — the C3 splice — and n is erased. A forked ordering
// (several effect successors or several predecessors) fails closed, the
// same discipline forwardLoad applies to region loads: the splice would
// not know which single order to preserve.
func splicePast(mctx *ir.MutContext, n ir.Node) error {
	var pred ir.Node
	if edges := ir.ImpactBy(n); len(edges) == 1 {
		pred = edges[0].Def()
	} else if len(edges) > 1 {
		return fmt.Errorf("memopt: node %d (%s) holds %d ImpactBy edges (an orderable chain is linear)",
			n.BasePtr().ID(), n.KindOf(), len(edges))
	}
	var succEdges []ir.SSAUse
	for _, u := range ir.UsesOf(n) { // snapshot before any SetDef mutates the list
		user := u.User()
		if user == nil {
			continue
		}
		for _, e := range ir.ImpactBy(user) {
			if e == u { // this use record IS one of user's ImpactBy edges
				succEdges = append(succEdges, u)
				break
			}
		}
	}
	if len(succEdges) > 1 {
		return fmt.Errorf("memopt: node %d (%s) has %d effect successors (at most one in an orderable chain)",
			n.BasePtr().ID(), n.KindOf(), len(succEdges))
	}
	for _, e := range succEdges {
		if pred != nil {
			if err := mctx.SetDef(e, pred); err != nil {
				return fmt.Errorf("memopt: splice successor of node %d: %w", n.BasePtr().ID(), err)
			}
		} else if err := mctx.RemoveImpactBy(e); err != nil {
			return fmt.Errorf("memopt: drop head edge of node %d: %w", n.BasePtr().ID(), err)
		}
	}
	return mctx.Erase(n)
}

// hasDataUse reports whether any live use of n is a DATA use (an operand
// slot, not an ImpactBy ordering edge) — the fill source load's liveness
// test: ordering edges are spliced away by the removal, a data use means
// the value is still read.
func hasDataUse(n ir.Node) bool {
	for _, u := range ir.UsesOf(n) {
		user := u.User()
		if user == nil {
			continue
		}
		effect := false
		for _, e := range ir.ImpactBy(user) {
			if e == u {
				effect = true
				break
			}
		}
		if !effect {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Shared access predicates.
// ---------------------------------------------------------------------------

// isGMEMAccess reports whether k is a HIR GMEM access (C4: the LIR SMEM/
// FRAG families carry no effect chains and never reach the region rules;
// rule (d) reaches the buffer-store pair directly, D-f).
func isGMEMAccess(k ir.Kind) bool { return k == ir.KindMemLoad || k == ir.KindMemStore }

// isBufferStore reports whether k is an LIR buffer store of the fill
// families — rule (d)'s subject (D-f).
func isBufferStore(k ir.Kind) bool { return k == ir.KindFragStore || k == ir.KindMemStoreSMem }

// isBufferLoad reports whether k is an LIR buffer load of the fill
// families — the readership side of rule (d)'s liveness predicate.
func isBufferLoad(k ir.Kind) bool { return k == ir.KindFragLoad || k == ir.KindMemLoadSMem }

// tensorOf returns n's tensor operand node — operand 0 of a load, operand 1
// of a store (the region key, C2c) — or nil.
func tensorOf(n ir.Node) ir.Node {
	ops := ir.OperandsOf(n)
	switch n.KindOf() {
	case ir.KindMemLoad:
		if len(ops) > 0 {
			return ops[0]
		}
	case ir.KindMemStore:
		if len(ops) > 1 {
			return ops[1]
		}
	}
	return nil
}

// axesOf returns n's subscript axes in order — the operands past the
// tensor (loads: [tensor, axes...]; stores: [value, tensor, axes...]).
func axesOf(n ir.Node) []ir.Node {
	ops := ir.OperandsOf(n)
	switch n.KindOf() {
	case ir.KindMemLoad:
		if len(ops) > 1 {
			return ops[1:]
		}
	case ir.KindMemStore:
		if len(ops) > 2 {
			return ops[2:]
		}
	}
	return nil
}

// storeValue returns the stored value — a store's operand 0 (the store
// returns input#0, doc §2.5; the LIR buffer store families share the
// [value, dst, bivs...] layout) — or nil.
func storeValue(n ir.Node) ir.Node {
	switch n.KindOf() {
	case ir.KindMemStore, ir.KindFragStore, ir.KindMemStoreSMem:
		if ops := ir.OperandsOf(n); len(ops) > 0 {
			return ops[0]
		}
	}
	return nil
}

// sameProvenPath reports whether two GMEM accesses are PROVEN to name the
// same runtime coordinate (C2b/C2c): same tensor operand node identity,
// same axes arity, pairwise ValueEqual axes under the accumulated
// side-effect analysis. The comparison is identity-keyed, never a value
// computation: two structurally-unequal subscripts may still alias at
// runtime, so an unproven pair simply blocks the rule (fail-closed) —
// which is exactly what makes the rules sound under unknown aliasing.
func sameProvenPath(se sideeffect.SideEffects, a, b ir.Node) bool {
	ta, tb := tensorOf(a), tensorOf(b)
	if ta == nil || tb == nil || ta.BasePtr() != tb.BasePtr() {
		return false
	}
	aa, ab := axesOf(a), axesOf(b)
	if len(aa) != len(ab) {
		return false
	}
	for i := range aa {
		va, okA := aa[i].(ir.Value)
		vb, okB := ab[i].(ir.Value)
		if !okA || !okB || !va.ValueEqual(vb, se) {
			return false
		}
	}
	return true
}

// firstStoreBehind walks n's ImpactBy chain backwards, passing through
// loads (reads never write), and returns the first store reached — the
// most-recent-store discipline rule (a) forwards against. Nil when the
// walk falls off the chain head without meeting a store.
func firstStoreBehind(n ir.Node) ir.Node {
	for {
		edges := ir.ImpactBy(n)
		if len(edges) != 1 {
			return nil
		}
		d := edges[0].Def()
		if d == nil || d.BasePtr().Erased() {
			return nil
		}
		switch d.KindOf() {
		case ir.KindMemStore:
			return d
		case ir.KindMemLoad:
			n = d // pass through the read
		default:
			return nil
		}
	}
}

// hasEffectSuccessor reports whether any live access of r or the Return
// holds s in its ImpactBy payload — the successor test of the dead-store
// rule (c). Snapshotted members already erased contribute nothing: Erase
// killed their records, and every live edge was spliced past them.
func hasEffectSuccessor(r region, end ir.Node, s ir.Node) bool {
	for _, e := range ir.ImpactBy(end) {
		if d := e.Def(); d != nil && d.BasePtr() == s.BasePtr() {
			return true // Return-wired: observable by construction (INV-D)
		}
	}
	for _, a := range r.accesses {
		if a == s || a.BasePtr().Erased() {
			continue
		}
		for _, e := range ir.ImpactBy(a) {
			if d := e.Def(); d != nil && d.BasePtr() == s.BasePtr() {
				return true
			}
		}
	}
	return false
}

// holdsImpactEdgeAt reports whether user holds def among its ImpactBy
// effect inputs — the successor-edge detector the splice paths use to find
// the records a removal must re-point (or drop).
func holdsImpactEdgeAt(user ir.Node, def ir.Node) bool {
	for _, e := range ir.ImpactBy(user) {
		if d := e.Def(); d != nil && d.BasePtr() == def.BasePtr() {
			return true
		}
	}
	return false
}
