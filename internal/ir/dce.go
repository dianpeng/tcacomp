package ir

// This file is the building-phase dead-code elimination the Phase-1 review
// mandates (P1R-08): the builder may emit chains whose intermediate values
// are never consumed (an abandoned Map before a Store, a Reduce whose result
// is overwritten), and wiping them belongs to construction, not to the pass
// pipeline — passes never see building debris.
//
// The algorithm is LEVEL-AWARE (two-level IR, quick-261017-vdj):
//
//   - Level 1 (before Materialize): a SEA MARK-SWEEP. MARK roots at the
//     graph's params plus the end node (R2 — the graph's single KindReturn;
//     its operand and ImpactBy paths reach every live store) and follows
//     operand slots, payload uses (store targets, iteration dims, lambda
//     params, ImpactBy effect inputs), and lambda bodies — exactly Walk's
//     child relation. SWEEP the Graph.nodes allocation registry: a node with
//     zero live uses and not marked reachable is erased through
//     MutContext.Erase in allocation-order passes (a pass that erases nothing
//     ends the loop); killing a dead user frees its def's next pass. The end
//     node is a mark seed, so the stores its paths reach are never erased by
//     construction; a store NO end path reaches is unreachable debris and is
//     wiped like any other dead node.
//   - Level 2 (after Materialize): the review's original three-step
//     statement model below (mark live statements, backward fixpoint, wipe),
//     kept unchanged for passes that run after scheduling.
//
// Both levels share the conservative Erase discipline: a node with live uses
// is skipped, never force-erased; cyclic dead clusters where every member
// keeps one live-looking use survive untouched (no corruption, residual
// deadness). Builder-produced graphs share only statement ROOTS across
// statements, so the wipe is total for them.

// BuildingDCE runs building-phase dead-code elimination over g, wiping dead
// nodes in place. It runs BEFORE the pass pipeline (pkg.Builder calls it at
// Build time, before Materialize), so passes never observe construction
// debris. The algorithm is level-aware (see file header): a sea mark-sweep at
// Level 1, the review's statement fixpoint at Level 2. A nil graph is a
// no-op; the returned error reports only genuine mutation failures (Erase on
// a live-used node is skipped, never an error).
func BuildingDCE(g *Graph) error {
	if g == nil {
		return nil
	}
	if g.materialized || g.cfg != nil {
		dead := markDeadStatements(g)
		if len(dead) == 0 {
			return nil
		}
		ctx := &MutContext{Pass: "building-dce"}
		return wipeDeadStatements(g, dead, ctx)
	}
	return buildingDCESea(g)
}

// buildingDCESea is the Level-1 sweep: mark everything reachable from the
// params and the end node (Walk's child relation — operand slots, payload
// uses including ImpactBy effect inputs, lambda bodies), then erase every
// unmarked node that no live use keeps alive, in allocation-order passes:
// erasing a dead user kills its operand/payload records, freeing its defs for
// the next pass. A pass that erases nothing ends the loop.
func buildingDCESea(g *Graph) error {
	live := markReachable(g)
	ctx := &MutContext{Pass: "building-dce"}
	erased := make(map[Node]bool)
	for pass := 0; pass <= len(g.nodes); pass++ {
		progress := false
		for _, n := range g.nodes {
			if n == nil || erased[n] || n.BasePtr().erased || live[n] {
				continue
			}
			if len(UsesOf(n)) > 0 {
				continue // still referenced: a live user, or one not yet erased
			}
			if err := ctx.Erase(n); err != nil {
				return err
			}
			erased[n] = true
			progress = true
		}
		if !progress {
			break
		}
	}
	return nil
}

// markReachable returns the set of nodes reachable from the Level-1 roots —
// the graph's params plus its end node (R2) — following exactly the child
// relation Walk uses (operand slots, payload uses, lambda bodies). These
// nodes are untouchable for the sea sweep: they are live by root status or
// feed a live node. All params are seeds (not only the RoleTensor sea
// roots): params are graph-owned signature nodes whose ref lists the
// def-use checker audits, so the sweep must never erase them.
func markReachable(g *Graph) map[Node]bool {
	live := make(map[Node]bool)
	var stack []Node
	push := func(n Node) {
		if n == nil || live[n] {
			return
		}
		live[n] = true
		stack = append(stack, n)
	}
	push(g.end)
	for i := len(g.params) - 1; i >= 0; i-- {
		push(g.params[i])
	}
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if impl := asImpl(n); impl != nil {
			impl.forEachOperand(func(u SSAUse) {
				if u.rec != nil && !u.rec.killed && u.rec.def != nil {
					push(u.rec.def.node)
				}
			})
		}
		if pb, ok := n.(payloadBearer); ok {
			pb.forEachPayloadUse(func(u SSAUse) {
				if u.rec != nil && !u.rec.killed && u.rec.def != nil {
					push(u.rec.def.node)
				}
			})
		}
		if lb, ok := n.(lambdaBearer); ok {
			lb.forEachLambda(func(lam *Lambda) { push(lam.Body) })
		}
	}
	return live
}

// isEffectKind reports whether kind k carries an observable side effect in
// v0. The memory-store family (GMEM/SMEM/FRAG) and kernel returns are the
// only effects; everything else — including loads and loop nests without an
// effect in their body (caught by hasEffectNode's reachability walk) — is
// dead-code-eligible when unreferenced.
func isEffectKind(k Kind) bool {
	switch k {
	case KindMemStore, KindMemStoreSMem, KindFragStore, KindReturn:
		return true
	}
	return false
}

// hasEffectNode reports whether any node reachable from n (operand and
// payload edges, lambda bodies) is side-effecting.
func hasEffectNode(n Node) bool {
	seen := make(map[Node]bool)
	var stack []Node
	stack = append(stack, n)
	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if cur == nil || seen[cur] {
			continue
		}
		seen[cur] = true
		if isEffectKind(cur.KindOf()) {
			return true
		}
		if impl := asImpl(cur); impl != nil {
			impl.forEachOperand(func(u SSAUse) {
				if u.rec != nil && u.rec.def != nil {
					stack = append(stack, u.rec.def.node)
				}
			})
		}
		if pb, ok := cur.(payloadBearer); ok {
			pb.forEachPayloadUse(func(u SSAUse) {
				if u.rec != nil && u.rec.def != nil {
					stack = append(stack, u.rec.def.node)
				}
			})
		}
		if lb, ok := cur.(lambdaBearer); ok {
			lb.forEachLambda(func(lam *Lambda) { stack = append(stack, lam.Body) })
		}
	}
	return false
}

// markDeadStatements returns the set of dead statements: side-effect-free
// statements whose root def's every use belongs to another dead statement.
// Live roots are side-effecting statements; the fixpoint then propagates
// liveness backward along def-use edges (a statement used by a live
// statement is live). Uses with statement-less users fail conservative.
func markDeadStatements(g *Graph) map[*Statement]bool {
	var all []*Statement
	for s := g.cfg.entry.head; s != nil; s = s.next {
		all = append(all, s)
	}
	live := make(map[*Statement]bool, len(all))
	owner := statementOwnership(g)
	for _, s := range all {
		if s.Kind() == StmtEffect || hasEffectNode(s.rhs) {
			live[s] = true
		}
	}
	// Backward fixpoint: statement s is live iff any use of its root def is
	// owned by a live statement (or by no statement — fail conservative).
	// Uses inside s's own tree are internal and never keep s alive.
	for changed := true; changed; {
		changed = false
		for _, s := range all {
			if live[s] {
				continue
			}
			keep := false
			for _, u := range UsesOf(s.rhs) {
				if u.rec == nil || u.rec.killed || u.rec.user == nil {
					continue
				}
				user := u.rec.user
				if owner[user] == s {
					continue // internal use: s's own tree references its root
				}
				if owner[user] == nil || live[owner[user]] {
					keep = true
					break
				}
			}
			if keep {
				live[s] = true
				changed = true
			}
		}
	}
	dead := make(map[*Statement]bool, len(all))
	for _, s := range all {
		if !live[s] {
			dead[s] = true
		}
	}
	return dead
}

// statementOwnership maps every node reachable from the block's statements
// (operand, payload, and lambda-body edges — the SAME child relation Walk
// uses, NOT the printer's stage-only StmtNodes) to its owning statement:
// the first statement, in list order, whose subtree reaches it. Nodes
// reachable from no statement are unowned (nil).
func statementOwnership(g *Graph) map[Node]*Statement {
	owner := make(map[Node]*Statement)
	for s := g.cfg.entry.head; s != nil; s = s.next {
		seen := make(map[Node]bool)
		var visit func(n Node)
		visit = func(n Node) {
			if n == nil || seen[n] {
				return
			}
			seen[n] = true
			if _, taken := owner[n]; !taken {
				owner[n] = s
			}
			if impl := asImpl(n); impl != nil {
				impl.forEachOperand(func(u SSAUse) {
					if u.rec != nil && u.rec.def != nil {
						visit(u.rec.def.node)
					}
				})
			}
			if pb, ok := n.(payloadBearer); ok {
				pb.forEachPayloadUse(func(u SSAUse) {
					if u.rec != nil && u.rec.def != nil {
						visit(u.rec.def.node)
					}
				})
			}
			if lb, ok := n.(lambdaBearer); ok {
				lb.forEachLambda(func(lam *Lambda) { visit(lam.Body) })
			}
		}
		visit(s.rhs)
	}
	return owner
}

// wipeDeadStatements erases the nodes of every dead statement. Nodes are
// attempted in deterministic DFS pre-order per statement (reverse statement
// order across passes); MutContext.Erase's live-use check skips nodes still
// referenced from surviving statements — their dead-side records die as the
// dead users are erased, so the next pass succeeds. A pass that erases
// nothing ends the loop. Erase unlinks a wiped statement's root from the
// intrusive list through its back-pointer (O(1), P1R-02).
func wipeDeadStatements(g *Graph, dead map[*Statement]bool, ctx *MutContext) error {
	var all []*Statement
	for s := g.cfg.entry.head; s != nil; s = s.next {
		all = append(all, s)
	}
	// Owned nodes per statement, in deterministic DFS pre-order: the first
	// statement (in list order) whose subtree reaches a node owns it
	// (owner map), so cross-statement operand defs are attributed to the
	// EARLIER statement and never wiped with the later dead consumer.
	owner := make(map[Node]*Statement)
	ownedBy := make(map[*Statement][]Node, len(all))
	for _, s := range all {
		seen := make(map[Node]bool)
		var visit func(n Node)
		visit = func(n Node) {
			if n == nil || seen[n] {
				return
			}
			seen[n] = true
			if _, taken := owner[n]; !taken {
				owner[n] = s
				ownedBy[s] = append(ownedBy[s], n)
			}
			if impl := asImpl(n); impl != nil {
				impl.forEachOperand(func(u SSAUse) {
					if u.rec != nil && u.rec.def != nil {
						visit(u.rec.def.node)
					}
				})
			}
			if pb, ok := n.(payloadBearer); ok {
				pb.forEachPayloadUse(func(u SSAUse) {
					if u.rec != nil && u.rec.def != nil {
						visit(u.rec.def.node)
					}
				})
			}
			if lb, ok := n.(lambdaBearer); ok {
				lb.forEachLambda(func(lam *Lambda) { visit(lam.Body) })
			}
		}
		visit(s.rhs)
	}
	erased := make(map[Node]bool, len(all))
	for pass := 0; pass <= len(all); pass++ {
		progress := false
		for i := len(all) - 1; i >= 0; i-- {
			s := all[i]
			if !dead[s] {
				continue
			}
			for _, n := range ownedBy[s] {
				if erased[n] || n.BasePtr().erased {
					continue
				}
				if len(UsesOf(n)) > 0 {
					continue // still referenced: a surviving user, or one not yet erased
				}
				if err := ctx.Erase(n); err != nil {
					return err
				}
				erased[n] = true
				progress = true
			}
		}
		if !progress {
			break
		}
	}
	return nil
}
