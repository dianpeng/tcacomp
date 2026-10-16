package ir

// This file is the building-phase dead-code elimination the Phase-1 review
// mandates (P1R-08): the builder may emit chains whose intermediate values
// are never consumed (an abandoned Map before a Store, a Reduce whose result
// is overwritten), and wiping them belongs to construction, not to the pass
// pipeline — passes never see building debris.
//
// The model is the review's three steps:
//
//  1. MARK: a statement is dead iff it is side-effect-free (no reachable
//     store/return node — StmtEffect roots are live by classification) and
//     every use of its root def belongs to a dead statement (uses owned by
//     unreachable, statement-less nodes keep the statement alive:
//     fail conservative).
//  2. PROPAGATE: backward fixpoint — a statement whose root def is used by a
//     live statement becomes live itself.
//  3. WIPE: erase every node owned by a dead statement through
//     MutContext.Erase, consumers before producers (reverse statement order,
//     reverse StmtNodes order within a statement), skipping nodes that still
//     carry live uses (shared with a surviving statement). Skipped nodes lose
//     their dead-side records as the dead users die, so they re-enter the
//     surviving graph cleanly. Erase unlinks wiped statements from the
//     intrusive list through the root's back-pointer (O(1), P1R-02).
//
// Builder-produced graphs share only statement ROOTS across statements, so
// the wipe is total for them; hand-built raw graphs with non-root sharing
// are handled conservatively (no corruption, possibly residual statements).

// BuildingDCE runs building-phase dead-code elimination over g, wiping dead
// statements in place. It runs BEFORE the pass pipeline (pkg.Builder calls
// it at Build time), so passes never observe construction debris. A nil
// graph is a no-op; the returned error reports only genuine mutation
// failures (Erase on a live-used node is skipped, never an error).
func BuildingDCE(g *Graph) error {
	if g == nil {
		return nil
	}
	dead := markDeadStatements(g)
	if len(dead) == 0 {
		return nil
	}
	ctx := &MutContext{Pass: "building-dce"}
	return wipeDeadStatements(g, dead, ctx)
}

// isEffectKind reports whether kind k carries an observable side effect in
// v0. Stores, kernel returns, and tile stores are the only effects;
// everything else — including loads and loop nests without an effect in
// their body (caught by hasEffectNode's reachability walk) — is
// dead-code-eligible when unreferenced.
func isEffectKind(k Kind) bool {
	switch k {
	case KindStore, KindReturn, KindStoreTile:
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
