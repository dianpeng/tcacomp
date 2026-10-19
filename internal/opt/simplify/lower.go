// Package simplify implements the Simplification stage of the pipeline
// (quick-261019-ktm, doc/optana/lower_itercluster.md): three registered
// passes that rewrite the frontend's IterCluster-format graphs into plain
// SSA data flow almost immediately after construction.
//
//	simplify-lower   (102)  IterCluster format → Iter format
//	simplify-inline  (103)  lambda parameter inlining (requires lower)
//	simplify-verify  (104)  post-simplify invariants (requires inline)
//
// They slot after dce cleanup (100/101) and before gvn (110), so GVN sees
// — and dedupes — the lowered TensorDimOf/Iter/InductionVariable nodes.
package simplify

import (
	"errors"
	"fmt"

	"tcacomp/internal/ir"
	"tcacomp/internal/opt"
)

// init self-registers the lowering pass: Priority 102 — after dce/dce-verify
// (100/101), before gvn (110). The registry scale in internal/opt/registry.go
// documents the anchors.
func init() {
	opt.Register(opt.PassSpec{
		ID:       "simplify-lower",
		Priority: 102,
		Factory:  func() opt.Pass { return LowerPass{} },
	})
}

// LowerPass rewrites the IterCluster format into the Iter format. The
// IterCluster format (what the frontend will emit for map/reduce stages):
// an IterCluster whose Input list holds MemLoads, bridged into a Map
// through IterClusterProject nodes (Input slot i ↔ lambda param i), with
// the as-order riding a trailing As node. The pass rewrites it into
// data-dependency form:
//
//	one Iter per as-dim, bounded by TensorDimOf(<first pairing's tensor>,
//	<axis>), init/step 0/1 (D5);
//	one InductionVariable per dim (interned: a dim shared by several loads
//	yields ONE iv, first-wins);
//	each MemLoad axis slot rewired to the dim's iv;
//	each project-bearing stage input slot rewired to the cluster's input
//	def directly;
//	projects and processed clusters erased.
//
// Legacy builder output — a cluster wrapping a Map, no projects — is a
// documented no-op (D11): the pass touches nothing.
//
// All mutations ride one MutContext{Pass: "simplify-lower"} through the
// choke points (SetDef for rewiring, Erase for removal); Erase refusals
// propagate as errors — no force-erasure, no direct field writes.
type LowerPass struct{}

// ID returns the pipeline name "simplify-lower".
func (LowerPass) ID() string { return "simplify-lower" }

// Requires: lowering has no ordering preconditions.
func (LowerPass) Requires() []string { return nil }

// Produces: builder-output stage — the pass lowers within the HIR suite
// (Iter/TensorDimOf/InductionVariable are catalog kinds), so the
// constructed legality window re-checks the graph.
func (LowerPass) Produces() opt.Stage { return opt.StageConstructed }

// Preserved: nothing — every analysis is invalidated.
func (LowerPass) Preserved() []string { return nil }

// dimLowering is the per-dim interned triple of the Iter format.
type dimLowering struct {
	tdim ir.Node
	iter ir.Node
	iv   ir.Node
}

// Run performs the lowering over the Level-1 sea. Gates first (nil and
// materialized/CFG-carrying graphs are deterministic errors evaluated
// before any mutation, so a refused graph is untouched); then three
// sweeps, all in allocation order for determinism:
//
//	(1) per cluster with projecting maps: derive the ordered dim list from
//	    the projecting maps' As operands (map allocation order, As order,
//	    dedup), intern the TensorDimOf/Iter/InductionVariable triple per
//	    dim (first MemLoad axis pairing wins), fail closed on a dim no
//	    axis covers;
//	(2) rewire every processed cluster's MemLoad axis slots to their dim's
//	    iv, and every project's uses to the project's cluster input def
//	    (bounds-checked, fail closed);
//	(3) erase the projects (now use-free), then the processed clusters —
//	    Erase refusals propagate (a hybrid graph still piping a cluster
//	    surfaces the refusal; never force-erase).
//
// Clusters with ZERO projecting maps: MemLoad inputs are a live uncovered
// iteration space (error); non-MemLoad inputs only (the legacy wrapper)
// skip the cluster entirely — the documented no-op.
func (LowerPass) Run(_ *opt.Context, g *ir.Graph) error {
	if g == nil {
		return errors.New("simplify-lower: nil graph")
	}
	if g.Materialized() || g.CFG() != nil {
		return errors.New("simplify-lower: graph already materialized (simplify runs on the Level-1 sea)")
	}

	nodes := g.AllocatedNodes()

	// Collect clusters and projects in allocation order; index the
	// projects by their operand-0 cluster.
	var clusters []ir.Node
	var projects []ir.Node
	for _, n := range nodes {
		switch n.KindOf() {
		case ir.KindIterCluster:
			clusters = append(clusters, n)
		case ir.KindIterClusterProject:
			projects = append(projects, n)
		}
	}
	if len(clusters) == 0 && len(projects) == 0 {
		return nil // nothing in the IterCluster format anywhere: no-op
	}

	// The maps/reduces carrying projects, in allocation order, per cluster.
	projectingStages := make(map[*ir.Base][]ir.Node)
	for _, p := range projects {
		cl := ir.OperandsOf(p)[0]
		if cl == nil {
			continue
		}
		for _, u := range ir.UsesOf(p) {
			if user := u.User(); user != nil {
				projectingStages[cl.BasePtr()] = appendUnique(projectingStages[cl.BasePtr()], user)
			}
		}
	}

	mctx := &ir.MutContext{Pass: "simplify-lower"}

	// Interning tables for the Run: one lowering per dim node, shared
	// const0/const1 (TF32 per the DimVar test convention, D5) — created
	// LAZILY on the first lowered dim so legacy no-op graphs allocate
	// nothing.
	lowered := make(map[*ir.Base]*dimLowering)
	var const0, const1 ir.Node

	// memLoadInputs returns the cluster's MemLoad input defs, in input order.
	memLoadInputs := func(cl ir.Node) []ir.Node {
		var loads []ir.Node
		for _, in := range ir.OperandsOf(cl) {
			if in != nil && in.KindOf() == ir.KindMemLoad {
				loads = append(loads, in)
			}
		}
		return loads
	}

	// lowerDim interns (or returns) the triple for dim, bound by the given
	// first pairing (tensor, axisIdx).
	lowerDim := func(dim, tensor ir.Node, axisIdx int) *dimLowering {
		if dl, ok := lowered[dim.BasePtr()]; ok {
			return dl
		}
		if const0 == nil {
			const0 = g.NewConst("simplify-lower", ir.InternScalar(ir.TF32), 0)
			const1 = g.NewConst("simplify-lower", ir.InternScalar(ir.TF32), 1)
		}
		tdim := g.NewTensorDimOf("simplify-lower", tensor, axisIdx)
		iter := g.NewIter("simplify-lower", dim, tdim, const0, const1)
		iv := g.NewInductionVariable("simplify-lower", iter)
		dl := &dimLowering{tdim: tdim, iter: iter, iv: iv}
		lowered[dim.BasePtr()] = dl
		return dl
	}

	// Sweep 1: derive dims per processed cluster and intern the triples.
	var processed []ir.Node
	for _, cl := range clusters {
		stages := projectingStages[cl.BasePtr()]
		if len(stages) == 0 {
			// No projecting map. A MemLoad input is a live uncovered
			// iteration space (fail closed); the legacy wrapper (non-load
			// inputs only) skips the cluster entirely.
			if len(memLoadInputs(cl)) > 0 {
				return fmt.Errorf("simplify-lower: cluster node %d carries MemLoad inputs but no projecting map (live uncovered iteration space)",
					cl.BasePtr().ID())
			}
			continue
		}

		// Ordered dim list: projecting maps in allocation order, As
		// operands in As order, dedup by node identity.
		var dims []ir.Node
		for _, st := range stages {
			as := ir.AsOf(st)
			if as == nil {
				continue
			}
			for _, d := range ir.OperandsOf(as) {
				if d == nil {
					continue
				}
				dims = appendUnique(dims, d)
			}
		}

		// Coverage is bidirectional, both fail-closed: every as-dim must
		// find a MemLoad axis pairing (else its iter has no bound), and
		// every MemLoad axis that is a dim InputParam must be covered by
		// some projecting map's As (else the load keeps an uncovered
		// iteration axis).
		inDims := make(map[*ir.Base]bool, len(dims))
		for _, d := range dims {
			inDims[d.BasePtr()] = true
		}
		for _, ld := range memLoadInputs(cl) {
			for slot, ax := range ir.OperandsOf(ld) {
				if slot == 0 || ax == nil {
					continue
				}
				if role, ok := ir.RoleOf(ax); ok && role == ir.RoleDim && !inDims[ax.BasePtr()] {
					return fmt.Errorf("simplify-lower: dim node %d (%s) is not covered by any projecting map's as-order (MemLoad axis binds it)",
						ax.BasePtr().ID(), ax.KindOf())
				}
			}
		}

		// Per dim: the FIRST MemLoad axis pairing (loads in input order,
		// axes in slot order) is the dim's bound. Validate the cluster's
		// full coverage BEFORE creating anything (a dim no axis covers is
		// an iteration space with no bound — fail closed, naming the dim),
		// then intern the triples.
		type pairing struct {
			tensor ir.Node
			axis   int
		}
		pairs := make([]pairing, len(dims))
		for i, dim := range dims {
			bound := false
			for _, ld := range memLoadInputs(cl) {
				ops := ir.OperandsOf(ld)
				for slot := 1; slot < len(ops); slot++ {
					if ops[slot] != nil && ops[slot].BasePtr() == dim.BasePtr() {
						pairs[i] = pairing{tensor: ops[0], axis: slot - 1}
						bound = true
						break
					}
				}
				if bound {
					break
				}
			}
			if !bound {
				return fmt.Errorf("simplify-lower: dim node %d (%s) is not covered by any projecting map's as-order (no MemLoad axis binds it)",
					dim.BasePtr().ID(), dim.KindOf())
			}
		}
		for i, dim := range dims {
			lowerDim(dim, pairs[i].tensor, pairs[i].axis)
		}
		processed = append(processed, cl)
	}

	// Sweep 2a: rewire each processed cluster's MemLoad axis slots whose
	// def has an interned iter to the dim's iv.
	for _, cl := range processed {
		for _, ld := range memLoadInputs(cl) {
			axisDefs := ir.OperandsOf(ld) // snapshot: slots hold these defs now
			for slot := 1; slot < len(axisDefs); slot++ {
				dim := axisDefs[slot]
				if dim == nil {
					continue
				}
				dl, ok := lowered[dim.BasePtr()]
				if !ok {
					continue // not a lowered dim (e.g. a constant axis): untouched
				}
				for _, u := range ir.UsesOf(dim) {
					if user := u.User(); user != nil && user.BasePtr() == ld.BasePtr() {
						if err := mctx.SetDef(u, dl.iv); err != nil {
							return fmt.Errorf("simplify-lower: rewire load %d axis: %w", ld.BasePtr().ID(), err)
						}
					}
				}
			}
		}
	}

	// Sweep 2b: rewire every project's uses to the project's cluster input
	// def (bounds-checked, fail closed).
	for _, p := range projects {
		cl := ir.OperandsOf(p)[0]
		if cl == nil {
			return fmt.Errorf("simplify-lower: project node %d has no cluster operand", p.BasePtr().ID())
		}
		idx, ok := ir.IterClusterProjectIndex(p)
		if !ok {
			continue // unreachable: p is a project by collection
		}
		inputs := ir.OperandsOf(cl)
		if idx < 0 || idx >= len(inputs) || inputs[idx] == nil {
			return fmt.Errorf("simplify-lower: project node %d indexes cluster input %d of %d (out of range)",
				p.BasePtr().ID(), idx, len(inputs))
		}
		for _, u := range ir.UsesOf(p) {
			if err := mctx.SetDef(u, inputs[idx]); err != nil {
				return fmt.Errorf("simplify-lower: rewire project %d: %w", p.BasePtr().ID(), err)
			}
		}
	}

	// Sweep 3: erase the projects (use-free after 2b), then the processed
	// clusters — propagate Erase refusals.
	for _, p := range projects {
		if err := mctx.Erase(p); err != nil {
			return fmt.Errorf("simplify-lower: erase project %d: %w", p.BasePtr().ID(), err)
		}
	}
	for _, cl := range processed {
		if err := mctx.Erase(cl); err != nil {
			return fmt.Errorf("simplify-lower: erase cluster %d: %w", cl.BasePtr().ID(), err)
		}
	}
	return nil
}

// appendUnique appends n (node identity) when absent; deterministic order.
func appendUnique(list []ir.Node, n ir.Node) []ir.Node {
	for _, e := range list {
		if e.BasePtr() == n.BasePtr() {
			return list
		}
	}
	return append(list, n)
}
