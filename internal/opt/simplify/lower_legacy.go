// lower_legacy.go — the pkg-builder (legacy) IterCluster dialect lowering
// (quick-261024-k2i, amending the D11 no-op): the pkg rungs wrap each stage
// map in an IterCluster whose non-map operands are the chain's dim
// InputParams, and reference the map's tensor elements through bare
// paramRefs of the tensor defs — no MemLoad/project structure at all. The
// pipeline downstream (tiling, schedule, codegen) consumes the Iter format:
// per-dim DimVar→BoundedDimVar→Iter nests with InductionVariable axes, and
// MemLoad(tensor, ivs) element accesses. This file lowers the legacy
// dialect into exactly that shape, reusing the projecting dialect's
// interned lowerDim machinery so shared (dim, parent) prefixes fuse across
// dialects and GVN sees one nest per chain position.
//
// The dim↔axis pairing the bounds need is recovered by NAME: the frontend
// driver declares each shape dim as a RoleDim InputParam whose debug name
// is the dim's declared name (dynamic dims) or the canonical "e<extent>"
// spelling (static dims — the same spelling pkg.Local documents), and a
// tensor param's interned type carries its DimRefs in axis order. The
// frontend's semantic checks guarantee one dimvar per shape dim per chain,
// so the name match is unambiguous within a chain.

package simplify

import (
	"fmt"
	"strconv"

	"tcacomp/internal/ir"
)

// legacyChain is one processed legacy cluster's resolution state: the
// cluster (for erasure), the wrapped map (the pipe value after erasure),
// and the cluster-LOCAL dim→record table, keyed by the dim param's debug
// name (unique within a chain — dim declarations dedupe) because the
// *ir.Base table keys do not carry the concrete InputParam node the name
// accessor needs.
type legacyChain struct {
	cl        ir.Node
	m         ir.Node
	root      *legacyChain // root cluster of the same chain (self for roots)
	byDim     map[*ir.Base]*dimLowering
	dimByName map[string]*dimLowering
}

// lowerLegacyClusters lowers every legacy-dialect cluster in allocation
// order: root clusters (dims + wrapped map) grow the interned iter nest
// and swap their map's paramRef operands for MemLoads over the nest's
// IVs; piped clusters (no dims, wrapped map over a pipe) are pure wrappers
// — their users rewire to the map and the cluster dies. After every
// cluster is processed, the chains' MemStore axis slots (still dim
// InputParams in the builder dialect) rewire to the owning chain's IVs.
// Erase/SetDef refusals propagate; nothing is force-erased.
func lowerLegacyClusters(g *ir.Graph, mctx *ir.MutContext, clusters []ir.Node, projecting map[*ir.Base][]ir.Node, lowerDim func(dim, parent, tensor ir.Node, axisIdx int) *dimLowering) error {
	roots := map[*ir.Base]*legacyChain{}  // root map -> chain (reduce pipes)
	pipeds := map[*ir.Base]*legacyChain{} // piped map -> chain (store values)
	var order []*legacyChain

	for _, cl := range clusters {
		if len(projecting[cl.BasePtr()]) > 0 {
			continue // projecting dialect: sweep 1 owns this cluster
		}
		ops := ir.OperandsOf(cl)
		if len(ops) == 0 {
			return fmt.Errorf("simplify-lower: cluster node %d carries no operands (neither legacy wrapper nor projecting format)", cl.BasePtr().ID())
		}
		// Operand layout: leading RoleDim InputParams (the chain dims), then
		// the wrapped map. The builder always wraps exactly one map.
		var dims []ir.Node
		var inputs []ir.Node
		for _, o := range ops {
			if o == nil {
				continue
			}
			if role, ok := ir.RoleOf(o); ok && role == ir.RoleDim && len(inputs) == 0 {
				dims = append(dims, o)
				continue
			}
			inputs = append(inputs, o)
		}
		if len(inputs) != 1 || inputs[0].KindOf() != ir.KindMap {
			return fmt.Errorf("simplify-lower: legacy cluster %d wraps %v, want exactly one map (the builder dialect wraps each stage map once)",
				cl.BasePtr().ID(), kindsOf(inputs))
		}
		m := inputs[0]
		mops := ir.OperandsOf(m)

		// D11 dialect gate: the pkg builder's maps carry ONLY paramRef
		// tensor operands, the lambda, the as-order, and (piped) the
		// producer stage's map/reduce as the pipe. A map with any other
		// operand kind (a const, a memload, an expression) is the
		// documented D11 no-op wrapper — NOT the builder dialect — and
		// passes through untouched (TestLowerLegacyNoOp pins a const
		// operand; lowering it would invent an iteration space the chain
		// never declared).
		if !isBuilderMap(mops) {
			continue
		}

		lc := &legacyChain{cl: cl, m: m, byDim: map[*ir.Base]*dimLowering{}, dimByName: map[string]*dimLowering{}}
		order = append(order, lc)

		if len(dims) == 0 {
			// Piped cluster: no iteration space of its own. The wrapped
			// map's operand 0 is the pipe; operand paramRefs, when present,
			// bind the PRODUCING chain's live dims.
			pipe := mops[0]
			if pipe == nil {
				return fmt.Errorf("simplify-lower: piped map %d carries no pipe operand", m.BasePtr().ID())
			}
			var root *legacyChain
			if pipe.KindOf() == ir.KindReduce {
				if p := ir.OperandsOf(pipe)[0]; p != nil {
					root = roots[p.BasePtr()]
				}
			}
			if root == nil {
				return fmt.Errorf("simplify-lower: piped map %d's pipe does not resolve to a lowered root map of this run", m.BasePtr().ID())
			}
			if err := lc.lowerOperandRefs(g, mctx, m, root.dimByName); err != nil {
				return err
			}
			pipeds[m.BasePtr()] = lc
			lc.root = root
		} else {
			// Root cluster: intern the iter nest in dim order (the chain's
			// innermost axis is last — the frontend driver parks the
			// reduced axes there), pairing each dim with the first map
			// operand whose tensor type mentions it.
			var parentIter ir.Node
			for _, dim := range dims {
				tensor, axis, err := firstPairing(dim, mops)
				if err != nil {
					return fmt.Errorf("simplify-lower: legacy cluster %d: %w", cl.BasePtr().ID(), err)
				}
				dl := lowerDim(dim, parentIter, tensor, axis)
				lc.byDim[dim.BasePtr()] = dl
				lc.dimByName[ir.InputParamNameOf(dim)] = dl
				parentIter = dl.iter
			}
			if err := lc.lowerOperandRefs(g, mctx, m, lc.dimByName); err != nil {
				return err
			}
			roots[m.BasePtr()] = lc
			lc.root = lc
		}

		// The cluster is a pure wrapper once its map's operands stand alone:
		// every use (a reduce's pipe, a store's value, another cluster's
		// operand) takes the map instead, and the cluster dies.
		if err := mctx.ReplaceAllUsesWith(cl, m); err != nil {
			return fmt.Errorf("simplify-lower: rewire legacy cluster %d to its map: %w", cl.BasePtr().ID(), err)
		}
		if err := mctx.Erase(cl); err != nil {
			return fmt.Errorf("simplify-lower: erase legacy cluster %d: %w", cl.BasePtr().ID(), err)
		}
	}

	// Store axes: the builder dialect's structural stores carry dim
	// InputParams; the schedule/codegen consume IVs. The owning chain is
	// the one whose map the store's value flows from — directly (piped
	// map) or through the chain's reduce (direct store after reduce).
	for _, n := range g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemStore {
			continue
		}
		ops := ir.OperandsOf(n)
		if len(ops) < 2 {
			continue
		}
		value := ops[0]
		if value == nil {
			continue
		}
		var lc *legacyChain
		if c, ok := pipeds[value.BasePtr()]; ok {
			lc = c.root
		} else if value.KindOf() == ir.KindReduce {
			if piped := ir.OperandsOf(value)[0]; piped != nil {
				if c, ok := roots[piped.BasePtr()]; ok {
					lc = c
				}
			}
		}
		if lc == nil {
			continue // not a legacy chain's store (projecting dialect or pre-lowered)
		}
		for _, dim := range ops[2:] {
			if dim == nil {
				continue
			}
			dl, ok := lc.byDim[dim.BasePtr()]
			if !ok {
				return fmt.Errorf("simplify-lower: store %d axis %d binds dim node %d, which is not a lowered dim of its chain (unlowered legacy axis)",
					n.BasePtr().ID(), axisSlot(n, dim), dim.BasePtr().ID())
			}
			for _, u := range ir.UsesOf(dim) {
				if user := u.User(); user != nil && user.BasePtr() == n.BasePtr() {
					if err := mctx.SetDef(u, dl.iv); err != nil {
						return fmt.Errorf("simplify-lower: rewire store %d axis: %w", n.BasePtr().ID(), err)
					}
				}
			}
		}
	}
	return nil
}

// lowerOperandRefs replaces one legacy map's paramRef operands with
// MemLoads over the chain's IVs: the paramRef wraps the tensor def with no
// axis information (the builder validated axes handle-side), so the axes
// rebuild from the tensor's interned type — DimRef j resolves to the dim's
// record in the owning chain's name-keyed table.
func (lc *legacyChain) lowerOperandRefs(g *ir.Graph, mctx *ir.MutContext, m ir.Node, dimByName map[string]*dimLowering) error {
	for _, op := range ir.OperandsOf(m) {
		if op == nil || op.KindOf() != ir.KindParam {
			continue // the As node, the lambda, the pipe def
		}
		tdef := ir.OperandsOf(op)[0]
		if tdef == nil {
			return fmt.Errorf("simplify-lower: legacy map %d operand %d is a paramRef over nothing", m.BasePtr().ID(), op.BasePtr().ID())
		}
		// Scalar captures (e.g. init values, scalar [scale]-style captures)
		// are paramRefs wrapping a scalar InputParam — they pass through
		// unchanged: the lambda receives the scalar directly, no MemLoad
		// or iteration axis is involved. Only tensor paramRefs lower to
		// MemLoads over the chain's IVs.
		if t, ok := ir.InputParamTypeOf(tdef); ok && t != nil {
			if nt, ok := t.(*ir.NumericType); ok && nt.IsScalar() {
				continue
			}
		}
		refs, err := tensorDimRefs(tdef)
		if err != nil {
			return fmt.Errorf("simplify-lower: legacy map %d operand %d: %w", m.BasePtr().ID(), op.BasePtr().ID(), err)
		}
		axes := make([]ir.Node, len(refs))
		for j, ref := range refs {
			dl := matchDimRef(dimByName, ref)
			if dl == nil {
				return fmt.Errorf("simplify-lower: legacy map %d operand %d axis %d (%v) matches no lowered dim of its chain",
					m.BasePtr().ID(), op.BasePtr().ID(), j, ref)
			}
			axes[j] = dl.iv
		}
		load := g.NewMemLoad("simplify-lower", tdef, axes)
		if err := mctx.ReplaceAllUsesWith(op, load); err != nil {
			return fmt.Errorf("simplify-lower: rewire legacy map %d operand %d to a load: %w", m.BasePtr().ID(), op.BasePtr().ID(), err)
		}
		if err := mctx.Erase(op); err != nil {
			return fmt.Errorf("simplify-lower: erase legacy paramRef %d: %w", op.BasePtr().ID(), err)
		}
	}
	return nil
}

// firstPairing finds the bound pairing of one chain dim: the first map
// operand (paramRef order) whose tensor type mentions the dim — named refs
// match the dim param's name, static extents the canonical "e<extent>"
// spelling the frontend driver declares static dims under.
func firstPairing(dim ir.Node, mops []ir.Node) (tensor ir.Node, axis int, err error) {
	for _, op := range mops {
		if op == nil || op.KindOf() != ir.KindParam {
			continue
		}
		tdef := ir.OperandsOf(op)[0]
		if tdef == nil {
			continue
		}
		refs, err := tensorDimRefs(tdef)
		if err != nil {
			continue // operands are tensors; a non-tensor here fails lowerOperandRefs later
		}
		for j, ref := range refs {
			if dimRefMatches(dim, ref) {
				return tdef, j, nil
			}
		}
	}
	return nil, 0, fmt.Errorf("dim node %d (%s) has no pairing: no map operand's tensor type mentions it",
		dim.BasePtr().ID(), ir.InputParamNameOf(dim))
}

// tensorDimRefs reads one tensor def's interned DimRefs in axis order.
// MemNew-backed locals (the frontend's let-locals) are deliberately
// untyped value objects, so their dims recover from the store that
// initializes them — the builder's structural store pins the local's axes
// in order; a local is always stored before it is read.
func tensorDimRefs(tdef ir.Node) ([]ir.DimRef, error) {
	if t, ok := ir.InputParamTypeOf(tdef); ok && t != nil {
		nt, ok := t.(*ir.NumericType)
		if !ok || nt.IsScalar() {
			return nil, fmt.Errorf("tensor def %d has a non-tensor type %T", tdef.BasePtr().ID(), t)
		}
		return nt.Dims(), nil
	}
	if tdef.KindOf() != ir.KindMemNew {
		return nil, fmt.Errorf("tensor def %d is neither a typed InputParam nor a MemNew local", tdef.BasePtr().ID())
	}
	// Local: recover the ordered dims from its initializing store.
	for _, u := range ir.UsesOf(tdef) {
		user := u.User()
		if user == nil || user.KindOf() != ir.KindMemStore {
			continue
		}
		ops := ir.OperandsOf(user)
		if len(ops) < 3 || ops[1] == nil || ops[1].BasePtr() != tdef.BasePtr() {
			continue
		}
		refs := make([]ir.DimRef, 0, len(ops)-2)
		for _, ax := range ops[2:] {
			if ax == nil {
				continue
			}
			if role, ok := ir.RoleOf(ax); !ok || role != ir.RoleDim {
				return nil, fmt.Errorf("local %d's store axis is not a dim param (%s)", tdef.BasePtr().ID(), ax.KindOf())
			}
			refs = append(refs, ir.DimRef{Name: ir.InputParamNameOf(ax)})
		}
		if len(refs) > 0 {
			return refs, nil
		}
	}
	return nil, fmt.Errorf("local %d has no initializing store to recover its dims from", tdef.BasePtr().ID())
}

// dimRefMatches reports whether one chain dim node is the declared dim of
// one tensor-type DimRef: named refs by the dim param's name, static
// extents by the driver's canonical "e<extent>" spelling.
func dimRefMatches(dim ir.Node, ref ir.DimRef) bool {
	if ref.Name != "" {
		return ir.InputParamNameOf(dim) == ref.Name
	}
	if ref.Extent == 0 {
		return false // a written constant 0 has no dim (design §2.4)
	}
	return ir.InputParamNameOf(dim) == "e"+strconv.FormatUint(ref.Extent, 10)
}

// matchDimRef resolves one tensor-type DimRef against a chain's
// name-keyed dim table: named refs by the dim's declared name, static
// extents by the driver's canonical "e<extent>" spelling.
func matchDimRef(dimByName map[string]*dimLowering, ref ir.DimRef) *dimLowering {
	name := ref.Name
	if name == "" {
		if ref.Extent == 0 {
			return nil // a written constant 0 has no dim (design §2.4)
		}
		name = "e" + strconv.FormatUint(ref.Extent, 10)
	}
	return dimByName[name]
}

// axisSlot reports which operand slot of store n holds dim (diagnostic
// payload for the fail-closed store-axis error).
func axisSlot(n, dim ir.Node) int {
	for i, o := range ir.OperandsOf(n) {
		if o != nil && o.BasePtr() == dim.BasePtr() {
			return i
		}
	}
	return -1
}

// isBuilderMap reports whether one wrapped map uses the pkg builder's
// operand signature, which the legacy sweep knows how to lower. The builder
// always produces maps whose non-lambda, non-as operands are ALL
// KindParam (tensor paramRefs) — either all tensor operands (root map), one
// pipe operand (map/reduce) plus optional paramRefs (piped map). Any other
// operand kind (const, memload, binop, …) marks a D11-dialect map that the
// sweep must pass through untouched (TestLowerLegacyNoOp pins this for a
// const-only map; lowering it would invent an iteration space the chain
// never declared).
func isBuilderMap(mops []ir.Node) bool {
	for _, op := range mops {
		if op == nil {
			continue
		}
		k := op.KindOf()
		// Lambda body, as-order node, and pipe operands are expected structure
		// in both builder and non-builder maps — skip them; the tensor
		// operands are what distinguish the dialects.
		if k == ir.KindLambda || k == ir.KindAs {
			continue
		}
		// Pipe operands (map or reduce feeding a piped stage) are expected
		// in builder piped maps — they aren't tensor paramRefs but are part
		// of the builder shape.
		if k == ir.KindMap || k == ir.KindReduce {
			continue
		}
		// All other operands must be paramRefs (KindParam) in the builder
		// dialect. A const, memload, expression, etc. means this is the D11
		// no-op wrapper.
		if k != ir.KindParam {
			return false
		}
	}
	return true
}

// kindsOf renders a node list's kinds (diagnostic payload).
func kindsOf(nodes []ir.Node) []ir.Kind {
	ks := make([]ir.Kind, len(nodes))
	for i, n := range nodes {
		if n == nil {
			ks[i] = ir.Kind(0) // "unknown" slot
			continue
		}
		ks[i] = n.KindOf()
	}
	return ks
}
