package ir

import "slices"

// node.go is the IR's structural core: the shared Base cell, the node
// interfaces (Node/Value/ScalarExprNode/nodeImpl), the operand-slot storage
// (operandList), and the Graph root with its allocation/wiring primitives.
// The concrete node families live in the layered node_xxx.go files next to
// this one (R4a); the per-kind metadata table lives in catalog.go (R4e).
//
// The exported graph-construction surface (constructors) is split across the
// node_xxx.go files beside the node types they build — the ONLY way to create
// graphs, params, and nodes from outside package ir (pkg/ builder, pass
// tests, visualization tests). Statements are NOT constructed here — Level 1
// is a statement-free sea of nodes; the Level-2 statement list is produced
// solely by Materialize (schedule.go) plus the NewStatement factory (cfg.go)
// for passes that run after scheduling. Construction is exempt from MutContext
// (RESEARCH Pattern 2) but records PassOrigin so verifier diagnostics are
// meaningful from birth (D-17).
//
// Constructors deliberately do NOT validate kinds or operands: kind
// acceptance is the legality window's job (IR-03) and operand wiring is the
// verifier's job (PASS-06). This lets cross-package tests construct
// out-of-window or unwired graphs to prove rejection.
//
// Every reference in the graph is an SSAUse (review P1R-01): operand slots
// hold SSAUse handles, and payload fields (lambda params, store target
// accesses, iteration dims/bounds, map as-orders, parameter refs, ImpactBy
// effect inputs) hold SSAUse records wired exactly like operand uses — a
// fresh ref record in the def's SSAVar, owned by the referencing node. Names
// appear ONLY as InputParam debug payload and are never used for resolution
// or identity.

// Span is a source location range. It is the zero value throughout Phase 1:
// there is no frontend yet, so nodes carry no source spans (IR-06).
type Span struct {
	File            string
	Line, Col       int
	EndLine, EndCol int
}

// Loc is the minimal location record every node carries (IR-06): the
// frontend span (zero in Phase 1) and the pass that originated the node.
type Loc struct {
	Span       Span
	PassOrigin string
}

// Base is the shared cell embedded in every IR node (D-15). It holds the
// node's stable ID, kind, location, last-touched-by stamp, and the node's
// SSAVar — the sole pointee of the SSA handle layer (directive 2): the uses
// that point AT this node as their def live in Base.ssa's ref list.
//
// Base exposes no public mutators (D-15): the only sanctioned mutation path
// is the MutContext choke-point API in mutate.go (IR-04).
type Base struct {
	id            uint64
	kind          Kind
	loc           Loc
	lastTouchedBy string
	ssa           *SSAVar    // this node's SSA cell; allocated at construction (1:1, ID == node ID)
	erased        bool       // tombstone set by Erase; IDs are never reused (IR-05)
	surplus       int        // operands supplied beyond the fixed-arity slot count at construction; the verifier flags > 0 (fail-closed, WR-01)
	graph         *Graph     // owning graph, set at construction; lets Erase unlink from the block
	stmt          *Statement // the statement whose RHS root this node is, nil for non-root nodes; lets Erase unlink in O(1)
	stmtID        uint64     // builder-pinned statement ID for Materialize (PinStatementID); 0 = Materialize draws from the counter
}

// BasePtr returns the shared cell, implementing Node.
func (b *Base) BasePtr() *Base { return b }

// KindOf returns the node's kind, implementing Node.
func (b *Base) KindOf() Kind { return b.kind }

// ID returns the node's stable per-graph integer ID (IR-05).
func (b *Base) ID() uint64 { return b.id }

// LastTouchedBy returns the name of the last pass that mutated this node
// through a choke-point mutation (D-17); empty for never-mutated nodes.
func (b *Base) LastTouchedBy() string { return b.lastTouchedBy }

// Loc returns the node's location record (IR-06).
func (b *Base) Loc() Loc { return b.loc }

// Node is the root IR node interface (D-15, go/ast shape): every concrete
// node family satisfies it through its embedded *Base.
type Node interface {
	BasePtr() *Base
	KindOf() Kind
}

// ScalarExprNode is the marker interface for scalar-expression node kinds
// (const, param, binop, unaryop, select, cast, call). Passes type-switch
// over families rather than enumerating kinds.
type ScalarExprNode interface {
	Node
	scalarExprNode()
}

// Value is the RHS-capable node interface (directive 2): every statement root
// that produces a referenceable value implements it. A none Value is a Value
// but just nothing — the return root overrides IsNone to report exactly that,
// so it can sit in a statement RHS slot while binding no SSA name. Every
// concrete node embeds *Base, which promotes IsNone() == false; only
// returnNode overrides it. Node stays the universal root (D-15).
type Value interface {
	Node
	IsNone() bool
}

// IsNone reports whether the node is the none Value. *Base promotes this to
// every node kind; only returnNode overrides it (true).
func (b *Base) IsNone() bool { return false }

// Graph is one kernel/function's IR: a name, the kernel input-parameter
// nodes, and — once Materialize runs (Level 2) — a single-block CFG
// (IR-02). Between construction and Materialize the graph is a pure sea of
// nodes: cfg is nil, Block()/CFG() return nil (T-quick-02 nil-safety), and
// ordering of side effects rides the explicit ImpactBy effect chains (see
// effect.go). Node and statement IDs come only from NewID: per-graph
// monotonic, starting at 1, never reused even across Erase (IR-05). Every
// reference anywhere in the graph is an SSAUse; every definition is an
// SSADef — names survive only as InputParam debug payload.
type Graph struct {
	name         string
	params       []Node // InputParam nodes (kernel roles only), in declaration order
	cfg          *CFG   // Level 2 only: nil from NewGraph until Materialize
	nextID       uint64 // next ID to hand out; monotonic
	nodes        []Node // every node ever allocated, in allocation order (never shrinks; erased nodes stay tombstoned) — the sea's sweep registry
	end          Node   // R2: the graph's ONE end node (the first KindReturn created); nil until a return exists — Level-1 Walk/Verify/DCE root beside the RoleTensor params (R3)
	materialized bool   // Materialize has run; cfg and the statement list exist
}

// NewID returns the graph's next monotonic ID. This is the ONLY source of
// node and statement IDs; IDs are never reused (IR-05).
func (g *Graph) NewID() uint64 {
	id := g.nextID
	g.nextID++
	return id
}

// Name returns the graph's name.
func (g *Graph) Name() string { return g.name }

// Params returns a copy of the graph's kernel input-parameter nodes in
// declaration order. There is deliberately NO name→param lookup: names are
// debug payload, and resolution happens at construction time (review P1R-01).
func (g *Graph) Params() []Node { return append([]Node(nil), g.params...) }

// Block returns the CFG's entry basic block — the graph's single block in v0.
// It is nil until Materialize runs (Level 1 is a statement-free sea of nodes,
// T-quick-02 nil-safety): construction and passes must not consult the block.
func (g *Graph) Block() *BasicBlock {
	if g == nil || g.cfg == nil {
		return nil
	}
	return g.cfg.entry
}

// CFG returns the graph's control-flow graph (exactly one block in v0), or nil
// before Materialize.
func (g *Graph) CFG() *CFG { return g.cfg }

// Materialized reports whether Materialize has run: the graph is Level 2 with
// a CFG and a statement list. Codegen-adjacent surfaces (print, DOT) require
// it; passes operate on the sea and never need it.
func (g *Graph) Materialized() bool { return g != nil && g.materialized }

// nodeImpl is the internal interface every concrete node type satisfies:
// the public Node surface plus operand-slot access for chain maintenance.
// All mutation and verification flows through these hooks, never through
// concrete node fields. Operand slots hold SSAUse value-type handles; a slot
// is wired iff its handle's record is non-nil with a non-nil target.
type nodeImpl interface {
	Node
	forEachOperand(fn func(SSAUse)) // visit each operand slot in canonical order
	addSlot() SSAUse                // append one operand slot and return its (zero) handle
	setSlot(i int, u SSAUse)        // write slot i (i < slot count); construction wiring only
	removeSlot(u SSAUse) bool       // drop u's slot from the operand storage (record identity)
}

// operandList is embedded by every concrete node type to provide the
// operand-slot half of nodeImpl uniformly. The slice stores SSAUse handles BY
// VALUE; each handle points at a heap ref record shared with the def's SSAVar
// ref list, so slice growth/removal moves handle copies only — the records
// (and the chain state they carry) never move and never dangle (CR-01/CR-02).
// A zero handle (rec == nil) is an unwired slot.
type operandList struct {
	ops []SSAUse // one SSAUse per operand, in canonical operand order
}

func (l *operandList) forEachOperand(fn func(SSAUse)) {
	for _, u := range l.ops {
		fn(u)
	}
}

func (l *operandList) addSlot() SSAUse {
	l.ops = append(l.ops, SSAUse{}) // a fresh slot is a zero (unwired) handle
	return l.ops[len(l.ops)-1]
}

func (l *operandList) setSlot(i int, u SSAUse) {
	l.ops[i] = u
}

func (l *operandList) removeSlot(u SSAUse) bool {
	if u.rec == nil {
		return false
	}
	for i, s := range l.ops {
		if s.rec == u.rec { // record-identity comparison
			l.ops = slices.Delete(l.ops, i, i+1) // drops a handle copy; the record stays valid
			return true
		}
	}
	return false
}

// rawNode is the generic fallback for KindInvalid and any kind without a
// dedicated struct (including kinds >= KindEnd constructed by rejection
// tests). Its operand count is exactly the constructor's operand count.
type rawNode struct {
	*Base
	operandList
}

// lambdaBearer is implemented by nodes carrying Lambda payloads; the
// verifier walks lambda bodies as graph edges alongside operand uses.
type lambdaBearer interface {
	forEachLambda(fn func(*Lambda))
}

// payloadBearer is implemented by nodes carrying non-operand SSAUse payloads
// (parameter refs, iteration dims/bounds, lambda params, as-orders, store
// targets, ImpactBy effect inputs). The verifier treats payload uses exactly
// like operand slots in both chain directions; Walk follows them as graph
// edges.
type payloadBearer interface {
	forEachPayloadUse(fn func(SSAUse))
}

// allocNode creates the concrete node for kind k with exactly nSlots
// zero-valued operand slots; the ID comes only from g.NewID() and the
// PassOrigin is stamped from origin. Every node gets its SSAVar at
// allocation (1:1, ID == node ID — IR-05's single NewID source, so statement
// numbering can never drift from the node IDs). The kind→struct dispatch is
// the ONE place a new node kind joins the concrete suite (R4a: add the struct
// in its node_xxx.go file, add a catalog.go row, add the case here).
func (g *Graph) allocNode(k Kind, origin string, nSlots int) nodeImpl {
	b := &Base{id: g.NewID(), kind: k, loc: Loc{PassOrigin: origin}, graph: g}
	var n nodeImpl
	switch k {
	case KindConst:
		n = &constNode{Base: b}
	case KindInputParam:
		n = &inputParamNode{Base: b}
	case KindParam:
		n = &paramRefNode{Base: b}
	case KindBinOp:
		n = &binOpNode{Base: b}
	case KindUnaryOp:
		n = &unaryOpNode{Base: b}
	case KindSelect:
		n = &selectNode{Base: b}
	case KindCast:
		n = &castNode{Base: b}
	case KindCall:
		n = &callNode{Base: b}
	case KindIter:
		n = &iterNode{Base: b}
	case KindIterCluster:
		n = &iterClusterNode{Base: b}
	case KindMap:
		n = &mapNode{Base: b}
	case KindReduce:
		n = &reduceNode{Base: b}
	case KindMemLoad:
		n = &memLoadNode{Base: b}
	case KindMemStore:
		n = &memStoreNode{Base: b}
	case KindMemNew:
		n = &memNewNode{Base: b}
	case KindReturn:
		n = &returnNode{Base: b}
	case KindDimVar:
		n = &dimVarNode{Base: b}
	case KindBoundedDimVar:
		n = &boundedDimVarNode{Base: b}
	case KindMemLoadSMem:
		n = &memLoadSMemNode{Base: b}
	case KindMemStoreSMem:
		n = &memStoreSMemNode{Base: b}
	case KindMemNewSMem:
		n = &memNewSMemNode{Base: b}
	case KindFragLoad:
		n = &fragLoadNode{Base: b}
	case KindFragStore:
		n = &fragStoreNode{Base: b}
	case KindFragNew:
		n = &fragNewNode{Base: b}
	case KindForLoop:
		n = &forLoopNode{Base: b}
	default:
		n = &rawNode{Base: b}
	}
	def, _ := n.(Value) // total for in-repo nodes: every concrete node embeds *Base
	b.ssa = &SSAVar{id: b.id, node: n, define: def}
	for i := 0; i < nSlots; i++ {
		n.addSlot()
	}
	g.nodes = append(g.nodes, n)
	return n
}

// wireOperands wires each operand into its slot as an SSAUse handle over a
// fresh ref record in the def's SSAVar, exactly as the AddUse choke point
// does. A nil operand leaves the slot a zero handle (unwired) for the
// verifier to flag. Operands beyond a fixed-arity kind's slot count are NOT
// wired, but they are COUNTED and returned as the surplus so NewNode can
// record them: silent drops are fail-open, and construction in this design
// is fail-closed — the verifier flags any surplus under the use-def
// invariant (WR-01).
func wireOperands(n nodeImpl, operands ...Node) int {
	slots := lenOf(n)
	surplus := 0
	for i, val := range operands {
		if i >= slots {
			surplus++
			continue
		}
		if val == nil || val.BasePtr().ssa == nil {
			continue // leave the zero handle (foreign nodes without a var stay unwired)
		}
		rec := val.BasePtr().ssa.addRef(n)
		n.setSlot(i, SSAUse{rec: rec})
	}
	return surplus
}

// wirePayloadUse links val as a payload (non-operand-slot) reference of
// user, creating a fresh ref record in the def's SSAVar exactly as operand
// wiring does. A nil val (or a foreign node without a var) leaves a zero
// handle for the verifier to flag. The user back-pointer may be nil when the
// owning node does not exist yet (Lambda records); NewMap/NewReduce fix the
// records up to the stage node when attaching the lambda.
func wirePayloadUse(user Node, val Node) SSAUse {
	if val == nil || val.BasePtr().ssa == nil {
		return SSAUse{}
	}
	return SSAUse{rec: val.BasePtr().ssa.addRef(user)}
}

// fixupLambdaUsers retargets a freshly-built lambda record's parameter use
// records at their owning stage node (they were created with a nil user
// before the node existed).
func fixupLambdaUsers(n Node, lam *Lambda) {
	for i := range lam.params {
		if r := lam.params[i].rec; r != nil {
			r.user = n
		}
	}
}

// NewGraph creates an empty named graph in its Level-1 form: a pure sea of
// nodes with NO CFG and no statements. The single-block CFG (Level 2) is
// produced only by Materialize (schedule.go), after construction and passes
// and before codegen — Block()/CFG() return nil until then.
func NewGraph(name string) *Graph {
	return &Graph{name: name, nextID: 1}
}

// ReserveStatementID returns the graph's next monotonic ID and advances the
// counter, letting a builder pin a statement's ID before the statement's RHS
// nodes exist. A fluent builder constructs the statement's chain nodes before
// the statement materializes, which would otherwise push statement IDs behind
// every chain node. Reservation moves the allocation to chain-entry time; IDs
// still come only from the graph's monotonic counter (IR-05) and must be
// consumed by PinStatementID, in construction order.
func (g *Graph) ReserveStatementID() uint64 {
	id := g.nextID
	g.nextID++
	return id
}

// PinStatementID attaches a builder-reserved statement ID (see
// ReserveStatementID) to root, so Materialize creates root's statement with
// exactly that ID instead of drawing from the counter. Level 2's statement
// IDs therefore stay dense across the chain nodes exactly as the pre-scheduler
// flow produced them, which the committed canonical goldens pin byte-for-byte.
// Pinning is the ONLY way a statement ID is reserved: Materialize assigns all
// unpinned effectful nodes from the counter in emission order. A nil root is
// ignored.
func (g *Graph) PinStatementID(root Node, id uint64) {
	if root == nil {
		return
	}
	root.BasePtr().stmtID = id
}

// NewNode is the low-level constructor: it creates a node of ANY kind —
// including KindInvalid and kinds >= KindEnd — with operands wired as Uses.
// It deliberately performs no kind validation (see file comment), but
// operands supplied beyond a fixed-arity kind's slot count are recorded as
// surplus on the node so the verifier can reject the graph (fail-closed,
// WR-01); variadic kinds (call, map, the memory family) and the raw fallback
// wire every operand.
func (g *Graph) NewNode(k Kind, origin string, operands ...Node) Node {
	arity := fixedArity(k)
	nSlots := arity
	if nSlots < 0 {
		nSlots = len(operands)
	}
	n := g.allocNode(k, origin, nSlots)
	if arity >= 0 {
		n.BasePtr().surplus = wireOperands(n, operands...)
	} else {
		wireOperands(n, operands...)
	}
	return n
}
