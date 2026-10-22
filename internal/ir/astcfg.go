package ir

// astcfg.go is the AST-style CFG product of the final optimization pass
// (quick-261021-ta3, doc/optana/schedule.md "AST style CFG"): the
// sea-of-nodes' second, STRUCTURAL view of the kernel, consumed lastly by
// final_lower and code generation (doc/optana/final_lower.md). It coexists
// with — and never touches — the legacy single-block CFG, Materialize, the
// canonical printer, and every golden (locked decision D-01): the AST CFG
// lives in its OWN Graph field (g.ast, see node.go), is nil until the
// "schedule" pass attaches it, and is invisible to Level-2 machinery.
//
// The shape follows the spec recursively:
//
//	ASTKernel            — one kernel: signature params, GMEM allocations,
//	                       per-LNT loop metadata, and a Body that is always
//	                       a BlockGroup ("Body: Which is a Group").
//	ASTBlock (Group|Loop) — the spec's Block: a Group is "an arbitrary group
//	                       of statements recursively" (prologue/epilogue
//	                       landing pads and kernel body); a Loop carries the
//	                       final-loop materialization header (iv, bound,
//	                       dosoc, loc) and is classified Seq/CTA/Thr from
//	                       the loop's soc KeyedAttribute (0/1/2).
//	ASTItem              — the spec's "Statement is recursively defined: it
//	                       can be a single real statement OR another Block":
//	                       exactly one of Node / Block is non-nil.
//
// Persistent per-node scheduling state (locked decision D-03): alongside
// these types, Base carries schedSN/schedBlock — the scheduling number and
// the SN→block linkage final_lower (and reducer lowering) reads AFTER the
// pass. Same discipline as the ty pass-state cell: never printed, never
// serialized, never verified — a schedule is not a chain invariant.
//
// This package deliberately knows NOTHING about opt/tiling: per-LNT runtime
// mapping construction data stays in the preserved "tiling-record" cache
// entry (the schedule pass preserves it; final_lower fetches it — importing
// tiling here would invert the dependency).

import (
	"errors"
	"fmt"
)

// BlockKind classifies an ASTBlock: the Group kind (arbitrary statement
// group — prologue/epilogue/kernel body) and the three Loop kinds from the
// loop's soc attribute (doc/optana/schedule.md "Block": (1) CTALoop,
// (2) ThrLoop, (3) SeqLoop; the soc KeyedAttribute encodes 0=seq, 1=cta,
// 2=thr — node_loop.go M1).
type BlockKind int

const (
	// BlockGroup is an arbitrary recursively-nested statement group: the
	// kernel body, and the prologue/epilogue pads around every loop nest
	// (spec: "Mostly inserted to include runtime object initialization or
	// LICM landing"; "All LNT will have a header block, Group").
	BlockGroup BlockKind = iota
	// BlockSeqLoop is a sequential loop (soc 0 — the zero-value default,
	// matching tiling.SoCType: a loop with no soc metadata schedules as
	// sequential; this is the deliberate fail-soft default for kernels
	// with no LNT metadata and for unit-test graphs).
	BlockSeqLoop
	// BlockCTALoop is a CTA-parallel loop (soc 1).
	BlockCTALoop
	// BlockThrLoop is a thread-parallel loop (soc 2).
	BlockThrLoop
)

// String returns the diagnostic name of the block kind.
func (k BlockKind) String() string {
	switch k {
	case BlockGroup:
		return "Group"
	case BlockSeqLoop:
		return "SeqLoop"
	case BlockCTALoop:
		return "CTALoop"
	case BlockThrLoop:
		return "ThrLoop"
	default:
		return fmt.Sprintf("BlockKind(%d)", int(k))
	}
}

// ASTItem is one statement slot of an ASTBlock — the spec's recursive
// statement: exactly one field is non-nil. Node carries a scheduled IR node
// (a statement of the block); Block carries a nested loop block.
type ASTItem struct {
	Node  Node      // a scheduled statement; nil when Block is set
	Block *ASTBlock // a nested loop block; nil when Node is set
}

// ASTBlock is one AST-style CFG block (the spec's Block). The fields are
// unexported; the read accessors are the only face. items is append-only
// via AppendItem — immutable-style construction, deterministic order.
type ASTBlock struct {
	kind  BlockKind // Group vs Seq/CTA/ThrLoop
	loop  Node      // the KindForLoop this Loop block materializes; nil for Groups
	iv    Node      // loop header info, ops[0] (induction variable); nil for Groups
	bound Node      // loop header info, ops[1] (upper bound); nil for Groups
	dosoc int       // the loop's dosoc map dimension; -1 when absent
	loc   int       // level of concurrency; 1 for sequential loops (and Groups)
	items []ASTItem // statements and nested loop blocks, placement order
}

// Kind returns the block's kind: Group or the Seq/CTA/ThrLoop class.
func (b *ASTBlock) Kind() BlockKind { return b.kind }

// Loop returns the KindForLoop node this Loop block materializes, or nil
// for a Group.
func (b *ASTBlock) Loop() Node { return b.loop }

// IV returns the loop header's induction-variable operand (ops[0]), or nil
// for a Group.
func (b *ASTBlock) IV() Node { return b.iv }

// Bound returns the loop header's upper-bound operand (ops[1]), or nil for
// a Group.
func (b *ASTBlock) Bound() Node { return b.bound }

// DoSoC returns the loop's dosoc map dimension, or -1 when the loop
// carries no dosoc annotation.
func (b *ASTBlock) DoSoC() int { return b.dosoc }

// LoC returns the loop's level of concurrency (1 for sequential loops and
// for Groups).
func (b *ASTBlock) LoC() int { return b.loc }

// Items returns a fresh copy of the block's statement items in placement
// order; the receiver's slice is never aliased.
func (b *ASTBlock) Items() []ASTItem { return append([]ASTItem(nil), b.items...) }

// AppendItem appends one statement item (a scheduled node or a nested loop
// block) — the only mutator besides the constructors.
func (b *ASTBlock) AppendItem(it ASTItem) { b.items = append(b.items, it) }

// NewASTGroup creates an empty BlockGroup — a kernel body or one of the
// prologue/epilogue pads around a loop-nest cluster (spec step 1.2).
func NewASTGroup() *ASTBlock {
	return &ASTBlock{kind: BlockGroup, dosoc: -1, loc: 1}
}

// NewASTLoop builds a Loop block from a live KindForLoop node (the LBBN
// construction of spec step 1.1, "mapping specific LIR loop node to
// corresponding Loop BB"). It reads the loop header via OperandsOf — the
// iv/bound are ops[0]/ops[1] (node_loop.go M1) — and classifies the block
// kind from the trailing KeyedAttribute operands (ops[3:]) by key: "soc"
// via KeyedAttributeOf + ConstValueOf (0 → BlockSeqLoop, 1 → BlockCTALoop,
// 2 → BlockThrLoop; an absent soc defaults to BlockSeqLoop — the deliberate
// fail-soft zero-value convention, the same default tiling.SoCType uses, so
// kernels with no LNT metadata and unit-test graphs schedule as sequential
// loops), "dosoc" (absent → -1), "loc" (absent → 1). A soc value outside
// {0,1,2} is a fail-closed error: silently scheduling an unknown
// materialization class would emit the wrong loop kind downstream.
// Non-KeyedAttribute trailing operands and non-const attribute values are
// skipped (the loopAnnotations precedent, internal/opt/tiling/lower.go).
func NewASTLoop(loop Node) (*ASTBlock, error) {
	if loop == nil || loop.BasePtr() == nil {
		return nil, errors.New("ir: NewASTLoop on a nil node")
	}
	if loop.KindOf() != KindForLoop {
		return nil, fmt.Errorf("ir: NewASTLoop requires a KindForLoop node, got %s (node %d)",
			loop.KindOf(), loop.BasePtr().ID())
	}
	ops := OperandsOf(loop)
	if len(ops) < 3 {
		return nil, fmt.Errorf("ir: for-loop node %d carries %d operands, want >= 3 (iv, bound, body)",
			loop.BasePtr().ID(), len(ops))
	}
	blk := &ASTBlock{
		kind:  BlockSeqLoop, // the fail-soft zero-value default (no soc metadata)
		loop:  loop,
		iv:    ops[0],
		bound: ops[1],
		dosoc: -1,
		loc:   1,
	}
	for _, attr := range ops[3:] {
		if attr == nil || attr.KindOf() != KindKeyedAttribute {
			continue
		}
		key, val := KeyedAttributeOf(attr)
		switch key {
		case "soc":
			v, ok := ConstValueOf(val)
			if !ok {
				return nil, fmt.Errorf("ir: for-loop node %d carries a non-const soc annotation", loop.BasePtr().ID())
			}
			switch int(v) {
			case 0:
				blk.kind = BlockSeqLoop
			case 1:
				blk.kind = BlockCTALoop
			case 2:
				blk.kind = BlockThrLoop
			default:
				return nil, fmt.Errorf("ir: for-loop node %d carries unknown soc %v (want 0=seq, 1=cta, 2=thr)",
					loop.BasePtr().ID(), v)
			}
		case "dosoc":
			if v, ok := ConstValueOf(val); ok {
				blk.dosoc = int(v)
			}
		case "loc":
			if v, ok := ConstValueOf(val); ok {
				blk.loc = int(v)
			}
		}
	}
	return blk, nil
}

// ASTGmemAlloc is one GMEM allocation of the kernel (spec 4.2): the
// KindMemNew node plus its extent value (op[0]). Every GMEM temporary gets
// an externally-provided launcher buffer at codegen time — after
// compilation the full allocation-site list is known, types included.
type ASTGmemAlloc struct {
	New    Node // the KindMemNew allocation node
	Extent Node // its extent value (op[0])
}

// ASTLoopMeta is one loop's SoC materialization metadata (spec 4.3[2] —
// consumed by reducer lowering: a reducer lowers differently under a
// SEQ/WARP/GLOBAL loop). Soc/Dosoc/LoC mirror the block classification
// (0=seq, 1=cta, 2=thr).
type ASTLoopMeta struct {
	Loop  Node // the KindForLoop node
	Soc   int  // 0=seq, 1=cta, 2=thr
	Dosoc int  // the loop's dosoc map dimension; -1 when absent
	Loc   int  // level of concurrency; 1 for sequential loops
}

// ASTLNT is one loop-nest tree's metadata (spec 4.3): the nest root plus
// its loops' SoC info outer→inner. The per-LNT runtime-mapping
// construction data lives separately in the preserved "tiling-record"
// cache entry — final_lower fetches it; ir does NOT import opt/tiling.
type ASTLNT struct {
	Root  Node          // the nest's root (outermost) KindForLoop
	Loops []ASTLoopMeta // outer→inner per-loop SoC metadata
}

// ASTKernel is the scheduled kernel (spec "Scheduling the CFG"): the Body
// is always a BlockGroup ("Body: Which is a Group") whose items are, per
// LNT in forest-root allocation order, [prologue Group, LBBN root block,
// epilogue Group]; Params are the kernel input params in declaration order;
// Gmem lists every live GMEM allocation in allocation order; LNTs lists the
// per-LNT loop metadata in forest-root allocation order.
type ASTKernel struct {
	Body   *ASTBlock      // the kernel body — always a BlockGroup
	Params []Node         // kernel input params, declaration order
	Gmem   []ASTGmemAlloc // every live GMEM allocation, allocation order
	LNTs   []ASTLNT       // per-LNT SoC metadata, forest-root allocation order
}

// AST returns the graph's AST-style CFG — the "schedule" pass's product
// (quick-261021-ta3, D-01) — or nil while unattached. It is a SEPARATE
// field from the legacy Level-2 cfg: Materialize never sets it, and Level-2
// machinery (Block/CFG/printer/goldens) never reads it.
func (g *Graph) AST() *ASTKernel {
	if g == nil {
		return nil
	}
	return g.ast
}

// AttachAST installs k as the graph's AST-style CFG. It is fail-closed: a
// nil graph, nil kernel, nil Body, a non-Group Body (the spec: "Body:
// Which is a Group"), or an already-attached AST (one schedule per graph —
// re-runs are a manager-level re-run of the whole pipeline, not an in-place
// reschedule) all refuse with an error and leave the graph untouched.
func (g *Graph) AttachAST(k *ASTKernel) error {
	if g == nil {
		return errors.New("ir: AttachAST on a nil graph")
	}
	if k == nil {
		return errors.New("ir: AttachAST with a nil kernel")
	}
	if k.Body == nil {
		return errors.New("ir: AttachAST with a nil Body")
	}
	if k.Body.Kind() != BlockGroup {
		return fmt.Errorf("ir: AttachAST requires a BlockGroup Body, got %s", k.Body.Kind())
	}
	if g.ast != nil {
		return errors.New("ir: AST CFG already attached")
	}
	g.ast = k
	return nil
}

// SchedSN returns the node's scheduling number (quick-261021-ta3, D-03): the
// rank of the AST block the schedule pass landed it in — 0 is the
// legitimate prologue/kernel-body rank (loop-invariant by construction), a
// positive rank r maps to the depth-r loop block of the owning LNT. Pass
// state: nil-safe at the Base level (the zero value), never
// printed/serialized/verified.
func (b *Base) SchedSN() int { return b.schedSN }

// SchedBlock returns the AST block the schedule pass landed the node in, or
// nil while unscheduled — that is how final_lower tells leftovers from
// placed nodes (D-03: the linkage persists on the node, readable after the
// pass). Pass state, never printed/serialized/verified.
func (b *Base) SchedBlock() *ASTBlock { return b.schedBlock }

// SetSched stamps the node's persistent scheduling state — SN and landing
// block together (D-03), so the two can never disagree. It is the single
// writer both for the pass's anchoring/propagation phase (block nil) and
// for its placement phase (block stamped). A zero SN is the legitimate
// rank-0 prologue; a nil block means "not scheduled yet".
func (b *Base) SetSched(sn int, blk *ASTBlock) {
	b.schedSN = sn
	b.schedBlock = blk
}
