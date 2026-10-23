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
// EPILOGUE CONTRACT (quick-261022-b2m, rulings 6-7): epilogue Groups are no
// longer permanently-empty pads — they receive the AFTER-NEST statements:
// kernel-level consumers of in-loop values or effects (reduction shapes,
// post-nest stores chained after in-loop stores) and the kernel end node,
// which lands in the LAST LNT's epilogue. SchedSN and SchedBlock are
// COMPLEMENTARY on those nodes: SN is the propagated DEPTH REQUIREMENT (0 =
// loop-invariant, prologue/kernel Group; r with a Loop block = in-loop at
// depth r; r with an epilogue Group = after that nest, still requiring
// depth r) while the block is the landing zone. final_lower's reducer rule
// ("using Reducer's own SN to learn its tightest Loop BB — OR using
// reducer's first operand") reads exactly this duality: the reducer's own
// SN names the reduced loop's depth, and its operand's SchedBlock IS that
// loop's block. Every prologue/epilogue Group exposes its owning nest via
// Nest() (nil outside clusters); loop blocks keep Loop() as their nest
// identity.
//
// PROTOTYPE BLOCK (quick-261022-e2u, user ruling 2026-10-22 D-1..D-5):
// live KindInputParam and live KindMemNew (GMEM — the launcher-provided
// buffer, spec 4.2) nodes are placeholders to known states, NOT statements:
// they anchor at SN -1 (PrototypeSN) into a kernel-signature Prototype
// block (BlockPrototype) carried by the dedicated ASTKernel.Prototype field
// and rendered FIRST in the compiled kernel. This makes the SN→lexical-scope
// mapping TOTAL (-1 signature / 0 kernel / r depth-r loop) with zero special
// conditions in the algorithm — propagation's meet is a max, so a -1 dep
// contributes nothing. Params []Node and Gmem []ASTGmemAlloc stay as
// STRUCTURED metadata views (D-5); the prototype block is the canonical
// SCHEDULE view final_lower consumes. MemNewSMem/FragNew allocations and
// extent consts stay NORMALLY scheduled at SN 0 (D-4) — they are in-kernel
// declarations and values, not placeholders.
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
	// BlockPrototype is the kernel-signature block (quick-261022-e2u,
	// D-1/D-2): rendered FIRST in the compiled kernel, its items are the
	// live KindInputParam and KindMemNew placeholder nodes anchored at
	// PrototypeSN. Appended AFTER the loop kinds on purpose — append-only:
	// BlockGroup must stay the zero value.
	BlockPrototype
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
	case BlockPrototype:
		return "Prototype"
	default:
		return fmt.Sprintf("BlockKind(%d)", int(k))
	}
}

// PrototypeSN is the scheduling number of the kernel-signature scope — the
// Prototype block (quick-261022-e2u, D-1). It makes the SN→lexical-scope
// mapping TOTAL:
//
//	-1 = signature scope (the Prototype block: live KindInputParam and
//	     KindMemNew placeholders to known states)
//	 0 = kernel scope (the prologue/kernel Group)
//	 r >= 1 = depth-r loop
//
// Anchoring at -1 needs zero special conditions downstream: SN
// propagation's meet operator is a MAX over deps, so a -1 dep contributes
// nothing (the anchored-set exclusion during propagation/placement is the
// only mechanism — exactly the loop-header discipline).
const PrototypeSN = -1

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
	nest  *ASTLNT   // owning nest of a prologue/epilogue Group (ruling 7, quick-261022-b2m); nil for Loop blocks and Groups outside any cluster
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

// Nest returns the owning loop-nest tree of a prologue/epilogue Group —
// the O(1) back-pointer final_lower uses to map an epilogue-landed node
// back to its producing nest (ruling 7, quick-261022-b2m) — or nil for
// Loop blocks (their nest identity is Loop()) and for Groups outside any
// cluster (the kernel body, the no-loop case). The schedule pass is the
// sole producer: the pointer is set at cluster construction time via
// NewASTGroupInNest and is stable for the kernel's life.
func (b *ASTBlock) Nest() *ASTLNT { return b.nest }

// Items returns a fresh copy of the block's statement items in placement
// order; the receiver's slice is never aliased.
func (b *ASTBlock) Items() []ASTItem { return append([]ASTItem(nil), b.items...) }

// AppendItem appends one statement item (a scheduled node or a nested loop
// block) — the primary construction mutator besides the constructors.
func (b *ASTBlock) AppendItem(it ASTItem) { b.items = append(b.items, it) }

// InsertItemAt inserts one statement item at index, shifting the existing
// items at and after index back (P-6, quick-261022-l1o). Fail-closed bounds:
// a negative index or one beyond the current length (append position
// excluded — index == len appends, mirroring slice-insert semantics) is an
// error leaving the block untouched. final-lower is the first consumer (the
// KernelInfo prepend, the reducer bracket insertions); the utility stays
// generic — no final-lower semantics live here.
func (b *ASTBlock) InsertItemAt(index int, it ASTItem) error {
	if index < 0 || index > len(b.items) {
		return fmt.Errorf("ir: ASTBlock.InsertItemAt index %d out of bounds [0, %d]", index, len(b.items))
	}
	b.items = append(b.items, ASTItem{})
	copy(b.items[index+1:], b.items[index:])
	b.items[index] = it
	return nil
}

// RemoveItemsIf drops every item for which pred reports true, preserving
// the relative order of the survivors, and returns the number removed (P-6,
// quick-261022-l1o). final-lower is the first consumer (the erased-node
// sweep); the utility stays generic.
func (b *ASTBlock) RemoveItemsIf(pred func(ASTItem) bool) int {
	if pred == nil {
		return 0
	}
	kept := b.items[:0]
	removed := 0
	for _, it := range b.items {
		if pred(it) {
			removed++
			continue
		}
		kept = append(kept, it)
	}
	b.items = kept
	return removed
}

// NewASTGroup creates an empty BlockGroup — a kernel body or one of the
// prologue/epilogue pads around a loop-nest cluster (spec step 1.2) — with
// NO owning nest (Nest() returns nil; ruling 7, quick-261022-b2m).
func NewASTGroup() *ASTBlock {
	return &ASTBlock{kind: BlockGroup, dosoc: -1, loc: 1}
}

// NewASTGroupInNest creates an empty BlockGroup whose Nest() back-pointer
// is lnt (ruling 7, quick-261022-b2m): ONE loop-nest cluster's LEXICAL
// Group (quick-261023-lex) — the head region (prologue items), the embedded
// LBBN root, and the after-nest (epilogue) items all live as items of this
// ONE Group — so final_lower can recover the owning nest of any node in the
// cluster in O(1) and codegen's C++ scoping is correct (a sibling prologue
// Group would close before the nest opens). The schedule pass is the sole
// producer.
func NewASTGroupInNest(lnt *ASTLNT) *ASTBlock {
	return &ASTBlock{kind: BlockGroup, dosoc: -1, loc: 1, nest: lnt}
}

// NewASTPrototype creates the empty kernel-signature Prototype block
// (quick-261022-e2u, D-2): rendered FIRST in the compiled kernel — the
// dedicated ASTKernel.Prototype field is its carrier, and the block is NOT
// an item inside Body. Its items are the live KindInputParam and KindMemNew
// placeholder nodes, each stamped SetSched(PrototypeSN, proto) by the
// schedule pass. Same defaults as NewASTGroup (dosoc -1, loc 1, Nest()
// nil).
func NewASTPrototype() *ASTBlock {
	return &ASTBlock{kind: BlockPrototype, dosoc: -1, loc: 1}
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
// materialization class would emit the wrong loop kind downstream. Since
// quick-261022-b2m (ME-02) ALL THREE keys fail closed on a non-const value:
// a corrupted annotation must never be masked into the documented default.
// Non-KeyedAttribute trailing operands are skipped (the loopAnnotations
// precedent, internal/opt/tiling/lower.go).
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
			v, ok := ConstValueOf(val)
			if !ok {
				// ME-02 (quick-261022-b2m): a corrupted dosoc annotation must
				// fail closed exactly like soc — silently dropping it would
				// leave the block at the default and propagate that default
				// into ASTLoopMeta, masking the corruption from final_lower's
				// per-LNT runtime mapping and reducer specialization.
				return nil, fmt.Errorf("ir: for-loop node %d carries a non-const dosoc annotation", loop.BasePtr().ID())
			}
			blk.dosoc = int(v)
		case "loc":
			v, ok := ConstValueOf(val)
			if !ok {
				// ME-02 (quick-261022-b2m): same fail-closed rule as soc/dosoc.
				return nil, fmt.Errorf("ir: for-loop node %d carries a non-const loc annotation", loop.BasePtr().ID())
			}
			blk.loc = int(v)
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

// ASTKernel is the scheduled kernel (spec "Scheduling the CFG"): the
// Prototype is the kernel-signature block rendered FIRST (quick-261022-e2u,
// D-2); the Body is always a BlockGroup ("Body: Which is a Group") whose
// items are, per LNT in forest-root allocation order, that LNT's LEXICAL
// Group (quick-261023-lex): [head items (the old prologue region), the
// embedded LBBN root block, after-nest items (the old epilogue region)] —
// there is no separate prologue/tail block; the prototype is NOT an item
// inside Body;
// Params are the kernel input params in declaration order; Gmem lists every
// live GMEM allocation in allocation order; LNTs lists the per-LNT loop
// metadata in forest-root allocation order. Since quick-261022-b2m (ruling
// 7) LNTs is a []*ASTLNT — the schedule pass is the sole producer and every
// prologue/epilogue Group points at its stable LNT record through Nest(),
// so consumers can compare Group identity and nest identity
// interchangeably (final_lower maps an epilogue-landed node back to its
// owning nest).
type ASTKernel struct {
	// Prototype is the kernel-signature block (quick-261022-e2u, D-2),
	// rendered FIRST: one BlockPrototype whose items are exactly the live
	// KindInputParam and KindMemNew placeholder nodes at PrototypeSN — the
	// canonical SCHEDULE view of the kernel signature (D-5). Params and
	// Gmem below stay as STRUCTURED metadata views of the same facts:
	// Params keeps declaration order; Gmem pairs each MemNew with its
	// extent (op[0]) for final_lower's launcher-buffer materialization.
	Prototype *ASTBlock      // the kernel-signature block — always a BlockPrototype
	Body      *ASTBlock      // the kernel body — always a BlockGroup
	Params    []Node         // kernel input params, declaration order (metadata view, D-5)
	Gmem      []ASTGmemAlloc // every live GMEM allocation, allocation order, extent paired (metadata view, D-5)
	LNTs      []*ASTLNT      // per-LNT SoC metadata, forest-root allocation order (stable records; Nest() points here)
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
// nil graph, nil kernel, a nil Prototype, a non-BlockPrototype Prototype
// (quick-261022-e2u, D-2 — validated FIRST), nil Body, a non-Group Body
// (the spec: "Body: Which is a Group"), or an already-attached AST (one
// schedule per graph — re-runs are a manager-level re-run of the whole
// pipeline, not an in-place reschedule) all refuse with an error and leave
// the graph untouched.
func (g *Graph) AttachAST(k *ASTKernel) error {
	if g == nil {
		return errors.New("ir: AttachAST on a nil graph")
	}
	if k == nil {
		return errors.New("ir: AttachAST with a nil kernel")
	}
	if k.Prototype == nil {
		return errors.New("ir: AttachAST with a nil Prototype")
	}
	if k.Prototype.Kind() != BlockPrototype {
		return fmt.Errorf("ir: AttachAST requires a BlockPrototype Prototype, got %s", k.Prototype.Kind())
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

// SchedSN returns the node's scheduling number (quick-261021-ta3 D-03;
// semantics extended by quick-261022-b2m ruling 6 and quick-261022-e2u
// D-1): the propagated DEPTH REQUIREMENT, complementary to the landing
// zone SchedBlock — -1 means signature scope (a placeholder: live
// KindInputParam/KindMemNew landed in the Prototype block), 0 means
// loop-invariant (prologue/kernel Group), r > 0 means "requires depth r".
// With a Loop block that reads "in-loop at depth r"; with an epilogue
// Group it reads "AFTER that nest, still requiring depth r" — never an
// after-nest rank. That complementarity is exactly final_lower's reducer
// dual route: a Reduce lands in its nest's epilogue with SN == the reduced
// loop's depth, and its first operand's SchedBlock IS that loop's block —
// either route identifies the loop. Pass state: nil-safe at the Base level
// (the zero value), never printed/serialized/verified.
func (b *Base) SchedSN() int { return b.schedSN }

// SchedBlock returns the AST block the schedule pass landed the node in —
// the LANDING ZONE, complementary to SchedSN's depth requirement (ruling 6,
// quick-261022-b2m) — or nil while unscheduled; that is how final_lower
// tells leftovers from placed nodes (D-03: the linkage persists on the
// node, readable after the pass). Pass state, never
// printed/serialized/verified.
func (b *Base) SchedBlock() *ASTBlock { return b.schedBlock }

// SetSched stamps the node's persistent scheduling state — SN and landing
// block together (D-03), so the two can never disagree at placement time.
// It is the single writer both for the pass's anchoring/propagation phase
// (block nil — SN is provisional depth state) and for its placement phase
// (block stamped — epilogue nodes KEEP their propagated SN; ruling 6). A
// zero SN is the legitimate rank-0 prologue; -1 is the legitimate
// signature-scope rank (PrototypeSN — the SN floor since quick-261022-e2u);
// a nil block means "not scheduled yet".
func (b *Base) SetSched(sn int, blk *ASTBlock) {
	b.schedSN = sn
	b.schedBlock = blk
}
