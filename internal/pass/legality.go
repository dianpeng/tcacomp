package pass

import (
	"fmt"

	"tcacomp/internal/ir"
)

// Stage identifies a pipeline stage: the unit IR-07's legality windows are
// centered on. The order below mirrors the PM-enforced pipeline order in
// ROADMAP.md ("Pipeline Staging").
type Stage int

const (
	StageConstructed   Stage = iota // builder output: abstract + scalar-expr families
	StageLinearized                 // effect-linearize done: no pipes
	StageParallelized               // annotations only
	StageIterLowered                // loop family appears; no raw iter nodes
	StageReduceLowered              // reducers lowered to collective implementations
	StageCacheLowered               // cache tiers bound (SMEM/FRAG)
	StageFinal                      // codegen-ready
)

// String returns the canonical lowercase name of the stage.
func (s Stage) String() string {
	switch s {
	case StageConstructed:
		return "constructed"
	case StageLinearized:
		return "linearized"
	case StageParallelized:
		return "parallelized"
	case StageIterLowered:
		return "iter-lowered"
	case StageReduceLowered:
		return "reduce-lowered"
	case StageCacheLowered:
		return "cache-lowered"
	case StageFinal:
		return "final"
	default:
		return fmt.Sprintf("stage(%d)", int(s))
	}
}

// LegalityAction is the per-Kind verdict inside a stage's legality window,
// adapted from MLIR's ConversionTarget model (Legal / Illegal / Dynamic).
type LegalityAction int

const (
	Legal   LegalityAction = iota // the kind is legal in this window
	Illegal                       // the kind is rejected in this window
	Dynamic                       // the kind is legal only if its predicate passes
)

// LegalityWindow is the set of node kinds legal at one pipeline stage.
// Absence from Actions is the unknown-kind signal and fails CLOSED (IR-03):
// a Kind introduced before its window row lands is illegal, never silently
// legal — including real Phase-1 kinds checked against a stage whose row
// has not been written yet.
type LegalityWindow struct {
	Actions    map[ir.Kind]LegalityAction
	Predicates map[ir.Kind]func(ir.Node) error // consulted for Dynamic kinds
}

// windows is the central stage -> window table (one place, not per-pass).
// Only StageConstructed carries a populated row in Phase 1; later stages
// get rows as Phases 3-4's lowering passes land (RESEARCH Pattern 4).
var windows map[Stage]*LegalityWindow

func init() {
	windows = make(map[Stage]*LegalityWindow)
	windows[StageConstructed] = constructedWindow()
}

// constructedWindow marks EVERY Phase-1 kind (KindIter..KindForLoop — the
// whole abstract, scalar-expr, parameter, and lowered node suite) explicitly
// Legal. The entries are written out one by one so a future kind appended
// before KindEnd is visibly absent here until the window is revised.
func constructedWindow() *LegalityWindow {
	w := &LegalityWindow{
		Actions:    make(map[ir.Kind]LegalityAction),
		Predicates: make(map[ir.Kind]func(ir.Node) error),
	}
	for k := ir.KindIter; k <= ir.KindForLoop; k++ {
		w.Actions[k] = Legal
	}
	return w
}

// WindowFor returns the legality window for stage s. It NEVER returns nil
// or a window with a nil map: a stage with no table row gets a fail-closed
// EMPTY window, so every real kind is rejected there (pinned checker
// revision) — absence from the Actions map is what rejects, not any enum
// zero value.
func WindowFor(s Stage) *LegalityWindow {
	if w, ok := windows[s]; ok && w != nil {
		return w
	}
	return &LegalityWindow{
		Actions:    map[ir.Kind]LegalityAction{},
		Predicates: map[ir.Kind]func(ir.Node) error{},
	}
}

// checkLegality walks every node in g in deterministic order (BB statement
// order, operand order — via ir.Walk, never map iteration) and checks each
// node's Kind against the window for stage s. A Kind is rejected iff it is
// KindInvalid, >= KindEnd, or has NO entry in the window's Actions map.
// Kinds with an entry pass when Legal, fail with the pinned message when
// Illegal, and run their predicate when Dynamic, reporting its error. The
// walk stops at the first violation (ir.Stop) — the first error wins.
func checkLegality(s Stage, g *ir.Graph) error {
	w := WindowFor(s)
	var first error
	ir.Walk(g, ir.VisitorFunc(func(n ir.Node) ir.VisitResult {
		k := n.KindOf()
		act, known := w.Actions[k]
		switch {
		case k == ir.KindInvalid || k >= ir.KindEnd || !known:
			first = fmt.Errorf("stage %s: node %d kind %s illegal in this window",
				s, n.BasePtr().ID(), k)
		case act == Illegal:
			first = fmt.Errorf("stage %s: node %d kind %s illegal in this window",
				s, n.BasePtr().ID(), k)
		case act == Dynamic:
			pred := w.Predicates[k]
			if pred == nil {
				first = fmt.Errorf("stage %s: node %d kind %s dynamic but no predicate registered",
					s, n.BasePtr().ID(), k)
				break
			}
			if err := pred(n); err != nil {
				first = fmt.Errorf("stage %s: node %d kind %s: %w",
					s, n.BasePtr().ID(), k, err)
			}
		}
		if first != nil {
			return ir.Stop
		}
		return ir.Descend
	}))
	return first
}
