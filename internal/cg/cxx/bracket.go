// bracket.go is the R-A bracket-law structural verifier (2026-10-24 ruling,
// R-B golden-audit protocol): a mechanical brace-depth scanner over an
// EMITTED kernel text asserting the reduce bracket's statement order —
//
//	Init's enclosing loop is a for-loop (the output-element loop — X's
//	parent; the innermost element loop for multi-dim outputs, the enclosing
//	single loop at T==1) AND Init is the last statement before X opens;
//	X (the reduced dim's tile-between loop, or the single K loop at T==1)
//	opens immediately after Init; the Reduce call sits inside the
//	INNERMOST loop nested in X (the tile-inside walk Y at T>1) or directly
//	inside X when X carries no nested loop (the T==1 form); the Combine
//	local is the first statement after X's closing brace, at the parent's
//	depth; the gated single-owner store follows with no scope closing
//	below the parent between Combine and the store.
//
// The scanner is deliberately textual (the golden-level R-A proof R-B
// demands): it never consults the IR, so a placement regression in the
// lowering cannot satisfy it by construction. Emitter comments and blank
// lines are skipped; every other line is code with brace bookkeeping.
package cxx

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	bracketInitLine    = regexp.MustCompile(`reducer(\d+)\.Init\(\);`)
	bracketReduceLine  = regexp.MustCompile(`reducer(\d+)\.Reduce\(`)
	bracketCombineLine = regexp.MustCompile(`=\s*reducer(\d+)\.Combine\(\);`)
)

// blCodeLine is one code line of the emitted text with its brace depth at
// the END of the line (statements carry their enclosing scope's depth; a
// `for (...) {` header carries body depth; a lone `}` carries the closed
// scope's enclosing depth).
type blCodeLine struct {
	text  string
	depth int
}

// ScanBracketLaw runs the R-A structural verification over an emitted
// kernel text. It fails closed on every deviation: a missing or duplicated
// bracket, a mis-parented Init, a Reduce outside the innermost reduced-dim
// loop, a Combine not immediately after X's close at the parent depth, or
// a scope close between Combine and the gated store.
func ScanBracketLaw(cu string) error {
	var codes []blCodeLine
	depth := 0
	for _, raw := range strings.Split(cu, "\n") {
		t := strings.TrimSpace(raw)
		if t == "" || strings.HasPrefix(t, "//") {
			continue // emitter comments and blanks are not code
		}
		depth += strings.Count(t, "{") - strings.Count(t, "}")
		codes = append(codes, blCodeLine{t, depth})
	}
	// Exactly one reducer bracket per kernel in v0.
	initIdx := -1
	reducer := ""
	for i, c := range codes {
		m := bracketInitLine.FindStringSubmatch(c.text)
		if m == nil {
			continue
		}
		if initIdx >= 0 {
			return fmt.Errorf("bracket-law: multiple Init calls (%s and %s) — one reducer bracket per kernel in v0", reducer, m[1])
		}
		initIdx, reducer = i, m[1]
	}
	if initIdx < 0 {
		return fmt.Errorf("bracket-law: no reducerN.Init() call in the text")
	}
	d := codes[initIdx].depth // the parent scope's depth (X's parent)

	// (a) Init's enclosing scope is a for-loop body (the output-element
	// loop — X's parent): walking backward over plain sibling statements at
	// depth d (bounds/index locals legitimately precede Init in the same
	// block), the innermost scope opener must be a for header whose body
	// depth == d.
	if initIdx == 0 {
		return fmt.Errorf("bracket-law: Init has no preceding statement — no enclosing output-element loop")
	}
	enclosing := -1
	for i := initIdx - 1; i >= 0; i-- {
		if codes[i].depth < d {
			break // left the parent scope without finding its opener
		}
		if codes[i].depth == d && strings.Contains(codes[i].text, "for (") && strings.Contains(codes[i].text, "{") {
			enclosing = i
			break
		}
		if codes[i].depth == d && codes[i].text == "}" {
			break // a sibling scope closed at the same depth — not our opener
		}
	}
	if enclosing < 0 {
		return fmt.Errorf("bracket-law: Init at depth %d is not enclosed by a for-loop body (no for header opens depth %d around it)", d, d)
	}

	// (b) X opens immediately after Init.
	if initIdx+1 >= len(codes) {
		return fmt.Errorf("bracket-law: nothing follows Init — X missing")
	}
	xHead := codes[initIdx+1]
	if !strings.Contains(xHead.text, "for (") || xHead.depth != d+1 {
		return fmt.Errorf("bracket-law: the statement after Init must be X's for header at depth %d, got %q at depth %d", d+1, xHead.text, xHead.depth)
	}

	// X's close: the first lone brace returning to depth d.
	xClose := -1
	for i := initIdx + 2; i < len(codes); i++ {
		if codes[i].depth <= d {
			if codes[i].text != "}" || codes[i].depth != d {
				return fmt.Errorf("bracket-law: the first scope event after X leaves depth %d at %q — malformed bracket", d+1, codes[i].text)
			}
			xClose = i
			break
		}
	}
	if xClose < 0 {
		return fmt.Errorf("bracket-law: X never closes back to depth %d", d)
	}

	// Loops nested directly in X's body (headers at depth d+2): the T>1
	// form carries exactly one (Y, the tile-inside walk); the T==1 form
	// carries none (X IS the single K loop).
	nested := 0
	for i := initIdx + 2; i < xClose; i++ {
		if strings.Contains(codes[i].text, "for (") && codes[i].depth == d+2 {
			nested++
		}
	}

	// (c) Reduce sits inside the innermost reduced-dim loop.
	redIdx := -1
	for i := initIdx + 2; i < xClose; i++ {
		m := bracketReduceLine.FindStringSubmatch(codes[i].text)
		if m != nil && m[1] == reducer {
			redIdx = i
			break
		}
	}
	if redIdx < 0 {
		return fmt.Errorf("bracket-law: no reducer%s.Reduce( inside X", reducer)
	}
	switch {
	case nested == 0:
		if codes[redIdx].depth != d+1 {
			return fmt.Errorf("bracket-law: the T==1 form puts Reduce directly inside X (depth %d), got depth %d", d+1, codes[redIdx].depth)
		}
	case nested == 1:
		if codes[redIdx].depth < d+2 {
			return fmt.Errorf("bracket-law: Reduce must sit inside Y (depth >= %d), got depth %d", d+2, codes[redIdx].depth)
		}
	default:
		return fmt.Errorf("bracket-law: X carries %d directly nested loops, want exactly Y (1) or the T==1 single form (0)", nested)
	}

	// (d) Combine is the first statement after X's close, at the parent's
	// depth, and the closer is X's brace.
	if xClose+1 >= len(codes) {
		return fmt.Errorf("bracket-law: nothing follows X's close — Combine missing")
	}
	cm := bracketCombineLine.FindStringSubmatch(codes[xClose+1].text)
	if cm == nil || cm[1] != reducer {
		return fmt.Errorf("bracket-law: the first statement after X's close must be the reducer%s Combine local, got %q", reducer, codes[xClose+1].text)
	}
	if codes[xClose+1].depth != d {
		return fmt.Errorf("bracket-law: Combine must sit at X's parent depth %d, got %d", d, codes[xClose+1].depth)
	}
	combineIdx := xClose + 1

	// (e) The gated single-owner store follows; no scope closes below the
	// parent between Combine and the store's Set.
	gateIdx, setIdx := -1, -1
	for i := combineIdx + 1; i < len(codes); i++ {
		if codes[i].depth < d {
			return fmt.Errorf("bracket-law: the bracket's parent scope closed before the gated store (line %q)", codes[i].text)
		}
		if gateIdx < 0 && strings.Contains(codes[i].text, "is_first_in_cta()") {
			gateIdx = i
		}
		if gateIdx >= 0 && strings.Contains(codes[i].text, ".Set(") {
			setIdx = i
			break
		}
		if codes[i].depth == d && codes[i].text == "}" {
			// A plain close at the parent depth after Combine without a
			// store: the bracket ended store-less.
			break
		}
	}
	if setIdx < 0 {
		return fmt.Errorf("bracket-law: no is_first_in_cta()-gated .Set( after Combine")
	}
	return nil
}
