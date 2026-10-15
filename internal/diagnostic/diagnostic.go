package diagnostic

import (
	"errors"
	"fmt"
	"strings"
)

// Severity ranks a diagnostic's impact (TOOL-01, REQUIREMENTS order):
// Error fails compilation, Warning reports a likely problem, Remark is
// informational. The zero value is Error so an unranked diagnostic cannot
// silently pass as benign.
type Severity int

const (
	// Error fails compilation; Warning is a likely problem; Remark is a note.
	Error Severity = iota
	Warning
	Remark
)

// String returns the lowercase canonical severity name.
func (s Severity) String() string {
	switch s {
	case Error:
		return "error"
	case Warning:
		return "warning"
	case Remark:
		return "remark"
	default:
		return "unknown"
	}
}

// Span is a half-open source range [Line:Col, EndLine:EndCol). The zero Span
// means "unknown location" (IR-06: spans stay zero until the frontend lands,
// so diagnostics must render sensibly without one).
type Span struct {
	File            string
	Line, Col       int
	EndLine, EndCol int
}

// known reports whether the span carries any location information.
func (s Span) known() bool {
	return s.File != "" || s.Line != 0 || s.Col != 0 || s.EndLine != 0 || s.EndCol != 0
}

// Diagnostic is one structured compiler message (TOOL-01): a severity, an
// optional source span, a human message, and ordered structured context
// lines (doc/syntax.md §10 style: "expected: ...", "actual: ...").
type Diagnostic struct {
	Severity Severity
	Span     Span
	Message  string
	Context  []string
}

// String renders the canonical one-paragraph form:
//
//	<file>:<line>:<col>: <severity>: <message>
//	  <context line 1>
//	  <context line 2>
//
// The location prefix is omitted when the span is unknown; context lines
// are indented two spaces.
func (d Diagnostic) String() string {
	var b strings.Builder
	if d.Span.known() {
		fmt.Fprintf(&b, "%s:%d:%d: ", d.Span.File, d.Span.Line, d.Span.Col)
	}
	fmt.Fprintf(&b, "%s: %s", d.Severity, d.Message)
	for _, c := range d.Context {
		b.WriteString("\n  ")
		b.WriteString(c)
	}
	return b.String()
}

// Bag accumulates diagnostics in insertion order (TOOL-01). The zero Bag is
// ready to use.
type Bag struct {
	diags []Diagnostic
}

// Add appends d to the bag.
func (b *Bag) Add(d Diagnostic) {
	b.diags = append(b.diags, d)
}

// Diagnostics returns a copy of the accumulated diagnostics in insertion
// order; mutating the result does not affect the bag.
func (b *Bag) Diagnostics() []Diagnostic {
	return append([]Diagnostic(nil), b.diags...)
}

// Err returns a non-nil error iff the bag holds at least one Error-severity
// diagnostic; warnings and remarks alone never fail compilation. The error
// text is the rendered form of every error-severity diagnostic, joined by
// newlines.
func (b *Bag) Err() error {
	var msgs []string
	for _, d := range b.diags {
		if d.Severity == Error {
			msgs = append(msgs, d.String())
		}
	}
	if len(msgs) == 0 {
		return nil
	}
	return errors.New(strings.Join(msgs, "\n"))
}
