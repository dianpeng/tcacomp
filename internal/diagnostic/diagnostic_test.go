package diagnostic

import (
	"strings"
	"testing"
)

func TestSeverityNames(t *testing.T) {
	cases := []struct {
		sev  Severity
		want string
	}{
		{Error, "error"},
		{Warning, "warning"},
		{Remark, "remark"},
		{Severity(99), "unknown"},
	}
	for _, c := range cases {
		if got := c.sev.String(); got != c.want {
			t.Errorf("Severity(%d).String() = %q, want %q", int(c.sev), got, c.want)
		}
	}
}

func TestBagErrNonNilOnlyWithErrorSeverity(t *testing.T) {
	var b Bag
	if err := b.Err(); err != nil {
		t.Fatalf("empty bag: Err() = %v, want nil", err)
	}

	b.Add(Diagnostic{Severity: Warning, Message: "suspicious"})
	b.Add(Diagnostic{Severity: Remark, Message: "note"})
	if err := b.Err(); err != nil {
		t.Fatalf("warnings/remarks only: Err() = %v, want nil", err)
	}

	b.Add(Diagnostic{Severity: Error, Message: "broken"})
	if err := b.Err(); err == nil {
		t.Fatal("with an error-severity diagnostic: Err() = nil, want non-nil")
	} else if !strings.Contains(err.Error(), "broken") {
		t.Fatalf("Err() = %q, want it to contain the error message", err.Error())
	}
}

func TestDiagnosticsReturnsInsertionOrderCopy(t *testing.T) {
	var b Bag
	b.Add(Diagnostic{Severity: Error, Message: "first"})
	b.Add(Diagnostic{Severity: Warning, Message: "second"})

	ds := b.Diagnostics()
	if len(ds) != 2 || ds[0].Message != "first" || ds[1].Message != "second" {
		t.Fatalf("Diagnostics() = %v, want [first second] in order", ds)
	}
	ds[0].Message = "mutated"
	if b.Diagnostics()[0].Message != "first" {
		t.Fatal("Diagnostics() does not return a copy: bag was mutated through the result")
	}
}

func TestStringRendersSpanMessageAndContext(t *testing.T) {
	d := Diagnostic{
		Severity: Error,
		Span:     Span{File: "kernel.tca", Line: 3, Col: 7},
		Message:  "shape mismatch",
		Context:  []string{"expected: tensor<f32, [M, N]>", "actual: tensor<f32, [M]>"},
	}
	got := d.String()
	want := "kernel.tca:3:7: error: shape mismatch\n" +
		"  expected: tensor<f32, [M, N]>\n" +
		"  actual: tensor<f32, [M]>"
	if got != want {
		t.Fatalf("String() =\n%q\nwant\n%q", got, want)
	}
}

func TestStringWithoutSpanOmitsLocation(t *testing.T) {
	d := Diagnostic{Severity: Warning, Message: "no location"}
	if got, want := d.String(), "warning: no location"; got != want {
		t.Fatalf("String() = %q, want %q", got, want)
	}
}
