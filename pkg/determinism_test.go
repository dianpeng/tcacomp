package pkg_test

import (
	"bytes"
	"testing"

	"tcacomp/internal/ir"
	"tcacomp/internal/visualization"
)

// TestDeterminism (VER-08) builds the doc-pinned oracle twice through two
// fresh builders and byte-compares the canonical prints. The printer's
// ordering guarantees (D-01..D-04: statement order, chain stage order,
// operand slot order, no map iteration) make this a pure function of the
// construction sequence; the committed golden in TestMatmulScaleGolden is
// the cross-process anchor, this test is the same-process tripwire. Run
// with -count=10 to trip ordering that only surfaces across repeated
// builds (e.g. interning caches keyed by map iteration).
func TestDeterminism(t *testing.T) {
	g1 := buildMatmulScale(t)
	g2 := buildMatmulScale(t)
	p1 := ir.Print(g1)
	p2 := ir.Print(g2)
	if p1 != p2 {
		t.Fatalf("Print is not deterministic across two fresh builds:\n--- build 1 ---\n%s\n--- build 2 ---\n%s", p1, p2)
	}
}

// TestDeterminismDot (VER-08) extends the byte-stability check to the DOT
// visualization emitter, which shares the printer's numbering and ordering
// (D-05): two independently built oracles must render identical .dot text.
func TestDeterminismDot(t *testing.T) {
	g1 := buildMatmulScale(t)
	g2 := buildMatmulScale(t)
	var d1, d2 bytes.Buffer
	if err := visualization.EmitDot(&d1, g1); err != nil {
		t.Fatalf("EmitDot build 1: %v", err)
	}
	if err := visualization.EmitDot(&d2, g2); err != nil {
		t.Fatalf("EmitDot build 2: %v", err)
	}
	if !bytes.Equal(d1.Bytes(), d2.Bytes()) {
		t.Fatalf("EmitDot is not deterministic across two fresh builds:\n--- build 1 ---\n%s\n--- build 2 ---\n%s", d1.String(), d2.String())
	}
}
