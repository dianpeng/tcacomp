// Package golden implements the shared golden-file test harness (TOOL-05,
// D-11..D-14): one generic implementation serves every output kind (IR text,
// .dot, diagnostics) with the standard -update flow, exact byte comparison,
// and a first-differing-line failure report.
//
// The shape follows the Go standard library's own golden pattern (verified
// in $GOROOT/src/cmd/gofmt/gofmt_test.go): a single package-level -update
// flag registration (Pitfall 7: package initialization runs exactly once per
// test binary, so many importers cannot double-register), fixtures under
// testdata/ (ignored by the go tool), and byte-exact compare (D-13) so stale
// goldens fail loudly in CI, which never passes -update (D-12).
package golden

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"strings"
	"testing"
)

// update is the single -update flag registration for the whole module
// (Pitfall 7). `go test ./... -update` rewrites goldens; CI runs without it.
var update = flag.Bool("update", false, "update golden files")

// Assert compares got against the golden file at path, failing the test on
// mismatch. With -update it rewrites the golden instead (D-12).
func Assert(t *testing.T, path string, got []byte) {
	t.Helper()
	if err := Check(path, got); err != nil {
		t.Fatal(err)
	}
}

// Check is Assert's error-returning core, exported so tests can assert the
// mismatch report itself without intercepting t.Fatal.
func Check(path string, got []byte) error {
	got = normalize(got)
	if *update {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			return fmt.Errorf("update golden %s: %w", path, err)
		}
		return nil
	}
	want, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read golden %s: %v (run with -update)", path, err)
	}
	if !bytes.Equal(want, got) { // D-13: exact byte compare
		return fmt.Errorf("golden mismatch: %s\n%s", path, firstDiff(want, got))
	}
	return nil
}

// normalize collapses trailing newlines to exactly one: golden files end
// with a single newline regardless of how the emitter terminated its last
// line (Pitfall 4 hygiene — trailing whitespace beyond that is significant).
func normalize(b []byte) []byte {
	b = bytes.TrimRight(b, "\n")
	return append(b, '\n')
}

// firstDiff renders a hand-rolled first-differing-line report with up to 3
// lines of preceding context from the golden (a locator, not a full diff —
// RESEARCH "Don't Hand-Roll" table).
func firstDiff(want, got []byte) string {
	wl := bytes.Split(want, []byte("\n"))
	gl := bytes.Split(got, []byte("\n"))

	n := min(len(wl), len(gl))
	at := n // at == n with equal lengths means "prefix-equal"; handle below
	for i := 0; i < n; i++ {
		if !bytes.Equal(wl[i], gl[i]) {
			at = i
			break
		}
	}

	var b strings.Builder
	switch {
	case at == n && len(wl) == len(gl):
		fmt.Fprintf(&b, "no differing line found (lengths %d vs %d)", len(wl), len(gl))
	case at == n:
		fmt.Fprintf(&b, "first difference at end of shorter input (line %d)", n+1)
	default:
		fmt.Fprintf(&b, "first difference at line %d", at+1)
	}

	ctx := 3
	lo := max(0, at-ctx)
	for j := lo; j < at && j < len(wl); j++ {
		fmt.Fprintf(&b, "\n  want %4d| %s", j+1, wl[j])
	}
	if at < len(wl) {
		fmt.Fprintf(&b, "\n  want %4d| %s", at+1, wl[at])
	}
	if at < len(gl) {
		fmt.Fprintf(&b, "\n  got  %4d| %s", at+1, gl[at])
	}
	return b.String()
}
