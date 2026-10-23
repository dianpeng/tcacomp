// Package e2e builds and runs the committed golden kernels on the GPU and
// compares their numerical output against CPU references computed in Go
// (doc/rt.md §8.3; CLAUDE.md §(d) build-integration decision). Everything is
// gated behind TCACOMP_E2E=1 so unit-test runs stay hermetic; fixtures are
// content-addressed under os.TempDir. Stdlib only.
package e2e

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// e2eTimeout bounds every toolchain/exec invocation (cmake configure, build,
// fixture run, sanitizer run).
const e2eTimeout = 5 * time.Minute

// skipUnlessE2E gates every GPU test: TCACOMP_E2E=1 plus a findable cmake and
// nvcc (the ~/.local/bin cmake install may not be on PATH; exec.LookPath
// falls back to the known absolute locations).
func skipUnlessE2E(t TB) {
	t.Helper()
	if os.Getenv("TCACOMP_E2E") != "1" {
		t.Skip("TCACOMP_E2E != 1 — skipping GPU e2e (hermetic unit run)")
	}
	if _, ok := FindTool("cmake", "/home/ps/.local/bin/cmake"); !ok {
		t.Skip("cmake not found on PATH or at the known fallback location")
	}
	if _, ok := FindTool("nvcc", "/usr/local/cuda-12.2/bin/nvcc"); !ok {
		t.Skip("nvcc not found on PATH or at the known fallback location")
	}
}

// TB is the narrow testing-facing interface used by the harness (satisfied by
// *testing.T and *testing.B).
type TB interface {
	Helper()
	Fatalf(format string, args ...any)
	Logf(format string, args ...any)
	Skip(args ...any)
}

// FindTool resolves a tool via PATH, then the given absolute fallbacks.
func FindTool(name string, fallbacks ...string) (string, bool) {
	if p, err := exec.LookPath(name); err == nil {
		return p, true
	}
	for _, f := range fallbacks {
		if st, err := os.Stat(f); err == nil && !st.IsDir() && st.Mode()&0111 != 0 {
			return f, true
		}
	}
	return "", false
}

// FixtureCacheKey hashes the fixture file contents (sorted names) into a
// content-addressed cache directory name under os.TempDir()/tcacomp-e2e.
func FixtureCacheKey(files map[string][]byte) string {
	names := make([]string, 0, len(files))
	for n := range files {
		names = append(names, n)
	}
	sort.Strings(names)
	h := sha256.New()
	for _, n := range names {
		h.Write([]byte(n))
		h.Write([]byte{0})
		h.Write(files[n])
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// BuildFixture materializes the content-addressed fixture, configures and
// builds it with CMake (default Unix Makefiles generator — ninja is absent on
// this machine; NEVER pass -G Ninja), and returns the fixture binary path.
// The cmake+build step is skipped when the binary already exists (a .done
// sentinel makes the completion visible to concurrent readers).
func BuildFixture(t TB, files map[string][]byte, binName, runtimeDir string) string {
	t.Helper()
	dir := filepath.Join(os.TempDir(), "tcacomp-e2e", FixtureCacheKey(files))
	bin := filepath.Join(dir, "build", binName)
	done := filepath.Join(dir, ".done")
	if st, err := os.Stat(done); err == nil && !st.IsDir() {
		if _, err := os.Stat(bin); err == nil {
			return bin // cache hit — the fixture content defines the binary
		}
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("e2e: mkdir fixture %s: %v", dir, err)
	}
	for name, content := range files {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, content, 0o644); err != nil {
			t.Fatalf("e2e: write fixture file %s: %v", p, err)
		}
	}

	cmake, _ := FindTool("cmake", "/home/ps/.local/bin/cmake")
	nvcc, _ := FindTool("nvcc", "/usr/local/cuda-12.2/bin/nvcc")
	buildDir := filepath.Join(dir, "build")
	cfgArgs := []string{"-S", dir, "-B", buildDir,
		fmt.Sprintf("-DCMAKE_CUDA_COMPILER=%s", nvcc),
		fmt.Sprintf("-DTCACOMP_RUNTIME_DIR=%s", runtimeDir)}
	if out, err := RunCmd(t, "cmake (configure)", cmake, cfgArgs...); err != nil {
		t.Fatalf("e2e: cmake configure failed:\n%s", out)
	}
	if out, err := RunCmd(t, "cmake (build)", cmake, "--build", buildDir, "-j8"); err != nil {
		t.Fatalf("e2e: cmake build failed:\n%s", out)
	}
	if _, err := os.Stat(bin); err != nil {
		t.Fatalf("e2e: fixture binary missing after build: %s", bin)
	}
	if err := os.WriteFile(done, []byte("done\n"), 0o644); err != nil {
		t.Fatalf("e2e: write sentinel %s: %v", done, err)
	}
	return bin
}

// RunCmd runs argv with a fixed argument slice (no shell — untrusted input
// never reaches interpolation), captures combined output, and enforces the
// e2e timeout.
func RunCmd(t TB, what, name string, args ...string) (string, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), e2eTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	cmd.Env = os.Environ()
	err := cmd.Run()
	out := buf.String()
	if ctx.Err() == context.DeadlineExceeded {
		return out, fmt.Errorf("%s: timed out after %s", what, e2eTimeout)
	}
	if err != nil {
		return out, fmt.Errorf("%s: %w", what, err)
	}
	return out, nil
}

// ParseFixtureOutput extracts the machine-parseable lines the runners print:
//
//	CANARY <n>
//	CHECKSUM <double>
//	SAMPLES <%a ...>   (first 8 values)
//	OUT <%a ...>       (full output dump, when the runner emits it)
func ParseFixtureOutput(t TB, out string) (canary int, checksum float64, samples []float64, full []float64) {
	t.Helper()
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		switch fields[0] {
		case "CANARY":
			if _, err := fmt.Sscanf(fields[1], "%d", &canary); err != nil {
				t.Fatalf("e2e: malformed CANARY line: %q", line)
			}
		case "CHECKSUM":
			if _, err := fmt.Sscanf(fields[1], "%g", &checksum); err != nil {
				t.Fatalf("e2e: malformed CHECKSUM line: %q", line)
			}
		case "SAMPLES":
			samples = parseHexFloats(t, fields[1:])
		case "OUT":
			full = parseHexFloats(t, fields[1:])
		}
	}
	return canary, checksum, samples, full
}

func parseHexFloats(t TB, fields []string) []float64 {
	t.Helper()
	vals := make([]float64, 0, len(fields))
	for _, f := range fields {
		v, err := strconv.ParseFloat(f, 64)
		if err != nil {
			t.Fatalf("e2e: malformed hex-float %q: %v", f, err)
		}
		vals = append(vals, v)
	}
	return vals
}
