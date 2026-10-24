package syntax

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"tcacomp/internal/diagnostic"
	"tcacomp/internal/syntax/ast"
	"tcacomp/internal/syntax/lexer"
)

// Loader tests (design §6 Task 5): include resolution, cycle detection, the
// T-k2i-01 path policy, runtime inheritance, and module-scope checks run
// across files. Fixtures are written to t.TempDir; none of these tests
// touches the network or the GPU.

// identityKernel is the smallest semacheck-clean kernel for fixtures.
const identityKernel = `kernel %s(t: tensor<dtype=f32, shape=[M, N]>) -> tensor<dtype=f32, shape=[M, N]> {
  dimvar (a, b);
  let out = map on(t[a][b]) (x: f32) -> f32: x;
  return out;
}`

func sprintfKernel(name string) string { return fmt.Sprintf(identityKernel, name) }

// writeFixtures writes name→src files into dir and returns dir.
func writeFixtures(t *testing.T, dir string, files map[string]string) string {
	t.Helper()
	for name, src := range files {
		full := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// loadFixtures runs the loader on dir/root.tca and returns the module and
// all diagnostics.
func loadFixtures(t *testing.T, dir string) (*ast.Module, []diagnostic.Diagnostic) {
	t.Helper()
	fset := lexer.NewFileSet()
	m, diags := loadModule(fset, filepath.Join(dir, "root.tca"))
	return m, diags
}

func errsOf(diags []diagnostic.Diagnostic) []diagnostic.Diagnostic {
	var errs []diagnostic.Diagnostic
	for _, d := range diags {
		if d.Severity == diagnostic.Error {
			errs = append(errs, d)
		}
	}
	return errs
}

func wantErr(t *testing.T, diags []diagnostic.Diagnostic, want string) {
	t.Helper()
	errs := errsOf(diags)
	for _, e := range errs {
		if strings.Contains(e.Message, want) {
			return
		}
	}
	t.Fatalf("no error mentioning %q in %v", want, errs)
}

func wantNoErrs(t *testing.T, diags []diagnostic.Diagnostic) {
	t.Helper()
	if errs := errsOf(diags); len(errs) != 0 {
		t.Fatalf("errors = %v, want 0", errs)
	}
}

// TestModuleIncludeBasics: a root includes one file; the included kernel
// lands after the root's kernels; module-scope checks see both.
func TestModuleIncludeBasics(t *testing.T) {
	dir := writeFixtures(t, t.TempDir(), map[string]string{
		"root.tca":   "@include \"common.tca\";\n" + sprintfKernel("main"),
		"common.tca": sprintfKernel("helper"),
	})
	m, diags := loadFixtures(t, dir)
	wantNoErrs(t, diags)
	if len(m.Kernels) != 2 {
		t.Fatalf("kernels = %d, want 2 (root first, included after)", len(m.Kernels))
	}
	if m.Kernels[0].Name != "main" || m.Kernels[1].Name != "helper" {
		t.Fatalf("kernel order = %s, %s", m.Kernels[0].Name, m.Kernels[1].Name)
	}
}

// TestModuleDupKernelsAcrossFiles: rule 1 sees through includes.
func TestModuleDupKernelsAcrossFiles(t *testing.T) {
	dir := writeFixtures(t, t.TempDir(), map[string]string{
		"root.tca":   "@include \"common.tca\";\n" + sprintfKernel("dup"),
		"common.tca": sprintfKernel("dup"),
	})
	_, diags := loadFixtures(t, dir)
	wantErr(t, diags, "duplicate kernel name")
}

// TestModuleIncludeCycle: root→a→b→a is rejected, naming the cycle.
func TestModuleIncludeCycle(t *testing.T) {
	dir := writeFixtures(t, t.TempDir(), map[string]string{
		"root.tca": "@include \"a.tca\";",
		"a.tca":    "@include \"b.tca\";",
		"b.tca":    "@include \"a.tca\";",
	})
	_, diags := loadFixtures(t, dir)
	wantErr(t, diags, "cycle")
}

// TestModuleMissingInclude: a stat failure is a clean diagnostic, not a crash.
func TestModuleMissingInclude(t *testing.T) {
	dir := writeFixtures(t, t.TempDir(), map[string]string{
		"root.tca": "@include \"nope.tca\";",
	})
	_, diags := loadFixtures(t, dir)
	wantErr(t, diags, "nope.tca")
}

// TestModuleIncludePathPolicy: absolute paths and .. escapes are rejected
// (T-k2i-01: includes resolve within the includer's directory only).
func TestModuleIncludePathPolicy(t *testing.T) {
	abs := filepath.Join(t.TempDir(), "evil.tca")
	if err := os.WriteFile(abs, []byte(sprintfKernel("evil")), 0o644); err != nil {
		t.Fatal(err)
	}
	root := writeFixtures(t, t.TempDir(), map[string]string{
		"root.tca": "@include \"" + abs + "\";",
	})
	_, diags := loadFixtures(t, root)
	wantErr(t, diags, "must be relative")

	root2 := writeFixtures(t, t.TempDir(), map[string]string{
		"root.tca": "@include \"../escape.tca\";",
	})
	_, diags2 := loadFixtures(t, root2)
	wantErr(t, diags2, "escapes")
}

// TestModuleDiamondInclude: two files both include common.tca; the kernels
// from common.tca appear exactly once.
func TestModuleDiamondInclude(t *testing.T) {
	dir := writeFixtures(t, t.TempDir(), map[string]string{
		"root.tca":   "@include \"a.tca\";\n@include \"b.tca\";\n" + sprintfKernel("main"),
		"a.tca":      "@include \"common.tca\";",
		"b.tca":      "@include \"common.tca\";",
		"common.tca": sprintfKernel("shared"),
	})
	m, diags := loadFixtures(t, dir)
	wantNoErrs(t, diags)
	if len(m.Kernels) != 2 {
		t.Fatalf("kernels = %d, want 2 (diamond dedup)", len(m.Kernels))
	}
}

// TestModuleRuntimeInheritance: an included @runtime fills the module's
// runtime when the root declares none; the root's own @runtime wins.
func TestModuleRuntimeInheritance(t *testing.T) {
	dir := writeFixtures(t, t.TempDir(), map[string]string{
		"root.tca":   "@include \"common.tca\";",
		"common.tca": "@runtime(cxx_path=\"runtime/x.h\");",
	})
	m, diags := loadFixtures(t, dir)
	wantNoErrs(t, diags)
	if m.Runtime == nil || len(m.Runtime.Args) != 1 || m.Runtime.Args[0].Value.Text != "runtime/x.h" {
		t.Fatalf("runtime = %+v, want inherited cxx_path", m.Runtime)
	}

	dir2 := writeFixtures(t, t.TempDir(), map[string]string{
		"root.tca":   "@runtime(cxx_path=\"root.h\");\n@include \"common.tca\";",
		"common.tca": "@runtime(cxx_path=\"common.h\");",
	})
	m2, diags2 := loadFixtures(t, dir2)
	wantNoErrs(t, diags2)
	if m2.Runtime == nil || m2.Runtime.Args[0].Value.Text != "root.h" {
		t.Fatalf("runtime = %+v, want root's own", m2.Runtime)
	}
}
