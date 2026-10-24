package semacheck

import (
	"strings"
	"testing"

	"tcacomp/internal/diagnostic"
	"tcacomp/internal/syntax/ast"
	"tcacomp/internal/syntax/lexer"
	"tcacomp/internal/syntax/parser"
)

// checkKernel parses one kernel source and runs the per-kernel checks,
// returning the diagnostics (design §6 Task 6: one positive and one negative
// test per §4 rule family, no exhaustive matrices).
func checkKernel(t *testing.T, src string) []diagnostic.Diagnostic {
	t.Helper()
	f := (&lexer.FileSet{}).AddFile("test.tca", src)
	m, pdiags := parser.Parse(f)
	for _, d := range pdiags {
		if d.Severity == diagnostic.Error {
			t.Fatalf("parse failed: %s", d)
		}
	}
	if len(m.Kernels) != 1 {
		t.Fatalf("kernels = %d, want 1", len(m.Kernels))
	}
	return Check(m.Kernels[0])
}

func errorsOf(diags []diagnostic.Diagnostic) []diagnostic.Diagnostic {
	var errs []diagnostic.Diagnostic
	for _, d := range diags {
		if d.Severity == diagnostic.Error {
			errs = append(errs, d)
		}
	}
	return errs
}

// wantError asserts exactly one error mentioning want is present.
func wantError(t *testing.T, diags []diagnostic.Diagnostic, want string) {
	t.Helper()
	errs := errorsOf(diags)
	if len(errs) != 1 {
		for _, d := range diags {
			t.Logf("diag: %s", d)
		}
		t.Fatalf("errors = %d, want exactly 1 mentioning %q", len(errs), want)
	}
	if !strings.Contains(errs[0].Message, want) {
		t.Fatalf("error %q does not mention %q", errs[0].Message, want)
	}
}

func wantClean(t *testing.T, diags []diagnostic.Diagnostic) {
	t.Helper()
	if errs := errorsOf(diags); len(errs) != 0 {
		for _, d := range errs {
			t.Logf("diag: %s", d)
		}
		t.Fatalf("errors = %d, want 0", len(errs))
	}
}

// validKernel is the positive fixture reused across the rule families: a
// scaled matmul contracting the middle axis, piped through an elementwise
// scale — structurally Appendix A with renamed dims.
const validKernel = `kernel k(
  t: tensor<dtype=f32, shape=[M, N]>,
  u: tensor<dtype=f32, shape=[N, Q]>,
  s: f32,
) -> tensor<dtype=f32, shape=[M, Q]>
{
  dimvar (a, b, c);
  let out = map on(t[a][b], u[b][c]) as (a, b, c) (x: f32, y: f32) -> f32: x * y
    | reduce(init = 0.0f) on(acc: f32, %0: f32) as (a, c) -> f32: acc + %0
    | map on() [s] (v: f32) -> f32: v * s
    ;
  return out;
}`

// TestKernelScope: rule 5 (one namespace, no shadowing) — negative: a dimvar
// named like a tensor parameter; positive: the valid fixture.
func TestKernelScope(t *testing.T) {
	wantClean(t, checkKernel(t, validKernel))
	dup := strings.Replace(validKernel, "dimvar (a, b, c);", "dimvar (a, b, c, t);", 1)
	wantError(t, checkKernel(t, dup), `duplicate name "t"`)
}

// TestImplicitDims: rule 6 — a return-only dim is unbindable.
func TestImplicitDims(t *testing.T) {
	bad := strings.Replace(validKernel, "shape=[M, Q]>\n{", "shape=[M, Z]>\n{", 1)
	wantError(t, checkKernel(t, bad), `"Z"`)
}

// TestTypes: rules 7/8 — a written lambda param annotation that disagrees
// with the inferred operand element type.
func TestTypes(t *testing.T) {
	bad := strings.Replace(validKernel, "(x: f32, y: f32)", "(x: i32, y: f32)", 1)
	wantError(t, checkKernel(t, bad), "param annotation")
}

// TestBindingsAndAxes: rules 9/10 — a subscript naming a shape dim instead of
// a dimvar.
func TestBindingsAndAxes(t *testing.T) {
	bad := strings.Replace(validKernel, "on(t[a][b], u[b][c])", "on(t[a][N], u[b][c])", 1)
	wantError(t, checkKernel(t, bad), "rule 10")
}

// TestAsPermutation: rule 11 — a root map as clause omitting a bound dimvar.
// Standalone kernel: shrinking the root as cascades legitimately into the
// downstream stages of the shared fixture, so the negative is isolated.
func TestAsPermutation(t *testing.T) {
	bad := `kernel k(t: tensor<dtype=f32, shape=[M, N]>) -> tensor<dtype=f32, shape=[M]> {
  dimvar (a, b);
  let out = map on(t[a][b]) as (a) (x: f32) -> f32: x;
  return out;
}`
	wantError(t, checkKernel(t, bad), "rule 11")
}

// TestReduceAxes: rule 11/12 — a reduce as clause dropping two axes.
func TestReduceAxes(t *testing.T) {
	bad := strings.Replace(validKernel, "as (a, c) -> f32", "as (a) -> f32", 1)
	wantError(t, checkKernel(t, bad), "minus exactly one")
}

// TestLambdas: rule 13 — piped map lambda arity off by one.
func TestLambdaArity(t *testing.T) {
	bad := strings.Replace(validKernel, "[s] (v: f32) -> f32:", "[s] (v: f32, w: f32) -> f32:", 1)
	wantError(t, checkKernel(t, bad), "rule 13")
}

// TestCaptures: rule 14 — capturing a tensor is rejected; an unused capture
// warns but does not error.
func TestCaptures(t *testing.T) {
	bad := strings.Replace(validKernel, "[s] (v: f32) -> f32: v * s", "[t] (v: f32) -> f32: v", 1)
	wantError(t, checkKernel(t, bad), "not a kernel scalar param")

	warn := strings.Replace(validKernel, "[s] (v: f32) -> f32: v * s", "[s] (v: f32) -> f32: v", 1)
	diags := checkKernel(t, warn)
	if errs := errorsOf(diags); len(errs) != 0 {
		t.Fatalf("errors = %v, want 0 (unused capture is a warning only)", errs)
	}
	found := false
	for _, d := range diags {
		if d.Severity == diagnostic.Warning && strings.Contains(d.Message, "never used") {
			found = true
		}
	}
	if !found {
		t.Fatalf("want an unused-capture warning, got %v", diags)
	}
}

// TestExprTyping: rules 17-20 — mixed operand types, a non-registry cast, and
// an unknown builtin.
func TestExprTyping(t *testing.T) {
	bad := strings.Replace(validKernel, "x * y", "x + 1", 1)
	wantError(t, checkKernel(t, bad), "operand types differ")

	cast := strings.Replace(validKernel, "x * y", "f32(x)", 1)
	wantError(t, checkKernel(t, cast), "redundant cast")

	call := strings.Replace(validKernel, "x * y", "nosuchf(x)", 1)
	wantError(t, checkKernel(t, call), "not a registered pure scalar builtin")
}

// TestSelectTyping: rule 18 — a non-bool select condition.
func TestSelectTyping(t *testing.T) {
	bad := strings.Replace(validKernel, "x * y", "x > y ? x : y", 1)
	wantClean(t, checkKernel(t, bad)) // comparison + select is well-typed
	bad2 := strings.Replace(validKernel, "x * y", "x ? x : y", 1)
	wantError(t, checkKernel(t, bad2), "must be bool")
}

// TestStatements: rules 25/26 — a chain not beginning with a root map.
func TestStatements(t *testing.T) {
	bad := strings.Replace(validKernel,
		"let out = map on(t[a][b], u[b][c]) as (a, b, c) (x: f32, y: f32) -> f32: x * y",
		"let out = map on() [s] (v: f32) -> f32: v * s", 1)
	wantError(t, checkKernel(t, bad), "rule 26")
}

// TestReturnChecks: rule 25 — returning a non-let name.
func TestReturnChecks(t *testing.T) {
	bad := strings.Replace(validKernel, "return out;", "return s;", 1)
	wantError(t, checkKernel(t, bad), "not a let-bound tensor")
}

// TestPipeRefScope: a PipeRef in a root map body has no pipe slot.
func TestPipeRefScope(t *testing.T) {
	bad := strings.Replace(validKernel, "x * y", "x * %0", 1)
	wantError(t, checkKernel(t, bad), "no pipe slot")
}

// TestModuleScope: rules 1, 3, 4 via CheckModule — duplicate kernels, a bad
// @runtime, and an unknown annotation.
func TestModuleScope(t *testing.T) {
	parse := func(t *testing.T, src string) *ast.Module {
		t.Helper()
		f := (&lexer.FileSet{}).AddFile("test.tca", src)
		m, diags := parser.Parse(f)
		for _, d := range diags {
			if d.Severity == diagnostic.Error {
				t.Fatalf("parse failed: %s", d)
			}
		}
		return m
	}
	dup := "kernel a() -> f32 { return x; }\nkernel a() -> f32 { return x; }"
	if diags := errorsOf(CheckModule(parse(t, dup))); len(diags) != 1 ||
		!strings.Contains(diags[0].Message, "duplicate kernel name") {
		t.Fatalf("dup kernels diags = %v", diags)
	}
	rt := "@runtime(cxx_path=3);\nkernel a() -> f32 { return x; }"
	if diags := errorsOf(CheckModule(parse(t, rt))); len(diags) != 1 ||
		!strings.Contains(diags[0].Message, "string literal") {
		t.Fatalf("runtime diags = %v", diags)
	}
	ann := "@bogus() kernel a() -> f32 { return x; }"
	if diags := errorsOf(CheckModule(parse(t, ann))); len(diags) != 1 ||
		!strings.Contains(diags[0].Message, "unknown annotation") {
		t.Fatalf("annotation diags = %v", diags)
	}
}

// TestAppendixAResolved runs the normative Appendix A kernel through Check
// and asserts the resolution cells §5.1 promises: every omitted annotation
// filled, written ones verified.
func TestAppendixAResolved(t *testing.T) {
	const appendixA = `@include "tca-common.tca";

@runtime(
  cxx_path="runtime/tca_runtime.h",
);

@name(cxx="MatMulScale")
kernel matmul_scale(
  lhs: tensor<dtype=f32, shape=[M, N]>,
  rhs: tensor<dtype=f32, shape=[N, Q]>,
  scale: f32,
) -> tensor<dtype=f32, shape=[M, Q]>
{
  dimvar (a, b, c);

  let output = map
    on(lhs[a][b], rhs[b][c])
    as (a, b, c)
    (l: f32, r: f32) -> f32:
      l * r
  | reduce(init = 0.0f)
    on(acc: f32, %0: f32)
    as (a, c) -> f32:
      acc + %0
  | map
    on()
    [scale] (v: f32) -> f32:
      v * scale
  ;

  return output;
}`
	f := (&lexer.FileSet{}).AddFile("appendix_a.tca", appendixA)
	mod, pdiags := parser.Parse(f)
	for _, d := range pdiags {
		if d.Severity == diagnostic.Error {
			t.Fatalf("parse failed: %s", d)
		}
	}
	if diags := errorsOf(CheckModule(mod)); len(diags) != 0 {
		t.Fatalf("module diags = %v", diags)
	}
	k := mod.Kernels[0]
	if diags := Check(k); len(diags) != 0 {
		for _, d := range diags {
			t.Logf("diag: %s", d)
		}
		t.Fatalf("semacheck diags = %d, want 0", len(diags))
	}

	letStmt := k.Body[1].(*ast.LetDecl)
	if letStmt.Type == nil {
		t.Fatal("let type must be filled by resolution")
	}
	tt := letStmt.Type.(*ast.TensorType)
	if tt.Elem.Elem != ast.EF32 || len(tt.Shape) != 2 ||
		tt.Shape[0].Name != "M" || tt.Shape[1].Name != "Q" {
		t.Fatalf("resolved let type = %v", tt)
	}
	chain := letStmt.Chain
	root := chain.Stages[0].(*ast.MapStage)
	if root.As == nil {
		t.Fatal("root as must be filled")
	}
	red := chain.Stages[1].(*ast.ReduceStage)
	piped := chain.Stages[2].(*ast.MapStage)
	if len(red.As) != 2 || red.As[0] != "a" || red.As[1] != "c" {
		t.Fatalf("reduce as = %v", red.As)
	}
	if piped.As == nil {
		t.Fatal("piped as must be filled")
	}
	if len(piped.As) != 2 || piped.As[0] != "a" || piped.As[1] != "c" {
		t.Fatalf("piped as = %v", piped.As)
	}
	// All lambda return annotations were written and verified; the param
	// specs kept theirs. Roles are filled.
	for _, p := range k.Params {
		if p.Role == ast.ParamRoleUnknown {
			t.Fatalf("param %s role unset", p.Name)
		}
	}
	if k.Params[0].Role != ast.TensorParam || k.Params[2].Role != ast.ScalarParam {
		t.Fatalf("param roles = %v/%v", k.Params[0].Role, k.Params[2].Role)
	}
}
