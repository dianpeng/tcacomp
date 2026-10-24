package parser

import (
	"testing"

	"tcacomp/internal/syntax/ast"
)

// appendixASource is the normative fixture of doc/syntax-design.md Appendix A
// (kept byte-identical to the design doc).
const appendixASource = `@include "tca-common.tca";

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

// TestAppendixAParse parses the Appendix A fixture: zero diagnostics and the
// expected declaration/statement/stage skeleton.
func TestAppendixAParse(t *testing.T) {
	m, diags := Parse(testFile("appendix_a.tca", appendixASource))
	if len(diags) != 0 {
		for _, d := range diags {
			t.Logf("diag: %s", d)
		}
		t.Fatalf("got %d diagnostics, want 0", len(diags))
	}
	if len(m.Includes) != 1 || m.Includes[0].Path != "tca-common.tca" {
		t.Fatalf("includes = %+v", m.Includes)
	}
	if m.Runtime == nil || len(m.Runtime.Args) != 1 || m.Runtime.Args[0].Key != "cxx_path" {
		t.Fatalf("runtime = %+v", m.Runtime)
	}
	if len(m.Kernels) != 1 {
		t.Fatalf("kernels = %d, want 1", len(m.Kernels))
	}
	k := m.Kernels[0]
	if k.Name != "matmul_scale" {
		t.Fatalf("kernel name = %q", k.Name)
	}
	if len(k.Anns) != 1 || k.Anns[0].Name != "name" {
		t.Fatalf("annotations = %+v", k.Anns)
	}
	if len(k.Params) != 3 {
		t.Fatalf("params = %d, want 3", len(k.Params))
	}
	if len(k.Body) != 3 {
		t.Fatalf("body = %d stmts, want 3 (dimvar, let, return)", len(k.Body))
	}
	if dv, ok := k.Body[0].(*ast.DimVarDecl); !ok || len(dv.Names) != 3 ||
		dv.Names[0] != "a" || dv.Names[1] != "b" || dv.Names[2] != "c" {
		t.Fatalf("dimvar decl = %#v", k.Body[0])
	}
	letStmt, ok := k.Body[1].(*ast.LetDecl)
	if !ok || letStmt.Name != "output" {
		t.Fatalf("let stmt = %#v", k.Body[1])
	}
	if letStmt.Type != nil {
		t.Fatalf("let annotation must be omitted in Appendix A, got %#v", letStmt.Type)
	}
	chain := letStmt.Chain
	if len(chain.Stages) != 3 {
		t.Fatalf("chain stages = %d, want 3", len(chain.Stages))
	}
	root, ok := chain.Stages[0].(*ast.MapStage)
	if !ok {
		t.Fatalf("stage 0 = %T, want *ast.MapStage", chain.Stages[0])
	}
	if len(root.Bindings) != 2 || root.Bindings[0].Tensor != "lhs" ||
		len(root.Bindings[0].Dims) != 2 || root.Bindings[0].Dims[0] != "a" ||
		root.Bindings[1].Tensor != "rhs" || root.Bindings[1].Dims[1] != "c" {
		t.Fatalf("root bindings = %+v", root.Bindings)
	}
	if len(root.As) != 3 || root.As[0] != "a" || root.As[1] != "b" || root.As[2] != "c" {
		t.Fatalf("root as = %v", root.As)
	}
	if len(root.Lambda.Params) != 2 || root.Lambda.RetType == nil ||
		root.Lambda.RetType.Elem != ast.EF32 {
		t.Fatalf("root lambda = %+v", root.Lambda)
	}
	red, ok := chain.Stages[1].(*ast.ReduceStage)
	if !ok {
		t.Fatalf("stage 1 = %T, want *ast.ReduceStage", chain.Stages[1])
	}
	if red.Init == nil || len(red.OnParams) != 2 || !red.OnParams[1].Pipe ||
		red.OnParams[1].Name != "%0" || red.RetType.Elem != ast.EF32 {
		t.Fatalf("reduce = init %v params %+v ret %v", red.Init, red.OnParams, red.RetType)
	}
	if len(red.As) != 2 || red.As[0] != "a" || red.As[1] != "c" {
		t.Fatalf("reduce as = %v", red.As)
	}
	piped, ok := chain.Stages[2].(*ast.MapStage)
	if !ok {
		t.Fatalf("stage 2 = %T, want *ast.MapStage", chain.Stages[2])
	}
	if len(piped.Bindings) != 0 || len(piped.Lambda.Captures) != 1 ||
		piped.Lambda.Captures[0] != "scale" || len(piped.Lambda.Params) != 1 {
		t.Fatalf("piped map = %+v", piped)
	}
	if piped.As != nil {
		t.Fatalf("piped as must be omitted in Appendix A, got %v", piped.As)
	}
	ret, ok := k.Body[2].(*ast.ReturnStmt)
	if !ok || ret.Name != "output" {
		t.Fatalf("return stmt = %#v", k.Body[2])
	}
	// Return type shape [M, Q] with implicit symbolic dims.
	retTy, ok := k.Ret.(*ast.TensorType)
	if !ok || len(retTy.Shape) != 2 || retTy.Shape[0].Name != "M" || retTy.Shape[1].Name != "Q" {
		t.Fatalf("kernel return type = %#v", k.Ret)
	}
}
