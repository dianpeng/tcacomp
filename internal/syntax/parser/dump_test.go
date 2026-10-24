package parser

import (
	"fmt"
	"strings"
	"testing"

	"tcacomp/internal/golden"
	"tcacomp/internal/syntax/ast"
)

// dump renders one module's AST as a canonical indented tree (test-only:
// design §8.12 keeps a formatter out of the product, but the parse tree
// still needs a stable golden rendering). Positions are deliberately
// omitted — attribution is covered by the lexer and semacheck tests.
func dump(m *ast.Module) string {
	var b strings.Builder
	b.WriteString("module\n")
	for _, inc := range m.Includes {
		fmt.Fprintf(&b, "  include %q\n", inc.Path)
	}
	if m.Runtime != nil {
		fmt.Fprintf(&b, "  runtime %s\n", dumpKeyed(m.Runtime.Args))
	}
	for _, k := range m.Kernels {
		dumpKernel(&b, k)
	}
	return b.String()
}

func dumpKeyed(args []*ast.KeyedConst) string {
	var parts []string
	for _, a := range args {
		fmtArg := a.Value.Text
		if a.Value.Kind == ast.ConstString {
			fmtArg = fmt.Sprintf("%q", a.Value.Text)
		}
		parts = append(parts, a.Key+"="+fmtArg)
	}
	return "(" + strings.Join(parts, ", ") + ")"
}

func dumpKernel(b *strings.Builder, k *ast.Kernel) {
	for _, ann := range k.Anns {
		fmt.Fprintf(b, "  annotation @%s %s\n", ann.Name, dumpKeyed(ann.Args))
	}
	fmt.Fprintf(b, "  kernel %s\n", k.Name)
	for _, p := range k.Params {
		fmt.Fprintf(b, "    param %s %s\n", p.Name, dumpTypeRef(p.Type))
	}
	fmt.Fprintf(b, "    ret %s\n", dumpTypeRef(k.Ret))
	for _, s := range k.Body {
		dumpStmt(b, s, "    ")
	}
}

func dumpTypeRef(t ast.TypeRef) string {
	switch t := t.(type) {
	case *ast.TensorType:
		var dims []string
		for _, d := range t.Shape {
			if d.Dynamic || d.Name != "" {
				dims = append(dims, d.Name)
			} else {
				dims = append(dims, fmtUint(d.Extent))
			}
		}
		return "tensor<" + string(t.Elem.Elem) + ",[" + strings.Join(dims, ",") + "]>"
	case *ast.ScalarType:
		return string(t.Elem)
	}
	return "?"
}

func fmtUint(v uint64) string {
	const digits = "0123456789"
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = digits[v%10]
		v /= 10
	}
	return string(buf[i:])
}

func dumpStmt(b *strings.Builder, s ast.Stmt, ind string) {
	switch s := s.(type) {
	case *ast.DimVarDecl:
		fmt.Fprintf(b, "%sdimvar %s\n", ind, strings.Join(s.Names, " "))
	case *ast.ReturnStmt:
		fmt.Fprintf(b, "%sreturn %s\n", ind, s.Name)
	case *ast.LetDecl:
		ann := ""
		if s.Type != nil {
			ann = ": " + dumpTypeRef(s.Type)
		}
		fmt.Fprintf(b, "%slet %s%s\n", ind, s.Name, ann)
		dumpChain(b, s.Chain, ind+"  ")
	}
}

func dumpChain(b *strings.Builder, c *ast.StageChain, ind string) {
	for _, st := range c.Stages {
		switch st := st.(type) {
		case *ast.MapStage:
			var ops []string
			for _, bind := range st.Bindings {
				ops = append(ops, bind.Tensor+subscripts(bind.Dims))
			}
			as := ""
			if st.As != nil {
				as = " as(" + strings.Join(st.As, ",") + ")"
			}
			fmt.Fprintf(b, "%smap on(%s)%s\n", ind, strings.Join(ops, ", "), as)
			dumpLambda(b, st.Lambda, ind+"  ")
		case *ast.ReduceStage:
			init := ""
			if st.Init != nil {
				init = "init=" + dumpExpr(st.Init)
			}
			as := ""
			if st.As != nil {
				as = " as(" + strings.Join(st.As, ",") + ")"
			}
			fmt.Fprintf(b, "%sreduce(%s) on(%s)%s -> %s: %s\n",
				ind, init, dumpParamSpecs(st.OnParams), as, st.RetType.Elem, dumpExpr(st.Body))
		}
	}
}

func subscripts(dims []string) string {
	var b strings.Builder
	for _, d := range dims {
		b.WriteString("[" + d + "]")
	}
	return b.String()
}

func dumpParamSpecs(specs []*ast.ParamSpec) string {
	var parts []string
	for _, s := range specs {
		t := ""
		if s.Type != nil {
			t = ":" + string(s.Type.Elem)
		}
		if s.Pipe {
			parts = append(parts, "%0"+t)
		} else {
			parts = append(parts, s.Name+t)
		}
	}
	return strings.Join(parts, ", ")
}

func dumpLambda(b *strings.Builder, l *ast.Lambda, ind string) {
	caps := ""
	if len(l.Captures) > 0 {
		caps = "[" + strings.Join(l.Captures, ",") + "] "
	}
	ret := ""
	if l.RetType != nil {
		ret = " -> " + string(l.RetType.Elem)
	}
	fmt.Fprintf(b, "%s%s(%s)%s: %s\n", ind, caps, dumpParamSpecs(l.Params), ret, dumpExpr(l.Body))
}

func dumpExpr(e ast.Expr) string {
	switch e := e.(type) {
	case *ast.BasicLit:
		return e.Value
	case *ast.NameRef:
		return "name(" + e.Name + ")"
	case *ast.PipeRef:
		return "%0"
	case *ast.BinaryExpr:
		return "(" + string(e.Op) + " " + dumpExpr(e.L) + " " + dumpExpr(e.R) + ")"
	case *ast.UnaryExpr:
		return "(" + string(e.Op) + " " + dumpExpr(e.X) + ")"
	case *ast.SelectExpr:
		return "(select " + dumpExpr(e.Cond) + " " + dumpExpr(e.Then) + " " + dumpExpr(e.Else) + ")"
	case *ast.CallExpr:
		var args []string
		for _, a := range e.Args {
			args = append(args, dumpExpr(a))
		}
		return "call(" + e.Name + " " + strings.Join(args, ", ") + ")"
	case *ast.CastExpr:
		return "(cast " + string(e.Target) + " " + dumpExpr(e.X) + ")"
	}
	return "?"
}

// TestAppendixAGolden pins the Appendix A parse tree (design §6 Task 6:
// lexer/parser round-trips on the example programs, goldens).
func TestAppendixAGolden(t *testing.T) {
	m, diags := Parse(testFile("appendix_a.tca", appendixASource))
	if len(diags) != 0 {
		t.Fatalf("parse diagnostics: %v", diags)
	}
	golden.Assert(t, "testdata/appendix_a.txt", []byte(dump(m)))
}
