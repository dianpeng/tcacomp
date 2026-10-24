// Package semacheck is the TCA frontend's semantic checker
// (doc/syntax-design.md §4, §5.1): it validates one kernel against the §4
// rules (one namespace, bindings/axes, lambdas, expression typing,
// statements) and resolves the omitted annotation cells IN PLACE per the
// §5.1 forward pass — no parallel typed-AST hierarchy. Module-scope rules
// 1-4 are CheckModule's job; the per-kernel rules 5-26 + resolution live in
// Check.
//
// Unlike ast/parser, this package MAY import internal/ir: it owns the
// ast.Elem ↔ ir.ElemType conversion, the operator-spelling → opcode tables,
// the shared cast registry (ir.CastPairOK), and the builtin registry
// (ir.Builtin) — single-sourced validation mirroring pkg/expr.go (ruling R5).
package semacheck

import (
	"fmt"

	"tcacomp/internal/diagnostic"
	"tcacomp/internal/ir"
	"tcacomp/internal/syntax/ast"
	"tcacomp/internal/syntax/lexer"
)

// Check validates one kernel (design §4 rules 5-26) and resolves its omitted
// annotation cells in place (§5.1). It returns every diagnostic found
// (errors and warnings); a kernel with any Error-severity diagnostic must be
// treated as having no typed AST (fail-fast at kernel granularity, §4).
func Check(k *ast.Kernel) []diagnostic.Diagnostic {
	c := newChecker(k)
	c.collectSignature()
	c.checkBody()
	return c.diags
}

// checker holds one kernel's semantic-checking state: the single-namespace
// symbol table (rule 5) and the diagnostic accumulator.
type checker struct {
	k     *ast.Kernel
	diags []diagnostic.Diagnostic

	tensorParams map[string]*ast.Param   // tensor parameters by name
	scalarParams map[string]ir.ElemType  // scalar parameters by name
	dims         map[string]ast.ShapeDim // implicit signature dims (rule 6)
	dimvars      map[string]*dimvarInfo  // declared dimvars (structural axis → extent source)
	lets         map[string]*letInfo     // let-bound tensors in declaration order
}

// dimvarInfo records one declared dimvar and the shape dim it is structurally
// bound to (its extent source, rule 26). Two dimvars bound to the same shape
// dim are distinct axes; one dimvar bound to two different shape dims is a
// semacheck error (inconsistent structural binding).
type dimvarInfo struct {
	pos lexer.Pos
	src ast.ShapeDim // extent source; zero value until the dimvar is bound
	set bool
}

// letInfo records one let-bound tensor: its element type and the extent
// source of each of its axes (the final stage's as-axes, §5.1 step 5).
type letInfo struct {
	pos   lexer.Pos
	elem  ir.ElemType
	shape []ast.ShapeDim
}

func newChecker(k *ast.Kernel) *checker {
	return &checker{
		k:            k,
		tensorParams: map[string]*ast.Param{},
		scalarParams: map[string]ir.ElemType{},
		dims:         map[string]ast.ShapeDim{},
		dimvars:      map[string]*dimvarInfo{},
		lets:         map[string]*letInfo{},
	}
}

func (c *checker) errorf(pos lexer.Pos, format string, args ...any) {
	c.diags = append(c.diags, diagnostic.Diagnostic{
		Severity: diagnostic.Error,
		Span:     spanOf(pos),
		Message:  fmt.Sprintf(format, args...),
	})
}

func (c *checker) warnf(pos lexer.Pos, format string, args ...any) {
	c.diags = append(c.diags, diagnostic.Diagnostic{
		Severity: diagnostic.Warning,
		Span:     spanOf(pos),
		Message:  fmt.Sprintf(format, args...),
	})
}

func spanOf(pos lexer.Pos) diagnostic.Span {
	span := diagnostic.Span{Line: pos.Line, Col: pos.Col}
	if pos.File != nil {
		span.File = pos.File.Name
	}
	return span
}

// elemOf converts an ast element spelling onto the IR element type set
// (design §2.4, 1:1). Unknown spellings are impossible: the parser only
// produces element-type keywords. The IR zero value is TBool, so spellings
// are validated against the table rather than defaulted.
func elemOf(e ast.Elem) (ir.ElemType, bool) {
	switch e {
	case ast.EBool:
		return ir.TBool, true
	case ast.EF32:
		return ir.TF32, true
	case ast.EF16:
		return ir.TF16, true
	case ast.EBF16:
		return ir.TBF16, true
	case ast.EI8:
		return ir.TI8, true
	case ast.EI16:
		return ir.TI16, true
	case ast.EI32:
		return ir.TI32, true
	case ast.EU8:
		return ir.TU8, true
	case ast.EU16:
		return ir.TU16, true
	case ast.EU32:
		return ir.TU32, true
	}
	return ir.TBool, false
}

// isIntElem / isFloatElem mirror pkg/expr.go's operand classes (rule 17).
func isIntElem(e ir.ElemType) bool {
	switch e {
	case ir.TI8, ir.TI16, ir.TI32, ir.TU8, ir.TU16, ir.TU32:
		return true
	}
	return false
}

func isFloatElem(e ir.ElemType) bool {
	switch e {
	case ir.TF32, ir.TF16, ir.TBF16:
		return true
	}
	return false
}

func isArithElem(e ir.ElemType) bool { return isIntElem(e) || isFloatElem(e) }

// elemName renders an element type for diagnostics.
func elemName(e ir.ElemType) string { return e.String() }

// binOpOf maps an AST binary operator spelling onto the IR opcode table
// (design §2.7: the operator set is the same table, 1:1).
func binOpOf(op ast.BinOp) (ir.BinOpCode, bool) {
	switch op {
	case "||":
		return ir.BinOpLOr, true
	case "&&":
		return ir.BinOpLAnd, true
	case "|":
		return ir.BinOpOr, true
	case "^":
		return ir.BinOpXor, true
	case "&":
		return ir.BinOpAnd, true
	case "==":
		return ir.BinOpEq, true
	case "!=":
		return ir.BinOpNE, true
	case "<":
		return ir.BinOpLT, true
	case "<=":
		return ir.BinOpLE, true
	case ">":
		return ir.BinOpGT, true
	case ">=":
		return ir.BinOpGE, true
	case "<<":
		return ir.BinOpShL, true
	case ">>":
		return ir.BinOpShR, true
	case "+":
		return ir.BinOpAdd, true
	case "-":
		return ir.BinOpSub, true
	case "*":
		return ir.BinOpMul, true
	case "/":
		return ir.BinOpDiv, true
	case "%":
		return ir.BinOpMod, true
	}
	return ir.BinOpInvalid, false
}

// unaryOpOf maps an AST unary operator spelling onto the IR opcode table.
func unaryOpOf(op ast.UnaryOp) (ir.UnaryOpCode, bool) {
	switch op {
	case "+":
		return ir.UnaryOpPos, true
	case "-":
		return ir.UnaryOpNeg, true
	case "!":
		return ir.UnaryOpLNot, true
	case "~":
		return ir.UnaryOpNot, true
	}
	return ir.UnaryOpInvalid, false
}

// collectSignature registers parameters and the return type (rules 5-8):
// one-namespace duplicates, implicit dims from shapes (R1), shape required
// and rank ≥ 1 on signature tensors, return-only dims rejected (rule 6).
func (c *checker) collectSignature() {
	for _, p := range c.k.Params {
		c.registerParam(p)
	}
	switch ret := c.k.Ret.(type) {
	case *ast.TensorType:
		c.checkSignatureTensor(ret, "kernel return type")
		for _, d := range ret.Shape {
			if d.Dynamic || d.Name != "" {
				if _, ok := c.dims[d.Name]; !ok {
					c.errorf(ret.Pos(), "return shape dim %q does not appear in any parameter shape (unbindable, rule 6)", d.Name)
				}
			}
		}
	case *ast.ScalarType:
		if _, ok := elemOf(ret.Elem); !ok {
			c.errorf(ret.Pos(), "unknown element type %q", ret.Elem)
		}
	}
}

// registerParam registers one kernel parameter in the namespace (rule 5) and
// fills its role (design §3.1).
func (c *checker) registerParam(p *ast.Param) {
	if c.taken(p.Name) {
		c.errorf(p.Pos(), "duplicate name %q (one namespace: params, dims, dimvars, lets must all be distinct, rule 5)", p.Name)
		return
	}
	switch t := p.Type.(type) {
	case *ast.TensorType:
		p.Role = ast.TensorParam
		c.checkSignatureTensor(t, "parameter "+p.Name)
		c.tensorParams[p.Name] = p
	case *ast.ScalarType:
		p.Role = ast.ScalarParam
		elem, ok := elemOf(t.Elem)
		if !ok {
			c.errorf(t.Pos(), "unknown element type %q", t.Elem)
			return
		}
		c.scalarParams[p.Name] = elem
	default:
		c.errorf(p.Pos(), "parameter %s requires a type", p.Name)
	}
}

// checkSignatureTensor validates a tensor type in a signature position
// (parameter or return): shape required, rank ≥ 1 (rule 7), and registers its
// implicit dims, rejecting conflicting redeclarations of the same dim name.
func (c *checker) checkSignatureTensor(t *ast.TensorType, what string) {
	if _, ok := elemOf(t.Elem.Elem); !ok {
		c.errorf(t.Pos(), "%s: unknown element type %q", what, t.Elem.Elem)
	}
	if t.Shape == nil {
		c.errorf(t.Pos(), "%s: tensor shape is required in a kernel signature (rank is always static, §2.4)", what)
		return
	}
	if len(t.Shape) < 1 {
		c.errorf(t.Pos(), "%s: tensor rank must be ≥ 1 (rule 7)", what)
		return
	}
	for _, d := range t.Shape {
		if !d.Dynamic && d.Name == "" {
			continue // compile-time constant extent (may be 0)
		}
		if prev, ok := c.dims[d.Name]; ok {
			if prev.Extent != d.Extent || prev.Dynamic != d.Dynamic {
				c.errorf(d.Pos(), "shape dim %q redeclared with a different extent", d.Name)
			}
			continue
		}
		c.dims[d.Name] = d
	}
}

// taken reports whether name is already registered in the kernel namespace.
func (c *checker) taken(name string) bool {
	if _, ok := c.tensorParams[name]; ok {
		return true
	}
	if _, ok := c.scalarParams[name]; ok {
		return true
	}
	if _, ok := c.dims[name]; ok {
		return true
	}
	if _, ok := c.dimvars[name]; ok {
		return true
	}
	if _, ok := c.lets[name]; ok {
		return true
	}
	return false
}
