package ast

import "tcacomp/internal/syntax/lexer"

// --- types (design §3.1; TypeRef is an interface) ---

// TypeRef is a type annotation: *TensorType or *ScalarType.
type TypeRef interface {
	Node
	typeRefNode()
}

// Elem is the element-type spelling set (design §2.4), carried as the exact
// source keyword so this package stays free of internal/ir imports (the
// layering gate). Semacheck maps Elem onto ir.ElemType 1:1.
type Elem string

// The element-type spellings (design §2.4; f8 is intentionally absent).
const (
	EBool Elem = "bool"
	EF32  Elem = "f32"
	EF16  Elem = "f16"
	EBF16 Elem = "bf16"
	EI8   Elem = "i8"
	EI16  Elem = "i16"
	EI32  Elem = "i32"
	EU8   Elem = "u8"
	EU16  Elem = "u16"
	EU32  Elem = "u32"
)

// TensorType is 'tensor' '<' 'dtype' '=' scalarType [',' 'shape' '=' '['
// shapeDimList ']'] [','] '>' (design §2.4).
type TensorType struct {
	pos   lexer.Pos
	Elem  ScalarType
	Shape []ShapeDim // nil = shape omitted (legal only on local annotations)
}

func (n *TensorType) astNode()       {}
func (n *TensorType) typeRefNode()   {}
func (n *TensorType) Pos() lexer.Pos { return n.pos }

// NewTensorType builds a TensorType at pos; shape nil means omitted.
func NewTensorType(pos lexer.Pos, elem ScalarType, shape []ShapeDim) *TensorType {
	return &TensorType{pos: pos, Elem: elem, Shape: shape}
}

// ScalarType is one bare element type (design §2.4 scalarType).
type ScalarType struct {
	pos  lexer.Pos
	Elem Elem
}

func (n *ScalarType) astNode()       {}
func (n *ScalarType) typeRefNode()   {}
func (n *ScalarType) Pos() lexer.Pos { return n.pos }

// NewScalarType builds a ScalarType at pos.
func NewScalarType(pos lexer.Pos, elem Elem) *ScalarType {
	return &ScalarType{pos: pos, Elem: elem}
}

// ShapeDim is one tensor shape dimension (design §2.4): a uintLit compile-time
// constant extent (Name empty) or an identifier declaring an implicit symbolic
// dim (Name set, Extent == 0, R1). Dynamic records the spelling so a written
// constant 0 (legal per §2.4) stays distinguishable from a symbolic dim,
// keeping the design's "Name set ⇔ dynamic" invariant for every other case.
type ShapeDim struct {
	pos     lexer.Pos
	Name    string // symbolic dim parameter name (empty for constant extents)
	Extent  uint64 // 0: dynamic/symbolic or a written constant 0 (see Dynamic)
	Dynamic bool   // true when the dim was written as an identifier
}

func (n *ShapeDim) astNode()       {}
func (n *ShapeDim) Pos() lexer.Pos { return n.pos }

// NewShapeDim builds a ShapeDim at pos.
func NewShapeDim(pos lexer.Pos, name string, extent uint64, dynamic bool) *ShapeDim {
	return &ShapeDim{pos: pos, Name: name, Extent: extent, Dynamic: dynamic}
}
