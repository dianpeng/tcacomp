package ir

import (
	"encoding/binary"
	"hash/fnv"
	"slices"
)

// ElemType is the IR scalar element type set, mirroring doc/syntax-design.md
// §2.4 (f8 is intentionally absent: its concrete formats are unspecified).
type ElemType int

const (
	TBool ElemType = iota
	TF32
	TF16
	TBF16
	TI8
	TI16
	TI32
	TU8
	TU16
	TU32
)

// String returns the canonical source spelling of the element type.
func (t ElemType) String() string {
	switch t {
	case TBool:
		return "bool"
	case TF32:
		return "f32"
	case TF16:
		return "f16"
	case TBF16:
		return "bf16"
	case TI8:
		return "i8"
	case TI16:
		return "i16"
	case TI32:
		return "i32"
	case TU8:
		return "u8"
	case TU16:
		return "u16"
	case TU32:
		return "u32"
	default:
		return "?"
	}
}

// DimRef is one tensor shape dimension. Extent == 0 means a dynamic /
// symbolic runtime dim (Name holds the dim parameter name); Extent > 0 is a
// compile-time constant extent.
type DimRef struct {
	Name   string // symbolic dim parameter name (empty for constant extents)
	Extent uint64 // 0: dynamic/symbolic; >0: compile-time constant
}

// Type is the extensible IR type interface (directive 1): NumericType is the
// concrete numeric component today; tuple/struct/etc. join later as new
// implementations without touching NumericType or the printer's numeric
// paths. Consumers needing element/dim details type-assert to *NumericType;
// Equal is the cross-component comparison point.
//
// Interning makes interface == identity for numerics (D-1): the dynamic type
// *NumericType is a comparable pointer and interns are deduped, so types are
// interface-compared and map-keyable, and deterministic emission needs no
// tie-breaking. The zero rank (len(dims) == 0) is a scalar.
type Type interface {
	Equal(other Type) bool
}

// NumericType is the tensor/scalar type component (directive 1). It is
// interned, immutable, and identity-compared via the Type interface.
type NumericType struct {
	elem ElemType
	dims []DimRef
}

// Elem returns the element type.
func (t *NumericType) Elem() ElemType { return t.elem }

// Rank returns the number of shape dimensions; 0 means a scalar type.
func (t *NumericType) Rank() int { return len(t.dims) }

// IsScalar reports whether the type is a scalar (rank 0).
func (t *NumericType) IsScalar() bool { return len(t.dims) == 0 }

// Dims returns a copy of the shape dimensions (empty for scalars).
func (t *NumericType) Dims() []DimRef { return slices.Clone(t.dims) }

// Equal compares this numeric against another Type: content compare when
// other is a *NumericType, false otherwise — including a nil Type. It never
// panics and never special-cases a particular implementation, so any future
// Type component can be compared against a numeric safely.
func (t *NumericType) Equal(other Type) bool {
	o, ok := other.(*NumericType)
	return ok && t.elem == o.elem && slices.Equal(t.dims, o.dims)
}

// internTable is the package-level type interner. The map is keyed by
// content hash and used only for lookup — never iterated — so interning
// cannot leak map-order nondeterminism into emission (Pitfall 1).
//
// SECURITY NOTE (T-1-01-04): the FNV-1a hash below is an interning dedup
// key only. It is not a security hash and must never be used where collision
// resistance matters; bucket membership is always confirmed by full content
// comparison (numericEquals).
var internTable = struct {
	byHash map[uint64][]*NumericType
}{byHash: make(map[uint64][]*NumericType)}

// typeContentHash computes the FNV-1a content hash of an element type plus
// its dimension list.
func typeContentHash(elem ElemType, dims []DimRef) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(elem))
	h.Write(buf[:])
	for _, d := range dims {
		binary.LittleEndian.PutUint64(buf[:], d.Extent)
		h.Write(buf[:])
		h.Write([]byte(d.Name))
		h.Write([]byte{0})
	}
	return h.Sum64()
}

func numericEquals(a *NumericType, elem ElemType, dims []DimRef) bool {
	return a.elem == elem && slices.Equal(a.dims, dims)
}

// InternScalar returns the interned scalar type for elem.
func InternScalar(elem ElemType) Type {
	return InternTensor(elem, nil)
}

// InternTensor returns the interned tensor type with the given element type
// and shape. A nil or empty dims slice yields a scalar type. The dims slice
// is copied, so callers may reuse it.
func InternTensor(elem ElemType, dims []DimRef) Type {
	h := typeContentHash(elem, dims)
	for _, t := range internTable.byHash[h] {
		if numericEquals(t, elem, dims) {
			return t
		}
	}
	t := &NumericType{elem: elem, dims: slices.Clone(dims)}
	internTable.byHash[h] = append(internTable.byHash[h], t)
	return t
}
