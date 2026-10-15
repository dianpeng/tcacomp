package ir

import (
	"encoding/binary"
	"hash/fnv"
	"slices"
)

// ElemType is the IR scalar element type set, mirroring doc/syntax.md §3.3
// (f8 is intentionally absent: its concrete formats are unspecified).
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

// Type is an interned, immutable IR type (D-18): identity == equality, so
// types are pointer-compared and map-keyable, and deterministic emission
// needs no tie-breaking. The zero rank (len(dims) == 0) is a scalar.
type Type struct {
	elem ElemType
	dims []DimRef
}

// Elem returns the element type.
func (t *Type) Elem() ElemType { return t.elem }

// Rank returns the number of shape dimensions; 0 means a scalar type.
func (t *Type) Rank() int { return len(t.dims) }

// IsScalar reports whether the type is a scalar (rank 0).
func (t *Type) IsScalar() bool { return len(t.dims) == 0 }

// Dims returns a copy of the shape dimensions (empty for scalars).
func (t *Type) Dims() []DimRef { return slices.Clone(t.dims) }

// internTable is the package-level type interner. The map is keyed by
// content hash and used only for lookup — never iterated — so interning
// cannot leak map-order nondeterminism into emission (Pitfall 1).
//
// SECURITY NOTE (T-1-01-04): the FNV-1a hash below is an interning dedup
// key only. It is not a security hash and must never be used where collision
// resistance matters; bucket membership is always confirmed by full content
// comparison (typesEqual).
var internTable = struct {
	byHash map[uint64][]*Type
}{byHash: make(map[uint64][]*Type)}

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

func typesEqual(a *Type, elem ElemType, dims []DimRef) bool {
	return a.elem == elem && slices.Equal(a.dims, dims)
}

// InternScalar returns the interned scalar type for elem.
func InternScalar(elem ElemType) *Type {
	return InternTensor(elem, nil)
}

// InternTensor returns the interned tensor type with the given element type
// and shape. A nil or empty dims slice yields a scalar type. The dims slice
// is copied, so callers may reuse it.
func InternTensor(elem ElemType, dims []DimRef) *Type {
	h := typeContentHash(elem, dims)
	for _, t := range internTable.byHash[h] {
		if typesEqual(t, elem, dims) {
			return t
		}
	}
	t := &Type{elem: elem, dims: slices.Clone(dims)}
	internTable.byHash[h] = append(internTable.byHash[h], t)
	return t
}
