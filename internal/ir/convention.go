package ir

import (
	"fmt"
	"reflect"
)

// convention.go machine-checks the Input/ImpactBy-only convention
// (quick-261019-ktm revision, D16): an IR node's inputs live ONLY in its
// Input (operand) list — data dependencies — and its ImpactBy list —
// side-effect dependencies. No node struct may carry a field whose type
// holds (or could hold) SSA/Node references outside those two lists; the
// only sanctioned payload spelling is a DIRECT []SSAUse field on an effect
// bearer (the ImpactBy effect-input lists of memLoad/memStore/return).
//
// The check is TYPE-LEVEL and shape-based, not value-based: a forbidden
// field trips the violation even when it holds a zero value, because the
// convention is a structural property of the node suite — a field that CAN
// hide a reference is already a violation the moment it is declared.

// invConvention names the Input/ImpactBy-only sweep.
const invConvention = "convention"

// hiddenRefFieldType classifies a struct-field type under the convention:
// it reports whether a field of type t carries (or could carry) SSA/Node
// references outside the two sanctioned lists. isEffectBearer admits the
// one sanctioned payload spelling — a direct []SSAUse field, the ImpactBy
// effect-input list. Nested records get no allowance: only the bearer's
// own direct field is its ImpactBy list, so a Lambda record classifies as
// hidden even on an effect bearer. A nil type (untyped) classifies hidden:
// fail-closed — what cannot be proven reference-free is a violation.
func hiddenRefFieldType(t reflect.Type, isEffectBearer bool) bool {
	if t == nil {
		return true // untyped ⇒ unprovable ⇒ hidden (fail-closed)
	}
	ssaUse := reflect.TypeOf(SSAUse{})
	switch t.Kind() {
	case reflect.Struct:
		if t == ssaUse {
			return true // a single use is never an ImpactBy list
		}
		for i := 0; i < t.NumField(); i++ {
			if hiddenRefFieldType(t.Field(i).Type, false) {
				return true
			}
		}
		return false
	case reflect.Map:
		return hiddenRefFieldType(t.Key(), false) || hiddenRefFieldType(t.Elem(), false)
	case reflect.Slice, reflect.Array, reflect.Pointer:
		if t.Kind() == reflect.Slice && t.Elem() == ssaUse && isEffectBearer {
			return false // the ImpactBy effect-input list
		}
		return hiddenRefFieldType(t.Elem(), false)
	case reflect.Interface:
		// A field of interface type t can hold IR nodes iff the Node
		// interface is assignable to it — Node itself, any interface Node
		// satisfies (any superset, any). Interfaces Node does not satisfy
		// (Type, marker faces) stay legal metadata.
		return reflect.TypeOf((*Node)(nil)).Elem().Implements(t)
	}
	return false
}

// checkConvention sweeps every ALLOCATED node's concrete struct fields —
// everything except the embedded *Base machinery cell and the embedded
// operandList (the Input list itself) — and flags any field whose type can
// carry SSA/Node references (D16). Sweeping g.nodes (allocation order)
// rather than the root walk makes the check cover unreachable and forged
// nodes too; a node is its struct shape wherever the graph can see it.
func checkConvention(g *Graph, vs *[]Violation) {
	baseT := reflect.TypeOf((*Base)(nil))
	opsT := reflect.TypeOf(operandList{})
	for _, n := range g.nodes {
		t := reflect.TypeOf(n)
		if t == nil || t.Kind() != reflect.Pointer {
			continue // foreign Node implementations: no struct to inspect
		}
		st := t.Elem()
		_, isEffectBearer := n.(effectBearer)
		for i := 0; i < st.NumField(); i++ {
			f := st.Field(i)
			if f.Anonymous && (f.Type == baseT || f.Type == opsT) {
				continue // the Base cell and the Input list are sanctioned
			}
			if hiddenRefFieldType(f.Type, isEffectBearer) {
				*vs = append(*vs, nodeViolation(n, invConvention,
					fmt.Sprintf("field %q carries SSA/node references outside the Input/ImpactBy lists", f.Name)))
			}
		}
	}
}
