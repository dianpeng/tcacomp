package ir

// node_var.go holds the graph-owned variable nodes (KindInputParam) and the
// ParamRole taxonomy. Kernel roles (RoleDim, RoleTensor, RoleScalar) are
// owned by the Graph's param list; RoleLambda marks lambda-bound parameters,
// which are graph-external by design (the review's IR modeling: lambda
// parameters are InputParams, inlined at lowering).

// ParamRole discriminates InputParam nodes. Kernel roles (RoleDim,
// RoleTensor, RoleScalar) are owned by the Graph's param list; RoleLambda
// marks lambda-bound parameters, which are graph-external by design (the
// review's IR modeling: lambda parameters are InputParams, inlined at
// lowering).
type ParamRole int

const (
	RoleDim    ParamRole = iota // shape dimension (runtime shape symbol; untyped)
	RoleTensor                  // tensor parameter (interned tensor type)
	RoleScalar                  // scalar parameter (interned scalar type)
	RoleLambda                  // lambda-bound parameter (map/reduce lambda argument)
)

// inputParamNode is a kernel or lambda input parameter (review P1R-01): one
// graph-owned node per parameter, referenced exclusively through SSAUse
// records. name/typ/role are debug/diagnostic payload — NEVER resolution or
// identity. ops is empty (arity 0); the SSAVar IS the parameter's def.
type inputParamNode struct {
	*Base
	operandList
	name string
	typ  Type // nil for dim params (dims are not typed values in v0)
	role ParamRole
}

// NewInputParam creates one graph-owned kernel input parameter (dim, tensor,
// or scalar) and appends it to the graph's param list. The returned node IS
// the parameter's SSADef pointee: every reference to it elsewhere in the
// graph is an SSAUse record; the name is debug payload only (review P1R-01).
func (g *Graph) NewInputParam(origin string, role ParamRole, name string, typ Type) Node {
	n := g.allocNode(KindInputParam, origin, 0).(*inputParamNode)
	n.name, n.typ, n.role = name, typ, role
	g.params = append(g.params, n)
	return n
}

// NewLambdaInputParam creates a lambda-bound InputParam node (role
// RoleLambda). Like kernel parameters it is an SSA def referenced only
// through SSAUse records, but it is NOT registered in the graph's param
// list — lambda params are graph-external by design (review P1R-01; they are
// inlined at lowering).
func (g *Graph) NewLambdaInputParam(origin, name string, typ Type) Node {
	n := g.allocNode(KindInputParam, origin, 0).(*inputParamNode)
	n.name, n.typ, n.role = name, typ, RoleLambda
	return n
}

// RoleOf returns n's parameter role when n is an InputParam node, and
// whether it is one. It is the exported read face of the role taxonomy —
// passes (e.g. the scheduler, the sea-root set) discriminate params by role
// without asserting the concrete node type.
func RoleOf(n Node) (ParamRole, bool) {
	ip, ok := n.(*inputParamNode)
	if !ok {
		return 0, false
	}
	return ip.role, true
}
