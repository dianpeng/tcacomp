package ast

import "fmt"

// Visitor is the go/ast-shaped tree walker callback (design §3.1): Visit is
// invoked for each node; the returned visitor replaces v for that node's
// children (returning nil prunes the subtree).
type Visitor interface {
	Visit(n Node) Visitor
}

// Walk traverses n in source order — declaration order, binding order,
// operand order — so every diagnostic printer and future formatter is stable
// (design §3.1). A nil node is a no-op.
func Walk(v Visitor, n Node) {
	if v == nil || n == nil {
		return
	}
	switch n := n.(type) {
	case *Module:
		walkList(v, n.Includes)
		if n.Runtime != nil {
			Walk(v, n.Runtime)
		}
		walkList(v, n.Kernels)
	case *Include:
	case *RuntimeAnn:
		walkList(v, n.Args)
	case *Annotation:
		walkList(v, n.Args)
	case *KeyedConst:
	case *Kernel:
		walkList(v, n.Anns)
		walkList(v, n.Params)
		if n.Ret != nil {
			Walk(v, n.Ret)
		}
		walkList(v, n.Body)
	case *Param:
		if n.Type != nil {
			Walk(v, n.Type)
		}
	case *TensorType:
		Walk(v, &n.Elem)
		for i := range n.Shape {
			Walk(v, &n.Shape[i])
		}
	case *ScalarType:
	case *ShapeDim:
	case *DimVarDecl:
	case *LetDecl:
		if n.Type != nil {
			Walk(v, n.Type)
		}
		Walk(v, n.Chain)
	case *ReturnStmt:
	case *StageChain:
		walkList(v, n.Stages)
	case *MapStage:
		walkList(v, n.Bindings)
		Walk(v, n.Lambda)
	case *ReduceStage:
		if n.Init != nil {
			Walk(v, n.Init)
		}
		walkList(v, n.OnParams)
		Walk(v, &n.RetType)
		Walk(v, n.Body)
	case *Binding:
	case *Lambda:
		walkList(v, n.Params)
		if n.RetType != nil {
			Walk(v, n.RetType)
		}
		Walk(v, n.Body)
	case *ParamSpec:
		if n.Type != nil {
			Walk(v, n.Type)
		}
	case *BasicLit:
	case *NameRef:
	case *PipeRef:
	case *BinaryExpr:
		Walk(v, n.L)
		Walk(v, n.R)
	case *UnaryExpr:
		Walk(v, n.X)
	case *SelectExpr:
		Walk(v, n.Cond)
		Walk(v, n.Then)
		Walk(v, n.Else)
	case *CallExpr:
		walkList(v, n.Args)
	case *CastExpr:
		Walk(v, n.X)
	default:
		// Unreachable for the closed node set; a future node kind that
		// forgets to register here must fail loudly rather than silently
		// skip its children.
		panic("ast.Walk: unknown node type " + fmt.Sprintf("%T", n))
	}
}

// walkList walks a homogeneous node list in order.
func walkList[T Node](v Visitor, ns []T) {
	for _, n := range ns {
		Walk(v, n)
	}
}
