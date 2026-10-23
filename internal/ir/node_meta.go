package ir

// node_meta.go holds KindKernelMeta (quick-261023-cgq, the author's
// instrumentation ruling): the graph-owned metadata node carrying the
// FRONTEND-specified emitted symbol names — the kernel name and the host
// launcher name — from construction to codegen. The frontend syntax is not
// locked yet; this node IS the carry contract, so the builder (and later
// the parser) sets it and the backend reads it. Names are STRING PAYLOADS
// (the runtimeQueryNode.field / callNode.method precedent), never operands.
//
//   - Zero operands (arity 0), pure metadata: never a statement, never
//     scheduled — the schedule pass anchors it in the Prototype block
//     beside the InputParam placeholders, and codegen reads it there.
//   - The graph holds it in a dedicated field (NewKernelMeta is the sole
//     producer; a second call replaces the first), and the Level-1 DCE
//     seeds it as a root — an unreferenced metadata node is never swept.
//   - An empty payload means "derive the name at codegen" (the Google-style
//     PascalCase fallback over the graph name), so the frontend may pin
//     only one of the two symbols.

// kernelMetaNode: the kernel naming metadata; string payloads only.
type kernelMetaNode struct {
	*Base
	operandList
	kernelName   string
	launcherName string
}

// NewKernelMeta sets the graph's emitted-name metadata: the __global__
// kernel symbol and the extern "C" launcher symbol. Empty strings defer to
// the backend's derivation. The node is anchored at PrototypeSN by the
// schedule pass; it is never a kernel parameter.
func (g *Graph) NewKernelMeta(origin, kernelName, launcherName string) Node {
	n := g.allocNode(KindKernelMeta, origin, 0).(*kernelMetaNode)
	n.kernelName = kernelName
	n.launcherName = launcherName
	g.meta = n
	return n
}

// KernelNamesOf returns the metadata node's (kernel, launcher) payloads and
// whether n is a KernelMeta node — the exported read face (the payload
// read-face precedent, node_expr.go), so codegen never asserts the concrete
// node type.
func KernelNamesOf(n Node) (string, string, bool) {
	if m, ok := n.(*kernelMetaNode); ok {
		return m.kernelName, m.launcherName, true
	}
	return "", "", false
}

// Meta returns the graph's kernel naming metadata node, or nil when the
// frontend set none (codegen derives both symbols).
func (g *Graph) Meta() Node {
	if g == nil {
		return nil
	}
	return g.meta
}
