// Package visualization emits Graphviz DOT dumps of the IR (TOOL-03, D-05):
// text-only emission mirroring the canonical text printer's SSA numbering
// and node rendering, with one cluster per statement and explicit def-to-user
// edges. No graphviz library and no `dot` binary dependency — rendering is a
// dev-only tool; the compiler only ever produces the .dot text.
package visualization

import (
	"bufio"
	"fmt"
	"io"

	"tcacomp/internal/ir"
)

// EmitDot writes g as a deterministic DOT digraph to w: `digraph <name> {`,
// then per statement a `subgraph cluster_<stmtID>` (the cluster prefix is the
// verified Graphviz syntactic convention for grouping), containing one
// `n<id> [label=<canonical>];` per node and `n<defID> -> n<userID>;` edges in
// operand order. Node order and label text come from the text printer's own
// walk and rendering (ir.StmtNodes / ir.Label under ir.ComputeNumbering), so
// .dot output mirrors the text numbering exactly (D-05); edges are explicit
// use-def records, including the chain pipe inputs the text printer renders
// as stage-local `%0`. Fully deterministic: statement order, operand order,
// no map iteration (RESEARCH Pitfall 1).
//
// DOT emission is a Level-2 surface: the clusters ARE the statement list, so
// an unmaterialized graph returns an error instead of a misleading empty
// digraph (call Materialize first).
func EmitDot(w io.Writer, g *ir.Graph) error {
	if g == nil {
		return nil
	}
	if !g.Materialized() {
		return fmt.Errorf("visualization: graph %q is not materialized", g.Name())
	}
	bw := bufio.NewWriter(w)
	num := ir.ComputeNumbering(g)
	fmt.Fprintf(bw, "digraph %s {\n", g.Name())
	for stmt := g.Block().First(); stmt != nil; stmt = stmt.Next() {
		fmt.Fprintf(bw, "  subgraph cluster_%d {\n", stmt.ID())
		fmt.Fprintf(bw, "    label=\"statement #%d\";\n", stmt.ID())
		nodes := ir.StmtNodes(stmt)
		for _, n := range nodes {
			fmt.Fprintf(bw, "    n%d [label=%q];\n", n.BasePtr().ID(), ir.Label(n, num))
		}
		for _, n := range nodes {
			for _, op := range ir.OperandsOf(n) {
				if op == nil {
					continue // unwired slot: verifier owns rejection, DOT stays total
				}
				fmt.Fprintf(bw, "    n%d -> n%d;\n", op.BasePtr().ID(), n.BasePtr().ID())
			}
			for _, body := range ir.LambdaBodies(n) {
				if body == nil {
					continue
				}
				fmt.Fprintf(bw, "    n%d -> n%d;\n", body.BasePtr().ID(), n.BasePtr().ID())
			}
		}
		fmt.Fprintf(bw, "  }\n")
	}
	fmt.Fprintf(bw, "}\n")
	return bw.Flush()
}
