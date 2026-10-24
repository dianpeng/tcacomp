// This file is the frontend's public module object (doc/syntax-design.md §6
// Task 5): ParseModule is the single entry point a driver (CLI, tests)
// calls to go from a source file to typed, buildable kernels — load,
// include resolution, module + per-kernel semantic checks, and fail-fast on
// the first Error-severity diagnostic batch.

package syntax

import (
	"fmt"
	"strings"

	"tcacomp/internal/diagnostic"
	"tcacomp/internal/ir"
	"tcacomp/internal/syntax/ast"
	"tcacomp/internal/syntax/build"
	"tcacomp/internal/syntax/lexer"
)

// Module is the typed module object the build driver consumes: the module's
// kernels in merge order, the module-global runtime annotation (nil when
// none), and the root source path the module was loaded from.
type Module struct {
	Kernels []*Kernel
	Runtime *ast.RuntimeAnn
	Source  string
}

// Kernel is one typed, semachecked kernel ready to lower: it wraps the AST
// node (whose cells semacheck filled) and builds through the pkg rungs.
type Kernel struct {
	ast *ast.Kernel
}

// Name returns the kernel's declared name.
func (k *Kernel) Name() string { return k.ast.Name }

// AST returns the wrapped semachecked kernel AST (the filled cells the
// lowering driver consumes).
func (k *Kernel) AST() *ast.Kernel { return k.ast }

// Build lowers the kernel through the pkg rungs (design §6 Task 4's mapping
// table) and returns the verified graph: pkg.Build's BuildingDCE +
// Materialize + VerifyEach gate has already run, so a returned graph is
// internally consistent. The graph is Level-2 (materialized); a full
// optimization pipeline run must Dematerialize it first (see
// ir.Graph.Dematerialize).
func (k *Kernel) Build() (*ir.Graph, error) {
	return build.Lower(k.ast)
}

// ParseModule loads, include-resolves, and semachecks the module rooted at
// path, failing fast on the first batch of Error-severity diagnostics
// (fail-fast at kernel granularity, design §2.3). The returned module's
// kernels are typed and ready to Build.
func ParseModule(fset *lexer.FileSet, path string) (*Module, error) {
	m, diags := loadModule(fset, path)
	if errs := errorDiags(diags); len(errs) > 0 {
		return nil, fmt.Errorf("frontend: %s:\n%s", path, joinDiags(errs))
	}
	mod := &Module{Runtime: m.Runtime, Source: path}
	for _, k := range m.Kernels {
		mod.Kernels = append(mod.Kernels, &Kernel{ast: k})
	}
	return mod, nil
}

// errorDiags filters the Error-severity diagnostics (the only severity that
// blocks the build; warnings ride along in the returned slice from the
// loader but never fail ParseModule).
func errorDiags(diags []diagnostic.Diagnostic) []diagnostic.Diagnostic {
	var errs []diagnostic.Diagnostic
	for _, d := range diags {
		if d.Severity == diagnostic.Error {
			errs = append(errs, d)
		}
	}
	return errs
}

// joinDiags renders the diagnostic batch, one per line.
func joinDiags(diags []diagnostic.Diagnostic) string {
	lines := make([]string, len(diags))
	for i, d := range diags {
		lines[i] = "  " + d.String()
	}
	return strings.Join(lines, "\n")
}
