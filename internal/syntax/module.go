// Package syntax is the TCA frontend's module loader (doc/syntax-design.md
// §2.3, §4.1, design Task 3/5): it reads one root source file, lexes and
// parses it, resolves its @include graph (relative paths only, cycle
// detection, include-guard dedup), merges the kernel tables, and runs the
// module-scope and per-kernel semantic checks. The returned diagnostics are
// the complete fail-fast report; the merged module tree is best-effort.
package syntax

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"tcacomp/internal/diagnostic"
	"tcacomp/internal/syntax/ast"
	"tcacomp/internal/syntax/lexer"
	"tcacomp/internal/syntax/parser"
	"tcacomp/internal/syntax/semacheck"
)

// loadModule loads the module rooted at path (read via os, so path may be
// absolute or relative to the process working directory): the root file's
// include graph is resolved relative to each including file — include paths
// must be relative and must not escape the including file's directory
// (threat T-k2i-01) — and the merged module is module-checked (rules 1, 3,
// 4) and per-kernel semachecked (rules 5-26 + §5.1 resolution, whose filled
// cells the build driver consumes). The exported entry point is ParseModule
// (api.go), which wraps this loader and fails fast on diagnostics.
//
// The returned module holds every parsed kernel in merge order (include
// files depth-first, then the root's own kernels); diagnostics are returned
// separately, and any Error-severity diagnostic means the corresponding
// declaration must be treated as absent (fail-fast at kernel granularity).
func loadModule(fset *lexer.FileSet, path string) (*ast.Module, []diagnostic.Diagnostic) {
	ld := &loader{
		fset:      fset,
		processed: map[string]bool{},
	}
	root, diags := ld.load(path, map[string]bool{filepath.Clean(path): true})
	if root == nil {
		return &ast.Module{}, diags
	}
	diags = append(diags, ld.resolveIncludes(root, path, map[string]bool{filepath.Clean(path): true})...)
	merged := &ast.Module{
		Runtime: root.Runtime,
		Kernels: root.Kernels,
	}
	for _, inc := range root.Includes {
		merged.Includes = append(merged.Includes, inc)
	}
	// Include contributions land after the root's own kernels in merge
	// order; a runtime on an included file fills in only when the root
	// declares none (the root file owns the build configuration).
	if merged.Runtime == nil {
		merged.Runtime = ld.firstRuntime
	}
	diags = append(diags, semacheck.CheckModule(merged)...)
	for _, k := range merged.Kernels {
		diags = append(diags, semacheck.Check(k)...)
	}
	return merged, diags
}

// loader carries one ParseModule session's include graph state.
type loader struct {
	fset         *lexer.FileSet
	processed    map[string]bool // cleaned paths already merged (include guard)
	firstRuntime *ast.RuntimeAnn
}

// load reads, lexes, and parses one file, returning its module tree (include
// requests unresolved).
func (ld *loader) load(path string, stack map[string]bool) (*ast.Module, []diagnostic.Diagnostic) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, []diagnostic.Diagnostic{{
			Severity: diagnostic.Error,
			Span:     diagnostic.Span{File: path},
			Message:  fmt.Sprintf("cannot read source file: %v", err),
		}}
	}
	f := ld.fset.AddFile(path, string(src))
	return parser.Parse(f)
}

// resolveIncludes walks one file's include requests depth-first, merging each
// included module's kernels after the root's (deterministic include order)
// and validating the threat-model include policy.
func (ld *loader) resolveIncludes(m *ast.Module, path string, stack map[string]bool) []diagnostic.Diagnostic {
	var diags []diagnostic.Diagnostic
	for _, inc := range m.Includes {
		resolved, d := resolveIncludePath(inc, path)
		if d != nil {
			diags = append(diags, *d)
			continue
		}
		inc.Resolved = resolved
		if stack[resolved] {
			diags = append(diags, diagnostic.Diagnostic{
				Severity: diagnostic.Error,
				Span:     spanOf(inc.Pos()),
				Message:  fmt.Sprintf("@include cycle: %s is already being loaded", resolved),
			})
			continue
		}
		if ld.processed[resolved] {
			continue // include guard: a diamond-included file merges once
		}
		sub, subDiags := ld.load(resolved, nil)
		diags = append(diags, subDiags...)
		if sub == nil {
			continue // missing file: load reported it
		}
		if sub.Runtime != nil && ld.firstRuntime == nil {
			ld.firstRuntime = sub.Runtime
		}
		ld.processed[resolved] = true
		inner := map[string]bool{}
		for k := range stack {
			inner[k] = true
		}
		inner[resolved] = true
		diags = append(diags, ld.resolveIncludes(sub, resolved, inner)...)
		m.Kernels = append(m.Kernels, sub.Kernels...)
	}
	return diags
}

// resolveIncludePath validates and resolves one include path relative to the
// including file's directory. Threat T-k2i-01: relative paths only, and no
// escaping the including file's directory.
func resolveIncludePath(inc *ast.Include, includingFile string) (string, *diagnostic.Diagnostic) {
	bad := func(msg string) *diagnostic.Diagnostic {
		return &diagnostic.Diagnostic{
			Severity: diagnostic.Error,
			Span:     spanOf(inc.Pos()),
			Message:  msg,
		}
	}
	if filepath.IsAbs(inc.Path) {
		return "", bad(fmt.Sprintf("@include path %q must be relative to the including file (threat T-k2i-01)", inc.Path))
	}
	dir := filepath.Dir(includingFile)
	resolved := filepath.Clean(filepath.Join(dir, inc.Path))
	within := resolved == dir || strings.HasPrefix(resolved, dir+string(filepath.Separator))
	if !within {
		return "", bad(fmt.Sprintf("@include path %q escapes the including file's directory (threat T-k2i-01)", inc.Path))
	}
	if _, err := os.Stat(resolved); err != nil {
		return "", bad(fmt.Sprintf("missing include file %q: %v", inc.Path, err))
	}
	return resolved, nil
}

func spanOf(pos lexer.Pos) diagnostic.Span {
	span := diagnostic.Span{Line: pos.Line, Col: pos.Col}
	if pos.File != nil {
		span.File = pos.File.Name
	}
	return span
}
