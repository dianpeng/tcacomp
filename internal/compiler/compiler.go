// Package compiler is the user-facing FACADE (fix2.md R-F/D-g, the
// 2026-10-25 author ruling): ONE entry performing the syntax → backend-code
// transformation, built on the syntax module and the pipeline module. The
// facade owns NO pass logic and NO emission logic — it is exactly the
// composition syntax.Build → Dematerialize → pipeline.Run → cg/cxx.Emit,
// threaded with the backend options. External users do not care about pass
// composition; cmd/tcacomp and the e2e harness are thin consumers of this
// package, never re-assembly points.
package compiler

import (
	"fmt"

	"tcacomp/internal/cg"
	"tcacomp/internal/cg/cxx"
	"tcacomp/internal/pipeline"
	"tcacomp/internal/syntax"
	"tcacomp/internal/syntax/lexer"
)

// CompileKernel compiles one semachecked frontend kernel into its backend
// files: the pkg-rungs Build (the returned graph materializes as its final
// construction step), Dematerialize (the pass pipeline consumes the Level-1
// sea), the full pass composition (pipeline.Run — the HIR and LOWER windows
// plus the cleanup tail), and the CUDA C++ emission.
func CompileKernel(k *syntax.Kernel, opts cg.Options) (*cg.Result, error) {
	g, err := k.Build()
	if err != nil {
		return nil, err
	}
	// pkg.Build materializes as its final construction step; the pipeline's
	// scheduling/tiling passes consume the Level-1 sea, so dematerialize
	// first (the pipeline's tail re-attaches the AST the emitter reads).
	if err := g.Dematerialize(); err != nil {
		return nil, err
	}
	if err := pipeline.Run(g); err != nil {
		return nil, err
	}
	return cxx.Emit(g, opts)
}

// CompileFile parses and semachecks the source file at path, then compiles
// every kernel in the module, merging the per-kernel emissions into one
// Result. A file name emitted by two kernels with DIFFERING content is a
// named error (deterministic fail-closed); identical duplicates collapse.
func CompileFile(path string, opts cg.Options) (*syntax.Module, *cg.Result, error) {
	mod, err := syntax.ParseModule(lexer.NewFileSet(), path)
	if err != nil {
		return nil, nil, err
	}
	if len(mod.Kernels) == 0 {
		return mod, nil, fmt.Errorf("compiler: %s: no kernels", path)
	}
	merged := &cg.Result{Files: map[string][]byte{}}
	for _, k := range mod.Kernels {
		res, err := CompileKernel(k, opts)
		if err != nil {
			return mod, nil, fmt.Errorf("kernel %s: %w", k.Name(), err)
		}
		for name, content := range res.Files {
			if prev, ok := merged.Files[name]; ok {
				if string(prev) != string(content) {
					return mod, nil, fmt.Errorf("compiler: kernels emit conflicting content for %s", name)
				}
				continue
			}
			merged.Files[name] = content
		}
	}
	return mod, merged, nil
}
