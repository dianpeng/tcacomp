// Command tcacomp is the TCA compiler driver: it compiles a frontend source
// file (doc/syntax-design.md) through the full flow — parse, semacheck, lower
// via the pkg rungs, optimize, CUDA C++ codegen — and emits the generated
// translation unit plus a CMakeLists.txt the user builds against the tcacomp
// runtime (the compiler emits code; it does not drive nvcc, PROJECT.md).
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"tcacomp/internal/cg"
	"tcacomp/internal/cg/cxx"
	"tcacomp/internal/opt"
	_ "tcacomp/internal/opt/all" // register every pipeline pass
	"tcacomp/internal/syntax"
	"tcacomp/internal/syntax/lexer"
)

// sm86Options are the A6000 target's emission constraints (cg.Options):
// 100 CTAs of 512 threads, compute capability 8.6.
var sm86Options = cg.Options{MaxCTA: 100, MaxThreadsPerCTA: 512, TargetSM: 86}

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	if len(args) == 0 {
		usage()
		return 2
	}
	switch args[0] {
	case "compile":
		return compile(args[1:])
	case "help", "-h", "--help":
		usage()
		return 0
	default:
		fmt.Fprintf(os.Stderr, "tcacomp: unknown command %q\n", args[0])
		usage()
		return 2
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `usage:
  tcacomp compile [flags] <input.tca>   compile a TCA source file to CUDA C++

flags:
`)
	fs := flag.NewFlagSet("compile", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.PrintDefaults()
}

// compile implements `tcacomp compile <input.tca>`: every kernel in the
// module is lowered, optimized, and emitted into the output directory.
func compile(args []string) int {
	fs := flag.NewFlagSet("compile", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	outDir := fs.String("o", "tcacomp-out", "output directory for the emitted `<name>.cu` + `CMakeLists.txt`")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "tcacomp compile: exactly one input file required\n")
		return 2
	}
	src := fs.Arg(0)

	mod, err := syntax.ParseModule(lexer.NewFileSet(), src)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tcacomp: %v\n", err)
		return 1
	}
	if len(mod.Kernels) == 0 {
		fmt.Fprintf(os.Stderr, "tcacomp: %s: no kernels\n", src)
		return 1
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "tcacomp: create output dir: %v\n", err)
		return 1
	}

	for _, k := range mod.Kernels {
		if err := compileKernel(k, *outDir); err != nil {
			fmt.Fprintf(os.Stderr, "tcacomp: kernel %s: %v\n", k.Name(), err)
			return 1
		}
	}
	return 0
}

// compileKernel builds one kernel through the pkg rungs, runs the full
// optimization pipeline over the dematerialized graph, and writes the
// emitted files into outDir.
func compileKernel(k *syntax.Kernel, outDir string) error {
	g, err := k.Build()
	if err != nil {
		return err
	}
	// pkg.Build materializes as its final construction step; the pipeline's
	// scheduling/tiling passes consume the Level-1 sea, so dematerialize
	// first (the pipeline re-materializes for codegen).
	if err := g.Dematerialize(); err != nil {
		return err
	}
	if err := opt.NewManager(opt.BuildPipeline()).Run(g); err != nil {
		return err
	}
	res, err := cxx.Emit(g, sm86Options)
	if err != nil {
		return err
	}
	for name, content := range res.Files {
		dst := filepath.Join(outDir, name)
		if err := os.WriteFile(dst, content, 0o644); err != nil {
			return fmt.Errorf("write %s: %w", dst, err)
		}
	}
	fmt.Printf("tcacomp: compiled %s -> %s\n", k.Name(), outDir)
	return nil
}
