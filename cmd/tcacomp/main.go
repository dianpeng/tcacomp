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
	"tcacomp/internal/compiler"
	"tcacomp/internal/syntax"
	"tcacomp/internal/syntax/lexer"
)

// sm86Options are the A6000 target's emission constraints (cg.Options):
// compute capability 8.6 and the §4.3 MaxCTA policy constant 512 (the
// gridDim cap ValidateLaunch checks — the launch DIMENSIONS themselves
// are annotation-derived per fix4 §5.1, never supplied here).
var sm86Options = cg.Options{MaxCTAPolicy: 512, TargetSM: 86}

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

// compileKernel compiles one kernel through the compiler facade (the single
// syntax→backend composition — parse was already done; fix2 R-F) and writes
// the emitted files into outDir.
func compileKernel(k *syntax.Kernel, outDir string) error {
	res, err := compiler.CompileKernel(k, sm86Options)
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
