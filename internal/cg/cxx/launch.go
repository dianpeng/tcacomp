package cxx

import (
	"fmt"
	"strings"

	"tcacomp/internal/ir"
)

// launcher emits the USER-FACING host wrapper (doc/cg.md §9, the Tensor
// contract protocol — quick-261023-ten): every tensor parameter arrives as
// tca::Tensor<T, Rank> (device memory + shape + layout in one object), the
// kernel's dim parameters are DERIVED from the tensors and cross-validated
// (the mutually understood contract: every occurrence of a dim name must
// agree), scalars pass through, and a dim no tensor carries stays an
// explicit uint32_t argument. Launch totals are baked from the tiling
// Params (LR-15), the runtime validates divisibility, the dynamic-smem
// opt-in rides the > 48 KB path (D5), and a CUDA_CHECK follows the launch.
// The kernel itself lives in the anonymous namespace — the launcher is the
// TU's only exported symbol (the scalfold engine precedent).
func (e *emitter) launcher() (string, error) {
	// dimSuppliers: every (tensor param index, axis) a dim name appears at,
	// in signature order — the first supplies the value, the rest are
	// agreement checks. A dim with no tensor occurrence stays an explicit
	// launcher argument.
	type site struct {
		tensor ir.Node
		param  int // k.Params index of the tensor
		axis   int
	}
	sites := map[string][]site{}
	var tensorOrder []ir.Node
	for pi, p := range e.k.Params {
		role, _ := ir.RoleOf(p)
		if role != ir.RoleTensor {
			continue
		}
		_, dims, err := e.tensorTypeOf(p)
		if err != nil {
			return "", err
		}
		tensorOrder = append(tensorOrder, p)
		for ai, d := range dims {
			if d.Extent > 0 {
				continue // constant extent — nothing to derive or check
			}
			name := d.Name
			if _, seen := sites[name]; !seen {
				sites[name] = nil
			}
			sites[name] = append(sites[name], site{tensor: p, param: pi, axis: ai})
		}
	}

	// Launcher parameters: scalars, then tensors (const& for read-only, &
	// for stored outputs), then free dims, then the stream.
	var params []string
	var callArgs []string // the kernel call's argument expressions
	var body []string
	derived := map[string]string{} // dim name → the derived local
	for _, p := range e.k.Params {
		role, _ := ir.RoleOf(p)
		name := ir.InputParamNameOf(p)
		switch role {
		case ir.RoleDim:
			if ss := sites[name]; len(ss) > 0 {
				// Derived from the tensors below; the kernel call reuses it.
				callArgs = append(callArgs, name)
				continue
			}
			params = append(params, "uint32_t "+name)
			callArgs = append(callArgs, name)
		case ir.RoleScalar:
			ts, err := scalarTypeOfParam(p)
			if err != nil {
				return "", err
			}
			params = append(params, ts+" "+name)
			callArgs = append(callArgs, name)
		case ir.RoleTensor:
			ts, dims, err := e.tensorTypeOf(p)
			if err != nil {
				return "", err
			}
			if e.stored[p] {
				params = append(params, fmt.Sprintf("tca::Tensor<%s, %du>& %s", ts, uint(len(dims)), name))
			} else {
				params = append(params, fmt.Sprintf("const tca::Tensor<%s, %du>& %s", ts, uint(len(dims)), name))
			}
			callArgs = append(callArgs, name+".Data()")
		}
	}
	// Derive each dim from its first tensor site; assert agreement with
	// every other site (the contract check — mismatched shapes are a hard
	// user error, named per dim).
	for _, p := range tensorOrder {
		_, dims, err := e.tensorTypeOf(p)
		if err != nil {
			return "", err
		}
		for ai, d := range dims {
			if d.Extent > 0 {
				continue
			}
			if _, ok := derived[d.Name]; ok {
				continue
			}
			derived[d.Name] = ir.InputParamNameOf(p) // the supplier site
			body = append(body, fmt.Sprintf("  const uint32_t %s = %s.Dim(%du);", d.Name, ir.InputParamNameOf(p), uint(ai)))
		}
	}
	for _, p := range tensorOrder {
		_, dims, err := e.tensorTypeOf(p)
		if err != nil {
			return "", err
		}
		tn := ir.InputParamNameOf(p)
		for ai, d := range dims {
			if d.Extent > 0 {
				continue
			}
			if derived[d.Name] == tn {
				continue // the supplier site defines it — nothing to check
			}
			body = append(body, fmt.Sprintf("  TCA_REQUIRE(%s.Dim(%du) == %s, \"%s.Dim(%d) must equal %s (dim %s)\");",
				tn, uint(ai), d.Name, tn, ai, d.Name, d.Name))
		}
	}

	params = append(params, "cudaStream_t tca_stream")
	var b strings.Builder
	fmt.Fprintf(&b, "void %s(%s) {\n", e.launcherSym, strings.Join(params, ", "))
	for _, l := range body {
		b.WriteString(l + "\n")
	}
	fmt.Fprintf(&b, "  tca::rt::ValidateLaunch(dim3(%du, 1u, 1u), dim3(%du, 1u, 1u), %du);\n",
		uint(e.opts.MaxCTA), uint(e.opts.MaxThreadsPerCTA), uint(e.smemCap))
	if e.smemCap > 48*1024 {
		fmt.Fprintf(&b, "  cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, %du);\n",
			e.kernelSym, uint(e.smemCap))
	}
	fmt.Fprintf(&b, "  %s<<<dim3(%du, 1u, 1u), dim3(%du, 1u, 1u), %du, tca_stream>>>(%s);\n",
		e.kernelSym, uint(e.opts.MaxCTA), uint(e.opts.MaxThreadsPerCTA), uint(e.smemCap), strings.Join(callArgs, ", "))
	fmt.Fprintf(&b, "  TCA_CUDA_CHECK(cudaGetLastError());\n")
	fmt.Fprintf(&b, "}\n")
	return b.String(), nil
}

// cmake emits the generated CMakeLists.txt (ruling D1; the stack decision
// "emit .cu + CMake, not raw nvcc"): CUDA_ARCHITECTURES pinned to the
// target SM — never "native" (reproducibility) — C++20 both languages, and
// a link against the tcacomp runtime target.
func (e *emitter) cmake() (string, error) {
	sm := e.opts.TargetSM
	if sm == 0 {
		sm = 86 // the single-target default (A6000); explicit is better
	}
	if sm < 30 || sm > 99 {
		return "", fmt.Errorf("cxx: TargetSM %d is not a plausible major*100+minor", sm)
	}
	var b strings.Builder
	b.WriteString("# generated by tcacomp — do not edit\n")
	b.WriteString("cmake_minimum_required(VERSION 3.24)\n")
	fmt.Fprintf(&b, "project(%s LANGUAGES CXX CUDA)\n", e.kernelSym)
	b.WriteString("set(CMAKE_CXX_STANDARD 20)\n")
	b.WriteString("set(CMAKE_CXX_STANDARD_REQUIRED ON)\n")
	b.WriteString("set(CMAKE_CUDA_STANDARD 20)\n")
	b.WriteString("set(CMAKE_CUDA_STANDARD_REQUIRED ON)\n")
	fmt.Fprintf(&b, "add_library(%s OBJECT %s.cu)\n", e.kernelSym, e.name)
	fmt.Fprintf(&b, "set_target_properties(%s PROPERTIES CUDA_ARCHITECTURES %d)\n", e.kernelSym, sm)
	fmt.Fprintf(&b, "target_compile_options(%s PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)\n", e.kernelSym)
	fmt.Fprintf(&b, "target_link_libraries(%s PRIVATE tcacomp_runtime CUDA::cudart)\n", e.kernelSym)
	return b.String(), nil
}
