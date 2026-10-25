package cxx

import (
	"fmt"
	"strings"

	"tcacomp/internal/cg"
	"tcacomp/internal/ir"
)

// emitter is the walk state (doc/cg.md §4): the AST walk, the on-the-fly
// SSA symbol table (the seed doc's point 3 — SSAUse is not resolved, so the
// emitter generates local symbols as it walks), and the deterministic
// counters behind every name. One emitter per Emit call.
type emitter struct {
	g        *ir.Graph
	k        *ir.ASTKernel
	opts     cg.Options
	name     string // emitted symbol base name
	comments cg.CommentLevel

	sb   strings.Builder  // kernel body text
	out  *strings.Builder // current emission target (kernel body vs a lambda function)
	deep int              // brace depth (indentation)

	// pre-scanned facts (deterministic: allocation / declaration order).
	stored    map[ir.Node]bool   // tensor params written by a GMEM MemStore
	buffers   []ir.Node          // live SMEM-class MemNew, allocation order
	frags     []ir.Node          // live FRAG-class MemNew, allocation order
	smemOff   map[ir.Node]int    // SMEM-class MemNew → pool byte offset (v0: disjoint, D8)
	smemCap   int                // total pool bytes (v0: Σ extents, zero reuse)
	fillSrc   map[ir.Node]bool   // MemLoads serving as a fill's source (skipped as items)
	bufTensor map[ir.Node]string // tiered MemNew → source tensor name (semantic view names)
	staging   map[ir.Node]bool   // SMEM-class MemNew nodes serving as reducer staging rows (R1 — inline ctor views, no kernelTop decl)
	gated     map[ir.Node]bool   // MemStores of a reduced value (R3 — single-owner gated on is_first_in_cta)

	// symbols.
	tensorView   map[ir.Node]string   // RoleTensor param → TensorView local
	smemView     map[ir.Node]string   // MemNewSMem → SmemView local
	fragLocal    map[ir.Node]string   // FragNew → Frag local
	iters        map[ir.Node]string   // DimVar node → owning loop's iterator local
	dimScope     []map[ir.Node]string // dim InputParam → iterator local, loop-scoped stack
	objName      map[ir.Node]string   // runtime construction → local
	lambdas      []ir.Node            // surviving KindLambda in first-reference order
	lambdaOwned  map[ir.Node]bool     // the operand closure of every surviving lambda — skipped in the kernel walk (hoisted file-scope instead)
	lambdaParams map[ir.Node]string   // the lambda currently being emitted: param → local
	curGrid      string               // the enclosing LNT's GridMapping local
	curThread    string               // the enclosing LNT's ThreadMapping local

	// symbol derivation (quick-261023-cgq): the KindKernelMeta payloads or
	// the Google-style PascalCase fallback; collision-safe local naming.
	kernelSym   string
	launcherSym string
	paramNames  map[string]bool   // kernel signature names — the collision guard
	usedNames   map[string]bool   // every emitted local
	pureRef     map[ir.Node]bool  // pure items actually referenced (B1: skip the dead)
	cseText     map[string]string // pure expr text → its first local (B2)

	// value environments are function-scoped (kernel body vs each hoisted
	// lambda): consts and materialized locals never cross function
	// boundaries even when the SSA node is shared.
	values map[ir.Node]string // materialized value → local
	valN   int
	constN int
	itN    int
	mapN   int // mapping construction counter
	redN   int // reducer construction counter
}

// Emit produces the full trio (doc/cg.md §2, ruling D1): "<name>.cu" with
// the __global__ kernel plus its extern "C" host launcher, and
// "CMakeLists.txt" building it against the tcacomp runtime. It consumes the
// graph read-only.
func Emit(g *ir.Graph, opts cg.Options) (*cg.Result, error) {
	if g == nil {
		return nil, fmt.Errorf("cxx: nil graph")
	}
	k := g.AST()
	if k == nil {
		return nil, fmt.Errorf("cxx: no attached ASTKernel — the schedule and final-lower passes must run first")
	}
	name := opts.KernelName
	if name == "" {
		name = g.Name()
	}
	if name == "" {
		return nil, fmt.Errorf("cxx: empty kernel name (set Options.KernelName)")
	}
	e := &emitter{
		g: g, k: k, opts: opts, name: name,
		stored:      map[ir.Node]bool{},
		smemOff:     map[ir.Node]int{},
		fillSrc:     map[ir.Node]bool{},
		bufTensor:   map[ir.Node]string{},
		staging:     map[ir.Node]bool{},
		gated:       map[ir.Node]bool{},
		tensorView:  map[ir.Node]string{},
		smemView:    map[ir.Node]string{},
		fragLocal:   map[ir.Node]string{},
		iters:       map[ir.Node]string{},
		objName:     map[ir.Node]string{},
		lambdaOwned: map[ir.Node]bool{},
		paramNames:  map[string]bool{},
		usedNames:   map[string]bool{},
		pureRef:     map[ir.Node]bool{},
		cseText:     map[string]string{},
		values:      map[ir.Node]string{},
	}
	e.out = &e.sb
	e.comments = opts.Comments.Resolved()
	if err := e.gates(); err != nil {
		return nil, err
	}
	if err := e.deriveSymbols(); err != nil {
		return nil, err
	}
	if err := e.prescan(); err != nil {
		return nil, err
	}

	sig, err := e.signature()
	if err != nil {
		return nil, err
	}
	// Body statements emit one indent level into the function (the kernel
	// top declarations set the level; quick-261023-ten's indentation fix).
	e.deep = 1
	top, err := e.kernelTop()
	if err != nil {
		return nil, err
	}
	if err := e.walkBody(); err != nil {
		return nil, err
	}
	e.deep = 0
	lamText, err := e.lambdaSection()
	if err != nil {
		return nil, err
	}
	launch, err := e.launcher()
	if err != nil {
		return nil, err
	}

	var cu strings.Builder
	fmt.Fprintf(&cu, "// generated by tcacomp — deterministic output; do not edit\n")
	fmt.Fprintf(&cu, "#include \"tca/runtime.h\"\n\n")
	cu.WriteString(lamText)
	if e.comments >= cg.CommentSections {
		fmt.Fprintf(&cu, "// %s: launch %d CTAs x %d threads (tiling Params; the runtime asserts divisibility)\n",
			e.kernelSym, e.opts.MaxCTA, e.opts.MaxThreadsPerCTA)
		if e.smemCap > 0 {
			fmt.Fprintf(&cu, "// dynamic SMEM pool: %d B (v0: disjoint offsets, zero reuse)\n", e.smemCap)
		}
	}
	// The kernel lives in the anonymous namespace — the launcher is the
	// TU's only exported symbol (quick-261023-ten, the scalfold engine
	// precedent: kernels are internal, wrappers are the surface).
	fmt.Fprintf(&cu, "namespace {\n\n")
	fmt.Fprintf(&cu, "__global__ void __launch_bounds__(%d) %s(%s) {\n", e.opts.MaxThreadsPerCTA, e.kernelSym, sig)
	cu.WriteString(top)
	cu.WriteString(e.sb.String())
	cu.WriteString("}\n\n")
	cu.WriteString("}  // namespace\n\n")
	cu.WriteString(launch)

	cmake, err := e.cmake()
	if err != nil {
		return nil, err
	}
	return &cg.Result{Files: map[string][]byte{
		e.name + ".cu":   []byte(cu.String()),
		"CMakeLists.txt": []byte(cmake),
	}}, nil
}

// gates is the fail-closed entry surface (doc/cg.md §4): the whitelist of
// §1.2 (kinds eliminated before codegen must not be live) and the v0
// emission boundary — GMEM temporaries (KindMemNew) have no fixture and no
// launcher contract yet, so they fail closed instead of emitting garbage.
func (e *emitter) gates() error {
	banned := map[ir.Kind]string{
		ir.KindMap:                "map (final-lower Map Lower must eliminate it)",
		ir.KindReduce:             "reduce (final-lower Reducer Lower must eliminate it)",
		ir.KindIter:               "iter (tile-lower must retire it)",
		ir.KindIterCluster:        "itercluster (simplify-lower must eliminate it)",
		ir.KindIterClusterProject: "iterclusterproject (simplify-lower must eliminate it)",
		ir.KindAs:                 "as (eliminated with its Map)",
		ir.KindRuntimeGridQuery:   "runtimegridquery (final-lower must rewrite it to an attribute read)",
		ir.KindRuntimeThreadQuery: "runtimethreadquery (final-lower must rewrite it to an attribute read)",
		ir.KindInductionVariable:  "inductionvariable (tile-lower must RAUW it onto dimvars)",
	}
	for _, n := range e.g.AllocatedNodes() {
		if n == nil || n.BasePtr().Erased() {
			continue
		}
		if why, bad := banned[n.KindOf()]; bad {
			return fmt.Errorf("cxx: live node %d is %s — %s", n.BasePtr().ID(), n.KindOf(), why)
		}
	}
	if len(e.k.Gmem) > 0 {
		return fmt.Errorf("cxx: %d GMEM temporaries (KindMemNew) — v0 emission supports none (no fixture, no launcher contract)", len(e.k.Gmem))
	}
	if e.opts.MaxCTA <= 0 || e.opts.MaxThreadsPerCTA <= 0 {
		return fmt.Errorf("cxx: launch totals must be positive (got %d CTAs / %d threads)", e.opts.MaxCTA, e.opts.MaxThreadsPerCTA)
	}
	return nil
}

// prescan derives the facts the walk needs: which tensors are stored (the
// signature's const-qualification), the SMEM pool layout (D4/D5/D8: one
// pool, disjoint offsets in allocation order, zero reuse), FRAG tiles, and
// the fill-source loads that must not emit as standalone items.
func (e *emitter) prescan() error {
	for _, n := range e.g.AllocatedNodes() {
		if n == nil || n.BasePtr().Erased() {
			continue
		}
		switch n.KindOf() {
		case ir.KindMemStore:
			if ops := ir.OperandsOf(n); len(ops) > 1 && ops[1] != nil {
				e.stored[ops[1]] = true
			}
		case ir.KindMemNew:
			// fix3 D-h: one allocation kind, three storage classes — the
			// class decides the pool side (SMEM: dynamic-pool offsets;
			// FRAG: register tiles). GMEM temporaries are rejected at the
			// gates before the prescan runs.
			sc, _ := ir.StorageClassOf(n)
			switch sc {
			case ir.StorageSMem:
				ext, err := e.byteExtent(n)
				if err != nil {
					return err
				}
				off := e.smemCap
				if off%16 != 0 { // 16-byte alignment per buffer view
					off += 16 - off%16
				}
				e.smemOff[n] = off
				e.smemCap = off + ext
				e.buffers = append(e.buffers, n)
			case ir.StorageFrag:
				e.frags = append(e.frags, n)
			}
		}
	}
	// Signature names are the local-name collision guard.
	for _, p := range e.k.Params {
		e.paramNames[ir.InputParamNameOf(p)] = true
	}
	// Fill-source loads: the value operand of every fill store. They exist
	// only to carry the loader's source coordinates; emitting them as items
	// would perform a dead global load per round. Also record each buffer's
	// source tensor for the semantic <tensor>_tile view names.
	var walk func(blk *ir.ASTBlock) error
	walk = func(blk *ir.ASTBlock) error {
		for _, it := range blk.Items() {
			if it.Node != nil {
				if isFillStore(it.Node) {
					ops := ir.OperandsOf(it.Node)
					if len(ops) > 0 && ops[0] != nil {
						e.fillSrc[ops[0]] = true
						if src := ir.OperandsOf(ops[0]); len(src) > 0 && src[0] != nil {
							if _, ok := e.bufTensor[ops[1]]; !ok {
								e.bufTensor[ops[1]] = ir.InputParamNameOf(src[0])
							}
						}
					}
				}
				continue
			}
			if it.Block != nil {
				if err := walk(it.Block); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if err := walk(e.k.Body); err != nil {
		return err
	}
	e.scanReducerFacts()
	e.buildPureRefs()

	// The lambda-owned closure: the scheduler anchors the surviving COMBINER
	// lambdas' parameters and bodies as items (typically in the owning LNT's
	// prologue), but their emission is the file-scope function — the
	// kernel-body walk skips the whole closure. Only the reducer
	// constructions' lambda operands are combiners: since fix2 D-b the
	// producing Maps' lambdas also survive final-lower (the lower no longer
	// sweeps them — they are dead-in-the-sea items the pipeline tail keeps),
	// and their closures are EMITTED statements (the piped value chain), so
	// marking them would skip live statements.
	for _, n := range e.g.AllocatedNodes() {
		if n == nil || n.BasePtr().Erased() {
			continue
		}
		switch n.KindOf() {
		case ir.KindRuntimeSequentialReducer, ir.KindRuntimeWarpReducer, ir.KindRuntimeGlobalReducer:
			ops := ir.OperandsOf(n)
			if len(ops) >= 2 && ops[1] != nil && ops[1].KindOf() == ir.KindLambda {
				e.markClosure(ops[1])
			}
		}
	}
	return nil
}

// scanReducerFacts pre-collects the R1/R3 reducer emission facts (the
// constructions emit BEFORE the data-flow that uses them in walk order, so
// the reverse lookups happen up front):
//
//   - STAGING: every warp/global construction's staging MemNewSMem operand
//     (R1). Staging rows participate in the pool offset accounting (the
//     prescan's disjoint D8 layout) but get NO kernelTop view declaration —
//     their only textual form is the INLINE SmemView construction inside
//     the reducer ctor.
//   - GATED: every GMEM MemStore whose value-operand closure consumes a
//     Combine call (R3). Such stores are per-element epilogue stores inside
//     a reducer's element loop — the runtime-owned ownership gate
//     (is_first_in_cta, R2) wraps them so exactly one CTA thread writes
//     each output element.
func (e *emitter) scanReducerFacts() {
	var combineCalls []ir.Node
	for _, n := range e.g.AllocatedNodes() {
		if n == nil || n.BasePtr().Erased() {
			continue
		}
		switch n.KindOf() {
		case ir.KindRuntimeWarpReducer, ir.KindRuntimeGlobalReducer:
			if ops := ir.OperandsOf(n); len(ops) == 4 && ops[3] != nil {
				e.staging[ops[3]] = true
			}
		case ir.KindRuntimeCallMethod:
			if ir.RuntimeMethodOf(n) == "Combine" {
				combineCalls = append(combineCalls, n)
			}
		}
	}
	if len(combineCalls) == 0 {
		return
	}
	for _, n := range e.g.AllocatedNodes() {
		if n == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindMemStore {
			continue
		}
		cl := e.closureOf(ir.OperandsOf(n)[0])
		for _, c := range combineCalls {
			if cl[c] {
				e.gated[n] = true
				break
			}
		}
	}
}

// closureOf collects n's transitive operand closure (self included, live
// nodes only) — the prescan's expression-reach helper.
func (e *emitter) closureOf(n ir.Node) map[ir.Node]bool {
	set := make(map[ir.Node]bool)
	var visit func(x ir.Node)
	visit = func(x ir.Node) {
		if x == nil || x.BasePtr() == nil || x.BasePtr().Erased() || set[x] {
			return
		}
		set[x] = true
		for _, d := range ir.OperandsOf(x) {
			visit(d)
		}
	}
	visit(n)
	return set
}

// buildPureRefs computes which scheduler-anchored PURE items are actually
// referenced by emitted statements or loop headers (B1, quick-261023-cgq):
// the operand closure of every non-pure statement item plus every loop
// block's header operands. An unreferenced pure item (e.g. a const whose
// only use became an inline iterator literal) emits nothing — dead locals
// are warnings and reviewer noise, never code.
func (e *emitter) buildPureRefs() {
	var add func(n ir.Node)
	add = func(n ir.Node) {
		if n == nil || e.pureRef[n] {
			return
		}
		e.pureRef[n] = true
		switch n.KindOf() {
		case ir.KindDimVar, ir.KindKeyedAttribute:
			// Header/annotation carriers: their children (iv init/step,
			// soc/dosoc/loc/tile/loader values) emit as inline literals,
			// never as locals — descending would false-reference them.
			return
		case ir.KindMemNew:
			// Allocation declarations (F0, 2026-10-23; every storage class
			// since fix3 D-h): their byte-extent
			// consts render as inline literals in the hoisted preamble
			// declarations (the pool views, the Frag tiles, the reducer
			// staging slices) — never as locals. Descending would
			// false-reference the extent const into a dead local (the same
			// rationale as the DimVar/KeyedAttribute stop).
			return
		}
		for _, op := range ir.OperandsOf(n) {
			add(op)
		}
	}
	var walk func(blk *ir.ASTBlock)
	walk = func(blk *ir.ASTBlock) {
		for _, it := range blk.Items() {
			if it.Node != nil {
				if !isPureItem(it.Node) {
					add(it.Node)
				}
				continue
			}
			if it.Block != nil {
				// ONLY the bound's MAX expression: the bdimvar's iv operand
				// (and the loop node's annotation consts) emit as inline
				// literals, so walking them would false-reference their
				// consts (B1). Body statements are covered by the recursion.
				if bops := ir.OperandsOf(it.Block.Bound()); len(bops) == 2 {
					add(bops[1])
				}
				walk(it.Block)
			}
		}
	}
	walk(e.k.Body)
}

// isPureItem reports whether the item node is a pure value (hoistable,
// inline-able) rather than an effectful statement or construction.
func isPureItem(n ir.Node) bool {
	switch n.KindOf() {
	case ir.KindConst, ir.KindParam, ir.KindBinOp, ir.KindUnaryOp, ir.KindSelect,
		ir.KindCast, ir.KindCall, ir.KindTensorDimOf, ir.KindRuntimeGetAttribute, ir.KindDimVar:
		return true
	}
	return false
}

// markClosure adds n and its transitive operand closure to lambdaOwned.
func (e *emitter) markClosure(n ir.Node) {
	if n == nil || e.lambdaOwned[n] {
		return
	}
	e.lambdaOwned[n] = true
	for _, op := range ir.OperandsOf(n) {
		e.markClosure(op)
	}
}

// byteExtent returns a MemNewSMem/FragNew's extent operand as a byte count
// (the tile-lower pass allocates u32-const byte extents).
func (e *emitter) byteExtent(n ir.Node) (int, error) {
	ops := ir.OperandsOf(n)
	if len(ops) != 1 || ops[0] == nil {
		return 0, fmt.Errorf("cxx: allocation %d carries no extent operand", n.BasePtr().ID())
	}
	v, ok := ir.ConstValueOf(ops[0])
	if !ok {
		return 0, fmt.Errorf("cxx: allocation %d extent is not a constant (v0 requires compile-time sizes)", n.BasePtr().ID())
	}
	if v < 0 || v != float64(int(v)) {
		return 0, fmt.Errorf("cxx: allocation %d extent %v is not a whole byte count", n.BasePtr().ID(), v)
	}
	return int(v), nil
}

// isFillStore reports whether n is a FILL — a tiered (SMEM/FRAG-class)
// MemStore whose value operand is a tile-shaped GMEM MemLoad (fix3 D-h:
// the retired v0-era loader annotation left the IR; selection is DERIVED
// at codegen from the storage class per R-H/R-J — SMEM window fills take
// the cooperative LoadTileCta, FRAG fills the per-thread LoadTileThread,
// the same tier rule the attribute recorded).
func isFillStore(n ir.Node) bool {
	if n == nil || n.KindOf() != ir.KindMemStore {
		return false
	}
	if sc, ok := ir.StorageClassOf(n); !ok || sc == ir.StorageGMEM {
		return false
	}
	ops := ir.OperandsOf(n)
	if len(ops) < 3 || ops[0] == nil || ops[0].KindOf() != ir.KindMemLoad {
		return false
	}
	sc, ok := ir.StorageClassOf(ops[0])
	return ok && sc == ir.StorageGMEM
}

// line writes one indented line into the current emission target.
func (e *emitter) line(s string) {
	for i := 0; i < e.deep; i++ {
		e.out.WriteString("  ")
	}
	e.out.WriteString(s)
	e.out.WriteString("\n")
}

// deriveSymbols pins the emitted function symbols (quick-261023-cgq): the
// KindKernelMeta payloads when the frontend set them (validated as C++
// identifiers, fail-closed), else the Google-style PascalCase fallback over
// the graph name. Launcher defaults to <Kernel>Launch.
func (e *emitter) deriveSymbols() error {
	base := e.name
	if e.opts.KernelName != "" {
		base = e.opts.KernelName
	}
	kernel, launcher := pascalIdent(base), ""
	if m := e.g.Meta(); m != nil && m.KindOf() == ir.KindKernelMeta {
		mk, ml, _ := ir.KernelNamesOf(m)
		if mk != "" {
			kernel = mk
		}
		if ml != "" {
			launcher = ml
		}
	}
	if err := checkIdent("kernel", kernel); err != nil {
		return err
	}
	if launcher == "" {
		launcher = kernel + "Launch"
	}
	if err := checkIdent("launcher", launcher); err != nil {
		return err
	}
	e.kernelSym, e.launcherSym = kernel, launcher
	return nil
}

// c emits one instrumentation comment line at the current indentation when
// the level allows (quick-261023-cgq; section comments — the goldens' level).
func (e *emitter) c(format string, args ...any) {
	if e.comments < cg.CommentSections {
		return
	}
	e.line(fmt.Sprintf("// "+format, args...))
}

// provenance annotates a materialized local with its minting pass at the
// verbose level.
func (e *emitter) provenance(n ir.Node) string {
	if e.comments < cg.CommentVerbose {
		return ""
	}
	return "  // [" + n.BasePtr().Loc().PassOrigin + "]"
}

// local mints a collision-safe local name: never a kernel parameter name
// and never a previously minted local (deterministic fallbacks).
func (e *emitter) local(base string) string {
	for _, cand := range []string{base, "tca_" + base, base + "_x"} {
		if !e.paramNames[cand] && !e.usedNames[cand] {
			e.usedNames[cand] = true
			return cand
		}
	}
	for i := 2; ; i++ {
		cand := fmt.Sprintf("%s_%d", base, i)
		if !e.paramNames[cand] && !e.usedNames[cand] {
			e.usedNames[cand] = true
			return cand
		}
	}
}
