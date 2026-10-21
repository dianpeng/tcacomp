package opt

import (
	"fmt"
	"sort"
	"sync"
)

// registry.go is the global pass registry (quick-261018-l36 requirement 9
// point 2). Every optimization sub-module (dce, schedule, ...) self-registers
// its PassSpec from a func init(), so importing the sub-module — directly or
// through internal/opt/all — is the single act that makes the pass available
// to BuildPipeline. The database/sql.Register precedent applies: registration
// happens at program init, so every failure mode is a PANIC with a clear
// message — deterministic, surfaced at startup, never a silently missing
// pass.
//
// The priority scale is the DEFAULT pipeline position vocabulary:
//
//	100   early cleanup (dce — erase before anything observes the graph)
//	101   dce-verify (the post-sweep invariant check)
//	102   simplify-lower (IterCluster format → Iter format)
//	103   simplify-inline (lambda parameter inlining)
//	104   simplify-verify (the post-simplify invariant check)
//	105   type inference (typeinfer — every node typed after Simplification, before gvn)
//	110   global CSE (gvn — dedup pure computations graph-wide)
//	111   gvn-verify (the post-dedup invariant check)
//	112   memory optimization (memopt — store-to-load forwarding + store-store/dead-store elimination, post-GVN)
//	113   memopt-verify (the post-memopt invariant/fixpoint check)
//	114   loop tiling (decide: overlap analysis + PSched search + decision record)
//	115   tiling-verify (the post-decide record consistency check)
//	116   tile-lower (the decide→lower pair's lowering half: annotated tiled
//	      loop nests + map-query intrinsics + SMEM/FRAG cache insertion)
//	...   transformation passes slot between as they land
//	900   scheduling (schedule — sea-of-nodes → AST-style CFG + LICM; the
//	      final_lower transition point)
//
// Smaller priority = earlier. Equal priorities keep registration order
// (stable sort), which is also the order init()s run in when only all.go
// imports the sub-modules.

// PassSpec is one registry entry: everything BuildPipeline needs to
// construct the default pipeline without importing every pass package.
type PassSpec struct {
	// ID is the pass's pipeline identity; it must equal Factory().ID().
	ID string
	// Priority is the default pipeline position; smaller = earlier.
	Priority int
	// Requires is the declarative mirror of Pass.Requires. The Manager
	// remains authoritative at run time: it re-checks the ACTUAL Pass's
	// Requires against the pipeline, regardless of construction order.
	Requires []string
	// Factory constructs a fresh pass instance. It must be cheap and pure —
	// Register constructs one throwaway instance to cross-check the ID, and
	// BuildPipeline invokes it once per pipeline build.
	Factory func() Pass
}

// registry is the process-global spec table. Registration happens in func
// init()s, but tests may register later, so every access takes the mutex.
var registry = struct {
	sync.Mutex
	specs []PassSpec
}{}

// Register adds spec to the global registry. It panics — with the offending
// ID in the message where one exists — on:
//
//   - an empty ID,
//   - a nil Factory,
//   - an ID that is already registered (duplicate registration is almost
//     always a copy-paste of another pass's init, and failing at init is the
//     cheapest possible diagnosis),
//   - a Factory whose instance's ID() does not equal spec.ID (the registry
//     would hand out passes under the wrong name).
func Register(spec PassSpec) {
	registry.Lock()
	defer registry.Unlock()
	if spec.ID == "" {
		panic("opt: Register called with an empty pass ID")
	}
	if spec.Factory == nil {
		panic(fmt.Sprintf("opt: Register(%q) called with a nil Factory", spec.ID))
	}
	for _, s := range registry.specs {
		if s.ID == spec.ID {
			panic(fmt.Sprintf("opt: Register called twice for pass %q", spec.ID))
		}
	}
	if got := spec.Factory().ID(); got != spec.ID {
		panic(fmt.Sprintf("opt: Register(%q): Factory().ID() = %q, must equal the spec ID", spec.ID, got))
	}
	registry.specs = append(registry.specs, spec)
}

// Lookup returns the registered spec for id, and whether one exists.
func Lookup(id string) (PassSpec, bool) {
	registry.Lock()
	defer registry.Unlock()
	for _, s := range registry.specs {
		if s.ID == id {
			return s, true
		}
	}
	return PassSpec{}, false
}

// Registered returns every registered spec, priority ASCENDING, stable within
// a priority (registration order). The returned slice is a fresh copy, so
// callers may reorder it freely.
func Registered() []PassSpec {
	registry.Lock()
	out := append([]PassSpec(nil), registry.specs...)
	registry.Unlock()
	sort.SliceStable(out, func(i, j int) bool { return out[i].Priority < out[j].Priority })
	return out
}
