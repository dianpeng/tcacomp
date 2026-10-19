package analysis

import "tcacomp/internal/ir"

// factory.go is the abstract-factory half of the registry
// (quick-261019-jck, mirroring internal/opt/factory.go): BuildAll turns
// the registry's specs into constructed analyses over one graph snapshot,
// giving callers every registered analysis without importing a single
// analysis sub-module — importing internal/analysis/all is enough.

// BuildAll constructs every registered analysis over g, keyed by Name().
// A nil g is legal: nil passes through to each factory, whose contract is
// all-conservative defaults on a nil graph. Keying by Name() matches
// opt.Cache.Put's bookkeeping, so a manager can seed its cache from
// BuildAll in one loop.
func BuildAll(g *ir.Graph) map[string]Analysis {
	specs := Registered()
	out := make(map[string]Analysis, len(specs))
	for _, spec := range specs {
		out[spec.Name] = spec.Factory(g)
	}
	return out
}
