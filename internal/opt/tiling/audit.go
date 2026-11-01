package tiling

// audit.go is the fix7 §3.4 PRICED-side reuse audit hook (design/fix/
// fix7.md §3.4): a read-only quote of the priced model for a graph's
// promoted sites, so an e2e audit can compare the judge's priced reuse
// against the emission's actual refill structure WITHOUT reaching into
// package internals. The math rides the SHARED pricing functions
// (siteWindowBytesRaw/siteWindowBytes/siteReuseCount — judge.go owns the
// model and this hook only reads it, never copies it), the SAME
// analyzeOverlap discovery the decide and lower passes run, and the
// record's own decisions. Nothing here feeds back into pricing,
// searching, or lowering — and judge.go/search.go/record.go stay
// untouched (the fix7 §3.3 scope guard).

import (
	"fmt"

	"tcacomp/internal/analysis/loopforest"
	"tcacomp/internal/analysis/sideeffect"
	"tcacomp/internal/ir"
)

// PricedDim is one dim's priced facts in the audit report.
type PricedDim struct {
	Name string  // the dim InputParam's spelling ("e4" for a const dim)
	SoC  SoCType // the record decision's space of concurrency
	Tile int     // the record decision's tile
}

// PricedSite is one promoted site's priced facts: the judge's reuse dims
// and factors, the priced block footprint, and the raw window — the
// numbers the §3.4 audit compares against the emitted refill structure.
type PricedSite struct {
	Tensor      string       // the source tensor's InputParam name
	Class       CacheClass   // the re-derived R2/C1b class
	Covered     []PricedDim  // the covered dims in tensor order
	Tie         string       // the tie dim's spelling ("" for a nil tie)
	ReuseDims   []PricedDim  // the reuse dims ({tie} ∪ inner-than-tie − covered)
	RawBytes    int          // siteWindowBytesRaw — the traffic footprint
	WindowBytes int          // siteWindowBytes — the priced block footprint (min(raw, MaxFragPerThread) for FRAG)
	Reuse       int          // siteReuseCount — the priced reuse factor
}

// PricedSites quotes the priced model for every promoted site of g under
// the given record. The caller runs the decide pass (and its HIR
// prerequisites) first — the record is its product; the class is
// re-derived with the SAME R2/C1b tier rule the lower applies
// (lower_fragchunk.go fragClassPlan), and sites the rule refuses are
// skipped (the lower fails them closed at surgery).
func PricedSites(g *ir.Graph, rec *Record, p Params) ([]PricedSite, error) {
	f := loopforest.Analyze(g)
	se := sideeffect.Analyze(g)
	facts := analyzeOverlap(g, f, se, p).Trees
	decisions := make(map[ir.Node]IterDecision)
	for _, root := range f.Roots() {
		for _, it := range treeItersOf(root) {
			d, ok := rec.IterDecisionOf(it)
			if !ok {
				return nil, fmt.Errorf("tiling audit: iter node %d has no record entry under its BIV", it.BasePtr().ID())
			}
			decisions[it] = d
		}
	}
	dimName := func(it ir.Node) string {
		dim, err := dimOfIter(it)
		if err != nil || dim == nil {
			return fmt.Sprintf("iter%d", it.BasePtr().ID())
		}
		return ir.InputParamNameOf(dim)
	}
	pricedDim := func(it ir.Node) PricedDim {
		d := decisions[it]
		return PricedDim{Name: dimName(it), SoC: d.SoC, Tile: d.TileSize}
	}
	var out []PricedSite
	for _, lf := range facts {
		lntDecisions := make(map[ir.Node]IterDecision, len(lf.Levels))
		for _, it := range lf.Levels {
			lntDecisions[it] = decisions[it]
		}
		ctx := blockDimCtx{blockDim: blockDimOf(lntDecisions)}
		for _, sf := range lf.CS {
			id := sf.Load.BasePtr().ID()
			sd, ok := rec.Sites[id]
			if !ok || !sd.Promoted {
				continue
			}
			class, planned, err := (&lowerRun{decisions: decisions, overrides: map[ir.Node]int{}}).fragClassPlan(sf)
			if err != nil {
				return nil, err
			}
			if !planned {
				continue // the refusal shapes: the lower fails them closed at surgery
			}
			cov, err := siteCoveredTensorOrder(sf, id)
			if err != nil {
				return nil, err
			}
			covered := make([]PricedDim, len(cov))
			for i, it := range cov {
				covered[i] = pricedDim(it)
			}
			reuse := make([]PricedDim, len(sf.ReuseDims))
			for i, it := range sf.ReuseDims {
				reuse[i] = pricedDim(it)
			}
			tie := ""
			if sf.Tie != nil {
				tie = dimName(sf.Tie)
			}
			out = append(out, PricedSite{
				Tensor:      ir.InputParamNameOf(sf.Tensor),
				Class:       class,
				Covered:     covered,
				Tie:         tie,
				ReuseDims:   reuse,
				RawBytes:    siteWindowBytesRaw(sf, lntDecisions, class),
				WindowBytes: siteWindowBytes(sf, lntDecisions, class, p),
				Reuse:       siteReuseCount(sf, lntDecisions, ctx, class),
			})
		}
	}
	return out, nil
}
