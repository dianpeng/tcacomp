package opt

import (
	"fmt"

	"tcacomp/internal/ir"
)

// Pass is one transformation or analysis-shaped pipeline step. The five
// methods are the whole contract the manager codes against (RESEARCH
// Pattern 3, MLIR-narrowed to a flat single-BB pipeline):
//
//   - ID: stable name used in pipeline syntax, preconditions, and dumps.
//   - Requires: pass IDs that must appear EARLIER in the pipeline (PASS-01).
//   - Produces: the pipeline stage whose legality window is checked after
//     this pass runs (IR-07).
//   - Preserved: analysis names kept valid; empty = invalidate all (PASS-03).
//   - Run: the transformation itself; a non-nil return aborts the run.
type Pass interface {
	ID() string
	Requires() []string
	Produces() Stage
	Preserved() []string
	Run(*Context, *ir.Graph) error
}

// Manager runs an ordered flat pipeline over one graph (PASS-01). It owns
// the analysis cache and dispatches instrumentation callbacks around every
// pass and at pipeline end (PASS-02). Pass failure aborts the whole run
// (MLIR signalPassFailure semantics).
type Manager struct {
	passes          []Pass
	instrumentation []Instrumentation
	cache           *Cache
}

// NewManager returns a Manager that will run passes in the given order.
func NewManager(passes []Pass) *Manager {
	return &Manager{
		passes: passes,
		cache:  NewCache(),
	}
}

// AddInstrumentation registers a PM callback. BeforePass fires in
// registration order, AfterPass in REVERSE registration order (stack-like,
// MLIR-verified).
func (pm *Manager) AddInstrumentation(in Instrumentation) {
	if in == nil {
		return
	}
	pm.instrumentation = append(pm.instrumentation, in)
}

// Cache returns the manager's analysis cache, so callers can pre-seed
// analyses and tests can observe preservation across runs (PASS-03).
func (pm *Manager) Cache() *Cache { return pm.cache }

// Run executes the pipeline over g in order. Per pass: preconditions are
// checked, BeforePass fires in registration order, the pass runs, AfterPass
// fires in reverse order, a pass error aborts with the pass named, the
// produced stage's legality window checks the whole graph, and the analysis
// cache invalidates everything not on the preservation whitelist.
//
// After the LAST pass — including when the pass list is EMPTY — AfterPipeline
// fires on every registered instrumentation implementing PipelineEnd, in
// reverse registration order; the first non-nil error aborts Run with
// "pipeline end: %w". This makes empty-pipeline verification meaningful.
func (pm *Manager) Run(g *ir.Graph) error {
	ran := make(map[string]bool)
	for _, p := range pm.passes {
		if err := pm.checkPreconditions(p, ran); err != nil {
			return fmt.Errorf("pipeline order: %w", err)
		}
		for _, in := range pm.instrumentation {
			in.BeforePass(p, g)
		}
		err := p.Run(&Context{Pass: p.ID(), Cache: pm.cache}, g)
		for i := len(pm.instrumentation) - 1; i >= 0; i-- {
			pm.instrumentation[i].AfterPass(p, g, err)
		}
		if err != nil {
			return fmt.Errorf("pass %s: %w", p.ID(), err)
		}
		ran[p.ID()] = true
		if err := checkLegality(p.Produces(), g); err != nil {
			return err
		}
		pm.cache.Invalidate(p.Preserved())
	}
	for i := len(pm.instrumentation) - 1; i >= 0; i-- {
		if pe, ok := pm.instrumentation[i].(PipelineEnd); ok {
			if err := pe.AfterPipeline(g); err != nil {
				return fmt.Errorf("pipeline end: %w", err)
			}
		}
	}
	return nil
}

// checkPreconditions rejects a pass whose Requires names a pass ID that has
// not appeared earlier in the pipeline (PASS-01). ran holds the IDs of
// passes that already ran this pipeline.
func (pm *Manager) checkPreconditions(p Pass, ran map[string]bool) error {
	for _, req := range p.Requires() {
		if !ran[req] {
			return fmt.Errorf("pass %q requires %q earlier", p.ID(), req)
		}
	}
	return nil
}
