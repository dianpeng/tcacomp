package pass

import (
	"tcacomp/internal/analysis"
)

// Cache holds cached analysis results keyed by analysis name (PASS-03,
// MLIR's invalidate-all-by-default preservation contract). A transform pass
// invalidates everything it does not explicitly preserve: a pass with an
// empty Preserved() whitelist clears the whole cache.
type Cache struct {
	analyses map[string]analysis.Analysis
}

// NewCache returns an empty analysis cache.
func NewCache() *Cache {
	return &Cache{analyses: make(map[string]analysis.Analysis)}
}

// Get returns the cached analysis with the given name, or nil.
func (c *Cache) Get(name string) analysis.Analysis {
	return c.analyses[name]
}

// Put stores a under its Name().
func (c *Cache) Put(a analysis.Analysis) {
	if a == nil {
		return
	}
	if c.analyses == nil {
		c.analyses = make(map[string]analysis.Analysis)
	}
	c.analyses[a.Name()] = a
}

// Invalidate clears EVERY cached analysis, then re-keeps only the names on
// the preservation whitelist (invalidate-all default, PASS-03).
func (c *Cache) Invalidate(preserved []string) {
	kept := make(map[string]analysis.Analysis)
	for _, name := range preserved {
		if a, ok := c.analyses[name]; ok {
			kept[name] = a
		}
	}
	c.analyses = kept
}
