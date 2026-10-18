package opt

// Context is the pass-local context handed to every Pass.Run. It carries
// the running pass's ID so passes can stamp diagnostics and mutations with
// their own name (D-17), and the manager's analysis cache so passes SHARE
// analysis results (quick-261018-l36 requirement 9 point 6): a pass reads a
// previously-computed analysis via Cache.Get, publishes its own via Cache.Put,
// and preservation/invalidation stays governed by the pass's Preserved()
// whitelist plus Cache.Invalidate — the manager wires pm.cache in before
// every Run, so all passes of one pipeline observe the same cache.
type Context struct {
	Pass  string
	Cache *Cache
}
