package pass

// Context is the pass-local context handed to every Pass.Run. It carries
// the running pass's ID so passes can stamp diagnostics and mutations with
// their own name (D-17).
type Context struct {
	Pass string
}
