package opt

// factory.go is the abstract-factory half of the registry (quick-261018-l36
// requirement 9 point 2): BuildPipeline turns the registry's priority-ordered
// specs into concrete Pass instances, giving callers the DEFAULT pipeline
// without importing a single pass sub-module — importing internal/opt/all is
// enough.

// BuildPipeline returns a fresh instance of every registered pass, in
// Registered() order: priority ascending, registration order among equals.
//
// Priority ordering is the DEFAULT pipeline order only. Pass.Requires is
// still enforced by Manager.Run at run time for correctness regardless of
// construction order: a pipeline assembled by hand, filtered, or reordered
// gets the same precondition checks as this default one.
func BuildPipeline() []Pass {
	specs := Registered()
	passes := make([]Pass, 0, len(specs))
	for _, spec := range specs {
		passes = append(passes, spec.Factory())
	}
	return passes
}
