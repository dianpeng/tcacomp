package flags

// F holds global compiler constants. Phase 5's cmd/ wires the CLI flags to
// these fields; Phase 1 sets the pinned defaults programmatically.
var F = struct {
	VerifyEach   bool   // verify IR after every pass (TOOL-06; --verify-each=0 disables)
	DumpAfter    string // comma-separated pass IDs to dump after, or "all"; "" = off
	DumpGraphviz bool   // route dumps to .dot emission instead of text
	PassTiming   bool   // accumulate and report per-pass durations
}{
	VerifyEach:   true,
	DumpAfter:    "",
	DumpGraphviz: false,
	PassTiming:   false,
}
