package analysis

// An Analysis is a named, read-only computation over a graph snapshot.
// Read-only rule: analyses NEVER mutate the IR — they receive no MutContext
// and never call internal/ir mutation functions. The package exists so
// transform passes can declare what they preserve (PASS-03). This
// interface is the shared contract every analysis sub-module implements
// and the registry/BuildAll construct against.
type Analysis interface {
	Name() string
}
