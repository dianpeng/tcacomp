// Package opt implements the optimization-pipeline layer: the pass manager,
// legality windows, instrumentation, the shared analysis cache, and the
// global pass registry with its abstract factory
// (quick-261018-l36 requirement 9).
//
// Layout:
//
//   - This top level (internal/opt) holds the machinery SHARED by all
//     passes: the Pass contract and Manager (manager.go), the run Context
//     with the shared analysis Cache (context.go, cache.go), instrumentation
//     (instrument.go), legality windows (legality.go), and the registry +
//     default-pipeline factory (registry.go, factory.go).
//   - Each optimization lives in its own sub-module folder
//     (internal/opt/dce, internal/opt/schedule, ...) and self-registers via
//     a func init() calling opt.Register with its PassSpec — priority (asc,
//     registration-stable) is the default pipeline position.
//   - internal/opt/all blank-imports every pass sub-module; importing it is
//     what fires the registrations, and opt.BuildPipeline then returns the
//     default priority-ordered pipeline.
//   - Pass.Requires is enforced by Manager.Run at run time regardless of
//     construction order; shared analysis results flow through
//     Context.Cache with Preserved()-driven invalidation.
//
// The package imports only internal/ir and internal/analysis; pass
// sub-modules import internal/opt (never each other), so the import graph
// stays acyclic by construction.
package opt
