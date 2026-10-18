// Package all is the composition root of the pass layer
// (quick-261018-l36 requirement 9 point 3): it blank-imports every pass
// sub-module so their func init() self-registrations fire. Importing
// tcacomp/internal/opt/all is the single act that makes BuildPipeline see
// the full default pipeline — no pass package needs a direct import anywhere
// else.
//
// Every new pass sub-module's blank import accumulates here.
package all

import (
	_ "tcacomp/internal/opt/schedule"
)
