// Package all is the composition root of the analysis layer
// (quick-261019-jck): it blank-imports every analysis sub-module so their
// func init() self-registrations fire. Importing
// tcacomp/internal/analysis/all is the single act that makes
// analysis.Lookup/Registered/BuildAll see the full analysis set — no
// analysis package needs a direct import anywhere else.
//
// Every new analysis sub-module's blank import accumulates here.
package all

import (
	_ "tcacomp/internal/analysis/sideeffect"
)
