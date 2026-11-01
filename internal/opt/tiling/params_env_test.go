// params_env_test.go pins the env-gated Params overlay (fix8 §3 slice 0,
// the pinned-decision ceiling probe): paramsFromEnv starts from
// DefaultParams() and overlays the five TCACOMP_TILING_* variables the
// probe's 3-knob pin recipe needs. The contract under test:
//
//   - IDENTITY: with every var unset the result is field-identical to
//     DefaultParams() — the hook is provably zero-default-path (the sha256
//     zero-diff gate against the 1c manifest is the e2e form of this pin).
//   - OVERRIDES: each var set in isolation applies exactly its parsed
//     value; every other field stays DefaultParams().
//   - FAIL-FAST: a malformed value (non-integer, empty, zero, negative,
//     empty list segment) is a NAMED error identifying the variable and
//     the offending value; mustParamsFromEnv panics with that same error
//     (the registry Factory signature returns opt.Pass only — the panic
//     IS the fail-fast choke). The pass's own Params.Validate() keeps
//     ownership of the semantic floors (entries >= 2): a below-floor but
//     well-formed value PARSES and then fails Validate.
package tiling

import (
	"os"
	"reflect"
	"strings"
	"testing"
)

// paramsEnvOverlayVars is the complete overlay surface — nothing else in
// Params is env-settable (the probe pins decisions, not policy).
var paramsEnvOverlayVars = []string{
	"TCACOMP_TILING_CTA_LOC",
	"TCACOMP_TILING_TILES",
	"TCACOMP_TILING_MAX_CTA",
	"TCACOMP_TILING_NOMINAL_EXTENT",
	"TCACOMP_TILING_WARP_LOC",
}

// unsetAllTilingEnv defensively clears the whole overlay surface — the
// test process may inherit a developer shell carrying probe pins (the
// identity pin must measure the hook's default path, not the shell).
func unsetAllTilingEnv(t *testing.T) {
	t.Helper()
	for _, v := range paramsEnvOverlayVars {
		if err := os.Unsetenv(v); err != nil {
			t.Fatalf("unset %s: %v", v, err)
		}
	}
}

// TestParamsFromEnvIdentityDefault pins the zero-default-path contract:
// all overlay vars unset => field-identical to DefaultParams(), no error.
// The must-variant agrees (it panics only on error).
func TestParamsFromEnvIdentityDefault(t *testing.T) {
	unsetAllTilingEnv(t)

	got, err := paramsFromEnv()
	if err != nil {
		t.Fatalf("paramsFromEnv() with every var unset errored: %v", err)
	}
	if want := DefaultParams(); !reflect.DeepEqual(got, want) {
		t.Fatalf("paramsFromEnv() with every var unset =\n%+v\nwant DefaultParams()\n%+v", got, want)
	}
	if got := mustParamsFromEnv(); !reflect.DeepEqual(got, DefaultParams()) {
		t.Fatalf("mustParamsFromEnv() with every var unset = %+v, want DefaultParams()", got)
	}
}

// TestParamsFromEnvOverrides pins each var in isolation: the named field
// is replaced by the parsed value and EVERY other field stays
// DefaultParams() (the overlay never invents unrelated changes).
func TestParamsFromEnvOverrides(t *testing.T) {
	cases := []struct {
		name  string
		vars  map[string]string
		mut   func(p *Params)
		field string
	}{
		{
			name:  "CTA_LOC comma list",
			vars:  map[string]string{"TCACOMP_TILING_CTA_LOC": "16,4"},
			mut:   func(p *Params) { p.CtaLoCPresets = []int{16, 4} },
			field: "CtaLoCPresets",
		},
		{
			name:  "CTA_LOC single entry",
			vars:  map[string]string{"TCACOMP_TILING_CTA_LOC": "512"},
			mut:   func(p *Params) { p.CtaLoCPresets = []int{512} },
			field: "CtaLoCPresets",
		},
		{
			name:  "TILES",
			vars:  map[string]string{"TCACOMP_TILING_TILES": "16"},
			mut:   func(p *Params) { p.TileSizes = []int{16} },
			field: "TileSizes",
		},
		{
			name:  "MAX_CTA",
			vars:  map[string]string{"TCACOMP_TILING_MAX_CTA": "512"},
			mut:   func(p *Params) { p.MaxCTA = 512 },
			field: "MaxCTA",
		},
		{
			name:  "NOMINAL_EXTENT",
			vars:  map[string]string{"TCACOMP_TILING_NOMINAL_EXTENT": "131072"},
			mut:   func(p *Params) { p.NominalExtent = 131072 },
			field: "NominalExtent",
		},
		{
			name:  "WARP_LOC",
			vars:  map[string]string{"TCACOMP_TILING_WARP_LOC": "2"},
			mut:   func(p *Params) { p.WarpLoCPresets = []int{2} },
			field: "WarpLoCPresets",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			unsetAllTilingEnv(t)
			for k, v := range tc.vars {
				t.Setenv(k, v)
			}

			got, err := paramsFromEnv()
			if err != nil {
				t.Fatalf("paramsFromEnv() errored: %v", err)
			}
			want := DefaultParams()
			tc.mut(&want)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("paramsFromEnv() = %+v, want DefaultParams() with %s overridden: %+v", got, tc.field, want)
			}
		})
	}
}

// TestParamsFromEnvMalformed pins fail-fast-at-the-boundary: every
// malformed form is a NAMED error carrying the variable name and the
// offending value, and mustParamsFromEnv panics with that same error —
// never a silent fallback to defaults.
func TestParamsFromEnvMalformed(t *testing.T) {
	bad := []string{"16,x", "0", "-4", "", "abc"}
	for _, vars := range paramsEnvOverlayVars {
		for _, val := range bad {
			t.Run(vars+"="+val, func(t *testing.T) {
				unsetAllTilingEnv(t)
				t.Setenv(vars, val)

				_, err := paramsFromEnv()
				if err == nil {
					t.Fatalf("paramsFromEnv() accepted %s=%q", vars, val)
				}
				msg := err.Error()
				if !strings.Contains(msg, vars) {
					t.Fatalf("error %q does not name the variable %s", msg, vars)
				}
				if !strings.Contains(msg, val) {
					t.Fatalf("error %q does not carry the offending value %q", msg, val)
				}

				panicked := true
				func() {
					defer func() {
						r := recover()
						if r == nil {
							panicked = false
							return
						}
						got, ok := r.(error)
						if !ok {
							t.Fatalf("mustParamsFromEnv panicked with %T (%v), want the paramsFromEnv error", r, r)
						}
						if got.Error() != msg {
							t.Fatalf("mustParamsFromEnv panic = %q, want the paramsFromEnv error %q", got.Error(), msg)
						}
					}()
					_ = mustParamsFromEnv()
				}()
				if !panicked {
					t.Fatalf("mustParamsFromEnv() did not panic on %s=%q", vars, val)
				}
			})
		}
	}
}

// TestParamsFromEnvEmptyListSegment pins the no-empty-segments rule of
// the comma-list grammar: "16,,4" and "16," are malformed (an unset entry
// is never silently skipped).
func TestParamsFromEnvEmptyListSegment(t *testing.T) {
	for _, val := range []string{"16,,4", "16,", ",16"} {
		unsetAllTilingEnv(t)
		t.Setenv("TCACOMP_TILING_CTA_LOC", val)

		_, err := paramsFromEnv()
		if err == nil {
			t.Fatalf("paramsFromEnv() accepted TCACOMP_TILING_CTA_LOC=%q (empty segment)", val)
		}
		if !strings.Contains(err.Error(), "TCACOMP_TILING_CTA_LOC") {
			t.Fatalf("error %q does not name the variable", err.Error())
		}
	}
}

// TestParamsFromEnvBelowValidateFloor pins the gate separation: a
// well-formed but below-floor value (TILES="1", CTA_LOC="1") PARSES
// cleanly — and the resulting Params fails the EXISTING Params.Validate()
// floor (entries >= 2). Validate stays the single owner of the semantic
// floors; the env hook adds no second gate and no bypass.
func TestParamsFromEnvBelowValidateFloor(t *testing.T) {
	cases := []struct {
		name string
		vars map[string]string
	}{
		{name: "TILES=1", vars: map[string]string{"TCACOMP_TILING_TILES": "1"}},
		{name: "CTA_LOC=1", vars: map[string]string{"TCACOMP_TILING_CTA_LOC": "1"}},
		{name: "MAX_CTA=1", vars: map[string]string{"TCACOMP_TILING_MAX_CTA": "1"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			unsetAllTilingEnv(t)
			for k, v := range tc.vars {
				t.Setenv(k, v)
			}

			p, err := paramsFromEnv()
			if err != nil {
				t.Fatalf("paramsFromEnv() rejected a well-formed value: %v", err)
			}
			if verr := p.Validate(); verr == nil {
				t.Fatalf("Params from %v passed Validate(); want the existing >=2 floor to fail it", tc.vars)
			}
		})
	}
}
