// params_env.go is the fix8 §3 slice-0 probe hook: an env-gated Params
// overlay that lets the pinned-decision ceiling probe pin the decision
// menu the search enumerates (CtaLoCPresets / TileSizes / MaxCTA /
// NominalExtent / WarpLoCPresets) WITHOUT any production default-path
// change — with every TCACOMP_TILING_* variable unset the result is
// field-identical to DefaultParams(), proven end-to-end by the sha256
// zero-diff gate against the 1c suite manifest.
//
// Scope (fix8 §3, measurement only): the hook overlays ONLY the five
// menu-shaped fields the probe's 3-knob pin recipe needs. Decisions stay
// SEARCH-CHOSEN — the hook never overrides a record; it changes what the
// search is allowed to express, which is exactly the "what does the
// current model deliver at each parallelism point" measurement. The
// three tiling-family registry factories (decide/verify/lower) call
// mustParamsFromEnv, so search and verification read the SAME env in
// lockstep by construction — no new override semantics, no drift window.
//
// Boundary discipline (threat T-u8a-01): values are parsed with strict
// strconv integer syntax, no empty list segments, positive only; a
// malformed value is a NAMED error identifying the variable and the
// offending value — never a silent fallback to defaults. The pass's own
// Params.Validate() keeps sole ownership of the semantic floors (entries
// >= 2, MaxCTA >= 2): a well-formed but below-floor value parses and then
// fails Validate exactly as a hand-built Params would. mustParamsFromEnv
// panics with the named error — the opt.Pass Factory signature returns no
// error, so the panic IS the fail-fast choke (mirroring the fail-loud
// policy of every registry factory).
package tiling

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// paramsEnvPrefix namespaces every variable this hook reads.
const paramsEnvPrefix = "TCACOMP_TILING_"

// parseStrictPositiveInt parses one overlay scalar: strict base-10
// integer syntax, no leading '+', value >= 1. Zero and negatives are
// malformed at the boundary (never legal integer settings for these
// fields); the semantic floors above 1 remain Validate's.
func parseStrictPositiveInt(varName, raw string) (int, error) {
	if strings.HasPrefix(raw, "+") {
		return 0, fmt.Errorf("tiling: %s: %q is not a strict integer (no leading '+')", varName, raw)
	}
	n, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("tiling: %s: %q is not an integer: %v", varName, raw, err)
	}
	if n < 1 {
		return 0, fmt.Errorf("tiling: %s: %q must be a positive integer", varName, raw)
	}
	if int64(int(n)) != n {
		return 0, fmt.Errorf("tiling: %s: %q overflows int", varName, raw)
	}
	return int(n), nil
}

// parseStrictPositiveList parses one comma list: non-empty, every segment
// a strict positive integer, no empty segments (an omitted entry is never
// silently skipped).
func parseStrictPositiveList(varName, raw string) ([]int, error) {
	segments := strings.Split(raw, ",")
	out := make([]int, 0, len(segments))
	for _, seg := range segments {
		// A strict segment carries no surrounding whitespace and is never
		// empty ("16,,4" and "16," are malformed, not silently dropped).
		if seg == "" || seg != strings.TrimSpace(seg) {
			return nil, fmt.Errorf("tiling: %s: %q has an empty or padded list segment", varName, raw)
		}
		v, err := parseStrictPositiveInt(varName, seg)
		if err != nil {
			return nil, fmt.Errorf("tiling: %s: %q has a malformed entry: %v", varName, raw, err)
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("tiling: %s: %q is an empty list", varName, raw)
	}
	return out, nil
}

// paramsFromEnv starts from DefaultParams() and overlays the fix8 §3
// slice-0 pin variables. With every variable unset the result is
// field-identical to DefaultParams(). A variable that is SET but
// malformed fails with a named error (set-but-empty is malformed — the
// probe never wants an ambiguous empty pin). The caller-owned
// Params.Validate() is NOT run here: the three passes run it at Run
// entry, and that gate stays the single semantic floor.
func paramsFromEnv() (Params, error) {
	p := DefaultParams()

	if raw, ok := os.LookupEnv(paramsEnvPrefix + "CTA_LOC"); ok {
		v, err := parseStrictPositiveList("TCACOMP_TILING_CTA_LOC", raw)
		if err != nil {
			return Params{}, err
		}
		p.CtaLoCPresets = v
	}
	if raw, ok := os.LookupEnv(paramsEnvPrefix + "TILES"); ok {
		v, err := parseStrictPositiveList("TCACOMP_TILING_TILES", raw)
		if err != nil {
			return Params{}, err
		}
		p.TileSizes = v
	}
	if raw, ok := os.LookupEnv(paramsEnvPrefix + "MAX_CTA"); ok {
		v, err := parseStrictPositiveInt("TCACOMP_TILING_MAX_CTA", raw)
		if err != nil {
			return Params{}, err
		}
		p.MaxCTA = v
	}
	if raw, ok := os.LookupEnv(paramsEnvPrefix + "NOMINAL_EXTENT"); ok {
		v, err := parseStrictPositiveInt("TCACOMP_TILING_NOMINAL_EXTENT", raw)
		if err != nil {
			return Params{}, err
		}
		p.NominalExtent = v
	}
	if raw, ok := os.LookupEnv(paramsEnvPrefix + "WARP_LOC"); ok {
		v, err := parseStrictPositiveList("TCACOMP_TILING_WARP_LOC", raw)
		if err != nil {
			return Params{}, err
		}
		p.WarpLoCPresets = v
	}
	return p, nil
}

// mustParamsFromEnv is the registry-factory form of paramsFromEnv: the
// opt.Pass Factory signature returns no error, so the named boundary
// error panics instead — fail-loud at pass construction, before any IR
// is touched (a malformed probe pin must never silently compile with
// default Params).
func mustParamsFromEnv() Params {
	p, err := paramsFromEnv()
	if err != nil {
		panic(err)
	}
	return p
}
