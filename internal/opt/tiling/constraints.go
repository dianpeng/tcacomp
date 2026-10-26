package tiling

// constraints.go is the SINGLE SOURCE of the fix4 §2.3 constraint
// arithmetic — MAPM (expanded-instance dim caps), MALoC (factorization
// product bounds), and CuMAP (the CTA-dim presence) — shared by the
// feasible-area search (incremental, over PARTIAL vectors: a digit is
// rejected the moment it violates a running bound) and tiling-verify
// (completed-vector re-derivation over the published record). Neither
// side duplicates a formula: the search prunes with the same pred that
// verify re-checks with.
//
// The counters are monotone in the digits placed so far (counts only
// grow, products only multiply by entries >= 2), so checking a partial
// prefix is sound: a prefix violation can never be repaired by later
// digits.

// partState is the running factorization state of one (partial or
// complete) schedule vector: the expanded-instance counts and the
// factorization products (fix4 §2.3, checked over expanded instances and
// factorizations).
type partState struct {
	ctaDims int // #CTA-assigned dims placed (1 CTA instance each)
	thrDims int // #expanded THR instances placed (WARP dims + CTA dims)
	ctaLoC  int // running Π CTA LoC (>= 1; the empty product is 1)
	warpK   int // running Π warp K (>= 1; the empty product is 1)
}

// initialPartState is the empty-vector state (all counters zero, both
// products the empty product 1).
func initialPartState() partState { return partState{ctaLoC: 1, warpK: 1} }

// place returns the state after placing one digit with the given SoC and
// absolute LoC (LoC is ignored for SEQ dims — they carry the implicit 1).
func (s partState) place(soc SoCType, loc int) partState {
	next := s
	switch soc {
	case SoCTA:
		next.ctaDims++
		next.thrDims++
		next.ctaLoC *= loc
	case SoCWARP:
		next.thrDims++
		next.warpK *= loc
	}
	return next
}

// mapmOK reports whether PLACING this digit keeps the partial vector
// inside the MAPM caps (fix4 §2.3 constraint 1: #CTA dims <=
// MaxPartitionCta; #THR instances — WARP dims + CTA dims — <=
// MaxPartitionThr). Checked at placement time: a violating digit rejects
// the whole subtree below it, never a completed vector.
func (s partState) mapmOK(soc SoCType, p Params) bool {
	switch soc {
	case SoCTA:
		return s.ctaDims+1 <= p.MaxPartitionCta && s.thrDims+1 <= p.MaxPartitionThr
	case SoCWARP:
		return s.thrDims+1 <= p.MaxPartitionThr
	default:
		return true // SEQ places no expanded instance
	}
}

// malocOK reports whether PLACING this digit keeps the partial vector
// inside the MALoC bounds (fix4 §2.3 constraint 2: per LNT, Π CTA LoC <=
// MaxCTA and Π warp K <= MaxWarpsPerCTA).
func (s partState) malocOK(soc SoCType, loc int, p Params) bool {
	switch soc {
	case SoCTA:
		return s.ctaLoC*loc <= p.MaxCTA
	case SoCWARP:
		return s.warpK*loc <= p.MaxWarpsPerCTA
	default:
		return true
	}
}

// cumapOK reports the CuMAP constraint (fix4 §2.3 constraint 3): at least
// one CTA-assigned dim. A COMPLETED vector without a CTA dim is skipped
// by the search; an LNT where no CTA-carrying vector exists at all is
// structural infeasibility — a hard error (fix4 §2.6).
func (s partState) cumapOK() bool { return s.ctaDims >= 1 }

// partitionCount is the fix4 §2.5 tiebreak metric: expanded instances per
// the §1.3 table — a CTA dim contributes 1 CTA instance + 1 THR instance,
// a WARP dim 1 THR instance, SEQ none (R-PARTITION: 2×#CTA + #WARP, <= 6
// under MAPM).
func (s partState) partitionCount() int { return 2*s.ctaDims + (s.thrDims - s.ctaDims) }

// locPresetOK reports whether an absolute LoC belongs to its SoC's preset
// list (fix4 §2.7's first re-check): CTA -> CtaLoCPresets, WARP ->
// WarpLoCPresets, SEQ -> the implicit 1.
func locPresetOK(soc SoCType, loc int, p Params) bool {
	switch soc {
	case SoCTA:
		return intIn(loc, p.CtaLoCPresets)
	case SoCWARP:
		return intIn(loc, p.WarpLoCPresets)
	default:
		return loc == 1
	}
}

// intIn reports membership in an int list.
func intIn(v int, xs []int) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}
