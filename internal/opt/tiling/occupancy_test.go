// occupancy_test.go pins the SM-occupancy factor of the tiling search
// objective (quick-261029-n63, the locked design:
// score' = traffic × (1 − idleFrac) × (ε + (1−ε)·fill)).
//
// THE SINGLE SOURCE of the arithmetic under test is occupancyOf /
// occupancyCompute in constraints.go — these pins exercise that function
// directly and never re-derive the formula. The mm_small numbers are the
// design's headline case: at NominalExtent 4096, a single-CTA-dim schedule
// at LoC=64/tile=256 wastes 75% of its grid (activeGrid 16 of 64), while
// the same schedule at LoC=16 keeps every CTA busy (idleFrac 0) — for
// equal traffic the occupancy factor prefers LoC=16 by ~4×.
package tiling

import (
	"math"
	"testing"

	"tcacomp/internal/ir"
)

// occDefaultParams is the A6000/sm_86 pin set under test.
func occDefaultParams() Params { return DefaultParams() }

// TestOccupancyMMSmallLoC64 pins the design's worked number: LoC=64,
// tile=256 at NominalExtent=4096, blockDim=64, post-eviction smem 64 KiB.
//	 grid        = 64
//	 activeGrid  = min(64, ceil(4096/256)=16) = 16
//	 idleFrac    = 1 − 16/64 = 0.75
//	 ctasPerSM   = min(48/2, 1536/64, 101376/66560) = min(24, 24, 1) = 1
//	 capacity    = 84 × 1 = 84
//	 fill        = 16/84 ≈ 0.19048
//	 factor      = 0.25 × (0.15 + 0.85×0.19048) ≈ 0.25 × 0.31190
func TestOccupancyMMSmallLoC64(t *testing.T) {
	p := occDefaultParams()
	cta := []IterDecision{{SoC: SoCTA, LoC: 64, TileSize: 256}}
	o := occupancyCompute(cta, 64, 64*1024, p)
	if o.grid != 64 {
		t.Fatalf("grid = %d, want 64", o.grid)
	}
	if o.activeGrid != 16 {
		t.Fatalf("activeGrid = %d, want 16 (min(64, ceil(4096/256)))", o.activeGrid)
	}
	if o.blockDim != 64 {
		t.Fatalf("blockDim = %d, want 64", o.blockDim)
	}
	if o.ctasPerSM != 1 {
		t.Fatalf("ctasPerSM = %d, want 1 (the 64 KB SMEM window binds: 101376/66560)", o.ctasPerSM)
	}
	if o.capacity != 84 {
		t.Fatalf("capacity = %d, want 84 (SMs × ctasPerSM)", o.capacity)
	}
	if math.Abs(o.idleFrac-0.75) > 1e-12 {
		t.Fatalf("idleFrac = %v, want 0.75", o.idleFrac)
	}
	wantFill := 16.0 / 84.0
	if math.Abs(o.fill-wantFill) > 1e-12 {
		t.Fatalf("fill = %v, want %v", o.fill, wantFill)
	}
	wantFactor := 0.25 * (0.15 + 0.85*wantFill)
	if math.Abs(o.factor-wantFactor) > 1e-12 {
		t.Fatalf("factor = %v, want %v", o.factor, wantFactor)
	}
}

// TestOccupancyMMSmallLoC16 pins the flip side: the SAME traffic at
// LoC=16 keeps the whole grid busy — idleFrac 0, same fill (the 64 KB
// window still binds at one CTA/SM), factor ≈ 0.31190 — beating the
// LoC=64 schedule's 0.25×0.31190 by exactly the idleFrac ratio (4×).
func TestOccupancyMMSmallLoC16(t *testing.T) {
	p := occDefaultParams()
	cta := []IterDecision{{SoC: SoCTA, LoC: 16, TileSize: 256}}
	o := occupancyCompute(cta, 64, 64*1024, p)
	if o.grid != 16 || o.activeGrid != 16 {
		t.Fatalf("grid/activeGrid = %d/%d, want 16/16", o.grid, o.activeGrid)
	}
	if o.idleFrac != 0 {
		t.Fatalf("idleFrac = %v, want 0 (activeGrid == grid)", o.idleFrac)
	}
	if o.ctasPerSM != 1 || o.capacity != 84 {
		t.Fatalf("ctasPerSM/capacity = %d/%d, want 1/84", o.ctasPerSM, o.capacity)
	}
	wantFactor := 1.0 * (0.15 + 0.85*(16.0/84.0))
	if math.Abs(o.factor-wantFactor) > 1e-12 {
		t.Fatalf("factor = %v, want %v", o.factor, wantFactor)
	}

	// The ~4× re-choice: equal traffic, LoC=16's factor is 4× LoC=64's.
	lo := occupancyCompute([]IterDecision{{SoC: SoCTA, LoC: 64, TileSize: 256}}, 64, 64*1024, p)
	ratio := o.factor / lo.factor
	if math.Abs(ratio-4.0) > 1e-9 {
		t.Fatalf("LoC=16-vs-64 factor ratio = %v, want exactly 4 (idleFrac 0 vs 0.75, equal fill)", ratio)
	}
}

// TestOccupancyCapacityZero pins the total clamp: a post-eviction smem
// footprint that cannot fit one CTA's reservation on the SM zeroes
// ctasPerSM, hence capacity, hence fill — the factor degrades to
// (1−idleFrac)×ε and NO division by zero can panic the compiler
// (threat T-n63-02).
func TestOccupancyCapacityZero(t *testing.T) {
	p := occDefaultParams()
	cta := []IterDecision{{SoC: SoCTA, LoC: 16, TileSize: 256}}
	// smemPerCTA = SmemPerSM − ReservedSmemPerCTA + 1: one byte over the
	// numerator — the smem term clamps to 0.
	o := occupancyCompute(cta, 64, p.SmemPerSM-p.ReservedSmemPerCTA, p)
	if o.ctasPerSM != 0 {
		t.Fatalf("ctasPerSM = %d, want 0 (smemPerCTA exceeds the SMEM numerator)", o.ctasPerSM)
	}
	if o.capacity != 0 {
		t.Fatalf("capacity = %d, want 0", o.capacity)
	}
	if o.fill != 0 {
		t.Fatalf("fill = %d, want 0 (capacity 0 clamps)", o.fill)
	}
	wantFactor := 1.0 * 0.15 // (1−idleFrac)=1, ε + (1−ε)·0 = ε
	if math.Abs(o.factor-wantFactor) > 1e-12 {
		t.Fatalf("factor = %v, want %v (the ε-only floor)", o.factor, wantFactor)
	}
}

// TestOccupancyMultiDim pins the per-dim product shape: each CTA dim
// contributes its LoC to the grid and min(LoC, ceil(Nominal/T)) to the
// active grid — the uniform-extent model (no per-dim extents exist at
// compile time, so NominalExtent applies to every CTA dim).
func TestOccupancyMultiDim(t *testing.T) {
	p := occDefaultParams()
	// 2 CTA dims, LoC 16 × 8 = 128 CTAs; per-dim active caps: both tiles
	// 256 → 16 each → activeGrid 256... no: min(16,16)=16, min(8,16)=8 →
	// active = 128. Idle-free only if every dim's LoC <= its cap.
	cta := []IterDecision{
		{SoC: SoCTA, LoC: 16, TileSize: 256},
		{SoC: SoCTA, LoC: 8, TileSize: 256},
	}
	o := occupancyCompute(cta, 128, 8*1024, p)
	if o.grid != 128 {
		t.Fatalf("grid = %d, want 128 (16×8)", o.grid)
	}
	if o.activeGrid != 128 {
		t.Fatalf("activeGrid = %d, want 128 (both LoCs under the 16 cap)", o.activeGrid)
	}
	if o.idleFrac != 0 {
		t.Fatalf("idleFrac = %v, want 0", o.idleFrac)
	}
	// blockDim 128 = 4 warps: ctasPerSM = min(48/4, 1536/128, 101376/9216)
	// = min(12, 12, 11) = 11 — the smem term binds first here.
	if o.ctasPerSM != 11 {
		t.Fatalf("ctasPerSM = %d, want 11 (smem term binds: 101376/9216 = 11)", o.ctasPerSM)
	}
	if o.capacity != 84*11 {
		t.Fatalf("capacity = %d, want %d", o.capacity, 84*11)
	}
	// fill clamps at 1 once activeGrid >= capacity.
	if o.fill != 1 {
		t.Fatalf("fill = %v, want 1 (activeGrid %d >= capacity %d)", o.fill, o.activeGrid, o.capacity)
	}
}

// TestOccupancyOfSharedSource pins that occupancyOf — the map-facing
// wrapper the search calls — extracts the CTA dims and delegates to the
// SAME arithmetic (no duplicated formula anywhere): for the mm_small
// LoC=64 single-CTA-dim schedule over a real decision map the product
// must equal occupancyCompute on the extracted slice, and blockDim must
// come from blockDimOf (Π warp K × 32), not a re-derivation.
func TestOccupancyOfSharedSource(t *testing.T) {
	p := occDefaultParams()
	_, _, _, iters := fix33Graph()
	itersMap := map[ir.Node]IterDecision{
		iters[0]: {SoC: SoCTA, LoC: 64, TileSize: 256},          // M
		iters[1]: {SoC: SoCSequential, LoC: 1, TileSize: 256},   // N
		iters[2]: {SoC: SoCWARP, LoC: 2, TileSize: 32},          // K → blockDim 64
	}
	viaWrapper := occupancyOf(itersMap, 64*1024, p)
	viaCore := occupancyCompute([]IterDecision{{SoC: SoCTA, LoC: 64, TileSize: 256}}, 64, 64*1024, p)
	if viaWrapper != viaCore {
		t.Fatalf("occupancyOf = %+v, want exactly occupancyCompute's %+v (the wrapper must not re-derive)",
			viaWrapper, viaCore)
	}
	if viaWrapper.blockDim != 64 {
		t.Fatalf("blockDim = %d, want 64 from blockDimOf (Π warp K × 32)", viaWrapper.blockDim)
	}
}

// TestOccupancyParamsDefaults pins the six A6000/sm_86 occupancy fields
// on DefaultParams — same pinning discipline as MaxCache (threat
// T-n63-01's default half).
func TestOccupancyParamsDefaults(t *testing.T) {
	p := DefaultParams()
	if p.SMs != 84 {
		t.Fatalf("SMs default = %d, want 84 (A6000 GA102)", p.SMs)
	}
	if p.WarpsPerSM != 48 {
		t.Fatalf("WarpsPerSM default = %d, want 48 (A6000 GA102)", p.WarpsPerSM)
	}
	if p.ThreadsPerSM != 1536 {
		t.Fatalf("ThreadsPerSM default = %d, want 1536 (48 × 32)", p.ThreadsPerSM)
	}
	if p.SmemPerSM != 102400 {
		t.Fatalf("SmemPerSM default = %d, want 102400 (the sm_86 per-SM byte count)", p.SmemPerSM)
	}
	if p.ReservedSmemPerCTA != 1024 {
		t.Fatalf("ReservedSmemPerCTA default = %d, want 1024 (the sm_86 per-block reservation)", p.ReservedSmemPerCTA)
	}
	if p.OccEps != 0.15 {
		t.Fatalf("OccEps default = %v, want 0.15 (the locked design's ε)", p.OccEps)
	}
}

// TestOccupancyParamsValidate pins the Validate gate on the six new
// fields (threat T-n63-01's boundary half): every occupancy count must
// be positive and OccEps must lie in (0, 1].
func TestOccupancyParamsValidate(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*Params)
	}{
		{"sms zero", func(p *Params) { p.SMs = 0 }},
		{"sms negative", func(p *Params) { p.SMs = -1 }},
		{"warps per sm zero", func(p *Params) { p.WarpsPerSM = 0 }},
		{"threads per sm zero", func(p *Params) { p.ThreadsPerSM = 0 }},
		{"smem per sm zero", func(p *Params) { p.SmemPerSM = 0 }},
		{"reserved smem zero", func(p *Params) { p.ReservedSmemPerCTA = 0 }},
		{"occ eps zero", func(p *Params) { p.OccEps = 0 }},
		{"occ eps negative", func(p *Params) { p.OccEps = -0.1 }},
		{"occ eps over one", func(p *Params) { p.OccEps = 1.5 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := DefaultParams()
			tc.mut(&p)
			if err := p.Validate(); err == nil {
				t.Fatalf("%s must fail Validate", tc.name)
			}
		})
	}
	// The boundary value OccEps == 1 is legal (the (0,1] interval's closed
	// end): the factor degenerates to (1−idleFrac) × 1 — pure fill-blind
	// selection, a valid explicit choice.
	p := DefaultParams()
	p.OccEps = 1
	if err := p.Validate(); err != nil {
		t.Fatalf("OccEps == 1 (the closed interval end) must validate: %v", err)
	}
}
