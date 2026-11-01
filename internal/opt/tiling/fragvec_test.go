package tiling

// ---------------------------------------------------------------------------
// fix7 §5 slice 1c RED pin (quick-261101-oqg, design/fix/fix7.md §5):
// the chunk-policy VECTOR-WIDTH BIAS.
// ---------------------------------------------------------------------------
//
// The loader's 128-bit fast path (runtime/tca/loader.h copy_row) gates per
// row on trailing % 4 == 0 (float) AND 16-byte endpoint alignment. Slice 1b
// pinned the six amortized kernels' chunk trailing extents at 4/8/16/64
// (all 16B-clean) — but nothing in the policy GUARANTEES a budget-forced
// chunk stays there: the 1a halving loop snaps dims to RAW halves
// (24 -> 12 -> 6), so a budget squeeze can land a fill destination's
// trailing extent on a non-vector-width multiple (6) and the runtime gate
// silently fails on EVERY row (correct via the scalar fallback, slow by
// construction). Slice 1c biases the halving loop: when a reducible SEQ
// dim's TRUE static extent is a multiple of vecElems (16/elemBytes = 4 for
// f32), each halving step snaps DOWN to the largest multiple of vecElems
// that is <= the raw half AND keeps fragChunkBytes <= the priced budget
// (siteWindowBytes — the ONE shared source, untouched); when no multiple
// of vecElems satisfies the budget, the dim keeps the 1a raw-halving
// floor-1 behavior EXACTLY (no silent clamp, the budget gate never
// violated, the fail-closed error unchanged).
//
// This pin crafts the record DIRECTLY against planSiteChunk (the 1a chunk
// policy's ONE shape-choice site — lower_fragchunk.go) with a covered set
// [C1b-WARP outer dim (per-lane chunk 1, absorbing no budget), SEQ trailing
// dim with a TRUE static extent], the priced budget forcing the trailing
// chunk past a vector-width boundary. The fixture iter chain mirrors
// internal/cg/cxx/fragchunk_test.go's idiom (DimVar + BoundedDimVar over a
// const max + Iter), and the BIV max const operand IS the staticIterExtent
// source (the NEVER-ir.TypeOf rule).
//
// Pin arithmetic (recorded per the 1b discipline, in-subtest):
//
//	covered [W(T=32) -> lane chunk ceil(32/32)=1, S(true ext E)]
//	raw window bytes  = 4 B * 1 * E;  budget = min(raw, MaxFragPerThread)

import (
	"testing"

	"tcacomp/internal/ir"
)

// vecBiasRun builds a lowerRun + SiteFacts for one covered [WARP outer,
// SEQ trailing(static extent seqExtent)] site and plans its chunk. The
// WARP dim needs no forcing INS iter here: planSiteChunk is the chunk
// ARITHMETIC owner — the class decision (fragClassPlan) is upstream and
// covered by the 1a pins; this fixture drives the halving loop only.
func vecBiasRun(t *testing.T, seqExtent, maxFrag int) (*lowerRun, ir.Node, ir.Node) {
	t.Helper()
	g := ir.NewGraph("fragvec")
	u32 := ir.InternScalar(ir.TU32)
	dw := g.NewInputParam("ctor", ir.RoleDim, "w", nil)
	ds := g.NewInputParam("ctor", ir.RoleDim, "s", nil)
	x := g.NewInputParam("ctor", ir.RoleTensor, "x",
		ir.InternTensor(ir.TF32, []ir.DimRef{{Name: "w"}, {Name: "s"}}))

	mkIter := func(dim ir.Node, ext float64, parent ir.Node) ir.Node {
		dv := g.NewDimVar("ctor", dim, g.NewConst("ctor", u32, 0), g.NewConst("ctor", u32, 1))
		biv := g.NewBoundedDimVar("ctor", dv, g.NewConst("ctor", u32, ext))
		return g.NewIter("ctor", biv, parent)
	}
	wIter := mkIter(dw, 32, nil)                      // WARP-covered outer, T=32
	sIter := mkIter(ds, float64(seqExtent), wIter)    // SEQ trailing, static extent
	ivW := g.NewInductionVariable("ctor", wIter)
	ivS := g.NewInductionVariable("ctor", sIter)
	env := ir.NewEnv()
	load := g.NewMemLoadChained("ctor", env, x, []ir.Node{ivW, ivS})

	lr := &lowerRun{
		g:      g,
		params: DefaultParams(),
		decisions: map[ir.Node]IterDecision{
			wIter: {SoC: SoCWARP, LoC: 1, TileSize: 32},
			sIter: {SoC: SoCSequential, LoC: 1, TileSize: seqExtent},
		},
		fragChunk: map[ir.Node]int{},
	}
	lr.params.MaxFragPerThread = maxFrag
	sf := SiteFacts{
		Load:      load,
		Tensor:    x,
		Covered:   []ir.Node{wIter, sIter},
		INS:       []ir.Node{wIter, sIter},
		AxesIters: [][]ir.Node{{wIter}, {sIter}},
	}
	if err := lr.planSiteChunk(load.BasePtr().ID(), sf); err != nil {
		t.Fatalf("planSiteChunk over the vec-bias fixture: %v", err)
	}
	return lr, wIter, sIter
}

// TestFragChunkVecBias pins the §5 snap-to-vector-width rule:
//
//  1. "snap to vector width": true extent 24 (24 % 4 == 0),
//     MaxFragPerThread = 24 -> budget = min(4*1*24, 24) = 24 B.
//     1a raw halving: [1,24] = 96 B > 24 -> 12 (48 B > 24) -> 6 (24 B) —
//     trailing 6 lands OFF the vector width and the vec16 row gate dead-
//     fires. The bias: step to raw half 12, then snap DOWN to the largest
//     multiple of 4 keeping the budget: 12 (48 B > 24), 8 (32 B > 24),
//     4 (16 B <= 24) — expected chunk trailing 4.
//  2. "extent not vector-compatible keeps raw halving": true extent 30
//     (30 % 4 != 0), MaxFragPerThread = 24 -> budget 24 B: [1,30] = 120
//     -> 15 (60) -> 7 (28) -> 3 (12 <= 24). Expected 3 — the 1a behavior
//     byte-preserved where the snap's precondition (extent % vecElems ==
//     0) does not hold; never a silent clamp.
//  3. "largest multiple within budget, never below": true extent 24,
//     MaxFragPerThread = 48 -> budget = 48 B: [1,24] = 96 > 48 -> raw half
//     12, snap: 12 (48 B <= 48) — expected 12. The bias must not crush the
//     chunk past what the budget allows (the snap stops at the largest
//     qualifying multiple, not the smallest).
//
// RED gate (pre-1c): the policy halves to the raw half — subtest 1 plans
// trailing 6 (want 4) and FAILS; subtests 2 and 3 are the controls that
// must stay green at RED and GREEN alike. The WARP dim never halves in
// all subtests (lane chunk 1, pinned via fragChunk[wIter] == 1).
func TestFragChunkVecBias(t *testing.T) {
	t.Run("snap to vector width", func(t *testing.T) {
		lr, wIter, sIter := vecBiasRun(t, 24, 24)
		if got := lr.fragChunk[wIter]; got != 1 {
			t.Fatalf("the WARP-covered dim's lane chunk is ceil(32/32) = 1 and never halves, got %d", got)
		}
		if got := lr.fragChunk[sIter]; got != 4 {
			t.Fatalf("fix7 §5 1c: a vector-compatible true extent (24 %% 4 == 0) under a 24 B budget must snap the halving step to the largest multiple of 4 within budget (raw half 12 -> 8 -> 4), got trailing chunk %d (the 1a raw policy plans 6 — the vec16 row gate dead-fires on trailing %% 4 != 0)", got)
		}
	})
	t.Run("extent not vector compatible keeps raw halving", func(t *testing.T) {
		lr, wIter, sIter := vecBiasRun(t, 30, 24)
		if got := lr.fragChunk[wIter]; got != 1 {
			t.Fatalf("the WARP-covered dim's lane chunk is ceil(32/32) = 1 and never halves, got %d", got)
		}
		if got := lr.fragChunk[sIter]; got != 3 {
			t.Fatalf("fix7 §5 1c control: a true extent NOT a multiple of vecElems (30 %% 4 != 0) keeps the 1a raw halving exactly (30 -> 15 -> 7 -> 3 under the 24 B budget), got %d", got)
		}
	})
	t.Run("largest multiple within budget", func(t *testing.T) {
		lr, wIter, sIter := vecBiasRun(t, 24, 48)
		if got := lr.fragChunk[wIter]; got != 1 {
			t.Fatalf("the WARP-covered dim's lane chunk is ceil(32/32) = 1 and never halves, got %d", got)
		}
		if got := lr.fragChunk[sIter]; got != 12 {
			t.Fatalf("fix7 §5 1c control: the snap stops at the LARGEST multiple of vecElems within budget (raw half 12 fits the 48 B budget exactly), got %d", got)
		}
	})
}
