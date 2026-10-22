package finallower

// consts.go is final-lower's LITERAL CANONICALIZATION step: GVN (priority
// 110) dedups pure nodes long before the passes that mint most const atoms —
// tile-lower's rebuilt loop bounds, schedule-era expressions, final-lower's
// own constructions (mapping portions, KeyedAttribute values, re-iteration
// init/step) — and nothing re-dedups them afterward. Without this step the
// codegen-ready IR carries duplicate literal nodes for the same (type,
// value) pair (a tiled matmul showed eleven distinct "64" consts).
//
// The step walks the allocated nodes in allocation order and keeps the
// FIRST const of each (type, value) pair as the canonical def; every later
// duplicate is RAUW'd onto it through the MutContext choke point and
// erased. Consts are immutable leaves, so sharing is unconditionally sound
// (no def-site mutation exists to poison other users), and a const's
// SchedBlock placement is scheduling metadata — a literal is loop-invariant
// everywhere, so rewiring uses across blocks cannot invalidate placement.
// Type identity is the NumericType interning relation (types.go:
// "interned, immutable, and identity-compared"), so a pointer-keyed map is
// exact, never speculative. Untyped consts (no typeinfer product) are left
// alone. Runs BEFORE the erased-item sweep so the duplicates' AST items are
// swept with everything else, and before the final ir.Verify gate.
import (
	"fmt"
	"math"

	"tcacomp/internal/ir"
)

// canonicalizeConsts dedups literal constants: at most one live const node
// per (type, value) pair survives (the lowest-ID def of the pair).
func (lr *lowering) canonicalizeConsts() error {
	type constKey struct {
		ty   *ir.NumericType
		bits uint64
	}
	canon := make(map[constKey]ir.Node)
	for _, n := range lr.g.AllocatedNodes() {
		if n == nil || n.BasePtr() == nil || n.BasePtr().Erased() || n.KindOf() != ir.KindConst {
			continue
		}
		v, ok := ir.ConstValueOf(n)
		if !ok {
			continue
		}
		// The declared payload type (ConstTypeOf — set at construction,
		// interned) is the canonical key; the typeinfer-computed cell
		// (TypeOf) is frequently nil for consts minted after typeinfer
		// (tile-lower/schedule/final-lower construct untyped-computed
		// atoms), which is exactly the population this step exists for.
		ty := ir.ConstTypeOf(n)
		if ty == nil {
			ty = ir.TypeOf(n)
		}
		nt, ok := ty.(*ir.NumericType)
		if !ok || nt == nil {
			continue // nothing safe to key on
		}
		k := constKey{nt, math.Float64bits(v)}
		c, ok := canon[k]
		if !ok {
			canon[k] = n
			continue
		}
		if err := lr.mctx.ReplaceAllUsesWith(n, c); err != nil {
			return fmt.Errorf("final-lower: canonicalizing const #%d onto #%d: %w",
				n.BasePtr().ID(), c.BasePtr().ID(), err)
		}
		if err := lr.mctx.Erase(n); err != nil {
			return fmt.Errorf("final-lower: erasing duplicate const #%d: %w", n.BasePtr().ID(), err)
		}
	}
	return nil
}
