package cxx

import (
	"fmt"

	"tcacomp/internal/ir"
)

// socOf maps a Loop block's class to the soc selector (0=seq, 1=cta, 2=thr
// — the materialization primitive, runtime_objects.md §5).
func socOf(blk *ir.ASTBlock) (int, error) {
	switch blk.Kind() {
	case ir.BlockSeqLoop:
		return 0, nil
	case ir.BlockCTALoop:
		return 1, nil
	case ir.BlockThrLoop:
		return 2, nil
	default:
		return -1, fmt.Errorf("cxx: loop block carries non-loop block kind %s", blk.Kind())
	}
}

// walkLoop materializes one Loop block as a runtime iterator for-statement
// (doc/cg.md §7): the soc annotation selects the iterator class; the
// iterator's Has() is the CTA-UNIFORM round walk (all threads of the CTA
// agree on the trip count — collectives stay uniformly reached, tca.md
// §5.12 "predicated, not exited"); per-thread/per-CTA validity is enforced
// inside the shim's views and the loaders, never by an emitted branch. The
// compiler emits no control flow of its own anywhere.
func (e *emitter) walkLoop(blk *ir.ASTBlock) error {
	soc, err := socOf(blk)
	if err != nil {
		return err
	}
	loop := blk.Loop()
	iv := blk.IV()
	bound := blk.Bound()
	if loop == nil || iv == nil || bound == nil {
		return fmt.Errorf("cxx: loop block is missing its header nodes")
	}
	if err := e.requireSocAttr(loop, soc); err != nil {
		return err
	}
	bops := ir.OperandsOf(bound)
	if len(bops) != 2 || bops[1] == nil {
		return fmt.Errorf("cxx: loop %d bound is not a (iv, max) bdimvar", loop.BasePtr().ID())
	}
	maxExpr, err := e.emitExpr(bops[1])
	if err != nil {
		return err
	}
	// The dimvar identity (init, step) rides the iterator ctor as literals
	// — header facts, not reused SSA values.
	ivOps := ir.OperandsOf(iv)
	if len(ivOps) != 3 {
		return fmt.Errorf("cxx: loop %d iv is not a (dim, init, step) dimvar", loop.BasePtr().ID())
	}
	initLit, err := constLiteral(ivOps[1])
	if err != nil {
		return err
	}
	stepLit, err2 := constLiteral(ivOps[2])
	if err2 != nil {
		return err2
	}
	it := fmt.Sprintf("tca_it%d", e.itN)
	e.itN++

	var ctor string
	switch soc {
	case 0:
		ctor = fmt.Sprintf("tca::rt::SeqIterator(%s, %s, %s)", initLit, stepLit, maxExpr)
	case 1:
		if e.curGrid == "" {
			return fmt.Errorf("cxx: cta loop %d has no GridMapping in scope (the LNT prologue must construct it first)", loop.BasePtr().ID())
		}
		if blk.DoSoC() < 0 {
			return fmt.Errorf("cxx: cta loop %d carries no dosoc", loop.BasePtr().ID())
		}
		ctor = fmt.Sprintf("tca::rt::CtaIterator(%s, %du, %du, %s, %s, %s)",
			e.curGrid, uint(blk.DoSoC()), uint(blk.LoC()), initLit, stepLit, maxExpr)
	case 2:
		if e.curThread == "" {
			return fmt.Errorf("cxx: thr loop %d has no ThreadMapping in scope (the LNT prologue must construct it first)", loop.BasePtr().ID())
		}
		if blk.DoSoC() < 0 {
			return fmt.Errorf("cxx: thr loop %d carries no dosoc", loop.BasePtr().ID())
		}
		ctor = fmt.Sprintf("tca::rt::ThrIterator(%s, %du, %du, %s, %s, %s)",
			e.curThread, uint(blk.DoSoC()), uint(blk.LoC()), initLit, stepLit, maxExpr)
	}

	e.line(fmt.Sprintf("for (auto %s = %s; %s.Has(); %s.Next()) {", it, ctor, it, it))
	e.deep++
	e.iters[iv] = it
	dim := ivOps[0]
	e.dimScope = append(e.dimScope, map[ir.Node]string{dim: it})
	if err := e.walkItems(blk); err != nil {
		return err
	}
	e.dimScope = e.dimScope[:len(e.dimScope)-1]
	delete(e.iters, iv)
	e.deep--
	e.line("}")
	return nil
}

// requireSocAttr gates on the soc annotation existing on the ForLoop itself
// (tile-lower stamps it at every construction site; its absence is a
// pipeline bug) and agreeing with the block class.
func (e *emitter) requireSocAttr(loop ir.Node, soc int) error {
	for _, op := range ir.OperandsOf(loop)[3:] {
		key, val := ir.KeyedAttributeOf(op)
		if key != "soc" || val == nil {
			continue
		}
		v, ok := ir.ConstValueOf(val)
		if !ok || int(v) != soc {
			return fmt.Errorf("cxx: loop %d soc annotation %v disagrees with its block class %d", loop.BasePtr().ID(), v, soc)
		}
		return nil
	}
	return fmt.Errorf("cxx: loop %d carries no soc annotation", loop.BasePtr().ID())
}
