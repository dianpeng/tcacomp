# Deferred items — quick-261024-bz9

Out-of-scope latent defects discovered during execution (recorded, not
fixed — each needs an author-level ruling):

1. **SEQ T==1 single form cannot serve an epilogue-positioned consumer**
   (blocks the plan's literal rank-3 T==1 shape B and its documented
   {CTA,1}/{SEQ,1}/{SEQ,1} fallback). The SEQ-single rewrite's index
   expression IS the raw loop dimvar (`d.indexExpr = d.singleIV`), so any
   consumer scheduled outside the nest — the reduce-tail store — uses the
   dimvar from outside the loop's operand/payload closure and the
   loop-scope verifier rejects the graph AT TILE-LOWER time
   ("dimvar used outside the loop's body operand subtree"). The 1-between
   golden shape passes only because the CTA form's index expression is an
   Add shared with the in-nest load. Candidate fixes (author decision):
   an AST-aware loop-scope notion for scheduled graphs, or wrapping the
   SEQ-single index expression. Workaround used: shape B is the rank-2
   T==1 endpoint ({CTA,100,1}/{WARP,8,1}) which exercises all three
   stated shape-B purposes (consumer move + same-scope store, WARP-single
   mirror gate numerics, WarpReducer at T==1).

2. **T>1 WARP non-reduced dim in a moved consumer — the adjacent unruled
   coverage-hole shape.** The ruled thread-partitioned-consumer refusal
   covers the single IV of a WARP-SoC dim. The T>1 WARP analog (a moved
   store whose coordinate keeps a thread-varying `rank*T + betweenIV`
   term of a WARP dim while the reduced dim is SEQUENTIAL) leaves the same
   coverage hole under the is_first_in_cta gate but is NOT covered by the
   ruling's single-IV formulation. Unreachable when the reduced dim is
   WARP (the one-WARP-dim policy); flagged for a ruling extension.

3. **The `betweenBlocks` empty-path nil bug (fixed here, but the sibling
   audit is worth an author glance):** any other caller relying on
   "nil means not found" over possibly-empty results should be audited —
   the `append(empty-slice)` nil-return trap is silent in Go.
