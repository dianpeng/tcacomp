# Review: doc/optana/final_lower.md (Final Lowering design doc)

- Target: `doc/optana/final_lower.md` (265 lines, untracked WIP as of 2026-10-22). The
  doc specifies the last pass before code generation: reducer lowering, map
  elimination, and runtime-object construction (KernelInfo, per-LNT mapping objects)
  over the AST-style CFG produced by the schedule pass.
- Reviewer method: the target was read fully with line numbers; every design element
  is checked against the live system — `internal/ir/` (astcfg.go, node_lambda.go,
  node_intrinsic.go, kind.go, catalog.go, mutate.go), `internal/opt/`
  (simplify/inline.go, simplify/verify.go, typeinfer/rules.go,
  typeinfer/typeinfer.go, tiling/lower_rewrite.go, tiling/lower_cache.go,
  legality.go) — and the doc suite (`doc/tca.md` §2.5/§2.7/§4.2/§4.3/§5.9/§5.10/
  §5.11/§5.12, `doc/cg.md`, `doc/opt.md`, `doc/optana/schedule.md`,
  `doc/runtime_objects.md`). `doc/cg.md` (33 lines) and `doc/opt.md` (33 lines) are
  also untracked working copies; cited by their current line numbers.
- Element verdict tags (exactly one per audited element E1–E10):
  **CONFIRMED** (the design claim holds against the cited system fact),
  **GAP** (the design is underspecified on a point the implementation must know —
  reported as a gap, never silently interpreted), **CONTRADICTION** (the design as
  written conflicts with a cited system fact or with itself), **OUT-OF-SCOPE-OK**
  (forward scaffolding consistent with the system's stated v0 boundaries).
- Finding severities (consolidated table §7): **BLOCKING** (implementing the doc as
  written produces wrong behavior), **CONTRADICTION**, **GAP**, **COMPAT**
  (consistent with the cited system, possibly with named required companions),
  **MINOR** (doc hygiene: typos, internal drift).
- Citation convention: every bare `L<n>` refers to the target doc
  (doc/optana/final_lower.md L1–L265). `§` cites another document by section;
  `path/file.go:L<n>` cites source files. Prior author rulings
  (`loop-tiling-design-rulings`, `schedule-pass-review-rulings` memory files) are
  established decisions and are cited, not re-litigated.
- The doc is NOT modified by this review. No code changes; the only file created is
  this review.

## 1. Verdict summary

**Verdict: compatible-with-named-fixes** (required-fix list in §8; audit basis §3–§4,
runtime-object boundary §5, MLIR assessment §6, consolidated findings §7).

The architecture the doc proposes — a `RuntimeCallMethod` expression node over
runtime-object constructor nodes (`Runtime{Sequential,Warp,Global}Reducer`,
`RuntimeConstruct("KernelInfo")`, `RuntimeGridMapping`/`RuntimeThreadMapping`), Map
elimination, and per-LNT mapping construction — is COMPLEMENTARY to what already
exists: `doc/runtime_objects.md` §11 (L723–L737) already records exactly this surface
as [spec'd] pending final-lower (inventory entries 40–43, L68–L71), the query
intrinsics the tile-lower pass already emits (`internal/ir/node_intrinsic.go:39-54`)
were designed against these objects, and the schedule pass's persistent
SchedSN/SchedBlock state was built for this consumer
(`internal/ir/astcfg.go:35-38`, `:407-428`). But the doc is not implementable as
written: the reducer-lowering algorithm's four insertion points wrap the wrong loop
whenever the reduced dimension is tiled (FL-1, BLOCKING — contradicts `doc/tca.md`
§5.10 L723–L728), the reduced-dimension identification mechanism is unspecified
(FL-2/FL-3), and a dozen specification gaps (node taxonomy, string-operand convention,
lambda-invocation mechanism, portion data source, query-object binding) must be closed
before any backend can treat the output as mechanical translation input.

### Coverage table (user evaluation points → review sections)

| # | User point | Review section |
|---|------------|----------------|
| 1 | Per-element compatibility: lambdaNode convention, Reduce-inline fix, reducer-lower 4-insertion algorithm, Runtime* nodes, Map elimination, KernelInfo, per-LNT mapping objects — each vs doc/tca.md, astcfg.go, doc/opt.md pipeline, doc/cg.md expectations | §3 (E1–E9) |
| 2 | IR purity: does the design leak backend/CUDA specifics into the IR in a way that blocks future platforms (AMD/NPU/TPU) or a lower IR | §3 E10 |
| 3 | Mechanical-translation completeness: everything a CUDA C++ or MLIR backend would still have to invent | §4 |
| 4 | Runtime-object boundary: enumerate the objects the design implies, cross-check BOTH directions vs doc/runtime_objects.md | §5 |
| 5 | MLIR translation complexity: per-construct mapping sketch + overall rating | §6 |

## 2. Method and evidence base

What was read, once, in full: the target (L1–L265); `internal/ir/astcfg.go`,
`internal/ir/node_lambda.go`, `internal/ir/node_intrinsic.go`, `internal/ir/kind.go`,
`internal/ir/catalog.go`; `internal/opt/simplify/inline.go`,
`internal/opt/simplify/verify.go` (invariant headers and the lambda walk),
`internal/opt/typeinfer/typeinfer.go` (untyped whitelist),
`internal/opt/typeinfer/rules.go` (KindLambda/KindParam rules); the intrinsic
emission sites in `internal/opt/tiling/lower_rewrite.go`
(:322, :330, :357, :374, :386) and `internal/opt/tiling/lower_cache.go` (:461);
`doc/tca.md` §2.5/§2.7 (L74–L137), §4.2–4.3 (L275–L299), §5.9–§5.12 (L612–L805);
`doc/cg.md`; `doc/opt.md`; `doc/optana/schedule.md` (L1–L181); `doc/runtime_objects.md`
(inventory L25–L75, §1–§11, appendices). Grep was used for point lookups
(ClearLambdaParams, KindParam typing, unknown-kind legality) rather than re-reads.

Decision rule applied throughout: where the target is silent or ambiguous on a point
the implementation must know, the review reports a GAP with the system fact that makes
the gap load-bearing; it does not invent an interpretation and review the invention.
Where the target conflicts with a cited fact, the finding names the fact and the fix
direction, and the verdict is CONTRADICTION. Prior rulings (D3 one-map-per-LNT
portion model; Issue #1 reducer Init/Combine placement; SchedSN/SchedBlock
complementarity; PrototypeBlock TOTAL SN mapping) are applied as ground truth.

## 3. Per-element compatibility audit (user point 1)

### E1. lambdaNode convention (L13–L24) — CONFIRMED (one sub-claim does not hold; FL-6)

- Doc: the lambda is a stateless function; Input list = [parameter definitions...,
  body] with the body LAST (L16–L19); parameters are DEFINITIONS, not references
  (L18–L20); the lambda's type is its body expression's type (L22–L23);
  "Make sure the lambdaNode follows the above requirements" (L24).
- Ground: the layout already holds exactly. `internal/ir/node_lambda.go:47-64` —
  lambdaNode ops = [params..., body], "the parameter atoms lead in binding order,
  the body expression closes the list"; the params are InputParam atoms, definitions
  by construction (`internal/ir/node_lambda.go:14-19`: "the parameter atoms
  (RoleLambda InputParams) in binding order"); `NewLambdaNode` (L57–L64) appends
  body last. The catalog row agrees (`internal/ir/catalog.go:54`: "lambda
  expression: (param declarations..., body)"). So requirements (1) and (2) are
  already true — the "Fix or Ensure" is a no-op for them.
- The type claim (3) does NOT hold and is not the convention:
  `internal/opt/typeinfer/typeinfer.go:65-75` lists `KindLambda` in the CLOSED
  untyped whitelist ("Lambda is a function expression for which no function Type
  exists yet"), and `internal/opt/typeinfer/rules.go:272-273` returns nil for it.
  Making "type = body type" true would be a deliberate typeinfer change (remove the
  whitelist entry, add a body-typing rule) with unassessed consequences (a function
  expression typed as its result conflates the callable with the call result);
  the doc motivates nothing. Reported as FL-6 (GAP): drop the claim or specify the
  typeinfer amendment and its motivation.

### E2. Pipeline fix "[1] Remove inline for Reduce node's Lambda" (L26–L32) — CONFIRMED (coherent; ripple surface enumerated)

- Doc: keep the Reduce's body as a separate lambda (L29–L31, "its body is
  represented as SSAUse -> <lambdaNode>"); Map's body inline is still kept (L32).
- Ground: the target of the fix is `internal/opt/simplify/inline.go:29-47` — the
  pass "inlines every map/reduce lambda parameter" with the D9 binding for Reduce
  (L33–L35): `[piped, attr(init)]` binds `[acc, now]` as **acc↔piped,
  now↔init-value**. That binding is data-flow WIRING, not combiner semantics: after
  inlining, the reduce body is a pure expression over the piped value and the init
  constant (`internal/opt/simplify/inline.go:186-214` RAUWs every paramRef) — an
  expression that is NOT the combiner `lambda(acc, now)` a specialized reducer must
  apply per element. Keeping Map inline while exempting Reduce is therefore not just
  coherent but semantically REQUIRED: the combiner must survive as a function of two
  formals (acc, now) for `RuntimeCallMethod(%reducer, "Init", <Lambda>)` (L151) to
  hand the runtime something callable; a Map needs no specialization (L91–L92, and
  `doc/tca.md` §5.10's per-element `map(...)` stays inlined at L738). The
  trailing-lambda carrier the doc assumes already exists
  (`internal/ir/node_lambda.go:111-137`, LambdaOf).
- Ripple surface (each verified, each a required companion of the fix):
  1. **simplify-verify** — `internal/opt/simplify/verify.go:34-41` enforces "FULLY
     INLINED MAP/REDUCE BODIES: no lambda body contains a paramRef resolving to a
     parameter still declared in that stage's own record" (walk at L81–L112). As
     written it fails on every exempted Reduce; the invariant must be split
     (Map-only, or Reduce-exempt) — a spec-level companion the doc does not name.
  2. **GVN** — lambda nodes are pure and GVN-eligible
     (`internal/ir/node_lambda.go:50-51`); retained params participate in identity
     ("two lambdas over the same atoms and body node are one value"), so CSE still
     behaves. No conflict.
  3. **typeinfer** — retained paramRefs are typeable today
     (`internal/opt/typeinfer/rules.go:116` KindParam, `:120` KindInputParam). No
     conflict.
  4. **memopt / tiling** — lambdas see values only (`doc/tca.md` §4.2 L280: "A
     lambda sees operand values, not induction variables, tensor addresses, or
     indices"), so reduce bodies carry no MemLoad/Iter subscripts; the tiling record
     is keyed by BIV/MemLoad identity and is untouched. No conflict.
  5. **scheduling** — an un-inlined body is a pure subgraph rooted at the param
     atoms (KindInputParam, no operands → SN 0), so the schedule pass lands the
     combiner body in the prologue Group, not inside the loop; the Reduce node
     itself keeps its deep SN via its piped operand and the completion-edge
     repointing (`internal/ir/astcfg.go:27-41`, `:407-428`). This is acceptable
     only because the body is a function definition to be EXTRACTED as a callable,
     not emitted as a statement — a duty the doc gestures at (L160–L162) but never
     specifies (FL-10, FL-26).
  6. **DCE** — the body stays live through the Reduce → lambda-node operand chain;
     no orphaning. No conflict.

### E3. Reducer-lower algorithm: dual route + 4 insertion points (L111–L129) — CONTRADICTION (FL-1 BLOCKING, FL-2 GAP)

- Doc: identify currentLoop via "Reducer's own SN to learn its tighted Loop BB — OR
  using reducer's first operand to learn its tighted Loop BB, which is the last
  dimension" (L112–L114); then (2.1) construct the object at the first Group of the
  Loop BB Cluster (L117–L118), (2.2) insert `Init` in the parental loop of
  currentLoop — or the first Group if none (L119–L121), (2.3) inside currentLoop map
  `<ReducePrevious>` to `current_value()` and call `Reduce(first operand)`
  (L122–L125), (2.4) insert `Combine` in currentLoop's parental loop/block, right
  after the currentLoop statement (L126–L129).
- Ground, three-way:
  1. The dual route itself is CONFIRMED machinery: `internal/ir/astcfg.go:35-38`
     ("final_lower's reducer rule ('using Reducer's own SN to learn its tightest
     Loop BB — OR using reducer's first operand') reads exactly this duality") and
     `:414-418` (a Reduce lands in its nest's epilogue with SN == the reduced
     loop's depth; its first operand's SchedBlock IS that loop's block). Both
     routes identify the SAME loop — the tile-INSIDE loop of the reduced dimension,
     where the piped map value is produced.
  2. tca.md's placement law (`doc/tca.md` §5.10 L723–L728): `Init` immediately
     BEFORE the tile-between loop `@r_o`, `Combine` immediately AFTER it; "They are
     never placed inside the tiling loops of `@r`. This way one reduction spans all
     tiles of `@r`, and no per-tile partials are produced" (L725); worked example
     L730–L748 (Init at L733 before `seq_for(@k_o)` L735, Combine at L744 after it);
     `Reduce` replaces the abstract reduce inside `@r_i` (L726). The loop-tiling
     Issue #1 ruling agrees ("reducer Init before element loop / Combine after")
     and `doc/runtime_objects.md` §8.4 L638 records the same contract.
  3. The pipeline guarantees the split: tile-lower runs before scheduling before
     final lower (`doc/opt.md` L26–L30), and it EMITS between/inside loop pairs for
     every T>1 dimension (`doc/runtime_objects.md` §5.4 value table, L394–L403).
- Adjudication (the critical question): after tile-lower, the reduced dimension is
  TWO loops. The doc never mentions the between/inside split anywhere; its
  "currentLoop" (L114) is defined as the loop identified by the dual route, which is
  the tile-inside loop `@r_i`. Under that (only supported) reading:
  - 2.2 puts `Init` in the parental loop of `@r_i` — for the matmul nest
    `@m_o → @k_o → @m_i → @k_i` (tca.md §5.9.1 L630–L636) that is `@m_i`, i.e.
    INSIDE the tile-between loop `@k_o`;
  - 2.4 puts `Combine` "right after the currentLoop statement", i.e. inside `@m_i`
    immediately after `@k_i` — also inside `@k_o`.
  Both then execute once per tile: per-tile `Init` re-initializes the accumulator
  every round and per-tile `Combine` publishes a one-tile partial — exactly the
  "per-tile partials" tca.md L725 forbids, and numerically wrong for any multi-tile
  reduction (K=512 at T=64 → 8 silently-lost tiles; the worked example
  `doc/runtime_objects.md` Appendix A L743–L750 is exactly such a K=512/T=64
  kernel). The alternative reading (currentLoop = `@r_o`) is not supported by the
  doc's own words — "tightest Loop BB" (L112) and the operand's SchedBlock
  (astcfg.go L417–L418) both point at `@r_i` — and would additionally break 2.3,
  whose `Reduce(first operand)` value only exists inside `@r_i`.
  - Special case: when the reduced dim is untiled (T==1, single loop —
    `doc/runtime_objects.md` §5.4 table L396/L398), currentLoop is that single loop
    and 2.2/2.4 reduce to tca.md L755's simple form ("Init before the `@r` loop and
    Combine after it"). The contradiction is therefore specifically on the NORMAL
    tiled path, not the edge case.
- Verdict: CONTRADICTION. The algorithm as written misplaces Init/Combine one level
  too deep whenever the reduced dimension is tiled (the default). Required fix (see
  §8 item 1): currentLoop for placement (2.2/2.4) must resolve to the tile-BETWEEN
  loop `@r_o` — Init immediately before it, Combine immediately after (tca.md
  L725) — while 2.3's Reduce call stays inside `@r_i`. The between partner is
  discoverable from `@r_i` (the inside loop's offset/round chain references it;
  golden `doc/runtime_objects.md` L747–L750 shows the shared-dim between/inside
  dimvar pair) or from the LNT loop metadata (`internal/ir/astcfg.go:326-333`),
  but the doc must SAY so (FL-2).

### E4. Reduce operand layout and "last dimension" semantics (L34–L38, L141–L144) — CONTRADICTION (with GAP sub-items FL-3, FL-5)

- Doc: "Performs reduction against last dimension of provided input" (L35); a Reduce
  "BINDS statically to a dimension, which binds to a iteration" (L36); operand
  layout "#0: SSAUse -> current loop/iteration's value, #1: KeyedAttribute, #2:
  SSAUse -> <Lambda> node" (L141–L144).
- Ground:
  1. Semantics: `doc/tca.md` §2.5 L81 — reduce "Reduces the **last dimension in the
     producing map's `as` order**", and §2.7 rule 3 (L133) makes the producing map's
     `as` the ONLY dimension-order source ("A downstream `reduce` reduces the last
     variable of `as`"). "Last dimension of provided input" is a different claim:
     the input tensor's own axis order is not authoritative (a transposed store or
     permuted `as` decouples them, §2.7 rule 6 L136); the producing MAP's as-order
     is. CONTRADICTION (wording-level but semantic: it names the wrong lookup key).
  2. Layout: the builder wires `[piped, KeyedAttribute("init"), lambda]`
     (`internal/ir/node_lambda.go:100-108`) — but the catalog's layout contract is
     "piped, KeyedAttribute(init), trailing As?, trailing lambda node"
     (`internal/ir/catalog.go:36`), and `AsOf` deliberately tolerates a trailing As
     on a Reduce (`internal/ir/node_lambda.go:158-178`). The doc's flat #2=Lambda
     breaks the moment an As is present (its own pre-lowering sketch L97–L102
     includes `As(...)` — the two halves of the doc disagree). FL-5 (MINOR).
  3. Identification mechanism: tca.md §5.9 L618–L620 pins the mechanism in the spec
     world — "every `reduce` is annotated with the induction variable it reduces,
     written `reduce<@r>`... Tiling splits that variable into two loops, so 'the
     innermost dimension' no longer identifies it unambiguously." In the IR TODAY
     no reduce<@r>-equivalent annotation exists: `NewReduce` wires no As
     (node_lambda.go L105–L108) and no dimension payload; the reduced dim is only
     DERIVABLE — Reduce.op0 → producing Map → `AsOf(map)` → last as-operand
     (tca.md L133's rule made mechanical). The doc's dual route (L112–L114)
     identifies a LOOP, not the reduced DIMENSION, and says nothing about the
     producing map's As. GAP (FL-3): the doc must state the identification
     mechanism (carry an annotation, or define the op0→AsOf derivation, including
     the reduce-after-reduce chain case, tca.md L93) — and note that this must run
     BEFORE Map elimination (E7) rewires Reduce.op0 away from the Map.
- Verdict: CONTRADICTION (the L35 claim conflicts with tca.md L81), plus the FL-3
  identification GAP that blocks implementing even a correct placement rule.

### E5. RuntimeCallMethod node (L131–L136, L145–L178) — GAP (with internal contradictions FL-8/FL-9/FL-11)

- Doc: new expression node `RuntimeCallMethod`, "calling a method on a runtime IR
  node": input #0 = another Runtime node, #1 = "name of methiod" [sic], #2–K =
  parameters (L132–L136); usage sketch L147–L178 including
  `RuntimeCallMethod(%reducer, "Init", <Lambda>)` (L151), `"...", "current_value")`
  (L156), `"...", "Reduce", %current_value)` (L171), `"...", "Combine")` (L173–175);
  the lambda is invoked via "SSAUse for lambda to invoke it, ie rewritten as
  something like, ExprCall" (L160–L162), and "This node effectively codegen (C++)
  as reduce_result = lambda(reduce_current, previous)" (L164–L167); special body
  operands `<ReducePrevious>` and `<...IR node for current...>` (L104–L107).
- Ground and findings (each a named gap or contradiction):
  1. **Taxonomy unspecified (FL-7, GAP)**: kind, tier, family, arity are unstated.
     New kinds are legal — append-only within tier before `KindEnd` since
     KindVersion 5 (`internal/ir/kind.go:12-19`, `:30`) — but each needs the
     six-companion set the intrinsics already established
     (`internal/ir/node_intrinsic.go:21-24`): node struct + catalog row,
     allocNode case, verify shape case, typeinfer rule, legality-window row
     (unknown kinds fail CLOSED, `internal/opt/legality.go:57`),
     contentHash/contentEqual. None are named.
  2. **Method-name-as-string OPERAND vs the established string PAYLOAD precedent
     (FL-8, CONTRADICTION)**: operands are nodes — `wireOperands` takes Nodes
     (`internal/ir/node_intrinsic.go:42`); the two precedents for string selectors
     are the intrinsic `field` payload and callNode.name
     (`internal/ir/catalog.go:68-69`; `internal/ir/node_intrinsic.go:29-43`), both
     PAYLOADS, never operands. A bare string cannot ride input slot #1; the method
     name must be a payload (hashed via the mixString precedent) or a distinct kind
     per method. The doc's shape is unimplementable as drawn.
  3. **Naming drift (FL-9, MINOR)**: `RuntimeCallMethod` (L132) vs
     `RuntimeMethodCall` (L210–L211) — the doc names the same node two ways.
  4. **Lambda invocation undefined (FL-10, GAP)**: "something like, ExprCall"
     (L161) — no such kind exists; `KindCall` is "registered pure builtin call"
     ONLY (`internal/ir/catalog.go:48`), not user-lambda invocation; and
     `<ReducePrevious>`/`<current>` (L104–L107) are never given kinds, layout, or
     scoping. Under the (correct) exempt-from-inline fix E2, the combiner body's
     acc/now references are ordinary KindParam refs to the lambda's declared
     InputParams (`internal/ir/node_lambda.go:14-19`) — the doc should say the
     formals ARE the previous/current carriers, not new node vocabulary.
  5. **Internal contradiction on who applies the lambda (FL-11)**: L164–L167
     materializes an explicit call node `lambda(reduce_current, previous)` in the
     lowered IR, while L168–L171 says "since lambda is provided to Reducer object,
     internally it will call the lambda to do the job" — and
     `doc/runtime_objects.md` §11 L733 records the internal-call contract ("The
     Reduce lambda body is NOT inlined — the reducer calls it internally"). Both
     cannot hold: if the reducer owns the lambda (Init carries it, L151), the
     explicit call node must NOT exist, and `%current_value = <Input #1 of Reduce
     IR Node>` (L158) is dead sketch (it also contradicts the doc's own layout,
     where #1 is the KeyedAttribute — another FL-9 typo).
- Verdict: GAP — the node is the right shape of mechanism (a uniform method-call
  expression over runtime objects, matching inventory entry 43
  `doc/runtime_objects.md` L71) but its contract is underspecified at every
  load-bearing point above.

### E6. The 3 runtime reducer specialization nodes (L137–L140) — GAP (Global = OUT-OF-SCOPE-OK as v0-dead scaffolding)

- Doc: `RuntimeSequentialReducer`, `RuntimeWarpReducer`, `RuntimeGlobalReducer`
  (L137–L140); the exposed Reducer interface is identical across specializations
  except the constructor (L80–L83).
- Ground: the selection rule is unstated in the doc but fully determined by the
  system: schedule.md 4.3[2] (L136–L139) — each loop's SoC information "Will be
  used for reducer lowering. Reducer will be lowered into different types,
  SEQ/WARP/GLOBAL" — carried today as `ASTLoopMeta{Soc, Dosoc, LoC}`
  (`internal/ir/astcfg.go:315-324`, "consumed by reducer lowering: a reducer
  lowers differently under a SEQ/WARP/GLOBAL loop"; Soc 0=seq/1=cta/2=thr,
  `:319-323`). The doc never says: WHICH loop's Soc (the reduced dim's — see E3/E4
  identification), nor the soc→node mapping (0→Sequential, 2→Warp, 1→Global).
  GAP (FL-13). Family is likewise unstated: `FamilyCollective` is reserved with
  zero members exactly for "reducers/collectives" (`internal/ir/kind.go:182-184`)
  — the natural home; the doc should assign it (FL-14).
- `RuntimeGlobalReducer` and CTA-parallel reduced dims: `doc/tca.md` §5.10 L717 —
  "v0 scheduling policy: no preset assigns CTA parallelism to a reduced dimension.
  S6 runs a full reduction on one CTA... Global reduction support can therefore be
  deferred." The Global node is therefore dead code on every v0-reachable input —
  consistent forward scaffolding (the interface uniformity claim L80–L83 keeps it
  cheap): **OUT-OF-SCOPE-OK**, provided the doc SAYS it is v0-unreachable
  (otherwise implementers must build grid-wide collectives — tca.md L713–L715 —
  for no reachable input).

### E7. Map elimination (L180–L185) — CONTRADICTION (FL-22: the "no Lambda" premise is false; the identity claim is right)

- Doc: "Remove Map's IRNODE with Body Expression's SSAUse" (L182–L183); "Map is an
  idenetity IRNode that just return value of evaluation of its Body Expression"
  (L184); "Notes MAP must be inlined before, hence it DOES NOT HAVE Lambda at all"
  (L185).
- Ground:
  1. The identity claim is CORRECT post-inline: after simplify-inline the body's
     paramRefs were RAUW'd onto the Map's value operands
     (`internal/opt/simplify/inline.go:186-214`), so the body is a pure expression
     over the Map's own inputs and the Map is semantically a projector onto its
     body. Replacing Map uses with the body node (RAUW + erase) is mechanically
     expressible with the existing choke points.
  2. The "DOES NOT HAVE Lambda at all" premise is FALSE: inline strips the lambda
     PARAMETERS — `ClearLambdaParams` kills dead param DECLARATIONS
     (`internal/opt/simplify/inline.go:112-131`; `internal/ir/mutate.go:203-213`)
     — but the lambda NODE itself survives as the Map's trailing operand holding
     `[body]` (`internal/ir/node_lambda.go:47-64`; no pass erases it — the only
     lambda-node erasure in `internal/opt/` is inline.go's param-atom cleanup,
     L134–L147). What the doc means is "has no lambda PARAMETERS". The correction
     matters because elimination must also dispose of the orphaned trailing lambda
     node (and the trailing As node, which AsOf reads,
     `internal/ir/node_lambda.go:158-178`) or leave them to a post-lower DCE that
     the pipeline does not yet schedule (the loop-tiling rulings memory explicitly
     defers "post-lower DCE for rewrite debris" to final-lower/codegen).
- Verdict: CONTRADICTION with the implementation's actual post-inline shape (the
  fix is a one-line correction plus a cleanup rule, FL-22).

### E8. KernelInfo construction (L187–L214) — CONFIRMED (concept + placement reconcilable; node-shape gap cross-ref FL-7/FL-8)

- Doc: one runtime object for "all cuda kernel related information", "inserted at
  very first of the kernel" (L191–L192), "PREPEND to the starting block" (L197),
  "insert the code into the **very first Group block in kernel**" (L206), consumed
  for `total_cta` / `total_thread_per_cta` (L209–L211) and by the mapping
  constructors (L249–L262).
- Ground: `KernelInfo` is already an inventory object — entry 40
  (`doc/runtime_objects.md` L68) with the §11 row (L731: "One runtime object at
  kernel entry holding all cuda kernel info: total_cta(), total_thread_per_cta().
  Constructed at the very first block") — and the mapping objects are already
  specified to be "constructed at LNT entry from KernelInfo totals + the portion
  partition for this LNT" (§2.1 L196–L199). Placement: the kernel Body IS a Group
  (`internal/ir/astcfg.go:335-341`; schedule.md L35–L36 "Body: Which is a Group")
  whose items are per-LNT [prologue, LBBN, epilogue]; prepending KernelInfo to the
  Body Group puts it before every cluster — the first statement of the kernel. The
  Prototype block renders FIRST (`internal/ir/astcfg.go:43-55`, schedule.md
  L144–L161) but it is the SIGNATURE scope (InputParam/MemNew placeholders, not
  statements), so "very first Group block in kernel" remains true among statement
  blocks. KernelInfo is loop-invariant → SN 0 (schedule.md L170–L173), the same
  scope the allocation carve-out already uses. Compatible as placed.
- Residual gaps (cross-referenced, not re-litigated here): the
  `RuntimeConstruct("KernelInfo")` node carries its class name as a string
  ARGUMENT (L207) — same operand-vs-payload violation as FL-8, plus an unspecified
  kind/family (FL-7); who BEYOND the mapping constructors consumes
  total_cta/total_thread_per_cta (e.g., nothing else in v0) is answerable but
  unstated. The doc is also silent on the Prototype/Gmem metadata the kernel
  carries (`internal/ir/astcfg.go:355` — Gmem exists "for final_lower's
  launcher-buffer materialization") — final_lower owns that materialization and
  the doc never mentions it (tracked in §4).

### E9. Per-LNT RuntimeGridMapping / RuntimeThreadMapping (L216–L264) — GAP (adjudicated COMPLEMENTARY to the existing query intrinsics; two load-bearing specs missing: FL-17, FL-18)

- Doc: per LNT, 2 runtime objects; "PREPEND to the starting block of each Loop BB
  Cluster" (L220); C++ sketch `GridMapping<3, 2, 5, -1>` / `ThreadMapping<2, 2,
  -1>` constructed from kernel_info (L228–L242); IR sketch
  `RuntimeGridMapping(%kernel_info, 3, 2, 5, -1)` / `RuntimeThreadMapping(...)`
  (L249–L262) with the note "we do not use RuntimeConstruct, but use
  RuntimeGridMapping/ThreadMapping IR NODE" (L247).
- Ground and adjudication (complementary vs contradictory):
  1. **Complementary, not contradictory.** The tile-lower pass ALREADY emits the
     QUERY side: `KindIntrinsicGridMap`/`KindIntrinsicThreadMap` nodes carrying
     only a const-u32 DoSoC operand + a string field payload, at five body sites
     (`internal/opt/tiling/lower_rewrite.go:322`, `:330`, `:357`, `:374`, `:386`;
     `internal/opt/tiling/lower_cache.go:461`; constructors
     `internal/ir/node_intrinsic.go:35-54`). runtime_objects.md records the
     objects as "[spec'd object; queries emitted]" (inventory entries 7–8, L35–L36)
     and specifies construction "at LNT entry from KernelInfo totals + the portion
     partition for this LNT" (§2.1 L196–L199) — i.e., the system was DESIGNED
     expecting exactly these construction nodes (inventory entry 41, L69). The
     doc's nodes complete that design; they do not collide with the queries.
  2. **Portion semantics consistent with the D3 ruling**: ONE GridMapping + ONE
     ThreadMap per LNT, multi-dimensional, Σ CTA portions = PORTION_CTA = 10, Σ
     warp portions = PORTION_WARP = 8, last portion may be −1 (remainder) —
     loop-tiling-design-rulings memory (D3); runtime_objects.md §2 L165–L169. The
     doc's example `2, 5, −1` of 10 (L232–L234) and `2, −1` of 8 (L238–L240)
     satisfies both sums. (The example shows 3 grid dims — legal as a general-model
     illustration, but v0's searcher policy is ≤1 CTA dim + ≤1 independent WARP dim
     per LNT — runtime_objects.md Appendix B L772 — which the doc does not
     restate; MINOR, FL-20. Two comment typos: L261 "2, // 20%" vs L239
     "2/8 = 25%"; FL-20.)
  3. **FL-17 (GAP): the portion DATA SOURCE is unspecified.** The portions live in
     the preserved "tiling-record" analysis-cache entry — the designed hand-off:
     `internal/ir/astcfg.go:63-66` ("per-LNT runtime mapping construction data
     stays in the preserved 'tiling-record' cache entry (the schedule pass
     preserves it; final_lower fetches it — importing tiling here would invert the
     dependency)") and `:326-329`. The doc says nothing about where 3/2/5/−1 come
     from; the record (per-Iter BIV-keyed SoC/LoC/tile_size + WarpPortion —
     loop-tiling rulings memory, D2/D3) is the only source. Must be stated.
  4. **FL-18 (GAP): the BINDING between queries and constructed objects is
     unspecified.** The query intrinsics carry NO object operand
     (`internal/ir/node_intrinsic.go:26-43`) — they resolve "the GridConcurrencyMap
     constructed for the ENCLOSING LNT nest" by scope (runtime_objects.md §2.1
     L184–L185, §1.6 L152–159). The doc never states the rule (lexical enclosure:
     a query inside an LNT cluster resolves to that cluster's preceding
     construction). Codegen and any MLIR backend must be told this explicitly.
  5. **Internal placement contradiction (FL-15, CONTRADICTION-within-doc)**: the
     prose says "starting block of each Loop BB Cluster" (L220) but the IR comment
     says "insert the code into the **very first Group block in kernel**" (L245) —
     a copy-paste from the KernelInfo section. Per-LNT scoping (D3 ruling,
     runtime_objects.md §1.6) requires the PROLOGUE Group of each cluster — which
     exists for exactly this purpose (`internal/ir/astcfg.go:205-211`,
     NewASTGroupInNest; schedule.md L41–L42 "All LNT will have a header block,
     Group" and L85). The prose is right; the comment is wrong; the doc must fix
     itself.
- Verdict: GAP — complementary architecture, two load-bearing specifications
  (portion source, query binding) plus one self-contradiction to fix before
  implementation.

### E10. IR purity / backend-leak assessment (user point 2; §3.x of the doc's own structure) — CONFIRMED (same platform-model commitment the system already makes; no NEW semantic leak)

- Vocabulary introduced by the doc: GridMapping/ThreadMapping, Warp/Global reducer,
  total_cta, total_thread_per_cta, KernelInfo, Runtime* node names, "cuda kernel
  related information" (L191).
- Ground: the system's IR and specs ALREADY commit to this platform model at the
  same depth: soc annotations 0/1/2 = seq/cta/thr on loops
  (`internal/ir/astcfg.go:74-77`; runtime_objects.md §9 L656), BlockCTALoop /
  BlockThrLoop / BlockSeqLoop block kinds (astcfg.go L91–L99), CBLId/CTLId
  linearization (doc/tca.md §5.7.1, cited in schedule.md), GridConcurrencyMap /
  ThreadMap as named runtime objects (runtime_objects.md §2),
  PORTION_CTA/PORTION_WARP constants (runtime_objects.md §1.2 L104), and the tca::
  namespace for all runtime objects (doc/cg.md L28–L31). The two-backend plan
  (CUDA C++ now, MLIR next — doc/cg.md L5–L11) never promised platform-neutral
  names; MLIR consumes the same model (§6 below).
- Naming leaks (cosmetic, renamable without IR semantics change): "Global" for the
  CTA-scope reducer, "cuda kernel related information" phrasing. Semantic leaks
  (CUDA-specific behavior baked into IR semantics): NONE found beyond what the
  soc/portion model already bakes in — the method-call surface
  (RuntimeCallMethod) is backend-neutral (any backend must provide the runtime
  objects; MLIR lowers them to calls, §6), the portion model is a platform
  abstraction (fractions of a 1-D normalized launch — runtime_objects.md §1.5
  L145–L150), and nothing in the doc puts warp shuffle/smem mechanics INTO the IR
  (those live in the runtime contracts, runtime_objects.md §7–§8).
- Verdict: CONFIRMED compatible — the doc extends the existing platform vocabulary
  rather than leaking a new one. The one forward-compat obligation worth naming:
  keep method names and constructor shapes in a runtime-owned registry (cg.md's
  tca:: namespace discipline) so a future backend can swap the object layer
  without IR changes.

## 4. Mechanical-translation completeness (user point 3)

Walking the post-final-lower IR exactly as the doc leaves it, what must a backend
(CUDA C++ today, MLIR next) STILL invent? Each item names the construct that fails
to carry the information and the disposition (carried elsewhere in the system =
contract reference; carried nowhere = GAP).

1. **Query→object resolution (construct: `intrinsicgridmap/threadmap` nodes).**
   The intrinsics carry no object operand (E9/FL-18); a backend must invent the
   lexical-enclosure rule (query resolves to the innermost enclosing LNT cluster's
   constructed mapping). Contract exists (runtime_objects.md §2.1 L184–L185) but is
   not in the doc. GAP in the doc, closed by FL-18's fix.
2. **Loop-header materialization (construct: Loop blocks).** NOT a gap in the
   system: the IR carries iv/bound (astcfg.go L149–L171), soc/dosoc/loc annotations
   (astcfg.go L240–304; runtime_objects.md §5.4), saturating tile_max bound chains
   (runtime_objects.md §10.1); the templates (thr_loop whole-body no-op, cta_loop
   guarded, seq_loop native) are RUNTIME contracts (runtime_objects.md §5
   L326–L369) that the backend implements, not invents. The doc adds nothing here
   and breaks nothing. COMPAT.
3. **Launch configuration (construct: nothing).** The host-side launcher needs
   gridDim.x/blockDim.x; these derive from the tiling Params (MaxCTA=100 /
   MaxThreadsPerCTA=512 — runtime_objects.md §1.1 L81 ff., §1.4), which are
   COMPILER-side thresholds not carried in the post-final-lower IR or the kernel
   metadata (schedule.md 4.2/4.3 carry GMEM and per-LNT mapping info only). The doc
   is silent; the compiled-output contract for the launcher is unspecified anywhere.
   GAP (FL-25).
4. **Reducer scratch and accumulator TILES (construct: RuntimeCallMethod scalar
   interface).** tca.md §5.10 L727: non-reduced dims between `@r_o` and Reduce
   force a TILE of accumulators ("reducer state is a tile of accumulators... Reduce
   indexes into the tile, and Init/Combine cover the whole tile"); L711: general
   combiners need VALIDITY TAGS; L686: "Scalar interface names are illustrative;
   tiled accumulator state additionally uses slot indices". The doc's scalar
   `init_`/`current_` interface (L72–L76) has no slot/tile/validity story at all —
   the backend would have to invent the entire accumulator-tile memory model.
   GAP (FL-24).
5. **Fused consumers after Combine (construct: the Reduce→Combine replacement).**
   tca.md §5.10 L728 places `%valred` consumers "after `Combine`, inside a
   re-iteration of the non-reduced inner dimensions" (worked example L744–L748).
   The doc replaces the Reduce with Combine (L173–L176) and never says where the
   fused `map | store` tail goes or that it must RE-ITERATE. The schedule pass
   lands such consumers in the nest epilogue (astcfg.go L27–L41) — the re-iteration
   loops are NOT in the scheduled IR (the nest's inside loops are not re-entered
   after the epilogue). GAP (FL-23) — the single largest "backend must invent"
   item after FL-1.
6. **Lambda materialization as a callable (construct: the retained lambdaNode).**
   The combiner must become a C++ callable (the `Lambda` argument of Init, L151).
   Capture/linkage mechanics — a `__device__` free function? a functor? how the
   param atoms become formal parameters — are unspecified (E5/FL-10, FL-26). The
   doc's only gesture is "rewritten as something like, ExprCall" (L161), which is
   not a mechanism. GAP.
7. **Reducer<T> T-binding (construct: Runtime*Reducer constructors).** The
   runtime objects are templates (`template <typename T> class Reducer`, L48); the
   doc never says where T comes from. tca.md §4.2 L293 supplies the rule the doc
   should cite (v0 reducers homogeneous: element/init/combiner/result one type —
   i.e., T = the init KeyedAttribute's IR type via ReduceInitOf,
   `internal/ir/node_lambda.go:180-194`). GAP (FL-27).
8. **RuntimeCallMethod dispatch (construct: the method-name payload).** How the
   generated code dispatches "Init"/"Reduce"/"Combine"/"current_value" — virtual,
   template, switch — and how the tca:: qualification applies (cg.md L28–L31) are
   unspecified; runtime_objects.md entry 43 records the node but not the dispatch.
   GAP (cross-ref FL-7/FL-8; enumerated for the runtime library in §5).
9. **GMEM launcher buffers (construct: Prototype/Gmem metadata).** Carried by the
   system (ASTKernel.Gmem pairs each MemNew with its extent "for final_lower's
   launcher-buffer materialization" — astcfg.go L353–L355; schedule.md 4.2
   L130–L132; Prototype block astcfg.go L43–L55): the backend reads kernel
   metadata, does not invent. The doc's silence is a documentation omission only
   (final_lower owns this surface). MINOR.
10. **Map-elimination debris (construct: eliminated Map's trailing As + lambda
    nodes).** E7/FL-22: the orphaned As and lambda([body]) nodes must be erased or
    explicitly left to a post-lower DCE that no pass currently schedules. MINOR-to-
    GAP depending on the chosen cleanup rule.
11. **SSAName / local symbols (construct: unresolved SSAUse).** NOT a gap: cg.md
    L18–L19 promises codegen "generat local symbol on the fly"; 2.1's "provide a
    SSAName" (L118) rides that mechanism. COMPAT.

Summary: items 1, 4, 5, 6, 7 are true information gaps in the doc (a backend would
have to INVENT semantics, with silent-wrong-answer risk for 4 and 5); items 3 and 8
are contract gaps to close in the runtime library boundary (§5); items 2, 9, 11 are
already carried by the system's contracts.

## 5. Runtime object boundary (user point 4)

### 5.1 The design's explicit runtime surface

Enumerated from the target (this is what the generated backend needs to RUN):

1. **Reducer<T> + Sequential/Warp/Global specializations** (L40–L83, L137–L140):
   uniform interface Init/Reduce/Combine + helpers `init_value()`/`current_value()`
   (L64–L76), constructor-only specialization (L80–L83).
2. **KernelInfo** (L187–L214): constructor + `total_cta`/`total_thread_per_cta`
   queries (L210–L211).
3. **Per-LNT GridMapping + ThreadMapping** (L216–L264): constructors taking
   (kernel_info, ndim, portions..., −1).
4. **The RuntimeCallMethod dispatch surface.** Method names that can appear,
   enumerated exhaustively from the doc: `Init` (L151), `current_value` (L156),
   `Reduce` (L171), `Combine` (L173–L175), `total_cta` and `total_thread_per_cta`
   (L210–L211), and the interface helper `init_value` (L67) — seven names. Any
   closed registry the runtime implements must be exactly this set (plus future
   warp-specialized queries, below).
5. **Lambdas as callables** (L151, L160–L171): the `Lambda` argument of Init.

### 5.2 Implicit reliance (pre-existing pipeline objects the output still needs)

Loop primitives `thr_loop`/`cta_loop`/`seq_loop` (runtime_objects.md §5, inventory
15–17); tile loaders/fills/buffers (inventory 19–27); barriers (§7, inventory 30);
masks (§8, inventory 31) — including `warp_index`/`lane_index`/`is_first` queries
(inventory 11) that the Warp reducer's down-reduce requires (§8.4 L636–L643); the
launch contract (§1, inventory 1–6); umin/umax builtins (inventory 12–13). The doc
neither duplicates nor disturbs these — correct layering.

### 5.3 Two-direction cross-check vs doc/runtime_objects.md

**Direction (a) — the doc implies, runtime_objects.md LACKS** (all GAPs to close in
the inventory; mergeable as rows):

| # | Implied by | Missing | Inventory ref |
|---|------------|---------|---------------|
| a1 | `RuntimeConstruct("KernelInfo")` generic node (L207) | A node-kind row: the KernelInfo OBJECT is entry 40, but the generic RuntimeConstruct KIND (or a dedicated KernelInfo kind, as the mappings got at L247) is unrecorded | entry 40 (L68) |
| a2 | `Reducer<T>` template (L48) | The T-binding rule (T = the init KeyedAttribute's IR type; tca.md §4.2 L293 homogeneous rule) — absent from BOTH docs | entry 42 (L70) |
| a3 | RuntimeCallMethod (L131–L136) | The C++ dispatch mechanism (virtual/template/switch) + tca:: qualification discipline (cg.md L28–L31) | entry 43 (L71) |
| a4 | Init's `Lambda` argument (L151) | The lambda's runtime representation (free `__device__` function vs functor; capture discipline) — §11 L733 says "the reducer calls it internally" but not what "it" is | entry 42 (L70) |
| a5 | "provide a SSAName" (L118) | Runtime-object symbol discipline for codegen (cg.md L18–L19's on-the-fly locals is the natural carrier; unstated) | — |
| a6 | tiled reductions generally | Accumulator-tile/slot indices + validity tags (tca.md L686/L711/L727) — the recorded Reducer interface (§11 L733) is scalar-only, same as the doc; shared gap | entry 42 (L70) |

**Direction (b) — runtime_objects.md already spec'd, checked for duplication or
drift:**

| # | Recorded | vs the doc | Verdict |
|---|----------|-----------|---------|
| b1 | Reducer interface §11 L733: `Init(init, lambda)` / `Reduce(value)` / `Combine()` + helpers, lambda NOT inlined | MATCHES doc L48–L77 exactly (no drift in the recorded surface) | COMPAT |
| b2 | …but tca.md §5.10 L688–L696 specifies argument-free `Init()` | Three-way divergence: tca.md vs final_lower.md+runtime_objects.md. The init value is already carried by the Reduce's `init` KeyedAttribute (§9 L660 records `Reducer.Init(T init = <value>)`); the LAMBDA parameter is the substantive difference (tca.md never says where the combiner comes from). **Recommendation (advisory): the final_lower.md + §11 form wins** — it is the concrete implementable surface; tca.md gets a one-line amendment (its L686 already flags "Scalar interface names are illustrative") | named divergence |
| b3 | Placement §8.4 L638 "Init BEFORE the element loop, Combine AFTER it" (+ loop-tiling Issue #1 ruling) | Doc 2.2/2.4 as read on tiled dims drift INSIDE the between loop (E3/FL-1) — the recorded contract wins; doc must fix | drift → FL-1 |
| b4 | Map construction inputs §2.1 L196–L199 ("constructed at LNT entry from KernelInfo totals + the portion partition for this LNT") | Doc's constructors MATCH (kernel_info operand + portions); only the L220-vs-L245 self-contradiction drifts (FL-15) | COMPAT + FL-15 |
| b5 | Per-LNT scoping §1.6 L152–L159, §2 L165–L169 (ONE map per kind per LNT) | Doc prose consistent; example portions 2+5+(−1)|10 and 2+(−1)|8 consistent with D3 | COMPAT |
| b6 | `init` KeyedAttribute §9 L660 | Doc L99 matches | COMPAT |
| b7 | Mask contract §8.1 (thread_map single supplier) + §8.4 ownership-mask warp down-reduce | Doc never mentions masks; Warp-reducer correctness DEPENDS on this contract — reference omission, not conflict | omission |
| b8 | KernelInfo queries §11 L731 (`total_cta()`, `total_thread_per_cta()`) | Doc L210–L211 matches | COMPAT |

## 6. MLIR translation complexity (user point 5)

Per-construct mapping sketch and rating (LOW / MED / HIGH), then overall. Context:
cg.md L5–L11 reserves `internal/cg/mlir` as the second backend and L15–L20 fixes the
assumed input shape (verbs already lowered; map/reduce lowered to runtime primitive
calls; SSAUse unresolved, locals generated on the fly).

| Post-final-lower construct | MLIR mapping sketch | Rating |
|----------------------------|---------------------|--------|
| Seq/CTA/ThrLoop blocks | `scf.for` with soc/dosoc/loc KeyedAttributes as an attribute dict; bounds are SSA u32 chains (arith). Parallel semantics have NO native scf.for form: keep the CUDA design's shape — plain scf.for + mapping knowledge in the intrinsics (option i) — rather than scf.forall (option ii, only if a pass later wants parallel semantics) | MED (mechanical under option i) |
| Group blocks | region/block sequencing; no construct — pure ordering | LOW |
| Prototype block | `func.func` signature: InputParams → typed args; GMem allocations → pointer args sized from kernel metadata (astcfg.go L353–L355; schedule.md 4.2) | LOW |
| RuntimeCallMethod | `func.call` to tca::-namespace runtime functions, object as a typed-pointer SSA value (runtime-struct alloca). Fits the per-LNT object model: objects are STATEFUL (reducer accumulators), which custom ops would have to model anyway; custom dialect ops pay off only for reduction pattern-matching (future). Method-name payload → callee symbol via a closed registry | LOW–MED |
| Runtime*Reducer construction + reduction lambda | constructor `func.call`s; the lambda as an MLIR function symbol passed by reference (private `func.func` + symbol-ref operand, indirect call at Reduce time inside the runtime) | MED — entirely due to lambda materialization (FL-10/FL-26), the one non-mechanical step |
| RuntimeGridMapping / RuntimeThreadMapping | constructor `func.call`s; portion constants as i32 args/attributes | LOW (once FL-17's source is stated) |
| KindIntrinsicGridMap/ThreadMap | two small custom-dialect ops (index/dimension variants) or plain accessor `func.call`s | LOW |
| KeyedAttribute payloads | MLIR attributes on the parent op | LOW |
| Unresolved SSAUse | on-the-fly local naming — MLIR's SSA builder numbering does it natively (cg.md L18–L19 promise holds verbatim) | LOW |

**Overall: MEDIUM.** Everything except (a) the lambda-as-function-value step and
(b) the parallel-loop semantics convention is a 1:1 mapping, and (b) dissolves under
option i (the current CUDA design's own answer). What MLIR wants that the IR does
not carry — tying back to §4: the query→object binding (FL-18; in MLIR there is no
lexical runtime scope, so the binding must become explicit — passing the mapping
object's SSA value into the accessor, a mechanical rewrite final_lower could adopt
to close FL-18 structurally for BOTH backends) and the accumulator-tile story
(FL-24; memref types would carry it naturally if specified). Assessment against
cg.md's assumed input: the doc's output DELIVERS point (2) (map/reduce lowered into
runtime primitive calls — via RuntimeCallMethod + Map elimination) and preserves
point (3) (SSAUse unresolved) — but only after the §8 fixes; as written, the FL-1
placement defect lowers WRONG reductions silently in any backend, MLIR included.

## 7. Consolidated findings table

| ID | Severity | Element | One-line statement | Evidence |
|----|----------|---------|--------------------|----------|
| FL-1 | BLOCKING | E3 | 2.2/2.4 wrap the tile-INSIDE loop on tiled reduced dims → per-tile Init/Combine = per-tile partials, contradicting tca.md's placement law | final_lower.md L112–129; doc/tca.md §5.10 L725, L730–748; runtime_objects.md §8.4 L638 |
| FL-2 | GAP | E3 | Doc never mentions the tile-between/tile-inside split; "the last dimension" is ambiguous exactly where tca.md warns | final_lower.md L35–L38, L112–L114; doc/tca.md §5.9 L618–L620 |
| FL-3 | GAP | E4 | Reduced-dimension identification mechanism unspecified (no reduce<@r> annotation in the IR; op0→Map→AsOf derivation unstated; must precede Map elimination) | final_lower.md L112–L114; internal/ir/node_lambda.go:100-108, :158-178; doc/tca.md L133 |
| FL-4 | CONTRADICTION | E4 | "last dimension of provided input" vs last dimension in the PRODUCING MAP's as order | final_lower.md L35; doc/tca.md §2.5 L81, §2.7 L133 |
| FL-5 | MINOR | E4 | Operand layout omits the optional trailing As the catalog (and the doc's own sketch) carry | final_lower.md L141–144 vs L97–102; internal/ir/catalog.go:36 |
| FL-6 | GAP | E1 | "type = body type" is not today's convention (KindLambda deliberately untyped); enforcing it is an unmotivated typeinfer change | final_lower.md L22–L23; internal/opt/typeinfer/typeinfer.go:65-75; internal/opt/typeinfer/rules.go:272-273 |
| FL-7 | GAP | E5 | RuntimeCallMethod kind/tier/family/arity + the six companions unspecified (legality fails CLOSED on unknown kinds) | final_lower.md L131–136; internal/ir/node_intrinsic.go:21-24; internal/opt/legality.go:57 |
| FL-8 | CONTRADICTION | E5 | Method name as a string OPERAND violates the node-operand convention; both precedents are string PAYLOADS | final_lower.md L135; internal/ir/node_intrinsic.go:29-43; internal/ir/catalog.go:68-69 |
| FL-9 | MINOR | E5 | RuntimeCallMethod vs RuntimeMethodCall naming drift; "Input #1 of Reduce" contradicts the doc's own layout | final_lower.md L132 vs L210–L211, L158 |
| FL-10 | GAP | E5 | Lambda invocation "something like, ExprCall" undefined (KindCall is registered-builtin-only); <ReducePrevious>/<current> nodes undefined — the retained KindParam refs are the real carriers | final_lower.md L104–L107, L160–L167; internal/ir/catalog.go:48; internal/ir/node_lambda.go:14-19 |
| FL-11 | CONTRADICTION | E5 | Explicit lambda-call node sketched AND reducer-internal application claimed; runtime_objects.md records internal-only | final_lower.md L164–L171; doc/runtime_objects.md §11 L733 |
| FL-12 | COMPAT | E2 | Reduce-inline exemption coherent (D9 acc↔piped/now↔init binding is wiring, not combiner semantics); ripple surface enumerated: simplify-verify split, SN-0 body landing, function-extraction duty | internal/opt/simplify/inline.go:29-47; internal/opt/simplify/verify.go:34-41; internal/ir/astcfg.go:407-428 |
| FL-13 | GAP | E6 | Specialization selection rule unstated (reduced loop's ASTLoopMeta.Soc: 0→Sequential, 2→Warp, 1→Global) | final_lower.md L137–L140; internal/ir/astcfg.go:315-324; doc/optana/schedule.md L136–139 |
| FL-14 | GAP | E6 | Node family unassigned; FamilyCollective is reserved-empty exactly for reducers | final_lower.md L137–140; internal/ir/kind.go:182-184 |
| FL-15 | CONTRADICTION | E9 | Per-LNT mapping placement: prose "starting block of each Loop BB Cluster" vs IR comment "very first Group block in kernel" — prose matches per-LNT scoping | final_lower.md L220 vs L245; internal/ir/astcfg.go:205-211; runtime_objects.md §1.6 L152-159 |
| FL-16 | COMPAT | E8 | KernelInfo concept + first-Group placement compatible (Body Group prepend; Prototype is the signature scope, not statements) — recorded as inventory entry 40 | final_lower.md L197–L207; internal/ir/astcfg.go:43-55, :335-341; runtime_objects.md L68, L731 |
| FL-17 | GAP | E9 | Portion DATA SOURCE unspecified; the preserved "tiling-record" cache entry is the designed carrier | final_lower.md L249–L262; internal/ir/astcfg.go:63-66, :326-329 |
| FL-18 | GAP | E9 | Query-intrinsic→object BINDING unspecified (lexical enclosure); codegen and MLIR both need it stated | internal/ir/node_intrinsic.go:26-43; runtime_objects.md §2.1 L184-185 |
| FL-19 | COMPAT | E9 | Construction nodes COMPLEMENTARY to the already-emitted query intrinsics; portions consistent with the D3 ruling (Σ=10/8, −1 remainder, one map per kind per LNT) | internal/opt/tiling/lower_rewrite.go:322-386; runtime_objects.md §2.1 L196-199, L35-36 |
| FL-20 | MINOR | E9 | Example shows 3 grid dims vs v0 ≤1-CTA-dim searcher policy; "20%" vs "2/8 = 25%" comment drift | final_lower.md L232–L234, L239 vs L261; runtime_objects.md Appendix B L772 |
| FL-21 | COMPAT | E10 | No new semantic backend leak — same platform-model commitment the system already makes (soc/CTA/Thr vocabulary, CBLId/CTLId, tca:: objects); naming leaks cosmetic, renamable | §3 E10 ground; doc/cg.md L5-11, L28-31 |
| FL-22 | CONTRADICTION | E7 | "Map DOES NOT HAVE Lambda at all" is false: inline strips lambda PARAMETERS, the lambda node with body survives; As/lambda debris disposal on elimination unspecified | final_lower.md L185; internal/opt/simplify/inline.go:112-131; internal/ir/mutate.go:203-213 |
| FL-23 | GAP | §4 | Fused-consumer re-iteration after Combine (tca.md's rule) absent from the doc — backend would invent the wrong placement | final_lower.md L173–L176 (silent); doc/tca.md §5.10 L728, L744-748 |
| FL-24 | GAP | §4 | Accumulator TILES + validity tags absent; scalar-only init_/current_ interface has no slot story (tca.md flags it) | final_lower.md L72–L76; doc/tca.md L686, L711, L727 |
| FL-25 | GAP | §4 | Host-side launch configuration (gridDim/blockDim derivation) carried nowhere in the compiled-output contract | §4 item 3; runtime_objects.md §1 |
| FL-26 | GAP | §4 | Lambda materialization as a C++ callable (capture/linkage) unspecified | final_lower.md L160–L162 |
| FL-27 | GAP | §4 | Reducer<T> T-binding unspecified; tca.md §4.2's homogeneous-types rule is the answer to cite | final_lower.md L48; doc/tca.md L293 |
| FL-28 | COMPAT | E3 | The dual SN/operand identification route is confirmed by the implementation's own contract comments (complementary SchedSN/SchedBlock) | internal/ir/astcfg.go:35-38, :414-418 |

Counts: 28 findings — 1 BLOCKING, 5 CONTRADICTION, 13 GAP, 7 COMPAT, 2 MINOR.

## 8. Verdict and required fixes

**Overall verdict: compatible-with-named-fixes.**

The architecture needs no rework: the RuntimeCallMethod-over-runtime-objects design
is the completion of a surface the system already committed to
(doc/runtime_objects.md §11 L723–L737, inventory entries 40–43), the per-LNT
mapping constructors complete the already-emitted query intrinsics (FL-19), the
pass slot exists (doc/opt.md L27–L30: scheduling → final lower → codegen), and the
schedule pass's persistent state was built for exactly this consumer
(internal/ir/astcfg.go:35-38). What blocks implementation as written is ONE
algorithmic defect (FL-1: the reducer insertion points wrap the wrong loop on the
normal tiled path, silently producing per-tile partials) plus the identification
gap that would prevent even a corrected placement rule from being implemented
(FL-2/FL-3); the remainder is specification closure (node taxonomy, payload
conventions, lambda contract, data sources, bindings) and doc hygiene. No finding
requires a new IR tier, a new pass, or an architectural change.

Required fixes (each scoped to the doc section it amends):

1. **§ Reducer Lower Integration (L111–L129) — FL-1/FL-2:** resolve the placement
   loop to the tile-BETWEEN loop `@r_o` of the reduced dimension: `Init`
   immediately before it, `Combine` immediately after (tca.md L725); the `Reduce`
   call stays inside `@r_i` (tca.md L726). State how `@r_o` is discovered from the
   identified inside loop (its offset/round dimvar chain, or the LNT loop metadata,
   internal/ir/astcfg.go:326-333).
2. **§ Reducer Lower Integration (L112–L114) — FL-3:** state the reduced-DIMENSION
   identification mechanism: carry a `reduce<@r>`-equivalent annotation, or define
   the Reduce.op0 → producing Map → `AsOf` → last-as-operand derivation
   (reduce-after-reduce chains included, tca.md L93), and require it to run BEFORE
   Map elimination rewrites op0.
3. **§ Overview of Reducer (L35) — FL-4:** "last dimension of provided input" →
   "last dimension in the producing map's `as` order" (tca.md L81/L133).
4. **§ Runtime Object Node (L141–L144) — FL-5:** operand layout must include the
   optional trailing As (internal/ir/catalog.go:36).
5. **§ Runtime Object Node (L131–L136) — FL-7/FL-8/FL-9:** specify kind, tier,
   family, arity and the six companions; the method name becomes a string PAYLOAD
   (callNode.name / intrinsic-field precedent), never a string operand; unify the
   RuntimeCallMethod/RuntimeMethodCall naming; fix the "Input #1" sketch line.
6. **§ Runtime Object Node (L104–L107, L160–L171) — FL-10/FL-11:** resolve the
   lambda-application contract to the reducer-internal form (runtime_objects.md
   §11 L733); delete the explicit call-node sketch; define <ReducePrevious>/<current>
   as the retained KindParam refs or drop the vocabulary.
7. **§ Fix or Ensure Lambda (L22–L23) — FL-6:** drop "type = body type" or specify
   the typeinfer amendment (removing KindLambda from the untyped whitelist) with
   its motivation.
8. **§ Map Lower (L180–L185) — FL-22:** correct "DOES NOT HAVE Lambda" to "has no
   lambda parameters (the lambda node with body survives inline)"; specify
   disposal of the orphaned As and lambda nodes on elimination.
9. **§ Reducer Lower (L137–L140) — FL-13/FL-14:** state the specialization
   selection (reduced loop's ASTLoopMeta.Soc: 0→Sequential, 2→Warp, 1→Global) and
   the family assignment (FamilyCollective); mark RuntimeGlobalReducer
   v0-unreachable forward scaffolding (tca.md L717).
10. **§ Per LNT's Mapping Object Construction (L245) — FL-15/FL-17/FL-18:** fix the
    placement comment to match the prose (each cluster's prologue Group); name the
    preserved "tiling-record" cache entry as the portion source; state the
    query→object lexical-enclosure binding.
11. **§ Reducer Lower (new subsection) — FL-23/FL-24/FL-27:** add the
    fused-consumer re-iteration rule (tca.md L728), the accumulator-tile/slot/
    validity-tag story (tca.md L686/L711/L727), and the T-binding rule (tca.md
    §4.2 L293).
12. **Cross-doc alignment (advisory, one-line amendments owned by the author):**
    tca.md §5.10 L688–L696 to note `Init(init, lambda)` as the concrete form
    (runtime_objects.md §11 L733 already records it); runtime_objects.md entries
    40/43 to record the KernelInfo node-kind decision, the closed seven-name method
    registry (§5.1 item 4), and the C++ dispatch mechanism.

## 9. Self-check

- Every element E1–E10 carries exactly one verdict tag and dual citations (target
  lines + system file:line or doc section) — verified per element in §3.
- The E3 tiled-Init/Combine question is explicitly adjudicated (both readings of
  "currentLoop" examined; the only supported reading contradicts tca.md L723-728).
- The runtime-object boundary is cross-checked in BOTH directions with
  inventory-entry references (§5.3 a1–a6, b1–b8).
- Every post-lowering construct has a sketched MLIR mapping and the overall
  complexity is rated with justification (§6, MEDIUM).
- One overall verdict (compatible-with-named-fixes) + a 12-item numbered
  required-fix list, each pointed at its doc section (§8).
- The coverage table maps user points 1–5 to sections (§1).
- doc/optana/final_lower.md is NOT modified by this review (the file is absent
  from the review's working tree and was only ever read); the only file created
  is doc/review/final_lower.md. No files under internal/, pkg/, runtime/ touched.

---

## 10. Resolution addendum (2026-10-22, post-review)

After this review, the reducer-placement fix was agreed with the author and
`doc/optana/final_lower.md` was amended. Status changes at the DESIGN-DOC
level (implementation still pending — no code has changed):

- **FL-1 (BLOCKING) — RESOLVED**: insertion points 2.2/2.4 now bracket
  X = the OUTERMOST loop over the reduced dim (tile-between on the tiled
  path; the single loop when T==1). The per-tile-partial reading is gone;
  tca.md §5.10's law is cited inline.
- **FL-2 — RESOLVED**: the tile-between/tile-inside split is explicit in the
  algorithm (step 1), including the home-loop vs bracket-loop distinction.
- **FL-3 — RESOLVED (design)**: @r identification specified — Input #0 →
  producing Map → `ir.AsOf` → last dim of the As order; fail-closed; ordered
  before Map Lower (which the doc now states as a sequencing requirement).
- **FL-4 — RESOLVED**: "Overview of Reducer" now says "last dimension of the
  producing Map's as order", matching tca.md §2.7 rule 3.
- **FL-13 — RESOLVED**: specialization selection stated — X's
  `ASTLoopMeta.Soc`: 0→Sequential, 2(thr)→Warp, 1(cta)→Global.
- **FL-17 — PARTIAL**: the new "Tiling Provenance Record" section establishes
  the designed carrier (preserved tiling-record, per-dim entries keyed by the
  DIM node: Between/Inside/Soc/Tile). The per-LNT mapping-portion fields are
  still to be specified when the mapping-construction section is amended.
- **FL-23 / FL-24 — RESOLVED (design)**: the amended steps carry tca.md
  §5.10 points 3–4 — the accumulator-tile slot index on Reduce
  (`Reduce(@m_i, %current_value)`) and fused-consumer re-iteration after
  Combine.

Also confirmed in discussion (no doc change needed): final lowering stays
AFTER scheduling — its insertion points, SoC specialization, and SN→block
linkage are all scheduler outputs, and pre-scheduling introduction of
Runtime* kinds would hit the fail-closed legality of every earlier pass
(doc/opt.md L27–L30 order stands).

### Round 2 (same day): remaining findings resolved — design now implementation-ready

All actionable findings are now closed at the design level. Author rulings:

- **FL-5 — RESOLVED**: full operand layout stated (#0 piped, #1
  KeyedAttribute("init"), #2 optional As before the lambda, last = lambda);
  reducer CONSTRUCTION carries the state — `RuntimeXXXReducer(%init, %lambda)`
  — so the `Init()` call is argument-free. This also reconciles the
  tca.md-vs-runtime_objects Init-signature divergence in tca.md's favor.
- **FL-6 — RESOLVED**: "lambda type = body type" dropped; KindLambda stays
  deliberately untyped; T is checked against the parameter-definition types
  at Reduce lowering (tca.md §4.2).
- **FL-7 / FL-14 — RESOLVED**: dedicated construction node per object (no
  generic RuntimeConstruct); exactly two generic nodes — RuntimeCallMethod
  (method payload, ordered side effect, never CSE) and RuntimeGetAttribute
  (attribute payload, pure read on immutable objects); families: new
  FamilyRuntime (construct/call/getattr) + FamilyCollective (the three
  reducers, its first members); six-companion registration checklist; LIR
  tier, final-lower-onward legality window.
- **FL-8 — RESOLVED**: names are string PAYLOADS (intrinsicNode.field
  precedent) with closed vocabularies — methods {Init, Reduce, Combine},
  attributes {total_cta, total_thread_per_cta, index, dimension}.
- **FL-9 — RESOLVED**: RuntimeCallMethod naming standardized; the sketch's
  "Input #1" corrected to Input #0.
- **FL-10 / FL-11 / FL-26 — RESOLVED**: runtime-internal combiner
  application — the lambda is handed over at construction; the (prev, cur)
  invocation is hidden from the compiler (prev = the runtime's member
  field); NO lambda-call IR node in v0; current_value/init_value are
  runtime-side helpers the lowering never emits. Lambda is pinned
  capture-free (params + consts only) — hoisting is trivially sound (MLIR:
  private func.func + symbol-ref; CUDA: file-scope __device__ function as
  the runtime template's Lambda parameter).
- **FL-15 — RESOLVED**: mapping objects prepend to the OWNING LNT's prologue
  Group (one pair per LNT), not the kernel's first Group.
- **FL-17 — RESOLVED (carrier)**: with the query rewrite below, the
  per-LNT mapping construction reads its portion data through the same
  preserved-record channel as the reducer provenance.
- **FL-18 — RESOLVED structurally**: final_lower rewrites every launch-map
  query into `RuntimeGetAttribute(%<LNT's mapping object>, field, dosoc)`
  (SN → block → owning nest); the binding becomes explicit SSA for both
  backends; the standalone query kinds are pre-final-lower material only.
- **FL-20 — RESOLVED**: example shrunk to the v0 policy (≤1 CTA dim, one
  independent warp dim), portions annotated as PortionCTA=10 /
  PortionWarp=8 fractions.
- **FL-22 — RESOLVED**: Map Lower debris disposal specified (RAUW →
  explicit As erasure → orphaned lambda left to mark-sweep DCE), ordered
  after Reducer Lower.
- **FL-25 — RESOLVED (scoping)**: launch configuration is a codegen
  contract item (tiling Params → kernel metadata → host launch wrapper +
  runtime launch validation), not an IR node.
- **FL-27 — RESOLVED**: T = element type of the Reduce's Input #0;
  homogeneity per tca.md §4.2 checked at lowering.
- Naming: the "Intrinsic" vocabulary is renamed to **Runtime** throughout
  (runtime objects opaque to the user; lowered nodes are mechanical literal
  representation with zero codegen assumptions).

**Verdict after round 2: GO — the design doc is implementation-ready.** All
28 findings are closed (resolved at design level or COMPAT-by-review);
implementation may proceed on doc/optana/final_lower.md as amended.
