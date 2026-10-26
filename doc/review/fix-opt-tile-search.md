# Review: doc/fix/fix-opt-tile-search.md — loop-tiling schedule search rewrite

**Reviewed:** 2026-10-26 (quick 261026-ft4)
**Against:** doc/optana/loop_tiling.md + settled rulings (F2/F4/F6/F8, R2/R3, 2026-10-20/21/24 rulings), and the implemented `internal/opt/tiling` (`search.go`, `record.go`, `verify.go`, `overlap.go`).

## Verdict

**Direction endorsed; not yet implementable as written.** The diagnosis is accurate on every count (confirmed against the code below), the generalization (per-dim tiles, searched LoC, eviction salvage, multi-dim MAPM) is the right next step, and the proposed module split is sound. But four findings are blocking: the objective function's reuse term regresses a settled ruling and cannot rank the candidates it is supposed to rank (C2), the GoC term does not encode the argument it is motivated by (C3), multi-dim same-category composition — explicitly ruled *future-until-pinned* — is admitted without being pinned (C1), and the budget model merges per-thread and per-CTA units into one pool (C4). Fix those plus the HIGH items and this doc is a good spec.

## Diagnosis confirmed — the current search is indeed trivial

All four "naive" claims check out against `search.go`:

| Doc claim | Code evidence |
|---|---|
| Uniform tile — one knob for every dim | `tileCandidates(p)` + single `tile` applied to all window levels (search.go:121, 179, 225); space ≤ 3^K × \|TileSizes\| = 243 × 4 = **972 candidates/LNT** at defaults |
| No LoC search | Portion completion (search.go:196-232): the CTA dim always takes full (LoC = MaxCTA); the WARP dim takes `WarpPortion` or auto (`AutoWarpPortion` = 4) |
| Budget gate is all-or-nothing | `smemFloor+res.smem > p.MaxCache` → the whole candidate is skipped (search.go:256-261); no eviction / partial-cache salvage |
| SoC space truncated | `v0PolicyClean`: ≤1 CTA dim + ≤1 independent WARP dim (search.go:512-523) |

So the doc's premise — the search needs to become a real optimizer over (SoC, LoC, TileSize) per dim with a cache-eviction feasibility step — is justified. The 531,441-candidate space at M=4 is a ~550× expansion over 972 and still tractable (see H2 note on cost).

## CRITICAL findings

### C1. Multi-dim same-category composition is admitted but not pinned — MALoC's sum model contradicts the product-based CTA grid

The doc's MAPM allows up to 3 CTA-assigned and 3 THR-assigned dims, but nothing in the doc says how *two* CTA dims' portions compose into the launch. This is precisely the corner the 2026-10-20/21 composition rulings left as FUTURE: *"same-category multi-dim (CTA+CTA, WARP+WARP independent) is FUTURE — combined-slicing arithmetic (product factors of the level's space, portions as grammar) must be pinned before the searcher may emit such schedules."* The current searcher enforces this as a hard skip (`v0PolicyClean`).

The settled model (F4) is asymmetric and the doc's uniform sum presentation matches neither side:

- **CTA dims compose by PRODUCT**: the grid is decomposed into tuple coordinates (reverse stride-shape of `cuda_linear_id`); "CTA total per LNT = product over CTA dims", bounded by gridDim.x.
- **WARP dims compose by SUM**: disjoint whole-warp shards; "threads = sum of disjoint whole-warp shards" ≤ warps/CTA.

Consequence for the doc's MALoC: `total_cta_loc ≤ MaxLoc.cta = 10` (a **sum** of portions) is not the CTA-side feasibility condition. Concrete failure: two CTA dims with LoC (5,5) — Σ = 10, passes MALoC; but under the product model that is 50 × 50 = 2500 CTAs against MaxCTA = 100 → launch-infeasible. Under a slice reading (50 + 50) it contradicts the tuple-decomposition GridMapping contract. Either way the constraint as written does not gate what it must gate.

**Fix (pick one, explicitly):**
1. **Keep the v0 policy** (≤1 CTA + ≤1 independent WARP per LNT) as a stated constraint in this doc — then MAPM 3+3 is aspirational dead weight and the LoC search space collapses usefully (a lone CTA dim is always full; see M3); or
2. **Pin the grammar**: per-dim extents derived from portions, product over CTA dims ≤ MaxCTA (portions as factor-selection), warp shards disjoint with Σ warps ≤ warps/CTA; derive MALoC from that arithmetic instead of asserting a flat sum.

The runtime map machinery already supports multi-dim tuple coordinates, so option 2 is a real possibility — but the portion→extent→product mapping has to be written down before anyone can implement or verify it.

### C2. The reuse term measures bytes *occupied*, not traffic *avoided* — it cannot rank equal-byte candidates and rewards waste

`reuse_score = (total_frag + total_smem) / MaxSRAM` is a **cache-bytes-occupancy** score. This is exactly the quantity the F8 ruling rejected: *"Objective: maximize GMEM traffic avoided = Σ bytes_per_tile × (reuse_count − 1), NOT cache bytes"* — which is what the current code computes and tests (`traffic += j.bytes * (j.reuse - 1)`, search.go:240).

Two defects:

- **Cannot rank what it ranks.** One site at 64 KB with reuse 2 and one site at 64 KB with reuse 64 both score 1.0; their traffic-avoided differs 32×. Bigger tiles buy more reuse per byte, and the score is blind to the multiplier — it optimizes *filling the SRAM*, not *what filling it buys*.
- **Rewards waste.** A large, low-reuse cache that happens to fit scores maximally. (Today the reuse ≥ 2 judge gate keeps reuse-1 sites out; but the objective itself would pay for them if admitted — the incentive points the wrong way.)

SRAM bytes are a **constraint** (a cost ceiling, MASRAM), not a benefit. The benefit of a cache is traffic saved. If a normalized score is wanted, normalize the traffic-avoided sum by a per-LNT reference (e.g., the same sum for the all-sequential uniform-tile candidate, or a theoretical upper bound) — not by the budget, which measures the wrong quantity.

### C3. The GoC term does not encode the granularity argument it is motivated by — and re-introduces the deleted near-constant parallelism term

The motivation (the 28×1024 example: 32×32 thread split beats a flat 1024) is an argument about **splitting a fixed thread budget across more dims**. The formula `goc_score = (total_cta_loc + total_thr_loc) / goc_max` is **quota consumption**. These are different things:

- 1 WARP dim @ LoC 8 vs 2 WARP dims @ LoC 4+4: **identical score** (Σ = 8 both), but the second is exactly the granularity win the example argues for. The formula rewards *bigger portions and more parallel dims*, never *finer splitting of the same budget*.
- Under the doc's own exhaust rule ("all concurrency will be exhausted regardlessly" — CTA dim full, last LoC = −1 remainder), Σ LoC is structurally pushed to its maximum. The 2026-10-20/21 ruling deleted the weighted parallelism term for precisely this reason: *"under mandatory per-LNT exhaustion total parallelism is CONSTANT, so it carries no ranking signal."* The same argument applies to Σ LoC.
- With α = 0.5, half the objective's resolution is spent on a term with essentially no ranking signal — effectively halving the reuse term's discrimination.
- True utilization depends on runtime shapes, which are dynamic (D12 ruling) and unknowable at compile time. If shape-aware utilization is genuinely wanted, the dormant `NominalExtent` Params knob (record.go) exists as the vehicle for a nominal-extent model — the doc should either specify that model or drop the term.

**Recommendation:** keep the objective single-term (traffic avoided, C2). If a second term survives, it needs (a) a formula that actually responds to dim-splitting, and (b) a normalization that doesn't collapse (per-term achievable ranges differ per LNT; a fixed goc_max makes the α tradeoff arbitrary across kernels).

### C4. FRAG and SMEM are merged into one 64 KB pool — dimensionally wrong

Constraint (3) sums FRAG bytes and SMEM bytes into one `total_frag + total_smem ≤ MaxSRAM` check. FRAG is **per-thread register** storage; SMEM is **per-CTA** storage. F6 ruling + the implemented Params keep them as **two separate budgets**: `MaxFragPerThread` = 256 B/thread (kernel-wide sum of per-thread bytes) and `MaxCache` = 64 KB SMEM (kernel-wide sum), both gated in the judge, the schedule feasibility, tiling-verify, and the lower's budget resolution. A single mixed pool is unsound whichever way you read it (counted per-thread it overstates SMEM headroom; counted ×threads it double-bills registers).

**Fix:** two gates — SMEM kernel-wide ≤ MaxCache; per-thread FRAG kernel-wide ≤ MaxFragPerThread — and the eviction step (H4) may trigger on either.

## HIGH findings

### H1. The assignment-slot data model is ambiguous: pre-tiling dims vs the "AFTER TILING" list

The Input section assigns a Schedule Assignment to each of the N (pre-tiling) iterations; the Constraints section says "the list of iteration is AFTER TILING … if input is 3 nests, now it is 6" and then sums "all iteration's # of CTA and THR/WARP" over that list. If the 6 post-tiling loops each carried an independent assignment, the doc's own structural rules already forbid it (CTA's TIL is forced THR; THR/SEQ's TIL is forced SEQ) — so the free slots are the M original window dims, and the constraint sums must run over the **expanded SoC-instance list**: a CTA dim contributes one CTA instance (its TBL) *plus one THR instance* (its coupled TIL) — which is exactly what the doc's "A CTA assignment implicitly uses 2 SoC" note says, and what makes MAPM.thr = 3 meaningful (e.g. 2 CTA + 1 WARP = 3 THR instances; 3 CTA + 1 WARP = 4 = infeasible). As written, "sum over the list" is not computable. **State the two-level correspondence explicitly**: assignment slots = the M window dims; all constraint sums over the expanded instance list derived from them.

### H2. M = 4 vs 5 — pick one; and mark the search-space arithmetic as an upper bound

"By default, set M = 4. Notes for now it is 5" → later "For now we clamp only 5 most nested loops" → but the arithmetic is (3·3·3)^4 = 531,441 (matches M=4; M=5 would be 27^5 ≈ 14.3M, 27× more). The current default is `WindowK = 5` (record.go). M=4 is the defensible choice — but then say the default *changes* 5→4. Also mark the count as an upper bound: the −1 remainder rule collapses the last LoC choice per category, and MAPM/CuMAP violations prune. Compile-time note: ~531K candidates × ≤ MaxCS(16) site judgments ≈ single-digit millions of cheap calls per LNT — fine in Go, but worth incremental constraint checks during enumeration (prune partial MAPM/MALoC violations instead of filtering after the fact, as the current odometer does) and note that per-site judgments depend only on (SoC/tile/portion of the dims the site touches) and are cacheable across candidates.

### H3. Search unit and cross-LNT budget policy are unspecified

R3/F6 ruling + implementation: the SMEM and FRAG sums are **kernel-wide**, threaded as running floors through LNTs in deterministic `Roots()` order (tiling.go:105-112) — an earlier LNT's chosen schedule gates later LNTs. The doc never says the search runs per-LNT (it must — the runtime maps are per-LNT scoped), nor how the 64 KB is shared. With eviction this matters more: greedy running-floors means an early LNT never evicts for a later one. State the policy (running floors in root order is the established answer — inherit it).

### H4. The "SRAM backtracking" is greedy spilling and is under-specified — pop order *is* the algorithm

What the doc describes (pop one cache out, recheck, repeat until fits) is Chaitin–Briggs-style **greedy spill**, not backtracking — no re-insertion, no search over eviction subsets. That's fine (and the analogy is apt — also fix the spelling, "Chattin-Briggs"), but then specify:

- **Pop order policy** — the whole quality of the step lives here. Natural rule: evict the least valuable site (smallest `bytes × (reuse − 1)`, or smallest traffic-avoided per byte) until both budgets fit. Alternatively, since sites per LNT are capped (MaxCS = 16), a real subset search is affordable — but pick one and name it.
- **Scoring after eviction** — the objective must be computed on the **post-eviction** site set (an evicted site contributes neither bytes nor traffic credit). The doc's ordering (backtrack → evaluate) implies this; make it explicit.
- **Per-site gates vs the aggregate budget** — sites failing their own gates (reuse ≥ 2, FRAG per-thread cap, CTA-covered refusal) are never in the candidate set; eviction only trims the aggregate. (Existing `judgeSite` behavior — inherit explicitly.)
- **Error contract changes.** Today, no feasible schedule is a hard error — "tiling is never disabled" (2026-10-24 ruling, search.go:148-158). With eviction, every structurally-valid candidate becomes feasible by evicting everything, so the no-feasible error vanishes and the worst case becomes *loops still tiled ≥ 2 with all sites demoted to GMEM*. Decide and document whether silent zero-cache degradation is acceptable or whether some minimum must hold (the 2026-10-24 ruling covered un-tiled loops, not zero-cache tiled loops — this is a genuine new decision point).

### H5. The tier rules and the window-bytes cost model are absent — and they change the numbers

The doc's storage classes (SEQ-covered → FRAG, THR-covered → SMEM) omit three settled rules, each of which changes feasibility or cost:

1. **CTA-covered sites are REFUSED** (R2 tier rule — no coherent fill variant under the v0 mechanics; implemented as the `ReasonCTACovered` gate).
2. **C1b precedence**: a thread-varying *non-covered* referenced coordinate (any INS iter assigned CTA or WARP) forces ClassFRAG regardless of the covered dims' SoC — this is the matmul lhs shape (per-thread row coordinate → per-thread window).
3. **Window-bytes model** (R2): a SMEM-class WARP-covered dim's footprint is `elem × LoC_d × 32 × T_d` — the **thread-group window**, not the tile. "Tile size … implicitly defines the SMEM size locally" under-prices WARP-window sites by the group factor (LoC×32). This exact factor is why the worked example's rhs needs ~1 MB against the 64 KB budget at defaults and is declined — without it the search will promote sites that cannot fit.

Inherit these rules explicitly (they are implemented and tested in `judgeSite` / `siteWindowBytes`), or consciously revise them — silently dropping them will regress the flagship decisions.

## MEDIUM findings

- **M1 — Determinism / tie-breaking unstated.** "Update the maximum one" must define strict `>` vs `≥` and a fixed enumeration order. The current contract is a fixed odometer order with first-strict-max wins, documented as an invariant (search.go:104-107) and load-bearing for golden tests. Keep it.
- **M2 — LoC semantics change breaks the record contract.** The doc's LoC is a **portion index** (CTA 1-10, THR 1-8); the published record's LoC is an **absolute count** (CTA dim: LoC == MaxCTA = 100, re-checked by tiling-verify, verify.go:204; WARP dim: warp count; SEQ: 1). Either keep absolute counts in the record (convert after search) or update verify.go's re-checks. The −1 sentinel must never reach the record — the lower pass consumes concrete LoC.
- **M3 — LoC preset edge semantics.** (a) Which dim is "the last" per category (innermost? textual order?) — define it. (b) Zero remainder: CTA (10, −1) leaves the second dim 0 portions — explicitly infeasible, the constraint check must reject it. (c) A lone CTA dim at LoC 1 or 5 is strictly dominated (same traffic, less parallelism) — force −1/full or let the objective handle it. (d) **F2 coupled-portion floor**: whenever a CTA dim exists, the independent warp portion must leave ≥ 1 warp portion for the CTA dim's coupled inside walk — a zero-portion coupled walk is a zero-stride infinite loop. The THR preset (1,4,8) includes 8 = full; with a CTA dim present that is an F2 violation. The constraint set must carry F2 or the searcher will emit infinite loops (the current code errors loudly on it, search.go:214-218).
- **M4 — goc_max formula.** `max(MaxLoc.cta + MaxLoc.thr, 2 × #iterations)`: the second arm never wins for M ≤ 9 (2×4 = 8 < 18); the inline comment "at most 3 + 3 will be allowed" contradicts 10 + 8 = 18 (3+3 is MAPM's *dim counts*, not LoC); and `total_cta_thr` should be `total_thr_loc` (typo — the constraint section defines `total_cta_loc` / `total_thr_loc`). If GoC survives C3 at all.
- **M5 — Tile preset list.** "Assuming it contains 32/64/128" — the current default is {32, 64, 128, 256} (record.go); dropping 256 is a real decision (256 is in-budget for rank-1 covered fp16/fp32 sites). Make the list a Params field, not an assumption, and note that **per-dim tiles multiply across covered dims** (two covered dims at T=128 = 128² × elem = 32 KB fp16 for a single site — half the budget), so the preset list is the main budget-pressure knob. Also state tile ≥ 2 legality (2026-10-24 ruling: tile 1 is not a legal schedule).
- **M6 — Naming continuity.** MaxPartition.cta/thr (new: dim-count caps 3/3) vs MaxLoc.cta/thr (= the existing `PortionCTA`/`PortionWarp` constants 10/8) vs MaxSRAM (= the existing `MaxCache` 65536). Map every new knob onto an existing Params field (or rename deliberately, once) — otherwise the implementation phase grows two vocabularies for the same knobs.
- **M7 — Constraint (5) is vacuous.** "Implicitly, the # of loop iterations constrains total # of SoC" — MAPM already caps dim counts. Delete, or restate as "the window length M caps the number of assignable dims."
- **Minor.** The findINS reference is correct (overlap.go:101) but site *formation* (sets, covered dims, tie, reuse dims) is `analyzeOverlap`/`newSiteFacts` — reference the pipeline, not just findINS. Typos: consequetive→consecutive, Heirachy→Hierarchy, Chattin-Briggs→Chaitin–Briggs, backtrace→spill/backtrack, weather→whether, Propotional, parition, constriants, recalcuate, candidative, managable.

## What is endorsed as-is

- The **structural TBL/TIL table** (CTA→(CTA,THR), THR→(THR,SEQ), SEQ→(SEQ,SEQ)) matches the implemented rewrite forms (CTA between/inside; independent WARP between at tile-chunk stride with sequential inside; coupled CTA inside at element granularity).
- **CuMAP** (≥1 CTA dim) matches the existing `hasCTALevel` gate and the CUDA thread-model requirement.
- **Per-dim tile sizes, searched LoC, eviction salvage, MAPM generalization** — all genuine improvements over the current ≤972-candidate uniform-tile search; the general shape of the new feasible area is right.
- **Deferring site discovery to the existing overlap machinery** — right call; that code is implemented and tested.
- **The Implementation section's module split** — right shape, and it maps cleanly onto the existing package (see below).

## Recommended minimal revision set (to unblock implementation)

1. **Pin the scope** (C1): keep the v0 ≤1+≤1 policy as a stated constraint, or pin the multi-dim grammar (product-of-extents ≤ MaxCTA with portions as factors; disjoint warp shards, Σ warps ≤ warps/CTA) and derive MALoC from it.
2. **Fix the objective** (C2/C3): reuse term = Σ bytes × (reuse − 1) (traffic avoided), normalized by a per-LNT reference if normalization is wanted; drop GoC or give it a shape model (NominalExtent) and a formula that responds to dim-splitting.
3. **Split the budgets** (C4): MaxCache (SMEM, kernel-wide) and MaxFragPerThread (per-thread, kernel-wide) as two gates; eviction may trigger on either.
4. **Specify the missing semantics** (H1/H3/H4): assignment slots = M window dims with constraint sums over expanded SoC instances; cross-LNT running floors in root order; pop-order policy + post-eviction scoring + the new error contract; inherit the tier rules and window-bytes model (H5) or revise them consciously.
5. **Fix the arithmetic and contracts** (H2/M2/M5): settle M (recommend 4, and say WindowK changes); tile presets as Params; LoC representation in the record (absolute counts vs portion indices).

## Implementation-section notes (for the executor when this doc is revised)

The module split maps onto the existing package without a rewrite:

- **(1) Iterator** — generalize `searchSchedule`'s odometer from a 3^K SoC vector to per-dim digits (SoC × TileSize × LoC), with incremental constraint pruning instead of post-filtering.
- **(2) Constraints + eviction** — extract from `evalSchedule`: LoC completion → the new per-dim preset/remainder logic; MAPM/MALoC/CuMAP checks over expanded instances; the two budget gates + the spill loop.
- **(3) Score** — `judgeSite` / `siteWindowBytes` / `siteReuseCount` are the cost model; keep them (plus the R2/C1b rules) and compute the score over the post-eviction site set.
- **(4)+(5) Update/return** — keep the existing determinism contract (fixed order, first-strict-max).
- **Config** — `Params` already is the "config as input with sane defaults" surface; extend it with the new knobs (MaxPartitionCta/Thr, LoC presets, α if it survives) and keep `Validate` fail-fast.

Invariants the rewrite must not silently drop: fixed enumeration order + first-strict-max (golden determinism); fail-closed on unknown iters/no-decision; kernel-wide floor threading; ≥1 CTA gate; tile ≥ 2 legality; the F2 floor; per-site gates (reuse ≥ 2, FRAG cap, CTA-covered refusal) as distinct from the aggregate budget.
