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

---

# Round 2 (2026-10-26) — re-review after revision

The revision addressed the four CRITICALs: absolute-count LoC replaces percentages (C1), reuse_score redefined over overlapping-dim tiles (C2), GoC clarified as partition counting (C3), single FRAG+SMEM pool defended (C4), M=4 settled (H2, partly). Verdict per point, then new findings the edit introduced.

## [1] Quota → absolute counts — CTA side RESOLVED; warp side regressed into a new ambiguity

**CTA: resolved and correct.** LoC = # CTAs per CTA dim, `MALoC.cta = Π LoC ≤ MaxLoC.cta`, and `gridDim.x = max over LNTs of Π` is exactly the product decomposition F4 prescribes ("CTA total per LNT = product over CTA dims"), with the smaller-LNT overflow handled by the existing whole-body-skip guards. The "percentage which is invalid" self-diagnosis is right. (But see N2/N3 — the bound is now vacuous and the presets are now the parallelism ceiling.)

**WARP: new CRITICAL.** `MALoC.thr = Mul all THR LoCs ≤ 32` silently switches the warp composition from the **disjoint-shard** model to a **tensor-product** warp space:

- Implemented model (F4 ruling + search.go/lower.go): independent WARP dims get *disjoint whole-warp shards* (a **sum**); the CTA dim's coupled inside walk takes the *remaining* warps (`coupledThreads = (PortionWarp − warpPortion) × warpsPerPortion × 32`, search.go:222); ThreadMap carries m+n tuple dims; an SMEM-class WARP dim's window is sized `LoC × 32 × T` (the shard's threads).
- The doc's Π model is a legitimate *alternative* (classic 2-D block decomposition: warp index factors as (w_a, w_b), each warp owns one tile pair) — but it is **not what the lower pass, the runtime ThreadMap, the R2 rank-in-index identity, and the window-bytes cost model implement**, and the structural section still says "partitioned by a partition of thread space" (sum language). The doc now contains both models at once.

**Pick one, per level:** (a) warps stay disjoint shards → `MALoC.thr = Σ LoC ≤ 32`, blockDim = (Σ warps) × 32, and everything downstream survives unchanged — **recommended**; or (b) warps go tensor-product → the TBL/TIL structural rules, coupled-walk ownership, window-bytes, fill mechanics, and ThreadMap must all be re-derived first (a codegen redesign, well beyond a search rewrite).

## [2] reuse_score — improved, but still not "actual reuse"

`reuse_size = Π(overlapped dims' tile_size)` captures tile volume (bigger overlapping tiles ⇒ more co-traversal captured) — a real improvement over occupied bytes. Three gaps remain:

1. **No re-traversal multiplier.** Benefit = cached volume × (times re-read − 1); the re-reads come from the inner walks of dims **not** in the covered set (the implemented reuse model: tie + inner non-covered dims' inside-walk trips). A compute dim covered by no site is invisible to the score: same tiles, 4 vs 1024 re-reads → identical score, ~256× different traffic avoided.
2. **Unit mismatch.** "Reuse bytes" currently has *element* units (no elem size) — fp8 vs fp16 sites incomparable — and dividing an element count by MaxSRAM (bytes) is dimensionally wrong.
3. **No −1.** A tile read once avoids nothing (the reuse ≥ 2 gate exists for exactly this).

**Recommendation unchanged:** `reuse_size := bytes × (reuse − 1)` — `judgeSite` already computes both factors. **Caveat if adopted:** Σ(bytes × (reuse−1)) is a *traffic* quantity and routinely exceeds MaxSRAM (the multiplier!), so `/MaxSRAM` normalization breaks (scores ≫ 1). Rank by raw traffic within the LNT, or normalize by the max seen during the search — not by the budget.

Also: "M ≥ 2 as hard constraints for findINS filter" is ambiguous — set **cardinality** ≥ 2 (the implemented singleton-set filter), overlap-**dim count** ≥ 2, or tensor rank ≥ CacheDims=2 (also implemented)? If it means dim count ≥ 2, note it would *exclude* the matmul K-overlap case (lhs/rhs overlap in exactly 1 dim) — the flagship.

## [3] GoC = partition count — intent ACCEPTED; the text must actually change

Counting partitions (dims) instead of LoC magnitudes fixes C3's core defect: 1 WARP dim (1/6) vs 2 WARP dims (2/6) now differ in the direction the 28×1024 argument wants, and the "at most 3 + 3" comment finally makes sense. Two fixes needed:

- The formula line still reads `(total_cta_loc + total_cta_thr)/goc_max` — under the new MALoC those names denote **products** (a lone CTA dim @ LoC 10 → 10/6 > 1). Write it as `(total_cta + total_thr)/goc_max` over the **same expanded-instance list MAPM counts** (a CTA dim contributes +1 CTA *and* +1 THR partition).
- `goc_max = max(3+3, 2 × #iterations)`: with M=4 the second arm yields 8 while only ≤6 partitions are constructible → the score caps at 0.75 forever. Use `goc_max = MaxPartition.cta + MaxPartition.thr = 6`.

Residual (accepted as heuristic): still shape-blind (runtime-dynamic extents — D12), so it rewards splitting whenever legal. That's fine as a tiebreaker; it's why α should probably favor reuse.

## [4] Single FRAG+SMEM pool — the L1 argument doesn't hold for registers; the pool needs one more gate

On sm_86, **registers are not L1**: the register file is a separate 256 KB SRAM per SM (64K × 32-bit, ≤ 255 regs/thread); L1 and SMEM share the 128 KB unified cache (≤ 100 KB SMEM opt-in). FRAG sites live in registers (`LoadTileThread`); they cost L1 only when they **spill** — at which point local-memory traffic (L1/L2-backed) makes them *worse than the GMEM load they replaced*, the exact regression F6 was ruled against.

Concrete unsoundness of the pool alone: `total_frag` is a sum of **per-thread** byte counts, so a schedule with total_frag = 60 KB + total_smem = 4 KB passes the 64 KB pool while needing 60 KB of registers **per thread** — 60× the ~1 KB architectural ceiling → guaranteed spill. The pool cannot see per-thread pressure because it mixes per-thread and per-CTA units.

**Resolution (author's call):** (a) two budgets — MaxCache (SMEM, per-CTA) + MaxFragPerThread (registers, per-thread), both already Params — cleanest, matches hardware; or (b) keep the single 64 KB pool as a policy simplification **plus a per-thread FRAG ceiling as a second gate** (256 B default, the existing knob). Pool-alone is not sound. ("Objective only accounts reuse" — agreed, and orthogonal: that's the objective; this is the constraint's unit problem.)

## [5] M = 4 — settled, but two stale lines contradict it

Delete "Notes for now it is 5" (Input) and "For now we clamp only 5 most nested loops" (Feasible Area). The arithmetic `(3·3·3)^4 = 531,441` is correct for M=4; note the WindowK default changes 5→4 in Params.

## New findings introduced by this revision

- **N1 — the −1 exhaust rule is dangling.** "The last CTA/THR LoC can be assigned −1, use up whatever left in runtime" presupposes the deleted percentage pool. Under absolute counts there is nothing "left" — −1 is undefined. Delete it or redefine (e.g., last dim per category takes ⌊policy_max / Π(others)⌋). The Objective's "all concurrency will be exhausted regardlessly" is the same stale quota language.
- **N2 — the presets are now the parallelism ceiling, and it dropped ~10×.** Old model: portions *exhausted* the space (CTA dim = full MaxCTA = 100 CTAs; warp dim = 16 warps = 512 threads at defaults). New model, presets as absolute counts with no functioning remainder rule: a CTA dim tops out at **10 CTAs** (vs 100), a warp dim at **8 warps = 256 threads** (vs 512). That is a large parallelism regression baked into the search space. Grow the presets (e.g., CTAs {2,5,10,25,50,100}, warps {1,2,4,8,16,32}) or restore an exhaust rule. Also decide whether the old runtime contract (threads/CTA a multiple of 8 warps, ≥ 256) survives — it was deleted, and blockDim = Π×32 is a multiple of 32 but no longer ≥ 256.
- **N3 — MaxLoC.cta = 2^31−1 is vacuous.** The old MaxCTA=100 policy bound is gone entirely; nothing real bounds the launch (see N2). MALoC.cta should carry an actual Params bound.
- **N4 — launcher section slips.** blockDim.x = "max of all LNT's total_cta_thr": (a) name — `total_cta_thr` → the MALoC-defined `total_thr_loc`; (b) unit — total_thr_loc is in **warps** under the new LoC, blockDim.x is **threads**, so × 32. (The gridDim.x max-over-LNTs + guard-bail shape is right and a genuinely good addition — kernel-level coherence the old doc lacked.)
- **N5 — MASRAM scope still unstated.** The launcher section proves kernel-level thinking, but per-LNT vs kernel-wide SMEM budget (the implemented running-floors model, R3) is still unsaid.

## Positive convergence worth noting

The new absolute-count LoC is *closer* to the published record's contract than the old portion-index proposal was (record LoC is already an absolute count). The remaining M2 delta narrows to: tiling-verify currently enforces CTA LoC == MaxCTA exactly (full exhaustion) — under multi-dim presets that check loosens to "LoC ∈ presets and Π == gridDim".

## Still open from Round 1 (untouched by this revision)

H1 (MAPM sums over the post-tiling list still not computable as written — count expanded instances), H4 (eviction: pop order, post-eviction scoring, error contract vs the 2026-10-24 tiling-never-disabled ruling), H5 (tier rules: CTA-covered refusal, C1b forced-FRAG, window-bytes `LoC×32×T` group factor), M1 (determinism/tie-break), M5 (tile presets vs current {32,64,128,256}), M6 (naming continuity), M7 (constraint 5 vacuous).

---

# Round 3 (2026-10-26) — the factorization ruling (author decision)

**Ruling:** BOTH levels partition by **factorization** — each WARP-SoC dim chooses K (warps), Π K ≤ 32; CTA already Π (round 2). Launch parameters fully dynamic ("no statically set one-size-fits-all").

## Assessment: ENDORSED — coherent, uniform, and more conventional than what it replaces — under one reading that must be confirmed

The clean reading is **exhaustive launch**: blockDim.x = Π K × 32 *exactly* — you launch precisely what the schedule factorizes. Then the model is watertight:

- Warp index w factors uniquely into coordinates (c_1..c_n) (reverse stride-shape); dim d's TBL distributes by residue class: warp c_d owns tiles ≡ c_d (mod K_d).
- A tile combination (t_1..t_n) is executed by exactly ONE warp — no redundancy, no idle warps, no co-ownership ambiguity. (K_d = 1 is vacuous distribution ≡ SEQ semantics — a dominated vector; prune it.)
- Π K ≤ 32 is pure hardware feasibility (1024 threads); this is literally how CUDA blocks/grids decompose — the shard model it replaces was the unusual design.
- **F2 dissolves**: no "remaining warps" concept exists, so the coupled-portion starvation hazard cannot arise. coupledThreads := blockDim (= Π K × 32).
- **Dynamic launch is already supported by the runtime model**: per-LNT map remapping (F5) + rank/extent guards let LNTs smaller than the launched max whole-body-skip; the reducer's warp identity is map-derived (`warp_index()`/`warp_count()` from ThreadMapping, runtime_objects.md #11) — survives a dynamic blockDim.

## The bill this ruling obligates (the honest cost — it outgrows "search rewrite")

1. **R2's reuse identity flips.** "All reuse is intra-thread; window-SMEM buys capacity, not sharing" was a *consequence of shards*. Under factorization, cross-warp sharing is real: a load whose subscripts miss warp dim d (e.g. B[k] with M also warp-distributed) is read by all W/K_d warps sharing the c_k coordinate — one fill serves them all. This makes the cost model *more accurate*, but every shard-derived formula is now stale: `siteReuseCount` (per-thread trips), `siteWindowBytes` (alloc = LoC×32), the C1b forced-FRAG rule, fill mechanics (LoadTileCta stripes).
2. **lower.go's loop-rewrite forms** (chunk-granularity strides over shard #threads) → residue-class ownership forms.
3. **tiling-verify + runtime contract**: "CTA LoC == MaxCTA" → "LoC ∈ presets ∧ Π == gridDim"; the assert-and-trap pre-condition (CTAs multiple of 10, threads ≥ 256 / multiple of 8 warps) is repealed — blockDim = Π K × 32 (any multiple of 32), gridDim = Π CTA LoC ≤ a real policy cap.
4. **The doc's structural section** ("partitioned by a partition of thread space regarding that CTA") must be rewritten in factorization terms — it still speaks shard language.
5. **Scope decision**: the search doc either carries the new bytes/reuse formulas itself, or explicitly points at loop_tile_lower.md / runtime_objects.md revisions. The searcher must NOT be implemented against judgeSite's shard-model numbers.

## N1 resolved concretely (recommendation)

Delete remainder-−1 — no quota pool exists. Replacement that fixes N2 + N3 in the same stroke: **−1 := absorb-the-slack** — the last dim per category takes ⌊bound / Π(others)⌋, with bound = 32 warps (warp category) / a real MaxCTA-style policy cap (CTA category). Effects: a lone warp dim at −1 launches 32 warps = 1024 threads (restores the old default's parallelism); a lone CTA dim at −1 launches the full policy grid; MaxLoC.cta stops being vacuous. Presets become pure factor choices: warps {2,4,8,16} + −1; CTAs {2,4,8,16,…} + −1 (drop 1 — dominated by SEQ).

## Decisions to confirm with the author

1. **Exhaustive reading** — blockDim = Π K × 32 exactly, launch-what-you-factorize? (The alternative — always launch 32 warps and let slack warp-coordinates bail — also works via guards but wastes residency and needs a co-ownership rule specified.)
2. **−1 semantics** — absorb-to-bound (recommended above), or delete entirely?
3. **The CTA policy cap** — gridDim.x needs a real bound (keep a MaxCTA-style Params knob; 2^31−1 is not a policy).

---

# Round 4 (2026-10-26) — factorization CONFIRMED; refactor authorized; final-decision checklist

**Author's rulings:** (1) warp allocation granularity = per warp (32 threads), Π LoC ≤ 32, blockDim = Π × 32 — the exhaustive launch reading is CONFIRMED; (2) −1/portions DELETED — precise allocations, launch config derived from the chosen schedule's settlement; (3) gridDim bound = the CUDA runtime's 2^31−1 or a more conservative Params value — actual gridDim = factorization of the CTA assignments per LNT (max over LNTs). The large refactor is AUTHORIZED ("current implementation is broken in terms of scheduling").

All three sound right. Two precision notes: (a) with −1 gone, the **preset lists are the parallelism ceiling** — a lone warp dim maxes at its largest preset (use 32 = 1024 threads if you want full blocks from single-dim schedules), a lone CTA dim at its largest CTA preset; (b) blockDim = max-over-LNTs of Π K × 32, with smaller LNTs' excess warps/CTAs whole-body-skipping via the per-LNT map + rank guards (the F5 mechanism) — this must be stated as the load-bearing cross-LNT rule.

## The definitive open-items checklist (what "crystal clear" still requires)

**A. Objective — 4 decisions**
- A1 [D] Reuse term := `bytes × (reuse − 1)` (traffic avoided; both factors already computed) — vs the doc's Π(overlapped T_d). Rec: traffic form.
- A2 [D] Normalization: none for within-LNT ranking (dividing traffic by MaxSRAM breaks — traffic ≫ budget). If GoC is kept, make the two terms scale-comparable per LNT or drop normalization.
- A3 [D] GoC: keep the partition-count form? If kept: `(total_cta + total_thr)/goc_max` over MAPM's expanded instances, `goc_max = 6`, delete the 2×N arm, fix the formula text (it still names the product variables). Rec: optional tiebreaker at α ≥ 0.75 favoring reuse, or drop.
- A4 [M] Determinism: fixed enumeration order, first strict max wins (keep the current INV; golden-test load-bearing).

**B. Cost model under factorization — 1 decision + derivations (the R2 bill, now in scope)**
- B1 [D] **Lane decomposition** within an owning warp's tile: 32 lanes split T elements — consecutive chunks (⌈T/32⌉ per lane, prefix masks match the existing lane-prefix contract) or strided (lane l takes ≡ l mod 32, matches the coupled-walk convention). Everything downstream (FRAG bytes, masks, reuse trips) keys off this. Rec: consecutive chunks.
- B2 [M] Per-thread FRAG bytes := ⌈T_d/32⌉ × elem × (other covered dims' factors) — a ~32× change from the shard model's per-thread T_d; all FRAG sizes re-derived.
- B3 [M] Window-bytes for SMEM sites under residue ownership: the site's group window = (warps sharing the relevant coordinate) × chunk — new formula replacing `alloc = LoC × 32`.
- B4 [M] Cross-warp reuse: loads whose subscripts miss a warp dim are read by all W/K_d warps sharing the coordinate — traffic formula counts cross-warp served reads.
- B5 [M] Tier rules re-derived: CTA-covered refusal (still valid?), C1b forced-FRAG (thread-varying coordinate → per-thread window), the new warp-shared-window class.
- B6 [D] The reuse ≥ 2 gate: per-thread trips or total-including-cross-warp? Rec: promote iff traffic avoided > 0 (total view).

**C. Constraints & presets — 5 decisions**
- C1 [D] Warp K presets: rec {2,4,8,16,32} (32 enables full-block single-dim schedules).
- C2 [D] CTA presets: rec {2,4,8,16,32,64} (or similar); single-dim ceiling = max preset.
- C3 [D] Tile presets: keep {32,64,128,256} (current) or the doc's {32,64,128}. Rec: keep 256 (rank-1 covered sites can afford it).
- C4 [D] **FRAG budget** (round-2 counterexample unanswered): the 64 KB pool alone admits 60 KB/thread registers → spill. Rec: keep MaxFragPerThread (256 B) as a hard per-thread gate alongside whatever pool you choose.
- C5 [D] gridDim policy cap: rec a Params knob with a real default (e.g., 512) rather than 2^31−1.
- C6 [M] Search-space arithmetic updates with the final preset sizes — e.g., (3 SoC × 4 tiles × 5 LoC)^4 ≈ 13M/LNT: fine only with incremental MAPM/MALoC pruning during enumeration + memoized per-site judgments; budget the preset lists accordingly.
- C7 [M] MAPM/MALoC rewritten over expanded instances (H1); constraint (5) deleted (M7); K=1/LoC=1 warp dims pruned (≡ SEQ, dominated); CuMAP kept.

**D. Eviction — 3 decisions (H4, untouched across all rounds)**
- D1 [D] Pop order: least traffic-value-per-byte first (rec), or smallest-bytes, or a real subset search (2^MaxCS = 65K worst case — affordable).
- D2 [M] Score computed on the post-eviction site set.
- D3 [D] Error contract: with eviction, every structurally-valid candidate is feasible; the worst case becomes loops-tiled-≥2-with-zero-caches. Rec: allow it silently BUT keep hard errors for structural infeasibility (no CTA dim, tile < 2) — this consciously extends the 2026-10-24 ruling; say so in the doc.

**E. Contracts — mechanical, but each is a work item (the refactor's real size)**
- E1 Record + verify: LoC absolute counts (now aligned); tiling-verify's `CTA LoC == MaxCTA` → `LoC ∈ presets ∧ Π == gridDim`; warp Π ≤ 32; F2 check DELETED; launch derivation (gridDim = max_LNT Π, blockDim = max_LNT Π × 32) — rec: recomputed by lower/codegen from per-iter decisions, no new record field.
- E2 lower.go: residue-class loop-rewrite forms; coupled CTA interior walk = blockDim-strided; fills per the re-derived tier rules.
- E3 Runtime: ThreadMapping warp coordinates from the factorization; assert-and-trap pre-conditions REPLACED (blockDim = Π K × 32, warp count ≤ 32); runtime_objects.md updated.
- E4 Pipeline position unchanged (tiling@114 → tiling-verify@115 → tile-lower); kernel-wide SMEM floors (R3) retained — still needs one sentence in the doc (N5).
- E5 Golden fixtures WILL change (new cost model → new chosen schedules): sanctioned regen + re-pin + e2e numerical verification on the A6000, per the fix3 C11 precedent.

**F. Doc mechanics** — delete the two stale "5" lines; rewrite the structural section in factorization terms; fix the GoC formula text + goc_max; launcher section names/units (blockDim = Π K × 32; `total_cta_thr` → warp-product name); typo pass; Params naming map (M6).

## Round 4 addendum — decision triage (author request)

Author will accept sane recommendations outright and wants only genuinely ambiguous items surfaced. Triage result:

**GO WITH RECOMMENDATION (no ruling needed):** A1 reuse term = bytes×(reuse−1) (the only form satisfying the author's stated "actual reuse" intent); A2 no budget normalization; A4 determinism = fixed order + first-strict-max; B1 lane decomposition = consecutive ⌈T/32⌉ chunks (per-lane contiguous windows, vec128-loader-compatible, fully coalesced at ≥8-elem lanes, prefix-mask contract intact — the strided alternative only wins for scalar loads); B2-B4 derived cost formulas (FRAG ⌈T/32⌉×elem×factors; residue-group window bytes; cross-warp served reads); B5 keep CTA-covered refusal for THIS refactor (revisit after the model lands); B6 gate = traffic avoided > 0 (total view, incl. cross-warp); C1 warp presets {2,4,8,16,32}; C2 CTA presets {2,4,8,16,32,64}; C3 tiles {32,64,128,256}; C5 gridDim Params cap default 512; C6 incremental pruning + memoized judgments; D1 eviction pop = least traffic-value-per-byte; D2 post-eviction scoring; GoC counts expanded instances like MAPM; E1-E5 contract items (verify re-checks, launch derivation in codegen, runtime contract replacement, kernel-wide floors, sanctioned golden regen); F doc mechanics.

**NEEDS AUTHOR RULING (3 items — genuine tradeoffs, no derivable answer):**
- R1. **FRAG per-thread ceiling (C4):** keep MaxFragPerThread = 256 B as a hard gate alongside the 64 KB pool? Author's "both cost L1" defense addressed: only SPILLED registers cost L1 (local memory); unspilled FRAG lives in the separate 256 KB register file. Without the ceiling, 60 KB/thread passes the pool and spills — the cache becomes slower than GMEM.
- R2. **GoC's authority (A3):** may granularity ever override real traffic differences? (a) tiebreak-only — traffic strictly primary, GoC breaks exact ties [rec: hypothetical utilization should never beat measurable bytes]; or (b) α-blend (α≈0.75) — but then the traffic term needs a defined per-LNT normalization for the blend to be meaningful.
- R3. **Zero-cache outcome (D3):** when the budget forces evicting ALL sites: (a) allow the tiled-uncached schedule + emit a warning [rec: the whole point of eviction is graceful degradation; erroring at zero makes eviction useless for tight budgets]; or (b) hard error, fail-loud (matches project culture + the 2026-10-24 ruling's spirit, changes observable compiler behavior).

---

# Round 5 (2026-10-26) — rulings locked; matmul inference on the objective

**Rulings:** R1 = yes (keep MaxFragPerThread=256B) **plus the FRAG definition**: a FRAG site = ALL covered dims SEQ-assigned → the window lives in a single thread's registers, walked wholly by that thread. Consequence (corrects Round-4 B2): **per-thread FRAG bytes = Π(covered T_d) × elem — no lane division** (the ⌈T/32⌉ lane split applies only to WARP-tile interiors, which are SMEM-class anyway). Min site = T₁·T₂·sizeof(dtype) per the author. Note: at 256 B, 2-D all-SEQ fragments (32×32 fp16 = 2 KB) decline — keep 256 B for v0 (single-dim fragments like matmul lhs = 128 B fit); the knob is exactly the hyperparameter-tunable surface the Implementation section wants. R2 = tiebreak-only (GoC never overrides traffic). R3 = allow zero-cache + warning.

## The inference: does traffic-objective + GoC tiebreak reach the canonical matmul schedule?

Model: C[M,N] = Σ_k A[M,K]·B[N,K], nest M→N→K, A refs (m,k), B refs (n,k), K reduced. findINS sites: A covered {k} tie N; B covered {n,k} tie M. Budgets SMEM 64 KB, FRAG/thread 256 B, Π warp LoC ≤ 32.

1. **M=CTA is FORCED, jointly by CuMAP + the tier rules.** All-warp vectors violate CuMAP. N=CTA ⇒ B's covered set contains a CTA dim ⇒ site REFUSED ⇒ B uncached ⇒ B re-read per m-row (O(M·N·K) traffic) — dominated. K=CTA ⇒ same for A. Only M=CTA leaves both sites cacheable (neither covers m). The structure falls out of the constraints — no heuristic needed.
2. **N=WARP beats N=SEQ on traffic, not just GoC.** N=SEQ ⇒ B all-SEQ-covered ⇒ FRAG at T_n·T_k·elem ≥ 2 KB > 256 B ⇒ declined ⇒ B uncached — dominated. N=WARP ⇒ B = SMEM window K_n·T_n·T_k·elem ≤ 64 KB, reuse = the CTA's T_m-row re-traversal of the window.
3. **Tiles: the objective's gradient = the classic arithmetic.** B contributes window·(T_m·rounds − 1), A contributes T_k·elem·(n-walk − 1) — both grow with tile span and re-traversal, so the search maximizes T_n·T_k until the 64 KB window binds (e.g. K_n=4, T_n=T_k=64 fp16 = 32 KB) and T_m to its max preset (T_m enters only through reuse — no budget covers m — so it maxes, which is genuinely traffic-optimal). This reproduces the canonical (T_m large, T_n·T_k budget-bound) shape.
4. **Result: the canonical schedule M=CTA / N=WARP / K=SEQ, max tiles — reached by constraint-forcing + traffic gradient, with GoC only reinforcing.** Notably the OLD model's emergent pick was M=CTA/N=SEQ/**K=WARP** with rhs declined (ReasonLowReuse — the shard model's intra-thread-only reuse couldn't see B's re-traversal); the factorization cost model's cross-warp sharing flips it to the classic shape. Evidence the new objective is not just different but better.

**The one caveat — the reduced dim K.** K=SEQ vs K=WARP (warp split-k, mechanically legal — warp-specialized reduction was built for it) can tie or flip on pure traffic depending on extents, and **GoC's partition-count tiebreak systematically prefers K=WARP** (4/6 vs 3/6 partitions). Split-k's reduction-staging cost (SMEM staging + barriers per output round) is unmodeled, so the tiebreak can pick split-k where classic K=SEQ is actually faster. Options: (a) accept split-k as a legitimate find (it IS optimal in some regimes; cost modeling is future NominalExtent-style work); (b) v0 policy "reduced dims default SEQ" — one line, the decide pass can see the reducer, removes the surprise from the golden regen. **Rec: (b) for v0, (a) when a reduction-cost term exists.** ← the single remaining question for the author.

1. **Doc finalization** — fold A/B/C/D answers into fix-opt-tile-search.md (+ the tier/cost formulas of B); author or a quick task.
2. **Decide-half rewrite** — cost model (B) + searcher (A/C/D) in internal/opt/tiling with unit tests; verify checks updated (E1).
3. **Lower-half rewrite** — residue-class forms + fill mechanics (E2) + runtime mapping contract (E3); goldens regen + re-pin (E5).
4. **E2E** — numerical acceptance + compute-sanitizer (memcheck/racecheck) on the A6000.
