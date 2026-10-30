---
phase: quick-261030-uai
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - testing/kernels/dedup/dup_expr.tca
  - testing/kernels/dedup/dup_const.tca
  - testing/kernels/fusion/axpy3.tca
  - testing/kernels/fusion/softmax.tca
  - testing/kernels/runtime/row_sum.tca
  - testing/kernels/runtime/scale_copy.tca
  - testing/runners/dup_expr.cu
  - testing/runners/dup_const.cu
  - testing/runners/axpy3.cu
  - testing/runners/softmax.cu
  - testing/runners/row_sum.cu
  - testing/runners/scale_copy.cu
  - design/fix/fix6.md
autonomous: true
requirements: [QUICK-261030-UAI]

must_haves:
  truths:
    - "All six in-scope suite kernels declare their reduced extent as a compile-time constant in the mixed dynamic/static form (row_sum/softmax use the const-MIDDLE form, not fully-static spellings)"
    - "The full GPU testing suite passes 14/14 with the rebuilt current-source bin/tcacomp"
    - "The eight unflipped kernels' emitted .cu are bit-identical (sha256) to the pre-flip manifest"
    - "The six flipped kernels' emitted .cu differ ONLY in the Launch signature (and const-extent propagation); their schedules are bit-identical modulo those signature lines per the 2026-10-30 probe"
    - "go build, go vet, go test all exit 0; module count 1"
  artifacts:
    - path: ".planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/before-sha256-manifest.txt"
      provides: "per-kernel sha256 of the emitted .cu for all 14 suite kernels on unmodified sources, built from current-source bin/tcacomp"
    - path: ".planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/after-sha256-manifest.txt"
      provides: "the same manifest after the const-K flip"
    - path: "design/fix/fix6.md"
      provides: "§2 updated with the const-shape-suite result (evidence line + gates outcome)"
  key_links:
    - from: "testing/runners/{dup_expr,dup_const,axpy3,softmax,row_sum,scale_copy}.cu"
      to: "generated Launch wrapper signatures"
      via: "prototype + call-site match against the ACTUAL emitted signature (D-fix6-02: prepended uint32_t e<extent> params)"
      pattern: "Launch\\(uint32_t e\\d+"
---

<objective>
Const-shape the six non-matmul testing/suite kernels per design/fix/fix6.md: re-declare the reduced extent as a compile-time constant in the mixed dynamic/static form, update the six runner .cu Launch prototypes/call sites to the new emitted signature, and validate that the full 14-kernel GPU suite stays green with schedules bit-identical to the pre-flip state.

Purpose: per fix6.md §2/§4, mixed const/static shapes are the legal vehicle for shape specialization today (fully-static is blocked by D-fix6-01 repeated-extents and D-fix6-03 cg-fill coordinate reconstruction). The 2026-10-30 probe proved the elementwise four emit bit-identical schedules under const-K; this task flips the suite so downstream fix6 work (per-shape specialization) has const-shaped fixtures. This is a pure enabler — no compiler change.
Output: six flipped .tca kernels, six updated runners, before/after sha256 manifests in the task dir, fix6.md §2 updated.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@design/fix/fix6.md
@testing/suite.toml
@testing/kernels/dedup/dup_expr.tca
@testing/kernels/dedup/dup_const.tca
@testing/kernels/fusion/axpy3.tca
@testing/kernels/fusion/softmax.tca
@testing/kernels/runtime/row_sum.tca
@testing/kernels/runtime/scale_copy.tca
@testing/orchestrator.py
</context>

<interfaces>
<!-- Current runner Launch faces (dynamic-extent form). After the flip each
     gains prepended `uint32_t e<extent>` params per D-fix6-02 — the exact
     spelling MUST be read from the emitted .cu, not assumed. -->

From testing/runners/scale_copy.cu (the others are isomorphic):
  void ScaleCopyLaunch(const tca::Tensor<float, 3u>& x, const tca::Tensor<float, 3u>& z,
                       ...scalars..., <out params>, unsigned seed /*or stream arg*/);
  call sites: ScaleCopyLaunch(xT, zT, outT, kGain, kBias, 0u);

D-fix6-02 example from fix6.md §4 (the expected NEW shape):
  ScaleCopyLaunch(uint32_t e4, ...)  // const dim params prepended
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Rebuild bin/tcacomp and record the BEFORE sha256 manifest</name>
  <files>bin/tcacomp, .planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/before-sha256-manifest.txt</files>
  <action>
    bin/tcacomp is a gitignored STALE prebuilt that shadows current source — rebuild it first: bash scripts/build.sh (pure Go, stdlib only, lands in bin/). Then run the emit-only path for ALL 14 suite kernels listed in testing/suite.toml to capture the pre-flip baseline: for each kernel path in suite.toml run bin/tcacomp compile -o <tmp>/<name> testing/kernels/<path>.tca, sha256sum the emitted <name>.cu, and write "sha256  <kernel-name>" lines (sorted by kernel name) to before-sha256-manifest.txt in the task dir. Use a fresh mktemp -d output root, one dir per kernel. Do NOT modify any kernel or runner file in this task — the manifest must reflect unmodified sources. Do not run the GPU suite here (emit-only is the sanctioned baseline per the task constraints; the suite runs once, after the flip, in Task 3).
  </action>
  <verify>
    <automated>test -s .planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/before-sha256-manifest.txt &amp;&amp; test $(grep -c '  ' .planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/before-sha256-manifest.txt) -eq 14 &amp;&amp; git diff --quiet -- testing/kernels testing/runners</automated>
  </verify>
  <done>bin/tcacomp rebuilt from current source; manifest exists with exactly 14 sha256 lines covering every suite.toml kernel; git diff on testing/ is empty (sources untouched).</done>
</task>

<task type="auto">
  <name>Task 2: Flip the six kernels to const-K mixed forms and match the runner Launch signatures</name>
  <files>testing/kernels/dedup/dup_expr.tca, testing/kernels/dedup/dup_const.tca, testing/kernels/fusion/axpy3.tca, testing/kernels/fusion/softmax.tca, testing/kernels/runtime/row_sum.tca, testing/kernels/runtime/scale_copy.tca, testing/runners/dup_expr.cu, testing/runners/dup_const.cu, testing/runners/axpy3.cu, testing/runners/softmax.cu, testing/runners/row_sum.cu, testing/runners/scale_copy.cu</files>
  <action>
    1. Flip the reduced extent to a compile-time const in each .tca kernel, preserving the tensor's rank and the position of the reduced axis:
       - dup_expr, dup_const: shape=[R, C, Ku] with Ku = 8u (trailing reduced).
       - axpy3, scale_copy: shape=[R, C, Ku] with Ku = 4u (trailing reduced).
       - row_sum, softmax: shape=[R, Ku, C] with Ku = 4u (row_sum) / 8u (softmax) — the const-MIDDLE mixed form. Do NOT use fully-static spellings: fix6.md §4 D-fix6-01 (repeated extents) and D-fix6-03 (cg fill coordinate reconstruction) block them; mixed dynamic/static is the proven-legal vehicle.
       Keep the dimvar declarations consistent with the new spelling (e.g. dimvar (i, j, k) unchanged; only the tensor shape lists change). Add a one-line comment in each kernel noting the const-K declaration and pointing at fix6.md.
    2. Read the ACTUAL new Launch signature before touching runners: compile ONE flipped kernel with bin/tcacomp compile -o <tmp> and read the emitted wrapper (D-fix6-02: prepended uint32_t e<extent> params, e.g. ScaleCopyLaunch(uint32_t e4, ...)). Do NOT assume the parameter order or count — the fix6.md §4 example is illustrative only. For row_sum/softmax (const middle dim) confirm which e-param spelling the emitter produces.
    3. Update the six runner .cu files: the Launch prototype AND every call site. Call sites pass the same baked constant the runner already uses for that dim (dK = 8 / 4 — the const extent value), preserving arg order per the actual emitted signature. Runner-baked dR/dC/dK constants stay unchanged (tensors keep the same runtime sizes).
    4. Do NOT touch testing/suite.toml: its dims fields are reporting mirrors of the runtime dims, which do not change with the const declaration (runner sources hold the load-bearing values). Record this no-op decision in the SUMMARY.
    5. Hermetic gates: go build ./..., go vet ./..., go test ./... all exit 0 (module count 1 — go list -m all | wc -l == 1).
  </action>
  <verify>
    <automated>go build ./... &amp;&amp; go vet ./... &amp;&amp; go test ./... &gt;/dev/null &amp;&amp; test $(go list -m all | wc -l) -eq 1</automated>
  </verify>
  <done>Six kernels declare the reduced extent as a compile-time const in the mixed form (const-middle for row_sum/softmax); runner prototypes and call sites match the actual emitted Launch signature (verified by compiling); go build/vet/test green; suite.toml untouched; unflipped eight kernels' sources untouched.</done>
</task>

<task type="auto">
  <name>Task 3: AFTER validation — GPU suite, manifest comparison, memcheck, fix6.md §2 update</name>
  <files>.planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/after-sha256-manifest.txt, design/fix/fix6.md</files>
  <action>
    1. Emit all 14 kernels again (same emit-only path as Task 1) into after-sha256-manifest.txt.
    2. Compare before/after per kernel:
       - The eight unflipped kernels MUST be bit-identical sha256. Any drift = RED FLAG, stop and investigate before proceeding.
       - The six flipped kernels WILL differ (the Launch signature change, and possibly const-extent propagation through the wrapper). This is EXPECTED — the 2026-10-30 probe (fix6.md §4) proved their schedules are bit-identical modulo the signature. The gate therefore is NOT raw sha256 equality but a schedule-identity diff: diff the before/after .cu for each flipped kernel and confirm the only differing lines are the Launch wrapper signature (e<extent> params) and const-extent propagation in the wrapper — no changes inside the kernel body, schedule, tile constants, or reducer setup. Drift beyond the signature/wrapper region = RED FLAG: stop, diagnose, do not weaken the gate.
    3. Full GPU suite: python3 -I testing/orchestrator.py --tcacomp ./bin/tcacomp --jobs 1 (correctness; single-job to avoid A6000 contention per the bench-jobs-1 protocol). All 14 must pass. Note the orchestrator's content-hash fixture cache keys on emitted source, so flipped kernels rebuild naturally; no --clean needed unless a stale cache is suspected.
    4. Memcheck: run python3 -I testing/orchestrator.py --tcacomp ./bin/tcacomp --sanitizer memcheck --jobs 1 --filter for each of the six flipped kernels (their .cu legitimately drifted). Must be clean. (The eight bit-identical kernels were memcheck-clean in prior phases and need no rerun.)
    5. Update design/fix/fix6.md §2: append the const-shape-suite result — the six kernels now compile as const-K mixed forms in the testing suite, before/after manifests recorded in the task dir, the eight unflipped bit-identical, the six flipped schedule-identical modulo the Launch signature (per the probe), suite 14/14 + memcheck clean. Also add one line to §4 noting the runner-prototype coupling (D-fix6-02) is now exercised by the suite. Do not reword §2's existing evidence; append.
  </action>
  <verify>
    <automated>test -s .planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/after-sha256-manifest.txt &amp;&amp; python3 -I -c "
import hashlib, pathlib, sys
task = pathlib.Path('.planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe')
def load(n):
    return dict(l.split()[::-1] for l in (task/n).read_text().splitlines() if l.strip())
before, after = load('before-sha256-manifest.txt'), load('after-sha256-manifest.txt')
flipped = {'dup_expr','dup_const','axpy3','softmax','row_sum','scale_copy'}
assert set(before) == set(after) == set(before.keys())
unflipped_same = all(before[k] == after[k] for k in before if k not in flipped)
print('unflipped bit-identical:', unflipped_same)
sys.exit(0 if unflipped_same else 1)
"</automated>
  </verify>
  <done>After manifest exists; the eight unflipped kernels' sha256 are unchanged (automated check); each flipped kernel's before/after .cu diff shows only Launch-signature/wrapper lines; GPU suite 14/14 at --jobs 1; memcheck clean on the six flipped kernels; fix6.md §2 updated.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| .tca kernel source → compiler | flipped const-extent declarations cross into the driver/build chain (dimKey, chain rungs, pkg reconciliation) — the exact paths D-fix6-01/03 broke on |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-uai-01 | Tampering | testing/kernels/*.tca edits | mitigate | flip is shape-spelling-only; before/after sha256 manifest + schedule-identity diff gate catches any unintended compiler-behavior drift; unflipped eight must stay bit-identical |
| T-uai-02 | Denial of Service | GPU suite runs | accept | local A6000, single-user; --jobs 1 prevents fixture contention (pinned protocol) |
</threat_model>

<verification>
- before/after manifests each cover all 14 kernels; unflipped eight bit-identical (automated).
- Flipped six: before/after .cu diff confined to Launch signature + const-extent propagation in the wrapper; kernel body/schedule/tile constants unchanged.
- go build/vet/test exit 0; go list -m all | wc -l == 1.
- python3 -I testing/orchestrator.py --tcacomp ./bin/tcacomp --jobs 1 → 14/14.
- --sanitizer memcheck clean on the six flipped kernels.
- design/fix/fix6.md §2 carries the appended result.
</verification>

<success_criteria>
The testing suite's six non-matmul kernels declare their reduced extent as a compile-time constant in the legal mixed dynamic/static form; the six runners match the actual emitted Launch signatures; the full GPU suite passes 14/14 with memcheck clean on the flipped kernels; the eight unflipped kernels' emitted .cu are bit-identical to the pre-flip manifest; the flipped six are schedule-identical modulo the Launch signature per the probe; fix6.md §2 records the result.
</success_criteria>

<output>
Create `.planning/quick/261030-uai-const-shape-the-testing-suite-kernels-pe/261030-uai-SUMMARY.md` when done, including the before/after manifest paths, the per-flipped-kernel diff attribution, the suite/memcheck results, and the suite.toml no-op decision.
</output>
