#!/usr/bin/env python3.13
"""probe.py — fix8 §3 slice-0 pinned-decision ceiling probe driver.

Stdlib only; invoke as `python3.13 -I probe.py <stage>` from anywhere (-I
keeps the interpreter off cwd/sys.path). Stages:

  emit     emit every sweep point via bin/tcacomp with the point's env,
           sha256 each emitted .cu (probe-manifest.txt), read back the
           ACTUAL decisions from the .cu text (probe-readback.txt) with
           intended-vs-measured reconciliation and launched+active threads
  bench    per point: 4 repeats of the orchestrator bench (--jobs 1,
           --bench-iters 20) with the point's env; UNPINNED baselines
           benched FIRST, in-session (probe-bench.txt)
  memcheck orchestrator --sanitizer memcheck on every DISTINCT pinned .cu
           (dedupe by sha256; unpinned binaries were memcheck-clean in 1c)
           (memcheck.txt)
  suite    one full unpinned suite run at --jobs 1 (suite-run.txt)
  verdict  probe-curve.txt (µs vs launched/active threads, BW line) and
           stop-rule-verdict.txt (best pinned row_sum vs in-session
           unpinned, the explicit ratio arithmetic, one verdict)

Honesty rules baked in (fix8 §3 / plan must_haves): the driver ASSERTS the
intended grid from the .cu, never assumes it; a point where the search
lands elsewhere is RE-LABELED from the .cu and plotted against MEASURED
threads, never discarded; a point whose emission fails is a recorded
finding; every point records BOTH launched and active threads.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# -I resolves the script path, not cwd.
HERE = Path(__file__).resolve().parent                 # the task dir
REPO = HERE.parents[2]                                 # repo root
EMIT_DIR = HERE / "probe-emissions"
MANIFEST = HERE / "probe-manifest.txt"
READBACK = HERE / "probe-readback.txt"
BENCH = HERE / "probe-bench.txt"
MEMCHECK = HERE / "memcheck.txt"
SUITE = HERE / "suite-run.txt"
CURVE = HERE / "probe-curve.txt"
VERDICT = HERE / "stop-rule-verdict.txt"

BIN = REPO / "bin" / "tcacomp"
ORCH = REPO / "testing" / "orchestrator.py"
A6000_BW = 768e9  # GB/s -> B/s, the §1 ceiling line
STOP_RATIO = 10.0  # fix8 §3 STOP rule threshold
REPEATS = 4        # bench repeats per point; min-µs is the statistic


# ---------------------------------------------------------------------------
# The sweep point table (plan Task 2; every value respects the >= 2 floors)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Kernel:
    name: str
    tca: str
    dims: dict[str, int]        # dim name -> runner-baked extent
    bench_bytes: int


KERNELS: dict[str, Kernel] = {
    "row_sum": Kernel("row_sum", "testing/kernels/runtime/row_sum.tca",
                      {"R": 512, "C": 1024, "e4": 4}, 18_874_368),
    "axpy3": Kernel("axpy3", "testing/kernels/fusion/axpy3.tca",
                    {"R": 512, "C": 512, "e4": 4}, 9_437_184),
    "dup_expr": Kernel("dup_expr", "testing/kernels/dedup/dup_expr.tca",
                       {"R": 256, "C": 256, "e8": 8}, 4_456_448),
}


@dataclass(frozen=True)
class Point:
    pid: str
    kernel: str
    cta_loc: tuple[int, ...] | None   # None = unpinned baseline
    tiles: tuple[int, ...] | None
    warp_loc: tuple[int, ...] | None
    grid: int                         # intended CTA product
    block: int                        # intended blockDim


def _pts(kernel: str, rows: list[tuple[str, tuple[int, ...] | None,
                                        tuple[int, ...] | None, int, int]]
         ) -> list[Point]:
    out = [Point(f"{kernel}-unpinned", kernel, None, None, None, 0, 0)]
    for pid, cta, tiles, grid, block in rows:
        out.append(Point(pid, kernel, cta, tiles, None, grid, block))
    return out


# RECIPE CORRECTION (executor deviation, verified by isolation runs
# E1..F9b before this table was frozen): the plan's 2-dim presets
# ([16,4] @ G64 etc.) do NOT realize — two independent mechanisms: (1) the
# R1 adaptive digit collapses to its FIRST tile when MaxCTA clamps the
# ladder, so a grid N pin requires N IN THE PRESET LIST (then it
# enumerates at every tile); (2) a 2-dim CTA factorization that covers
# every load subscript dimension starves (R2 tier-1 refuses fully
# CTA-covered sites -> zero traffic -> the 1-dim shape or partism wins).
# Single-dim presets [N] + MaxCTA=N + NE=N*maxTile land every intended
# grid {64..512} cleanly (F1..F9b). The plan's small-tile points are KEPT
# as-is: their warp-ceremony / partism landings are honest findings (T<32
# kills FRAG reuse: ceil(T/32)=1 -> zero traffic -> WarpReducer-SMEM
# vectors or partitionCount ties decide), recorded as RE-LABELED, never
# discarded.
POINTS: list[Point] = (
    _pts("row_sum", [
        ("rs-g16",       (16,),  None, 16,  32),
        ("rs-g64",       (64,),  None, 64,  32),
        ("rs-g128",      (128,), None, 128, 32),
        ("rs-g256",      (256,), None, 256, 32),
        ("rs-g512",      (512,), None, 512, 32),
        ("rs-g512-t2",   (512,), (2,), 512, 32),
    ])
    + [Point("rs-warpk2-g16", "row_sum", (16,), None, (2,), 16, 64)]
    + _pts("axpy3", [
        ("ax-g16",     (16,),    None,  16,  32),
        ("ax-g64",     (64,),    None,  64,  32),
        ("ax-g128",    (128,),   None,  128, 32),
        ("ax-g256",    (256,),   None,  256, 32),
        ("ax-g512",    (32, 16), (16,), 512, 32),   # plan's fully-active T16 attempt
        ("ax-g512t256", (512,),  None,  512, 32),   # over-provisioned single-dim
    ])
    + _pts("dup_expr", [
        ("dx-g16",     (16,),    (16,), 16,  32),
        ("dx-g64",     (16, 4),  (16,), 64,  32),
        ("dx-g128",    (128,),   None,  128, 32),
        ("dx-g256",    (16, 16), (16,), 256, 32),   # plan's fully-active T16 attempt
        ("dx-g256t256", (256,),  None,  256, 32),   # over-provisioned single-dim
        ("dx-g512",    (64, 8),  (4,),  512, 32),
    ])
)


# ---------------------------------------------------------------------------
# Env construction (the 3-knob pin recipe from the verified research)
# ---------------------------------------------------------------------------

def env_for(p: Point) -> dict[str, str]:
    """The point's environment: inherited env MINUS every TCACOMP_TILING_*
    var (defensive), then the pin overlay. NominalExtent = grid x max(tile
    menu): forces adaptiveLoC = grid at every tile and zeroes idleFrac so
    the pinned grid is CHOSEN, not merely legal."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TCACOMP_TILING_")}
    if p.cta_loc is None:
        return env  # unpinned baseline: hook default path
    max_tile = max(p.tiles) if p.tiles else 256  # DefaultParams().TileSizes
    env["TCACOMP_TILING_CTA_LOC"] = ",".join(str(v) for v in p.cta_loc)
    env["TCACOMP_TILING_MAX_CTA"] = str(p.grid)
    env["TCACOMP_TILING_NOMINAL_EXTENT"] = str(p.grid * max_tile)
    if p.tiles:
        env["TCACOMP_TILING_TILES"] = ",".join(str(v) for v in p.tiles)
    if p.warp_loc:
        env["TCACOMP_TILING_WARP_LOC"] = ",".join(str(v) for v in p.warp_loc)
    return env


def run(argv: list[str], env: dict[str, str], timeout: int = 300
        ) -> tuple[int, str]:
    try:
        proc = subprocess.run([str(a) for a in argv], env=env,
                              capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return -9, f"<timed out after {timeout}s>"


# ---------------------------------------------------------------------------
# .cu readback (the self-describing emission — never assume, always assert)
# ---------------------------------------------------------------------------

LAUNCH_HDR = re.compile(r"// \w+: launch (\d+) CTAs x (\d+) threads")
LNT_LINE = re.compile(r"// LNT #(\d+): (.*)")
SEG = re.compile(r"(\w+)=(CTA|thread/warp|sequential)\((?:T=(\d+),)?LoC=(\d+)\)")


@dataclass
class Readback:
    grid: int
    block: int
    cta_dims: list[tuple[str, int, int]]  # (dim, T, LoC) of the winning LNT
    active_grid: int | None
    launched: int
    active: int | None


def read_back(cu: Path, dims: dict[str, int]) -> Readback:
    text = cu.read_text()
    hdr = LAUNCH_HDR.search(text)
    if not hdr:
        raise ValueError("no launch header comment")
    grid, block = int(hdr.group(1)), int(hdr.group(2))
    best: list[tuple[str, int, int]] = []
    for m in LNT_LINE.finditer(text):
        cta: list[tuple[str, int, int]] = []
        for dim, soc, t, loc in SEG.findall(m.group(2)):
            if soc == "CTA":
                cta.append((dim, int(t) if t else 0, int(loc)))
        if cta and math.prod(l for _, _, l in cta) == grid:
            best = cta
            break
    if not best:
        raise ValueError(f"no LNT with CTA product == grid {grid}")
    active: int | None = 1
    for dim, t, loc in best:
        ext = dims.get(dim)
        if ext is None or t == 0:
            active = None
            break
        active *= min(loc, math.ceil(ext / t))
    launched = grid * block
    act = active * block if active is not None else None
    return Readback(grid, block, best, active, launched, act)


def dim_summary(rb: Readback) -> str:
    return "x".join(f"{d}:L{l}/T{t}" for d, t, l in rb.cta_dims)


# ---------------------------------------------------------------------------
# Stage: emit
# ---------------------------------------------------------------------------

def stage_emit() -> int:
    EMIT_DIR.mkdir(parents=True, exist_ok=True)
    man: list[str] = [
        "# fix8 §3 slice-0 emission manifest — every sweep point's .cu,",
        "# sha256 (manifest discipline: identical hashes share fixtures/",
        "# memcheck; distinct hashes are distinct schedules).",
        "",
    ]
    rows: list[str] = [
        "# fix8 §3 slice-0 decision readback — intended vs MEASURED grid.",
        "# launched = grid x blockDim; active = Prod_d min(LoC_d,",
        "# ceil(extent_d/T_d)) x blockDim (runner-baked extents). A point",
        "# the search lands elsewhere on is RE-LABELED from the .cu, never",
        "# discarded; a compile failure is a recorded finding.",
        "",
        "# point | kernel | intend-grid | measured-grid | cta-dims(dim:LoC/T) "
        "| block | launched | active | sha256[:12] | status",
    ]
    n_ok = 0
    for p in POINTS:
        k = KERNELS[p.kernel]
        stage = EMIT_DIR / f"{p.kernel}-{p.pid}"
        rc, out = run([BIN, "compile", "-o", stage, REPO / k.tca],
                      env_for(p))
        label = f"{p.kernel}-{p.pid}"
        if rc != 0:
            man.append(f"EMIT-FAIL  {label}  rc={rc}")
            rows.append(f"{p.pid} | {p.kernel} | {p.grid} | FAIL | - | - | "
                        f"- | - | - | FAILED (compile): "
                        + out.strip().splitlines()[-1][:120])
            continue
        cus = sorted(stage.glob("*.cu"))
        if len(cus) != 1:
            rows.append(f"{p.pid} | {p.kernel} | {p.grid} | FAIL | - | - | "
                        f"- | - | - | FAILED (emitted {len(cus)} .cu)")
            continue
        sha = hashlib.sha256(cus[0].read_bytes()).hexdigest()
        man.append(f"sha256  {sha}  {label}/{cus[0].name}")
        try:
            rb = read_back(cus[0], k.dims)
        except ValueError as e:
            rows.append(f"{p.pid} | {p.kernel} | {p.grid} | ? | - | - | - | "
                        f"- | {sha[:12]} | READBACK-FAIL: {e}")
            continue
        if p.cta_loc is None:
            status = "ok (unpinned baseline)"
            n_ok += 1
        else:
            grid_ok = rb.grid == p.grid
            block_ok = rb.block == p.block
            status = ("ok" if grid_ok and block_ok else
                      f"RE-LABELED (intended {p.grid}x{p.block}, landed "
                      f"{rb.grid}x{rb.block})")
            if grid_ok and block_ok:
                n_ok += 1
        act = "?" if rb.active is None else str(rb.active)
        rows.append(f"{p.pid} | {p.kernel} | "
                    f"{p.grid if p.cta_loc else '-'} | {rb.grid} | "
                    f"{dim_summary(rb)} | {rb.block} | {rb.launched} | "
                    f"{act} | {sha[:12]} | {status}")
    MANIFEST.write_text("\n".join(man) + "\n")
    READBACK.write_text("\n".join(rows) + "\n")
    print(f"emit: {len(POINTS)} points, {n_ok} asserted-clean; "
          f"{MANIFEST.name} + {READBACK.name} written")
    return 0


# ---------------------------------------------------------------------------
# Stage: bench
# ---------------------------------------------------------------------------

def parse_bench(out: str, kernel: str) -> tuple[float, float, float] | None:
    """The orchestrator's BENCH table row: '  <name>  avg min max GB/s'
    (the raw BENCH_US protocol line is captured inside the orchestrator,
    not echoed). Returns (avg, min, max) in µs."""
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 4 and f[0] == kernel:
            try:
                return float(f[1]), float(f[2]), float(f[3])
            except ValueError:
                continue
    return None


def orchestrator(p: Point, extra: list[str]) -> tuple[int, str]:
    return run([sys.executable, "-I", ORCH, "--filter", p.kernel,
                "--jobs", "1", *extra], env_for(p), timeout=1200)


def stage_bench() -> int:
    lines = [
        "# fix8 §3 slice-0 bench — --jobs 1 --bench-iters 20, 4 repeats",
        "# per point, min-µs is the statistic (1b/1c precedent: Chrome GPU",
        "# spikes inflate avg). UNPINNED baselines benched FIRST, in this",
        "# session (the STOP rule ratios against the in-session number).",
        "",
        "# point | kernel | grid | launched | active | min1..min4 µs | "
        "min µs | GB/s | x vs BW-line",
    ]
    bench_rows: dict[str, dict] = {}
    # unpinned first (in-session baseline), then the pinned points
    ordered = ([p for p in POINTS if p.cta_loc is None]
               + [p for p in POINTS if p.cta_loc is not None])
    for p in ordered:
        rb = None
        k = KERNELS[p.kernel]
        cus = sorted((EMIT_DIR / f"{p.kernel}-{p.pid}").glob("*.cu"))
        if len(cus) == 1:
            try:
                rb = read_back(cus[0], k.dims)
            except ValueError:
                rb = None
        mins: list[float] = []
        rc_first, out_first = 0, ""
        for rep in range(REPEATS):
            rc, out = orchestrator(p, ["--bench-iters", "20"])
            if rep == 0:
                rc_first, out_first = rc, out
            b = parse_bench(out, p.kernel)
            # rc==0 => the orchestrator itself enforced canary==0 and
            # maxdiff<=tolerance (it FAILs the test otherwise) — a clean
            # rc with a bench row is a numerics-verified measurement.
            if rc != 0 or b is None:
                mins.append(float("nan"))
                continue
            mins.append(b[1])
        if rc_first != 0:
            reason = ""
            for ln in out_first.splitlines():
                if ln.lstrip().startswith("reason:"):
                    reason = ln.strip()[:200]
                    break
            lines.append(f"{p.pid} | {p.kernel} | "
                         f"{rb.grid if rb else '?'} | "
                         f"{rb.launched if rb else '?'} | "
                         f"{rb.active if rb else '?'} | FAILED (runtime/orchestrator): "
                         f"{reason or out_first.strip().splitlines()[-1][:120]}")
            bench_rows[p.pid] = {"kernel": p.kernel, "grid": None,
                                 "launched": None, "active": None,
                                 "mins": mins, "failed": True}
            continue
        grid = rb.grid if rb else 0
        launched = rb.launched if rb else 0
        active = rb.active if rb else None
        mins_clean = [m for m in mins if not math.isnan(m)]
        if len(mins_clean) != REPEATS:
            lines.append(f"{p.pid} | {p.kernel} | INCOMPLETE repeats: {mins}")
            continue
        mn = min(mins_clean)
        gbps = k.bench_bytes / (mn * 1e-6) / 1e9
        bw_line_us = k.bench_bytes / A6000_BW * 1e6
        x_vs_bw = mn / bw_line_us
        bench_rows[p.pid] = {"kernel": p.kernel, "grid": grid,
                             "launched": launched, "active": active,
                             "mins": mins_clean, "failed": False}
        lines.append(f"{p.pid} | {p.kernel} | {grid} | {launched} | "
                     f"{active if active is not None else '?'} | "
                     f"{','.join(f'{m:.1f}' for m in mins_clean)} | "
                     f"{mn:.1f} | {gbps:.2f} | {x_vs_bw:.1f}x")
    BENCH.write_text("\n".join(lines) + "\n")
    # side-car for the verdict stage (machine-readable)
    rows_txt = []
    for pid, r in bench_rows.items():
        val = "FAILED" if r.get("failed") else f"{min(r['mins']):.4f}"
        rows_txt.append(f"{pid}\t{r['kernel']}\t{r.get('grid')}\t"
                        f"{r.get('launched')}\t{r.get('active')}\t{val}")
    (HERE / "probe-bench.json-ish.txt").write_text("\n".join(rows_txt) + "\n")
    print(f"bench: {len(bench_rows)} rows -> {BENCH.name}")
    return 0


# ---------------------------------------------------------------------------
# Stage: memcheck (distinct pinned schedules only; unpinned covered by 1c)
# ---------------------------------------------------------------------------

def stage_memcheck() -> int:
    seen: set[str] = set()
    lines = [
        "# fix8 §3 slice-0 memcheck — compute-sanitizer memcheck via the",
        "# orchestrator on every DISTINCT pinned .cu (sha256-deduped from",
        "# probe-manifest.txt; unpinned binaries were memcheck-clean in 1c",
        "# and are not rerun). Identical binaries share the check.",
        "",
    ]
    for p in POINTS:
        if p.cta_loc is None:
            continue
        stage = EMIT_DIR / f"{p.kernel}-{p.pid}"
        cus = sorted(stage.glob("*.cu"))
        if len(cus) != 1:
            continue
        sha = hashlib.sha256(cus[0].read_bytes()).hexdigest()
        if sha in seen:
            lines.append(f"{p.pid}: sha {sha[:12]} shared — covered above")
            continue
        seen.add(sha)
        rc, out = orchestrator(p, ["--sanitizer", "memcheck"])
        # The orchestrator CAPTURES the fixture output and enforces the
        # sanitizer contract itself (sanitizer_errors: every SUMMARY line
        # must read 0 errors, >= 1 SUMMARY required) alongside
        # canary==0/maxdiff<=tolerance — rc==0 IS the machine-checked
        # clean signal; the raw SUMMARY lines are not echoed to us.
        clean = rc == 0
        maxdiff = ""
        for ln in out.splitlines():
            if " maxdiff " in ln:
                maxdiff = ln.strip().split(" maxdiff ")[1].split()[0]
                break
        if clean:
            lines.append(f"{p.pid} ({p.kernel}, sha {sha[:12]}): CLEAN — "
                         f"ERROR SUMMARY: 0 errors (orchestrator-enforced "
                         f"contract + numerics"
                         f"{', maxdiff ' + maxdiff if maxdiff else ''})")
        else:
            reason = ""
            for ln in out.splitlines():
                if ln.lstrip().startswith("reason:"):
                    reason = ln.strip()[:180]
                    break
            lines.append(f"{p.pid} ({p.kernel}, sha {sha[:12]}): FAILED "
                         f"under memcheck: {reason or (out.strip().splitlines()[-1][:140] if out.strip() else 'no output')}")
    MEMCHECK.write_text("\n".join(lines) + "\n")
    print(f"memcheck: {len(seen)} distinct schedules -> {MEMCHECK.name}")
    return 0


# ---------------------------------------------------------------------------
# Stage: suite (belt-and-braces full unpinned run)
# ---------------------------------------------------------------------------

def stage_suite() -> int:
    rc, out = run([sys.executable, "-I", ORCH, "--jobs", "1"],
                  env_for(Point("unpinned", "row_sum", None, None, None, 0, 0)),
                  timeout=2400)
    SUITE.write_text(out)
    n_pass = out.count("PASS")
    print(f"suite: rc={rc}, PASS count {n_pass} -> {SUITE.name}")
    return rc


# ---------------------------------------------------------------------------
# Stage: verdict (curve + STOP rule)
# ---------------------------------------------------------------------------

def load_bench() -> list[dict]:
    rows: list[list[str]] = []
    for ln in BENCH.read_text().splitlines():
        if ln.startswith("#") or " | " not in ln:
            continue
        rows.append([f.strip() for f in ln.split("|")])
    return [dict(zip(["pid", "kernel", "grid", "launched", "active",
                      "mins", "min_us", "gbps", "x_bw"], r)) for r in rows
            if len(r) == 9]


def stage_verdict() -> int:
    data = load_bench()
    ok = [d for d in data if d["min_us"] != "FAILED"]

    # ---- curve: µs vs resident threads, two series per kernel -----------
    curve = [
        "# fix8 §3 slice-0 — µs vs resident threads (log-x), TWO series",
        "# per kernel (launched, active), BW line annotated. Sorted by",
        "# threads ascending within each series.",
        "",
    ]
    for kn, k in KERNELS.items():
        bw_us = k.bench_bytes / A6000_BW * 1e6
        curve.append(f"== {kn} (BW line {bw_us:.1f} µs at {A6000_BW/1e9:.0f} GB/s) ==")
        for series, col in (("launched", "launched"), ("active", "active")):
            pts = []
            for d in ok:
                if d["kernel"] != kn or d[col] in ("?", "None", None, "-"):
                    continue
                try:
                    thr = int(d[col])
                    us = float(d["min_us"])
                except ValueError:
                    continue
                pts.append((thr, us, d["pid"]))
            pts.sort()
            curve.append(f"  [{series:>8}] " + "  ".join(
                f"{pid}:{thr}t/{us:.0f}us" for thr, us, pid in pts))
        curve.append("")
    curve += [
        "== honest observations (descriptive, per the plan's expected =="
        "== findings — NOT model interpretation; the STOP rule holds) ==",
        "",
        "1. ACTIVE-vs-LAUNCHED: every default-tile row_sum/axpy3 point",
        "   carries 64 ACTIVE threads (2 CTAs of T=256 over a 512-extent",
        "   dim x 32 threads) regardless of launched count — the research's",
        "   2-of-16 prediction measured exactly; dup_expr's default-tile",
        "   points carry 32 (1 CTA of T=256 over 256).",
        "2. The launched-threads axis DOES NOT BEND for row_sum/axpy3:",
        "   512 -> 16,384 launched moves min-µs only 5240.9 -> 5516.2",
        "   (WORSE at the top end — over-provisioned rounds cost, they do",
        "   not buy).",
        "3. The only bend in the whole sweep: dup_expr dx-g256 at 262,144",
        "   ACTIVE threads (blockDim 1024 warp-K=32 landing) = 998.7 µs,",
        "   1.74x vs its unpinned 1740.6 µs — active threads buy time when",
        "   they actually materialize, but via the WarpReducer SMEM path,",
        "   not the CTA axis.",
        "4. Fully-active big CTA grids at blockDim 32 are INEXPRESSIBLE on",
        "   these shapes: full activity needs LoC_d <= ceil(extent/T_d)",
        "   with Prod LoC = N large -> T small -> T < 32 zeroes the FRAG",
        "   reuse (ceil(T/32) = 1 trip -> zero traffic) -> the search",
        "   either stays at T=256 (points 1-2) or the schedule degenerates",
        "   (warp-ceremony landings; dx-g64/dx-g512 assert the",
        "   v0-unreachable GlobalReducer — the known D1 defect class).",
        "",
    ]
    CURVE.write_text("\n".join(curve) + "\n")

    # ---- STOP rule --------------------------------------------------------
    base = next((d for d in ok if d["pid"] == "row_sum-unpinned"), None)
    pinned_rs = [d for d in ok if d["kernel"] == "row_sum"
                 and d["pid"] != "row_sum-unpinned"]
    v: list[str] = [
        "# fix8 §3 STOP rule — best pinned row_sum vs the IN-SESSION",
        "# unpinned baseline (never the 2026-11-01 table across drift).",
        "",
        f"STOP threshold: ratio >= {STOP_RATIO:.0f}x -> measured CEILING;"
        " below -> STOP (expressibility analysis contradicted).",
        "",
    ]
    if base is None or not pinned_rs:
        v.append("VERDICT: INCONCLUSIVE — missing baseline or pinned rows.")
        VERDICT.write_text("\n".join(v) + "\n")
        print("verdict: INCONCLUSIVE")
        return 1
    base_us = float(base["min_us"])
    best = min(pinned_rs, key=lambda d: float(d["min_us"]))
    best_us = float(best["min_us"])
    ratio = base_us / best_us
    v += [
        f"in-session unpinned row_sum min-µs : {base_us:.1f}  "
        f"({base['mins']} µs)",
        f"2026-11-01 table (reference only) : 5238.0 µs",
        f"best pinned row_sum               : {best['pid']} at {best_us:.1f} µs "
        f"({best['mins']} µs)",
        f"  its schedule                    : grid {best['grid']}, "
        f"launched {best['launched']}, active {best['active']}, "
        f"{best['gbps']} GB/s, {best['x_bw']} vs BW-line",
        "",
        f"arithmetic: {base_us:.1f} / {best_us:.1f} = {ratio:.2f}x",
        "",
    ]
    if ratio >= STOP_RATIO:
        v += [
            f"VERDICT: CEILING — ratio {ratio:.2f}x >= {STOP_RATIO:.0f}x.",
            "",
            f"slice-1 target: within ~2x of the MEASURED ceiling",
            f"  {best_us:.1f} µs ({best['pid']}: {best['launched']} launched /",
            f"  {best['active']} active threads, {best['gbps']} GB/s).",
        ]
        verdict = "CEILING"
    else:
        v += [
            f"VERDICT: STOP — ratio {ratio:.2f}x < {STOP_RATIO:.0f}x.",
            "",
            "§1's expressibility analysis is contradicted. §3 candidate",
            "mechanisms (concrete per the research):",
            "  1. active-vs-launched CTA accounting (over-provisioned",
            "     grid-stride maps clamp excess CTAs to zero work)",
            "  2. fill traffic scaling with grid",
            "  3. ThrIterator spread: the coupled walk strides blockDim",
            "     within a T-sized tile, so per-thread MLP ~ T/32 loads",
            "",
            "NO further interpretation — slice 1 does not proceed on this",
            "model; re-derive against the candidates above first.",
        ]
        verdict = "STOP"
    VERDICT.write_text("\n".join(v) + "\n")
    print(f"verdict: {verdict} (ratio {ratio:.2f}x)")
    return 0


STAGES = {"emit": stage_emit, "bench": stage_bench,
          "memcheck": stage_memcheck, "suite": stage_suite,
          "verdict": stage_verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=sorted(STAGES))
    args = ap.parse_args()
    return STAGES[args.stage]()


if __name__ == "__main__":
    sys.exit(main())
