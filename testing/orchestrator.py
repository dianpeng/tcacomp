#!/usr/bin/env python3.13
"""orchestrator.py — the tcacomp testing suite driver (stdlib only).

The user-facing validation harness: for every registered kernel it compiles
via the tcacomp CLI, builds the fixture through CMake (content-hash cached),
runs the binary on the GPU, and verifies canary == 0 AND maxdiff <= the
per-test tolerance against the in-binary CPU reference. Benchmark
(--bench-iters) and compute-sanitizer (--sanitizer) modes ride the same
pipeline.

Stdlib only (argparse, concurrent.futures, hashlib, os, re, shutil,
subprocess, sys, tempfile, time, tomllib, pathlib, statistics) — zero pip
installs, zero network, fixed argv lists everywhere, NEVER shell=True.
"""

import argparse
import concurrent.futures
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent          # testing/
REPO_ROOT = Path(__file__).resolve().parents[1]  # the repo root (any cwd)
SUITE_TOML = HERE / "suite.toml"
CMAKE_TEMPLATE = HERE / "cmake_template.txt"
CACHE_ROOT = Path(tempfile.gettempdir()) / "tcacomp-suite"
TEMPLATE_VERSION = b"tcacomp-suite-fixture-v1"  # bump on template-shape drift
TIMEOUT = 300  # seconds, every subprocess (mirrors internal/e2e e2eTimeout)

TOOL_FALLBACKS = {
    "cmake": ("/home/ps/.local/bin/cmake",),
    "nvcc": ("/usr/local/cuda-12.2/bin/nvcc",),
    "compute-sanitizer": ("/usr/local/cuda-12.2/bin/compute-sanitizer",),
}

# --------------------------------------------------------------------------
# ANSI reporting (only on a tty and with NO_COLOR unset)
# --------------------------------------------------------------------------

class Color:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def green(self, t: str) -> str: return self.wrap("32", t)
    def red(self, t: str) -> str: return self.wrap("31", t)
    def yellow(self, t: str) -> str: return self.wrap("33", t)
    def cyan(self, t: str) -> str: return self.wrap("36", t)
    def bold(self, t: str) -> str: return self.wrap("1", t)


# --------------------------------------------------------------------------
# Toolchain probing (mirrors internal/e2e/harness.go FindTool)
# --------------------------------------------------------------------------

def find_tool(name: str) -> str | None:
    """PATH first, then the known absolute fallbacks."""
    found = shutil.which(name)
    if found:
        return found
    for fallback in TOOL_FALLBACKS.get(name, ()):
        p = Path(fallback)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def probe_compiler(args: argparse.Namespace) -> list[str]:
    """The compiler: --tcacomp override, else repo-root bin/tcacomp, else the
    [go, run, ./cmd/tcacomp] argv prefix."""
    if args.tcacomp:
        return [args.tcacomp]
    prebuilt = REPO_ROOT / "bin" / "tcacomp"
    if prebuilt.is_file() and os.access(prebuilt, os.X_OK):
        return [str(prebuilt)]
    return ["go", "run", "./cmd/tcacomp"]


def run_argv(argv: list[str], timeout: int = TIMEOUT) -> tuple[int, str]:
    """Fixed argv list, never a shell. Returns (rc, combined output)."""
    if SHOW_ARGV:
        print(f"    $ {' '.join(str(a) for a in argv)}", file=sys.stderr)
    try:
        proc = subprocess.run(
            [str(a) for a in argv],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        return -9, f"<timed out after {timeout}s>"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

def load_registry() -> list[dict]:
    with open(SUITE_TOML, "rb") as f:
        data = tomllib.load(f)
    return data.get("test", [])


def select_tests(tests: list[dict], args: argparse.Namespace) -> list[dict]:
    out = []
    for t in tests:
        if args.category and t.get("category") != args.category:
            continue
        if args.filter and args.filter not in t.get("name", ""):
            continue
        out.append(t)
    return out


_NAME_SAFE = re.compile(r"[^A-Za-z0-9_]")


def sanitize_name(name: str) -> str:
    safe = _NAME_SAFE.sub("_", name)
    if not safe or safe[0].isdigit():
        safe = "t_" + safe
    return safe


# --------------------------------------------------------------------------
# Protocol parsing
# --------------------------------------------------------------------------

def parse_protocol(out: str) -> dict:
    """Extract CANARY/CHECKSUM/REF_CHECKSUM/MAXDIFF/SAMPLES/BENCH_US."""
    proto = {}
    for line in out.splitlines():
        fields = line.split()
        if not fields:
            continue
        tag = fields[0]
        try:
            if tag == "CANARY" and len(fields) >= 2:
                proto["canary"] = int(fields[1])
            elif tag == "CHECKSUM" and len(fields) >= 2:
                proto["checksum"] = float(fields[1])
            elif tag == "REF_CHECKSUM" and len(fields) >= 2:
                proto["ref_checksum"] = float(fields[1])
            elif tag == "MAXDIFF" and len(fields) >= 2:
                proto["maxdiff"] = float(fields[1])
            elif tag == "SAMPLES":
                proto["samples"] = fields[1:9]
            elif tag == "BENCH_US" and len(fields) >= 4:
                proto["bench"] = (float(fields[1]), float(fields[2]),
                                  float(fields[3]))
        except ValueError:
            proto.setdefault("malformed", []).append(line)
    return proto


def sanitizer_errors(tool: str, out: str) -> list[str]:
    """Mirror internal/e2e requireZeroSanitizerErrors: every *SUMMARY:* line
    mentioning errors must report "0 errors"; at least one SUMMARY required.
    Returns a list of violation strings (empty = clean)."""
    problems = []
    summaries = 0
    for line in out.splitlines():
        if "SUMMARY:" not in line:
            continue
        summaries += 1
        if "errors" in line and "0 errors" not in line:
            problems.append(f"{tool} reported errors: {line.strip()}")
    if summaries == 0:
        problems.append(f"{tool} produced no SUMMARY line")
    return problems


# --------------------------------------------------------------------------
# Per-test pipeline
# --------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str, category: str) -> None:
        self.name = name
        self.category = category
        self.status = "FAIL"       # PASS | FAIL | SKIP
        self.reason = ""
        self.t_compile = 0.0
        self.t_build = 0.0
        self.t_run = 0.0
        self.maxdiff = None
        self.cache = ""            # "cache" | "build" | ""
        self.bench = None          # (avg_us, min_us, max_us)
        self.gbps = None
        self.fixture_dir = None


def run_test(test: dict, tools: dict, args: argparse.Namespace) -> TestResult:
    res = TestResult(test["name"], test.get("category", "?"))
    name = test["name"]
    cxx_name = sanitize_name(name)

    # --- 1. COMPILE -------------------------------------------------------
    stage = Path(tempfile.mkdtemp(prefix=f"tcacomp-suite-{name}-"))
    t0 = time.monotonic()
    compile_argv = tools["compiler"] + [
        "compile", "-o", str(stage), str(REPO_ROOT / test["kernel"])]
    rc, out = run_argv(compile_argv)
    res.t_compile = time.monotonic() - t0
    if rc != 0:
        res.reason = "compile failed: " + tail(out)
        return res
    cus = sorted(stage.glob("*.cu"))
    if not cus:
        res.reason = "compile emitted no .cu"
        return res
    emitted = cus[0] if len(cus) == 1 else None
    if emitted is None:
        want = test.get("cxx", name)
        for c in cus:
            if c.stem.lower() == want.lower():
                emitted = c
                break
        if emitted is None:
            emitted = cus[0]

    runner_src = REPO_ROOT / test["runner"]
    # The shared runner prologue: copied beside runner.cu (quote-includes
    # resolve from the including file's directory) and hashed into the cache
    # key so a common.cuh change invalidates every fixture.
    common_cuh = runner_src.parent / "common.cuh"
    cmake_lists = (CMAKE_TEMPLATE.read_text()
                   .replace("%PROJECT%", f"tca_{cxx_name}")
                   .replace("%OBJLIB%", f"{cxx_name}_obj")
                   .replace("%SRCFILE%", "src.cu")
                   .replace("%BINNAME%", cxx_name))
    binname = cxx_name

    # --- 2/3. cache key + BUILD (cached) ----------------------------------
    key = hashlib.sha256()
    key.update(TEMPLATE_VERSION)
    key.update(emitted.read_bytes())
    key.update(runner_src.read_bytes())
    key.update(common_cuh.read_bytes())
    key.update(cmake_lists.encode())
    cache_dir = CACHE_ROOT / key.hexdigest()[:16]
    res.fixture_dir = cache_dir
    build_dir = cache_dir / "build"
    binary = build_dir / binname
    done = cache_dir / ".done"

    cache_hit = done.is_file() and binary.is_file()
    if not cache_hit:
        t0 = time.monotonic()
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(emitted, cache_dir / "src.cu")
        shutil.copyfile(runner_src, cache_dir / "runner.cu")
        if common_cuh.is_file():
            shutil.copyfile(common_cuh, cache_dir / "common.cuh")
        (cache_dir / "CMakeLists.txt").write_text(cmake_lists)
        rc, out = run_argv([
            tools["cmake"], "-S", str(cache_dir), "-B", str(build_dir),
            f"-DCMAKE_CUDA_COMPILER={tools['nvcc']}",
            f"-DTCACOMP_RUNTIME_DIR={args.runtime_dir}",
        ])
        if rc != 0:
            res.t_build = time.monotonic() - t0
            res.reason = "cmake configure failed: " + tail(out)
            return res
        rc, out = run_argv([tools["cmake"], "--build", str(build_dir), "-j8"])
        res.t_build = time.monotonic() - t0
        if rc != 0:
            res.reason = "cmake build failed: " + tail(out)
            return res
        if not binary.is_file():
            res.reason = f"fixture binary missing after build: {binary}"
            return res
        done.write_text("done\n")
        res.cache = "build"
    else:
        res.cache = "cache"
    shutil.rmtree(stage, ignore_errors=True)

    # --- 4. RUN (+ optional sanitizer / bench) ----------------------------
    run_argv_list = [str(binary)]
    if args.bench_iters > 0:
        run_argv_list += ["--bench", str(args.bench_iters)]
    label = "run"
    if args.sanitizer:
        run_argv_list = [tools["compute-sanitizer"], "--tool", args.sanitizer] \
            + run_argv_list
        label = f"sanitizer:{args.sanitizer}"
    t0 = time.monotonic()
    rc, out = run_argv(run_argv_list)
    res.t_run = time.monotonic() - t0
    if rc != 0:
        res.reason = f"{label} exited {rc}: " + tail(out)
        return res

    proto = parse_protocol(out)
    for bad in proto.get("malformed", []):
        res.reason = f"malformed protocol line: {bad!r}"
        return res

    # --- 6. SANITIZER check ----------------------------------------------
    if args.sanitizer:
        problems = sanitizer_errors(args.sanitizer, out)
        if problems:
            res.reason = "; ".join(problems)
            return res

    # --- 5. VERIFY ---------------------------------------------------------
    if "canary" not in proto:
        res.reason = "missing CANARY line: " + tail(out)
        return res
    if proto["canary"] != 0:
        res.reason = f"canary {proto['canary']} (want 0 — unwritten outputs)"
        return res
    tol = float(test.get("tolerance", 0.0))
    if tol == 0.0:
        if proto.get("checksum") != proto.get("ref_checksum"):
            res.reason = (f"bitwise mismatch: got {proto.get('checksum')!r} "
                          f"want {proto.get('ref_checksum')!r}")
            if proto.get("samples"):
                res.reason += f"; gpu samples {proto['samples']}"
            return res
        res.maxdiff = 0.0
    else:
        if "maxdiff" not in proto:
            res.reason = "missing MAXDIFF line: " + tail(out)
            return res
        res.maxdiff = proto["maxdiff"]
        if res.maxdiff > tol:
            res.reason = (f"maxdiff {res.maxdiff:g} > tolerance {tol:g}; "
                          f"checksum got {proto.get('checksum')!r} "
                          f"want {proto.get('ref_checksum')!r}")
            if proto.get("samples"):
                res.reason += f"; gpu samples {proto['samples']}"
            return res

    # --- 7. BENCH ----------------------------------------------------------
    if args.bench_iters > 0 and "bench" in proto:
        res.bench = proto["bench"]
        bb = test.get("bench_bytes")
        if bb and res.bench[0] > 0:
            res.gbps = bb / (res.bench[0] * 1e-6)

    res.status = "PASS"
    return res


def tail(out: str, n: int = 12) -> str:
    lines = [l for l in out.rstrip().splitlines() if l.strip()]
    return " | ".join(lines[-n:])


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def fmt_test_line(res: TestResult, color: Color, verbose: bool) -> str:
    tag = {"PASS": color.green("PASS"), "FAIL": color.red("FAIL"),
           "SKIP": color.yellow("SKIP")}[res.status]
    parts = [tag, f"{res.name}", f"[{res.category}]",
             f"compile {res.t_compile:.1f}s", f"build {res.t_build:.1f}s",
             f"run {res.t_run:.1f}s"]
    if res.cache:
        parts.append(f"({res.cache})")
    if res.maxdiff is not None:
        parts.append(f"maxdiff {res.maxdiff:g}")
    line = " ".join(parts)
    if res.status == "FAIL":
        line += f"\n    reason: {res.reason}"
        if res.fixture_dir:
            line += f"\n    fixture: {res.fixture_dir}"
    elif verbose and res.reason:
        line += f"\n    note: {res.reason}"
    return line


def report(results: list[TestResult], elapsed: float, color: Color, bench_mode: bool, verbose: bool) -> int:
    by_cat = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
        print(fmt_test_line(r, color, verbose))

    print()
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        npass = sum(1 for r in rs if r.status == "PASS")
        nfail = sum(1 for r in rs if r.status == "FAIL")
        nskip = sum(1 for r in rs if r.status == "SKIP")
        t = sum(r.t_compile + r.t_build + r.t_run for r in rs)
        print(f"  {cat}: {npass} passed, {nfail} failed, {nskip} skipped "
              f"({t:.1f}s)")

    if bench_mode:
        print()
        print(color.cyan("  BENCH (microseconds per iteration)"))
        print(color.cyan(f"  {'name':<14} {'avg us':>10} {'min us':>10} "
                         f"{'max us':>10} {'GB/s':>10}"))
        for r in results:
            if r.bench is None:
                continue
            gbps = f"{r.gbps:,.0f}" if r.gbps else "-"
            print(f"  {r.name:<14} {r.bench[0]:>10.1f} {r.bench[1]:>10.1f} "
                  f"{r.bench[2]:>10.1f} {gbps:>10}")

    npass = sum(1 for r in results if r.status == "PASS")
    nfail = sum(1 for r in results if r.status == "FAIL")
    nskip = sum(1 for r in results if r.status == "SKIP")
    final = (f"{npass} passed, {nfail} failed, {nskip} skipped, "
             f"total {elapsed:.1f}s")
    if nfail:
        print(color.bold(color.red(final)))
    else:
        print(color.bold(color.green(final)))
    return 0 if nfail == 0 else 1


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    global SHOW_ARGV
    parser = argparse.ArgumentParser(
        prog="orchestrator.py",
        description="tcacomp testing suite driver (stdlib only)")
    parser.add_argument("--filter", help="substring match on test name")
    parser.add_argument("--category", help="run only one category")
    parser.add_argument("--list", action="store_true",
                        help="print the registry and exit (no tools needed)")
    parser.add_argument("--bench-iters", type=int, default=0,
                        help="timed kernel iterations per test (0 = off)")
    parser.add_argument("--sanitizer", choices=["memcheck", "racecheck"],
                        help="run binaries under compute-sanitizer TOOL")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel tests (ThreadPoolExecutor workers)")
    parser.add_argument("--keep-builds", action="store_true",
                        help="print every fixture dir, not just failing ones")
    parser.add_argument("--clean", action="store_true",
                        help="rm -rf the suite build cache, then continue")
    parser.add_argument("--verbose", action="store_true",
                        help="print every subprocess argv + cache hit/miss")
    parser.add_argument("--tcacomp", help="override the compiler binary path")
    parser.add_argument("--runtime-dir", default=str(REPO_ROOT / "runtime"),
                        help="the tcacomp runtime source dir (cmake -D)")
    args = parser.parse_args(argv)

    if args.bench_iters < 0:
        parser.error("--bench-iters must be >= 0")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    color = Color(sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
    SHOW_ARGV = args.verbose

    tests = load_registry()

    if args.list:
        print(f"{'name':<14} {'category':<10} {'dims':<18} {'tol':<8} scalars")
        for t in tests:
            dims = "x".join(str(d) for d in t.get("dims", []))
            sc = ", ".join(f"{k}={v}" for k, v in t.get("scalars", {}).items())
            print(f"{t['name']:<14} {t.get('category', '?'):<10} "
                  f"{dims:<18} {t.get('tolerance', 0.0):<8} {sc}")
        print(f"\n{len(tests)} tests registered")
        return 0

    if args.clean and CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
        print(f"cleaned {CACHE_ROOT}")

    # Tool probing — a missing cmake/nvcc/compiler skips ALL tests (the
    # TCACOMP_E2E hermetic-skip philosophy: a skip is not a failure).
    tools = {}
    missing = []
    for tool in ("cmake", "nvcc"):
        path = find_tool(tool)
        if path is None:
            missing.append(tool)
        tools[tool] = path
    tools["compiler"] = probe_compiler(args)
    if not args.tcacomp and tools["compiler"][0] != "go":
        if not Path(tools["compiler"][0]).is_file():
            missing.append("tcacomp compiler")
    if args.sanitizer:
        san = find_tool("compute-sanitizer")
        if san is None:
            missing.append("compute-sanitizer")
        tools["compute-sanitizer"] = san

    if missing:
        print(color.yellow(
            f"ALL-SKIPPED: missing tool(s): {', '.join(missing)} — "
            f"nvcc/cmake/compiler absent, hermetic skip (not a failure)"))
        return 0

    selected = select_tests(tests, args)
    if not selected:
        print("no tests selected")
        return 0

    t0 = time.monotonic()
    if args.jobs == 1:
        results = [run_test(t, tools, args) for t in selected]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(lambda t: run_test(t, tools, args), selected))
    elapsed = time.monotonic() - t0

    if args.keep_builds:
        for r in results:
            if r.fixture_dir is not None:
                print(f"  fixture: {r.fixture_dir}")
    return report(results, elapsed, color, args.bench_iters > 0, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
