"""Rehearsal: run the whole assignment on a laptop, with no GPU, and check the result.

SMOKE=1 ./run_all.sh checks the chain on the lab box, but it needs a CUDA device and it
burns reserved time. This does the same job from a laptop, in two stages:

  Stage 1  Every Part A to E script is run for real, on both simulated cards, against
           the fake CUDA device in fakecuda.py. The scripts are not stubbed or mocked.
           The actual measurement code executes, allocates, OOMs, bisects and logs.
           Only the device underneath it is synthetic.
  Stage 2  make_figures.py and make_metrics.py run over whatever Stage 1 wrote.

Then it asserts the things that would otherwise be found on the clock, in the lab, at
the end of a 20 minute thermal run: every script exits 0, RUN_LOG.txt carries a header
block per part and a UUID on every line, every CSV is UUID labelled and records reps and
warmup, Part D finds an OOM bracket whose fitted coefficient matches theory, the thermal
log is not ragged, and METRICS.md has no unfilled cells.

The numbers are meaningless, the device is fake, and everything lands in a scratch
directory outside the repository. Only a run on the reserved card is a measurement.

    python scripts/rehearse.py                 # about a minute, prints pass or fail
    python scripts/rehearse.py --keep          # leave the tree in place to inspect
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# One card throttles and answers to the old nvidia-smi field name, the other does not
# throttle and only answers to the new one. One runs FP8 and the other refuses it.
# Between them every reporting branch gets walked.
CARDS = [
    {"key": "4090", "tag": "4090feedface", "break_fp8": True, "smi_fail_every": 5},
    {"key": "5090", "tag": "5090cafebabe", "break_fp8": False, "smi_fail_every": 0},
]

# Short where it costs nothing, full where the shape of the run is the point. Part B
# keeps all four sizes and Part D keeps the whole sweep plus the OOM search. Part E runs
# 30 s at 2 s sampling, long enough to cross the simulated throttle point.
STEPS = [
    ("part_a_provenance.py", []),
    ("part_b_precision.py", ["--warmup", "2", "--iters", "5"]),
    ("part_c_roofline.py", ["--warmup", "2", "--iters", "5"]),
    ("part_d_attention.py", ["--warmup", "1", "--iters", "3"]),
    ("part_e_thermal.py", ["--minutes", "0.5", "--interval", "2", "--n", "4096"]),
]

LOG_LINE = re.compile(r"^\[\d{4}-\d{2}-\d{2}T[\d:]+\] \[[A-E]\] \[GPU-[^\]]+\] ")


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        return ok

    @property
    def failures(self):
        return [name for name, ok, _ in self.rows if not ok]

    def report(self):
        print("checks")
        for name, ok, detail in self.rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def run_stage1(root, checks, verbose):
    """Every part script, for real, on each simulated card."""
    env = dict(os.environ, HW25_ROOT=root)
    for card in CARDS:
        print(f"\nstage 1: RTX {card['key']} (simulated)")
        for script, extra in STEPS:
            cmd = [sys.executable, os.path.join(HERE, "fakecuda.py"), "--card", card["key"]]
            if card["break_fp8"] and script.startswith("part_b"):
                cmd.append("--break-fp8")
            if card["smi_fail_every"] and script.startswith("part_e"):
                cmd += ["--smi-fail-every", str(card["smi_fail_every"])]
            cmd += [os.path.join(HERE, script), "--index", "0"] + extra
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            ok = proc.returncode == 0
            print(f"  {script:<24} exit {proc.returncode}")
            if verbose or not ok:
                print((proc.stdout or "").strip()[-2500:])
                print((proc.stderr or "").strip()[-2500:])
            checks.add(f"{script} runs on {card['key']}", ok, f"exit {proc.returncode}")


def run_stage2(root, checks, verbose):
    env = dict(os.environ, HW25_ROOT=root)
    print("\nstage 2: figures and METRICS.md")
    for script in ("make_figures.py", "make_metrics.py"):
        proc = subprocess.run([sys.executable, os.path.join(HERE, script)],
                              capture_output=True, text=True, env=env)
        ok = proc.returncode == 0
        print(f"  {script:<24} exit {proc.returncode}")
        if verbose or not ok:
            print((proc.stdout or "").strip())
            print((proc.stderr or "").strip())
        checks.add(f"{script} runs", ok, f"exit {proc.returncode}")


def check_run_log(root, checks):
    """The shared log format: a header block per part, a UUID on every line."""
    path = os.path.join(root, "RUN_LOG.txt")
    if not checks.add("RUN_LOG.txt written", os.path.exists(path)):
        return
    text = open(path).read()
    for part in "ABCDE":
        want = f"PART {part}"
        checks.add(f"RUN_LOG.txt has a {want} header block",
                   text.count(want) >= len(CARDS), f"{text.count(want)} blocks")
    body = [ln for ln in text.splitlines() if ln.startswith("[")]
    bad = [ln for ln in body if not LOG_LINE.match(ln)]
    checks.add("every RUN_LOG.txt entry is UUID labelled", not bad,
               f"{len(body)} entries" if not bad else f"{len(bad)} unlabelled, e.g. {bad[0][:70]}")
    checks.add("no unlabelled entries", "[no uuid]" not in text)
    checks.add("SID4 and SEED recorded in RUN_LOG.txt", "SID4=" in text and "SEED=" in text)


def check_csvs(root, checks):
    """Every CSV is UUID labelled on every row, and the thermal log is not ragged."""
    for path in sorted(glob.glob(os.path.join(root, "data", "*.csv"))):
        name = os.path.basename(path)
        rows = list(csv.DictReader(open(path)))
        if not checks.add(f"{name} has rows", bool(rows), f"{len(rows)} rows"):
            continue
        has_uuid = "uuid" in rows[0] and all((r.get("uuid") or "").startswith("GPU-")
                                             for r in rows)
        checks.add(f"{name} UUID labelled on every row", has_uuid)
        if "reps" in rows[0]:
            checks.add(f"{name} records warmup alongside reps", "warmup" in rows[0])
        if "vram_total_gib" in rows[0]:
            bad = []
            for r in rows:
                try:
                    if r["ok"] == "1" and float(r["peak_mem_gib"]) > float(r["vram_total_gib"]):
                        bad.append(r["seq_len"])
                except (TypeError, ValueError, KeyError):
                    continue
            checks.add(f"{name} no success exceeds physical VRAM", not bad,
                       "clean" if not bad else f"S = {bad}")

    for path in sorted(glob.glob(os.path.join(root, "logs", "part_e_thermal_*.csv"))):
        name = os.path.basename(path)
        raw = list(csv.reader(open(path)))
        widths = collections.Counter(len(r) for r in raw)
        checks.add(f"{name} rows all one width", len(widths) == 1, f"widths {dict(widths)}")
        errored = sum(1 for r in raw[1:] if any("error:" in c for c in r))
        checks.add(f"{name} survived {errored} failed sample(s)", True,
                   f"{len(raw) - 1} samples, {errored} errored")


def check_part_b(root, checks):
    """Warmup and repetition count, which the assignment asks be reported."""
    for path in sorted(glob.glob(os.path.join(root, "data", "part_b_*.csv"))):
        name = os.path.basename(path)
        rows = list(csv.DictReader(open(path)))
        timed = [r for r in rows if r["achieved_tflops"]]
        checks.add(f"{name} reports reps for every timed config",
                   timed and all(int(r["reps"]) > 0 for r in timed),
                   f"{len(timed)} timed configs")
        checks.add(f"{name} reports warmup for every timed config",
                   timed and all(int(r["warmup"]) > 0 for r in timed))
        checks.add(f"{name} reports a CV for every timed config",
                   timed and all(r["cv_pct"] != "" for r in timed))
        checks.add(f"{name} covers fp32, tf32, fp16 and bf16",
                   {"fp32", "tf32", "fp16", "bf16"} <= {r["precision"] for r in rows})
        fp8 = [r for r in rows if r["precision"] == "fp8"]
        recorded = fp8 and all(r["achieved_tflops"] or r["note"] for r in fp8)
        checks.add(f"{name} records fp8 as a result or a named failure", recorded,
                   (fp8[0]["note"] or "ran")[:60] if fp8 else "absent")


def check_part_d(root, checks):
    """The OOM bracket and the quadratic, the two things Part D is graded on."""
    for path in sorted(glob.glob(os.path.join(root, "data", "part_d_summary_*.json"))):
        name = os.path.basename(path)
        summary = json.load(open(path))["summary"]
        naive = summary.get("naive", {})
        checks.add(f"{name} naive OOM bracket found",
                   naive.get("largest_ok") and naive.get("smallest_fail"),
                   f"largest ok {naive.get('largest_ok')}, smallest fail "
                   f"{naive.get('smallest_fail')}")
        if naive.get("largest_ok") and naive.get("smallest_fail"):
            width = naive["smallest_fail"] - naive["largest_ok"]
            checks.add(f"{name} bracket narrowed by bisection", width <= 256,
                       f"{width} tokens wide")
        expected = naive.get("expected_quad_a")
        measured = naive.get("quad_a")
        if expected and measured:
            err = abs(measured - expected) / expected * 100
            checks.add(f"{name} fitted coefficient matches theory", err < 1.0,
                       f"measured {measured:.2f} against expected {expected}, {err:.3f}% off")
        checks.add(f"{name} quadratic fit is good", naive.get("r2", 0) > 0.999,
                   f"R^2 = {naive.get('r2')}")
        fused = summary.get("fused", {})
        checks.add(f"{name} fused has no quadratic term",
                   abs(fused.get("quad_a", 1e9)) < 1.0, f"a = {fused.get('quad_a')}")
        if fused.get("largest_ok") and naive.get("largest_ok"):
            checks.add(f"{name} fused outlives naive",
                       fused["largest_ok"] > naive["largest_ok"],
                       f"fused ok to {fused['largest_ok']}, naive to {naive['largest_ok']}")


def check_failure_modes(checks, verbose):
    """Replay the two failure modes the RTX 5090 lab run actually hit.

    The first is a host that serves allocations past physical VRAM instead of failing,
    so nothing raises and the allocator reports a peak larger than the card. Unchecked,
    that walks the OOM search roughly 50% past the real boundary. The second is a CUDA
    context that does not survive a failed allocation.
    """
    import tempfile
    for label, flag, expect in (
        ("oversubscribing host", "--oversubscribe", "implausible"),
        ("context dies after OOM", "--context-dies", "context_dead"),
    ):
        root = tempfile.mkdtemp(prefix="hw25-failmode-")
        for sub in ("data", "logs", "figures", "provenance"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        env = dict(os.environ, HW25_ROOT=root)
        cmd = [sys.executable, os.path.join(HERE, "fakecuda.py"), "--card", "5090", flag,
               os.path.join(HERE, "part_d_attention.py"), "--index", "0",
               "--warmup", "1", "--iters", "3"]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print(f"  part_d under {label:<24} exit {proc.returncode}")
        if verbose or proc.returncode != 0:
            print((proc.stdout or "").strip()[-2000:])
            print((proc.stderr or "").strip()[-2000:])
        if not checks.add(f"part_d survives a {label}", proc.returncode == 0,
                          f"exit {proc.returncode}"):
            continue

        path = glob.glob(os.path.join(root, "data", "part_d_*.csv"))
        path = [p for p in path if "summary" not in p]
        rows = list(csv.DictReader(open(path[0]))) if path else []
        kinds = {r.get("failure_kind", "") for r in rows}
        checks.add(f"{label} is classified as {expect}", expect in kinds,
                   f"kinds seen: {sorted(k for k in kinds if k)}")

        # The invariant that matters: nothing may be recorded as a success while
        # claiming more memory than the card physically holds.
        bad = []
        for r in rows:
            try:
                if r["ok"] == "1" and float(r["peak_mem_gib"]) > float(r["vram_total_gib"]):
                    bad.append((r["seq_len"], r["peak_mem_gib"], r["vram_total_gib"]))
            except (TypeError, ValueError, KeyError):
                continue
        checks.add(f"{label}: no success exceeds physical VRAM", not bad,
                   "clean" if not bad else f"{len(bad)} bad rows, e.g. {bad[0]}")

        summary = glob.glob(os.path.join(root, "data", "part_d_summary_*.json"))
        naive = json.load(open(summary[0]))["summary"]["naive"] if summary else {}
        if flag == "--oversubscribe":
            # 32*S^2 + 4096*S = 31.84 GiB solves to about S = 32,600, so the recovered
            # boundary has to land near there rather than 50k.
            ok = naive.get("largest_ok") or 0
            checks.add("oversubscribing host: boundary recovered near the predicted 32600",
                       31000 <= ok <= 33000, f"largest_ok = {ok}")
        else:
            checks.add("context death is recorded in the summary",
                       naive.get("context_died") is True,
                       f"context_died = {naive.get('context_died')}")
        shutil.rmtree(root, ignore_errors=True)


def check_outputs(root, checks):
    figures = sorted(os.listdir(os.path.join(root, "figures")))
    for want in ("part_b_tflops_vs_size", "part_c_roofline", "part_d_peak_memory",
                 "part_d_latency", "part_e_clock_temp"):
        got = len([f for f in figures if f.startswith(want)])
        checks.add(f"{len(CARDS)} x {want}.png", got == len(CARDS), f"found {got}")

    path = os.path.join(root, "METRICS.md")
    if not checks.add("METRICS.md written", os.path.exists(path)):
        return
    metrics = open(path).read()
    checks.add("a summary table per card",
               metrics.count("Table HW2.5.1") == len(CARDS),
               f"{metrics.count('Table HW2.5.1')} tables")
    # Only table rows count. The preamble legitimately explains the phrase.
    stale = [ln for ln in metrics.splitlines() if ln.startswith("|") and "not measured" in ln]
    checks.add("no unfilled table cells", not stale,
               "clean" if not stale else f"{len(stale)} row(s): {stale[:1]}")
    checks.add("plateau table present", "where each precision plateaus" in metrics)
    checks.add("quadratic fit reported", "Quadratic fit (naive)" in metrics)
    for card in CARDS:
        checks.add(f"card {card['tag']} present in METRICS.md", card["tag"] in metrics)
    checks.add("throttling card reports a reason", "SwPowerCap" in metrics)
    checks.add("non throttling card reports none", "| none |" in metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="do not delete the scratch tree")
    ap.add_argument("--root", default=None, help="scratch directory to build in")
    ap.add_argument("--verbose", action="store_true", help="show each script's output")
    args = ap.parse_args()

    root = args.root or tempfile.mkdtemp(prefix="hw25-rehearsal-")
    for sub in ("data", "logs", "figures", "provenance"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    print(f"rehearsing in {root}")
    print("the CUDA device is simulated, so none of these numbers is a measurement")

    checks = Checks()
    run_stage1(root, checks, args.verbose)
    run_stage2(root, checks, args.verbose)
    print("\nstage 3: failure modes seen in the lab")
    check_failure_modes(checks, args.verbose)
    print()
    check_run_log(root, checks)
    check_csvs(root, checks)
    check_part_b(root, checks)
    check_part_d(root, checks)
    check_outputs(root, checks)
    checks.report()

    print()
    if checks.failures:
        print(f"REHEARSAL FAILED: {len(checks.failures)} problem(s)")
        for name in checks.failures:
            print(f"  {name}")
    else:
        print(f"REHEARSAL PASSED, {len(checks.rows)} checks. Parts A to E run end to end, "
              f"log in the shared format, and the builders consume what they write.")
        print("The device was fake. Only a run on the reserved card is a result.")

    if args.keep:
        print(f"\nscratch tree kept at {root}")
    elif not args.root:
        shutil.rmtree(root, ignore_errors=True)
    return 1 if checks.failures else 0


if __name__ == "__main__":
    sys.exit(main())
