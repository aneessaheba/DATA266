"""Assembles METRICS.md, including Table HW2.5.1, from the CSVs and JSON summaries.
Every cell carries the UUID it came from, so each one is traceable to RUN_LOG.txt."""
import argparse
import glob
import json
import os

import common


def load_part(pattern, folder):
    """Map of uuid tag to parsed contents for every file matching the pattern."""
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, pattern))):
        tag = os.path.basename(path).rsplit("_", 1)[-1].split(".")[0]
        if path.endswith(".json"):
            with open(path) as fh:
                out[tag] = json.load(fh)
        else:
            out[tag] = common.read_csv(path)
    return out


def fmt(value, digits=2, suffix=""):
    if value in (None, "", "nan"):
        return "not measured"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def oom_cell(entry):
    if not entry:
        return "not measured"
    ok, fail = entry.get("largest_ok"), entry.get("smallest_fail")
    if fail is None:
        return f"no OOM up to S={ok} (largest tested)"
    return f"largest success S={ok}, smallest failure S={fail}"


# nvidia-smi throttle reason bits. GpuIdle and ApplicationsClocksSetting are not
# throttling, so they are never treated as an onset.
THROTTLE_BITS = [
    (0x0000000000000004, "SwPowerCap"),
    (0x0000000000000008, "HwSlowdown"),
    (0x0000000000000020, "SwThermalSlowdown"),
    (0x0000000000000040, "HwThermalSlowdown"),
    (0x0000000000000080, "HwPowerBrakeSlowdown"),
]


def decode_reasons(value):
    """Names of the throttling bits set in an nvidia-smi reason field, if any."""
    if not value:
        return []
    value = value.strip()
    try:
        bits = int(value, 16) if value.lower().startswith("0x") else int(value)
    except ValueError:
        # Some builds print text instead of a bitmask.
        low = value.lower()
        if "not active" in low or low in ("n/a", "unavailable", ""):
            return []
        return [value]
    return [name for bit, name in THROTTLE_BITS if bits & bit]


def throttle_onset(tag):
    """When the card first throttled, and why.

    Prefers nvidia-smi's own reason bits, the hardware stating that it is throttling,
    and falls back to a 95% of cold clock heuristic. The heuristic necessarily lags the
    real event, since the clock has to fall 5% before it trips, so when both are
    available both are reported and the reason bit time is the one used.
    """
    path = os.path.join(common.LOGS, f"part_e_thermal_{tag}.csv")
    if not os.path.exists(path):
        return None, ""
    rows = common.read_csv(path)
    samples = []
    for row in rows:
        try:
            samples.append((float(row["elapsed_s"]), float(row["clocks.current.sm"]), row))
        except (TypeError, ValueError):
            continue
    if not samples:
        return None, ""

    reason_onset = reason_note = None
    for elapsed, clock, row in samples:
        names = decode_reasons(row.get("clocks_throttle_reasons.active", ""))
        if names:
            reason_onset = elapsed
            reason_note = (f"nvidia-smi reported {', '.join(names)} at {elapsed:.0f} s, "
                           f"{row.get('temperature.gpu', '?')} C, "
                           f"{row.get('power.draw', '?')} W, SM clock {clock:.0f} MHz")
            break

    early = [s[1] for s in samples if s[0] <= 30.0]
    baseline = max(early) if early else None
    clock_onset = None
    if baseline:
        for elapsed, clock, row in samples:
            if clock < 0.95 * baseline:
                clock_onset = elapsed
                break

    if reason_onset is not None:
        note = reason_note
        if clock_onset is not None:
            note += (f". SM clock first fell below 95% of its cold {baseline:.0f} MHz at "
                     f"{clock_onset:.0f} s")
        return reason_onset, note
    if clock_onset is not None:
        row = next(r for e, c, r in samples if e == clock_onset)
        return clock_onset, (f"no throttle reason bits set, but SM clock fell from a cold "
                             f"{baseline:.0f} MHz to {float(row['clocks.current.sm']):.0f} "
                             f"MHz at {row.get('temperature.gpu', '?')} C")
    if baseline:
        return None, (f"no throttle reason bits set and SM clock held at or above 95% of "
                      f"{baseline:.0f} MHz for the whole run")
    return None, ""


PRECISION_ORDER = ["fp32", "tf32", "fp16", "bf16", "fp8"]


def plateau_rows(part_b):
    """Where each precision flattens out, read off the sweep instead of the figure.

    Plateau is the smallest N whose throughput is within 5% of the best that precision
    reached. If only the largest N qualifies, the sweep never showed a plateau, and that
    is reported rather than papered over.
    """
    lines = ["### Part B, where each precision plateaus", "",
             "Plateau is the smallest swept N reaching 95% of that precision's best "
             "measured throughput. The last column is why small matrices never plateau: "
             "at the smallest N the card delivers only a fraction of what it manages at "
             "the top of the sweep.", "",
             "| Precision | Best achieved TFLOPS | Plateau | Smallest N | "
             "Throughput at smallest N, as % of best |",
             "|---|---|---|---|---|"]
    any_row = False
    for precision in PRECISION_ORDER:
        pts = sorted((int(r["n"]), float(r["achieved_tflops"])) for r in part_b
                     if r["precision"] == precision and r["achieved_tflops"])
        if len(pts) < 2:
            continue
        any_row = True
        best = max(y for _, y in pts)
        largest_n = max(n for n, _ in pts)
        first_at_plateau = min(n for n, y in pts if y >= 0.95 * best)
        verdict = (f"not reached, still climbing at N={largest_n}"
                   if first_at_plateau == largest_n else f"N={first_at_plateau}")
        small_n, small_y = pts[0]
        lines.append(f"| {precision.upper()} | {best:.2f} | {verdict} | {small_n} | "
                     f"{small_y / best * 100:.1f}% |")
    lines.append("")
    return lines if any_row else []


def summary_table(part_b, part_c, part_d_sum, part_e_sum, tag):
    bf16 = [r for r in (part_b or []) if r["precision"] == "bf16" and r["achieved_tflops"]]
    best_bf16 = max(bf16, key=lambda r: float(r["achieved_tflops"])) if bf16 else None
    mem_rows = [r for r in (part_c or []) if "matmul" not in r["op"]]
    best_bw = max(mem_rows, key=lambda r: float(r["effective_bandwidth_gb_s"])) if mem_rows else None
    summary = (part_d_sum or {}).get("summary", {})
    part_e_sum = part_e_sum or {}

    if best_bf16:
        bf16_value = fmt(best_bf16["achieved_tflops"])
        bf16_note = f"N={best_bf16['n']}, {best_bf16['reps']} timed reps"
        pct_value = fmt(best_bf16["pct_of_peak"], 1, "%")
        pct_note = (f"vendor dense BF16 peak "
                    f"{fmt(best_bf16['theoretical_peak_tflops'], 1)} TFLOPS")
    else:
        bf16_value = pct_value = "not measured"
        bf16_note = pct_note = "Part B not run on this card"

    if best_bw:
        bw_value = fmt(best_bw["effective_bandwidth_gb_s"], 1)
        bw_note = (f"{best_bw['op']}, {fmt(best_bw['pct_of_spec_bandwidth'], 1)}% of "
                   f"{best_bw['spec_bandwidth_gb_s']} GB/s spec")
    else:
        bw_value, bw_note = "not measured", "Part C not run on this card"

    if part_e_sum:
        steady_value = fmt(part_e_sum.get("steady_over_peak_pct"), 1, "%")
        window = part_e_sum.get("steady_window_s", 300.0)
        duration = part_e_sum.get("duration_s") or 0.0
        steady_note = (f"peak {fmt(part_e_sum.get('peak_first30_tflops'))} TFLOPS in the "
                       f"first 30 s against steady "
                       f"{fmt(part_e_sum.get('steady_final5min_tflops'))}"
                       f" TFLOPS over the final {window / 60:.1f} min")
        if part_e_sum.get("run_too_short"):
            steady_note += (f". WARNING the run lasted only {duration / 60:.1f} min, not "
                            f"the 20 required, so this is not a steady state")
    else:
        steady_value, steady_note = "not measured", "Part E not run on this card"

    onset, throttle_note = throttle_onset(tag)
    if onset is not None:
        throttle_cell = f"{onset:.0f} s"
    elif throttle_note:
        throttle_cell = "none"
    else:
        throttle_cell = "not measured"
        throttle_note = "Part E thermal log not found for this card"

    d_config = (part_d_sum or {}).get("config", "Part D not run on this card")
    d_res = (part_d_sum or {}).get("refine_resolution", "?")

    return [
        "### Table HW2.5.1 summary", "",
        "| Measurement | Your GPU | Notes |",
        "|---|---|---|",
        f"| Peak achieved TFLOPS (BF16) | {bf16_value} | {bf16_note} |",
        f"| % of theoretical peak (BF16) | {pct_value} | {pct_note} |",
        f"| Effective bandwidth (GB/s) | {bw_value} | {bw_note} |",
        f"| Naive attention OOM length | {oom_cell(summary.get('naive'))} | {d_config} |",
        f"| Fused attention OOM length | {oom_cell(summary.get('fused'))} | "
        f"OOM bracket refined to {d_res} tokens |",
        f"| Steady state over peak throughput | {steady_value} | {steady_note} |",
        f"| Throttle onset (s, or none) | {throttle_cell} | {throttle_note} |",
        "",
    ]


def card_section(tag, identity, part_b, part_c, part_d, part_d_sum, part_e_sum):
    identity = identity or {}
    name = identity.get("gpu_name", f"unknown GPU ({tag})")
    lines = [f"## {name}", "",
             f"* GPU UUID: `{identity.get('uuid', tag)}`",
             f"* Driver {identity.get('driver_version', '?')} | CUDA (torch) "
             f"{identity.get('cuda_version', '?')} | torch {identity.get('torch_version', '?')}",
             f"* VRAM {identity.get('vram_total_mib', '?')} MiB | power limit "
             f"{identity.get('power_limit_w', '?')} W | host "
             f"{identity.get('host', '?')}", ""]
    lines += summary_table(part_b, part_c, part_d_sum, part_e_sum, tag)

    if part_b:
        lines += ["### Part B, achieved throughput", "",
                  "| N | Precision | Achieved TFLOPS | % of peak | Median ms | Reps | CV % | Note |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in part_b:
            median_ms = f"{float(row['median_s']) * 1e3:.3f}" if row["median_s"] else ""
            lines.append(
                f"| {row['n']} | {row['precision'].upper()} | {row['achieved_tflops'] or ''} "
                f"| {row['pct_of_peak'] or ''} | {median_ms} | {row['reps']} | "
                f"{row['cv_pct'] or ''} | {row.get('note', '') or ''} |")
        lines.append("")
        lines += plateau_rows(part_b)

    if part_c:
        lines += ["### Part C, roofline", "",
                  "| Operation | Effective GB/s | % of spec | Arithmetic intensity "
                  "(FLOPs per byte) | Ridge point | Side |", "|---|---|---|---|---|---|"]
        for row in part_c:
            lines.append(
                f"| {row['op']} | {row['effective_bandwidth_gb_s']} | "
                f"{row['pct_of_spec_bandwidth']} | "
                f"{row['arithmetic_intensity_flops_per_byte']} | "
                f"{row['ridge_point_flops_per_byte']} | {row['roofline_side']} |")
        lines.append("")

    if part_d:
        lines += ["### Part D, the cost of attention", "",
                  "| Impl | S | Peak memory (GiB) | Latency (ms) | Speedup vs naive | Status |",
                  "|---|---|---|---|---|---|"]
        for row in part_d:
            if row["phase"] != "sweep":
                continue
            status = "ok" if row["ok"] == "1" else f"OOM: {(row.get('note') or '')[:60]}"
            lines.append(
                f"| {row['impl']} | {row['seq_len']} | {row['peak_mem_gib'] or ''} | "
                f"{row['latency_ms'] or ''} | {row.get('speedup_vs_naive') or ''} | "
                f"{status} |")
        lines.append("")
        for impl, entry in ((part_d_sum or {}).get("summary") or {}).items():
            if "quad_a" in entry:
                lines.append(
                    f"Quadratic fit ({impl}): peak_mem_bytes = {entry['quad_a']:.6g}*S^2 + "
                    f"{entry['quad_b']:.6g}*S + {entry['quad_c']:.6g}, R^2 = "
                    f"{entry['r2']:.6f}. The S^2 coefficient is fitted to the measured "
                    f"sweep, not asserted.")
        lines.append("")

    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(common.ROOT, "METRICS.md"))
    args = ap.parse_args()

    identities = {}
    for path in sorted(glob.glob(os.path.join(common.DATA, "part_a_identity_*.json"))):
        with open(path) as fh:
            record = json.load(fh)
        identities[record["identity"]["uuid"][-12:]] = record["identity"]

    part_b = load_part("part_b_*.csv", common.DATA)
    part_c = load_part("part_c_*.csv", common.DATA)
    part_d = load_part("part_d_*.csv", common.DATA)
    part_d_sum = load_part("part_d_summary_*.json", common.DATA)
    part_e_sum = load_part("part_e_summary_*.json", common.DATA)

    tags = sorted(set(list(identities) + list(part_b) + list(part_c) + list(part_d)))
    out = ["# METRICS, HW2.5 GPU Assignment I", "",
           f"SID4={common.SID4} SEED={common.SEED}", "",
           "Every number below was produced by the scripts in `scripts/` and is echoed, "
           "UUID labelled, in `RUN_LOG.txt`. A cell reading `not measured` means that "
           "part has not been run on that card yet.", ""]
    if not tags:
        out += ["_No measurements found yet. Run the Part A to E scripts on the reserved "
                "workstation, then rerun `python scripts/make_metrics.py`._", ""]
    for tag in tags:
        out += card_section(tag, identities.get(tag), part_b.get(tag), part_c.get(tag),
                            part_d.get(tag), part_d_sum.get(tag), part_e_sum.get(tag))

    with open(args.out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
