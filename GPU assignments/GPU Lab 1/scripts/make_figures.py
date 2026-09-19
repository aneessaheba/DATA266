"""Builds every figure from the CSVs the part scripts wrote. Needs no GPU."""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"
PRECISION_ORDER = ["fp32", "tf32", "fp16", "bf16", "fp8"]

# FP16 and BF16 run the same kernels against the same peak, so their curves land on top
# of one another. Distinct dashes and markers keep the covered line readable instead of
# letting the last series drawn erase the one underneath.
DASHES = ["-", (0, (7, 3)), "-", (0, (1.6, 2.4)), (0, (5, 2, 1, 2))]
MARKERS = ["o", "s", "^", "D", "v"]


def place_end_labels(ax, labels):
    """Right hand series labels, pushed apart so overlapping curves stay legible.

    labels is a list of (x, y, text, color). Without this, two series that finish at the
    same value print their labels on top of each other and both become unreadable.
    """
    if not labels:
        return
    low, high = ax.get_ylim()
    gap = (high - low) * 0.05
    placed = []
    for x, y, text, color in sorted(labels, key=lambda item: item[1]):
        target = y
        if placed and target - placed[-1][1] < gap:
            target = placed[-1][1] + gap
        placed.append((x, target, text, color, y))
    for x, target, text, color, y in placed:
        # Anchor the text at the moved height, not the data point, or the push apart
        # above was for nothing and coincident series print over each other.
        ax.annotate(text, xy=(x, target), xytext=(9, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center", ha="left",
                    annotation_clip=False)
        if abs(target - y) > gap * 0.1:
            ax.annotate("", xy=(x, y), xytext=(x, target), textcoords="data",
                        arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8,
                                        alpha=0.7, shrinkA=0, shrinkB=0),
                        annotation_clip=False)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=12, pad=12, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def tag_of(path):
    return os.path.basename(path).rsplit("_", 1)[-1].replace(".csv", "").replace(".json", "")


def figure_b(path):
    """Achieved TFLOPS against matrix size, one line per precision, single figure."""
    rows = [r for r in common.read_csv(path) if r["achieved_tflops"]]
    if not rows:
        return
    gpu = rows[0]["gpu_name"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    present = [p for p in PRECISION_ORDER if any(r["precision"] == p for r in rows)]
    end_labels = []
    for slot, precision in enumerate(present):
        pts = sorted(((int(r["n"]), float(r["achieved_tflops"]))
                      for r in rows if r["precision"] == precision), key=lambda t: t[0])
        xs, ys = zip(*pts)
        color = SERIES[slot % len(SERIES)]
        ax.plot(xs, ys, color=color, linewidth=2, linestyle=DASHES[slot % len(DASHES)],
                marker=MARKERS[slot % len(MARKERS)], markersize=7.5 - 0.8 * slot,
                markeredgecolor=SURFACE, markeredgewidth=1.6, label=precision.upper(),
                zorder=3 + slot)
        end_labels.append((xs[-1], ys[-1], f"{precision.upper()}  {ys[-1]:.0f}", color))
    sizes = sorted({int(r["n"]) for r in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlim(min(sizes) * 0.85, max(sizes) * 1.9)
    ax.set_ylim(bottom=0)
    style(ax, f"Achieved matmul throughput by precision, {gpu}",
          "Matrix size N (log scale)", "Achieved TFLOPS")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left", ncol=len(present))
    place_end_labels(ax, end_labels)
    save(fig, os.path.join(common.FIGURES, f"part_b_tflops_vs_size_{tag_of(path)}.png"))


def figure_c(path):
    """Roofline: measured operations placed against the card's ridge point."""
    rows = common.read_csv(path)
    if not rows or not rows[0].get("ridge_point_flops_per_byte"):
        return
    gpu = rows[0]["gpu_name"]
    ridge = float(rows[0]["ridge_point_flops_per_byte"])
    peak_tflops = ridge * float(rows[0]["spec_bandwidth_gb_s"]) / 1000.0
    bw = float(rows[0]["spec_bandwidth_gb_s"])

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    xs = np.logspace(-2, 3, 200)
    roof = np.minimum(bw * xs / 1000.0, peak_tflops)
    ax.plot(xs, roof, color=INK2, linewidth=2, zorder=2, label="Roofline (vendor peak)")
    ax.axvline(ridge, color=GRID, linewidth=1.5, linestyle="--", zorder=1)
    ax.annotate(f"ridge {ridge:.0f} FLOPs per byte", (ridge, peak_tflops * 0.055),
                color=INK2, fontsize=9, rotation=90, va="bottom", ha="right",
                xytext=(-4, 0), textcoords="offset points")
    for slot, row in enumerate(rows):
        intensity = float(row["arithmetic_intensity_flops_per_byte"])
        achieved = float(row["achieved_tflops"])
        color = SERIES[slot % len(SERIES)]
        if achieved <= 0:
            # A pure copy does no useful FLOPs, so plot where bandwidth puts it.
            achieved = float(row["effective_bandwidth_gb_s"]) * max(intensity, 1e-3) / 1000.0
        ax.plot([intensity], [max(achieved, 1e-3)], marker="o", markersize=9, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4, linestyle="none")
        ax.annotate(f" {row['op']} ({row['roofline_side']})", (intensity, max(achieved, 1e-3)),
                    color=INK, fontsize=9, va="bottom", ha="left",
                    xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, f"Roofline placement of the measured operations, {gpu}",
          "Arithmetic intensity (FLOPs per byte, log scale)", "Achieved TFLOPS (log scale)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="lower right")
    save(fig, os.path.join(common.FIGURES, f"part_c_roofline_{tag_of(path)}.png"))


def figure_d(path):
    """Peak memory against sequence length, with the quadratic fitted to the naive data."""
    rows = [r for r in common.read_csv(path) if r["ok"] == "1" and r["phase"] == "sweep"]
    if not rows:
        return
    gpu = rows[0]["gpu_name"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    fit_note = ""
    for slot, impl in enumerate(["naive", "fused"]):
        pts = sorted(((int(r["seq_len"]), float(r["peak_mem_gib"]))
                      for r in rows if r["impl"] == impl), key=lambda t: t[0])
        if not pts:
            continue
        xs, ys = zip(*pts)
        color = SERIES[slot]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2,
                label=f"{impl} attention", zorder=4 + slot)
        ax.annotate(f" {impl}", (xs[-1], ys[-1]), color=INK, fontsize=9, va="center",
                    ha="left", xytext=(6, 0), textcoords="offset points")
        if impl == "naive" and len(xs) >= 3:
            coeffs = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 2)
            grid = np.linspace(min(xs), max(xs), 200)
            ax.plot(grid, np.polyval(coeffs, grid), color=color, linewidth=1.2,
                    linestyle="--", alpha=0.75, zorder=3,
                    label=f"fit {coeffs[0]:.3g}S2 + {coeffs[1]:.3g}S + {coeffs[2]:.3g} (GiB)")
            fit_note = f"measured S squared coefficient: {coeffs[0]:.4g} GiB per token squared"
    style(ax, f"Peak memory of attention against sequence length, {gpu}",
          "Sequence length S (tokens)", "Peak allocated memory (GiB)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    if fit_note:
        ax.annotate(fit_note, (0.02, 0.78), xycoords="axes fraction", color=INK2, fontsize=9)
    save(fig, os.path.join(common.FIGURES, f"part_d_peak_memory_{tag_of(path)}.png"))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for slot, impl in enumerate(["naive", "fused"]):
        pts = sorted(((int(r["seq_len"]), float(r["latency_ms"]))
                      for r in rows if r["impl"] == impl), key=lambda t: t[0])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=SERIES[slot], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=f"{impl} attention")
        ax.annotate(f" {impl}", (xs[-1], ys[-1]), color=INK, fontsize=9, va="center",
                    ha="left", xytext=(6, 0), textcoords="offset points")
    ax.set_yscale("log")
    style(ax, f"Forward latency of attention against sequence length, {gpu}",
          "Sequence length S (tokens)", "Forward latency (ms, log scale)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    save(fig, os.path.join(common.FIGURES, f"part_d_latency_{tag_of(path)}.png"))


def _floats(rows, key):
    out = []
    for row in rows:
        try:
            out.append(float(row[key]))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.asarray(out)


def figure_e(path):
    """Clock and temperature over the sustained run.

    One panel per measure sharing the time axis, never two y scales on one pair of axes,
    so a 10 GHz memory clock cannot flatten a 2.7 GHz SM clock into a straight line.
    """
    rows = common.read_csv(path)
    if not rows:
        return
    t = _floats(rows, "elapsed_s")
    panels = [
        ("clocks.current.sm", "SM clock (MHz)", SERIES[0], "{:.0f} MHz"),
        ("clocks.current.memory", "Memory clock (MHz)", SERIES[1], "{:.0f} MHz"),
        ("temperature.gpu", "Temperature (°C)", SERIES[2], "{:.0f} °C"),
        ("power.draw", "Power draw (W)", SERIES[3], "{:.0f} W"),
    ]

    # Throttle onset, counted only from the point the load actually started. Sampling
    # begins before the first matmul lands, and scanning across those idle leading
    # samples would report an onset of 0 s against an idle clock.
    #
    # The card's own reason bits come first, exactly as make_metrics.py does it, with
    # the 95% of cold clock heuristic only as a fallback. Using the heuristic alone
    # made this figure claim no throttling on a run where the table reported SwPowerCap
    # from the first loaded sample: the card sat at its power cap while holding 96.5%
    # of its cold clock, which is throttling the heuristic cannot see.
    sm = _floats(rows, "clocks.current.sm")
    onset = None
    reasons = ""
    start = 0
    if sm.size and not np.isnan(sm).all():
        busy = np.where(sm >= 0.5 * np.nanmax(sm))[0]
        start = int(busy[0]) if busy.size else 0
    for i in range(start, len(rows)):
        names = common.decode_throttle_reasons(
            rows[i].get("clocks_throttle_reasons.active", ""))
        if names:
            onset, reasons = float(t[i]), ", ".join(names)
            break
    if onset is None:
        early = sm[start:][t[start:] <= t[start] + 30.0] if sm.size else sm
        if early.size and not np.isnan(early).all():
            baseline = np.nanmax(early)
            below = np.where(sm[start:] < 0.95 * baseline)[0]
            if below.size:
                onset = float(t[start:][below[0]])

    fig, axes = plt.subplots(len(panels), 1, figsize=(8.6, 10.0), sharex=True)
    for ax, (key, label, color, unit) in zip(axes, panels):
        series = _floats(rows, key)
        ax.plot(t, series, color=color, linewidth=2, zorder=3)
        if not np.isnan(series).all():
            peak_i = int(np.nanargmax(series))
            span = np.nanmax(series) - np.nanmin(series)
            pad = span * 0.25 if span else max(abs(np.nanmax(series)) * 0.05, 1.0)
            ax.set_ylim(np.nanmin(series) - pad * 0.4, np.nanmax(series) + pad)
            # A peak near the top of the panel has no room for a label above it, so
            # flip underneath rather than clipping against the frame or the caption.
            low, high = ax.get_ylim()
            crowded = series[peak_i] > low + 0.85 * (high - low)
            ax.annotate("peak " + unit.format(np.nanmax(series)), (t[peak_i], series[peak_i]),
                        color=INK, fontsize=9,
                        va="top" if crowded else "bottom", ha="center",
                        xytext=(0, -9 if crowded else 8), textcoords="offset points")
        style(ax, "", "", label)
        if onset is not None:
            ax.axvline(onset, color=SERIES[4], linewidth=1.5, linestyle="--", zorder=1)

    axes[0].set_title("Clock, temperature and power under sustained load",
                      color=INK, fontsize=12, pad=14, loc="left")
    axes[-1].set_xlabel("Elapsed time (s)", color=INK2, fontsize=10)
    if onset is not None and reasons:
        caption = f"nvidia-smi reports {reasons} from {onset:.0f} s"
    elif onset is not None:
        caption = f"SM clock falls below 95% of its cold value at {onset:.0f} s"
    else:
        caption = ("no throttle reason bits set and SM clock held above 95% of its cold "
                   "value")
    if onset is not None:
        axes[0].annotate(caption, (onset, 1.0), xycoords=("data", "axes fraction"),
                         color=INK, fontsize=9, va="top", ha="left",
                         xytext=(6, -6), textcoords="offset points")
    else:
        axes[0].annotate(caption, (0.02, 0.08), xycoords="axes fraction",
                         color=INK2, fontsize=9)
    save(fig, os.path.join(common.FIGURES, f"part_e_clock_temp_{tag_of(path)}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=common.DATA)
    ap.add_argument("--logs", default=common.LOGS)
    args = ap.parse_args()

    for path in sorted(glob.glob(os.path.join(args.data, "part_b_*.csv"))):
        figure_b(path)
    for path in sorted(glob.glob(os.path.join(args.data, "part_c_*.csv"))):
        figure_c(path)
    for path in sorted(glob.glob(os.path.join(args.data, "part_d_*.csv"))):
        if "summary" not in path:
            figure_d(path)
    for path in sorted(glob.glob(os.path.join(args.logs, "part_e_thermal_*.csv"))):
        figure_e(path)


if __name__ == "__main__":
    main()
