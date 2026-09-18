"""Part E: 20 minutes of sustained matmul while nvidia-smi is sampled every 5 seconds."""
import argparse
import csv
import os
import subprocess
import threading
import time

import torch

import common

SAMPLE_FIELDS = [
    "timestamp", "clocks.current.sm", "clocks.current.memory", "temperature.gpu",
    "power.draw", "utilization.gpu", "utilization.memory", "clocks_throttle_reasons.active",
]

# Newer drivers renamed clocks_throttle_reasons to clocks_event_reasons. If the old name
# is rejected, every sampled row would come back as an error string and a 20 minute run
# would be wasted, so settle on a name this driver accepts before starting.
THROTTLE_FIELD_CANDIDATES = [
    "clocks_throttle_reasons.active",
    "clocks_event_reasons.active",
]

# What nvidia-smi says when a field name is wrong, as opposed to when the call simply
# failed. The two need different handling: try the next name, or try again.
_UNKNOWN_FIELD_MARKERS = ("not a valid field", "unknown field", "not supported")


def pick_throttle_field(index, attempts=3):
    """First throttle reason query name this nvidia-smi accepts, or None.

    Each candidate is probed more than once. A single transient nvidia-smi failure here
    would otherwise downgrade the whole 20 minute run to no throttle reasons and throw
    away the card's own account of why it slowed down, which is the primary evidence for
    Part E, leaving only the lagging clock heuristic.
    """
    for field in THROTTLE_FIELD_CANDIDATES:
        for attempt in range(attempts):
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits",
                     "-i", str(index)],
                    capture_output=True, text=True, check=True, timeout=10)
                value = out.stdout.strip()
                if value and not any(m in value.lower() for m in _UNKNOWN_FIELD_MARKERS):
                    return field
                break
            except subprocess.CalledProcessError as exc:
                text = f"{exc.stdout or ''} {exc.stderr or ''}".lower()
                if any(m in text for m in _UNKNOWN_FIELD_MARKERS):
                    break
                if attempt == attempts - 1:
                    print(f"  '{field}' failed {attempts} times: "
                          f"{(exc.stderr or '').strip()[:120]}", flush=True)
            except Exception as exc:
                if attempt == attempts - 1:
                    print(f"  '{field}' failed {attempts} times: {exc}", flush=True)
    return None


def sampler(stop_event, index, path, interval, t0, throttle_field):
    """Poll nvidia-smi every interval seconds until told to stop."""
    fields = [f for f in SAMPLE_FIELDS
              if f not in ("timestamp", "clocks_throttle_reasons.active")]
    if throttle_field:
        fields.append(throttle_field)
    query = ",".join(fields)
    # The header keeps the canonical column name whatever the driver calls the field, so
    # the builders parse one schema.
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["elapsed_s"] + SAMPLE_FIELDS)
        while not stop_event.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}",
                     "--format=csv,noheader,nounits", "-i", str(index)],
                    capture_output=True, text=True, check=True, timeout=10)
                values = [v.strip() for v in out.stdout.strip().split(",")]
            except Exception as exc:
                # One value per queried field, so an error row still has exactly the
                # width of the header. A ragged row would break the whole log. Keep the
                # message short so the log stays readable.
                detail = str(exc).strip().replace("\n", " ")
                if len(detail) > 80:
                    detail = f"{type(exc).__name__}: {detail[:77]}..."
                values = [f"error:{detail}"] * len(fields)
            if not throttle_field:
                values.append("unavailable")
            writer.writerow([round(time.time() - t0, 2),
                             time.strftime("%Y-%m-%d %H:%M:%S")] + values)
            fh.flush()
            stop_event.wait(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=8192, help="matmul size for the load")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--chunk", type=int, default=20, help="matmuls per throughput sample")
    args = ap.parse_args()

    common.require_cuda()
    common.seed_everything()
    torch.cuda.set_device(args.index)
    identity = common.gpu_identity(args.index)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    common.log_header("PART E sustained load and thermal behaviour", identity,
                      extra=f"minutes={args.minutes} interval={args.interval}s "
                            f"load=matmul N={args.n} {args.dtype}")

    tag = identity["uuid"][-12:]
    sample_path = os.path.join(common.LOGS, f"part_e_thermal_{tag}.csv")
    tput_path = os.path.join(common.DATA, f"part_e_throughput_{tag}.csv")

    a = torch.randn(args.n, args.n, device="cuda", dtype=dtype)
    b = torch.randn(args.n, args.n, device="cuda", dtype=dtype)
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    t0 = time.time()
    stop = threading.Event()
    throttle_field = pick_throttle_field(args.index)
    if throttle_field is None:
        common.log("E", "WARNING no throttle reason field accepted by this nvidia-smi. "
                        "Throttling will be inferred from the clock trace alone",
                   identity["uuid"])
    else:
        common.log("E", f"sampling throttle reasons via '{throttle_field}'",
                   identity["uuid"])
    thread = threading.Thread(target=sampler,
                              args=(stop, args.index, sample_path, args.interval, t0,
                                    throttle_field),
                              daemon=True)
    thread.start()

    flops_per_matmul = 2.0 * args.n ** 3
    tput_rows = []
    deadline = t0 + args.minutes * 60.0
    try:
        while time.time() < deadline:
            chunk_start = time.time()
            for _ in range(args.chunk):
                torch.matmul(a, b)
            torch.cuda.synchronize()
            chunk_end = time.time()
            tput_rows.append({
                "uuid": identity["uuid"], "gpu_name": identity["gpu_name"],
                "elapsed_s": round(chunk_end - t0, 2),
                "matmuls": args.chunk, "dtype": args.dtype, "n": args.n,
                "tflops": round(flops_per_matmul * args.chunk
                                / (chunk_end - chunk_start) / 1e12, 3),
            })
    finally:
        stop.set()
        thread.join(timeout=15)

    common.write_csv(tput_path, tput_rows)
    common.log("E", f"thermal log written to {os.path.relpath(sample_path, common.ROOT)} "
                    f"({args.interval}s sampling)", identity["uuid"])

    total = tput_rows[-1]["elapsed_s"] if tput_rows else 0.0
    # The assignment asks for the final 5 minutes. A run cut short must not still be
    # described as a 5 minute window, so clamp it and say which it was.
    window_s = min(300.0, total)
    short = total < 600.0
    first30 = [r["tflops"] for r in tput_rows if r["elapsed_s"] <= 30.0]
    final5 = [r["tflops"] for r in tput_rows if r["elapsed_s"] >= total - window_s]
    if first30 and final5:
        peak = max(first30)
        steady = sum(final5) / len(final5)
        label = (f"final {window_s / 60:.1f} min of a {total / 60:.1f} min run, SHORT, "
                 f"not the 20 minutes the assignment asks for"
                 if short else "final 5 min")
        if short:
            common.log("E", f"WARNING run lasted only {total:.0f} s, so the steady state "
                            f"window is the final {window_s:.0f} s, not 5 minutes. Rerun "
                            f"with --minutes 20 before submitting.", identity["uuid"])
        common.log("E", f"peak throughput (first 30 s) = {peak:.2f} TFLOPS, steady state "
                        f"({label}, n={len(final5)} samples) = {steady:.2f} TFLOPS, "
                        f"steady over peak = {steady / peak * 100:.1f}%", identity["uuid"])
        import json
        with open(os.path.join(common.DATA, f"part_e_summary_{tag}.json"), "w") as fh:
            json.dump({"uuid": identity["uuid"], "gpu_name": identity["gpu_name"],
                       "duration_s": total, "peak_first30_tflops": peak,
                       "steady_final5min_tflops": steady,
                       "steady_window_s": window_s, "run_too_short": short,
                       "steady_over_peak_pct": steady / peak * 100.0,
                       "load": f"matmul N={args.n} {args.dtype}"}, fh, indent=2)
    else:
        common.log("E", "not enough samples for the peak and steady comparison, run longer",
                   identity["uuid"])


if __name__ == "__main__":
    main()
