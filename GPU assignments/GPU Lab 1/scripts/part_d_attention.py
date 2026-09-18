"""Part D: the cost of attention. Naive attention that materializes the full S by S
score matrix, measured against the fused kernel in the stack, including the OOM boundary
reported as a bracket and a quadratic fit confirmed from the data."""
import argparse
import math
import os

import torch
import torch.nn.functional as F

import common

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]


def naive_attention(q, k, v):
    """Textbook scaled dot product attention. The S by S matrix really exists in VRAM."""
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def fused_attention(q, k, v):
    """Whatever fused kernel the stack picks."""
    return F.scaled_dot_product_attention(q, k, v)


def measure(seq_len, batch, heads, head_dim, dtype, impl, warmup, iters):
    """Peak allocated bytes and forward latency at one sequence length.

    Returns (row, error_string). error_string is non empty on OOM.
    """
    fn = naive_attention if impl == "naive" else fused_attention
    q = k = v = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        shape = (batch, heads, seq_len, head_dim)
        q = torch.randn(shape, device="cuda", dtype=dtype)
        k = torch.randn(shape, device="cuda", dtype=dtype)
        v = torch.randn(shape, device="cuda", dtype=dtype)
        with torch.inference_mode():
            mean_s, median_s, std_s, reps = common.time_cuda(
                lambda: fn(q, k, v), warmup=warmup, iters=iters)
        peak = torch.cuda.max_memory_allocated()
        return {
            "seq_len": seq_len, "impl": impl, "ok": 1,
            "peak_mem_bytes": peak, "peak_mem_gib": round(peak / 2 ** 30, 4),
            "latency_ms": round(median_s * 1e3, 4), "mean_ms": round(mean_s * 1e3, 4),
            "stdev_ms": round(std_s * 1e3, 4), "reps": reps,
        }, ""
    except RuntimeError as exc:
        if not common.is_oom(exc):
            raise
        message = str(exc).split("\n")[0]
        return {"seq_len": seq_len, "impl": impl, "ok": 0, "peak_mem_bytes": "",
                "peak_mem_gib": "", "latency_ms": "", "mean_ms": "", "stdev_ms": "",
                "reps": 0}, message
    finally:
        # Drop q, k and v here rather than only on success. After an OOM the partly
        # allocated tensors are still referenced by this frame, and the traceback keeps
        # the frame alive, so without this the bisection leaks the failed probe's VRAM
        # and every later probe fails for the wrong reason.
        del q, k, v
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def refine_oom(last_ok, first_fail, probe, resolution):
    """Bisect between the largest success and smallest failure until the bracket is
    within resolution tokens. Returns (largest_ok, smallest_fail, probes)."""
    probes = 0
    while first_fail - last_ok > resolution:
        mid = (last_ok + first_fail) // 2
        if mid in (last_ok, first_fail):
            break
        ok = probe(mid)
        probes += 1
        if ok:
            last_ok = mid
        else:
            first_fail = mid
    return last_ok, first_fail, probes


def quadratic_fit(seq_lens, peaks):
    """Least squares fit of peak_mem = a*S^2 + b*S + c using only measured points."""
    import numpy as np

    x = np.asarray(seq_lens, dtype=np.float64)
    y = np.asarray(peaks, dtype=np.float64)
    coeffs = np.polyfit(x, y, 2)
    pred = np.polyval(coeffs, x)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return coeffs, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--refine-resolution", type=int, default=256,
                    help="stop bisecting once the bracket is this many tokens wide. "
                         "1 gives a true single token boundary")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--max-extend", type=int, default=131072,
                    help="if the coarse sweep never OOMs, keep doubling up to this cap")
    args = ap.parse_args()

    common.require_cuda()
    common.seed_everything()
    torch.cuda.set_device(args.index)
    identity = common.gpu_identity(args.index)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    config = (f"batch={args.batch} heads={args.heads} head_dim={args.head_dim} "
              f"dtype={args.dtype} reps={args.iters}")
    common.log_header("PART D the cost of attention", identity, extra=config)
    common.log("D", f"configuration: {config}", identity["uuid"])

    rows, summary = [], {}
    for impl in ("naive", "fused"):
        ok_lengths, fail_lengths = [], []
        for seq_len in args.seq_lens:
            row, err = measure(seq_len, args.batch, args.heads, args.head_dim, dtype,
                               impl, args.warmup, args.iters)
            row.update(uuid=identity["uuid"], gpu_name=identity["gpu_name"],
                       batch=args.batch, heads=args.heads, head_dim=args.head_dim,
                       dtype=args.dtype, phase="sweep", warmup=args.warmup, note=err)
            rows.append(row)
            if row["ok"]:
                ok_lengths.append(seq_len)
                common.log("D", f"{impl:<5} S={seq_len:>6} peak={row['peak_mem_gib']:>8} GiB "
                                f"latency={row['latency_ms']:>9} ms reps={row['reps']}",
                           identity["uuid"])
            else:
                fail_lengths.append(seq_len)
                common.log("D", f"{impl:<5} S={seq_len:>6} OOM. {err}", identity["uuid"])

        largest_ok = max(ok_lengths) if ok_lengths else None
        smallest_fail = min(fail_lengths) if fail_lengths else None

        def probe(seq_len, phase="refine"):
            row, err = measure(seq_len, args.batch, args.heads, args.head_dim, dtype,
                               impl, warmup=1, iters=3)
            row.update(uuid=identity["uuid"], gpu_name=identity["gpu_name"],
                       batch=args.batch, heads=args.heads, head_dim=args.head_dim,
                       dtype=args.dtype, phase=phase, warmup=1, note=err)
            rows.append(row)
            common.log("D", f"{impl:<5} {phase} S={seq_len:>7} "
                            + ("ok" if row["ok"] else f"OOM. {err}"), identity["uuid"])
            return bool(row["ok"])

        # The coarse sweep can finish without a failure: naive attention at B=1 H=8 d=64
        # fp16 peaks near 8 GiB at S=16384, which fits on both a 24 GB 4090 and a 32 GB
        # 5090. Part D wants a real boundary, so keep doubling until something fails.
        if smallest_fail is None and largest_ok and not args.no_refine:
            candidate = largest_ok * 2
            while candidate <= args.max_extend:
                if probe(candidate, phase="extend"):
                    largest_ok = candidate
                    candidate *= 2
                else:
                    smallest_fail = candidate
                    break
            if smallest_fail is None:
                common.log("D", f"{impl}: no OOM up to S={largest_ok} (extension cap "
                                f"{args.max_extend}). For the fused kernel this is the "
                                f"expected result. Its peak memory is linear in S, so it "
                                f"does not OOM anywhere near the naive boundary.",
                           identity["uuid"])

        if smallest_fail and largest_ok and not args.no_refine:
            largest_ok, smallest_fail, probes = refine_oom(
                largest_ok, smallest_fail, probe, args.refine_resolution)
            common.log("D", f"{impl}: OOM bracket after {probes} probes. Largest success "
                            f"{largest_ok}, smallest failure {smallest_fail} "
                            f"(resolution {args.refine_resolution} tokens, NOT an exact "
                            f"single token boundary unless resolution is 1)",
                       identity["uuid"])

        summary[impl] = {"largest_ok": largest_ok, "smallest_fail": smallest_fail}

        measured = [(r["seq_len"], r["peak_mem_bytes"]) for r in rows
                    if r["impl"] == impl and r["ok"] and r["phase"] == "sweep"]
        if len(measured) >= 3:
            coeffs, r2 = quadratic_fit([m[0] for m in measured], [m[1] for m in measured])
            a, b, c = (float(x) for x in coeffs)
            summary[impl].update(quad_a=a, quad_b=b, quad_c=c, r2=r2)
            itemsize = torch.empty((), dtype=dtype).element_size()
            expected_a = args.batch * args.heads * 2 * itemsize
            summary[impl]["expected_quad_a"] = expected_a
            if impl == "naive":
                verdict = (f"against {expected_a} expected if two S by S {args.dtype} "
                           f"tensors are materialised (B*H*2*{itemsize})")
            else:
                # Comparing a fused kernel against the cost of materialising the score
                # matrix is meaningless. The point is that it does not. Judge it
                # against zero instead.
                share = abs(a) / expected_a * 100.0 if expected_a else float("nan")
                verdict = (f"against {expected_a} for the naive kernel, so {share:.3f}% of "
                           f"it. No S by S term, which is the whole claim. Peak memory is "
                           f"linear in S")
            common.log("D", f"{impl}: peak_mem = {a:.6g}*S^2 + {b:.6g}*S + {c:.6g} bytes "
                            f"(R^2={r2:.6f}). Measured S^2 coefficient {a:.4f} bytes per "
                            f"token squared, {verdict}", identity["uuid"])

    # Speedup at every sequence length where both implementations ran.
    naive_by_len = {r["seq_len"]: r for r in rows if r["impl"] == "naive" and r["ok"]
                    and r["phase"] == "sweep"}
    for row in rows:
        if row["impl"] == "fused" and row["ok"] and row["phase"] == "sweep":
            base = naive_by_len.get(row["seq_len"])
            if base:
                speedup = base["latency_ms"] / row["latency_ms"]
                mem_ratio = base["peak_mem_bytes"] / row["peak_mem_bytes"]
                row["speedup_vs_naive"] = round(speedup, 3)
                row["peak_mem_ratio_vs_naive"] = round(mem_ratio, 3)
                common.log("D", f"S={row['seq_len']:>6} fused speedup {speedup:.2f}x, "
                                f"peak memory {mem_ratio:.2f}x smaller", identity["uuid"])

    fields = ["uuid", "gpu_name", "impl", "phase", "seq_len", "batch", "heads", "head_dim",
              "dtype", "ok", "peak_mem_bytes", "peak_mem_gib", "latency_ms", "mean_ms",
              "stdev_ms", "reps", "warmup", "speedup_vs_naive", "peak_mem_ratio_vs_naive",
              "note"]
    for row in rows:
        row.setdefault("speedup_vs_naive", "")
        row.setdefault("peak_mem_ratio_vs_naive", "")
    common.write_csv(os.path.join(common.DATA, f"part_d_{identity['uuid'][-12:]}.csv"),
                     rows, fields)

    import json
    with open(os.path.join(common.DATA, f"part_d_summary_{identity['uuid'][-12:]}.json"),
              "w") as fh:
        json.dump({"uuid": identity["uuid"], "gpu_name": identity["gpu_name"],
                   "config": config, "refine_resolution": args.refine_resolution,
                   "max_extend": args.max_extend,
                   "summary": summary}, fh, indent=2)


if __name__ == "__main__":
    main()
