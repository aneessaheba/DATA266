"""Part B: dense matmul throughput at four sizes in FP32, TF32, FP16 and BF16, plus an
FP8 attempt whose failure is itself a reported finding."""
import argparse
import os

import torch

import common

SIZES = [1024, 4096, 8192, 16384]


def run_matmul(n, precision, warmup, iters):
    """One size and precision. Returns timing and achieved TFLOPS."""
    torch.backends.cuda.matmul.allow_tf32 = (precision == "tf32")
    torch.backends.cudnn.allow_tf32 = (precision == "tf32")
    dtype = {"fp32": torch.float32, "tf32": torch.float32,
             "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    mean_s, median_s, std_s, reps = common.time_cuda(lambda: torch.matmul(a, b),
                                                     warmup=warmup, iters=iters)
    flops = 2.0 * n ** 3
    del a, b
    torch.cuda.empty_cache()
    return {
        "tflops": flops / median_s / 1e12,
        "median_s": median_s,
        "mean_s": mean_s,
        "stdev_s": std_s,
        "cv_pct": (std_s / mean_s * 100.0) if mean_s else 0.0,
        "reps": reps,
    }


def run_fp8(n, warmup, iters):
    """torch._scaled_mm on FP8 E4M3. Returns (result, error_string)."""
    try:
        dtype = torch.float8_e4m3fn
    except AttributeError as exc:
        return None, f"torch build exposes no float8_e4m3fn: {exc}"
    try:
        a = torch.randn(n, n, device="cuda").to(dtype)
        # _scaled_mm wants the second operand column major.
        b = torch.randn(n, n, device="cuda").to(dtype).t().contiguous().t()
        scale = torch.tensor(1.0, device="cuda")

        def call():
            return torch._scaled_mm(a, b, scale_a=scale, scale_b=scale,
                                    out_dtype=torch.bfloat16)

        call()
        mean_s, median_s, std_s, reps = common.time_cuda(call, warmup=warmup, iters=iters)
        flops = 2.0 * n ** 3
        del a, b
        torch.cuda.empty_cache()
        return {"tflops": flops / median_s / 1e12, "median_s": median_s, "mean_s": mean_s,
                "stdev_s": std_s, "cv_pct": (std_s / mean_s * 100.0) if mean_s else 0.0,
                "reps": reps}, ""
    except Exception as exc:
        # Tooling maturity is a legitimate finding, so record it rather than crash.
        torch.cuda.empty_cache()
        return None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--skip-fp8", action="store_true")
    args = ap.parse_args()

    common.require_cuda()
    common.seed_everything()
    torch.cuda.set_device(args.index)
    identity = common.gpu_identity(args.index)
    _, specs = common.load_specs(identity["gpu_name"])
    peaks = (specs or {}).get("peak_tflops_dense", {})
    common.log_header("PART B precision and achieved throughput", identity,
                      extra=f"warmup={args.warmup} timed_reps={args.iters}")

    rows = []
    for n in args.sizes:
        for precision in ("fp32", "tf32", "fp16", "bf16"):
            try:
                res = run_matmul(n, precision, args.warmup, args.iters)
            except RuntimeError as exc:
                if not common.is_oom(exc):
                    raise
                torch.cuda.empty_cache()
                common.log("B", f"N={n} {precision}: OOM ({exc})", identity["uuid"])
                continue
            peak = peaks.get(precision)
            pct = (res["tflops"] / peak * 100.0) if peak else None
            rows.append({
                "uuid": identity["uuid"], "gpu_name": identity["gpu_name"], "n": n,
                "precision": precision, "achieved_tflops": round(res["tflops"], 3),
                "theoretical_peak_tflops": peak,
                "pct_of_peak": round(pct, 2) if pct is not None else "",
                "median_s": f"{res['median_s']:.6e}", "mean_s": f"{res['mean_s']:.6e}",
                "stdev_s": f"{res['stdev_s']:.6e}", "cv_pct": round(res["cv_pct"], 3),
                "reps": res["reps"], "warmup": args.warmup,
            })
            common.log("B", f"N={n:>5} {precision:<4} {res['tflops']:8.2f} TFLOPS"
                            + (f" ({pct:5.1f}% of {peak} peak)" if pct is not None else "")
                            + f" median={res['median_s']*1e3:.3f} ms reps={res['reps']}"
                            + f" cv={res['cv_pct']:.2f}%", identity["uuid"])

        if not args.skip_fp8:
            res, err = run_fp8(n, args.warmup, args.iters)
            if res is None:
                common.log("B", f"N={n} fp8: NOT AVAILABLE. {err}", identity["uuid"])
                rows.append({"uuid": identity["uuid"], "gpu_name": identity["gpu_name"],
                             "n": n, "precision": "fp8", "achieved_tflops": "",
                             "theoretical_peak_tflops": peaks.get("fp8"), "pct_of_peak": "",
                             "median_s": "", "mean_s": "", "stdev_s": "", "cv_pct": "",
                             "reps": 0, "warmup": args.warmup, "note": err})
            else:
                peak = peaks.get("fp8")
                pct = (res["tflops"] / peak * 100.0) if peak else None
                rows.append({"uuid": identity["uuid"], "gpu_name": identity["gpu_name"],
                             "n": n, "precision": "fp8",
                             "achieved_tflops": round(res["tflops"], 3),
                             "theoretical_peak_tflops": peak,
                             "pct_of_peak": round(pct, 2) if pct is not None else "",
                             "median_s": f"{res['median_s']:.6e}",
                             "mean_s": f"{res['mean_s']:.6e}",
                             "stdev_s": f"{res['stdev_s']:.6e}",
                             "cv_pct": round(res["cv_pct"], 3), "reps": res["reps"],
                             "warmup": args.warmup})
                common.log("B", f"N={n:>5} fp8  {res['tflops']:8.2f} TFLOPS"
                                + (f" ({pct:5.1f}% of peak)" if pct is not None else ""),
                           identity["uuid"])

    fields = ["uuid", "gpu_name", "n", "precision", "achieved_tflops",
              "theoretical_peak_tflops", "pct_of_peak", "median_s", "mean_s", "stdev_s",
              "cv_pct", "reps", "warmup", "note"]
    for row in rows:
        row.setdefault("note", "")
    common.write_csv(os.path.join(common.DATA, f"part_b_{identity['uuid'][-12:]}.csv"),
                     rows, fields)


if __name__ == "__main__":
    main()
