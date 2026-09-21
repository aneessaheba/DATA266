"""Part C: one memory bound op and one compute bound op, with effective bandwidth,
arithmetic intensity, and which side of the roofline each lands on."""
import argparse
import os

import torch

import common


def memory_bound_add(n_elements, dtype, warmup, iters):
    """c = a + b. Two streaming reads and one write, no reuse."""
    a = torch.randn(n_elements, device="cuda", dtype=dtype)
    b = torch.randn(n_elements, device="cuda", dtype=dtype)
    c = torch.empty_like(a)
    mean_s, median_s, std_s, reps = common.time_cuda(
        lambda: torch.add(a, b, out=c), warmup=warmup, iters=iters)
    itemsize = a.element_size()
    bytes_moved = 3.0 * n_elements * itemsize
    flops = 1.0 * n_elements
    del a, b, c
    torch.cuda.empty_cache()
    return {"op": "elementwise_add", "bytes": bytes_moved, "flops": flops,
            "median_s": median_s, "mean_s": mean_s, "stdev_s": std_s, "reps": reps}


def memory_bound_copy(n_elements, dtype, warmup, iters):
    """Device to device copy. One read and one write."""
    a = torch.randn(n_elements, device="cuda", dtype=dtype)
    c = torch.empty_like(a)
    mean_s, median_s, std_s, reps = common.time_cuda(
        lambda: c.copy_(a), warmup=warmup, iters=iters)
    itemsize = a.element_size()
    bytes_moved = 2.0 * n_elements * itemsize
    del a, c
    torch.cuda.empty_cache()
    return {"op": "device_copy", "bytes": bytes_moved, "flops": 0.0,
            "median_s": median_s, "mean_s": mean_s, "stdev_s": std_s, "reps": reps}


def compute_bound_matmul(n, dtype, warmup, iters):
    """Square matmul. O(n^3) work over O(n^2) traffic."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    mean_s, median_s, std_s, reps = common.time_cuda(
        lambda: torch.matmul(a, b), warmup=warmup, iters=iters)
    itemsize = a.element_size()
    bytes_moved = 3.0 * n * n * itemsize
    flops = 2.0 * n ** 3
    del a, b
    torch.cuda.empty_cache()
    return {"op": f"matmul_n{n}", "bytes": bytes_moved, "flops": flops,
            "median_s": median_s, "mean_s": mean_s, "stdev_s": std_s, "reps": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--elements", type=int, default=1 << 28, help="elements per vector op")
    ap.add_argument("--matmul-n", type=int, default=8192)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    common.require_cuda()
    common.seed_everything()
    torch.cuda.set_device(args.index)
    identity = common.gpu_identity(args.index)
    _, specs = common.load_specs(identity["gpu_name"])
    spec_bw = (specs or {}).get("bandwidth_gb_s")
    peaks = (specs or {}).get("peak_tflops_dense", {})
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    torch.backends.cuda.matmul.allow_tf32 = False
    common.log_header("PART C bandwidth bound against compute bound", identity,
                      extra=f"dtype={args.dtype} elements={args.elements} "
                            f"matmul_n={args.matmul_n} reps={args.iters}")

    # Ridge point: where the roofline turns from bandwidth limited to compute limited.
    peak_flops = peaks.get(args.dtype if args.dtype != "fp32" else "fp32")
    ridge = (peak_flops * 1e12) / (spec_bw * 1e9) if (peak_flops and spec_bw) else None
    if ridge:
        common.log("C", f"roofline ridge point = {ridge:.1f} FLOPs per byte "
                        f"({peak_flops} TFLOPS / {spec_bw} GB/s)", identity["uuid"])

    results = [
        memory_bound_add(args.elements, dtype, args.warmup, args.iters),
        memory_bound_copy(args.elements, dtype, args.warmup, args.iters),
        compute_bound_matmul(args.matmul_n, dtype, args.warmup, args.iters),
    ]

    rows = []
    for res in results:
        eff_bw = res["bytes"] / res["median_s"] / 1e9
        achieved_tflops = res["flops"] / res["median_s"] / 1e12
        intensity = res["flops"] / res["bytes"] if res["bytes"] else 0.0
        side = ("memory bound" if (ridge and intensity < ridge)
                else "compute bound" if ridge else "unknown, no vendor spec")
        rows.append({
            "uuid": identity["uuid"], "gpu_name": identity["gpu_name"], "op": res["op"],
            "dtype": args.dtype, "bytes_moved": int(res["bytes"]), "flops": int(res["flops"]),
            "median_s": f"{res['median_s']:.6e}",
            "effective_bandwidth_gb_s": round(eff_bw, 2),
            "spec_bandwidth_gb_s": spec_bw or "",
            "pct_of_spec_bandwidth": round(eff_bw / spec_bw * 100.0, 2) if spec_bw else "",
            "achieved_tflops": round(achieved_tflops, 3),
            "arithmetic_intensity_flops_per_byte": round(intensity, 4),
            "ridge_point_flops_per_byte": round(ridge, 2) if ridge else "",
            "roofline_side": side, "reps": res["reps"], "warmup": args.warmup,
        })
        common.log("C", f"{res['op']:<16} eff_bw={eff_bw:8.1f} GB/s"
                        + (f" ({eff_bw/spec_bw*100:5.1f}% of spec)" if spec_bw else "")
                        + f" AI={intensity:.4f} FLOPs per byte, {side}", identity["uuid"])

    common.write_csv(os.path.join(common.DATA, f"part_c_{identity['uuid'][-12:]}.csv"), rows)


if __name__ == "__main__":
    main()
