"""A fake CUDA device, so Parts A to E can be run and checked without a GPU.

The part scripts call common.require_cuda() and then allocate on device="cuda". This
stands a device underneath them so the real code runs end to end on a laptop:

* torch.cuda.is_available() and friends are stubbed out.
* Every device="cuda" allocation is rewritten to device="meta". Meta tensors carry real
  shapes and dtypes but own no storage, so a 16384 square matmul or a 32 GiB attention
  score matrix costs nothing and takes no time.
* A TorchDispatchMode charges each fresh tensor against a simulated VRAM budget and
  raises a genuine torch.cuda.OutOfMemoryError when it is exceeded. That is what makes
  the Part D OOM search real rather than a mock.
* The same mode counts work, and the fake torch.cuda.Event turns it into plausible
  elapsed times, including a per launch overhead, so small matrices come out
  inefficient for the same reason they do on real hardware.
* nvidia-smi is intercepted at subprocess.run, covering the Part A dump and the Part E
  sampling loop.

None of this produces a measurement. Run it from rehearse.py.

    python scripts/fakecuda.py --card 4090 -- scripts/part_d_attention.py --index 0
"""
import argparse
import os
import random
import subprocess
import sys
import time
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# Simulated cards. VRAM is what the budget enforces. The peaks and bandwidth only shape
# the synthetic timings.
CARDS = {
    "4090": {
        "name": "NVIDIA GeForce RTX 4090",
        "uuid": "GPU-fa4e0000-0000-4000-8000-4090feedface",
        "vram_mib": 24564, "power_limit_w": 450.0, "driver": "560.35.03",
        "sim_peaks": {"fp32": 82.6, "tf32": 82.6, "fp16": 165.2, "bf16": 165.2, "fp8": 330.3},
        "sim_bw_gb_s": 1008.0, "sm_clock": 2745, "mem_clock": 10501,
        # This card answers to the old query name.
        "throttle_field": "clocks_throttle_reasons.active", "throttles_at_s": 8.0,
    },
    "5090": {
        "name": "NVIDIA GeForce RTX 5090",
        "uuid": "GPU-b1ac0000-0000-5000-8000-5090cafebabe",
        "vram_mib": 32607, "power_limit_w": 575.0, "driver": "570.86.10",
        "sim_peaks": {"fp32": 104.8, "tf32": 104.8, "fp16": 209.5, "bf16": 209.5, "fp8": 419.0},
        "sim_bw_gb_s": 1792.0, "sm_clock": 2640, "mem_clock": 14000,
        # This one only answers to the new name, exercising the fallback in part_e.
        "throttle_field": "clocks_event_reasons.active", "throttles_at_s": None,
    },
}

LAUNCH_OVERHEAD_S = 3.0e-5
SYNC_SLEEP_S = 2.0e-3


class Vram:
    """A simulated allocator: current and peak bytes, and a ceiling that raises OOMs."""

    def __init__(self, total_bytes):
        self.total = total_bytes
        self.current = 0
        self.peak = 0
        # Seconds of simulated compute, accumulated per op at that op's own dtype rate,
        # so FP32 cannot come out faster than the FP32 peak.
        self.compute_s = 0.0
        self.bytes_moved = 0.0
        self.launches = 0

    def charge(self, nbytes, tensor):
        if self.current + nbytes > self.total:
            need = nbytes / 2 ** 30
            free = (self.total - self.current) / 2 ** 30
            raise torch.cuda.OutOfMemoryError(
                f"CUDA out of memory. Tried to allocate {need:.2f} GiB. GPU 0 has a total "
                f"capacity of {self.total / 2 ** 30:.2f} GiB of which {free:.2f} GiB is "
                f"free. [simulated by fakecuda.py]")
        self.current += nbytes
        self.peak = max(self.peak, self.current)
        weakref.finalize(tensor, self._release, nbytes)

    def _release(self, nbytes):
        self.current = max(0, self.current - nbytes)


VRAM = Vram(24 * 2 ** 30)

_MATMUL_NAMES = ("mm", "bmm", "matmul", "addmm", "baddbmm", "_scaled_mm", "linear")
_ATTENTION_HINT = "attention"


def _tensors(obj, out):
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _tensors(item, out)
    elif isinstance(obj, dict):
        for item in obj.values():
            _tensors(item, out)
    return out


def _matmul_flops(name, ins, outs):
    """2*M*N*K, inferred from the output shape and the shared dimension."""
    if not outs or outs[0].dim() < 2:
        return 0.0
    out = outs[0]
    k = 0
    for t in ins:
        if t.dim() >= 2 and t.shape[-2] == out.shape[-2]:
            k = max(k, t.shape[-1])
    if not k:
        k = max((t.shape[-1] for t in ins if t.dim() >= 1), default=0)
    return 2.0 * out.numel() * k


class SimulatedCuda(TorchDispatchMode):
    """Charges every fresh allocation to the budget and tallies work for the timer."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        ins = _tensors(args, []) + _tensors(kwargs, [])
        out = func(*args, **kwargs)
        outs = _tensors(out, [])

        input_ids = {id(t) for t in ins}
        for t in outs:
            # Skip views, which share storage with their base, and in place results,
            # where the op returned one of its own inputs. Both would double count.
            if id(t) in input_ids or t._base is not None:
                continue
            VRAM.charge(t.numel() * t.element_size(), t)

        name = str(func).split(".")[-1].rstrip("'>").split("(")[0]
        full = str(func)
        flops = 0.0
        if any(m in full for m in _MATMUL_NAMES):
            flops = _matmul_flops(name, ins, outs)
        elif _ATTENTION_HINT in full and ins:
            q = ins[0]
            if q.dim() == 4:
                b, h, s, d = q.shape
                flops = 4.0 * b * h * s * s * d
        if flops:
            VRAM.compute_s += flops / (_peak_for(ins, outs) * 1e12)
        VRAM.bytes_moved += sum(t.numel() * t.element_size() for t in ins + outs)
        VRAM.launches += 1
        return out


class FakeEvent:
    """Stands in for torch.cuda.Event, timing simulated work between records."""

    def __init__(self, enable_timing=False, **_):
        self.compute_s = self.bytes = self.launches = 0

    def record(self, *_, **__):
        self.compute_s = VRAM.compute_s
        self.bytes = VRAM.bytes_moved
        self.launches = VRAM.launches

    def synchronize(self):
        pass

    def query(self):
        return True

    def elapsed_time(self, other):
        """Milliseconds, whichever of compute and memory dominates, plus launch cost."""
        compute_s = max(other.compute_s - self.compute_s, 0.0)
        byts = max(other.bytes - self.bytes, 0.0)
        launches = max(other.launches - self.launches, 0)
        memory_s = byts / (_SIM["bw"] * 1e9)
        total = max(compute_s, memory_s) + launches * LAUNCH_OVERHEAD_S
        # A little jitter, so the coefficient of variation columns get real values
        # instead of a suspicious row of exact zeros.
        return total * random.uniform(0.995, 1.005) * 1e3


_SIM = {"peaks": {"fp32": 82.6, "tf32": 82.6, "fp16": 165.2, "bf16": 165.2, "fp8": 330.3},
        "bw": 1008.0}


def _peak_for(ins, outs):
    """The card's peak for the dtype this op actually ran in.

    Without this the simulator would clock an FP32 matmul at the tensor core rate and
    report it as 193% of the FP32 peak, which looks like a bug in the part scripts
    rather than in the fake timer.
    """
    dtypes = {t.dtype for t in ins} or {t.dtype for t in outs}
    peaks = _SIM["peaks"]
    if torch.float8_e4m3fn in dtypes or torch.float8_e5m2 in dtypes:
        return peaks.get("fp8", peaks["fp16"])
    if torch.float16 in dtypes:
        return peaks["fp16"]
    if torch.bfloat16 in dtypes:
        return peaks["bf16"]
    # FP32 inputs run at the TF32 rate only when the stack has been told they may.
    if getattr(torch.backends.cuda.matmul, "allow_tf32", False):
        return peaks["tf32"]
    return peaks["fp32"]


_REAL_SUBPROCESS_RUN = subprocess.run


def _smi_query_value(field, card, elapsed):
    """One query-gpu field, varying over time so Part E gets a realistic trace."""
    hot = card["throttles_at_s"] is not None and elapsed >= card["throttles_at_s"]
    table = {
        "uuid": card["uuid"],
        "name": card["name"],
        "driver_version": card["driver"],
        "memory.total": str(card["vram_mib"]),
        "power.limit": f"{card['power_limit_w']:.2f}",
        "clocks.current.sm": str(int(card["sm_clock"] * (0.93 if hot else 1.0))),
        "clocks.current.memory": str(card["mem_clock"]),
        "temperature.gpu": str(int(min(38 + elapsed * 2.2, 84))),
        "power.draw": f"{card['power_limit_w'] * (0.99 if hot else 0.87):.2f}",
        "utilization.gpu": "100",
        "utilization.memory": str(random.randint(40, 60)),
    }
    if field in (card["throttle_field"],):
        return "0x0000000000000004" if hot else "0x0000000000000000"
    return table.get(field, "N/A")


def _fake_smi(cmd, kwargs, card, t0, fail_every):
    query = next((a for a in cmd if a.startswith("--query-gpu=")), None)
    elapsed = time.time() - t0

    if query is None:
        body = [
            "NVSMI LOG", "",
            f"Timestamp                                 : {time.ctime()}",
            f"Driver Version                            : {card['driver']}",
            "CUDA Version                              : 12.8", "",
            "Attached GPUs                             : 1",
            "GPU 00000000:01:00.0",
            f"    Product Name                          : {card['name']}",
            f"    GPU UUID                              : {card['uuid']}",
            "    FB Memory Usage",
            f"        Total                             : {card['vram_mib']} MiB",
            "    Power Readings",
            f"        Power Limit                       : {card['power_limit_w']:.2f} W",
            "", "GENERATED BY fakecuda.py, NOT A REAL CAPTURE", "",
        ]
        return subprocess.CompletedProcess(cmd, 0, "\n".join(body) + "\n", "")

    fields = query.split("=", 1)[1].split(",")
    # A driver that does not know a field errors the whole query. That is how part_e
    # decides which throttle reason name to use, so it has to be modelled.
    for field in fields:
        if field.endswith("reasons.active") and field != card["throttle_field"]:
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(
                    6, cmd, "", f"Field \"{field}\" is not a valid field to query.\n")
            return subprocess.CompletedProcess(cmd, 6, "", "invalid field")

    # Occasionally fail a sample outright, so the sampler's error path and the row width
    # it writes are exercised rather than assumed.
    if fail_every and random.randint(1, fail_every) == 1:
        if kwargs.get("check"):
            raise subprocess.CalledProcessError(9, cmd, "", "simulated nvidia-smi failure")
        return subprocess.CompletedProcess(cmd, 9, "", "simulated failure")

    values = [_smi_query_value(f, card, elapsed) for f in fields]
    return subprocess.CompletedProcess(cmd, 0, ", ".join(values) + "\n", "")


def install(card_key="4090", vram_gb=None, smi_fail_every=0, break_fp8=False):
    """Stand up the fake device. Returns the card in use."""
    card = CARDS[card_key]
    total = int((vram_gb or card["vram_mib"] / 1024) * 2 ** 30)
    global VRAM
    VRAM = Vram(total)
    _SIM["peaks"], _SIM["bw"] = card["sim_peaks"], card["sim_bw_gb_s"]
    t0 = time.time()

    class Props:
        name = card["name"]
        total_memory = total
        major, minor = 8, 9
        multi_processor_count = 128

    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.get_device_properties = lambda *a, **k: Props()
    torch.cuda.get_device_name = lambda *a, **k: card["name"]
    torch.cuda.manual_seed_all = lambda *a, **k: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: setattr(VRAM, "peak", VRAM.current)
    torch.cuda.max_memory_allocated = lambda *a, **k: VRAM.peak
    torch.cuda.memory_allocated = lambda *a, **k: VRAM.current
    torch.cuda.synchronize = lambda *a, **k: time.sleep(SYNC_SLEEP_S)
    torch.cuda.Event = FakeEvent

    def rewrite(device):
        if device is None:
            return None
        if isinstance(device, torch.device):
            return torch.device("meta") if device.type == "cuda" else device
        if isinstance(device, str) and device.startswith("cuda"):
            return "meta"
        if isinstance(device, int):
            return "meta"
        return device

    def wrap_factory(fn):
        def wrapped(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = rewrite(kwargs["device"])
            return fn(*args, **kwargs)
        return wrapped

    for fname in ("randn", "rand", "zeros", "ones", "empty", "full", "tensor",
                  "arange", "eye", "empty_like", "zeros_like", "ones_like"):
        if hasattr(torch, fname):
            setattr(torch, fname, wrap_factory(getattr(torch, fname)))

    real_to = torch.Tensor.to

    def fake_to(self, *args, **kwargs):
        args = tuple(rewrite(a) if isinstance(a, (torch.device,)) or
                     (isinstance(a, str) and a.startswith("cuda")) else a for a in args)
        if "device" in kwargs:
            kwargs["device"] = rewrite(kwargs["device"])
        return real_to(self, *args, **kwargs)

    torch.Tensor.to = fake_to
    torch.Tensor.cuda = lambda self, *a, **k: real_to(self, "meta")

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and cmd and str(cmd[0]).endswith("nvidia-smi"):
            return _fake_smi(list(cmd), kwargs, card, t0, smi_fail_every)
        return _REAL_SUBPROCESS_RUN(cmd, *args, **kwargs)

    subprocess.run = fake_run

    if break_fp8:
        # Part B has to cope with a stack that will not run the lower precision at all.
        # That branch is a required finding, so the rehearsal needs to reach it.
        def no_fp8(*a, **k):
            raise RuntimeError(
                "torch._scaled_mm is not supported on this platform [simulated]")
        torch._scaled_mm = no_fp8

    mode = SimulatedCuda()
    mode.__enter__()
    return card


def main():
    ap = argparse.ArgumentParser(description="Run a part script on a fake CUDA device.")
    ap.add_argument("--card", default="4090", choices=sorted(CARDS))
    ap.add_argument("--vram-gb", type=float, default=None,
                    help="shrink the simulated VRAM to force an earlier OOM boundary")
    ap.add_argument("--smi-fail-every", type=int, default=0,
                    help="fail 1 in N nvidia-smi samples, to exercise the error path")
    ap.add_argument("--break-fp8", action="store_true",
                    help="make torch._scaled_mm raise, to exercise Part B's failure branch")
    ap.add_argument("target", help="the part script to run")
    ap.add_argument("target_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    install(args.card, args.vram_gb, args.smi_fail_every, args.break_fp8)

    import runpy
    sys.argv = [args.target] + [a for a in args.target_args if a != "--"]
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.target)))
    runpy.run_path(args.target, run_name="__main__")


if __name__ == "__main__":
    main()
