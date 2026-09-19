"""Shared helpers: step 0 parameters, GPU identity, CUDA timing, and RUN_LOG.txt."""
import csv
import datetime as _dt
import json
import os
import platform
import subprocess
import sys

try:
    import torch
except ImportError:
    torch = None

# Step 0, same convention as assignment 1. Student ID 018205330, so SID4 is 5330.
SID4 = int(os.environ.get("SID4", "5330"))
SEED = int(os.environ.get("SEED", str(SID4)))

# HW25_ROOT lets rehearse.py drive the chain into a scratch tree. Unset in a real run.
ROOT = os.environ.get("HW25_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGURES = os.path.join(ROOT, "figures")
LOGS = os.path.join(ROOT, "logs")
PROVENANCE = os.path.join(ROOT, "provenance")
RUN_LOG = os.path.join(ROOT, "RUN_LOG.txt")
SPECS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_specs.json")

for _d in (DATA, FIGURES, LOGS, PROVENANCE):
    os.makedirs(_d, exist_ok=True)


def seed_everything(seed=SEED):
    import random

    random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_cuda():
    if torch is None:
        sys.exit("PyTorch is not installed. Parts B to E need torch with CUDA.")
    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible. Parts B to E must run on the reserved "
                 "RTX 4090 or RTX 5090 workstation, not on a laptop.")


def nvidia_smi(query, index=0):
    """One query-gpu field as a string, or empty if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits",
             "-i", str(index)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def gpu_identity(index=0):
    """Everything that labels a measurement: UUID, name, driver, CUDA, VRAM, power cap."""
    props = (torch.cuda.get_device_properties(index)
             if (torch is not None and torch.cuda.is_available()) else None)
    return {
        "uuid": nvidia_smi("uuid", index) or "UUID-UNAVAILABLE",
        "gpu_name": (nvidia_smi("name", index) or (props.name if props else "unknown")),
        "driver_version": nvidia_smi("driver_version", index),
        "cuda_version": (torch.version.cuda or "") if torch else "",
        "vram_total_mib": nvidia_smi("memory.total", index),
        "power_limit_w": nvidia_smi("power.limit", index),
        "torch_version": torch.__version__ if torch else "",
        "host": platform.node(),
    }


def load_specs(gpu_name):
    """Vendor reference row for this card, matched loosely on the reported name."""
    with open(SPECS) as fh:
        specs = json.load(fh)
    for key, row in specs.items():
        if key.startswith("_"):
            continue
        if key.lower() in gpu_name.lower() or gpu_name.lower() in key.lower():
            return key, row
    for key, row in specs.items():
        if key.startswith("_"):
            continue
        tail = key.split()[-1]
        if tail.lower() in gpu_name.lower():
            return key, row
    return None, None


def is_oom(exc):
    """True for every way this stack reports running out of memory.

    torch.cuda.OutOfMemoryError only covers the caching allocator. At the OOM boundary
    the failure often comes from cuBLAS workspace allocation instead, as a plain
    RuntimeError. Treating that as a crash would abort the Part D bisection at the point
    it is trying to measure.
    """
    if torch is not None and isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in (
        "out of memory", "cublas_status_alloc_failed", "cuda error: out of memory",
        "alloc_failed", "cudaerrormemoryallocation",
        # torch.AcceleratorError is a RuntimeError subclass, so it is already caught by
        # the handlers. What was missing was recognising this message as a failure to
        # record rather than a crash to re raise. It surfaced on the lab run straight
        # after two very large failed allocations. It is not itself an out of memory
        # condition, so callers must follow it with cuda_context_alive() and stop if the
        # context did not survive.
        "device not ready", "cudaerrornotready",
    ))


def cuda_context_alive():
    """Force pending async CUDA errors to surface. False if the context is unusable.

    A failed allocation can leave errors queued that only appear at the next
    synchronize. Without this check a search would keep probing a dead context and
    record every later probe as a memory failure, producing a boundary that is an
    artifact of the crash rather than a measurement.
    """
    if torch is None:
        return False
    try:
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def vram_bytes(index=0):
    """(free, total) physical VRAM as the driver reports it, or (None, None).

    torch.cuda.max_memory_allocated reports the allocator's own bookkeeping. On a setup
    that silently spills past the card into host memory, that number can exceed what the
    card physically has, and no exception is raised. The driver figure from
    torch.cuda.mem_get_info is what a result has to be checked against.
    """
    if torch is None:
        return None, None
    try:
        free, total = torch.cuda.mem_get_info(index)
        return int(free), int(total)
    except Exception:
        return None, None


def environment_report(index=0):
    """Facts that decide whether a memory measurement can be believed."""
    free, total = vram_bytes(index)
    wsl = "unknown"
    try:
        with open("/proc/version") as fh:
            version = fh.read().lower()
        wsl = "yes" if ("microsoft" in version or "wsl" in version) else "no"
    except Exception:
        pass
    return {
        "driver_vram_total_gib": round(total / 2 ** 30, 3) if total else "",
        "driver_vram_free_gib": round(free / 2 ** 30, 3) if free else "",
        "nvidia_smi_memory_total_mib": nvidia_smi("memory.total", index),
        "running_under_wsl": wsl,
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "unset"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "platform": platform.platform(),
    }


# nvidia-smi throttle reason bits. GpuIdle and ApplicationsClocksSetting are not
# throttling, so they are never treated as an onset. Shared by the table and the figure
# builders so the two cannot disagree about whether the card throttled.
THROTTLE_BITS = [
    (0x0000000000000004, "SwPowerCap"),
    (0x0000000000000008, "HwSlowdown"),
    (0x0000000000000020, "SwThermalSlowdown"),
    (0x0000000000000040, "HwThermalSlowdown"),
    (0x0000000000000080, "HwPowerBrakeSlowdown"),
]


def decode_throttle_reasons(value):
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


def time_cuda(fn, warmup=10, iters=50):
    """Median and mean seconds per call, timed with CUDA events after a warmup.

    Returns (mean_s, median_s, stdev_s, iters), where stdev is the sample standard
    deviation over the timed repetitions. Every caller records both reps and warmup in
    its CSV, so the repetition count is visible in the writeup as Part B requires.

    The warmup runs first and is followed by a synchronize, so cuBLAS algorithm
    selection, lazy module loading and the clock ramp are all paid before the first
    timed iteration. CUDA events measure on device time only, so the host side
    synchronize below does not inflate the reported number.
    """
    import statistics

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
    return (
        statistics.fmean(samples),
        statistics.median(samples),
        statistics.stdev(samples) if len(samples) > 1 else 0.0,
        iters,
    )


def log(part, message, uuid=None):
    """Append a UUID labelled line to RUN_LOG.txt and echo it."""
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] [{part}] [{uuid or 'no uuid'}] {message}"
    with open(RUN_LOG, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def log_header(part, identity, extra=""):
    with open(RUN_LOG, "a") as fh:
        fh.write(f"\n{part} | SID4={SID4} SEED={SEED}\n")
        for key, value in identity.items():
            fh.write(f"  {key}: {value}\n")
        if extra:
            fh.write(f"  {extra}\n")
        fh.write("\n")
    print(f"{part} on {identity['gpu_name']} ({identity['uuid']})", flush=True)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        print(f"nothing to write to {path}")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))
