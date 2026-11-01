"""HW4.5 Part B: each memory technique alone, then the best combination.

Max batch comes from the doubling probe. Throughput, peak memory and accuracy come from a full
HW4 length fine tune at batch 32. Accumulation runs physical 16 x 8 = effective 128 for the same
number of optimiser steps as the baseline.
"""
import json, time
import torch
from torch.utils.data import DataLoader
from common import *

tr, va, te, _, names, info = load_data()
base_max = json.load(open(os.path.join(OUT, "part_a.json")))["max_batch"]
steps_base = EPOCHS * (len(tr) // BATCH)
CONFIGS = {
    "Baseline (HW4 settings)": dict(),
    "Mixed precision": dict(amp=True),
    "Gradient checkpointing": dict(ckpt=True),
    "Gradient accumulation": dict(accum=8, phys=16),
    "8-bit optimiser": dict(opt8=True),
    "Best combination": dict(amp=True, opt8=True),
}
results = []
for name, c in CONFIGS.items():
    amp, ckpt, opt8, accum = c.get("amp", False), c.get("ckpt", False), c.get("opt8", False), c.get("accum", 1)
    phys = c.get("phys", BATCH)
    if name == "Baseline (HW4 settings)" or accum > 1:
        max_b = base_max     # accumulation leaves per step memory unchanged
    else:
        rows = probe(tr, len(names), amp, ckpt, opt8, start=BATCH)
        max_b = max([r["batch"] for r in rows if not r["oom"]], default=0)
    model = build_model(len(names), ckpt)
    opt, opt_name = make_opt(model, opt8)
    scaler, g = make_scaler(amp), torch.Generator().manual_seed(SEED)
    dl = DataLoader(tr, batch_size=phys, shuffle=True, generator=g, drop_last=True)
    steps = 0
    reset_peak()
    sync()
    t = time.time()
    while steps < steps_base:
        steps += train_epoch(model, dl, opt, scaler, amp, accum, steps_base - steps)[1]
    sync()
    wall = time.time() - t
    _, acc, _, _ = evaluate(model, DataLoader(te, batch_size=128), amp)
    results.append(dict(technique=name, optimizer=opt_name, max_batch=max_b, effective_batch=phys * accum,
                        opt_steps=steps, wall_s=wall, samples_per_s=steps * phys * accum / wall,
                        peak_gb=peak_gb(), test_acc=acc))
    print(results[-1], flush=True)
    model = opt = None
    free()
save("part_b", dict(amp_dtype=str(AMP_DTYPE), rows=results))
