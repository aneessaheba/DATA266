"""HW4.5 Part C: DDP throughput, scaling and communication overhead.

torchrun --nproc_per_node=1 part_c_ddp.py   single GPU baseline
torchrun --nproc_per_node=2 part_c_ddp.py   the two GPU pairing
Per step overhead = step time with gradient all reduce minus step time under no_sync.
"""
import os, subprocess, time
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from common import *

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
local = int(os.environ["LOCAL_RANK"])
dist.init_process_group("nccl" if DEV == "cuda" else "gloo")
dev = f"cuda:{local}" if DEV == "cuda" else "cpu"
if DEV == "cuda":
    torch.cuda.set_device(local)
tr, _, _, _, names, _ = load_data()
STEPS = 3 if SMOKE else 20


def timed(model, opt, b, nosync):
    g = torch.Generator().manual_seed(SEED + rank)
    for s in range(STEPS + 3):
        if s == 3:
            sync()
            t = time.time()
        ids, mask, y = (x.to(dev) for x in tr[torch.randint(len(tr), (b,), generator=g)])
        if nosync:
            with model.no_sync():
                model(input_ids=ids, attention_mask=mask, labels=y).loss.backward()
        else:
            model(input_ids=ids, attention_mask=mask, labels=y).loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    sync()
    return (time.time() - t) / STEPS


rows = []
for b in ([8] if SMOKE else [8, 16, 32, 64, 128]):
    from transformers import BertForSequenceClassification
    torch.manual_seed(SEED)
    model = DDP(BertForSequenceClassification.from_pretrained(BERT, num_labels=len(names)).to(dev),
                device_ids=[local] if DEV == "cuda" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    step = timed(model, opt, b, False)
    compute = timed(model, opt, b, True)
    rows.append(dict(batch_per_gpu=b, global_batch=b * world, step_s=step, compute_s=compute,
                     comm_overhead_s=step - compute, samples_per_s=b * world / step))
    if rank == 0:
        print(rows[-1], flush=True)
    del model, opt
    free()

# standalone all reduce of one gradient sized buffer gives the effective bandwidth
n_bytes = sum(p.numel() for p in BertForSequenceClassification.from_pretrained(BERT, num_labels=len(names)).parameters()) * 4
buf = torch.ones(n_bytes // 4, device=dev)
for _ in range(3):
    dist.all_reduce(buf)
sync()
t = time.time()
for _ in range(10):
    dist.all_reduce(buf)
sync()
ar = (time.time() - t) / 10
bus_gbs = 2 * (world - 1) / world * n_bytes / ar / 1e9 if world > 1 else None

if rank == 0:
    topo = subprocess.run("nvidia-smi topo -m", shell=True, capture_output=True, text=True).stdout
    save(f"part_c_n{world}", dict(world=world, nodes=1, gpus=[torch.cuda.get_device_name(i) for i in range(world)]
                                  if DEV == "cuda" else [DEV], grad_mb=n_bytes / 1e6, allreduce_s=ar,
                                  bus_bandwidth_gbs=bus_gbs, topo=topo, rows=rows))
dist.destroy_process_group()
