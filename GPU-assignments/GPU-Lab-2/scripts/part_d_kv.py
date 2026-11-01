"""HW4.5 Part D: KV cache memory against context length at batch 8, and the longest context
that fits. Prefill runs on the base model (no LM head) so logits do not hide the KV cost.
KV memory is measured as allocated memory held by the cache after prefill."""
import torch
from transformers import AutoModel, AutoConfig
from common import *

if DEV != "cuda":
    raise SystemExit("Part D KV sweep needs a CUDA GPU")

B = 8
cfg = AutoConfig.from_pretrained(LLM)
head_dim = cfg.hidden_size // cfg.num_attention_heads
kv_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2   # K and V, fp16
model = AutoModel.from_pretrained(LLM, dtype=torch.float16).to(DEV).eval()


def try_len(L):
    kv = None
    try:
        free()
        before = torch.cuda.memory_allocated()
        reset_peak()
        with torch.no_grad():
            kv = model(torch.randint(1000, 20000, (B, L), device=DEV), use_cache=True).past_key_values
        sync()
        r = dict(context=L, kv_gb=(torch.cuda.memory_allocated() - before) / 1e9,
                 kv_formula_gb=B * L * kv_per_token / 1e9, peak_gb=peak_gb(), oom=False)
    except Exception as e:
        if not is_oom(e):
            raise
        r = dict(context=L, oom=True)
    kv = None
    free()
    print(r, flush=True)
    return r


rows, L = [], 512
while L <= 2 ** 20:
    rows.append(try_len(L))
    if rows[-1]["oom"]:
        break
    L *= 2
ok, bad = L // 2, L
for _ in range(3):   # bisect the bracket three times
    mid = (ok + bad) // 2
    rows.append(try_len(mid))
    ok, bad = (ok, mid) if rows[-1]["oom"] else (mid, bad)
save("part_d_kv", dict(model=LLM, batch=B, kv_bytes_per_token=kv_per_token, max_position=cfg.max_position_embeddings,
                       longest_ok=ok, shortest_fail=bad, rows=sorted(rows, key=lambda r: r["context"])))
