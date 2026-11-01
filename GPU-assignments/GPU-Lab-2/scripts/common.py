"""Shared setup for HW4.a and HW4.5: IDs, data, model, training loop, GPU helpers."""
import gc, json, os, platform, subprocess, time
import numpy as np
import torch
from torch.utils.data import TensorDataset

# Student ID 018205330, same step 0 convention as earlier assignments
SID4 = 5330
SEED = SID4
SLICE = SID4 % 1000          # 330
SMOKE = os.environ.get("SMOKE") == "1"   # tiny chain check, never submit
CARD = os.environ.get("CARD", "card")    # output label, e.g. rtx5090 or colab_t4

# HW4 fine tune settings, used unchanged as the HW4.5 baseline
BERT = "bert-base-uncased"
MAX_LEN, BATCH, LR = 64, 32, 2e-5
EPOCHS = 1 if SMOKE else 4
LLM = "Qwen/Qwen2.5-0.5B-Instruct"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", CARD)
FIG = os.path.join(ROOT, "figures")
for d in (OUT, FIG, os.path.join(ROOT, "provenance")):
    os.makedirs(d, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# bf16 needs Ampere or newer; a T4 falls back to fp16 with a GradScaler
AMP_DTYPE = torch.float16 if DEV == "cuda" and torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16


def seed_all():
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_data():
    """Rows SLICE to SLICE+5000 of the Twitter set, split 70/15/15 with SEED."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizer
    csv = os.path.join(ROOT, "data", "twitter.csv")   # optional override: columns text,label
    if os.path.exists(csv):
        df = pd.read_csv(csv)[["text", "label"]]
        names = sorted(df.label.astype(str).unique())
        df["label"] = df.label.astype(str).map(names.index)
    else:
        from datasets import load_dataset
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")
        df, names = ds.to_pandas(), ds.features["label"].names
    start, end = SLICE, min(SLICE + (300 if SMOKE else 5000), len(df))
    sub = df.iloc[start:end].reset_index(drop=True)
    tr, rest = train_test_split(sub, test_size=0.30, random_state=SEED, stratify=sub.label)
    va, te = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=rest.label)
    tok = BertTokenizer.from_pretrained(BERT)

    def tensors(part):
        enc = tok(list(part.text), padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(part.label.values))

    info = dict(dataset="data/twitter.csv" if os.path.exists(csv) else "cardiffnlp/tweet_eval (sentiment, train split)", rows=f"{start} to {end - 1}", n_rows=end - start, n_total=len(df),
                n_train=len(tr), n_val=len(va), n_test=len(te), classes=names)
    return tensors(tr), tensors(va), tensors(te), list(te.text), names, info


def build_model(n_labels, ckpt=False):
    from transformers import BertForSequenceClassification
    seed_all()
    m = BertForSequenceClassification.from_pretrained(BERT, num_labels=n_labels)
    if ckpt:
        m.gradient_checkpointing_enable()
    return m.to(DEV).train()   # from_pretrained returns eval mode, which disables checkpointing and dropout


def make_opt(model, opt8=False):
    if opt8:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(model.parameters(), lr=LR), "AdamW8bit"
        except Exception:   # no bitsandbytes build for this card: memory efficient Adafactor instead
            from transformers.optimization import Adafactor
            return Adafactor(model.parameters(), lr=LR, relative_step=False, scale_parameter=False), "Adafactor"
    return torch.optim.AdamW(model.parameters(), lr=LR), "AdamW"


def make_scaler(amp):
    return torch.amp.GradScaler("cuda", enabled=amp and AMP_DTYPE == torch.float16 and DEV == "cuda")


def fwd_bwd(model, batch, amp, scaler, scale=1.0):
    ids, mask, y = (t.to(DEV) for t in batch)
    with torch.autocast(DEV, dtype=AMP_DTYPE, enabled=amp):
        loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
    scaler.scale(loss * scale).backward()
    return loss.item()


def opt_step(opt, scaler):
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)


def train_epoch(model, loader, opt, scaler, amp=False, accum=1, max_steps=None):
    """One pass over loader. Returns mean loss and optimiser steps taken."""
    model.train()
    opt.zero_grad(set_to_none=True)
    losses, steps = [], 0
    for i, b in enumerate(loader):
        losses.append(fwd_bwd(model, b, amp, scaler, 1 / accum))
        if (i + 1) % accum == 0:
            opt_step(opt, scaler)
            steps += 1
            if max_steps and steps >= max_steps:
                break
    return float(np.mean(losses)), steps


@torch.no_grad()
def evaluate(model, loader, amp=False):
    model.eval()
    losses, preds, ys = [], [], []
    for ids, mask, y in loader:
        with torch.autocast(DEV, dtype=AMP_DTYPE, enabled=amp):
            out = model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV), labels=y.to(DEV))
        losses.append(out.loss.item())
        preds += out.logits.argmax(-1).tolist()
        ys += y.tolist()
    acc = float(np.mean(np.array(preds) == np.array(ys)))
    return float(np.mean(losses)), acc, preds, ys


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


def reset_peak():
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else 0.0


def free():
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


def is_oom(e):
    return isinstance(e, torch.OutOfMemoryError) or any(k in str(e).lower() for k in ("out of memory", "alloc_failed"))


def probe(train_ds, n_labels, amp=False, ckpt=False, opt8=False, start=1, steps=5):
    """Double the batch from start until OOM. Throughput is timed over steps after 2 warmup steps."""
    rows, b, cap = [], start, 64 if SMOKE else 2 ** 16
    while b <= cap:
        model = opt = None
        try:
            model = build_model(n_labels, ckpt)
            opt, _ = make_opt(model, opt8)
            scaler, g = make_scaler(amp), torch.Generator().manual_seed(SEED)
            reset_peak()
            for s in range(steps + 2):
                if s == 2:
                    sync()
                    t = time.time()
                fwd_bwd(model, train_ds[torch.randint(len(train_ds), (b,), generator=g)], amp, scaler)
                opt_step(opt, scaler)
            sync()
            thr = steps * b / (time.time() - t)
            rows.append(dict(batch=b, samples_per_s=thr, peak_gb=peak_gb(), epoch_s=len(train_ds) / thr, oom=False))
            print(rows[-1], flush=True)
        except Exception as e:
            if not is_oom(e):
                raise
            rows.append(dict(batch=b, oom=True))
            print(f"batch {b}: OOM", flush=True)
            break
        finally:
            model = opt = None
            free()
        b *= 2
    return rows


def gpu_info():
    def smi(q):
        try:
            return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader", "-i", "0"],
                                  capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    if DEV != "cuda":
        return dict(name=f"{platform.machine()} {DEV}", uuid="no GPU (smoke)", vram_gb=0)
    p = torch.cuda.get_device_properties(0)
    return dict(name=p.name, uuid=smi("uuid"), vram_gb=round(p.total_memory / 1e9, 1),
                driver=smi("driver_version"), cuda=torch.version.cuda, torch=torch.__version__)


def save(name, obj):
    obj = dict(card=CARD, gpu=gpu_info(), smoke=SMOKE, **obj)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    with open(os.path.join(ROOT, "RUN_LOG.txt"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {CARD} | {obj['gpu']['uuid']} | {name}"
                f"{' | SMOKE' if SMOKE else ''} | SID4={SID4} SEED={SEED} SLICE={SLICE}\n")
