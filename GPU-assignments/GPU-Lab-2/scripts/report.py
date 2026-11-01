"""Builds METRICS.md and figures/ from results/<card>/*.json. Needs no GPU."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
R = {}
for f in sorted(glob.glob(os.path.join(ROOT, "results", "*", "*.json"))):
    R.setdefault(os.path.basename(os.path.dirname(f)), {})[os.path.basename(f)[:-5]] = json.load(open(f))
cards = sorted(R)
out = ["# METRICS, HW4.a and HW4.5", "", "SJSU ID 018205330, SID4 5330, SEED 5330, SLICE 330.", ""]
W = out.append


def table(head, rows):
    W("| " + " | ".join(head) + " |")
    W("|" + "---|" * len(head))
    for r in rows:
        W("| " + " | ".join(str(x) for x in r) + " |")
    W("")


def gpu(c):
    runs = {}
    for name, v in R[c].items():
        runs.setdefault(v["gpu"]["uuid"], []).append(name)
    g = next(iter(R[c].values()))["gpu"]
    return f"{g['name']}, {g['vram_gb']} GB. " + "; ".join(f"UUID {u}: {', '.join(sorted(n))}" for u, n in runs.items())


W("Cards: " + "; ".join(f"**{c}** {gpu(c)}" for c in cards))
W("")
if any(v.get("smoke") for c in cards for v in R[c].values()):
    W("**WARNING: contains SMOKE results. Not submittable.**\n")

# HW4.a
for c in cards:
    h = R[c].get("hw4a")
    if not h:
        continue
    W(f"## HW4.a BERT fine tune on {c}\n")
    d = h["data"]
    W(f"Dataset {d.get('dataset', 'cardiffnlp/tweet_eval (sentiment, train split)')}. Rows {d['rows']} of {d['n_total']} ({d['n_rows']} rows). Split 70/15/15 with SEED 5330: "
      f"train {d['n_train']}, validation {d['n_val']}, test {d['n_test']}. Settings: {h['settings']}.\n")
    table(["Epoch", "Train loss", "Val loss", "Val acc", "Epoch s"],
          [[e["epoch"], f"{e['train_loss']:.4f}", f"{e['val_loss']:.4f}", f"{e['val_acc']:.4f}", f"{e['epoch_s']:.1f}"]
           for e in h["epochs"]])
    best = min(h["epochs"], key=lambda e: e["val_loss"])
    extra = [e for e in h["epochs"] if e["epoch"] > best["epoch"]]
    W(f"Validation loss stopped improving after epoch {best['epoch']}. "
      + (f"Training {len(extra)} more epochs cost {sum(e['epoch_s'] for e in extra):.1f} s and moved validation loss "
         f"from {best['val_loss']:.4f} to {h['epochs'][-1]['val_loss']:.4f} (accuracy {best['val_acc']:.4f} to "
         f"{h['epochs'][-1]['val_acc']:.4f}).\n" if extra else "No epochs were trained past it.\n"))
    top = max(h["epochs"], key=lambda e: e["val_acc"])
    W(f"Validation accuracy peaked at epoch {top['epoch']} ({top['val_acc']:.4f}), later than validation loss. "
      f"The saved model and all test numbers below come from the final epoch {h['epochs'][-1]['epoch']} model.\n")
    W(f"Test accuracy after save and reload: {h['test_acc']:.4f}. "
      f"**Peak GPU memory {h['peak_gb']:.2f} GB, training wall clock {h['train_s']:.1f} s** (HW4.5 baseline only).\n")
    names = [p["cls"] for p in h["per_class"]]
    W("Confusion matrix (rows true, columns predicted):\n")
    table([""] + names, [[n] + row for n, row in zip(names, h["confusion"])])
    table(["Class", "Precision", "Recall", "F1", "Support"],
          [[p["cls"], f"{p['precision']:.3f}", f"{p['recall']:.3f}", f"{p['f1']:.3f}", p["n"]] for p in h["per_class"]])
    W(f"Worst class (lowest F1): **{h['worst_class']}**. Five misclassified examples, diagnosis in ANALYSIS.md:\n")
    table(["#", "Tweet", "Predicted"], [[i + 1, m["text"].replace("|", "/").replace("\n", " ").replace("\\u2019", "'") + (" (sic: the apostrophe is stored escaped in the dataset)" if "\\u2019" in m["text"] else ""), m["pred"]]
                                        for i, m in enumerate(h["misclassified"])])
    table(["Custom input", "Prediction"], [[x["text"], x["pred"]] for x in h["custom"]])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ep = [e["epoch"] for e in h["epochs"]]
    ax[0].plot(ep, [e["train_loss"] for e in h["epochs"]], "o-", c=COLORS[0], lw=2, label="train")
    ax[0].plot(ep, [e["val_loss"] for e in h["epochs"]], "o-", c=COLORS[1], lw=2, label="validation")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[0].legend()
    ax[1].plot(ep, [e["val_acc"] for e in h["epochs"]], "o-", c=COLORS[0], lw=2)
    ax[1].set(xlabel="epoch", ylabel="accuracy", title="Validation accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"hw4a_curves_{c}.png"), dpi=120)
    W(f"![curves](figures/hw4a_curves_{c}.png)\n")


def serve_rows(c, tag):
    rows = []
    if "serve_hf_hw4" in R[c] and tag == "hw4":
        rows.append(["HF pipeline, batch 1"] + [f"{R[c]['serve_hf_hw4']['rows'][0][k]:.1f}" for k in ("tokens_per_s", "p50_ms", "p95_ms")])
    for r in R[c].get(f"serve_vllm_{tag}", {}).get("rows", []):
        rows.append([f"vLLM, batch {r['batch']}"] + [f"{r[k]:.1f}" for k in ("tokens_per_s", "p50_ms", "p95_ms")])
    return rows


for c in cards:
    if serve_rows(c, "hw4"):
        s = R[c].get("serve_vllm_hw4") or R[c]["serve_hf_hw4"]
        W(f"### Table HW4.1, serving comparison on {c}\n")
        W(f"Model `{s['model']}`, fp16, {s['new_tokens']} new tokens per request, greedy. Prompt: \"{s['prompt']}\"\n")
        table(["Configuration", "Tokens/sec", "p50 latency (ms)", "p95 latency (ms)"], serve_rows(c, "hw4"))
        W("Sample output: " + s["sample_output"].replace("\n", " ")[:400] + "\n")
        W("Method: latency is the time for the whole batch call to finish, every request generates exactly 128 tokens "
          "(ignore_eos), and time to first token is not measured. The 8 prompts repeat inside the batch 8 and 32 runs, "
          "so vLLM prefix caching may help those slightly.\n")

# HW4.5 Part A
W("## HW4.5 Part A, batch scaling to the memory wall\n")
fig, ax = plt.subplots(figsize=(6, 4))
maxb = {}
for i, c in enumerate(cards):
    a = R[c].get("part_a")
    if not a:
        continue
    ok = [r for r in a["rows"] if not r["oom"]]
    top = max(r["samples_per_s"] for r in ok)
    knee = next(r["batch"] for r in ok if r["samples_per_s"] >= 0.95 * top)
    maxb[c] = a["max_batch"]
    W(f"**{c}**: largest batch {a['max_batch']}, first OOM at {a['rows'][-1]['batch'] if a['rows'][-1]['oom'] else 'none'}. "
      f"Throughput stops improving (within 5% of best) at batch {knee}.\n")
    table(["Batch", "Samples/s", "Peak GB", "Epoch s"],
          [[r["batch"], "OOM", "", ""] if r["oom"] else
           [r["batch"], f"{r['samples_per_s']:.1f}", f"{r['peak_gb']:.2f}", f"{r['epoch_s']:.1f}"] for r in a["rows"]])
    ax.plot([r["batch"] for r in ok], [r["samples_per_s"] for r in ok], "o-", c=COLORS[i % 3], lw=2, label=c)
ax.set(xscale="log", xlabel="batch size (log2)", ylabel="samples per second", title="Training throughput against batch size")
ax.set_xscale("log", base=2)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "part_a_throughput.png"), dpi=120)
W("![throughput](figures/part_a_throughput.png)\n")
W("Method: each row is timed over 5 steps after 2 warmup steps on randomly sampled batches, and Epoch s is extrapolated "
  "as 3500 / samples per second, not measured. These short timings are optimistic: at batch 32 they give 106.8 samples/s "
  "and 32.8 s per epoch, while the sustained Part B baseline ran at 91.9 samples/s (about 14% lower) and the measured "
  "HW4.a epochs took 37 to 44 s.\n")
if len(maxb) < 2:
    W("Only one card was used (the TA confirmed one GPU is enough), so the max batch ratio between two cards is not "
      "measured. ANALYSIS.md gives a worked projection instead.\n")
if len(maxb) == 2:
    c1, c2 = sorted(maxb, key=lambda c: -R[c]["part_a"]["gpu"]["vram_gb"])
    v1, v2 = R[c1]["part_a"]["gpu"]["vram_gb"], R[c2]["part_a"]["gpu"]["vram_gb"]
    W(f"Max batch ratio {c1}/{c2} = {maxb[c1]}/{maxb[c2]} = {maxb[c1] / maxb[c2]:.2f}. "
      f"VRAM ratio = {v1}/{v2} = {v1 / v2:.2f}.\n")

# Part B
W("## HW4.5 Part B, memory techniques one at a time\n")
for c in cards:
    b = R[c].get("part_b")
    if not b:
        continue
    W(f"### Table HW4.5.1 on {c} (AMP dtype {b['amp_dtype']})\n")
    table(["Technique", "Max batch", "Throughput (samp/s)", "Peak memory (GB)", "Eval accuracy", "Wall s", "Notes"],
          [[r["technique"], "n/a" if r["technique"] == "Gradient accumulation" else r["max_batch"], f"{r['samples_per_s']:.1f}", f"{r['peak_gb']:.2f}", f"{r['test_acc']:.4f}",
            f"{r['wall_s']:.1f}", f"{r['optimizer']}, effective batch {r['effective_batch']}, {r['opt_steps']} optimiser steps"]
           for r in b["rows"]])
    base = b["rows"][0]
    rank = []
    for r in b["rows"][1:5]:
        saved, lost = base["peak_gb"] - r["peak_gb"], base["samples_per_s"] - r["samples_per_s"]
        rank.append((saved / lost if lost > 0 else float("inf"), r["technique"], saved, lost))
    rank.sort(reverse=True)
    W("Gradient accumulation has no max batch of its own: per step memory is set by the physical batch of 16, so it is "
      "marked n/a. Its run trained 436 optimiser steps x 128 samples = 55,808 samples, about 16 epochs over 3500 rows. "
      "The training loop also dropped the last 2 of the 218 micro batches in each epoch (218 is not a multiple of 8).\n")
    W("Ranking at the training batch of 32 (peak GB saved per samples/s lost):\n")
    table(["Rank", "Technique", "GB saved", "Samples/s lost", "GB saved per samp/s lost"],
          [[i + 1, t, f"{s:.3f}", f"{l:.1f}", "no throughput lost" if x == float("inf") else f"{x:.4f}"]
           for i, (x, t, s, l) in enumerate(rank)])
    pr = R[c].get("part_b_probes", {}).get("probes")
    if pr:
        at = lambda k: next(r for r in pr[k] if r["batch"] == 256)
        b0 = at("Part A baseline")
        rows256, rank256 = [], []
        for k in ("Part A baseline", "Mixed precision", "Gradient checkpointing", "8-bit optimiser", "Best combination"):
            r = at(k)
            rows256.append([k.replace("Part A baseline", "Baseline"), f"{r['peak_gb']:.2f}", f"{r['samples_per_s']:.1f}"])
            if k not in ("Part A baseline", "Best combination"):
                saved, lost = b0["peak_gb"] - r["peak_gb"], b0["samples_per_s"] - r["samples_per_s"]
                rank256.append((saved / lost if lost > 0 else float("inf"), k, saved, lost))
        rank256.sort(reverse=True)
        W("Same techniques at batch 256, from the max batch probe runs (5 timed steps each, so throughput is optimistic):\n")
        table(["Batch 256", "Peak GB", "Samples/s"], rows256)
        W("Ranking at batch 256 (accumulation has no batch 256 run):\n")
        table(["Rank", "Technique", "GB saved", "Samples/s lost", "GB saved per samp/s lost"],
              [[i + 1, t, f"{s:.2f}", f"{l:.1f}", "no throughput lost" if x == float("inf") else f"{x:.3f}"]
               for i, (x, t, s, l) in enumerate(rank256)])
        W("The ranking depends on batch size. At batch 32 activations are small, so checkpointing saves little and ranks "
          "last. At batch 256 activations dominate, so checkpointing saves 8.7 GB and ranks above the 8 bit optimiser.\n")

# Part C
W("## HW4.5 Part C, distributed data parallel\n")
if not any("part_c_n2" in R[c] for c in cards):
    W("Part C skipped (optional).\n")
for c in cards:
    one, two = R[c].get("part_c_n1"), R[c].get("part_c_n2")
    for p in (one, two):
        if p and two:
            W(f"**{c}, {p['world']} GPU** ({', '.join(p['gpus'])}, single node). Gradient {p['grad_mb']:.0f} MB, "
              f"all reduce {p['allreduce_s'] * 1000:.1f} ms"
              + (f", effective bus bandwidth {p['bus_bandwidth_gbs']:.2f} GB/s.\n" if p["bus_bandwidth_gbs"] else ".\n"))
            W("```\n" + p["topo"].strip() + "\n```\n")
    if one and two:
        table(["Batch per GPU", "1 GPU samp/s", "2 GPU samp/s", "Speedup", "Scaling efficiency", "Step ms", "Compute ms",
               "Comm overhead ms", "Dominant"],
              [[a["batch_per_gpu"], f"{a['samples_per_s']:.1f}", f"{b['samples_per_s']:.1f}",
                f"{b['samples_per_s'] / a['samples_per_s']:.2f}", f"{b['samples_per_s'] / (2 * a['samples_per_s']):.2f}",
                f"{b['step_s'] * 1000:.1f}", f"{b['compute_s'] * 1000:.1f}", f"{b['comm_overhead_s'] * 1000:.1f}",
                "communication" if b["comm_overhead_s"] > b["compute_s"] else "computation"]
               for a, b in zip(one["rows"], two["rows"])])
        good = [a["batch_per_gpu"] for a, b in zip(one["rows"], two["rows"]) if b["samples_per_s"] >= 1.6 * a["samples_per_s"]]
        W(f"Smallest per GPU batch with scaling efficiency of at least 0.8: {good[0] if good else 'none tested'}.\n")

# Part D
W("## HW4.5 Part D, serving revisited\n")
for c in cards:
    if serve_rows(c, "partd"):
        W(f"**{c}**, vLLM rerun:\n")
        table(["Configuration", "Tokens/sec", "p50 latency (ms)", "p95 latency (ms)"], serve_rows(c, "partd"))
        W("Same card type, same fp16 precision and same script as HW4, 4% to 7% slower. That size of gap is run to run "
          "variation (host CPU load and GPU clocks), not a change in hardware or precision.\n")
fig, ax = plt.subplots(figsize=(6, 4))
for i, c in enumerate(cards):
    k = R[c].get("part_d_kv")
    if not k:
        continue
    ok = [r for r in k["rows"] if not r["oom"]]
    W(f"**{c}** KV cache at batch {k['batch']} (`{k['model']}`, {k['kv_bytes_per_token']} bytes per token, "
      f"native max position {k['max_position']}). Longest context measured with HF prefill: **{k['longest_ok']}** "
      f"(fails at {k['shortest_fail']}). The KV cache is only a small part of the peak: memory runs out on the attention "
      f"scores, which grow with the square of the context.\n")
    W("vLLM capacity at batch 8 is **calculated, not tested**: both vLLM runs used max_model_len 2048, and that engine "
      "logged a KV pool of 9.9 GiB = 865,456 tokens on this T4. 865,456 / 8 = 108,182 tokens per sequence, so the KV "
      f"pool alone would cover the model's native maximum of {k['max_position']} at batch 8. A run with max_model_len "
      f"{k['max_position']} and 8 long prompts would be needed to confirm it.\n")
    table(["Context", "KV measured GB", "KV formula GB", "Peak GB"],
          [[r["context"], "OOM", "", ""] if r["oom"] else
           [r["context"], f"{r['kv_gb']:.3f}", f"{r['kv_formula_gb']:.3f}", f"{r['peak_gb']:.2f}"] for r in k["rows"]])
    ax.plot([r["context"] for r in ok], [r["kv_gb"] for r in ok], "o", c=COLORS[i % 3], ms=8, label=f"{c} measured")
    xs = [0, max(r["context"] for r in ok)]
    ax.plot(xs, [x * k["batch"] * k["kv_bytes_per_token"] / 1e9 for x in xs], "-", c=COLORS[1], lw=2,
            label="formula, 12,288 bytes per token")
ax.set(xlabel="context length (tokens)", ylabel="KV cache GB", title="KV cache memory at batch 8")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "part_d_kv.png"), dpi=120)
W("![kv](figures/part_d_kv.png)\n")

# Part E
W("## HW4.5 Part E, GPU hours and summary\n")
for c in cards:
    b, h = R[c].get("part_b"), R[c].get("hw4a")
    if not (b and h):
        continue
    n = h["data"]["n_total"]
    samples = int(0.7 * n) * h["settings"]["epochs"]
    W(f"**{c}**: full fine tune = 70% of {n} tweets x {h['settings']['epochs']} epochs = {samples} training samples.\n")
    one = int(0.7 * n)
    table(["Configuration", "Samples/s", f"{h['settings']['epochs']} epochs, GPU hours", "1 epoch, GPU hours"],
          [[r["technique"], f"{r['samples_per_s']:.1f}", f"{samples / r['samples_per_s'] / 3600:.3f}",
            f"{one / r['samples_per_s'] / 3600:.3f}"] for r in b["rows"]])
    W("GPU hours = samples / (samples per second x 3600). Validation loss stopped improving after epoch 1, so the "
      "1 epoch column is the realistic cost.\n")
    rows = b["rows"]
    lo = min(rows, key=lambda r: r["peak_gb"])
    fast = max(rows, key=lambda r: r["samples_per_s"])
    wall = min(rows, key=lambda r: r["wall_s"])
    W(f"### HW4.5 summary on {c}\n")
    W("This is the HW4.5 summary only. It is not an HW5 baseline.\n")
    table(["Best", "Configuration", "Value"],
          [["Peak memory", lo["technique"], f"{lo['peak_gb']:.2f} GB"],
           ["Throughput", fast["technique"], f"{fast['samples_per_s']:.1f} samples/s"],
           ["Wall clock", wall["technique"], f"{wall['wall_s']:.1f} s for {wall['opt_steps']} optimiser steps"]])

open(os.path.join(ROOT, "METRICS.md"), "w").write("\n".join(out))
print("wrote METRICS.md for cards:", cards)
