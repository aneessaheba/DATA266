# HW4.a and HW4.5 GPU Assignment II

SJSU ID 018205330, so SID4 = 5330, SEED = 5330, SLICE = 5330 % 1000 = 330.

Data: `cardiffnlp/tweet_eval` sentiment, train split (45,615 tweets, 3 classes). Rows 330 to 5329 are used,
split 70/15/15 with SEED. To use a course provided CSV instead, put it at `data/twitter.csv` with columns
`text,label`.

GPU: Colab T4 (`CARD=colab_t4`). One GPU is enough per the TA; the lab RTX 5090 (`CARD=rtx5090`) is an optional second card.
HW4.a and HW4.5 use the same dataset, row range and split.

## Files

| File | Does |
|---|---|
| scripts/common.py | IDs, data split, model, training loop, OOM probe, logging |
| scripts/hw4a.py | HW4.a BERT fine tune, curves, confusion matrix, save and reload, custom inference |
| scripts/serve.py | HF pipeline baseline and vLLM at batch 1, 8, 32 (HW4.a and Part D) |
| scripts/part_a.py | Part A batch doubling until OOM |
| scripts/part_b.py | Part B AMP, checkpointing, accumulation, 8 bit optimiser, best combination |
| scripts/part_d_kv.py | Part D KV cache memory against context length, longest context at batch 8 |
| scripts/report.py | builds METRICS.md and figures/ from results/ |
| run_all.sh | runs everything on one card |

Outputs: `results/<card>/*.json`, `figures/`, `provenance/nvidia-smi-q_<card>.txt` (UUID capture),
`RUN_LOG.txt`, `METRICS.md`.

## Run

Lab RTX 5090:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
CARD=rtx5090 bash run_all.sh
```

Colab (Runtime, change runtime type, T4 GPU). Upload `GPU_Lab_2.zip`, then:

```
!unzip -q GPU_Lab_2.zip -d /content/lab
%cd /content/lab
!pip install -q -r requirements.txt
!CARD=colab_t4 bash run_all.sh
!zip -qr /content/colab_results.zip results provenance figures RUN_LOG.txt
```

If vLLM fails to start on the T4, run `!pip install -q vllm==0.9.2` and repeat only the serving stages with
`!CARD=colab_t4 STAGES="serve d" bash run_all.sh`.

Part C is optional per the TA and was skipped; METRICS.md states "Part C skipped (optional)".
The GPU used is recorded in `provenance/nvidia-smi-L_<card>.txt`.

Chain check without a GPU: `SMOKE=1 CARD=mac bash run_all.sh`, then delete `results/mac`, `RUN_LOG.txt`,
`METRICS.md`, `figures/`. Never submit smoke results.

## Notebooks actually used

`hw45_colab_v2.ipynb` produced HW4.a, serving and Part D. `hw45_colab_v3.ipynb` reran Parts A and B after the eval mode fix.

## After the runs

Copy each card's `results/`, `provenance/` and its `RUN_LOG.txt` lines into this folder, run
`python scripts/report.py`, fill ANALYSIS.md and reservations.md, then

```bash
git init && git add . && git commit -m "HW4.5" && git tag hw4-5
```
