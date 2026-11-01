# HW4.a and HW4.5 GPU Assignment II

SJSU ID 018205330, so SID4 = 5330, SEED = 5330, SLICE = 5330 % 1000 = 330.

Data: `cardiffnlp/tweet_eval` sentiment, train split (45,615 tweets, 3 classes). Rows 330 to 5329 are used,
split 70/15/15 with SEED. HW4.a and HW4.5 use the same dataset, row range and split.

GPU: one Google Colab Tesla T4 (`CARD=colab_t4`). One GPU is enough per the TA. Part C is optional per the TA and was
skipped; METRICS.md states "Part C skipped (optional)".

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
| hw45_colab_v2.ipynb | the Colab run, with outputs, for HW4.a, serving and Part D |
| hw45_colab_v3.ipynb | the Colab rerun, with outputs, for Parts A and B after the eval mode fix |

Outputs:

* `results/colab_t4/*.json`: every measurement. `part_b_probes.json` holds the per batch probe rows printed in the v3
  notebook, used for the batch 256 ranking.
* `provenance/nvidia-smi-q_colab_t4_run1.txt` and `_run2.txt`: full GPU captures for the two Colab T4 instances
  (run1 = v2 notebook, run2 = v3 notebook). `nvidia-smi-L_colab_t4_run1.txt` and `_run2.txt` hold the short form.
* `figures/`, `RUN_LOG.txt` (times in UTC), `METRICS.md`, `ANALYSIS.md`.

## How it was run

The notebooks were created in Google Drive and opened in Colab on a T4, then Runtime, Run all. Each notebook mounts
Drive, copies the code folder from `MyDrive/GPU_Lab_2`, writes the fixed versions of any changed files, installs
`requirements.txt`, runs `run_all.sh`, and copies the results back to Drive. The v2 notebook's checksum cell still lists
`part_c_ddp.py`, which was later removed because Part C was skipped.

Versions recorded in the result files: torch 2.13.0+cu130 for the v2 run (installed with vLLM) and torch 2.11.0+cu130
for the v3 run, CUDA 13.0, driver 580.82.07. `requirements.txt` is unpinned and vLLM was the latest pip release on
1 Nov 2026; its exact version was not printed in the logs.

To rerun on one card: `pip install -r requirements.txt` then `CARD=colab_t4 bash run_all.sh`, and
`python scripts/report.py` rebuilds METRICS.md and figures/ from results/.
