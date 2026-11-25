# Assignment 12: GPTQ Quantization and Knowledge Distillation

SID4 = 5330, SEED = 5330, HP_ID = 2. Everything ran in one Google Colab Tesla T4 session. The two parts use different models and tasks, so they are reported separately.

## Part 1: GPTQ (Table HW11.1)

Base model `facebook/opt-125m`, revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6`. GPTQ at 4 bits, calibrated on Wikitext2 train, evaluated on Wikitext2 test.

| Configuration | Perplexity | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Base LLM | 31.76 | 291.6 / 395.8 | 0.805 | 250.5 |
| GPTQ quantized LLM | 34.13 | 558.9 / 731.8 | 0.680 | 125.3 |

## Part 2: Distillation on AG News (Table HW11.2)

Teacher: the provided `teacher_model` (BERT base). Student: `distilbert-base-uncased`, loss = alpha * CE + (1 - alpha) * KLD.

| Configuration | Accuracy | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Teacher (BERT base) | 0.9443 | 12.6 / 15.3 | 1.368 | 438.9 |
| Best distilled student (T = 1, alpha = 0.5) | 0.9446 | 6.3 / 6.9 | 1.057 | 267.8 |

* Temperature sweep at alpha 0.5: T = 1 gives 0.9446, T = 2 gives 0.9411, T = 5 gives 0.9432, T = 10 gives 0.9445.
* Alpha sweep at T = 1: 0.25 gives 0.9428, 0.5 gives 0.9446, 0.75 gives 0.9429.
* Business is the class that degrades most (0.9137 to 0.9021).

The training log, loss trends, per class accuracy, the five error examples and the full analysis are in the last cells of the notebook.

## Files

* `hw12.ipynb`, the notebook with outputs

## Rerunning

Open the notebook in Colab with a T4 GPU, place the instructor's `teacher_model` folder in `/content`, then run all cells (about one hour).
