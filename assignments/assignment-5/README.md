# Assignment 5

Fine tuning FLAN T5 Small with LoRA (PEFT) to summarize dialogues from the DialogSum dataset,
and comparing outputs before and after fine tuning.

## Personal parameters

SID4 = 5330, SEED = 5330, SLICE = 330, HP_ID = 2, CLS_A = 0, CLS_B = 3. They are stated in the
first cell of the notebook. Only SEED is used: Python random, NumPy and PyTorch are all seeded
with it, and the PyTorch seed is reset before each LoRA model is built. HP_ID does not apply
because the rank sweep (r = 4 and r = 16) is fixed for all students.

## Files

* `notebooks/assignment5_lora_dialogsum.ipynb`: the submission notebook with all outputs saved.
  Sections 1 to 7 match the seven tasks of the assignment.
* `baseline_outputs.csv`: the baseline summaries of the two test dialogues, before fine tuning.
* `models/lora_r16`: the saved LoRA adapters for rank 16 (main model).
* `models/lora_r4`: the saved LoRA adapters for rank 4 (experiment).

## Data

DialogSum from Hugging Face: 1,999 train, 499 validation and 499 test dialogues. The input is
the dialogue wrapped in an instruction style prompt ("Summarize the following conversation."),
of the kind FLAN T5 was trained on. The target is the human written summary. Each test dialogue
appears three times with different reference summaries, so the two sample dialogues are test
rows 0 and 3, and ROUGE uses every third test row (100 different dialogues).

## LoRA setup

| setting | value |
|---|---|
| rank r | 16 (main), 4 (experiment) |
| alpha | 2r (32 and 8), so alpha / r = 2 for both |
| dropout | 0.05 |
| target modules | attention query and value (q, v) |
| total parameters | 77,649,280 (r = 16), 77,133,184 (r = 4) |
| trainable parameters | 688,128 or 0.89% (r = 16), 172,032 or 0.22% (r = 4) |

Training uses the full train split for 5 epochs, batch size 8, AdamW with learning rate 0.001.
Decoding is greedy with at most 100 new tokens, the same for every model.

## Results

| model | final training loss | ROUGE 1 | ROUGE 2 | ROUGE L |
|---|---|---|---|---|
| baseline | none | 0.101 | 0.023 | 0.091 |
| LoRA r = 4 | 1.416 | 0.379 | 0.128 | 0.313 |
| LoRA r = 16 | 1.341 | 0.374 | 0.120 | 0.304 |

Before fine tuning the model outputs unrelated fragments ("Is this all correct?", "Talk to
the driver."). After fine tuning it writes real summaries that capture the main topics, but
still repeats itself and confuses some roles and facts.

r = 4 scores slightly higher on ROUGE and r = 16 reaches a lower training loss, but with one
seed and 100 dialogues the gap is too small to call a real difference, and on the two samples
the results are mixed. The smaller rank is a reasonable choice for cost, at a quarter of the
trainable parameters, but the results do not show that it is better.

## Running

Install `torch`, `transformers`, `datasets`, `peft`, `evaluate`, `rouge_score`, `pandas` and
`jupyter`, then open the notebook from the `notebooks` folder and run all cells. The dataset
and model download automatically. It picks MPS, then CUDA, then CPU, and was run on MPS on a
Mac, where training both models takes about 30 minutes.
