# AI use, HW4.a and HW4.5

I used an AI assistant (Claude) as a learning and experimentation aid for this assignment.

## How I used it

- **Learning the topics.** I asked it to explain concepts I needed for the lab: BERT fine tuning
  with BertForSequenceClassification, how vLLM batches requests and manages the KV cache,
  mixed precision with a GradScaler on a T4, gradient checkpointing, gradient accumulation,
  8 bit optimisers (bitsandbytes), and how to read p50 and p95 latency.
- **Experimenting.** I used it to help me plan the experiments, try different settings, and understand
  errors while running on Colab, for example the vLLM startup failure and the probe running in eval
  mode, which I fixed and reran as the v3 notebook.
- **Review.** After finishing, I asked it to review the repository against the assignment brief and
  used its feedback to check my analysis and fill gaps.

## What I did myself

- Ran every experiment on Colab (Tesla T4). All numbers in METRICS.md, RUN_LOG.txt and results/
  come from my own runs, recorded with the GPU UUIDs in provenance/.
- Checked the outputs, chose the final configuration, and wrote and verified the analysis in
  ANALYSIS.md against my measured results.

I take responsibility for all submitted code, results and conclusions.
