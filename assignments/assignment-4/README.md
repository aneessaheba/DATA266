# Assignment 4

A GPT style decoder only Transformer built from scratch, trained on the character level
Shakespeare corpus, and sampled with greedy, temperature and top-k decoding.

The multi-head masked self attention, the causal mask, the decoder block and the sampling
loop are all written by hand from `nn.Linear`, `nn.LayerNorm`, `nn.GELU` and `nn.Embedding`.
No `nn.Transformer`, no `nn.MultiheadAttention`, no HuggingFace transformer models.

## Files

* `notebooks/assignment4_mini_gpt.ipynb`: the submission notebook, with all outputs saved.
  Part 1 is character level tokenization and the sliding window dataset, Part 2 the
  architecture, Part 3 training, Part 4 the three decoding strategies, Part 5 the analysis.
* `HW4_Document.pdf`: the findings write up, matching the notebook's outline.
* `data/shakespeare.txt`: the corpus. 1,115,394 characters, 65 distinct characters.
* `data/training_loss.png`: the loss curve produced by the notebook.

## Model and training

3,225,153 parameters: hidden dimension 256, 4 heads, 4 decoder layers, context length 128.
Trained with Adam at a learning rate of 1e-3, batch size 64, for 6 epochs.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Training loss | 1.8955 | 1.3892 | 1.2651 | 1.1829 | 1.1064 | 1.0272 |

## Decoding comparison

Each setting is scored on two measures, averaged over 3 generations of 300 characters from
the prompt `ROMEO:`. Valid word rate is the fraction of words that occur in the corpus, so it
measures coherence. Distinct-3 is the fraction of character trigrams that are unique, so it
measures diversity.

| decoding | valid words | distinct-3 |
|---|---|---|
| greedy | **0.984** | 0.671 |
| temperature 0.5 | **0.984** | 0.753 |
| temperature 1.0 | 0.864 | 0.864 |
| temperature 1.5 | 0.755 | **0.905** |
| top-k, k = 2 | 0.932 | 0.746 |
| top-k, k = 40 | 0.869 | 0.859 |

Most coherent is greedy. It ties with temperature 0.5 on valid word rate, but it has the
lowest diversity of any setting, so it is the most conservative of the two. Most diverse is
temperature 1.5, which buys that diversity with the worst coherence.

Top-k at k = 40 produced output byte identical to temperature 1.0, because on a 65 character
vocabulary the 25 lowest ranked characters carry too little probability to ever be sampled.
Top-k only does real work when k is small relative to the number of characters the model is
actually considering.

## Rerunning the notebook

Requires `torch`, `matplotlib` and `jupyter`. Run from the `notebooks/` directory, since the
notebook reads the corpus at `../data/shakespeare.txt`:

```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 assignment4_mini_gpt.ipynb
```

The notebook picks MPS, then CUDA, then CPU. It was run on MPS, where training takes about 7
minutes. The seed is fixed, though GPU reductions are not bit exact across backends, so
sampled text may differ on a different device.
