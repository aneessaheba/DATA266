# Assignment 4: Mini GPT from Scratch

A decoder only Transformer built from `nn.Linear`, `nn.LayerNorm`, `nn.GELU` and
`nn.Embedding`, trained on character level Shakespeare, then sampled with greedy, temperature
and top k decoding. The multi head attention, the causal mask, the decoder block and the
sampling loop are written by hand. No `nn.Transformer`, no `nn.MultiheadAttention`, no
HuggingFace models.

## Personal parameters

| SID4 | SEED | SLICE | HP_ID | CLS_A | CLS_B |
|---|---|---|---|---|---|
| 5330 | 5330 | 330 | 2 | 0 | 3 |

`SEED` seeds Python, NumPy and PyTorch. The other four are stated as required by standing
requirement 0.1 but are not referenced by this assignment.

## Files

* `notebooks/assignment4_mini_gpt.ipynb`, the submission notebook with outputs saved
* `HW4_Document.pdf`, the findings write up
* `data/shakespeare.txt`, the corpus, 1,115,394 characters, 65 distinct characters
* `data/training_loss.png`, the loss curve

## Model and training

3,225,153 parameters: hidden dimension 256, 4 heads, 4 decoder layers, context length 128.
Adam, learning rate 1e-3, batch size 64, 6 epochs.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Training loss | 1.9114 | 1.3937 | 1.2691 | 1.1858 | 1.1083 | 1.0293 |

## Decoding comparison

Averaged over three generations of 300 characters from the prompt `ROMEO:`. Valid word rate
measures coherence, unique trigram rate measures diversity.

| decoding | valid words | unique trigrams |
|---|---|---|
| greedy | 0.952 | 0.574 |
| temperature 0.5 | **0.962** | 0.734 |
| temperature 1.0 | 0.924 | 0.831 |
| temperature 1.5 | 0.770 | **0.884** |
| top k = 2 | 0.943 | 0.730 |
| top k = 40 | 0.924 | 0.831 |

Most coherent is temperature 0.5. Greedy spells about as well but has the lowest diversity of
any setting and repeats itself heavily. Most diverse is temperature 1.5, at the cost of the
worst coherence. Top k at k = 40 produced output identical to temperature 1.0, because on a 65
character vocabulary the 25 lowest ranked characters are never sampled anyway.

## Rerunning

Requires `torch`, `numpy`, `matplotlib` and `jupyter`. Run from `notebooks/`, since the
notebook reads `../data/shakespeare.txt`:

```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 assignment4_mini_gpt.ipynb
```

The notebook picks MPS, then CUDA, then CPU. On MPS the full run takes about 12 minutes.
