# Assignment 3

Prompt engineering techniques and self attention with causal masking, implemented from
scratch.

## Files

* `notebooks/assignment3_prompting_and_self_attention.ipynb`: the submission notebook.
  Section 1 covers six prompt engineering techniques, two examples each, run through
  LangChain against a local llama3.1 8B model served by Ollama. Section 2 implements scaled
  dot product self attention and causal masking from raw PyTorch, with no
  nn.MultiheadAttention, nn.Transformer, or HuggingFace transformer class.
* `HW3_Document.pdf`: the findings write up, matching the notebook's outline.
* `AI_USE.md`: AI use disclosure.
* `data/`: the two heatmap images produced by the notebook (unmasked and causal self
  attention).
* `scripts/build_notebook_final.py`: generates the notebook from source. Not part of the
  submission; only needed to regenerate the notebook from scratch.

## Rerunning the notebook

Section 2 (self attention) runs on its own with no external dependencies beyond PyTorch.

Section 1 (prompt engineering) calls a local LLM through Ollama. Before rerunning it:

```
ollama pull llama3.1
ollama serve
```

then run the notebook top to bottom. Generation uses temperature 0 with a fixed seed, so
results are close to reproducible, though local LLM inference is not always perfectly
deterministic between runs.
