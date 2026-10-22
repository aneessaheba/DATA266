# Assignment 8: Mini Diffusion Model (DDPM to DDIM) on MNIST

Step 0 as in assignment 1. SID4 = 5330, SEED = 5330.

Required configuration: T = 100, β linear from 0.0001 to 0.02, batch size 128, 50 epochs. The noise predictor is a small U-Net (18 conv layers, 0.61M parameters, GroupNorm + SiLU, sinusoidal timestep embedding added in every residual block). EMA weights are used for sampling.

Samples are judged by a small MNIST classifier (98.7% test accuracy), 1000 samples per sampler.

| Sampler | Net evals | Time (s) | Confidence | Class entropy | Near 0/1 pixels |
|---|---|---|---|---|---|
| Real MNIST | - | - | 0.984 | 2.295 | 0.915 |
| DDPM | 100 | 39.2 | 0.839 | 1.818 | 0.847 |
| DDIM (η = 0) | 50 | 21.0 | 0.821 | 1.495 | 0.731 |
| DDIM (η = 0) | 20 | 8.4 | 0.819 | 1.502 | 0.728 |

With the required schedule, ᾱ_100 ≈ 0.36, so x_100 is not pure noise during training. Sampling from N(0, I) therefore gives digit-like scribbles. As an extension, the same model is retrained with β scaled by 10 (0.001 to 0.2, ᾱ_100 ≈ 2e-5). It then produces clear digits that match real MNIST statistics:

| Sampler (scaled β) | Net evals | Time (s) | Confidence | Class entropy | Near 0/1 pixels |
|---|---|---|---|---|---|
| DDPM | 100 | 46.7 | 0.930 | 2.290 | 0.916 |
| DDIM (η = 0) | 50 | 24.7 | 0.906 | 2.291 | 0.909 |
| DDIM (η = 0) | 20 | 9.9 | 0.900 | 2.289 | 0.905 |

DDIM is about 2× (50 steps) and 5× (20 steps) faster with only a small drop in quality. It is deterministic, so the same x_T gives the same digit at any step count. Full discussion is in the last cell of the notebook.

## Files

* `notebooks/assignment8_ddpm_ddim_mnist.ipynb`, the notebook with outputs
* `notebooks/ddpm_mnist_ema.pt`, EMA weights for the required schedule
* `notebooks/ddpm_mnist_ema_scaled_beta.pt`, EMA weights for the scaled schedule

## Rerunning

Requires `torch`, `torchvision`, `matplotlib` and `jupyter`. MNIST downloads to `data/` on first run. A full run takes about 2 hours on an Apple M-series GPU (two 50-epoch trainings).

```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace assignment8_ddpm_ddim_mnist.ipynb
```
