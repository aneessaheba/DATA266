# Assignment 7: GANs and AutoEncoders on FashionMNIST

Step 0 as in assignment 1. SID4 = 5330, SEED = 5330. No second hyperparameter configuration, since two models are already compared.

Questions 1a, 1b and 1c are answered in markdown cells in the notebook.

| Model | BCE per image | Pixel MSE |
|---|---|---|
| FC AE | 214.83 | 0.01032 |
| VAE (decoded from μ) | 225.75 | 0.01524 |
| VAE (decoded from sampled z) | 228.43 | 0.01642 |

The VAE's total loss (negative ELBO, 240.64) also includes 12.21 nats of KL, so it is not directly comparable to the FC AE loss. Only 6 of the 16 VAE latent dimensions are active.

The FC AE reconstructs better, the VAE generates realistic samples from N(0, I). Sampling the FC AE from a Gaussian fitted to its own codes narrows that gap considerably. Full analysis is in the last cell of the notebook.

## Files

* `notebooks/assignment7_gan_ae_fashionmnist.ipynb`, the notebook with outputs

## Rerunning

Requires `torch`, `torchvision`, `matplotlib` and `jupyter`. FashionMNIST downloads to `data/` on first run.

```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace assignment7_gan_ae_fashionmnist.ipynb
```
