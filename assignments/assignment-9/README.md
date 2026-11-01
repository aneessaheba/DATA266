# Assignment 9: Latent Diffusion on FashionMNIST

Step 0 as in assignment 1. SID4 = 5330, SEED = 5330. The two trained models are the pretrained VAE and the diffusion UNet.

| Part | Result |
|---|---|
| VAE | stabilityai/sd-vae-ft-mse, 128x128 input to 4x16x16 latent, s = 0.18215 |
| Reconstruction MSE | 0.00077 on 512 test images |
| Forward diffusion | T = 1000, β linear from 1e-4 to 0.02, shown at t = 0, 200, 500, 800, 1000 |
| UNet | 5.60M parameters, base 64, multipliers (1, 2, 4), self attention at 16x16 and bottleneck |
| Conditioning | 10 classes plus 1 null label, labels dropped with probability 0.1 |
| Training | 50 epochs, LR 2e-5, final noise loss 0.0998 |
| Sampling | CFG with w = 3, 64 images, labels 0 to 9 repeated |

The analysis is in the last cell of the notebook.

## Files

* `notebooks/assignment9_ldm_fashionmnist.ipynb`, the notebook with outputs

## Rerunning

Run on Google Colab with a T4 GPU (about 45 minutes). The first cell installs `diffusers`. FashionMNIST downloads on first run.
