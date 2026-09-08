# Measurement and Analysis

SID4=5330  SEED=5330  HP_ID=2

## 1. Multi-seed test accuracy (mean and standard deviation)

Data split kept fixed with `random_state=SEED` (5330) throughout. Baseline and HP_ID=2
modified networks were each trained three times using training seeds SEED, SEED+1, and
SEED+2, for both PyTorch and TensorFlow.

| Model                | acc (SEED) | acc (SEED+1) | acc (SEED+2) | Mean accuracy | Std dev |
|-----------------------|------------|---------------|---------------|----------------|---------|
| PyTorch baseline       | 0.7500     | 0.7414        | 0.7328        | 0.7414         | 0.0070  |
| PyTorch HP_ID=2        | 0.7500     | 0.7672        | 0.7586        | 0.7586         | 0.0070  |
| TensorFlow baseline    | 0.7586     | 0.7414        | 0.7500        | 0.7500         | 0.0070  |
| TensorFlow HP_ID=2     | 0.7500     | 0.7586        | 0.7328        | 0.7471         | 0.0108  |

Source: `metrics_nn_summary.csv`, produced by the multi-seed cell in `neural_networks.ipynb`.

## 2. SEED run loss curves: better, overfitting, underfitting, or indistinguishable

PyTorch: the HP_ID=2 model (lr=0.003) converges faster than baseline. Both its training and
validation loss drop more steeply and reach roughly 0.45 by epoch 30, versus about 0.59 to
0.60 for the baseline (lr=0.001). Training and validation loss stay close together for both
models throughout all 30 epochs, with no divergence between the two curves, so neither model
is overfitting. Verdict: the modified model is better, showing faster convergence and a lower
final loss with comparable generalization.

TensorFlow: the same faster convergence pattern is visible. The modified model's train and
validation loss drops more steeply than the baseline's, but this did not translate into a
higher single-seed test accuracy (baseline 0.7586 vs modified 0.7500). Train and validation
curves again track closely for both models, so neither is overfitting. Verdict:
indistinguishable from baseline on test accuracy, despite visibly faster loss convergence
during training.

Loss curve figures: `pytorch_loss_curves.png` and `tensorflow_loss_curves.png`, generated in
`neural_networks.ipynb`.

## 3. HP_ID and exact modified configuration

HP_ID = 2 (Learning-rate-high arm, from SID4 = 5330 mod 6 = 2).

Per the HW1 HP_ID mapping, HP_ID=2 uses hidden layers [64, 32] (unchanged from baseline),
learning rate 0.003 (baseline: 0.001), and 30 epochs (unchanged). This is the required
modified model for this assignment. As shown above, it outperforms baseline in PyTorch and is
roughly neutral (slightly lower mean, higher variance) in TensorFlow. Both results are
reported here as the honest outcome of the assigned configuration, per the assignment's
instructions.

## 4. CUDA: kernel vs CPU baseline timing

Kernel timed against a naive single threaded CPU baseline for N = 256, 1024, 4096. All GPU
timings measured with `cudaEvent_t`, averaged over 5 kernel launch repeats (kernel time) with
one untimed warm up launch excluded.

| Matrix size | CPU (ms)    | GPU kernel (ms) | H2D+D2H (ms) | Speedup    |
|-------------|-------------|------------------|--------------|------------|
| 256         | 19.770      | 0.062            | 0.371        | 45.703x    |
| 1024        | 3288.839    | 3.502            | 4.897        | 391.582x   |
| 4096        | 700001.548  | 195.519          | 78.519       | 2554.399x  |

Source: `metrics_cuda_timing.csv`, produced by `cuda.ipynb`. Raw console output is in
`RUN_LOG.txt`.

### Profiler output separating kernel time from transfer time

Profiler used: Nsight Compute (`ncu`), invoked via its full binary path
(`/usr/local/cuda-12.8/bin/ncu --set basic ./matmul 1024`) since it is not on the default
Colab PATH. Nsight Systems (`nsys`) is not installed in this Colab environment.

Nsight Compute profile of `matmulTiledKernel` at N=1024 (grid (64,64,1) x block (16,16,1),
Compute Capability 7.5, Tesla T4):

| Metric                     | Value              |
|------------------------------|-------------------|
| Duration (profiled)          | 5.80 ms           |
| Memory Throughput             | 74.34%            |
| Compute (SM) Throughput       | 74.34%            |
| DRAM Throughput               | 10.81%            |
| L1/TEX Cache Throughput       | 95.27%            |
| Achieved Occupancy            | 98.68%            |
| Theoretical Occupancy         | 100%               |
| Registers per thread          | 39                |

The profiled kernel duration (5.80 ms) is higher than the CUDA event measured kernel time
above (3.502 ms) because Nsight Compute's own instrumentation adds overhead while profiling.
This is expected and does not reflect the kernel's real, unprofiled runtime. The profiler
output isolates kernel execution ("Duration") separately from host device transfer, which is
not part of the `ncu` kernel level report at all (transfer time is captured instead via the
`cudaEvent` H2D+D2H measurement in the table above). Compute and Memory Throughput are both
around 74%, which Nsight Compute reports as well balanced, meaning the kernel is not clearly
bottlenecked on either arithmetic or memory bandwidth. Achieved Occupancy (98.68%) is close to
the theoretical maximum (100%), confirming the 16x16 tiled shared memory design uses the GPU's
resources efficiently. Full raw profiler output is in `RUN_LOG.txt`.

## 5. Crossover discussion

GPU end to end time already beats CPU time at the smallest size tested, N=256 (19.770 ms CPU
vs 0.433 ms GPU end to end, a 45.7x speedup), so the true crossover point in this environment
is smaller than 256, not between any of the tested sizes. The crossover is not at size zero
because the GPU path carries fixed costs that do not shrink with N: a roughly constant kernel
launch overhead and a near constant host to device/device to host transfer latency (about 0.37
ms observed at N=256). At very small N there isn't enough parallel work to amortize these
fixed costs, so a sufficiently tiny matrix multiply would in principle run faster on the CPU
alone. As N grows, the O(N^3) compute cost grows far faster than these fixed overheads, which
is why the GPU's advantage increases dramatically across the tested range, from about 46x at
N=256 to about 2554x at N=4096.
