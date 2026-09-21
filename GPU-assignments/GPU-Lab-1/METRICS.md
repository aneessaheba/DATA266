# METRICS, HW2.5 GPU Assignment I

SID4=5330 SEED=5330

Every number below was produced by the scripts in `scripts/` and is echoed, UUID labelled, in `RUN_LOG.txt`. A cell reading `not measured` means that part has not been run on that card yet.

## NVIDIA GeForce RTX 5090

* GPU UUID: `GPU-38ba4f51-a57b-4ca5-e2ab-74396ae57bdb`
* Driver 610.60 | CUDA (torch) 12.8 | torch 2.11.0+cu128
* VRAM 32607 MiB | power limit 575.00 W | host 5c9fd775c569

### Table HW2.5.1 summary

| Measurement | Your GPU | Notes |
|---|---|---|
| Peak achieved TFLOPS (BF16) | 226.87 | N=16384, 50 timed reps |
| % of theoretical peak (BF16) | 108.3% | vendor dense BF16 peak 209.5 TFLOPS |
| Effective bandwidth (GB/s) | 1555.2 | elementwise_add, 86.8% of 1792.0 GB/s spec |
| Naive attention OOM length | largest success S=32512, smallest failure S=32768. 1 probe(s) exceeded physical VRAM and were counted as failures, see the implausible rows in the Part D CSV | batch=1 heads=8 head_dim=64 dtype=fp16 reps=20 |
| Fused attention OOM length | no OOM up to S=131072 (largest tested) | OOM bracket refined to 256 tokens |
| Steady state over peak throughput | 94.3% | peak 230.52 TFLOPS in the first 30 s against steady 217.49 TFLOPS over the final 5.0 min |
| Throttle onset (s, or none) | 5 s | nvidia-smi reported SwPowerCap at 5 s, 52 C, 574.98 W, SM clock 2737 MHz. The first 1 sample(s), up to 5 s, were taken before the load ramped and are excluded |

### Part B, achieved throughput

| N | Precision | Achieved TFLOPS | % of peak | Median ms | Reps | CV % | Note |
|---|---|---|---|---|---|---|---|
| 1024 | FP32 | 40.269 | 38.42 | 0.053 | 50 | 35.961 |  |
| 1024 | TF32 | 26.779 | 25.55 | 0.080 | 50 | 23.713 |  |
| 1024 | FP16 | 96.145 | 45.89 | 0.022 | 50 | 37.123 |  |
| 1024 | BF16 | 20.281 | 9.68 | 0.106 | 50 | 21.519 |  |
| 1024 | FP8 | 41.196 | 9.83 | 0.052 | 50 | 62.577 |  |
| 4096 | FP32 | 64.782 | 61.81 | 2.122 | 50 | 3.265 |  |
| 4096 | TF32 | 101.686 | 97.03 | 1.352 | 50 | 4.951 |  |
| 4096 | FP16 | 209.833 | 100.16 | 0.655 | 50 | 10.132 |  |
| 4096 | BF16 | 179.694 | 85.77 | 0.765 | 50 | 11.896 |  |
| 4096 | FP8 | 376.388 | 89.83 | 0.365 | 50 | 11.777 |  |
| 8192 | FP32 | 64.474 | 61.52 | 17.053 | 50 | 1.425 |  |
| 8192 | TF32 | 112.975 | 107.8 | 9.732 | 50 | 0.471 |  |
| 8192 | FP16 | 212.909 | 101.63 | 5.164 | 50 | 1.499 |  |
| 8192 | BF16 | 224.206 | 107.02 | 4.904 | 50 | 1.208 |  |
| 8192 | FP8 | 450.094 | 107.42 | 2.443 | 50 | 1.78 |  |
| 16384 | FP32 | 65.505 | 62.5 | 134.282 | 50 | 0.486 |  |
| 16384 | TF32 | 116.254 | 110.93 | 75.663 | 50 | 0.361 |  |
| 16384 | FP16 | 214.576 | 102.42 | 40.993 | 50 | 0.63 |  |
| 16384 | BF16 | 226.866 | 108.29 | 38.772 | 50 | 1.682 |  |
| 16384 | FP8 | 436.706 | 104.23 | 20.142 | 50 | 1.214 |  |

### Part B, where each precision plateaus

Plateau is the smallest swept N reaching 95% of that precision's best measured throughput. The last column is why small matrices never plateau: at the smallest N the card delivers only a fraction of what it manages at the top of the sweep.

| Precision | Best achieved TFLOPS | Plateau | Smallest N | Throughput at smallest N, as % of best |
|---|---|---|---|---|
| FP32 | 65.50 | N=4096 | 1024 | 61.5% |
| TF32 | 116.25 | N=8192 | 1024 | 23.0% |
| FP16 | 214.58 | N=4096 | 1024 | 44.8% |
| BF16 | 226.87 | N=8192 | 1024 | 8.9% |
| FP8 | 450.09 | N=8192 | 1024 | 9.2% |

### Part C, roofline

| Operation | Effective GB/s | % of spec | Arithmetic intensity (FLOPs per byte) | Ridge point | Side |
|---|---|---|---|---|---|
| elementwise_add | 1555.16 | 86.78 | 0.0833 | 58.48 | memory bound |
| device_copy | 1501.32 | 83.78 | 0.0 | 58.48 | memory bound |
| matmul_n8192 | 48.2 | 2.69 | 1365.3333 | 58.48 | compute bound |

### Part D, the cost of attention

| Impl | S | Peak memory (GiB) | Latency (ms) | Speedup vs naive | Status |
|---|---|---|---|---|---|
| naive | 512 | 0.0177 | 0.1503 |  | ok |
| naive | 1024 | 0.0431 | 0.07 |  | ok |
| naive | 2048 | 0.1407 | 0.2926 |  | ok |
| naive | 4096 | 0.5236 | 1.2071 |  | ok |
| naive | 8192 | 2.0392 | 4.6977 |  | ok |
| naive | 16384 | 8.0704 | 17.9389 |  | ok |
| fused | 512 | 0.0119 | 0.0912 | 1.648 | ok |
| fused | 1024 | 0.0158 | 0.0319 | 2.194 | ok |
| fused | 2048 | 0.0158 | 0.0704 | 4.156 | ok |
| fused | 4096 | 0.0237 | 0.2131 | 5.664 | ok |
| fused | 8192 | 0.0394 | 0.8127 | 5.78 | ok |
| fused | 16384 | 0.0709 | 2.7823 | 6.448 | ok |

Quadratic fit (naive): peak_mem_bytes = 32*S^2 + 4096*S + 8.51968e+06, R^2 = 1.000000. The S^2 coefficient is fitted to the measured sweep, not asserted.
Quadratic fit (fused): peak_mem_bytes = 0.0331491*S^2 + 3413.6*S + 1.14493e+07, R^2 = 0.997363. The S^2 coefficient is fitted to the measured sweep, not asserted.

