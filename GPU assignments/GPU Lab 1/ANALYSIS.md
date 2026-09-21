## Part B: why small matrices never plateau

A matmul does 2N cubed FLOPs while touching only 3N squared elements, so arithmetic
intensity grows linearly with N and a small matrix simply does not carry enough work per
byte moved to keep the tensor cores fed. Two fixed costs dominate instead, and neither
shrinks with N. The first is per launch overhead, meaning kernel launch, cuBLAS algorithm
selection and wave quantisation at the tile level, which is roughly constant and is
amortised over work growing as N cubed, so it is a large fraction of the 0.053 ms median
at N=1024 and disappears entirely into the 134.282 ms at N=16384. The second is
occupancy: a 128 by 128 tiled GEMM at N=1024 produces only 64 tiles, which cannot fill
the 170 SMs on GB202, so well over half the machine sits idle for the whole kernel no
matter how fast the tensor cores are.

This is why the plateau only arrives at N=4096 for FP32 and FP16 and at N=8192 for TF32,
BF16 and FP8, and it is also why N=1024 carries by far the largest variance in the sweep,
with coefficients of variation between 21.52% and 62.58% against under 1.8% at N=8192. At
that size the kernel is short enough that launch and clock ramp noise is comparable to the
work itself, so the N=1024 column should be read as an order of magnitude rather than a
precise measurement.

## Part D: what the fused kernel avoids doing

The fused kernel never materialises the S by S score matrix in HBM at all, where the
naive version writes the scores, reads them back for the softmax, writes the weights,
then reads those back for the second matmul, so the quadratic term in the naive version
is HBM traffic as much as it is HBM capacity. Instead it tiles the computation and runs
the softmax online over blocks, keeping a running maximum and sum so each block can be
rescaled as later blocks arrive, which lets the QK product, the softmax and the AV
product stay fused in a single kernel with the intermediates held in SRAM and registers
rather than round tripping through HBM. It performs the same arithmetic, so what it saves
is bandwidth and capacity rather than FLOPs, which is exactly what my fit shows: the
S squared coefficient falls from 32 bytes per token squared to 0.0331, and peak memory at
S=16384 is 113.79 times smaller. The payoff compounds with sequence length, giving a
6.448x speedup at S=16384 and letting the fused path still run at S=131072 where the naive
path already fails at S=32768.

## Part F: the summary table

Table HW2.5.1 is generated into METRICS.md. Every cell traces to a UUID labelled line in
RUN_LOG.txt.

Commentary the table cannot carry:

Several Part B readings exceed 100% of the recorded theoretical peak, up to 110.93% at
N=16384 TF32. This is not a measurement error. The peak TFLOPS figures in gpu_specs.json
use NVIDIA rated boost clock of 2407 MHz. Part E measured sustained SM clock of 2737 MHz
under load, 1.14x the rated boost clock, which alone explains readings up to 114% of the
rated peak. This is necessary and sufficient to explain the observed inflation without
appealing to higher transient clocks. The inflation is not uniform across precisions at
N=16384, running 1.109x for TF32, 1.083x for BF16, 1.042x for FP8 and 1.024x for FP16,
which supports the clock explanation rather than undermining it: the heavier the tensor
operation, the more power it draws, so the lower the clock it can sustain against the
575 W cap Part E recorded as active from the first loaded sample onward.

Part D naive attention initially reported an OOM boundary of S=50688/50944, which was
wrong. The probe at S=32768 reported a peak of 32.13 GiB, exceeding the cards physical
31.84 GiB VRAM, on this WSL2 backed lab machine, allocations were silently spilling into
host memory instead of failing, so torch reported false successes with 9 to 10 second
latencies instead of raising an exception. This was caught by cross checking each
successful probes reported peak against torch.cuda.mem_get_info(), rejecting any result
that exceeds physical VRAM as failed with a distinct implausible marker. After the fix,
the naive OOM bracket is S=32512 ok, S=32768 fail, matching the independent prediction
from the fitted quadratic, 32 times S squared plus 4096 times S equals 31.84 GiB, solved
at S is approximately 32600.

A separate bug caused a hard crash during the first, unfixed run: torch.AcceleratorError,
CUDA error, device not ready, after two consecutive large failed allocations. This
exception type is a RuntimeError subclass, so it was already being caught, the real
defect was that is_oom() did not recognise device not ready as a caught failure and
re-raised it. Fixed by adding it as a recognised marker.

Part E throttle onset originally read as 0 seconds because the scan compared every
sample, including the pre ramp idle sample at 0.1 s, against the loaded baseline. Fixed
by excluding samples before the load ramps up. Corrected onset is 5 s, 52C, 574.98W,
matching the card reaching its power cap almost immediately after sustained load begins.
Unlike the other cells in Table HW2.5.1, this one is not printed directly by the run. It
is derived from the committed thermal log, which the Part E block in RUN_LOG.txt names as
logs/part_e_thermal_74396ae57bdb.csv, and the first loaded sample in that file reads
5.16 s, 2737 MHz, 52 C, 574.98 W with throttle bits 0x0000000000000004, which decodes to
SwPowerCap.

The lab Docker image ships PyTorch 2.1.2, which does not support the RTX 5090s Blackwell
architecture, CUDA capability sm_120, kernel launches fail with no kernel image is
available for execution on the device. PyTorch was upgraded in place inside the container
to 2.11.0+cu128 before any measurement was taken. This is necessary and affects every
number in this table.
