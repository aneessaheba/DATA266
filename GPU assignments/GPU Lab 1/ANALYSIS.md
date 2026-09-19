# Analysis, HW2.5

`METRICS.md` is generated and holds the numbers. This file is where the written answers
go, in your own words. Each section names the question, points at the generated number to
quote, and leaves the writing to you. Nothing here is auto filled.

## Part B: the plateau, and why small matrices never reach it

Quote from `METRICS.md`, section "Part B, where each precision plateaus", and
`figures/part_b_tflops_vs_size_<uuid>.png`.

**B1. At which size does each precision plateau?**

Give the size per precision from the plateau table. If a precision was still climbing at
N=16384, say so. "Not reached within the swept range" is a real answer, and the table
prints it when it happens.

**B2. Why do small matrices never plateau?**

The mechanism to write up in your own words:

* A matmul does 2N cubed FLOPs but touches only 3N squared elements, so arithmetic
  intensity grows with N. At N=1024 there is not enough work per byte to keep the tensor
  cores fed.
* Fixed per launch overhead, meaning kernel launch, cuBLAS heuristic selection and wave
  quantisation at the tile level, is a constant cost amortised over work that grows as N
  cubed. At small N it is a large share of the measured time. At large N it disappears
  into the noise.
* Occupancy. At N=1024 a 128 by 128 tiled GEMM produces only 64 tiles, which cannot fill
  all the SMs on either card, so part of the machine idles through the whole kernel.

Tie this to your own numbers. The plateau table's last column gives the fraction of peak
the card actually delivered at N=1024.

**B3. The lower precision, FP8 and below.**

Did `torch._scaled_mm` run on your card? If it did, give the achieved TFLOPS and the
percentage of the dense FP8 peak. If it did not, paste the exact error from the `note`
column of `data/part_b_<uuid>.csv` and say what you concluded from it. The assignment
counts the failure as a finding about tooling maturity, so quote the real message rather
than paraphrasing it.

## Part C: which side of the roofline

Quote from `METRICS.md`, section "Part C, roofline", and
`figures/part_c_roofline_<uuid>.png`.

**C1. Effective bandwidth.**

Give GB/s for the elementwise add and the copy, each as a percentage of the card's
specified bandwidth. Say which of the two got closer to spec and offer a reason. The copy
moves 2 bytes per element against the add's 3, and the read to write mix affects the
achievable fraction.

**C2. Arithmetic intensity and the ridge point.**

State the ridge point your card reports, peak FLOPS divided by peak bandwidth, printed in
the table, and the arithmetic intensity of each operation. Then say which side each falls
on. The elementwise add is around 0.08 FLOPs per byte against a ridge in the tens, so it
is bandwidth bound by roughly three orders of magnitude. The 8192 matmul is far to the
right of the ridge and is compute bound. Note that the ridge point is a property of the
card, not of the kernel.

## Part D: the cost of attention

Quote from `METRICS.md`, `data/part_d_summary_<uuid>.json`, and
`figures/part_d_peak_memory_<uuid>.png`.

**D1. Configuration.** State the head dimension and batch size you chose, and why. The
default is `batch=1 heads=8 head_dim=64 dtype=fp16`. Say if you changed it.

**D2. The quadratic, confirmed from your data.**

Give the fitted coefficient a from `peak_mem = a*S^2 + b*S + c` and the R squared. Then
compare a against `batch * heads * 2 * itemsize`, the bytes two S by S tensors, the
scores and the softmax output, would occupy. At batch 1, 8 heads, fp16 that predicts 32
bytes per token squared. Say how close your measured coefficient came, and account for
any gap. The allocator rounds block sizes up, and any transient the kernel keeps alive
lands in the same peak.

**D3. The OOM boundary.**

Report it as the bracket the script measured, the largest S that ran and the smallest S
that failed, and state the resolution you searched at, 256 tokens by default. Do not
round that into a single number. The assignment explicitly penalises claiming an exact
boundary you did not search for. If you want a single token, rerun with
`--refine-resolution 1` and say so.

**D5. The first run measured the host, not the card.**

Worth its own paragraph, because it is the most interesting thing that happened. The
first session reported naive attention succeeding at peaks of up to 76.77 GiB on a
31.84 GiB card, with latency 500 times higher than the trend. Explain what the host was
doing, why `torch.cuda.max_memory_allocated` did not catch it, and what the cross check
against `torch.cuda.mem_get_info` changed. Give both boundaries, the 50688 you first got
and the corrected one, and note that the corrected figure agrees with solving
`32*S^2 + 4096*S` against physical VRAM. A measurement that disagrees with its own
arithmetic by 50% is the useful part of this assignment.

**D4. What the fused kernel avoids doing.** Three or four sentences. The substance to
cover:

* It never materialises the S by S score matrix in HBM at all. The naive version writes
  the scores, reads them back for the softmax, writes the weights, then reads those back
  for the second matmul, so the quadratic term is HBM traffic as well as HBM capacity.
* It tiles the computation and runs the softmax online over blocks, keeping a running
  maximum and sum so a block's contribution can be rescaled as later blocks arrive. Only
  order S statistics ever leave the SM.
* The QK product, the softmax and the AV product are fused into one kernel, so the
  intermediates stay in SRAM and registers instead of round tripping through HBM.
* The saving is bandwidth, not arithmetic. It does the same FLOPs. That is why the
  speedup tracks how memory bound the naive version had become, and why peak memory goes
  from quadratic to linear in S.

Write your three or four sentences and support them with your own speedup column and your
own peak memory ratio at each S.

## Part E: sustained load and thermal behaviour

Quote from Table HW2.5.1, plus `figures/part_e_clock_temp_<uuid>.png` and
`logs/part_e_thermal_<uuid>.csv`.

**E1. Did it throttle, and against what ceiling?**

The table reports the nvidia-smi reason bits where they fired. `SwPowerCap` means the
power limit, `SwThermalSlowdown` or `HwThermalSlowdown` means temperature. Give the
reason, the second it first appeared, and the temperature and power draw at that moment.
If nothing fired, say so and give the clock's range across the run as evidence.

**E2. Peak against steady state.**

Give peak TFLOPS over the first 30 seconds, steady state TFLOPS over the final 5 minutes,
and the ratio as a percentage. Then say what it means for a training run: a benchmark
quoted from the first few seconds overstates sustained throughput by that margin.

**E3. Why this number is yours alone.**

One or two sentences. It depends on the chassis airflow, the room temperature, the card's
own cooler and its power limit. Note the ambient conditions if you know them.

## Part F: the summary table

Table HW2.5.1 is generated into `METRICS.md`. Every cell traces to a UUID labelled line in
`RUN_LOG.txt`.

Add any commentary the table cannot carry: a measurement you would not trust, a run you
had to repeat, a discrepancy between two sessions on the same card.
