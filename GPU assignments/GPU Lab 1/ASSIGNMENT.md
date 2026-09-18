# HW2.5 GPU Assignment I: Precision, Bandwidth, and the Cost of Attention

GPU Lab required: RTX 5090 or RTX 4090. Two weeks. Sits between HW3 and HW4, immediately
before attention and Transformers are taught in Week 3.

Follow the same step 0 instructions as assignment 1. The second model with different
hyperparameter configurations may be skipped for this assignment.

## Part A: onboarding and provenance

* Reserve and use time on either an RTX 5090 or an RTX 4090 workstation. Commit the
  reservation records and the GPU hours actually consumed against those reserved.
* Capture the complete `nvidia-smi -q` output on each machine and commit both files.
  Record GPU UUID, driver version, CUDA version, VRAM capacity and reported power limit.
* From vendor documentation record, for each card: architecture, memory type and
  bandwidth, tensor core generation, and the reduced precisions the tensor cores support.
  Cite the source. Every measurement in Parts B to E must be labelled with the UUID it
  came from.

## Part B: precision and achieved throughput

* Benchmark dense matmul at N = 1024, 4096, 8192 and 16384 in FP32, TF32, FP16 and BF16.
  Warm up before timing, time over enough repetitions that variance is small, and report
  the repetition count.
* Report achieved TFLOPS and achieved TFLOPS as a percentage of theoretical peak for that
  precision.
* Plot achieved TFLOPS against matrix size, one line per precision, on a single figure.
  State where each precision plateaus and why small matrices never do.
* If the stack exposes a lower precision, run it. If it does not, say what was tried and
  how it failed. That is a legitimate finding about tooling maturity.

## Part C: bandwidth bound against compute bound

* One memory bound operation, a large elementwise add or copy, and one compute bound
  operation, a large square matmul.
* For the memory bound case compute effective bandwidth in GB/s and report it as a
  percentage of specified bandwidth.
* Compute arithmetic intensity in FLOPs per byte for each and state which side of the
  roofline it falls on.

## Part D: the cost of attention

* Implement scaled dot product attention naively, materializing the full sequence by
  sequence attention matrix. State the head dimension and batch size chosen.
* Measure peak memory and forward latency at sequence lengths 512, 1024, 2048, 4096, 8192
  and 16384.
* Refine the OOM boundary to the smallest tested length that fails and the largest that
  succeeds. Report both. Do not claim an exact single token boundary without searching at
  single token resolution.
* Plot peak memory against sequence length and fit the curve. Confirm the quadratic term
  from the data and state the measured coefficient.
* Repeat with a fused or memory efficient attention implementation. Report the new OOM
  boundary and the speedup at each sequence length, and explain in three or four
  sentences what the fused kernel avoids doing.

## Part E: sustained load and thermal behaviour

* Run a sustained compute load for 20 minutes on each card, sampling GPU clock, memory
  clock, temperature, power draw and utilisation every 5 seconds. Commit the log.
* Plot clock and temperature against time on one figure. Identify whether throttling
  occurred, at what temperature or power ceiling, and how many seconds into the run.
* Report peak throughput in the first 30 seconds against steady state throughput in the
  final 5 minutes, as a percentage.

## Part F: analysis

One table summarising every measurement for the card:

| Measurement | Your GPU | Notes |
|---|---|---|
| Peak achieved TFLOPS (BF16) | | |
| % of theoretical peak (BF16) | | |
| Effective bandwidth (GB/s) | | |
| Naive attention OOM length | | |
| Fused attention OOM length | | |
| Steady state over peak throughput | | |
| Throttle onset (s, or none) | | |

Table HW2.5.1. Every cell must be traceable to a UUID labelled run in `RUN_LOG.txt`.

## Deliverables

Repository tagged `hw2-5`, capture of `nvidia-smi -q`, the thermal log, all figures,
`METRICS.md` with Table HW2.5.1, `RUN_LOG.txt`, `AI_USE.md`, and the reservation and GPU
hour record.
