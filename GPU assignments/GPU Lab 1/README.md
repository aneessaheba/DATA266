# HW2.5 GPU Assignment I: Precision, Bandwidth, and the Cost of Attention

Step 0, same convention as assignment 1. Student ID 018205330, so `SID4=5330` and
`SEED=5330`. HW2.5 skips the second hyperparameter configuration, so no `HP_ID` is used.
Override with environment variables if needed: `SID4=xxxx SEED=xxxx ./run_all.sh`.

Every measurement in Parts B to E is labelled with the GPU UUID it came from, both in the
CSV it lands in and in `RUN_LOG.txt`.

## Layout

```
scripts/      the measurement code, one file per part, plus figure and table builders
provenance/   full nvidia-smi -q dumps, one per card (Part A)
data/         per card CSV and JSON results (Parts A to E)
logs/         the 5 second thermal sampling log (Part E)
figures/      every plot, regenerated from data/ and logs/
METRICS.md    generated, Table HW2.5.1 plus the per part detail tables
ANALYSIS.md   the written answers Parts B to E ask for, in your own words
RUN_LOG.txt   generated, append only, the UUID labelled trace behind every cell
reservations.md   reservation records and GPU hours reserved against consumed
AI_USE.md     AI use disclosure
```

## Running it

Parts B to E need a CUDA device. They exit with a message if run anywhere else, so none
of this can be run on a laptop and quietly reported as a card measurement.

On the reserved RTX 4090 or RTX 5090 workstation:

```bash
pip install torch matplotlib numpy       # if the box does not already have them
./run_all.sh                             # GPU 0, full 20 minute thermal run
GPU=1 ./run_all.sh                       # a second card in the same chassis
```

The allocated lab machine logs in as `.\anees`, so it is almost certainly Windows, where
`run_all.sh` will not run. Use `powershell -ExecutionPolicy Bypass -File run_all.ps1`, or
run the seven python commands by hand. `LAB_RUNBOOK.md` lists them and covers the other
Windows specifics.

`SMOKE=1 ./run_all.sh` is a roughly one minute end to end chain check. It shortens every
part, not just Part E, and must not be submitted. Clear `data/`, `logs/`, `figures/` and
`RUN_LOG.txt` afterwards. `MINUTES=n` shortens Part E alone.

`LAB_RUNBOOK.md` has the pre flight checks, the time budget per part, and what to verify
before releasing the machine.

Each script can also be run alone, and all of them take `--index` to choose the GPU:

| Part | Script | Produces |
|---|---|---|
| A | `part_a_provenance.py` | `provenance/nvidia-smi-q_*.txt`, `data/part_a_identity_*.json` |
| B | `part_b_precision.py` | `data/part_b_*.csv` (FP32, TF32, FP16, BF16, and the FP8 attempt) |
| C | `part_c_roofline.py` | `data/part_c_*.csv` |
| D | `part_d_attention.py` | `data/part_d_*.csv`, `data/part_d_summary_*.json` |
| E | `part_e_thermal.py` | `logs/part_e_thermal_*.csv`, `data/part_e_throughput_*.csv` |
| | `make_figures.py` | everything in `figures/` |
| | `make_metrics.py` | `METRICS.md` including Table HW2.5.1 |
| | `rehearse.py` | nothing committed, a no GPU dry run of the whole assignment |
| | `fakecuda.py` | nothing committed, the simulated device `rehearse.py` runs on |

`make_figures.py` and `make_metrics.py` need no GPU, so figures and tables can be rebuilt
anywhere once the CSVs are in hand.

## Rehearsing without a GPU

`SMOKE=1 ./run_all.sh` checks the chain on the lab box but needs CUDA and spends reserved
time. `python scripts/rehearse.py` does the same job from a laptop in about a minute, and
runs 80 odd assertions over the result.

Stage 1 runs every Part A to E script for real against the fake CUDA device in
`scripts/fakecuda.py`. Nothing is mocked at the script level. The actual measurement code
allocates, times, OOMs, bisects and logs. Only the device beneath it is simulated:
`device="cuda"` is rewritten to `device="meta"`, so a 16384 square matmul or a 32 GiB
score matrix costs no memory and no time, while a `TorchDispatchMode` charges every
allocation against a simulated VRAM ceiling and raises a genuine
`torch.cuda.OutOfMemoryError` when it is exceeded. That is what makes the Part D OOM
search real rather than a mock. `nvidia-smi` is intercepted too, covering both the Part A
dump and the Part E sampling loop.

Stage 2 runs `make_figures.py` and `make_metrics.py` over whatever Stage 1 wrote.

It then asserts that every script exits 0 on both simulated cards, that `RUN_LOG.txt` has
a header block per part and a UUID on every line, that every CSV is UUID labelled row by
row and records both `reps` and `warmup`, that Part D found an OOM bracket and its fitted
quadratic coefficient matches `B*H*2*itemsize` to within 1%, that the thermal log is not
ragged even across failed samples, and that `METRICS.md` has no unfilled cells.

The two simulated cards differ on purpose. One throttles and answers to the old
`clocks_throttle_reasons.active` field, the other does not throttle and only answers to
the newer `clocks_event_reasons.active`. One runs FP8 and the other refuses it. Between
them every reporting branch gets walked.

It writes nothing into the repository and none of its numbers are measurements. Run it
after touching any script. It is the cheapest place to find a renamed column, a ragged
row or a broken OOM search.

## Choices worth knowing about

* **Timing.** CUDA events, after a warmup, with the median of N timed repetitions
  reported and the repetition count written into every CSV row. The coefficient of
  variation is recorded too, so small variance is a number rather than a claim.
* **Theoretical peaks** live in `scripts/gpu_specs.json`, transcribed 2026-09-18 from the
  NVIDIA Ada whitepaper (Appendix A, Table 2, pp. 29 to 31) and the RTX Blackwell
  whitepaper (Appendix A, Table 3, pp. 46 to 47) themselves, not from product pages or
  secondary sources, with a `source` list and a per number `derivation` field so every
  figure can be defended rather than asserted. Two traps are baked into how those tables
  are printed, and both are documented in the file:
  * Every tensor row reads `dense/sparse`, the sparse figure carrying the whitepaper's
    footnote 2. Use the dense number. The 5090 product page headline of 3352 AI TOPS is
    the FP4 sparse figure, exactly twice the 1676 dense rate, not four times.
  * Every FP16 and FP8 row appears twice, once with FP16 accumulate and once with FP32
    accumulate, the former exactly twice the latter. cuBLAS runs `torch.matmul` on half
    inputs with FP32 accumulate, so the FP32 accumulate row is the honest denominator:
    165.2 TFLOPS BF16 on the 4090 and 209.5 on the 5090. The FP16 accumulate figures are
    kept alongside as `peak_tflops_dense_fp16_accumulate` for reference only.
* **Part B plateau.** Small matrices never reach the plateau because the kernel is launch
  and memory latency dominated. Work grows as N cubed while the data it reads grows as N
  squared, so arithmetic intensity is too low at N=1024 to keep the tensor cores fed.
  `make_metrics.py` computes the plateau size from the sweep rather than leaving it to be
  eyeballed off the figure.
* **Part D OOM boundary** is reported as a bracket, the largest sequence length that ran
  and the smallest that failed, refined by bisection to `--refine-resolution` tokens
  (default 256). Pass `--refine-resolution 1` for a genuine single token boundary.
  Anything coarser must be reported as a bracket, as the assignment insists.
* **Part D sweep extension.** The coarse sweep does not necessarily OOM. At
  `batch=1 heads=8 head_dim=64` in fp16, naive attention peaks near 8 GiB at S=16384,
  which fits on both a 24 GB 4090 and a 32 GB 5090. So when the sweep finds no failure
  the script keeps doubling S, up to `--max-extend` (default 131072), until something
  actually fails, then bisects. Fused attention is expected to hit the cap without ever
  OOMing, since its peak memory is linear in S, and that is reported as the finding
  rather than as a failure.
* **Part D quadratic** is fitted to the measured peak memory sweep with `numpy.polyfit`,
  and the coefficient is compared against `batch * heads * 2 * itemsize`, the bytes two
  S by S tensors would take. The fit confirms the quadratic, it is not assumed.
* **Part E throttle onset** is taken from the nvidia-smi reason bits, the card stating
  that it is throttling, decoded to names like `SwPowerCap` or `HwThermalSlowdown`,
  because the 95% of cold clock heuristic necessarily lags the real event. Both times are
  reported when both exist. The sampler probes for `clocks_throttle_reasons.active` and
  falls back to the newer `clocks_event_reasons.active`, retrying each name so that one
  transient nvidia-smi failure cannot cost the whole run its throttle evidence.
* **Figures** use one panel per measure rather than two y axes, so a 10.5 GHz memory
  clock cannot flatten a 2.7 GHz SM clock into a straight line.

## Before submitting

* [x] `scripts/gpu_specs.json` verified against the vendor whitepapers, source cited
* [x] `SID4` and `SEED` set from student ID 018205330
* [ ] `provenance/` holds the full `nvidia-smi -q` for every card used
* [ ] `reservations.md` filled in, hours reserved against hours actually consumed
* [ ] Part E ran the full 20 minutes
* [ ] `METRICS.md` has no `not measured` cells left
* [ ] `ANALYSIS.md` filled in: the Part B plateau, the roofline sides, the measured
      quadratic coefficient, what the fused kernel avoids, and the throttle story
* [ ] `python scripts/rehearse.py` passes
* [ ] `AI_USE.md` filled in
* [ ] repository tagged `hw2-5`
