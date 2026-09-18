# Lab runbook, HW2.5

What to do on the reserved workstation, in order.

## Your reservation

ISB 840, machine 33, logging in as the local account `.\anees`. The slot runs from
Friday 18 September 2026 at 12:00 to Saturday 19 September 2026 at 11:59.

Check in on both days, Friday and Saturday. A student assistant is in ISB 836 from 12:00
to 14:00. If nobody checks in during that window the reservation is cancelled and the
machine goes to the next group, so check in on Saturday even though the measurements
will already be finished.

The measured work is about 40 minutes. Do the real run on Friday afternoon and keep
Saturday as the buffer for a rerun. Do not leave it to Saturday morning, because a
failed Part E would leave no time to repeat it.

The password is not recorded in this repository, and should not be. Keep it in a
password manager.

## If the machine is Windows

The `.\anees` login format means the lab machine is almost certainly Windows. That
changes a few things, none of them fatal, but all of them worth knowing before the clock
starts.

`./run_all.sh` will not run in cmd or PowerShell. Use the PowerShell wrapper:

```
powershell -ExecutionPolicy Bypass -File run_all.ps1
```

That wrapper has never been run on Windows. If it misbehaves, do not debug it on
reserved time. Run the five commands by hand instead, which is all the wrapper does:

```
python scripts\part_a_provenance.py --index 0
python scripts\part_b_precision.py --index 0
python scripts\part_c_roofline.py --index 0
python scripts\part_d_attention.py --index 0
python scripts\part_e_thermal.py --index 0 --minutes 20
python scripts\make_figures.py
python scripts\make_metrics.py
```

If WSL or Git Bash is installed, `./run_all.sh` works there and is the better option.

Other Windows specifics:

* `python` may be `py` on Windows. Check with `python --version` first, and pass
  `-Py py` to the wrapper if needed.
* `nvidia-smi` normally sits on the PATH with the driver. If it is not found, it is
  usually at `C:\Windows\System32\nvidia-smi.exe`. Every part needs it.
* Throttle reason bits can come back as N/A or unsupported on Windows. The sampler
  already falls back to the clock heuristic and logs a WARNING line, so check that line
  in `RUN_LOG.txt` rather than assuming the reason bits were captured. Either outcome is
  reportable for Part E, but you need to know which one you got.
* Windows reserves VRAM for the display, so usable memory is less than the 24 GB or
  32 GB on the box. The Part D OOM boundary will be lower than an equivalent Linux
  machine would give. That is a correct measurement of that machine, not a bug, and it
  is worth one sentence in `ANALYSIS.md`.
* Stop the machine sleeping or locking during the 20 minute Part E run. A display sleep
  partway through will show up as a clock and power artefact in the thermal log.

## Before you leave your laptop

* [ ] `python scripts/rehearse.py` runs every Part A to E script against a simulated CUDA
      device, then the figure and table builders, then about 80 assertions. Takes a
      minute and must print `REHEARSAL PASSED`. This is the cheapest place to find a
      broken column, a failed OOM search or a figure that writes nothing. The alternative
      is finding it at the end of a 20 minute thermal run on reserved time.
* [ ] Commit and push, so the box only has to clone.

`SID4` and `SEED` are already set to 5330, from student ID 018205330.

## On the box: pre flight, 2 minutes, do not skip

```bash
nvidia-smi                       # card present, and is it the one you reserved?
nvidia-smi -L                    # note the UUID, everything gets labelled with it
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import matplotlib, numpy; print('figures will build')"
```

If `torch.cuda.is_available()` is False, stop and fix that first. Every part except the
figure builders will exit immediately by design.

Then a one minute smoke test of the whole chain before committing to the real run:

```bash
SMOKE=1 ./run_all.sh
```

`SMOKE=1` shortens every part, not just Part E, so this really does take about a minute.
It produces junk numbers on purpose. Confirm it wrote `data/*.csv`, `figures/*.png` and a
`METRICS.md`, then delete those outputs so a smoke run can never be mistaken for the real
one:

```bash
rm -f data/*.csv data/*.json logs/*.csv figures/*.png RUN_LOG.txt
```

## The real run, about 40 minutes of GPU time

```bash
./run_all.sh 2>&1 | tee run_console.txt
```

Rough budget, so you know whether something has hung:

| Part | What it does | Expect |
|---|---|---|
| A | `nvidia-smi -q` capture, identity JSON | seconds |
| B | 4 sizes by 5 precisions, 10 warmup and 50 timed | 3 to 6 min |
| C | elementwise add, device copy, 8192 matmul | about 1 min |
| D | attention sweep, OOM extension, bisection | 5 to 10 min |
| E | sustained load, sampled every 5 s | 20 min exactly |

Note the wall clock start and end and put them straight into `reservations.md`. The
assignment wants hours consumed against hours reserved, and reconstructing that
afterwards from memory is how people lose marks.

## What to check before you release the machine

* [ ] `provenance/` has a full `nvidia-smi -q` dump, hundreds of lines, not empty
* [ ] `RUN_LOG.txt` has a PART A through PART E header block, all with the same UUID
* [ ] Part B: did FP8 run or fail? Either is fine, but read the failure text now. What
      you tried and how it failed is a required finding, and you want the real message
* [ ] Part D: `RUN_LOG.txt` shows an actual OOM and an `OOM bracket after N probes` line.
      If it says no OOM up to some S for naive, the extension cap was too low. Rerun that
      part with a higher `--max-extend`
* [ ] Part E: `logs/part_e_thermal_*.csv` has about 240 rows, 20 min at 5 s, and the
      throttle column holds hex values rather than `error:` or `unavailable`
* [ ] `METRICS.md` has no `not measured` cells and no short run warning

## If you get both cards

Run on the second card too. Nothing needs changing. Outputs are keyed by UUID, so the two
sets sit side by side and `make_metrics.py` emits a section per card.

```bash
GPU=1 ./run_all.sh
```

## Afterwards, back on the laptop

Figures and tables rebuild with no GPU, so iterate on the writeup locally:

```bash
python scripts/make_figures.py && python scripts/make_metrics.py
```

Then write the prose the assignment grades. `ANALYSIS.md` has a section per question: the
Part B plateau explanation, which side of the roofline each Part C op lands on, the
measured quadratic coefficient, what the fused kernel avoids doing, and the throttle
story. Each section names the generated number to quote. Fill that in, then `AI_USE.md`
and `reservations.md`, and tag:

```bash
git tag hw2-5 && git push --tags
```
