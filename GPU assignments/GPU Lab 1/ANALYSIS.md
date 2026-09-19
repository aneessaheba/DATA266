## Part F: the summary table

Table HW2.5.1 is in METRICS.md. Every cell traces to a UUID labelled line in
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

The lab Docker image ships PyTorch 2.1.2, which does not support the RTX 5090s Blackwell
architecture, CUDA capability sm_120, kernel launches fail with no kernel image is
available for execution on the device. PyTorch was upgraded in place inside the container
to 2.11.0+cu128 before any measurement was taken. This is necessary and affects every
number in this table.
