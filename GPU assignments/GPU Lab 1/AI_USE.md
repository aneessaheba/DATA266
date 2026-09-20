# AI Use Disclosure, HW2.5

## How I used AI

Used Claude to learn concepts. Asked it to explain how tensor cores work and why TF32,
FP16, and BF16 have different peak throughput. Asked it to explain arithmetic intensity
and the roofline model, memory bound against compute bound. Asked it why naive attention
memory grows quadratically with sequence length, and what a fused attention kernel avoids
doing. Asked it why GPUs throttle under sustained load. All benchmarks, logs, figures,
and numbers come from my own runs on the reserved GPU.

## Something the AI got wrong

AI explanations can be incomplete or imprecise, especially on hardware specs. It can mix
up dense and sparse peak TFLOPS, or give numbers for a different GPU or precision, so I
did not use its figures as the theoretical peak. I used the vendor documentation cited in
Part A for every spec number instead.

## How I checked everything

Cross checked AI explanations against NVIDIA documentation and course material. Used only
my own measurements for results, each labelled with the GPU UUID and traceable to
RUN_LOG.txt. Sanity checked the results against what the hardware can do: effective
bandwidth came in below specified bandwidth at 86.8% and 83.8%, while nine of twenty
achieved TFLOPS figures came in above the rated peak, which I traced to the card
sustaining 2737 MHz against its 2407 MHz rated boost clock rather than to a measurement
error. Confirmed the quadratic memory term by fitting my own data rather than assuming
it. Recorded the coefficient of variation for every configuration, which flagged N=1024
as launch latency dominated at 21.5% to 62.6% against under 1.8% at N=8192, and reported
that in the analysis rather than presenting those points as precise.
