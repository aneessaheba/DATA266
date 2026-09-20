AI Use Disclosure, HW2.5

1. How I used AI

Used Claude to learn concepts.
Asked it to explain how tensor cores work and why TF32, FP16, and BF16 have different peak throughput.
Asked it to explain arithmetic intensity and the roofline model (memory-bound vs. compute-bound).
Asked it why naive attention memory grows quadratically with sequence length, and what a fused attention kernel avoids doing.
Asked it why GPUs throttle under sustained load.
All benchmarks, logs, figures, and numbers come from my own runs on the reserved GPU.

2. Something the AI got wrong

AI explanations can be incomplete or imprecise, especially on hardware specs.
It can mix up dense and sparse peak TFLOPS, or give numbers for a different GPU or precision, so I did not use its figures as the "theoretical peak."
I used the vendor documentation cited in Part A for every spec number instead.

3. How I checked everything

Cross-checked AI explanations against NVIDIA documentation and course material.
Used only my own measurements for results, each labelled with the GPU UUID and traceable to RUN_LOG.txt.
Sanity-checked results: achieved TFLOPS below theoretical peak, effective bandwidth below specified bandwidth.
Confirmed the quadratic memory term by fitting my own data rather than assuming it.
Re-ran anything that looked off with more warm-up and repetitions.
