# AI Use Disclosure - HW1

## 1. How I used AI

- Used Gemini to help me understand the assignment topics better:
  - Autoregressive models
  - Feedforward neural networks
  - How training behaves (loss going down, epochs, etc.)
  - CUDA matrix multiplication

- Used Gemini to help me understand the code logic for PyTorch, TensorFlow, and CUDA:
  - How data preprocessing works
  - Training loss vs validation loss
  - What happens when you change the number of epochs
  - CUDA blocks and threads
  - How data moves between CPU and GPU memory
  - How profiling works
  - How to calculate speedup

## 2. Something the AI got wrong

- Gemini explained the standard deviation numbers incorrectly at first.
- It did not convert them from accuracy (a fraction) into percentage points properly.
- Example: a standard deviation of `0.0143` should be written as `1.43 percentage points`, not `0.0143%`. Gemini initially mixed these up.

## 3. How I checked everything

- I checked the explanations against my own executed notebook outputs.
- I looked at the loss curves, CUDA program output, and profiler results myself.
- I recalculated the standard deviations, GPU end-to-end times, and speedups on my own before putting them in my submission.
