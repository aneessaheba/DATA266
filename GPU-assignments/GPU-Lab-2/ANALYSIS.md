# ANALYSIS, HW4.a and HW4.5

All numbers come from METRICS.md and the UUID labelled lines in RUN_LOG.txt. GPU: Google Colab Tesla T4, 15.6 GB.
The TA confirmed one GPU is enough, so every part was run on this one card type. Colab assigned two T4 instances
over the session: UUID GPU e98b1f84 for HW4.a, serving and Part D, and UUID GPU 72d7994b for Parts A and B.

## HW4.a

**Data.** `cardiffnlp/tweet_eval` sentiment, train split, rows 330 to 5329 (5000 rows), split 70/15/15 with SEED 5330:
3500 train, 750 validation, 750 test. The same rows and split are used in every HW4.5 part.

**Epoch where validation stopped improving.** Validation loss was lowest after epoch 1 (0.718) and rose every epoch
after that, to 0.980 at epoch 4, while training loss kept falling to 0.200. Validation accuracy peaked later, at epoch 3
(0.688), but only moved between 0.665 and 0.688, so the rising loss is the clearer signal: the model became more
confident on the examples it got wrong. Training past epoch 1 cost 122.8 s, which is 77% of the 160.2 s run. The saved
model and the test results (accuracy 0.659) come from the final, overfit epoch 4 model.

**Five misclassified tweets from the worst class (negative, F1 0.623). All five were predicted neutral.**

1. "She's invited. And Ash may not go because of her..." (sic: the apostrophe is stored escaped in the dataset). The
   complaint is told as a story with no negative words, so the annoyance is only implied.
2. "You may remember Frank Gifford as a great football player..." It opens with praise and the negative turn is crude
   innuendo, which the model does not read as sentiment.
3. "It's simple, we want to be like Israel, a white Israel..." A political statement with no emotional words, so the
   hostility comes from context the model does not have.
4. "Please pull down Tom Brady's pants and undress the cheaters..." The insult "cheaters" is buried inside a polite
   sounding request, and the request wording wins.
5. "Kane may have made it 5 0, I may have lost my bet. But guess what..." Mocking sarcasm mixed with a factual sports
   update, which reads as neutral on the surface.

**Latency against throughput.** No tested vLLM batch size had worse per request latency than the HF pipeline baseline:
p50 was 743 ms at batch 1, 821 ms at batch 8 and 1153 ms at batch 32, against 5009 ms for HF at batch 1. Inside vLLM,
latency rises with batch size (55% higher at batch 32 than at batch 1) while total throughput keeps climbing from 172 to
1247 to 3553 tokens/s, so the point where latency would cross the baseline lies beyond batch 32 on this card. For an
interactive application the per request latency, especially p95, is the number that matters, because a user waits for
their own reply and does not see total throughput.

Method limits: latency here is the time for the whole batch call to finish with exactly 128 new tokens per request.
Time to first token, which matters most for an interactive app, was not measured. The 8 prompts repeat inside the batch
8 and 32 runs, so vLLM prefix caching may help those runs slightly.

## Part A

**Where throughput stops improving.** Throughput rises from 14.7 samples/s at batch 1 to 118.1 at batch 64, then stays flat
(119.7 at 128, 117.7 at 256). Batch 64 is where it stops improving. The largest batch that runs is 256 (11.7 GB peak), and
512 runs out of memory. Only one card was used, so there is no second curve to compare.

These throughputs come from 5 timed steps per batch size, and the epoch times are extrapolated from them, not measured.
They are optimistic: at batch 32 they give 106.8 samples/s and 32.8 s per epoch, while the sustained Part B baseline ran
at 91.9 samples/s (about 14% lower) and the measured HW4.a epochs took 37 to 44 s. The shape of the curve is still valid.

**Max batch ratio against VRAM ratio.** With one card the ratio is not measured, so here is a projection from the T4 data.
About 2.2 GB is fixed (weights, gradients, Adam states, CUDA context) and each extra sample adds about 37 MB of
activations ((11.71 minus 2.21 GB) / 255). A 32 GB card would fit about (32 minus 2.2) / 0.037 = 800 samples, which the
power of two search rounds down to 512. That gives 512 / 256 = 2.0 against a VRAM ratio of 32 / 15.6 = 2.05, close only
because of the rounding. In general the two ratios are not proportional, because the fixed memory does not scale with
VRAM and the power of two steps snap the result to the nearest lower power of two.

## Part B

**Gradient checkpointing cost.** At the training batch of 32, throughput fell from 91.9 to 66.1 samples/s, which is 28%
slower, because each layer's forward pass is run again during backward. In return the max batch went from 256 to 1024.

**Gradient accumulation.** Physical batch 16 times 8 steps gives an effective batch of 128. It took 619.7 s for the same
436 optimiser steps that the baseline did in 151.8 s, because each step processes four times as many samples. Those
steps cover 55,808 samples, about 16 epochs over the 3500 training rows. Since HW4.a already overfit after epoch 1, the
lower accuracy (0.633 against 0.648) is most likely overfitting from the extra epochs. Accumulation has no max batch of
its own, so it is marked n/a. The training loop also dropped the last 2 of the 218 micro batches in each epoch.

**Why the baseline accuracy differs from HW4.a.** The Part B baseline (0.648) and HW4.a (0.659) use the same settings
but are separate runs: Part B uses drop_last, giving 436 optimiser steps instead of 440, and the shuffle order differs.

**Ranking.** The ranking depends on batch size, which is the main lesson of this part:

* At batch 32, activations are small. Mixed precision is first (smaller and 2.8 times faster), then the 8 bit optimiser
  (0.64 GB saved for 2 samples/s lost), then accumulation, and checkpointing is last (0.43 GB for 26 samples/s).
* At batch 256, activations dominate. Mixed precision is still first, but checkpointing moves up to second: it saves 8.7 GB
  (11.71 to 3.05 GB) at 0.185 GB per samples/s lost, against 0.032 for the 8 bit optimiser, which only shrinks the fixed
  optimiser state.

**What to apply first and what to skip on a 32 GB card.** Apply mixed precision first: it is the only technique that is both
faster and smaller at every batch size, and accuracy did not change (0.648). On a 32 GB card I would not bother with
gradient checkpointing for this model, because the projected max batch without it is about 512, well past the batch 64
where throughput stops improving, so its 28% speed cost buys nothing.

## Part C

Part C skipped (optional).

## Part D

**Rerun against HW4.** The rerun on the same T4 card type gave 165, 1202 and 3324 tokens/s at batch 1, 8 and 32, against
172, 1247 and 3553 in the HW4 run. That is 4% to 7% lower with latency 5% to 7% higher. Hardware, precision (fp16) and
script did not change, so this is normal run to run variation from host CPU load and GPU clock behaviour, not a real
difference.

**KV cache against context length (batch 8).** The measured KV cache matches the formula of 12,288 bytes per token
(2 for K and V, 24 layers, 2 KV heads, 64 dims, 2 bytes) and grows linearly: 0.10 GB per 1024 tokens at batch 8.

**Longest context at batch 8.** Measured with HF prefill: **3328 tokens** fit and 3584 fails. The KV cache at 3328 is only
0.33 GB of the 13.2 GB peak. The limit is the attention scores, which grow with the square of the context. As a check by
calculation, fp32 scores for 8 sequences x 14 heads at 4096 tokens need 8 x 14 x 4096 x 4096 x 4 bytes = 7.52 GB on top
of everything else, which cannot fit.

vLLM was not tested at long context: both vLLM runs used max_model_len 2048. By calculation only, that engine's KV pool
was 9.9 GiB = 865,456 tokens on this T4 (from its startup log), so 865,456 / 8 = 108,182 tokens per sequence would fit,
more than the model's native maximum of 32,768. Confirming that vLLM can actually serve 32,768 tokens at batch 8 would
need a run with max_model_len 32,768 and 8 long prompts.

## Part E

**GPU hours for one full fine tune.** The full train split is 70% of 45,615 tweets = 31,930 tweets per epoch.

| Configuration | Samples/s | 4 epochs | 1 epoch |
|---|---|---|---|
| Baseline, batch 32 | 91.9 | 127,720 / 91.9 / 3600 = 0.39 h | 31,930 / 91.9 / 3600 = 0.097 h |
| Mixed precision + 8 bit, batch 32 | 263.8 | 127,720 / 263.8 / 3600 = 0.13 h | 31,930 / 263.8 / 3600 = 0.034 h |

**Configuration I would use.** Mixed precision plus the 8 bit AdamW optimiser at batch 32, for 1 epoch, so about
0.034 GPU hours. At batch 32 it had the lowest peak memory (1.74 GB), the highest throughput (263.8 samples/s) and the
shortest wall clock (52.9 s for 436 steps), with accuracy 0.652, slightly above the baseline. One epoch is enough because
validation loss stopped improving after epoch 1. The same combination reached 406.7 samples/s at batch 256 in the probe,
about 1.5 times faster, but its accuracy was never validated at that batch and would likely need a retuned learning
rate, so I keep batch 32, the setting accuracy was measured at.
