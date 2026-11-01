# ANALYSIS, HW4.a and HW4.5

All numbers come from METRICS.md and the UUID labelled lines in RUN_LOG.txt. GPU: Google Colab Tesla T4, 15.6 GB.
The TA confirmed one GPU is enough, so every part was run on this one card type. Colab assigned two T4 instances
over the session: UUID GPU e98b1f84 for HW4.a, serving and Part D, and UUID GPU 72d7994b for Parts A and B.

## HW4.a

**Data.** `cardiffnlp/tweet_eval` sentiment, train split, rows 330 to 5329 (5000 rows), split 70/15/15 with SEED 5330:
3500 train, 750 validation, 750 test. The same rows and split are used in every HW4.5 part.

**Epoch where validation stopped improving.** Validation loss was lowest after epoch 1 (0.718) and rose every epoch
after that, to 0.980 at epoch 4, while training loss kept falling to 0.200. Training past epoch 1 cost 122.8 s, which is
77% of the 160.2 s run, and only made the model overfit. Validation accuracy moved around 0.665 to 0.688, so the extra
epochs bought no real gain.

**Five misclassified tweets from the worst class (negative, F1 0.623). All five were predicted neutral.**

1. "She's invited. And Ash may not go because of her..." The complaint is told as a story with no negative words, so the
   annoyance is only implied.
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

## Part A

**Where throughput stops improving.** Throughput rises from 14.7 samples/s at batch 1 to 118.1 at batch 64, then stays flat
(119.7 at 128, 117.7 at 256). Batch 64 is where it stops improving. The largest batch that runs is 256 (11.7 GB peak), and
512 runs out of memory. Only one card was used, so there is no second curve to compare.

**Max batch ratio against VRAM ratio.** With one card there is no measured ratio. The T4 numbers show why the two ratios
are never equal: about 2.2 GB is fixed (weights, gradients, Adam states, CUDA context) and does not shrink with batch,
and only the rest holds activations at about 37 MB per sample. Doubling VRAM therefore more than doubles the room for
activations, and the power of two search rounds the result down, so the batch ratio does not follow the VRAM ratio.

## Part B

**Gradient checkpointing cost.** Throughput fell from 91.9 to 66.1 samples/s, which is 28% slower, because each layer's
forward pass is run again during backward. In return the max batch went from 256 to 1024, four times larger.

**Gradient accumulation.** Physical batch 16 times 8 steps gives an effective batch of 128. It took 619.7 s for the same
436 optimiser steps that the baseline did in 151.8 s, because each step now processes four times as many samples.
Accuracy was 0.633 against 0.648, since a larger batch with the same learning rate takes smaller steps per sample.

**Ranking.** Mixed precision is first: it saves memory and is 2.8 times faster. The 8 bit optimiser is second (0.64 GB saved
for 2 samples/s lost), accumulation third, and checkpointing last (0.43 GB saved for 26 samples/s lost).

**What to apply first and what to skip on a 32 GB card.** Apply mixed precision first: it is the only technique that is both
faster and smaller, and accuracy did not change (0.648). On a 32 GB card I would not bother with gradient checkpointing,
because this fine tune peaks under 3 GB at batch 32 and checkpointing only costs speed when memory is not the limit.

## Part C

Part C skipped (optional).

## Part D

**Rerun against HW4.** The rerun on the same T4 card type gave 165, 1202 and 3324 tokens/s at batch 1, 8 and 32, against
172, 1247 and 3553 in the HW4 run. That is 4% to 7% lower with latency 5% to 7% higher. Hardware and precision (fp16)
did not change, so this is normal run to run variation on a shared Colab machine, not a real difference.

**KV cache against context length (batch 8).** The measured KV cache matches the formula of 12,288 bytes per token
(2 for K and V, 24 layers, 2 KV heads, 64 dims, 2 bytes), and it grows linearly: 0.10 GB per 1024 tokens at batch 8.
Plain HF prefill fits up to 3328 tokens and fails at 3584, but the KV cache there is only 0.33 GB of the 13.2 GB peak.
The limit comes from the attention scores during prefill, which grow with the square of the context.

**Longest context at batch 8.** vLLM on this T4 reserved 9.9 GiB of KV cache, which holds 865,456 tokens (from the
vLLM startup log). Batch 8 at the model's full 32,768 token context needs only 262,144 tokens, so with vLLM the T4 serves
the model's maximum context of 32,768 at batch 8. Only the naive HF prefill is limited to 3328.

## Part E

**GPU hours for one full fine tune.** The full train split is 70% of 45,615 tweets = 31,930 tweets, times 4 epochs =
127,720 samples. At the baseline 91.9 samples/s that is 127,720 / 91.9 / 3600 = 0.39 GPU hours. With mixed precision and
the 8 bit optimiser at 263.8 samples/s it is 127,720 / 263.8 / 3600 = 0.13 GPU hours.

**Configuration I would use.** Mixed precision plus the 8 bit AdamW optimiser at batch 32. It had the lowest peak memory
(1.74 GB), the highest throughput (263.8 samples/s) and the shortest wall clock (52.9 s for 436 steps), with accuracy
0.652, slightly above the baseline. I would also stop after 1 or 2 epochs, since validation loss stopped improving after
epoch 1.
