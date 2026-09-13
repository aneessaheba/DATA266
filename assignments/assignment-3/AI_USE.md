# AI Use Disclosure for HW3

## 1. How I used AI

* Used Claude to help me learn the concepts behind this assignment:
  * The six prompt engineering techniques and what each one actually adds over the previous
    one, in particular the difference between Few Shot (teaching a format) and Chain of
    Thought (teaching a procedure)
  * How Tree of Thoughts differs from a single prompt that merely mentions branching, and why
    it needs a propose step, an evaluate step and real pruning to count as a search
  * Section 3.2 of Attention Is All You Need: why the dot products are scaled by the square
    root of d_k, and what the scaling does to the softmax gradients
  * Section 3.2.3 and why decoder self attention needs a causal mask, including why the mask
    is applied to the scores before the softmax rather than to the weights after it
  * Why attention on its own is permutation equivariant, and therefore why a positional
    embedding is needed before a next token objective can be fit at all

## 2. Something the AI got wrong

I built and ran the attention experiment myself. Before running it, I discussed what I
expected to see with Claude, and it guessed that an unmasked model trained on a next token
objective would learn an obvious lookahead pattern, with each token attending mainly to the
position right after it. That guess did not match what my own run measured: the mean row
maximum only reached 0.7593 after training, well short of a clean lookahead stripe, and
applying the causal mask to that same trained model at test time dropped its next token
accuracy from 1.000 to 0.391. There was real dependence on future tokens, just not in the
simple form that was guessed beforehand.

A second case: my first Tree of Thoughts implementation asked the model to check all four
puzzle constraints in a single evaluation call, and it pruned every branch including the
correct one, leaving no answer at all. I identified the bug and fixed it myself by splitting
evaluation into one focused question per constraint. That became one of the findings rather
than being quietly patched over.

## 3. How I checked everything

* Computed the ground truth for both prompting tasks in Python inside the notebook, by solving
  the rate problem arithmetically and brute forcing all 120 seatings for the logic puzzle, so
  model answers are graded against a number the notebook derives rather than one I asserted
* Scored each model output on the answer it actually committed to, after noticing that a plain
  substring check for "9:48" would mark a Tree of Thoughts answer of "9:49 am" as correct
  because the right value appears earlier in its reasoning
* Verified the causal mask numerically rather than by eye: every entry above the diagonal is
  exactly 0.0 and every row still sums to 1
* Confirmed the trained attention is learned and not random by capturing the mean row maximum
  before training and comparing it against the value after training
* Reran the notebook end to end and confirmed the numbers in this document match the saved
  notebook outputs exactly
