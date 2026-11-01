# Reservation and GPU hour record, HW4.5

No lab reservation was used. The TA confirmed one GPU is enough, so all runs used Google Colab (free tier, Tesla T4).

| Date | GPU UUID | What was run | Time (PDT, local) | Hours used |
|---|---|---|---|---|
| 31 Oct 2026 | `GPU-e98b1f84-8d7b-50bb-a5c6-96b3440c7e56` | first run, stopped at vLLM startup (bug fixed) | 20:20 to 20:40 | 0.33 |
| 31 Oct 2026 | `GPU-e98b1f84-8d7b-50bb-a5c6-96b3440c7e56` | HW4.a, serving, Parts A, B, D (v2) | 20:52 to 21:36 | 0.73 |
| 31 Oct 2026 | `GPU-72d7994b-bdfd-5779-ea43-fbfd70d47b68` | Parts A and B rerun after the eval mode fix (v3) | 22:05 to 22:35 | 0.50 |

Total GPU time used: about 1.6 hours. The submitted Parts A and B come from the v3 run; everything else from the v2 run.

RUN_LOG.txt timestamps are in UTC, 7 hours ahead of PDT. For example the v2 hw4a line at 04:03 UTC is 21:03 PDT.
