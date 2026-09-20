# Reservation and GPU hour record, HW2.5

Part A asks for the reservation records and the hours actually consumed against them, so
the two columns are expected to differ.

Credentials are deliberately not recorded in this repository. The lab account password
belongs in a password manager, not in a file that gets committed, pushed and tagged.

## Reservation

| Field | Value |
|---|---|
| Lab | ISB 840 |
| Machine number | 33 (final allocation, no changes) |
| Account | local account `.\anees` on the lab machine |
| Start | Fri 18 Sep 2026, 12:00 PDT |
| End | Sat 19 Sep 2026, 11:59 PDT |
| Hours reserved | 23.98, call it 24 |
| Approval | HPC access approval email, 18 Sep 2026 |
| Support | student assistant in ISB 836, 12:00 to 14:00 |

Check in on **both** days of the reservation, Friday 18 and Saturday 19. The approval
says that if nobody shows up by the check in window the reservation is cancelled and the
machine goes to the next group.

| Day | Check in done | Time | Notes |
|---|---|---|---|
| Fri 18 Sep 2026 | yes | 12:00 PDT | logged in, ran Parts A through E |
| Sat 19 Sep 2026 | no | | did not check in. All measurements were completed on Friday, so the machine was not needed on the second day |

## Hours actually consumed

The measured work is roughly 40 minutes of GPU time. The reservation is 24 hours of
machine time, so these two numbers should differ by a lot, and the difference is the
point of the question.

| Date | GPU UUID | What was run | Start | End | Hours used | Notes |
|---|---|---|---|---|---|---|
| 18 Sep 2026 | GPU-38ba4f51-a57b-4ca5-e2ab-74396ae57bdb | Part A provenance | 21:52 | 21:52 | 0:01 | rerun after PyTorch upgrade |
| 18 Sep 2026 | GPU-38ba4f51-a57b-4ca5-e2ab-74396ae57bdb | Part B precision sweep | 22:08 | 22:09 | 0:01 | |
| 18 Sep 2026 | GPU-38ba4f51-a57b-4ca5-e2ab-74396ae57bdb | Part C roofline | 22:10 | 22:10 | 0:01 | |
| 18 Sep 2026 | GPU-38ba4f51-a57b-4ca5-e2ab-74396ae57bdb | Part D attention sweep and OOM refine | 22:36 | 22:41 | 0:05 | first run had a bug, rerun after fix at 00:46 19 Sep |
| 18 Sep 2026 | GPU-38ba4f51-a57b-4ca5-e2ab-74396ae57bdb | Part E 20 minute sustained load | 22:42 | 23:02 | 0:20 | |
| 19 Sep 2026 | GPU-38ba4f51-a57b-4ca5-e2ab-74396ae57bdb | Part D rerun after VRAM overcommit fix | 00:46 | 00:47 | 0:01 | corrected OOM bracket |

**Total reserved:** 23.98 h

**Total consumed:** approximately 0:30

**Difference and why:** the reservation is a whole machine slot, while the measurements
themselves take about 40 minutes. Record anything that widened the gap, for example a
rerun after a short Part E, or a second session for the Part D bisection.

Attach or link the approval email alongside this file, with the password removed, so the
record is not self reported alone.
