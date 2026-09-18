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
| Fri 18 Sep 2026 | | | |
| Sat 19 Sep 2026 | | | |

## Hours actually consumed

The measured work is roughly 40 minutes of GPU time. The reservation is 24 hours of
machine time, so these two numbers should differ by a lot, and the difference is the
point of the question.

| Date | GPU UUID | What was run | Start | End | Hours used | Notes |
|---|---|---|---|---|---|---|
| | | Part A provenance | | | | |
| | | Part B precision sweep | | | | |
| | | Part C roofline | | | | |
| | | Part D attention sweep and OOM refine | | | | |
| | | Part E 20 minute sustained load | | | | |

**Total reserved:** 23.98 h

**Total consumed:** hh:mm

**Difference and why:** the reservation is a whole machine slot, while the measurements
themselves take about 40 minutes. Record anything that widened the gap, for example a
rerun after a short Part E, or a second session for the Part D bisection.

Attach or link the approval email alongside this file, with the password removed, so the
record is not self reported alone.
