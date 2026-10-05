# Software closure record

The pre-hardware software closure phase is superseded by the real deployment evidence in [REAL_ROBOT_RESULT.md](REAL_ROBOT_RESULT.md). The checkpoint loaded with its saved processors, produced finite 50×6 action chunks, and ran through the same async pipeline used for the real Stack2 demonstration.

The runtime was not a continuous 30 Hz stream of fresh predictions: WAN requests had P50 635 ms and P95 1,084 ms. At 30 Hz a 50-action chunk spans 1.667 s. The queue safely held the current target during 130 underflow ticks and never replayed a stale action. The task completed; a later retreat command crossed the saved shoulder-lift calibration bound and was correctly rejected.

See [BUGS_AND_FIXES.md](BUGS_AND_FIXES.md) for deployment regressions and fixes. Full per-tick evidence, device identities, calibration files, and videos are operator-local and are not stored in Git.
