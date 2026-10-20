# test: capstone-virtual, capstone-spatial-scrub, capstone-spatial; 1 repetition(s); host load bound 20.0

| benchmark | arm | reported s per repetition | median | ratio to capstone-spatial | QEMU CPU s (median) | nodes / collections / faults (sum over invocations, rep 1) | flagged repetitions |
|---|---|---|---|---|---|---|---|
| 502.gcc_r | capstone-virtual | **7** | 7 | 1.024 | -- | 5987/0/2051 | none |
| 502.gcc_r | capstone-spatial-scrub | **7** | 7 | 1.038 | -- | 1202/0/2052 | none |
| 502.gcc_r | capstone-spatial | **7** | 7 | 1.000 | -- | 1202/0/1765 | none |
| 544.nab_r | capstone-virtual | **124** | 124 | 1.039 | -- | 7194/0/446 | none |
| 544.nab_r | capstone-spatial-scrub | **120** | 120 | 1.006 | -- | 504/0/446 | none |
| 544.nab_r | capstone-spatial | **119** | 119 | 1.000 | -- | 504/0/446 | none |
| 538.imagick_r | capstone-virtual | **3** | 3 | 1.001 | -- | 2228/0/1543 | none |
| 538.imagick_r | capstone-spatial-scrub | **3** | 3 | 0.991 | -- | 362/0/1543 | none |
| 538.imagick_r | capstone-spatial | **3** | 3 | 1.000 | -- | 362/0/1542 | none |
| 519.lbm_r | capstone-virtual | **102** | 102 | 1.014 | -- | 40/0/104713 | none |
| 519.lbm_r | capstone-spatial-scrub | **101** | 101 | 1.006 | -- | 34/0/104713 | none |
| 519.lbm_r | capstone-spatial | **101** | 101 | 1.000 | -- | 34/0/104713 | none |

Reported seconds are runcpu's for the benchmark's own invocations; QEMU CPU seconds are the guest process's user+system time over the whole run, boot included. Counters come from the launcher's CAPSTONE_VM_STATS lines in the run directory's .err files.
