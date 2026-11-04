# C.5a on silicon: the kit session on board 2 (4 Nov), result

**The session.** Board 2 ran c5a-b2.sh, relaunched at 16:57 after a void first attempt (described at the end). It was pre-registered in PREREG.md and Addenda 1-5.
- **Before the flash:** the slot kit on bitstream 3.
- **The flash:** caplifive_vp2c_1fc2cd5aa, 17:32-17:34. The sealed sha256 was checked locally, and the resident name was read back after the power cycle.
- **After the flash:** the R-67, hpm, ObjBounds and slot kits on C.5a.
- **Grading.** Every row was graded here by script, and the rtl lane graded it independently.

**Graders.**
- R-67: land/rtl-r67-fixed 63e60355c408.
- hpm and ObjBounds: land/rtl-c5a-kits 5b6fb6a4ec9b. This version also checks the comparison dir's bitstream, and the runner's own copies from 6dae10292923 gave the same verdicts.
- slot: land/rtl-slot-kit 98ff08e6f3fb, with the cycle band.

## Readings

| Kit | Verdict | Readings |
|---|---|---|
| slot, bitstream 3 (board 2) | AS EXPECTED, ALL IN BAND | ldc 6.01, stc 8.01, pair 16.97 (same and other slot), sd+ld 5.97 / 1.95 cycles per op over the loop. This equals board 1's run to the cycle. |
| R-67 | **FIXED** | t1-fire `0x7101 0x800009e0 0x1 0x0 0x0 0x0 0x1 0x19 0x0 0x800009e0 0x59`, the pre-registered fixed vector: a jump through an ended code capability no longer runs its target |
| hpm | **FINDING** | l1d_miss (an exact field) reads 1 where the sim reads 0 on hpm-5, -1, -2, -3 and -4, and 257 where it reads 256 on hpm-7. hpm-6, which makes no data access, reads 0. l1d_acc equals the sim everywhere. The lifetime query's stall: hpm-3 509, hpm-4 254 (sim 511 / 256; ObjBounds 511 / 256; bitstream 3 1,530 / 1,530). Cycles: hpm-3 1,306, hpm-4 1,562 (bitstream 3 2,344 / 2,852). |
| ObjBounds | AS EXPECTED, cost pattern HOLDS (9 of 9 clauses) | Identity images and the witness are exact: stale nodes read 25 and the live control 0. objb-pcc does not return (the dead PCC is refused), as predicted. Cycles against bitstream 3: s2 0.557, s4 pass 2 0.614, s6 0.560, s7 0.549 lower; s3 1.057 and s4 pass 1 2.050 higher; s1 1.0018 and s5 pass 1 0.998 equal. |
| slot, C.5a | AS EXPECTED, ALL IN BAND | ldc **2.03**, stc **3.02**, pair **8.01** (same and other slot), sd+ld 5.97 / 2.00 cycles per op over the loop (sim 2.08 / 3.44 / 8.42) |

## What this says

- **The cached lifetime verdict reaches silicon.** An isolated LDC falls from 6.0 to 2.0 cycles over the loop, an STC from 8.0 to 3.0, and the STC→LDC pair from 17.0 to 8.0. The lifetime query's stall in the hpm kit falls to ObjBounds' value. Where the cache misses, cost rises as predicted (s3, s4 pass 1).
- **R-67 is fixed on silicon.**
- **Open: the extra L1D miss.** It is new on C.5a, because bitstream 3 and ObjBounds read the predicted value, and N = 1. It is one miss per run, not a cost. Whether it belongs to the bitstream or to this session's board state is decided by the hpm repeat in a fresh C.5a session (Addendum 6, perf/c5a-b2/hpm-r2).
- **Forwarding.** It was a store-buffer candidate. These readings confirm the sim, and the rtl lane's recommendation stands: forwarding would buy about 0 on C.5a alone, and at most about 2 per same-slot pair on top of stage A.

## The void first attempt

The first relaunch of the session (16:44-16:56) read `UART begin=False` on every slot image. Board 2's virtual switches 0, 1, 2, 5, 6 and 7 had been left on by the x264 train driver killed at 16:42, and sw[0] raises `uart_debug_takeover` (ariane_xilinx.sv:808), so the console pin carries the core's debug UART instead of the 16550. The switches were parked at 0, that run's directory was kept apart as `perf/c5a-b2/slot-bs3.noUART-switches`, and none of it is graded.

SPEC test on C.5a runs next (Addendum 5, `c5a-spec-b2.sh`). Its grade, S1-S3, will be RESULT.md.
