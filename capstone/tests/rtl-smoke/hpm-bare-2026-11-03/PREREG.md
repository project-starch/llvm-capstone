# hpm-bare-2026-11-03: pre-registration

This is written and pushed before any board run.

**Images.** `images/SHA256SUMS` pins them. They were built by `build.sh` from `tests/hpm-probe.S`, which equals
capstone-ariane `hpm-probe-b3` 87368cc69 and `hpm-probe-objb` fc1114913. Every image has `board_rec` 0x8000c000.

| image | sha256 |
|---|---|
| hpm-5 | dcc339ee4c1940ff0ef58ef0b98ccc5be84011f01e232113f59e1c790b61a004 |
| hpm-1 | 4a16a2203c8ba9c4547f2b1b6d66ad805cdef915c62fbc9fe4936bfc868d0b34 |
| hpm-2 | 22b8c6bf2ae23af45ad8dfef77d11103790c394301fdbb5db15dec9fe2fc0ecc |
| hpm-3 | 87938ddad16920fcb1e42f66ba366b2397c976f5ea5a7f38baf997b2e98c588a |
| hpm-4 | 1bcb3379d9c00ab087bc24a676ce352af49844b9fce163f67295f4502c9a7cbb |
| hpm-6 | 0bf4d19bbe4cd1304dcad5376f88ebd9488c4f037d552ef697c3702fa98356de |
| hpm-7 | 29d77f372692a5cd1b7245e951649d280b97ee425bf9f487ae02ff08a5388029 |

**Where the vectors come from.** They are simulated from THESE images through the testharness:
- memory delay 12, time_out 3,000,000;
- the recorder area is read by `board-vector.py`;
- bitstream 3's tokens: vp9 87368cc69, PROVENANCE `rtl_dirty=no`;
- ObjBounds: vp8 fc1114913, PROVENANCE `rtl_dirty=no`.

Every image ends in the wrapper's idle loop at the cycle limit, which is how a bare image ends. SR is 11 on every image.

**Field order:** load, store, l1d_acc, l1d_miss, branch, stall, cycles, instret, ev3, traps, cause.

| image | bitstream 3 (caplifive_vp2c_e7af48457) | ObjBounds (caplifive_vp2c_603528b8b) |
|---|---|---|
| hpm-5 empty | 0x0 0x0 0x0 0x0 0x100 0x0 0x33a 0x203 0x5 0x0 0x0 | same |
| hpm-1 ld | 0x100 0x0 0x100 0x0 0x100 0x0 0x468 0x303 0x5 0x0 0x0 | same |
| hpm-2 sd | 0x0 0x100 0x100 0x0 0x100 0x0 0x504 0x303 0x5 0x0 0x0 | same |
| hpm-3 ldc | 0x0 0x0 0x100 0x0 0x100 0x5fa 0x95a 0x303 0x5 0x0 0x0 | 0x0 0x0 0x100 0x0 0x100 0x1ff 0x565 0x303 0x5 0x0 0x0 |
| hpm-4 stc | 0x0 0x0 0x100 0x0 0x100 0x5fa 0xc10 0x303 0x5 0x0 0x0 | 0x0 0x0 0x100 0x0 0x100 0x100 0x71c 0x303 0x5 0x0 0x0 |
| hpm-6 inhibit | 0x0 0x0 0x0 0x0 0x0 0x0 0x493 0x303 0x5 0x0 0x0 | same |
| hpm-7 cold stride | 0x100 0x0 0x100 0x100 0x100 0x0 0x5155 0x403 0x5 0x0 0x0 | same |

**Grading rule.** `grade/hpm-grade.py <run dir> <bs3|objb> [<other arm's dir>]` applies it.
1. Each image's sha256 must equal its row above. The resident bitstream name in run.log must name the arm, or the
   verdict is WRONG BITSTREAM.
2. These fields must be EXACT: load, store, l1d_miss, branch, instret, ev3, traps and cause, plus every field of
   hpm-6. Any exact field off is a FINDING about the counters, not about the test.
3. cycles, stall and l1d_acc (outside hpm-6) are TIMING fields. Their simulated values come from delay 12, so they are
   reported and not required.
4. The cost pattern, with both arms read:
   - hpm-3 and hpm-4 have lower stall and lower cycles on ObjBounds, and the stall drop is within 15 % of the cycle
     drop. In simulation the drop per op is 3.98 / 3.96 for LDC and 4.98 / 4.95 for STC.
   - Every other image's cycles are equal across the bitstreams within 1 %.
5. Exit codes: 0 = AS EXPECTED (and the pattern HOLDS); 1 = anything else; 2 = NO DATA.

**Grader controls, run before this was pushed, on run directories fabricated from the vectors above:**
- bitstream 3 vs ObjBounds, each way: AS EXPECTED and HOLDS, rc 0.
- bitstream 3 data graded as objb: WRONG BITSTREAM, rc 1.
- hpm-3 with LDC counted as a load: FINDING, rc 1.
- ObjBounds replaced by bitstream 3 data: the pattern FAILS, rc 1.
- An empty directory: NO DATA, rc 2.

**What each reading decides.**
- If load and store read 0 on hpm-3 and hpm-4, LDC and STC are invisible to events 5 and 6 on silicon too. SPEC's
  capability-access counts then have to come from elsewhere.
- If the hpm-3 and hpm-4 stall drop equals the cycle drop, event 22 measures the lifetime check's non-overlapped cost
  on silicon.
- hpm-6 confirms that `mcountinhibit[8:3]` stops the counters. That is the bit the SPEC lane's firmware change clears.
- hpm-7 shows event 2 counting one miss per cold line.

**Limits.**
- Events 1, 4, 10 and 14 are not exercised. DTLB misses are 0 in bare M because there is no translation.
- One run per image, N = 1.

**History.** The first cut of arm 7 strode over `.data` (eedb57e45 / 58f84f0b6). Its kit simulation read 65 L1D misses
of 256, because the wrapper's prologue loads every data doubleword and `.data` is the D$ size. The arm was moved to its
own capability on 0x80100000 before anything ran on a board.

## Correction after board 1's reading (2026-11-03, bitstream 3, board 1, runtime lane's run 19:25-19:35)

Rule 2 said "every field of hpm-6" is exact. On board 1, hpm-6 read every counter as predicted (load, store, l1d_acc,
l1d_miss, branch and stall all 0; ev3 5; 0 traps). Its CYCLES read 1,074 against the simulator's 1,171, and under the
rule as written the arm graded FINDING for that reason alone.
- Cycles is a timing field on every other image: hpm-1 read 1,078 against 1,128 on the same boot. The arm exists to
  show that `mcountinhibit` stops the six counters, and cycles is mcycle, which the inhibit does not touch.
- The rule is corrected to: every field of hpm-6 EXCEPT cycles. This correction is made AFTER the reading and is
  recorded here as such.
- Under the rule as written the verdict is FINDING (hpm-6 cycles only). Under the corrected rule it is AS EXPECTED.
- The grader's controls were re-run after the change: LDC counted as a load gives FINDING, a wrong bitstream gives
  WRONG BITSTREAM, no check removed gives the pattern FAILS, and an hpm-6 counter reading 5 gives FINDING.

Board 1 by value, every exact field of all 7 images as pre-registered:
- hpm-3 and hpm-4 stall read 1,530, the simulator's count exactly;
- LDC and STC count as neither load nor store;
- hpm-7 counts 256 misses for 256 lines.

## Board 2's reading (2026-11-04, ObjBounds caplifive_vp2c_603528b8b, board 2, the runtime lane's run 04:52-05:09)

The rules are as corrected above; nothing was changed after this reading. Graded by `grade/hpm-grade.py`
(`run-b2-objb objb run-b1-bs3`), rc 0. Every image hash and the resident name were checked. The runtime lane's grade is
the same.

- Every exact field of all 7 images reads as pre-registered. hpm-3's stall reads 511 and hpm-4's 256, the simulator's
  ObjBounds counts exactly. **ARM objb: AS EXPECTED.**
- **The cost pattern, bitstream 3 (board 1) → ObjBounds (board 2), HOLDS:**
  - hpm-3: stall 1,530 → 511 (−1,019), cycles 2,344 → 1,331 (−1,013), 3.98 / 3.96 per LDC.
  - hpm-4: stall 1,530 → 256 (−1,274), cycles 2,852 → 1,583 (−1,269), 4.98 / 4.96 per STC.
  - hpm-1, -2, -5, -6 and -7 read equal cycles; the largest difference is hpm-7, 11,082 → 11,081.
- So on silicon the lifetime query's cost shows up as event 22 stall cycles, and removing the query removes about as many cycles (−1,013 of −1,019; −1,269 of −1,274). LDC and
  STC count as neither load nor store on both bitstreams.

**Limits.** N = 1 per bitstream, and the two bitstreams ran on different boards. Cycles are not the simulator's on
either board: hpm-7 reads 11,081 against the simulator's 20,821.
