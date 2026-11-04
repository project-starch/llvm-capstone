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

## The C.5a arm (pre-registered 2026-11-04, before any C.5a boot)

The C.5a bitstream is capstone-ariane 1fc2cd5aa (record: `docs/plans/2026-11-03-c5a-bitstream.md`). These seven images
were simulated on c5a-wt a642f4891, whose core is identical to 1fc2cd5aa, with PROVENANCE `rtl_dirty=no`, memory delay
12 and time_out 3,000,000. The ELFs reproduce `images/SHA256SUMS` byte for byte, and retired counts differ per image.

| image | C.5a vector |
|---|---|
| hpm-5 | 0x0 0x0 0x0 0x0 0x100 0x0 0x333 0x203 0x5 0x0 0x0 |
| hpm-1 | 0x100 0x0 0x100 0x0 0x100 0x0 0x461 0x303 0x5 0x0 0x0 |
| hpm-2 | 0x0 0x100 0x100 0x0 0x100 0x0 0x4fd 0x303 0x5 0x0 0x0 |
| hpm-3 | 0x0 0x0 0x100 0x0 0x100 0x1ff 0x55e 0x303 0x5 0x0 0x0 |
| hpm-4 | 0x0 0x0 0x100 0x0 0x100 0x100 0x715 0x303 0x5 0x0 0x0 |
| hpm-6 | 0x0 0x0 0x0 0x0 0x0 0x0 0x48c 0x303 0x5 0x0 0x0 |
| hpm-7 | 0x100 0x0 0x100 0x100 0x100 0x0 0x514e 0x403 0x5 0x0 0x0 |

- Every counter field equals ObjBounds'. hpm-3's stall is 511 and hpm-4's is 256. Cycles are exactly 7 below
  ObjBounds' on all seven images, which is the PC tracker's constant shift in the C.5a record.
- **What this kit can and cannot see.** Every EXACT field reads the same on bitstream 3, ObjBounds and C.5a. Only stall
  and cycles separate them, and those are timing fields. So the kit identifies C.5a only through the resident name and
  the cost pattern against a bitstream 3 run. It cannot tell C.5a from ObjBounds: both remove the query's stall.
- **The grade:** `grade/hpm-grade.py <run> c5a <bitstream 3 run>`. AS EXPECTED with the pattern HOLDING is rc 0. The
  arm REQUIRES the bitstream 3 dir; without it the grader gives NO DATA (rc 2).
- **Controls, run before any C.5a boot**, on mock dirs built from each tree's simulated vectors, with the committed
  images for the hash:
  - C.5a's vectors give rc 0;
  - bitstream 3's vectors under the C.5a name give COST PATTERN FAILS, rc 1;
  - the wrong resident name gives WRONG BITSTREAM, rc 1;
  - no bitstream 3 dir gives rc 2, and an empty dir gives rc 2.
  - The existing arms still read rc 0 on the real board 1 (bitstream 3) and board 2 (ObjBounds) runs.
- **Prediction:** AS EXPECTED, with the cost pattern HOLDING against board 1's bitstream 3 run.

## The C.5a reading (2026-11-04, caplifive_vp2c_1fc2cd5aa, board 2, the runtime lane's run 17:38-17:44)

**Grade.** `hpm-grade.py <run> c5a perf/hpm/run-b1-bs3` (land/rtl-c5a-kits 5b6fb6a4ec9b) gives **FINDING**, rc 1. The
comparison dir reads e7af48457, OK. Result lines: spec-silicon f45274ebafe0.

**The finding.** l1d_miss, an EXACT field, reads one more than the pre-registered value on every image that makes a data
access. With N = 1 it cannot yet be separated from the session's board state. It is a counter difference, not a cost.

| image | l1d_miss (pre-reg) | l1d_acc |
|---|---|---|
| hpm-1 .. hpm-5 | 1 (0) | equal to sim |
| hpm-7 | 257 (256) | equal to sim |
| hpm-6 (no data access) | 0 (0) | 0 |

- On bitstream 3 (board 1) and on ObjBounds (board 2), every image read l1d_miss as pre-registered.
- Simulating the same images on 1fc2cd5aa reads 0, so this is silicon against sim.
- No mechanism is known.
- A repeat in a fresh C.5a flash and session (perf/c5a-b2/hpm-r2, the runtime lane's Addendum 6) decides between the
  bitstream and the session.

**The lifetime query's cost on silicon.** The grader prints no cost pattern on a FINDING, so these figures come from the
raw SV lines of the three runs.

| | hpm-3 stall | hpm-4 stall | hpm-3 cycles | hpm-4 cycles |
|---|---|---|---|---|
| C.5a | 509 | 254 | 1,306 | 1,562 |
| ObjBounds | 511 | 256 | 1,331 | 1,583 |
| bitstream 3 | 1,530 | 1,530 | 2,344 | 2,852 |

- The cached verdict removes the query's stall as ObjBounds does.
- **CORRECTED 2026-11-04 (the same evening):** this line said every image runs 20-24 cycles below bitstream 3, and that
  is false for hpm-7.
  - hpm-5, -1, -2 and -6 run 21-24 cycles faster: ratios 0.974, 0.978, 0.979 and 0.981. That fails the pre-registered
    "equal within 1 %" clause.
  - hpm-7, the cold-miss image, reads 11,084 against 11,082, which holds.
  - So it is not a constant shift.
  - The grader now prints the cost pattern on a FINDING (land/rtl-c5a-kits cd572819613f), and these lines come from
    that output.
  - It is a second deviation in this one session, beside the extra L1D miss; both wait for the repeat.

## hpm-8: the L1I-miss event's positive control (added 2026-11-04, before any board run of it)

**Why.** Event 1 (L1I miss) was never exercised by hpm-1..7. The SPEC lane now wants to read it: is the excess per removed
STC/LDC pair in compiled code fetch? A reading from a counter that has never been shown to fire is not evidence.

**The source.** Event 1 is `l1_icache_miss_i` (`perf_counters.sv:108`), fed by `cva6_icache.sv` `miss_o`. That pulses
once per cacheable refill request (`:347`).

**The image.** hpm-8 is built from the same `tests/hpm-probe.S` with `-DHPM=8`:
- size 69,544 B, sha256 0094235951c3e398...;
- board_rec 0x80010000.

**Rebuild check.** Rebuilding hpm-1..7 from the edited source gives byte-identical images to the committed SHA256SUMS.

**The arm.**
- Counter 6 is reprogrammed to event 1.
- An 8 KiB block of 2,048 four-byte NOPs at 0x80000880..0x80002880 (`norvc`, checked in the disassembly) is called twice.
  That is 512 lines of 16 B, half the 16 KiB 4-way I-cache.
- It prints seven values: pass 1 misses, pass 2 misses, pass 1 cycles, pass 2 cycles, the mhpmevent6 readback, traps,
  and the trap cause.

**The vector.** Simulated from this image on bitstream 3's tokens (rtl-ids20-wt c3f5102f1, rtl_dirty=no, memory delay 12):
`0x203 0x1 0x6636 0x834 0x1 0x0 0x0`.
- 515 cold misses: 512 for the block, plus its call and loop lines.
- 1 warm miss.
- 26,166 against 2,100 cycles, which is 46.8 sim cycles per I-miss.

**Graded by `grade/hpm8-grade.py <run dir>`. Any bitstream may run it.**
- EXACT: SR 7, readback 1, traps 0, cause 0.
- COUNTS: pass 1 misses in 480..560, and pass 2 misses at most 8.
- Cycles are reported, with the per-miss cost. That cost is silicon's I-miss penalty, which the SPEC lane needs to turn an
  L1I-miss delta into cycles.
- AS EXPECTED (rc 0) means the event counts on that bitstream. FINDING (rc 1) means it does not, or an exact field is
  off. NO DATA is rc 2.

**Controls** (fake run dirs):

| case | result |
|---|---|
| the simulated vector | AS EXPECTED |
| pass 1 at 0 | FINDING |
| the warm pass not warm | FINDING |
| readback 2 | FINDING |
| a different image | FINDING |
| no data | NO DATA |

### hpm-8 on silicon (2026-11-05 03:16, board 1, caplifive_vp2c_e7af48457.bit, the runtime lane's run)

The run dir is `$CAPSTONE_WORK/perf/all-b1/hpm8`. It ran after the imagick dev/devmk counter boots in the same session, on
the same bitstream. `hpm8-grade.py` on it, run in the RTL lane and in the runtime lane, gives the same lines, rc 0:

- image OK (sha256 = SHA256SUMS), resident `caplifive_vp2c_e7af48457.bit`, SR 7;
- vector `0x203 0x1 0x3486 0x81b 0x1 0x0 0x0`: exact fields OK (readback 1, traps 0, cause 0);
- L1I misses: pass 1 515 (band 480..560), pass 2 1 (at most 8): COUNTS;
- cycles: pass 1 13,446, pass 2 2,075, so **22.1 cycles per L1I miss on silicon** (sim 46.8, memory delay 12).

**Verdict: AS EXPECTED.** The miss counts equal the simulation's exactly (515 / 1). Event 1 counts on bitstream 3, so an
L1I-miss delta read on bitstream 3 is a reading from a counter shown to fire. The runtime lane's use of it, the imagick
dev → devmk counter boots in which L1I barely moves and the cycle drop is 68-70 % issue stall, is recorded in that lane's
RESULT (`spec-silicon` 889e0ee98aad), not here.
