# CSMINT kit on silicon: board lane run plan (2026-11-23, before the seal exists and before any board run)

The kit is the rtl lane's, pre-registered at dev 4b02223df217 (`capstone/tests/rtl-smoke/csmint-bare-2026-11-23/PREREG.md`).
The PREREG holds the images, the vectors and the grading. This plan adds only where and how the kit runs, and it
amends nothing in the kit.

**Bitstream.** The first B5b-1 full build, capstone-ariane d0afdb40f, sealed as
`caplifive_…-pagedtbl-r76-r75d-b5b1_d0afdb40f.bit`. It was launched on apollo 2026-11-23 at 02:45Z under the 40 GB guard
(keys: rtl lane land/rtl-b5b1 507463601fa3, graded against R-75 (d)'s seal by `twc/b5b3/post-b5b3-full.sh`). **The kit
runs only if that grade reads PASS on every key.** Bitstream 1 (9e801ab5e) stays the registered default and the
fallback until the kit reads (the delegated bitstream rule, agreed with the runtime lane). The board and its slot are
the runtime lane's grant.

**Images.** The six of the PREREG table, in its order, every one expected to return:
1. sublet-life (control, SR 35);
2. csmint (SR 51);
3. csmint-vm (SR 30);
4. csmint-vm-nc (SR 30);
5. csmint-pmp (SR 16);
6. r79-pmp (SR 17, R-79's fixed row).
They come from a kit worktree `wt-csmintkit` at 4b02223df217. Every image's sha256 and rec matched the table on the day
(`sha256sum -c SHA256SUMS` passes).

**Procedure** (`<mnt-hdd>/capstone/csmintkit/run-csmintkit.sh OUTDIR BOARD BITNAME BITSHA256`).
- Prechecks, all before any board call:
  - the kit worktree at 4b02223df217 and unmodified;
  - each image's sha256 and rec against the PREREG table, which is parsed from the file;
  - the seal's local sha256 against the sha recorded at the seal;
  - the url file and the tools.
- Optionally wait for a pid, then for the board to be free, and book 45 min.
- Read back the resident. If it is not the seal, flash the seal with `flash-bitstream.py`, which checks the local sha256
  against the seal first, power-cycles and reads back; then read back again. A failed flash or readback STOPs before
  any load.
- The six images, one JTAG load each through `sup-bare-2026-10-03-ladder/run_sup_bare.py`, each with a 150 s timeout.
- On every exit, unbook. The script touches no marker. The seal is left resident; the default is unchanged.

**Grading** (`csmintkit/grade-csmintkit.py PREREG.md RUNDIR`). It parses the PREREG table and its six vectors, and
requires each SR to equal its vector's length (35, 51, 30, 30, 16, 17). The rules are the PREREG's:
- VOID unless the readback names the d0afdb40f seal and sublet-life reads SR 35 and its 35 values;
- CSMINT CONFIRMED when loads 2-5 read exactly;
- R-79 FIXED when load 6 reads its fixed row;
- otherwise the first failing load and its first differing position are printed. Every later load is collateral, and a
  non-return is a wedge.

**Instrument checks before arming.**
- The runner was dry-run (board calls stubbed, every other check real, with R-75 (d)'s seal standing in) through:
  - a flash, then six loads;
  - the seal already resident, so no flash;
  - a failed flash and a failed readback after the flash, each STOPping before any load and still unbooking;
  - a busy board;
  - a missing url file;
  - a wrong seal sha;
  - a modified kit tree.
- The grader was tested on fabricated runs built from the PREREG's own vectors:
  - all exact reads CONFIRMED and FIXED;
  - a wrong seal name, a bad control and a missing control each read VOID;
  - a wedge at csmint-vm, a differing value at csmint position 5, and R-79's pre-fix row at load 6 each name their
    first failure, with every later load collateral.

**The slot** (the runtime lane's grant, 2026-11-23, its option b: nobody waits behind a peer). Board 2, armed 03:17Z as
`csmintkit/orch-csm.sh` in tmux:
- It waits for the full build's grade, and continues only if grade-b5b3-full prints "B5B3-FULL PASS" and the chain's
  verdict is SEALED with the d0afdb40f name. The seal sha comes from that verdict line, and the kit checks it against
  the file again.
- Boundary A is the runtime lane's h16f exit. The kit polls its pid at 5 s, and the board must read free within 40 s.
- Boundary B is h16s's exit. It is used if h16f had already exited at the grade, or if A found the board busy before
  booking. Nothing is queued on board 2 after h16s.
- A pid counts only while its /proc start time equals the one recorded at arming, so a recycled pid cannot trigger a
  booking.
- Neither boundary alive at the grade: no booking, and the runtime lane is asked.
- Afterwards the seal stays resident. h16s reads the resident name and flashes P1 itself on the mismatch.
- The orchestrator was dry-run against a stub kit through each branch: a failing grade, a wrong seal name, a clean
  boundary A, A busy falling to B, a recycled pid, and neither boundary alive.

**Re-slotted (2026-11-23 09:17Z), at the runtime lane's order.** The orchestrator above (re-armed 04:13Z on the v2
grade) never booked. At h16f's exit board 2 goes to the nested lane's item-5 corpus first: SQLite and PHP bug corpora,
several boots, `nested/corpus5/board/item5-chain.sh`. That chain unbooks and then writes `ITEM5-BOARD-DONE` on every
exit path. Boundary A would have raced it, and boundary B was gone: the runtime lane had stopped the original h16s by
hand before it booked. The boundary-A wait was stopped in its pid-poll loop with no booking (chain.log: no BOOKED, then
END), and the old orchestrator ended on its "no booking" branch. The run is now `csmintkit/orch-csm2.sh`, in tmux
csm-b2c, run `run-b2c-20261123T0917`:
- it waits up to 68 h for ITEM5-BOARD-DONE, newer than its own start, which is inside the 72 h bound of h16s's waiter;
- it re-checks the grade and the seal exactly as before, then runs the same kit with no pid wait, and the board must
  read free within 1,800 s;
- on every outcome, including a timeout, a failing grade, no booking and a signal, it writes the runtime lane's
  `perf/natstage/csmint-b2.DONE` with that outcome. h16s now waits on that file (the runtime lane's wait-marker.sh).
- It was dry-run with every path pointed at scratch: a fresh marker (DRY kit, all six loads stubbed), a stale marker
  (no trigger), a timeout, a failing grade and a TERM. Each wrote its scratch DONE with the matching outcome, and the
  real markers were checked absent afterwards.

## Reading (2026-11-23), board 2: CSMINT KIT VOID by its PREREG (the control's last ten words)

Run `csmintkit/run-b2c-20261123T0917`, board 2, booking 1450. It booked at 13:51:43Z, 38 s after the nested lane's
ITEM5-BOARD-DONE ("pools:0 STOP-GATE1"), and was unbooked at 13:59:17Z.
- **Before the flash:** the resident read back as P1 `…_d83d3a633.bit`, not the seal (rc 1).
- **Flash:** `flash-bitstream.py` matched the local sha256 to the seal, uploaded the bitstream (HTTP 201), flashed it
  and power-cycled the board. The resident after the power cycle was `…-r75d-b5b1_d0afdb40f.bit`.
- **Readback after the flash:** the same name, name only (rc 3).
- **Loads:** all six returned, each with its full SR count and SUPTEST END.
- **Marker:** csmint-b2.DONE was written at 13:59:17Z. The seal is left resident; the default stays R-75 (d), with
  bitstream 1 the fallback.

| load | image | reading against the PREREG |
|---|---|---|
| 1 sublet-life (control) | eb281996594bcec1 | SR 35, 35 values: positions 0-23 and 34 exact, 24-33 DIFFER |
| 2 csmint | e9ba3863d3bc8f70 | SR 51, exact |
| 3 csmint-vm | 74f4c5896d0d2dd1 | SR 30, exact |
| 4 csmint-vm-nc | d73ceff87be78e06 | SR 30, exact |
| 5 csmint-pmp | 7d672a84e8d8dda5 | SR 16, exact |
| 6 r79-pmp | 8eb72553965cfc1c | SR 17, exact (R-79's fixed row) |

**Verdict: VOID, by the PREREG's own rule.** The PREREG voids the boot unless sublet-life reads "SR 35 and its 35
values", and `grade-csmintkit.py` printed "CSMINT KIT VOID: the sublet-life control is DIFFERS" (rc 1). So loads 2-6
are recorded as readings, not verdicts. They read the PREREG's vectors exactly, but this boot neither confirms CSMINT
nor shows R-79 fixed. Deciding that is the PREREG owner's call, not this record's.

**What differs.** The differing words are sublet-life's pool dump p11..p20. Each was pre-registered as 0x0 and read:
`0xff000100ff010003 0x94004000848882 0xff000380ff014006 0x41104000051402 0xff000400ff010006 0x14154100248810
0xff000400ff014006 0x80004000788801 0xff000500ff000006 0xa04d000fb1181`.

**Where they sit, from the source** (capstone-ariane d0afdb40f):
- The dump reads 20 quadwords from 0xbff00000 (`sublet-life.S`, the pool loop).
- The physical pool is `CAP_REVNODE_MEM_BASE + slot * 16` (`ex_stage.sv:2323`). The header quadwords are slots 0
  and 1 (`ridx`/`hidx`, `capstone_sublet_life.sv:111-116`). So p11..p20 are records 5 to 9.
- The header read on silicon, p1 = 0x8000000000010000 and p2 = 0x5, matches the PREREG and gives next = 5. Records 5
  to 9 have never been allocated.
- At reset the unit writes only the two header quadwords and slot 2's record (`capstone_sublet_life.sv:300-302`).
- The unit reads fresh candidates ahead of next only in a paged pool (S_O_N, `:386`). The physical pool is not paged
  (header word 0 bit 62 = 0).

Read that way, nothing in this boot wrote 0xbff00050..0xbff00098. The PREREG's zeros there would then be simulation's
zero-initialised memory, and the reference's zeroed table, rather than an output of the unit; silicon returns whatever
the DRAM held. This is an inference from the source, NOT a result. Two things would test it: rtl's check that no path
writes a Sublet slot at or above next, and the same image on silicon with those words cleared first, or with its dump
cut to records below next.

The part of the control the unit writes reads exactly: the fourteen trap causes, both header quadwords, records 2 to
4, and the trap count (6).

## Amendment, then re-run (2026-11-23): CSMINT CONFIRMED ON SILICON and R-79 FIXED ON SILICON, board 2, N=1

**The inference above was checked from source and the PREREG amended before any re-run** (rtl lane,
`land/rtl-csmkit-amend` d919aa1316e3). Nothing in the boot writes a flat-table slot at or above next. At reset the unit
writes only hidx 0, hidx 1 and slot 2. An allocation writes the slot it takes. A fresh candidate is read only when
paged_q is set, and the test stores nothing to the pool. So positions 24-33 (slots 5-9) were never a prediction. Boot
1450 stays VOID. For every re-run the control is graded on the readback, SR 35 and positions 0-23 and 34 (25 values),
and positions 24-33 are recorded but not graded. The images, hashes and other vectors are unchanged. The grader is
`csmintkit/grade-csmintkit2.py`, generated from v1 by an asserted patch. It refuses a PREREG without the amendment
(rc 2). It was tested before the run: a control change at position 5, 23 or 34 voids, a wrong readback voids, a
csmint-vm change at position 7 is the first failure with the later loads collateral, and an r79 change at position 16
reads NOT FIXED.

**Slot.** The runtime and nested lanes put the re-run after nested's item-5 re-slot and ahead of h16s. The orchestrator
is `orch-csm2.sh` (unchanged), in tmux csm-b2d, run `run-b2d-20261123T1407`, armed 14:07:56Z.

**Run.** ITEM5-BOARD-DONE ("… OK") was written at 21:10:37Z. The kit booked at 21:10:59Z (booking 1457) and unbooked at
21:18:37Z.
- Before the flash: P1 `…_d83d3a633.bit` was resident (rc 1).
- Flash: the local sha256 matched the seal, then the power cycle. The resident after the cycle was
  `…-r75d-b5b1_d0afdb40f.bit`.
- Readback after the flash: the same name (rc 3).
- csmint-b2.DONE was written at 21:18:37Z, and h16s started on it. Its runner reads the board back and flashes P1
  itself, sha-checked, if P1 is not resident.

| load | image | reading (`grade-csmintkit2.py`, amended PREREG) |
|---|---|---|
| 1 sublet-life (control) | eb281996594bcec1 | SR 35; the 25 graded positions exact |
| 2 csmint | e9ba3863d3bc8f70 | SR 51, exact |
| 3 csmint-vm | 74f4c5896d0d2dd1 | SR 30, exact |
| 4 csmint-vm-nc | d73ceff87be78e06 | SR 30, exact |
| 5 csmint-pmp | 7d672a84e8d8dda5 | SR 16, exact |
| 6 r79-pmp | 8eb72553965cfc1c | SR 17, exact: R-79's fixed row |

**Verdict: CSMINT CONFIRMED ON SILICON, and R-79 FIXED ON SILICON** (grader rc 0). The control passed on the Sublet
unit's own words, so loads 2-6 are verdicts.

Load 6 checked by hand against the raw UART:
`0x0 0x0 0x0 0x5a5a5a5a5a5a5a5a 0x7 0x80002000 0x5a5a5a5a5a5a5a5a 0x7 0x80002008 0x5a5a5a5a5a5a5a5a 0x7 0x80002000
0x77 0x77 0x77 0x77 0x3`. That is the PREREG's fixed row. R-79's pre-fix row, read on P1 (board 1,
`r79kit/run-b1-20261123T0130`), first differs from it at position 1 (0x7 where the fixed row reads 0x0).

Ungraded positions 24-33 this boot: `0xff000300ff010003 0x90004000800000 0xff000380ff014006 0x4000040000
0xff000400ff018006 0x14004000040000 0xff000480ff014006 0x80004000380001 0xff000500ff000006 0x20440009a0000`. Eight
of the ten differ from boot 1450's words. That fits memory nothing in the boot writes, and is not itself evidence about
the unit.

**Scope.** N=1, one board. The PREREG welcomes N=2 across boards, and the simulation is cycle-deterministic, so board
1 should read the same vectors. d0afdb40f is left resident on board 2 until the next chain reflashes it. Its status
for the default and as a candidate is to be agreed between the board, rtl and runtime lanes under the delegated rule.
Until then the default is unchanged.
