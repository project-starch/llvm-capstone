# Board runs for the rtl lane's tickcost kit (pointer record)

The predictions and grading are the rtl lane's: llvm-capstone land/rtl-tickcost 485e2fe8917e,
capstone/tests/rtl-smoke/tickcost-2026-11-08/PREREG.md (pushed before any run). This record only says how the SPEC
lane runs it on the board.

- Board 3, two boots, after the quantum run there (results/2026-11-08-quantum): stage tickcost-E1, then tickcost-E2
  (perf/tickcost/stages; run.sh calls the kit's tickcost-guest.sh with tickcost.ko c16c5c56585b84c1, arm E1 or E2).
- Kit bkit33-tickcost = bkit27-gtq (C.5a, module 65ede944a87ea742) with its own driver copy
  (perf/tickcost/driver, run_baked_rungs_fpga.py 286f322bed76424a): the only change reads aperture 224 ten more
  times after a wedge (BAKED_WEDGE_REPEAT_224=10); each aperture read is still a pair 0.6 s apart, unstable marked.
- The control rungs run first in each boot. The raw transcripts go to the rtl lane for grading.

## Result (8 Nov 13:46-14:14, board 3): both boots ran to their stuck arm; the grading is the rtl lane's

- Transcripts: spec/board/out-tc2-b3/tickcost-E1/guest/board.txt (sha256 prefix f1805cca75d0fdfb) and
  tickcost-E2/guest/board.txt (aa581864848cf32a). Controls, b0-stats and b0-hello returned 0 in both. The stage reads
  NO DATA and rc 2 by construction: E1 and E2 never return.
- Arms A-D returned in both boots with near-equal numbers (full-table scans 18.9 ms user, 37.7 ms kernel; cvprof's
  own body over full tables: 633 / 630 calls in 30 s, maxbody 37.7 ms).
- E1 ("stuck rdtime", a loop in the tick that reads the clock, which the firmware emulates): aperture 224 = 0xff on
  all 11 reads, no UNSTABLE, trap log 0x85 with tval 0x00000000000a0d63, commit pc ...fffe. E2 ("stuck register",
  no traps): 224 = 0x1c, stable.
- **What this changes in exitprof:** E1 reproduces the cvprof wedges' signature with a software loop in the tick
  handler. The rtl lane withdrew its reading that those wedges were a frozen hart (an RTL state), and withdrew its
  simulation calibration as a reference for silicon (sim read ex_commit.valid 0/75 for the same shape, silicon 11/11).
  The exitprof Correction's "a trap loop at M of a filtered cause" is the shape E1 shows. The stuck loop in the real
  runs is not identified. cvprof's cost alone did not hang the kernel in 30 idle seconds. The rtl lane's next
  instrument reads the stuck pc at the next wedge.

## Addendum 1 (8 Nov 15:xx, before its board call): the GDB reader's validation boot, board 1

The predictions and grading are the rtl lane's: land/rtl-tickcost 9c449dac93d6, PREREG.md addendum 2 (READER VALID:
all 8 pc samples in [tick + 0x154, tick + 0x160) with the disassembly showing the register loop; INVALID: any
0xca11ab1ebadcab1e, a pc outside that range, or UNREAD). How the SPEC lane runs it:
- Stage perf/tickgdb/stages/tickcost-E2g: tickcost-E2 with that commit's tickcost-guest.sh, which prints the `tick`,
  `cvprof_body` and `ktime_get` addresses at KERN_EMERG. Module unchanged (c16c5c56585b84c1).
- Kit bkit34-tickgdb = bkit33-tickcost with two changes: BAKED_WEDGE_GDB=1 in board-run.sh, and the driver copy
  perf/tickgdb/driver (run_baked_rungs_fpga.py 375b6458c96dc10f), which calls wedge_gdb.probe() after the wedge
  apertures and before the board is released; an exception out of probe() is written as `GDB probe FAILED <type>:
  <message>` into the transcript (a failed halt is one of the INVALID outcomes). wedge_gdb.py is that commit's file, unchanged.
- Board 1, after its Plain pairs chain (results/2026-11-07-plainreuse Addendum 4) ends; driver perf/tickgdb/gt-run-tg.sh.
- A second boot right after it: stage tickcost-E1g (E1, the clock-reading loop, with the same guest script), whose
  predictions are in the same addendum 2 (samples in tick + 0x170..0x17c, in ktime_get, or in firmware
  [0x80000000, 0x80200000), mstatus.MPP consistent with each).
- A cvprof wedge with the reader follows only after READER VALID, as a separate pre-registration.
