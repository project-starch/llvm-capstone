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
