# crevoke4b-bare-2026-11-25

This kit runs B5b-2 4(b) on silicon, with the interruptible CREVOKE, the store drain and the residual-2 guard. The tests are bare JTAG images, built for 88db40027's full build. The sources are capstone-ariane b5b2-crevoke4b at 438a9c70b, whose core is 88db40027's. The design, the semantics owner's rulings and the simulation readings are in llvm-capstone capstone/docs/plans/2026-11-25-crevoke-4b-rtl.md.

**Siblings:**
- `b5b2n-bare-2026-11-25` is 904cd9c6b's kit, on silicon at N=2. Its wrapper (`inc/`: the plain-M recorder and report, CSMINT minting) is used here byte-identical.
- `b5b2-bare-2026-11-24` is B5b-2's first kit (0c0458149).
- `sublet-life` is the control here too. It is rebuilt with this kit's id character, so its pool words differ from earlier kits' (see PREREG).

This kit claims nothing about the B5T mcf / M-16 wedge. That wedge is open, and its instrument is a separate hash (capstone-ariane b5b2-wedgeap).

**Images.** The tests are unchanged from 438a9c70b `verif/tests/custom/capstone/`, and the `-D` flags are as in testlist_b5b2.yaml. No image holds a custom-3 encoding (checked by build.sh). Every test mints through CSMINT.
- `sublet-life`: the known-good control, first in the boot.
- `b5b2-crevint` (no interrupt), `-mt` (the M timer during the CREVOKE), `-sh` (the no-Sstc shape: M sets STIP, and S's handler derives a record inside the run between executions), and `-lat` (the stop latency in mtime ticks, with its calibration).
- `b5b2-crevsv-none / -srev / -straw / -derive / -kill`: the saved position. In each, one thing is done between a stop and its restart.
- `b5b2-sbdrain-none / -fill / -fence / -mint / -sret`: does a lifetime operation see the hart's older committed stores to the table? `-cost0` / `-cost` measure the drain's cost per round.

**What silicon can and cannot show.**
- The answers and the table checksum are architectural.
- Whether a restart resumed or rescanned is not: the path is read from the unit's simulation trail only.
- Timing-dependent positions (cycles, timer-trap counts, latencies) are measured, not graded.

`build.sh` builds with riscv64-unknown-elf-gcc. Recorder registers come from pick-regs.py. `images/SHA256SUMS` pins the images. `PREREG.md` holds the vectors simulated from THESE images on both trees, and the grading and void rules. `expected-vectors.json` carries the same data for graders, with the class of every position. Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG`, and `SUP_REC_ADDR` from each `.rec`).
