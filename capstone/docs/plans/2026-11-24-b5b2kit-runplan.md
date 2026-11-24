# B5b-2 bare kit on silicon: board lane run plan (2026-11-24, written before any board run)

The kit is the rtl lane's, pre-registered at llvm-capstone `land/rtl-b5b2-plan` 126444a3325e
(`capstone/tests/rtl-smoke/b5b2-bare-2026-11-24/PREREG.md`). The PREREG holds the 16 images, their vectors and the
grading rule. This plan adds only where and how the kit runs, and it amends nothing in the kit.

**Bitstream.** B5b-2's seal, `caplifive_…-pagedtbl-r76-r75d-b5b2_0c0458149.bit`, sha256
b44291bacb220b8ae6dd02ff113cb68bcfff231893fae70741261c087bb53b9a. Its full build graded B5B2-FULL PASS (10/10 keys);
the record is `2026-11-24-b5b2-full-build.md`. The flash goes under the delegated bitstream rule. The rtl lane asked
for it and the runtime lane granted the board. The registry's default (R-75 (d)) and fallback (bitstream 1) are
unchanged; the seal is left resident afterwards, and the runtime lane's next board-3 chain reflashes on a readback
mismatch.

**The slot** (the runtime lane's grant, 2026-11-24). Board 3, 75 min, right after its l48p chain:
- The boundary is `thrun.sh 3 … l48p-b3.conf`, pid 2707050, /proc start time 640258594, checked at source here. A pid
  counts only while its start time matches, so a recycled pid cannot trigger a booking.
- Nothing of the runtime lane's is queued on board 3 behind l48p.

**Images.** The 16 of the PREREG table, in its order: sublet-life (the control) first, then the 15 B5b-2
configurations. Every one is expected to return, and the runner power-cycles before each. They come from a detached
kit worktree `wt-b5b2kit` at 126444a3325e. `images/SHA256SUMS` passes there.

**Procedure** (`<mnt-hdd>/capstone/b5b2kit/`):
- `orch-b5b2.sh OUT` (sha16 cefd1614c36e2fc8):
  - re-checks the full build's grade ("B5B2-FULL PASS") and its SEALED verdict with the 0c0458149 name; the seal sha
    comes from the verdict;
  - runs the kit with the pid boundary above (≤ 24 h) and the board free within 1,800 s;
  - on every exit, including a signal, writes `b5b2kit-b3.DONE` with the outcome for the runtime lane.
- `run-b5b2kit.sh OUT 3 BIT SHA` (962498c47988343f), derived from the CSMINT kit's runner:
  - prechecks, all before any board call:
    - the kit worktree at 126444a3325e and unmodified;
    - `kit-images.py` (7da38f2b816c272b): each image's sha256 and rec against the PREREG table, which is parsed from
      the file and must also agree with `expected-vectors.json` and `SHA256SUMS`;
    - the seal's local sha256;
    - the url file and the tools.
  - waits for the boundary pid, then for board-busy to read free;
  - starts a board-3 holder, then books 75 min; a 409 (the previous booking not yet released) is retried every 20 s
    for up to 15 min.
  - reads back the resident. If it is not the seal, flashes it with `flash-bitstream.py`, which checks the local sha256
    against the seal first, power-cycles and reads back; then reads back again. A failed flash or readback STOPs before
    any load.
  - runs the 16 images through `sup-bare-2026-10-03-ladder/run_sup_bare.py` (unchanged from the CSMINT kit's), one
    JTAG load each, ≤ 150 s;
  - reads the resident back after the loads;
  - unbooks on every exit after the booking.

**Grading** (`grade-b5b2kit.py KITDIR RUNDIR`, 6c648948046a6a6f). It applies the PREREG's rule:
- It first checks that `expected-vectors.json` equals the PREREG's code blocks, value for value (otherwise ERROR).
- **VOID** unless:
  - the readback names the 0c0458149 seal, both before the loads and after them;
  - every RUN line carries its image's sha256 prefix and rec;
  - the control sublet-life reads SR 35 with positions 1-24 and 35 exact. Positions 25-34 are printed and not graded.
- Each B5b-2 image is **AS PREDICTED** when SR and every value are exact. Otherwise it is a **MISS**, with its
  differing positions (1-based); no SR line is **NO READING**.
- Each image is its own power-cycled load, so a MISS does not make the later images collateral.

**Instrument checks before arming.**
- The runner was dry-run with the board calls stubbed and every other check real, through these cases:
  - a flash then 16 loads;
  - the seal already resident;
  - a failed flash, and a failed readback after the flash: each STOPs before any load, and both unbook;
  - a readback after the loads that misses the seal;
  - a busy board;
  - a 409 twice, then booked;
  - a 409 that never clears: STOP, no booking;
  - a wrong seal sha, a wrong image count, a wrong kit commit, a missing url file and an untracked file in the kit
    directory: each STOPs before booking;
  - a boundary pid ending, and a pid whose start time does not match.
- The orchestrator was dry-run through:
  - a PASS run;
  - a failing grade (no booking);
  - a TERM, which writes the DONE marker with "signal".
  - Every dry output went under `b5b2kit/dry/`, with the holder on a nonexistent board.
- The grader was run on fixtures built from the kit's own vectors:
  - all exact: 15/15;
  - the control's ungraded positions 25-34 changed: still 15/15;
  - control position 24 changed: VOID;
  - b5b2-raw position 16 changed: MISS at position 16;
  - nsedge with no SR line: NO READING;
  - an extra value in cptag2: MISS;
  - each VOID: a readback naming the B5b-1 seal, a readback after the loads that misses the seal, a wrong sha prefix on
    one RUN line, and a missing RUN line;
  - a tampered json: ERROR.
