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

## Re-slotted to board 1 (2026-11-24 15:36Z), at the runtime lane's order

The runtime lane withdrew the board-3 slot because l48p might run for hours more. It freed board 1 instead: the lead
ruled that SPEC train runs only at the very end, so it stopped its board-1 train chain and released booking 1486.
Nothing of its own was queued on board 1.

**Board 3 before the disarm.** l48p's pid exited at 15:34:00Z, before the withdrawal reached the board lane. The
board-3 run, `run-b3-20261124T1530`, then went ahead as the grant then stood:
- It booked board 3 (1487) at 15:34:01.
- It read rvbase `…l1d48-12way_bb56087ba.bit` as resident and flashed the B5b-2 seal, local sha checked first.
- The runner was sent TERM at 15:36:15 during the flash's power cycle. The flash completed: rc 0, and the resident
  after the power cycle is the seal.
- No image was loaded. Booking 1487 was unbooked (HTTP 200) and the holder exited.
- The runtime lane was told. Its next board-3 chain reflashes rvbase on the readback mismatch.

**Board 1.** `orch-b5b2-b1.sh` is `orch-b5b2.sh` with board 1 and no pid boundary. It was dry-run once, through a flash
and 16 loads. Run `run-b1-20261124T1536`, booking 1488, booked 15:36:57Z and unbooked 15:50:27Z.
- Before the flash: P1 `…pagedtbl_d83d3a633.bit` was resident.
- Flash: the local sha256 matched the seal, then the power cycle.
- Readback after the flash: the seal `…-r75d-b5b2_0c0458149.bit`.
- The 16 loads all returned rc 0, each with its full SR count, SUPTEST BEGIN and SUPTEST END.
- Readback after the loads: the seal again.

## Reading: B5B2-KIT 15/15 AS PREDICTED, the control exact (board 1, N=1)

`grade-b5b2kit.py`, rc 0. The readback names the 0c0458149 seal both before and after the loads, and every RUN line
carries its image's sha256 prefix and rec.

| load | image | sha16 | reading |
|---|---|---|---|
| 1 | sublet-life (control) | 991f87887e3f40ba | SR 35, AS PREDICTED on positions 1-24 and 35 |
| 2 | b5b2-illegal | d76a286aa87a65aa | SR 73, AS PREDICTED |
| 3 | b5b2-illegal-runv | 1e6591f0f0b5a8ae | SR 4, AS PREDICTED |
| 4 | b5b2-uprot | 08285ab00036d92f | SR 19, AS PREDICTED |
| 5 | b5b2-uprot-mis | b51889b8266d654d | SR 23, AS PREDICTED |
| 6 | b5b2-rule2 | 748635029c20760f | SR 14, AS PREDICTED |
| 7 | b5b2-rule2-pair | 408e7d85b84dcb3c | SR 21, AS PREDICTED |
| 8 | b5b2-sref | 624a531562bab3a4 | SR 20, AS PREDICTED |
| 9 | b5b2-sref-gen | 6fb462b500969cd1 | SR 20, AS PREDICTED |
| 10 | b5b2-scratch | 12adb230ffbdc8e1 | SR 54, AS PREDICTED |
| 11 | b5b2-mwrite | 17975ed92c56eb21 | SR 8, AS PREDICTED |
| 12 | b5b2-raw | 50ed437fdb47d25b | SR 38, AS PREDICTED (position 16 reads 0, the PREREG's layout-dependent prediction) |
| 13 | b5b2-rawlcc | e4764971b93d9e0b | SR 5, AS PREDICTED |
| 14 | b5b2-cptag | 8e74b5918604fd81 | SR 24, AS PREDICTED |
| 15 | b5b2-cptag2 | ecb129125c952ce3 | SR 13, AS PREDICTED |
| 16 | b5b2-nsedge | b75706e200fb76e2 | SR 17, AS PREDICTED |

**Checked by hand against the raw UART.**
- sref, sref-gen, raw and nsedge: each capture holds one SUPTEST BEGIN, one SR line, its SV lines and one SUPTEST END,
  and each file was written during its own load.
- sref and sref-gen differ exactly where their vectors do: position 12 reads 0x6 in sref and 0x6000000080000006 in
  sref-gen, and position 14 reads 0x0 and 0xc0000001. So each capture is its own image's, not a replay.
- The control's ungraded positions 25-34 read 0x0 on this boot. They are recorded only; DRAM after this power cycle
  happened to hold zeros there.

**Scope.** N=1, one board. The grading is the PREREG's, applied by the board lane's script. The rtl lane re-grades the
captures against `expected-vectors.json` with its own script. The B5b-2 seal stays resident on boards 1 and 3. The
registry's default (R-75 (d)) and fallback (bitstream 1) are unchanged; whether B5b-2 becomes a candidate is for the
board, rtl and runtime lanes to agree under the delegated rule.

## N=2 on board 3 (2026-11-24): attempt 1 VOID (runner), attempt 2 15/15 AS PREDICTED

rtl asked for this, and the runtime lane granted 20 minutes on board 3, where the seal was already resident.
`orch-b5b2-b3n2.sh` runs the kit with RK_NOFLASH: the resident must already be the seal, otherwise the run STOPs with
no flash and no load. It was dry-run through both branches.

**Attempt 1, booking 1489 (16:21:53-16:21:56Z): VOID, no load reached the board.**
- The readback named the seal. Then each of the 16 loads failed in about a second, before any console connection:
  `FPGA_BOARD='3': the boards are ['1', '2']`.
- The kit's pinned tree (126444a3325e, dev-based) carries a `fpga_console.py` with no board 3. Board 3's entries
  exist only on origin/park-board3.
- The dry runs stubbed the runner, so the board table was never resolved before the booking.
- The grader reads it VOID: the control is NO READING.
- The runner still exited 0, and its DONE line read "kit rc=0". The runtime lane caught this.

**Fixes before attempt 2:**
- Loads take the board driver from a second detached worktree, origin/park-board3 5980866f9f65 (RK_DRIVER_WT and
  RK_DRIVER_SHA).
  - Against the kit tree, its only differences are board 3's secret and lock, the shared-board watch, and the flash
    "error" state.
  - Its `run_sup_bare.py` is byte-identical to the kit's.
- A precheck asks the driver itself to accept the board before any booking. On the kit tree with board 3 it STOPs.
- Attempt 2 ran with the runner as it stood then. The load-count fix below was written and tested during the run and
  installed afterwards.

**Attempt 2, booking 1490 (16:23:21-16:34:43Z), run `run-b3n2b-20261124T1623`: 15/15 AS PREDICTED, the control
exact.**
- The resident read back as the seal, so there was no flash. It read back the same after the loads.
- All 16 loads returned rc 0, each with a uart.txt.
- `grade-b5b2kit.py` rc 0 gives every image the same verdict as board 1.
- The control's ungraded positions 25-34 read nonzero words on this board, `0x2b00a084508028 … 0x41940500100c0001`,
  where board 1's read zeros. That is consistent with uninitialised DRAM, which is why the PREREG does not grade them.

**The runner's outcome now counts loads** (installed after attempt 2; `run-b5b2kit.sh` sha16 dbf1594a8fd7e8da,
`orch-b5b2-b3n2.sh` 6b7f66a0eae0b8d2).
- After the loads it logs "loads attempted N, reached the console C, with an SR line S". C counts loads whose runner
  wrote a uart.txt.
- It exits 3 with LOADS FAILED when C < N, and the DONE line leads with the count.
- Negative test, attempt 1's shape for real: the actual `run_sup_bare.py`, the kit tree's driver, FPGA_BOARD=3.
  - It read rc 3, 16/0/0, LOADS FAILED.
  - That test mode runs only when the driver rejects the board. With the board-3 driver it refuses (rc 2).
- Other branches: a normal dry run reads rc 0, 16/16/16. Stubbed loads with no uart.txt read rc 3, 16/0/0.

**Status.** B5b-2's kit reads 15/15 AS PREDICTED on two boards (N=2). Default and fallback are unchanged, agreed by
the rtl and runtime lanes: B5b-2's seal runs bare kits only until the integer-M monitor exists, because the current
monitor executes CAPENTER at init. The seal is left resident on boards 1 and 3.
