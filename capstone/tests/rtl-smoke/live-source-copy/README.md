# RTL check for the live-source copy rule (plan step A.9)

`CapstoneLiveSourceCopy` (branch `compiler/movc-live-source-copy`) replaces every `movc` whose
source is read again with an adjacent `stc src, off(sp)` / `ldc dst, off(sp)` pair through one
16-byte slot. The correctness argument is that the pair behaves as `movc` does for NONLIN and
LINEAR sources and keeps an integer source, which `movc` on silicon nulls (C-32). QEMU cannot
check that argument, because QEMU's `stc`/`ldc` have no linear-slot semantics
(`CapstoneISASemantics.md`, ldc/stc rows). No board image carrying the rule is built before this
test has run on the resident RTL.

`lsc-stc-ldc-slot.S` is a directed test in the style of `verif/tests/custom/capstone/`. Each
failure path sets its own TESTNUM, listed in the file header.

| case | shape | pass condition |
|---|---|---|
| 0 | `LCC x0` on an integer | traps once (the detector works) |
| 2 | integer, stc then ldc, cold granule | source and destination hold the integer, destination untagged |
| 3 | the same, over a granule that held a NONLIN capability a moment before | destination untagged: the stale tag is not returned (S-10/S-10b family) |
| 4 | NONLIN | the source is still NONLIN; the destination is NONLIN with the same cursor and end |
| 5 | LINEAR, then a second ldc | source nulled; first reload LINEAR; second reload untagged |
| 6 | integer, NONLIN, integer pairs back to back on one granule | each reload has the right value and type |
| 7 | one stc, then three ldcs (integer) and two (NONLIN) | every reload equals the stored value (the slot-reuse shape) |
| 8 | ldc whose base is the register the previous ldc just wrote (S-07 shape) | reads the capability stored behind it |
| 9 | case 3 sixteen times, with four plain stores in flight each time | no bad leg |

## How to run it

On the host with the RTL build, per the `rtl-sim` skill:
1. Copy the file into `verif/tests/custom/capstone/`.
2. Add a testlist entry copied from `stc-register-clear`.
3. Run it at the resident revision, `6cbdaeeb4`: once at the default memory latency, and once with
   `S12_MEM_DELAY` set to a non-zero value.
   - Record the realised delay. `S12_MEM_DELAY=40` realises as 8 cycles (R-27 README).
   - Pin `--sv_seed` and record it with the result.

## What was run: PASS on `6cbdaeeb4` at both latencies (2026-09-29, RTL lane)

Result lines: `rtl-6cbdaeeb4.result-lines.txt` in this folder. `--sv_seed 1`; the verdict is read
from tohost, and the trap sites from the retirement trace.

- **The test as first written failed case 0 at both latencies**, on a cause number rather than on the
  shape: its handler expected 25, and this RTL raises UNEXPECTED_OPERAND_TYPE as **24** (the R-24 fix,
  in every bitstream from `054cea69b` on). The handler now expects 24. Nothing else changed.
- **With that fix: `RVTEST_PASS` at delay 0 (1,848 cycles) and at `S12_MEM_DELAY=40` (6,001
  cycles).** There are exactly three traps, all cause 24: case 0's, and case 5's two (the nulled LINEAR
  source and the cleared second reload). That is the prediction below, site for site, identical at
  both latencies. Cases 3 and 9, the S-10b stale-tag route, pass, and so does case 8, the S-07 shape.
- **A cold-miss extension, run locally and not part of this test.** It adds 17 legs, each on a fresh
  granule whose tag reached memory: STC NONLIN, fence, STC integer, LDC. Leg 0 is a positive control
  that skips the integer store and must count as bad. Result: exactly one bad leg (the control) and
  no extra traps, at both latencies. So 16 cold misses over a granule that held a tag read the integer
  back, untagged.
- **Not run:** the R-43 redesign `8f6a0af98`, and any board.

## Predicted reading, written before any run

- **Expected:** `RVTEST_PASS` at both latencies, with exactly three counted traps: case 0's, and
  case 5's two, for the nulled source and the cleared second reload. This follows from:
  - STC's writeback rules and LDC's granule clear (`capstone_dyn_unit.anvil`, `load_unit.sv:222-229`);
  - the load/store hazard check, which is word-granular (`store_buffer.sv`). An `ldc` at word 0
    matches an `stc` at word 0 of the same granule, so the load waits for the store.
- **If case 3 or 9 fails:** the S-10b tag route reaches the rule's own shape. The rule would then
  need a separating instruction, or a slot the function never stores a tagged value into, before
  any board use.
- **If case 5 fails:** the move-only argument does not hold for LINEAR sources. LINEAR copies are
  outside the Tier 4.1 contract, so this matters less, but the ISA semantics note would be wrong.
- **If case 8 fails:** it is S-07, not this rule. The same shape arises without the rule whenever
  a spilled pointer is reloaded and dereferenced.
