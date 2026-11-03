# q04-movc-bare-2026-11-03: Q-04 (b) on silicon, pre-registration

This is written and pushed before any board run.

**The question.** Does MOVC keep an untagged source on silicon? Q-04 (b) is the lead's decision of 2026-10-14
(option 1, batch item 5). If it holds on the sealed bitstreams, the lead can lift C-32. C-32 keeps board images
without `+movc-keeps-integer-source`. Lifting it removes every CapstoneLiveSourceCopy STC+LDC pair, which is
0.04-2.5 % of executed instructions on SPEC test (the runtime lane's count).

**Source** (read, not inferred). `core/anvil_build/capstone_flu_unit.anvil`, MOVC's preserve branch.
- It preserves NOT_CAP beside NONLIN at e7af48457 (bitstream 3) and at 603528b8b (ObjBounds). The file is
  byte-identical between the two (git diff empty).
- c7e6f604f, the field RTL before the batch, preserves NONLIN only.
- The SV that synthesis consumed is in the synth lane's build trees and was not read here. If it is needed, it is
  their grep of `capstone_flu_unit.anvil.sv` for the NOT_CAP compare.

**The image.**
- `images/movc-untagged.bin`, sha256 `960e6b7992bbeb2c653c02a03fbf415795294051a0b492390d2864dba35f15d7`, board_rec
  0x80001140, SR 26.
- It is built from capstone-ariane q04-board b534e60ca `movc-untagged-source.S` with `-DBOARD_INTS`, which records
  the readings before any check:
  - the 16 tagged rows' cursors (cases D/E/F; the board recorder keeps cursors, not types);
  - then s2, t0, t1 (case A), s4, s5 (B), s6 (C), s7, t2 (F) and s8 (G);
  - then 0x600d if every check passed, or the failing TESTNUM.

**The vectors, simulated from THIS image** (`r61-inner.sh`, memory delay 12, each tree's PROVENANCE `rtl_dirty=no`):
- Q-04 (b) PRESENT. This was read on bitstream 3's tokens (87368cc69) and on ObjBounds (b534e60ca = 603528b8b + tests),
  identical on both:
  `0x0 0x80001000 0x0 0x80001000 0x0 0x80001000 0x0 0x80001000 0x0 0x80001000 0x0 0x80001000 0x80001000 0x0 0x80001000 0x0 0x1122334455667788 0x1122334455667788 0x1122334455667788 0x0 0x0 0x55 0x77 0x0 0x0 0x600d`
- Q-04 (b) ABSENT, the control that must differ (c7e6f604f):
  `<the same 16 tagged values> 0x0 0x0 0x1122334455667788 0x0 0x0 0x55 0x0 0x0 0x0 0x1`
  - It differs at s2 and t0 (the untagged source nulled, and the read right after it), at s7 (case F's rd = x0
    row) and at the marker (TESTNUM 1).
  - The tagged rows are identical. They are the instrument's own controls: a consumed LIN source reads 0, and a kept
    NONLIN source keeps its cursor.

**The grading rule.** `grade/q04-grade.py <run dir> <bs3|objb>` applies it.
1. The loaded image's sha256 must be the one above, and the resident bitstream name must name the arm (e7af48457 or
   603528b8b).
2. Then the reading is one of three:
   - **PRESENT**: SR 26 and the first vector;
   - **ABSENT**: SR 26 and the second vector;
   - **FINDING**: anything else.
3. Exit codes: 0 PRESENT, 1 otherwise, 2 NO DATA.
4. Grader controls, run on fabricated directories before this was pushed:
   - present: rc 0;
   - absent: rc 1;
   - a bitstream-4 name: WRONG BITSTREAM, rc 1;
   - a perturbed vector: FINDING, rc 1;
   - empty: NO DATA, rc 2.

**Run order.** In the same boot gap as the hpm kit, after its hpm-5 control. One image per bare boot, on both
bitstreams (board 1 bitstream 3, board 2 ObjBounds).

**What it decides.**
- PRESENT on bitstream 3 is the silicon reading the lead needs for C-32. The board images run on bitstream 3.
- ABSENT on either bitstream means the source and the silicon disagree. That goes to the lead with the kit.
- FINDING goes to the lead too, as a new defect folder.

**Limits.** N = 1 per bitstream. Only the integer rows decide; types are not visible to the board recorder.
