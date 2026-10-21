# Q-30's silicon half, measured: a cursor excursion self-heals, and the issue guard separates at 114 cycles against 6

*RTL lane, 2026-10-19 late. Two arms, one evening, both pre-registered before they ran and both reading exactly
what was written down. Supersedes one sentence of
`19-10-2026_20-46-00_s23-containment-closed-by-the-issue-guard-rtl-half.md`, which said the guard was argued and
not demonstrated. Arms and readings on `capstone-ariane` `virtual-profile-2-c`: `cap-cursor-window` at
`4eecdf450` (pre-registered `7212ad46a`), `s23-issue-guard` at `849ea1d9b` (pre-registered `ae40eddd1`).*

## Q-30: does the QEMU compact-profile cursor defect have a silicon analogue?

The runtime lane found that a virtual-context escape stores x1..x31 with `cap_compress` and the resume reloads
them with `cap_uncompress`, so a register capability whose cursor sits outside its representable window returns
from a preemption with different bounds — and their allocator had one, a group capability hoisted with its cursor
near 0.

**Answer: not at the preemption, and not at a plain cursor excursion.** `cap-cursor-window` on their own geometry
(0x1f0 bytes at 0x3fbbabe000), 39 readings, every substantive prediction confirmed:

| arm | reading |
|---|---|
| the home cursor, and +0x800 | base and top **exact** |
| +0x4000 and +0x10004000 | **wrong, by exactly the cursor's own 16 KiB-region change** |
| all the way back | **exact again** — the self-healing, measured |
| the allocator's transit at cursor 0 | base `0xffffffffffffe000`, top `0xffffffffffffe1f0` — the garbage decode to the digit |
| back in at +0x40 | **restored**, and its SHRINK **passed** reading [+0x40, +0x80) |
| a bit-11-crossing geometry | exact, then restored |
| **a SHRINK issued while the cursor is at 0** | **cause 0x1d (29), trap count 1** |

**The mechanism, and the asymmetry that decides it:** `compress_bounds` does **not** take the cursor
(`ariane_pkg.sv:785`); only `decompress_bounds` does (`:673`). The register file holds the compressed word
(`cap_metadata_t`, `:645-650`), operands are decompressed on entry to EX (`ex_stage.sv:874-876`) and results
recompressed on exit (`:1739`, `:1753`). The out-of-range correction is ±2^(E+14) on **both** bounds
(`:709-716`) while the kept fields are `start[13:0]` and `end[11:0]`, invariant under that shift — so the word
survives any excursion and only the decode is transiently wrong. **QEMU's register holds the FAT value**, so a
load that decodes at a bad cursor burns the garbage in permanently; that is why its preemption is a real loss and
silicon's excursion is not.

**Where silicon does bite:** an instruction that **derives** bounds from the bad decode — SHRINK, TIGHTEN,
CDERIVE. SHRINK's operand check (`capstone_flu_unit.anvil:268`) refuses with 29 rather than corrupting. In the
allocator's own sequence the SHRINK runs after the cursor is back in range, so that path is safe.

**Two retractions from this lane on the way to that answer, both the same evening.** First "the loss happens at
the cincoffset's own writeback, so the class is wider on silicon" — wrong, from reading the writeback path
without checking the compressor's signature. Then, by hand, "a geometry whose start and end differ in bit 11 will
behave differently" — refuted by the model and then by the arm: with `iE = 0`, `T[13:12]` is derived from
`B[13:12]` (`:698`), so the two corrections always fire together. **The model is what caught both**, which is the
reusable part: every one of the 39 values was computed from a transcription of the two encoder functions on the
exact inputs before the run.

**One pre-registration error, and it points at a real habit.** W0's fourth line was registered with each ladder
offset applied to the *original* base, while the arm applies them **cumulatively**, so the pre-registered cursor
was 0x4000 low and its decoded bounds with it. Driven by the cursor sequence the arm actually produces, the same
model matches all 39 readings. A model of a function must be driven by the arm's own state sequence, not by its
nominal parameters — the failure reads as a hardware discrepancy, which is the expensive direction.

## The issue guard, now measured

`s23-issue-guard`, four arms, each anchored by a CDERIVE so positions are marked rather than counted. Gap from
the anchor's derive event to the op's own owed window:

| arm | hold | op | gap |
|---|---|---|---|
| A1 | cache-missing load | CREVOKE (**guarded**) | **114 cycles** |
| A2 | cache-missing load | node query (unguarded) | **6 cycles** |
| A3 | three dependent divides | CREVOKE (guarded) | 212 |
| A4 | three dependent divides | node query (unguarded) | 140 |

**The pair separates: 114 against 6.** The unguarded query issues six cycles after its anchor while the missing
load is still outstanding; the guarded CREVOKE waits 114. So `issue_read_operands.sv:1557-1565` does what its
source says, and the one real objection in the earlier audit — that the guard was argued and never demonstrated —
is answered.

**And the negative control for the method worked in the same run**, which is what makes the pair trustworthy: A3
against A4 is 212 against 140, reproducing `s23-kill-window`'s 216 against 140-ish within a few cycles and its
difference of **72 exactly**. The serial divider is not pipelined, so it stalls both arms and that pair cannot
discriminate. The method's own failure mode was reproduced deliberately beside the measurement that works.

Harness SUCCESS, all ten readings as pre-registered, and the span counts exact — query 16, derive 7, kill 2, the
50 OWED CHANGE lines written down before the run.

Nothing here touches whether a flushed walk can outlive its instruction: the arm raises no flush at all, and its
three FLUSH SEEN lines are the harness's own CSR write. That question still has exactly one generator left, a
level-held flush rising while the unit is busy (`dom_switch_busy_i`, `controller.sv:215-225`).

## Also from this evening

`R-58` and `R-59` are filed (dev `155a33ecc006`): the switcher never saves or restores the caller's `srevroot`,
and a frame srevroot word of 1..4095 passes the non-zero precheck and installs 0. Both source-derived, both
carrying their settling arm, neither with a reading yet. The RTL fix for R-58 is this lane's and is deliberately
**not** pre-registered yet — id 67 in the caller's snapshot extends SAVE and RESTORE together and the stride
places it at +944, but the reference model and any software definition of the snapshot size have to move with it,
so it belongs in the next switcher revision rather than in the first build that happens to be running.
