# s23-cpmp-rearm as a bare JTAG arm: the must-read arm of the batch bitstream's first boot (pre-registered 2026-10-21, before any board run)

The batch bitstream is cut from capstone-ariane `c721c1a2a` (items 1, 2, 3, 4-restructured, 5 and 6). Its first boot
needs one arm that reads DIFFERENTLY on it than on every pre-batch bitstream — the boards hold tree A `8522d72fe` and
tree B `8de3c48cf` (the board lane's correction, 2026-10-21; `current-state.md`'s "resident bf3db3175" is stale) — in a direction that no
other change in the batch can produce, so that "the new bitstream is running" and "item 1 is in it" are the same
reading. That arm is S-23's data face, `s23-cpmp-rearm` (18 probes, 20 recorded deltas), whose six re-arm probes flip
ALLOWED -> DENIED on item 1's namespace-aware invalidation and on nothing else.

Built by `build.sh` from the arm's source at the batch tip (copied into `tests/` and checked equal at build time),
with the bare M-mode wrapper of `sup-bare-2026-10-03-ladder` (env/) and the node-aware header of
`vp1-bare-2026-10-17` (inc/, copied). Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG` = the image,
`SUP_REC_ADDR` from the `.rec`, `FPGA_BITSTREAM` = the resident name); the report is the UART's `SR <count>` /
`SV <value>` lines. No SDK, no monitor, no Linux. The arm sets its own `mtvec` and counts M-mode traps; each probe is
one load from S through one CPMP entry with the counter read across it, recorded as the counter DELTA: 0 = ALLOWED,
1 = DENIED. The wrapper's three scratch registers are x8, x4 and x2, which the arm never names.

| image | sha256 | board_rec | SR | readings (SV, in order), from the simulation of the committed tree |
|---|---|---|---|---|
| `images/s23-cpmp-rearm.bin` | `bcefd969e128113f0c58e18571023f1269efce816d0587a85e987577b9e3c0fe` | 0x80004040 | 25 | see the two vectors below |

## The two vectors, simulated through the Anvil prologue (each run directory carries its PROVENANCE)

Print order: `a0` (CAPENTER's grant: its CURSOR, `0x80800000`, the base of the [0x80800000, 0x82000000) grant the arm
runs in — the simulation's own print_cap line reads `Reg[10]: Cursor: 0000000080800000` on all three pre-batch trees and
on the batch, and silicon reads the same), the marker `0x5a23`, the twenty probe deltas Q1 .. Q13d in source order, the
last trap cause, the total trap count, the end marker `0x55`.

| probe | what it does | PRE-BATCH: `bf3db3175`, tree A `8522d72fe`, tree B `8de3c48cf` — all three IDENTICAL (same rev-node unit hash 6d0f9699a247d012; the arm overlaid, rtl_dirty=yes for that reason) | batch `c721c1a2a` | reading |
|---|---|---|---|---|
| Q1 | slot 0 installed, probe | 0 ALLOWED | 0 ALLOWED | INSTRUMENT CONTROL: DENIED here voids the whole run |
| Q2 | the FRTY sweep, probe slot 0 | 0 | 0 | the handler alone is harmless |
| Q3 | REVOKE h0, probe | 1 DENIED | 1 DENIED | the broadcast clears the validity bit |
| Q4 | the FRTY sweep again | 1 | 1 | an id-preserving rewrite cannot re-arm it |
| **Q5** | `csrsi csupctl, 4`, the M strobe | **0 ALLOWED** | **1 DENIED** | **the re-arm is gone** |
| Q6 | probe slot 1 after REVOKE h2 | 0 | 0 | INDEX CONTROL |
| Q7 | the matching index | 1 | 1 | |
| **Q8** | srevroot write, SPACED | **0 ALLOWED** | **1 DENIED** | **the re-arm is gone** |
| Q9 | a fresh clear | 1 | 1 | |
| **Q10** | srevroot, BACK-TO-BACK | **0 ALLOWED** | **1 DENIED** | **the re-arm is gone** |
| Q11a, Q11b | controls | 0, 0 | 0, 0 | |
| Q11c, Q11d | true denies | 1, 1 | 1, 1 | |
| **Q12a, Q12b** | one strobe, two slots | **0, 0** | **1, 1** | **the strobe's GLOBAL reach is gone** |
| Q13a | control | 0 | 0 | |
| Q13b, Q13c | true denies | 1, 1 | 1, 1 | |
| **Q13d** | | **0 ALLOWED** | **1 DENIED** | **the re-arm is gone** |
| last cause | | 0x5 | 0x5 | |
| **total traps** | | **8** | **14** | the eight true denies, plus the six flips on the batch |

Full vectors, SV in order:

- pre-batch, ONE vector for all three: `0x80800000 0x5a23 0x0 0x0 0x1 0x1 0x0 0x0 0x1 0x0 0x1 0x0 0x0 0x0 0x1 0x1 0x0 0x0 0x0 0x1 0x1 0x0 0x5 0x8 0x55`
  (SUCCESS after 17,745 cycles on each; vp9-wt/.sup/s23-resident-bf3db3175/, vp9-wt/.sup/s23-treeA-8522d72fe/,
  vp8-wt/.sup/s23-treeB-8de3c48cf/ — byte-identical vectors, identical cycle counts, identical rev-node unit)
- batch `c721c1a2a`: `0x80800000 0x5a23 0x0 0x0 0x1 0x1 0x1 0x0 0x1 0x1 0x1 0x1 0x0 0x0 0x1 0x1 0x1 0x1 0x0 0x1 0x1 0x1 0x5 0xe 0x55`
  (SUCCESS after 17,770 cycles; vp8-wt/.sup/s23-batch/; byte-identical to item 1's acceptance run of 2026-10-21, with
  the `SUP ns-flush RISE` line present 7 times, so Q8's setup is proven live on the batch tree)

**How to grade a boot.** SR must be 25, the first value must be `0x80800000` (the grant's cursor; anything else is a
different grant geometry or a different image, and the probe pages are then not where the arm thinks) and Q1 must read
0, or the instrument did not work and nothing else is evidence.

**CORRECTION 2026-10-21 22:40, before the tree A run and the batch boot.** The first version of this file gave the
first value as `0x0` in both vectors. That was an artefact of `capprint-readings.py`, which reports a CAPPRINT operand
as "the last value written" to that register in the INTEGER trace; `a0` is written by CAPENTER, a capability write the
integer trace never shows, so the tool reported the register's stale integer view. The simulation's own `print_cap`
line for that CAPPRINT reads `Cursor: 0000000080800000` on every tree, which is what `board_rec` stores. Board 2 (tree
B) read `0x80800000` there and matched every other value, which is how the artefact was caught — the board lane's
grader had added an `a0` check this file did not ask for, and it was right to. Every other value in both vectors is an
integer register (trap-count deltas, the cause, the count, the markers) and is unaffected.
Then the six flip probes and the total: 14 with Q5/Q8/Q10/Q12a/Q12b/Q13d all DENIED is the batch bitstream with item 1
in it; 8 with them all ALLOWED is a pre-batch bitstream — tree A, tree B or bf3db3175 read the same — or a batch
bitstream without item 1, which the lane would have to explain. A tree A run BEFORE the batch is the silicon-side
instrument control: SR 25, Q1 0 and total 8 there prove the wrapper, board_rec, the trap counter and the probe pages
work on silicon, leaving the batch's first boot one unknown, the flips. Any mix is a finding, not a pass. The eight true-deny probes and the six controls must read the same
on both, or the instrument has moved and the flips are not evidence either.

**What this arm cannot distinguish.** It reads the CPMP validity bit through item 1's gate only; items 2-6 do not
touch its path (the sweep on `c721c1a2a` reads identical in value to the `c7e6f604f` baseline on every arm, with one
accepted +1-cycle shift at a REVOKE from item 6). So it certifies item 1's presence, not the batch's correctness.
