# Silicon check list for bitstream 1: R-60, R-68, R-72, R-32 (pre-registered 2026-11-21)

Written before any board run. Each fix is in P1 d83d3a633's ancestry (R-72 5af46d00e, R-68 213378a11, R-32 0ab365c26,
R-60 item 4 from 86fc0475c on) and has had simulation-only acceptance. Bitstream 1 is P1 + the R-76 ccsrrw term
(capstone-ariane 9e801ab5e lineage, sealed 2026-11-20), and the four fixes' files are byte-identical to P1's there.
Vectors come from simulating THESE images (`out/*.elf`, built by `build.sh`) through the testharness: S12_MEM_DELAY 12,
time_out 3,000,000, read from the recorder area at board_rec by `board-vector.py`. Every tree was clean (dirty=no).

| image | sha256 | board_rec | SR | bitstream 1 (fixed) vector |
|---|---|---|---|---|
| `va-runv.bin` (control) | `f0ed3fac351b60f3cd92b98ad4e0d6376984e0b08c632e3b13612280e49221a4` | 0x8000c000 | 15 | `0x2 0x2 0x1234 0x2 0xb 0x0 0x0 0x55 0x0 0x77 0x77 0x5 0xb 0x0 0x9` |
| `alias-store.bin` (R-60) | `46ca6a4475b06ec0085c2401a111ef7432d772d793fb5ac2b83dbeb6c04f9393` | 0x80009000 | 8 | `0x5a5a 0x1234 0x0 0x1 0x18 0x5a5a 0x7 0x58` |
| `pcc-end.bin` (R-72) | `f0935c425401b34d46890e0faa7c37477ee69d4679e9b0118c4ccb275704de25` | 0x80003000 | 5 | `0x0 0x1c 0x0 0x0 0x1` |
| `pcc-end-in.bin` (R-72 control) | `6e67467e0074c4404754f5e29acceb2c6d47a18c0e16603c72324ab093de460c` | 0x80003000 | 5 | `0x123 0x1c 0x4 0x4 0x1` |
| `pcc-end-out.bin` (R-72 control) | `1dc587cd8489fe89cd6cf11c987e2669080034550905ceed274bbe8ea207aba4` | 0x80003000 | 5 | `0x0 0x1c 0x0 0x0 0x1` |
| `shrinkto.bin` (R-32) | `34311f6d26259cb19b6ef06588e8d22f2a20534287d63f6a7348d3bc97c26863` | 0x80004000 | 16 | `0x0 0x1000 0x0 0x40 0x0 0x40 0x0 0x1 0x0 0x11 0x0 0x7ff 0xfc0 0x1000 0x10 0x30` |
| `niss-superpage.bin` (R-68) | `f34bd6e59092b95cf83b82a91b085278ed79080c0b69378464845f82385f1ca6` | 0x80007040 | 36 | graded by cycle RATIO, below |

**Pre-fix vectors, the other side of each pair**, simulated on these images:

| issue | tree | vector |
|---|---|---|
| R-60 | 8522d72fe (tree A, before item 4) | alias-store `0x5a5a 0x1234 0x0 0x0 0x0 0x9999 0x1 0x58`: no trap, the sentinel overwritten a span away, the capability still tagged |
| R-72 | 1fc2cd5aa (C.5a, before 5af46d00e) | pcc-end `0x123 0x1c 0x4 0x4 0x1` (the straddler commits); pcc-end-in and pcc-end-out read as on bitstream 1 |
| R-32 | 1fc2cd5aa | shrinkto `0x0 0x1000 0x0 0x40 0x0 0x3f 0x0 0x0 0x0 0x10 0x0 0x7fe 0xfc0 0xfff 0x10 0x2f`, which is r32-shrinkto-bare-2026-11-06's pre-fix vector, the one board 2 read on caplifive_vp2c_1fc2cd5aa on 2026-11-06 (that run's grade file is not committed) |
| R-68 | 687849ee0 (stage-b-base, no fix, simulated with the FPGA's NI rule) | below |

The control va-runv reads the same 15 values on the three trees it was simulated on (bitstream 1, 1fc2cd5aa, 8522d72fe), and they equal its confirmed silicon vector in
r58-r59-q24-bare-2026-10-26 (bf3db3175's).

## R-68: by ratio, because silicon has no NI trace

The image prints a witness, then (cycles over 32 iterations, the loaded paddr) for each of 17 arms, then the trap
count (`niss-superpage.S` header). It was simulated with `+define+CAPSTONE_FPGA_NI` (the FPGA's rule [0, 0x8000_0000)):

| arm | bitstream 1 (fixed) | 687849ee0 (pre-fix) |
|---|---|---|
| 1 2M-00 / 2 2M-01 | 287 / 279 (1.03) | 570 / 275 (2.07) |
| 10 1G-low / 9 1G | 331 / 333 (0.99) | 572 / 325 (1.76) |
| 17 CLINT-1G / 16 CLINT | 1916 / 1875 (1.02) | 901 / 1365 (0.66) |
| traps | 0 | 0 |

Silicon latencies differ from the simulation's, so no cycle value is predicted. The keys are the ratios:
- **FIXED:** arm1/arm2 and arm10/arm9 both <= 1.25, and arm17/arm16 >= 0.85.
- **PRE-FIX shape:** arm1/arm2 >= 1.5, or arm17/arm16 <= 0.75.
- Anything between the two is a finding, reported with all 17 cycle values.
- Each arm's paddr word must equal the simulation's, or that arm read through a wrong mapping and is VOID.
The residual is unchanged by any reading: the test cannot tell the fix from deleting the splice, because every rule
boundary is 1 GiB-aligned (ISSUES R-68).

## Grading

- **VOID** unless va-runv reads SR 15 and exactly its 15 values.
- **R-60 CONFIRMED FIXED ON SILICON** when alias-store reads its 8 values exactly. A vector with 0x9999 at position 6
  is the alias landing: the fix is absent from what is resident.
- **R-72 CONFIRMED FIXED** when pcc-end reads its fixed vector and pcc-end-in and pcc-end-out read theirs. If a control
  differs, the R-72 triple is VOID.
- **R-32 CONFIRMED FIXED** when shrinkto reads its 16 fixed values. With 0x3f at position 6 it is the pre-fix bitstream.
- **R-68**: by the ratio keys above.
- One image per JTAG load, the control first in each session, per the board-run skill. Any other vector is a finding,
  graded value by value against the tables above.
- Cite readings by image hash, never by label.

## Silicon reading (board 2, resident P1 `caplifive_…-pagedtbl_d83d3a633.bit`, 2026-11-21 02:43-02:49 local)

Run by the board lane, one image per JTAG load, the control first (record llvm-capstone board/meas-pw 4d64c19cb53b,
procedure and amendments in `capstone/docs/plans/2026-11-21-silchk-b1-board2.md` there). The resident is P1, not
bitstream 1: the only ccsrrw in any image is in va-runv's M-mode setup, so the R-76 term cannot move a vector, and the
four fixes' files are byte-identical between the two trees.

Graded here a second time, by a script independent of the board lane's grader, from the raw `uart.txt` of each load
(the last SUPTEST BEGIN..END block), with each image's sha256 recomputed against the table above. The script was
negative-tested on a copy with shrinkto position 6 set to 0x3f and arm 17's word set unequal to arm 16's: it failed on
both.

| image | reading | grade |
|---|---|---|
| va-runv f0ed3fac351b60f3 | SR 15, all 15 values | control PASS |
| alias-store 46ca6a4475b06ec0 | SR 8, the fixed vector exactly | R-60 FIXED ON SILICON |
| pcc-end-in 6e67467e0074c440 / pcc-end-out 1dc587cd8489fe89 | SR 5 each, their vectors exactly | R-72 controls as predicted |
| pcc-end f0935c425401b34d | SR 5, the fixed vector exactly | R-72 FIXED ON SILICON |
| shrinkto 34311f6d26259cb1 | SR 16, the fixed vector (position 6 0x40) | R-32 SHRINKTO FIXED ON SILICON |
| niss-superpage f34bd6e59092b95c | SR 36, witness 0xa00000800, traps 0; ratios arm1/arm2 1.000, arm10/arm9 1.000, arm17/arm16 0.999; all 14 DRAM paddr words equal the simulation's; arm 16 = arm 17 = 0x0 | R-68 FIXED by the ratio keys |

R-68 cycles per arm 1-17: 250 250 250 250 272 272 272 272 275 275 250 1832 304 326 328 1818 1817. Arm 12 reads the
board's bootrom doubleword 0x3049107300800913 (the simulation's bootrom differs) and is not graded, per the board lane's
amendment 2, which restricted the word check to the DRAM arms before niss-superpage was read. R-68's residual stands:
the test cannot tell the fix from deleting the splice. N = 1, board 2.

## Addendum 2026-11-26: these images on Line B's final Full core b42c8c644, in simulation only (pre-registered before the run)

**Why.** The four fixes are in b42c8c644's ancestry, through P1 d83d3a633. Every fix file changed after P1, so
ancestry does not settle it. By source, each fix commit's added lines are present at b42c8c644:
- R-68 213378a11: 9 of 9.
- R-32 0ab365c26: 19 of 19.
- R-72 5af46d00e: commit_stage 5 of 5. Its issue_read_operands line that is missing is the DYN-bypass lever, not R-72.
- R-60: r60_* is present in ex_stage.sv and ariane_pkg.sv.
This run checks the behaviour.

**The tree.** capstone-ariane 4bbdbc6ed (s2pre-wt). Its core differs from b42c8c644 by one `$display` in a
simulation-only block of capstone_sublet_life.sv. Settings: rtl_dirty=no, S12_MEM_DELAY=12, time_out 3,000,000, and
the images rebuilt here, which reproduce SHA256SUMS byte for byte.

**What Line B cannot run, and why:**
- Every image but niss-superpage mints with the custom-3 debug family, and so does this P1 wrapper's recorder. Under
  the Sublet format that family is illegal (decoder.sv:1227-1234), so the run defines CAPSTONE_SIM_DEBUG_MINT, the
  simulation-only scaffold.
- So these readings test the RTL fixes on Line B. They are NOT a silicon kit for Line B: that needs CSMINT ports and
  sublet-life as the control.
- va-runv uses CSRUNV, which B5b-2 step 8a deletes.
- pcc-end and both its controls use CJALR, which is illegal under Sublet in every build (decoder.sv:1555-1557).

**Predictions:**
- **alias-store:** this kit's fixed vector, `0x5a5a 0x1234 0x0 0x1 0x18 0x5a5a 0x7 0x58`.
- **shrinkto:** its 16 fixed values, with position 6 = 0x40.
- **niss-superpage**, run with +CAPSTONE_FPGA_NI as well: the FIXED ratio keys, i.e. arm1/arm2 and arm10/arm9 at
  most 1.25 and arm17/arm16 at least 0.85, with traps 0. The paddr words are recorded and are not graded here.
- **va-runv, pcc-end, pcc-end-in, pcc-end-out:** each ends at an illegal-instruction trap at its deleted instruction,
  and NOT its bitstream-1 vector. The recorder's contents are recorded.

**Grading:**
- A value-for-value match on alias-store and shrinkto, and the ratio keys on niss-superpage, read as FIXED ON LINE B
  IN SIMULATION.
- Any other vector is a finding, reported with its values.
- If the wrapper's recorder does not run under the scaffold (an empty recorder on every image), the run is VOID and
  says nothing about the fixes.

### Result on b42c8c644's core (2026-11-26)

The run: s2pre-wt `.sup/sck`, plus `.sup/sckni` with +CAPSTONE_FPGA_NI. rtl 4bbdbc6ed, dirty=no, PROVENANCE by the Anvil
prologue. The images were rebuilt and reproduce SHA256SUMS. Vectors were read by board-vector.py from each trace.

- **niss-superpage, R-68: FIXED ON LINE B IN SIMULATION.**
  - With the NI rule: arm1/arm2 287/279 (1.03), arm10/arm9 331/333 (0.99), arm17/arm16 1916/1875 (1.02), traps 0.
  - These are bitstream 1's simulated cycles value for value. Every paddr word is the same in the two runs (with and
    without NI).
  - Without the NI rule, only the device arms 12, 16 and 17 move.
- **va-runv, pcc-end, pcc-end-in, pcc-end-out: as predicted.** Each traps cause 2 at its deleted instruction and does
  not read its bitstream-1 vector. va-runv's recorder fills with trap prints. The pcc-end triple reads the same
  `0x0 0x2 ...` in all three arms.
- **alias-store (R-60) and shrinkto (R-32): VOID.** Each test executes CAPENTER in its own code, at
  cap-alias-store.S:59 and shrinkto-readback.S:46. The trap is ILLEGAL_INSTR at 0x80000258 and 0x80000264, word
  0x1ab5105b = custom-2 funct3 1 funct7 0x0d.
  - CAPENTER is illegal under the Sublet format (decoder.sv:1508-1512), and capmode is tied to 0 there
    (csr_regfile.sv:349). So neither test ever runs in the capability mode it was written for.
  - alias-store's `0x5a5a 0x1234 0x1 0x1 0x2 0x9999 0x7 0x58` is an integer-mode run after a cause-2 trap. Its
    0x9999 is NOT the R-60 alias.
  - shrinkto recorded only the trap (count 1, 0x2).
  - **Not predicted:** the pre-registration's census covered the mints, CSRUNV and CJALR, but not CAPENTER. A census
    over every encoding b5b2-illegal.S lists finds CAPENTER in four of the five tests; niss-superpage alone uses no
    deleted encoding.
- **R-72 on Line B is UNREACHABLE by construction**, read from source at b42c8c644 (an rtl-oracle read, with the
  gating lines re-read here).
  - The check at commit_stage.sv:423-428 requires `priv_lvl_i == M && capmode_i`, and under the Sublet format
    `capmode_d = 1'b0` (csr_regfile.sv:349). Its only other writer is reset (:3453; the register's update is :3549).
  - The PC metadata itself CAN become a capability on Line B. CBNZ is decoded under Sublet (decoder.sv:1563-1569; the
    reference keeps cscbnz, capstone-qemu 9289114e06 insn32.decode:970) and loads the PC metadata from its tagged
    operand. So R-72 returns if capmode is ever restored, or in an IDS55 build.
  - Recorded, not traced: CBNZ setting the PC metadata from protected U, and commit_stage.sv:458 and below adopting a
    PC record id without a capmode gate (read as inert).

**What R-60 and R-32 need on Line B:** a Line B-native test.
- Mint through CSMINT in M or S. The FLU ops involved (CINCOFFSET and SHRINKTO) are not deleted.
- Check the result through LDC/STC or a protected-U access, not through capability mode.
- Use the B5b-2 wrapper and sublet-life as the control.
Until then both are FIXED IN RTL on Line B by source only (their fix lines are present at b42c8c644).
