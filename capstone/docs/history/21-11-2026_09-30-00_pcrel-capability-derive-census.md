# The PC-relative capability derive and load: per-benchmark census (2026-11-21)

The runtime lane asked for this census in the stage-2 shape: per benchmark, the static sites and dynamic executions
of each candidate, the instructions and fetch bytes removed, the encoding cost, and how each keeps authority with gp.

**Result.** The useful candidate is (a), one instruction for `auipc` + `scc rd, gp`.
- It removes 1.71 % of perl's instructions (1.96 % of fetch bytes) and 0.96 % of gcc's (1.12 %).
- Elsewhere it removes 0.15 % or less.
- (b), a PC-relative capability load, adds at most 0.09 % on perl beyond (a), and only if a range-limited
  encoding is accepted.

## Candidates, and what an encoding can actually remove

Under the application SDK's ABI (`-capstone-gp-free -capstone-image-gp`), a global's capability is formed as
`auipc rX, %pcrel_hi(sym)`, `addi rX, rX, %pcrel_lo(sym)` and `scc rd, gp, rX`. With `-capstone-fold-global-lo` the
addi folds into the following access.

- **(a) PC-relative derive, `scc.pcrel rd, imm20`:** rd = gp with cursor pc + (imm20 << 12).
  - It replaces `auipc` + `scc`. A 32-bit instruction cannot also hold the low 12 bits, so the addi stays (as a
    `cincoffsetimm` on the result) or stays folded into the access.
  - **One instruction (4 bytes) is removed per site, in both shapes.**
- **(b) PC-relative capability load or store:** the (a) sequence where the scc result's only use is one ldc/stc at a
  constant offset.
  - A full-range PC-relative capability load does not fit 32 bits either. Beyond (a) it removes one more
    instruction only with a range-limited offset (a 20-bit byte offset reaches ±512 KiB of the pc).
  - Its column below is that upper bound.

**Which sites count.** The census counts a site only where the encoding removes something:
- the scc's integer operand rX comes from an `auipc` within six instructions (an `addi rX, rX` may sit between, and
  nothing else touches rX);
- rX is dead after the scc: written before it is read again, with no branch or call first, which is conservative.

## Instrument

- Arm CRVC0 (the stage-2 baseline: FRCDKMSa3's SDK and flags, the compiler from 76061a895cb3).
- Shift-1 twin profiles, every command window, mapped onto each benchmark's `.full` image (`s2census.py`'s arena
  mapping).
- Program and library are separated by function symbol against the linked archives.

Checks:
- perl's total `scc gp` share reads 2.70 % of instructions, the figure in the stage-2 gap table.
- perl's hottest scc site is `auipc a2; addi a2; scc a2, gp, a2`, counted.
- Its next two hottest take an integer address reloaded from the stack (`ld a0, 0x278(sp); scc s3, gp, a0`) and are
  correctly not counted.

## Readings

| row | scc gp, % of insns | (a) sites | (a) executions | insns removed | fetch bytes removed | of which program | (b) further, upper bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| perl test-sw | 2.70 % | 14,491 | 1,444,962,346 | 1.71 % | 1.96 % | 100 % | 0.09 % insns |
| gcc train01 | 1.81 % | 61,597 | 116,971,542 | 0.96 % | 1.12 % | 100 % | 0.02 % insns |
| nab test | 0.35 % | 1,092 | 10,681,830 | 0.15 % | 0.17 % | 15 % | 0.00 % |
| xz test | 0.19 % | 509 | 24,475,319 | 0.05 % | 0.06 % | 100 % | 0.00 % |
| x264 test | 0.10 % | 1,473 | 184,401,271 | 0.04 % | 0.04 % | 100 % | 0.00 % |
| imagick test | 0.03 % | 6,575 | 7,668 | 0.01 % | 0.01 % | 64 % | 0.00 % |
| mcf test | 0.01 % | 86 | 1,031,975 | 0.00 % | 0.00 % | 14 % | 0.00 % |
| lbm test | 0.00 % | 68 | 432 | 0.00 % | 0.00 % | 2 % | 0.00 % |

- (a) reaches about half of the `scc gp` executions: 1.44 G of perl's 2.28 G, and 117 M of gcc's 221 M.
- The rest re-derive from an integer address that is still live, mostly reloaded from the stack, so no fusion
  removes anything. Keeping the capability live instead of the integer is a different, compiler-side lever.
- **Overlap with stage 2:** in the CRVC mode many of these scc are already 2-byte c.scc. For those sites (a)
  replaces 6 bytes with 4, so in a CRVC image its fetch saving is about half the column above; the instruction
  saving is unchanged.

## Encoding cost

- (a) needs rd and a 20-bit immediate, a U-type, which takes a whole major opcode. In P1's decoder
  (`core/decoder.sv` at d83d3a633) custom-0 holds the UART debug printer's encoding and custom-2/custom-3 hold
  Capstone. custom-1 (0x2b) is the free major opcode, and (a) would take all of it.
- (b) in its range-limited form needs another U-type-sized field, and only custom-1 is free. (a) and (b) compete for
  the same opcode, and (a) is worth twenty times more.

## Authority: why neither candidate can mint a capability

- (a) takes its authority from gp, exactly as `scc rd, gp, rX` does: type, permissions, bounds and lifetime node
  come from gp, and only the cursor is new.
  - The PCC contributes only its integer address, never its authority. A form that derived from the PCC would hand
    out a capability over the code with the PCC's permissions, which the image does not hold as data.
  - The result is bounded by gp's bounds, so a cursor outside them behaves as an out-of-window cursor does today
    (R-60).
  - Under `-capstone-gp-captable` gp covers only the capability table, and the derive would not apply; the
    compiler would emit it only under image-gp.
- (b) accesses memory through the same gp-derived capability, checked against gp's bounds and permissions as the
  ldc/stc it replaces. It reads or writes nothing gp could not.

## Limits

- Static liveness, address order, a six-instruction window: a lower bound on the reachable sites.
- Fetch bytes and instructions, not cycles.
- The libc and runtime are counted as built: their sites would need a rebuild with the feature (perl and gcc:
  under 1 % of the executions).
