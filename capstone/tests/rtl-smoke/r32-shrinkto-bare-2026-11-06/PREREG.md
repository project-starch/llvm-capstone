# R-32 on silicon (SHRINKTO's size): pre-registered 2026-11-06, before any board run

One image, one cold bare boot. Its vector is the recorder's words at `board_rec`. They were taken from simulating the
image itself through the Anvil prologue (`S12_MEM_DELAY=12`, `+time_out=3000000`). The image prints 16 values and parks in
its CAP_PASS spin at 0x80000c84. The simulation ends at the time-out, after that spin.

It was simulated on three trees, all at `rtl_dirty=no`:
- **C.5a:** capstone-ariane c5a-eval 36a87ec91. Its core/ and corev_apu/ equal 1fc2cd5aa, the C.5a bitstream's.
- **Stage A:** stage-b-base 687849ee0, which is stage A 14f39f356 plus tests and two sim-only ifdef blocks.
- **Stage-B batch:** stage-b-batch fe4810dd1, which is stage A plus the R-32 fix (0ab365c26 = a3b7a22c0) and R-68's fix.

| image | sha256 | board_rec | SR |
|---|---|---|---|
| `shrinkto.bin` | `34311f6d26259cb19b6ef06588e8d22f2a20534287d63f6a7348d3bc97c26863` | 0x80004000 | 16 |

Each pair of positions gives the child's LCC start and end, each minus the 4096-byte parent's address:

| pos | arm | C.5a and stage A (simulated) | stage-B batch (simulated) | QEMU / the ISA's intent |
|---|---|---|---|---|
| 0, 1 | the parent | 0, 4096 | 0, 4096 | 0, 4096 |
| 2, 3 | SHRINK control, integer bounds [0, 64) | 0, 64 | 0, 64 | 0, 64 |
| 4, 5 | SHRINKTO 64 | 0, 63 | 0, 64 | 0, 64 |
| 6, 7 | SHRINKTO 1 | 0, 0 | 0, 1 | 0, 1 |
| 8, 9 | SHRINKTO 17 | 0, 16 | 0, 17 | 0, 17 |
| 10, 11 | SHRINKTO 2047 (largest immediate) | 0, 2046 | 0, 2047 | 0, 2047 |
| 12, 13 | cursor 4032, SHRINKTO 64 (cursor + imm == parent end) | 4032, 4095 | 4032, 4096 | 4032, 4096 |
| 14, 15 | cursor 16, SHRINKTO 32 | 16, 47 | 16, 48 | 16, 48 |

The pre-fix vector (C.5a, stage A), in print order:
`0x0 0x1000 0x0 0x40 0x0 0x3f 0x0 0x0 0x0 0x10 0x0 0x7fe 0xfc0 0xfff 0x10 0x2f`

The fixed vector (stage-B batch), in print order:
`0x0 0x1000 0x0 0x40 0x0 0x40 0x0 0x1 0x0 0x11 0x0 0x7ff 0xfc0 0x1000 0x10 0x30`

The C.5a and stage-A simulations match value for value. Every SHRINKTO end on them is one below the batch's, and the
four control values match on all three trees.

## Grading on C.5a (1fc2cd5aa) or stage A (c9c7952bd)

`r32-grade.py <uart.txt>` (default `--expect prefix`) reads the last `SUPTEST BEGIN R` block of the capture.
- **VOID** unless SR is 16 with `SUPTEST END`, and positions 0-3 read 0 4096 0 64. A short vector ending in
  `SUPTEST END`, with the controls intact, is not void. It is a FINDING: the trap handler recorded mcause after the arm
  that faulted.
- **R-32 CONFIRMED ON SILICON:** the pre-fix vector exactly. SHRINKTO gives imm - 1 bytes on this bitstream, and the
  compiler's +1 gives compiled code exactly N bytes.
- **NOT AS EXPECTED:** the fixed vector exactly. The bitstream would then hold imm bytes, and a +1 image would be one
  byte wider than its object. That contradicts the source line in 1fc2cd5aa, and it is a finding about the build.
- Any other vector is a finding. The grader prints it position by position, tagged PRE-FIX / FIXED / NEITHER.

## Grading on a stage-B bitstream (when one exists)

`r32-grade.py <uart.txt> --expect fixed`: the fixed vector is the R-32 fix on silicon. The VOID rule is the same.

The grader was dry-run on synthetic captures before any board run: the pre-fix vector, the fixed vector, a replayed
earlier boot followed by the real one, a trap after position 4, a broken control, a one-value change, no END, no data,
and the wrong test letter. Each took the branch named above in both modes.
