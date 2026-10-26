# fdiv.d / fsqrt.d rounding on silicon: pre-registered 2026-10-26, before any board run

One image for the board: `fp-all`. Its vector is the recorder's words at `board_rec`, taken from simulating the image
itself on dcc6daf6e's RTL through the Anvil prologue (`S12_MEM_DELAY=12`, `+time_out=3000000`). The image finished its
ten prints and parks; the cycle limit ends the simulation. The two one-case images are listed for the record.

| image | sha256 | board_rec | SR | vector on dcc6daf6e |
|---|---|---|---|---|
| `fp-all.bin` | `07692df2c3710353ea8ae8512f79292ff92ef7bb511f4ce8f2f09b538f2e7dc4` | 0x80007000 | 10 | `0x17 0x13 0x3 0xc1d880ad3d2bf68c 0xc1d880ad3d2bf68d 0xf 0x4094146748614d06 0x4094146748614d07 0xc100794b211e882c 0x1` |
| `fp-c3.bin` (sim only) | `ddf722c8dd31d72b7027d1f4890c181ffa296127f675ba5ee4f54835b7f961e2` | 0x80007000 | 10 | `0x1 0x0 0x3 0xc1d880ad3d2bf68c 0xc1d880ad3d2bf68d 0xffffffffffffffff 0x0 0x0 0xc1d880ad3d2bf68c 0x1` |
| `fp-c15.bin` (sim only) | `8f98f1a1ede5aec155e0dd1b9851f2dc8ef4493022ad68496c4ff173e8a9492b` | 0x80007000 | 10 | `0x0 0x1 0xffffffffffffffff 0x0 0x0 0xf 0x4094146748614d06 0x4094146748614d07 0x3fc8e79b6fb34410 0x1` |

Positions (0-based):
- 0 and 1: the fdiv and fsqrt mismatch counts.
- 2 to 4: the first fdiv miss (index, got, want).
- 5 to 7: the first fsqrt miss (index, got, want; -1 and zeros when there is none).
- 8: the last case's quotient. It is the positive control that the loop reached the end: `fp-all`'s is the table's
  correctly rounded last a/b.
- 9: fflags (NX).

## Grading

- **VOID** unless SR is 10 and position 8 is 0xc100794b211e882c: the loop did not finish, or the image is not this one.
- **CONFIRMED ON SILICON** (the rounding defect is on the board): `fp-all` reads the vector above exactly.
- **NOT ON SILICON**: positions 0 and 1 read 0 with position 8 exact. The board's divider is correctly rounded on these
  512 cases, and the sim finding does not transfer. That would be a finding in itself about the bitstream's build.
- Any other vector is a finding, graded value by value. Different counts with the same first misses would point at a
  partial difference.
