# R-61 on silicon: pre-registered 2026-10-24, before any board run

Three images, one session, in this order (the ordering rule: the known-good control first; all three are expected to
RETURN, none to wedge). Each vector is the recorder's words at `board_rec` (count at board_rec, values at
board_rec + 8·(1..SR)), taken by simulating the image itself on the bitstream's RTL through the Anvil prologue
(`rtl_dirty=no`, `S12_MEM_DELAY=12`, `+time_out=3000000`; a first attempt at 600,000 cycles was void: it never left the
wrapper's prologue loop, count 0 on all three with identical retired counts). Retired counts differ per image.

| image | sha256 | board_rec | SR |
|---|---|---|---|
| `images/r61-control.bin` | `c710e0d5156c2c0d54a6b7cc7445a0231dac79b7e120b7dba71269d11f31e9bb` | 0x80013000 | 36 |
| `images/r61-stale2.bin` | `993876a985b934b9034d895f565f1bd6ae471fa5960b376e1045f07cc49164fe` | 0x80014000 | 23 |
| `images/r61-stale.bin` | `f4d9ea5b47bd8b40ce02a01dd4a16f0f0c18f07b23806033ed7f5ff9e638dc54` | 0x80014000 | 22 |

## Vectors (SV, in order)

**r61-control** (vb-trans's full vp2 arm list), identical on c721c1a2a and on the fixed tree, and value for value the
vb-trans-vp2 board vector pre-registered on bf3db3175 (`vp2-bare-2026-10-17`):
`0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xd 0x0 0x0 0x5 0x99 0x0 0x5 0xf 0x0 0x0 0x0 0x5 0xf 0x0 0x5 0x0 0x5 0x1c 0x0 0x0 0x5 0x0`

**r61-stale2** (arms 0 and 1, then a refused consuming ldc, then a fresh entry whose only access is a legal load):

| tree | vector |
|---|---|
| **c721c1a2a (the R-61 prediction)** | `0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xf 0x0 0x0 0x5 0xf 0x4 0x80006008 0x0 0x0` |
| fixed tree dcc6daf6e | `0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xf 0x0 0x0 0x5 0x2 0x18 0x420590db 0x1234 0x0` |

They differ at positions 18-21 (0-based) only: csupcause 15 against 2 (the RETURN refused in a virtual context);
csupepc - stub7 = 4 (the load) against 0x18 (the RETURN); csuptval 0x80006008 (the load's own address, `data` + 8)
against 0x420590db (the RETURN's encoding); the copied word 0 (the load never completed) against 0x1234. Positions 14-16
are the refused ldc itself (5, 15, 0), the same on both trees.

**r61-stale** (the same, the second entry a load from an unmapped page):

| tree | vector |
|---|---|
| **c721c1a2a** | `0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xf 0x0 0x0 0x5 0xf 0x0 0x0 0x0` |
| fixed tree dcc6daf6e | `0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xf 0x0 0x0 0x5 0xd 0x0 0x0 0x0` |

They differ at position 18 only: 15 against the page fault's 13.

## Grading

- **VOID** unless r61-control reads SR 36 and all 36 values exactly, and each other image's SR is its table's.
- **R-61 CONFIRMED ON SILICON** when r61-stale2 reads the c721c1a2a vector exactly (and r61-stale its c721c1a2a vector).
- **Not reproduced on silicon** when r61-stale2 reads the fixed tree's vector on c721c1a2a: a finding about the
  simulation-silicon difference, to the RTL lane before any claim.
- Any other vector is a finding, graded value by value against the positions above.
- N=2 across both boards is welcome; the mechanism is a register's value and the simulation is cycle-deterministic, so
  the two boards must agree exactly.
