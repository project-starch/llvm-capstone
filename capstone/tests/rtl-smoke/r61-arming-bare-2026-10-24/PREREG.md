# R-61's arming conditions on silicon: pre-registered 2026-10-24, before any board run

Four images, one session, in this order (the control first; all four are expected to RETURN). Each vector is the
recorder's words at `board_rec`, from simulating the image itself on the bitstream's RTL through the Anvil prologue
(`rtl_dirty=no`, `S12_MEM_DELAY=12`, `+time_out=3000000`); retired counts differ per image.

| image | sha256 | board_rec | SR |
|---|---|---|---|
| `images/r61a-control.bin` | `d6becdd7209b19ab51c881bdd0136fd24655e9f4e23e04229eada96913fb1223` | 0x80013000 | 36 |
| `images/r61a-stale0.bin` | `30e1d23c1c2c497e48c3bdc46cd779ae9e1ad35911c21034089fd8bee8cbff42` | 0x80014000 | 19 |
| `images/r61a-stale3.bin` | `231b16cddbcbe5881f9b711545c68a236d5b079ea6c293743f6e7361035147d3` | 0x80014000 | 22 |
| `images/r61a-stale4.bin` | `9be0fc798e9e544f143deeaf8927cde167ece28f334d370a1c3cedc84fbd6426` | 0x80014000 | 22 |

## Vectors (SV, in order)

**r61a-control**, identical on c721c1a2a and on the fixed tree, and equal to `r61-control`'s confirmed silicon vector:
`0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0xd 0x0 0x0 0x5 0x99 0x0 0x5 0xf 0x0 0x0 0x0 0x5 0xf 0x0 0x5 0x0 0x5 0x1c 0x0 0x0 0x5 0x0`

**r61a-stale0** (arms 0 and 1, then stub7's legal load with NO LDC before it), identical on BOTH trees, a normal exit:
`0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0x2 0x18 0x420590db 0x1234 0x0`

**r61a-stale3** (arm 6's NON-consuming ldc from a read-only page, which completes: 5, 2; then stub7) and **r61a-stale4**
(the same with the page first remapped writable and D clear) read the same vectors as each other:

| tree | vector |
|---|---|
| **c721c1a2a (the prediction)** | `0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0x2 0x0 0x5 0xf 0x4 0x80006008 0x0 0x0` |
| fixed tree 0311ca24a | `0x0 0x5 0x2 0x28 0x420590db 0x77 0x1234 0x0 0x5 0x2 0x5a5a 0x0 0x0 0x5 0x2 0x0 0x5 0x2 0x18 0x420590db 0x1234 0x0` |

They differ at positions 17-20 (0-based) only: csupcause 15 against 2 (the RETURN refused); csupepc - stub7 = 4 (the load)
against 0x18; csuptval 0x80006008 (the load's own address) against the RETURN's encoding; the copied word 0 against
0x1234. Positions 13-14 (5, 2) are arm 6's non-consuming ldc completing normally on both trees.

## Grading

- **VOID** unless r61a-control reads SR 36 and its 36 values exactly, and each image's SR is its table's.
- **VOID for the arming question** unless r61a-stale0 reads its normal-exit vector exactly: without it, a 15 in stale3
  or stale4 could be stub7's load faulting by itself.
- **Arming condition CONFIRMED ON SILICON** per image when r61a-stale3 (non-consuming ldc, read-only page) or
  r61a-stale4 (non-consuming ldc, writable page with D clear) reads the c721c1a2a vector exactly.
- Reading the fixed tree's vector on c721c1a2a is "not reproduced on silicon", a finding to the RTL lane before any claim.
  Any other vector is a finding, graded value by value against the positions above.
- N=2 across both boards; the two must agree exactly.
