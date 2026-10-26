# R-63 on silicon (fdiv/fsqrt under every rounding mode): pre-registered 2026-10-26, before any board run

One image, one cold bare boot. Its vector is the recorder's words at `board_rec`, taken from simulating the image
itself through the Anvil prologue (`S12_MEM_DELAY=12`, `+time_out=3000000`; the image finishes its 16 prints and
parks). It was simulated on two trees: bitstream 2's RTL (capstone-ariane r62-divsqrt e7af48457 with `core/Flist.cva6`
pointed back at the vendored normaliser, so the core is dcc6daf6e's; vp8 .sup/fprm-base, the build log naming the
vendored file) and the fixed RTL (e7af48457 as committed; .sup/fprm-fix, naming core/fpu_div_sqrt_fix).

| image | sha256 | board_rec | SR |
|---|---|---|---|
| `fp-rm.bin` | `45a7397cc4d4f099176ef8eaf45ff6a693d133d62924163e2b38d37459d5baa8` | 0x80003000 | 16 |

Positions 0-14 are, for frm = 0 (RNE), 1 (RTZ), 2 (RDN), 3 (RUP) and 4 (RMM) in that order: 1/3, -1/3, sqrt(2).
Position 15 is fflags.

| frm | correctly rounded (exact rationals) | bitstream 2 (simulated, this image) |
|---|---|---|
| RNE | 0x3fd5555555555555 0xbfd5555555555555 0x3ff6a09e667f3bcd | the same |
| RTZ | 0x3fd5555555555555 0xbfd5555555555555 0x3ff6a09e667f3bcc | the same |
| RDN | 0x3fd5555555555555 0xbfd5555555555556 0x3ff6a09e667f3bcc | 0x3fd5555555555556 0xbfd5555555555555 0x3ff6a09e667f3bcd (RUP's) |
| RUP | 0x3fd5555555555556 0xbfd5555555555555 0x3ff6a09e667f3bcd | 0x3fd5555555555555 0xbfd5555555555556 0x3ff6a09e667f3bcc (RDN's) |
| RMM | 0x3fd5555555555555 0xbfd5555555555555 0x3ff6a09e667f3bcd | 0x3fd5555555555555 0xbfd5555555555555 0x3ff6a09e667f3bcc (truncated) |

The bitstream-2 vector, in print order:
`0x3fd5555555555555 0xbfd5555555555555 0x3ff6a09e667f3bcd 0x3fd5555555555555 0xbfd5555555555555 0x3ff6a09e667f3bcc 0x3fd5555555555556 0xbfd5555555555555 0x3ff6a09e667f3bcd 0x3fd5555555555555 0xbfd5555555555556 0x3ff6a09e667f3bcc 0x3fd5555555555555 0xbfd5555555555555 0x3ff6a09e667f3bcc 0x1`

The fixed-RTL vector (what a bitstream with e7af48457 should read), in print order:
`0x3fd5555555555555 0xbfd5555555555555 0x3ff6a09e667f3bcd 0x3fd5555555555555 0xbfd5555555555555 0x3ff6a09e667f3bcc 0x3fd5555555555555 0xbfd5555555555556 0x3ff6a09e667f3bcc 0x3fd5555555555556 0xbfd5555555555555 0x3ff6a09e667f3bcd 0x3fd5555555555555 0xbfd5555555555555 0x3ff6a09e667f3bcd 0x1`

## Grading on bitstream 2 (caplifive_vp2c_dcc6daf6e)

- **VOID** unless SR is 16, and positions 0-5 (RNE and RTZ, the same on both trees and correct) read exactly as above.
- **R-63 CONFIRMED ON SILICON:** the bitstream-2 vector exactly.
- **NOT ON SILICON:** the fixed-RTL vector. The board would then round the directed modes correctly, which contradicts
  the source reading; that is a finding about the bitstream's build.
- Any other vector is a finding, graded value by value. RDN/RUP exchanged while RMM is correct (or the reverse) would
  split the two halves of R-63.
