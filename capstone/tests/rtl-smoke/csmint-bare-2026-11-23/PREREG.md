# CSMINT on silicon (B5b-1 step 3): pre-registered 2026-11-23, before any board run and before the bitstream exists

**Bitstream:** the first B5b-1 full build of capstone-ariane b5b1-step3 d0afdb40f (`…-pagedtbl-r76-r75d-b5b1_d0afdb40f.bit`
when sealed; its sha256 is recorded at the seal and checked before the flash). Each vector is the recorder's words at
`board_rec`, from simulating THESE images on d0afdb40f's RTL through the Anvil prologue: `rtl_dirty=no`,
`S12_MEM_DELAY=12`, `+time_out=3000000`. rc=2 is CAP_PASS's spin after a complete recorder.

**Checked against the reference-derived expectations, by script.** Every vector equals its in-tree test's
pre-registered expectation (capstone-ariane d0afdb40f: sublet-life.expect, csmint.expect, csmint-vm.expect VECTOR and
NONCANON, csmint-pmp.expect RIGHT), with the image addresses shifted by the wrapper's +0x1000. Those expectations were
computed by capstone-qemu c1985bb529's own code before their in-tree runs. The ids (for example a17's 0x80000007) are
not addresses and are not shifted.

| order | image | sha256 | board_rec | SR |
|---|---|---|---|---|
| 1 | `images/sublet-life.bin` | `eb281996594bcec1c1d6c922ab02eefbc196321654c3bb911b2175d82808c491` | 0x80003100 | 35 |
| 2 | `images/csmint.bin` | `e9ba3863d3bc8f7005d328f1ba1a767b71e81e0251bf29aa0aaef0122d262c9f` | 0x80004180 | 51 |
| 3 | `images/csmint-vm.bin` | `74f4c5896d0d2dd10868d4aa74711d150fceaaa3a76dcb1687ed46a6eed7c969` | 0x8000c000 | 30 |
| 4 | `images/csmint-vm-nc.bin` | `d73ceff87be78e06ef409ead5ec65816d726d5860d2413262e3476a2f5d940e4` | 0x8000c000 | 30 |
| 5 | `images/csmint-pmp.bin` | `7d672a84e8d8dda5bc259463d04820cc3af30eb7ff1a9941a96542c2dc965d37` | 0x800071c0 | 16 |
| 6 | `../r79-pmp-bare-2026-11-23/images/r79-pmp.bin` | `8eb72553965cfc1c98fd33b4caa44a0b0f1051e661092b63ed41d1cf4d954ec5` | 0x80007080 | 17 |

The order puts the Sublet control first. Every image is expected to RETURN.

## Vectors (SV, in order)

**sublet-life**: `0x0 0x0 0x0 0x19 0x0 0x0 0x19 0x0 0x0 0x0 0x19 0x19 0x2 0x2 0x8000000000010000 0x5 0x300000002 0x7 0xc00000000 0x100000000 0x1000000000 0x400000000 0x0 0x200000000 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x6`

**csmint**: `0x0 0x3 0x80004010 0x6 0x0 0x4 0x80004010 0x8 0x4 0x80004104 0x6 0x80004148 0x5a5a5a5a5a5a5a5a 0x6 0x80004168 0x5a5a5a5a5a5a5a5a 0x1d 0x0 0x1d 0x1d 0x1d 0x1d 0x5a5a5a5a5a5a5a5a 0x1d 0x0 0x5 0x1d 0x0 0x6 0x2 0x0 0x7 0x0 0x0 0x19 0x80000007 0x0 0x1d 0x0 0x5a5a5a5a5a5a5a5a 0x8000000000010000 0x9 0x100000008 0x8 0x0 0x0 0x0 0x0 0x0 0x0 0xd`

**csmint-vm**: `0x0 0x3 0x6 0xf 0x80003000 0x5a5a5a5a5a5a5a5a 0xf 0x80004000 0x5a5a5a5a5a5a5a5a 0xd 0x40000000 0xf 0x40000000 0xf 0x80003000 0x5a5a5a5a5a5a5a5a 0xd 0x80008100 0x0 0x4 0x8 0x77 0x77 0x77 0x77 0x8000000000010000 0x5 0x0 0x3 0x6`

**csmint-vm-nc**: `0x0 0x3 0x6 0xf 0x80003000 0x5a5a5a5a5a5a5a5a 0xf 0x80004000 0x5a5a5a5a5a5a5a5a 0xd 0x40000000 0xf 0x40000000 0xf 0x80003000 0x5a5a5a5a5a5a5a5a 0xd 0x80008100 0x0 0x4 0x8 0xd 0x800080008100 0xf 0x800080008140 0x8000000000010000 0x5 0x0 0x3 0x8`

**csmint-pmp**: `0x0 0x0 0x5a5a5a5a5a5a5a5a 0x7 0x80002000 0x5a5a5a5a5a5a5a5a 0x7 0x80002010 0x5a5a5a5a5a5a5a5a 0x0 0x3 0x77 0x77 0x77 0x2 0x2`

**r79-pmp** (R-79's kit, its fixed row, PREREG at dev 9d3a8afbce52):
`0x0 0x0 0x0 0x5a5a5a5a5a5a5a5a 0x7 0x80002000 0x5a5a5a5a5a5a5a5a 0x7 0x80002008 0x5a5a5a5a5a5a5a5a 0x7 0x80002000 0x77 0x77 0x77 0x77 0x3`

## Grading

- **VOID** unless both hold:
  - the boot's readback names the sealed bitstream;
  - sublet-life reads SR 35 and its 35 values.
  sublet-life is step 2a's pool arms: the Sublet format, CDERIVE, CREVOKE and CRENEW on silicon. Its pass is itself a
  reading (the Sublet unit on silicon), and its failure voids everything after it.
- **CSMINT CONFIRMED ON SILICON** when csmint, csmint-vm, csmint-vm-nc and csmint-pmp each read their vector exactly.
- **R-79 FIXED ON SILICON** when r79-pmp reads its fixed row exactly. Its pre-fix row is CONFIRMED on board 2 (dev
  819780239744).
- Any other vector is a finding, graded value by value against its in-tree test's header.
  - The first differing position names the arm.
  - Readings after the first failure in a boot are collateral, not results (the ordering rule).
  - A non-return is a wedge: read no further.
- N=2 across boards is welcome. The simulation is cycle-deterministic, so the boards should agree exactly.
- The default bitstream stays bitstream 1 until this kit reads (the delegated rule, agreed with the runtime lane).
