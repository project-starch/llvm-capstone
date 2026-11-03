# R-67 kit: pre-registration

This is written and pushed before any board run.

**Images** (board_rec 0x80003040, SR 11 each; `images/SHA256SUMS`):

| image | sha256 |
|---|---|
| t1-norevoke | ba89db3f915a53b5fe4f7ebee3fd2880a4ffbf838c8a5eb5431e781ea0e4a1e5 |
| t1-untagged | 92aea79fa39a020614febc597ca8c21ac5ec8aaa0b951fdaee7a4b14ec263d9e |
| t1-oob | 60d0db28ee338fd0c752e72f02fd5bef4629ab8591b444c990ca0135d43b7439 |
| t1-fire | 6482059c119290e9edede3337760085d7344478840ef80edb7313d6434be1f92 |

**Vectors.**
- They were simulated from THESE images on rtl-ids20-wt 4dcb67e4e (e7af48457's RTL tokens) and on vp8 038b540a3
  (ObjBounds 603528b8b + tests), each PROVENANCE `rtl_dirty=no`, memory delay 12. The two trees read identically.
- Readings R1..R11 are those of `tests/lv-t1-body.h`: opening marker, C, validity before, validity after, marker word
  before the jump, marker word after it, trap count, last mcause, last mepc − tgt, last mtval, closing marker.

| image | vector (bitstream 3 = ObjBounds) |
|---|---|
| t1-norevoke | 0x7101 0x800009e0 0x1 0x1 0x0 0x4d41524b 0x0 0x0 0xffffffff7ffff620 0x0 0x59 |
| t1-untagged | 0x7101 0x800009f0 0x1 0x1 0x0 0x0 0x1 0x18 0xfffffffffffffb10 0x800009f0 0x59 |
| t1-oob | 0x7101 0x800009e0 0x1 0x1 0x0 0x0 0x1 0x1c 0x10 0x800009f0 0x59 |
| t1-fire | 0x7101 0x800009e0 0x1 0x0 0x0 0x4d41524b 0x0 0x0 0xffffffff7ffff620 0x0 0x59 |

The prediction is **HOLE on both sealed bitstreams**: t1-fire reads validity 0 after the crevoke, the marker written,
and 0 traps. With the fix (1fc2cd5aa, in-tree simulation) t1-fire reads R6 = 0, R7 = 1, R8 = 25 and R9 = 0. The fixed
tree's image vector is to be simulated before that bitstream is run.

**The grading rule** (`grade/r67-grade.py`):
1. Every image hash and the resident bitstream name must match, else MISMATCH.
2. The three controls must read their vectors exactly, else VOID: the boot says nothing.
3. t1-fire is then graded as one of:
   - **HOLE**: the vector above;
   - **FIXED**: R4 0, R6 0, at least one trap, cause 25 or 1, R9 0;
   - **FINDING**: anything else.
4. Exit codes: 0 when the verdict is the arm's prediction (HOLE for bs3/objb, FIXED for `fixed`), 1 otherwise, and 2
   for NO DATA.

**Limits.**
- N = 1 per bitstream.
- The hole is shown for CJALR only. CALL/RETURN query the node by design, and an escape or resume installing a dead
  code capability is not exercised here.

## Readings (graded by `grade/r67-grade.py`; nothing was changed after them)

- **Board 1, bitstream 3** (caplifive_vp2c_e7af48457), 2026-11-03 19:25-19:35: `r67-grade.py run-b1-bs3 bs3` gives rc 0.
  All four vectors read exactly as tabled above, so **t1-fire is HOLE**.
- **Board 2, ObjBounds** (caplifive_vp2c_603528b8b), 2026-11-04: `r67-grade.py run-b2-objb-r67 objb` gives rc 0. All
  four vectors read exactly as tabled, so **t1-fire is HOLE**.
  - The first pass (04:52-05:09) read NO DATA on t1-norevoke because the GDB load_image timed out.
  - t1-norevoke alone was re-run in the next booking (05:10-05:11) and read its vector exactly.
  - The other three images are the first pass's files, byte-identical.
- The runtime lane graded both boards with the same result.
