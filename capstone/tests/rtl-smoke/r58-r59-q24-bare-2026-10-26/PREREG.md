# R-58, R-59 and Q-24 on silicon: pre-registered 2026-10-26, before any board run

Six images, one session, in this order (the control first; all six are expected to RETURN). Each vector is the
recorder's words at `board_rec`, from simulating the image itself on dcc6daf6e's RTL through the Anvil prologue
(`rtl_dirty=no`, `S12_MEM_DELAY=12`, `+time_out=3000000`); retired counts differ per image.

| image | sha256 | board_rec | SR | vector on dcc6daf6e |
|---|---|---|---|---|
| `va-runv.bin` (control) | `274823b4df9e3a4a077e36029e5a4cc7e8a45cac7f9b1e124c0d9cde4caa286a` | 0x8000c000 | 15 | `0x2 0x2 0x1234 0x2 0xb 0x0 0x0 0x55 0x0 0x77 0x77 0x5 0xb 0x0 0x9` |
| `va-srevroot.bin` (R-58) | `2fbafdfd2eaec3c1ce31b5e8a664eea64ff558c17f6801b29014401fc47056a3` | 0x8000d000 | 20 | `0x2 0x2 0x1234 0x2 0xb 0x0 0x0 0x55 0x0 0x77 0x77 0x5 0xb 0x80007000 0x80007000 0x80007000 0x80008000 0x0 0x0 0x9` |
| `va-runv-r59.bin` (R-59) | `f551e9d2492ca6babba146c80f46b7687c3a4ef17480dea88b2d90168e695308` | 0x8000c000 | 15 | `0x2 0x2 0x1234 0x2 0x1d 0xfffffffffffffff0 0x0 0x0 0xfffffffffffffff0 0x0 0x0 0x5 0x1d 0x0 0x9` |
| `va-runv-r59-low.bin` (R-59) | `981fcf633194ee2039c7e191f504353f148f11104426fa616eadcabc5d91a5ac` | 0x8000c000 | 15 | the same as va-runv-r59 |
| `q24-deleg.bin` (Q-24 control) | `6fd2c0b60fe340ab2c0b4c1336eafc207acfbf5e8ef09f8e241037362ef60ca8` | 0x80003000 | 5 | `0x0 0x0 0x0 0x18 0x0` |
| `q24-deleg-on.bin` (Q-24) | `0e82f2adca300741f04f2d23110b27701d5377bcb2ad251e1bde2262856965f8` | 0x80003000 | 5 | `0x1000000 0x18 0x0 0x9 0xc` |

**The control** equals the va-runv board vector pre-registered on bf3db3175 in `vp2-bare-2026-10-17`, value for value.

**R-58** (va-srevroot, va-runv's vector plus five readings, positions 13-17 0-based): the caller writes srevroot = tbl
(0x80007000) and the frame carries tbl2 (0x80008000). Position 13 is what the caller wrote (tbl), position 14 what it reads
back after the escape: **tbl, its own** (0x80007000). On the pre-fix RTL it read tbl2, the context's table (the arm's
header, run srv2). Positions 15-16 print tbl and tbl2 for the comparison, 17 is the CSNODE variant's slot (0).

**R-59** (both arms): the frame's table word is tbl with bit 11 set, or 0x800 alone. Position 4 is +24, **0x1d (29): refused
at entry** on the refuse path (kind 2, S never left, the refuse event's pc 16 before ctx_ecall), and the context never
runs (data[8] 0 at position 9 where va-runv reads 0x77). Without the fix, by va-runv.S's own header: the first word is
admitted and reads exactly as va-runv (the masked root IS tbl), and the second is admitted with srevroot 0 (+24 not 29).

**Q-24:** q24-deleg-on reads medeleg = 0x1000000 (the bit took; the pre-fix mask reads 0), scause 0x18 seen in S at the
fault (+0), and M sees only S's ecall (9, 0xc past the fault). The control keeps the 24 in M at the fault and S's handler
never runs.

## Grading

- **VOID** unless va-runv reads SR 15 and its 15 values exactly.
- **Each fix CONFIRMED ON SILICON** when its image(s) read the vector above exactly: R-58 by va-srevroot, R-59 by both
  r59 images, Q-24 by q24-deleg-on with q24-deleg as its control (a q24-deleg reading that differs voids the Q-24 pair).
- Any other vector is a finding, graded value by value against the readings named above.
