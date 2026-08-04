# Archived reproducer packages — the issue is FIXED

A package moves here when the defect it reproduces has been **fixed in silicon** and the fix
has been verified on the board. It is archived rather than deleted for three reasons:

1. It is the evidence the defect was real. The R-14 investigation cost many board sessions and
   produced several retractions; the package records what was actually measured, not the
   conclusions that did not survive.
2. It is a **regression test for a bitstream**. The board is reflashed from time to time and
   not every bitstream carries every fix. Running an archived package against a new bitstream
   answers "does this one still have the fix?" in one boot.
3. Its probe technique is reusable — bounds/type measurement via `lcc` that cannot wedge, and
   one-variable pairs whose sources differ by a single constant.

**Do not hand an archived package to the board owner as an open issue.** Check the resolution
banner at the top of each `00-README.md` first.

| dir | issue | fixed by | verified |
|---|---|---|---|
| `R14-frame-pad/` | **R-14** | `caplifive_fixed_forward.bit` (`capstone-ariane 7aac52f93`, operand forwarding) | 2026-08-04 — `k1200` and `r14lp`, both previously failing, return 4 across two valid boots |
| `R14-strline-struct/` | **R-14** | same | same; this package was already superseded by `R14-frame-pad/` |

Active packages remain one level up in `capstone/tests/fpga-repros/`.
