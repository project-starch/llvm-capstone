# Representability in the virtual Capstone QEMU, measured before CSMINT

Date: 2026-10-06. QEMU `virtual-capstone-prototype` at `cbd3764a64`. The
tools and their full output live in
`capstone/capstone-qemu/tests/virtual-capstone-m1/representability/`.

## Question

`CSMINT` and the later removal of the bounds side store both assume that a
capability stored with STC comes back from LDC with the same bounds and
cursor. Does QEMU's 128-bit encoding hold every value QEMU lets a program
create?

## Method

Host programs compile the worktree's `target/riscv/cap_compress.c` and test
the round trip of `cap_compress` and `cap_uncompress`. A guest probe runs in
protected U on the worktree's QEMU and stores and reloads two values that
fail the round trip.

## Findings

| Measurement | Result |
|---|---|
| Lengths below 4096 | Exact at all 64 base offsets tested, every length. |
| Lengths in `[2^k, 2^(k+1))`, `12 <= k <= 29` | Exact only with base and end aligned to `2^(k-9)`. Of the other cases most decode wider, a few narrower or shifted. |
| Lengths from `2^30` | No alignment up to `2^k` makes a class exact. |
| Cursor window for a small object | 2048 below the base to 14335 above it; the span is four times the length class from 4096 up. |
| Cursors inside exact bounds | 719,581 of 719,586 exact; the five failures are one 1 GiB region. |
| Guest: SHRINK to `[0x40000001, 0x40001001)` | Accepted. The stored bits decode to `[0x40000000, 0x40001008)`. LDC returns the exact bounds. |
| Guest: 16-byte object, cursor moved by 64 KiB | Accepted. The stored bits decode to the 16 bytes at the cursor, `[0x40012000, 0x40012010)`. LDC returns the exact original bounds. |
| Instrument control | Comparing the reloaded value against the decoded bounds fails as it must. |

Two encoder defects explain the irregular cases:

- `cap_uncompress` computes `1 << (E + 14)` as `int` (`cap_compress.c:102-103`),
  which overflows from `E = 18`, regions of 1 GiB and more.
- `cap_compress` rounds the end up (`:54-55`) without raising the exponent
  when the rounding carries into the next length class. Such values were
  already unaligned, so a creation check that rejects them removes this
  case entirely.

## Conclusion

QEMU's SHRINK, SHRINKTO, SPLIT, CINCOFFSET and SCC perform no
representability check, and every memory tag carries the exact bounds beside
it, so a program cannot observe the loss today. Without that side store the
second guest case would turn a 64 KiB cursor move plus STC and LDC into
authority over memory the capability never covered.

The prototype therefore checks every created bound and every cursor move
before any change and raises cause 29 instead of rounding. Cursor movement
inside the bounds needs no check once the decoder overflow is fixed. The
rules and their software consequences are in the
[prototype ABI](../plans/virtual-capstone-abi.md#representability). The RTL
encoding was not examined; this prototype is QEMU only.
