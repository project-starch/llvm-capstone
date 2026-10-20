# R-60 window slack

`run.sh <capstone-qemu tree>` checks the claim `performCIncOffsetCombine` (CapstoneISelLowering.cpp, from #277) rests on.
Under R-60, a `cincoffset` whose result leaves the source's representable window writes it untagged. The combine
creates an intermediate cursor X+Y within 2048 bytes of an in-bounds access, so it is safe only if the window reaches
at least 2048 bytes below the base and past the end of every representable region.

**What it measures.** `win.c` is linked against the given tree's `target/riscv/cap_compress.c`: QEMU's own encoder,
decoder and R-60 predicate, `capstone_cursor_representable`. It does not re-implement them. It draws random requests
of 16 B to 8 GiB and keeps the regions that compress exactly. For each one it asks whether every cursor from base-2048
to end+2048 keeps its tag.

Recorded on capstone-qemu 02f03540d7 (csrunv-root-r59-r60), 2026-10-20:

- 103,045 exact regions, none short.
- The **positive control** demands 64 KiB of slack instead, and fails 6,841 of 6,842 regions. The check can therefore
  fire, and the windows are finite, as expected. run.sh exits 2 if the control does not fire.

**What it does not measure.** This is QEMU's model of the window, not the RTL's. That the board agrees rests on QEMU
and the RTL decoding compressed bounds identically, which this harness does not test. A window or decode difference
on silicon would need its own QEMU-vs-RTL enumeration at the window edges before this result could be carried over.
It is also a random sample, not an exhaustive enumeration.
