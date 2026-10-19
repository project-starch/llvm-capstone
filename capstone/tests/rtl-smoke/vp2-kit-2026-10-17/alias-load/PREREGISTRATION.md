# alias-load: is a vp2 load at an out-of-window cursor checked against cursor-relative bounds, or not checked at all?
Pre-registered 2026-10-20 before its board boot.

**Why.** diag31 read q15 case 31's out-of-window pointer as LCC-decoded one span up (+0x10000) with the cursor inside,
and the load did not fault (board lane records/diag31). That reading cannot tell "checked against the shifted bounds"
from "not checked". The RTL says checked (8522d72fe load_store_unit.sv:1318-1356, decode at cursor_a, gated on
ld_st_priv_lvl M, and a CSRUNV context runs at M: ISSUES.md:2347-2350). This test needs a cursor that is out of window
AND outside the shifted bounds.

**The program** (`alias-load.c`, image 4aa36d96885ff0ab, SDK t3-freeze-nb, -O1):
- A 20,160-byte child (E=2, span 64 KiB) of a 64 KiB-aligned 512 KiB static arena, at arena+0x20000.
- One load per process: `lbu` through the offset capability itself; LCC reads its base and end just before.
- `capstone-vexec alias-load.dom OFFSET` prints AL_PRE (cursor, decoded bounds), then AL_LOADED (the byte), then
  AL_DONE.
- Marker bytes: 0xa5 at the alias target, 0x5a at the outside target. Every target is inside the arena, so mapped.

| arm | OFFSET | cursor vs true bounds | the decode at the cursor (qemu_compact_repr model) |
|---|---|---|---|
| inb | 100 | inside | exact |
| past | 0x4ed0 | 16 past the end, in window | exact; cursor outside |
| alias | 0x10100 | out of window | +0x10000:+0x14ec0; cursor INSIDE |
| outside | 0x16000 | out of window | +0x10000:+0x14ec0; cursor OUTSIDE |

**QEMU control** (qkit7-rev2): inb loads value 0 and completes. past, alias and outside each fault cause 28 at
address = cursor. LCC reads the true bounds in every arm, since QEMU keeps them in registers.

**Board prediction** (tree A, one process per arm, in the order inb, alias, outside, past):
- inb: AL_LOADED value=0, AL_DONE.
- alias: AL_PRE decoded = child+0x10000 : child+0x14ec0; AL_LOADED value=165 (the marker); AL_DONE.
- outside: AL_PRE decoded shifted by the same span; domain fault cause=28 at address = cursor; no AL_LOADED.
- past: AL_PRE decoded exact; cause=28.

**What decides.**
- outside faults with 28 and alias loads 165: the load IS checked, against the bounds decoded at its own cursor, and
  aliasing is the whole mechanism.
- outside completes (reads 90): out-of-window loads are not checked at all.
- alias faulting: the case-31 mechanism is wrong.
