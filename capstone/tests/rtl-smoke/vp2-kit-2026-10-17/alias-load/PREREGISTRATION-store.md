# alias-store, the store twin of alias-load: pre-registered 2026-10-20 before its board boot

alias-load showed that out-of-window LOADS are checked against cursor-relative bounds and alias when that decode
contains the cursor. The RTL check covers stores too: `lsu_ctrl.fu inside {LOAD, STORE}`, load_store_unit.sv
8522d72fe:1318-1356. If a store aliases, an overrun silently writes memory a span away.

**The program** (`alias-store.c`, image b68b3d2586a91d07, SDK t3-freeze-nb, -O1): the same child and arena as
alias-load, with the markers 0xa5 at the alias target and 0x5a at the outside target.
- One store per process: `sb a0, 0(s5)`, where s5 = cincoffset(child, offset).
- The target byte is read back through the arena's own in-bounds capability.
- Output: AS_PRE (cursor, decoded, before), AS_STORED, AS_READBACK value, AS_DONE.

**QEMU control** (qkit7-rev2): inb stores and reads back 60 (0x3c). alias (before 165), outside (before 90) and past
each fault cause 28 at address = cursor, with no AS_STORED.

**Board prediction** (tree A; arms inb, alias, outside, past, one process each):
- inb: AS_READBACK value=60, AS_DONE.
- alias: decoded child+0x10000 : child+0x14ec0, before=165, then AS_STORED and AS_READBACK value=60: the store lands
  64 KiB away and overwrites the marker. AS_DONE.
- outside: decoded shifted by the same span, before=90; cause 28 at address = cursor; no AS_STORED.
- past: decoded exact; cause 28.

**What decides.**
- alias stores (readback 60) and outside faults: stores alias exactly as loads do, so a far overrun WRITES another
  object without a trap.
- alias faults: stores are checked differently from loads.
- outside stores: stores are not checked when out of window.
