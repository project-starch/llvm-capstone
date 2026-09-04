# S-12: the shape does not fault in a real capability domain either — 12,288 executions, zero faults

## The measurement

Rung `s12shape` runs the faulting shape with all three operands on one register, inside a real
capability domain on silicon:

    104f4  movc          a0, zero      the null the STC carries as its forwarded result
    104f8  stc           a0, 0x0(a1)   decoder.sv:1313 makes a0 this store's scoreboard rd
    104fc  ldc           a0, 0x0(a2)   the load that also claims a0
    10500  cincoffsetimm a0, a0, 0xb0  the consumer that reads a0

4096 iterations per entry, three boots.

| draw | control `k800` | `s12shape` | iterations |
|---|---|---|---|
| 1 | oracle 4, got 4, OK | `retval 20770` = `0x5122` | 4096 |
| 2 | oracle 4, got 4, OK | `retval 20770` | 4096 |
| 3 | oracle 4, got 4, OK | `retval 20770` | 4096 |

`0x5122` is the rung's "loop completed without faulting" verdict, written after the loop. The
progress counter reports 4096 every time, so the loop is proven to have executed rather than been
skipped, and the known-good control passed in every boot, so every boot carries a verdict.

**12,288 executions of the exact shape, in a domain, zero faults.** The same domain build of SQLite
faults at that same shape on essentially every run.

## What this eliminates

The domain context was the last untested variable of the three the search had:

* **the instruction shape alone** — refuted in bare-metal Verilator across three variants (warm,
  cache-missing, store-buffer-pressured), each with the tree's precondition counter firing every
  iteration and the outcome counter at 0;
* **the register relation** — established as necessary by the board ladder, to one byte;
* **the capability domain** — refuted here.

So the shape, the registers and the domain together are still not sufficient. Something SQLite
supplies beyond all three is required, and the search should move there rather than to further
permutations of the instruction sequence.

## Candidates that remain, in the order they look worth testing

1. **Scale of surrounding traffic.** SQLite's window sits inside a 4600-instruction function after
   ~1060 cap-table carves; this rung's loop is four instructions with nothing else in flight. The
   mechanism needs an STC *stalled* on a full store buffer, and a tight loop may simply never fill
   it however many times it runs.
2. **Slot geometry.** SQLite's load and store slots are frame offsets `s0-0x70` and `s0-0x120` on a
   monitor-carved stack; this rung uses one static buffer at +0 and +128.
3. **Dependency depth.** In SQLite the loaded capability comes from a spill written 18 instructions
   earlier by an incoming argument; here it is stored and reloaded immediately.

## Instrument notes, because this rung's zero is only worth its controls

The artifact was verified by disassembly before boarding: an earlier build had four separate
`__asm__` statements, and the compiler spilled between them and allocated three DIFFERENT registers
— `ldc a1`, a stack round-trip, `cincoffsetimm a0, a0`. That build would have returned a confident
clean result about a shape it did not contain. The four instructions are now one asm block with a
single tied early-clobber operand.

Three further plumbing bugs were caught under QEMU before any board time: passing the `res`
argument as a capability base (it is a plain pointer; a 16-byte-aligned static is the working
pattern), casting the buffer to `unsigned long` (strips the tag — QEMU asserted "cincoffsetimm with
an UNTAGGED rs1" and the compiler had warned), and writing `res[1]` in the QEMU variant (that region
is 4 bytes, so it is a capability OOB).
