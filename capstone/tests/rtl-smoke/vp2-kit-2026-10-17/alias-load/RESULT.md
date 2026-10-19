# alias-load: result on tree A (board 1, the board lane's records/alias-load.result-lines.txt at lane 925e4124dd0a)

Every pre-registered value holds:
- **inb:** decoded exact, value 0, AL_DONE.
- **alias:** decoded child+0x10000 : child+0x14ec0; value 165 (the marker at the cursor); AL_DONE; no fault.
- **outside:** decoded shifted by the same span; cause 28 at address = cursor; no AL_LOADED.
- **past** (in window): decoded exact; cause 28 at the cursor.

So a vp2 context's load at an out-of-window cursor IS bounds-checked, against the compressed bounds decoded at that
cursor (load_store_unit.sv 8522d72fe:1318-1356). It aliases, landing in memory a span away without a trap, exactly
when that decode contains the cursor. With the board lane's diag31 (q15 case 31: shifted by +0x10000, cursor inside,
no fault), that is case 31's MISSED on silicon. QEMU traps all three out-of-bounds arms with cause 28, because it
keeps exact bounds in registers between escapes.

Stores are not covered here.

**Correction to the pre-registrations' wording ("a far overrun aliases", 2026-10-20, from the board lane's claim-auditor,
lane c10210a7c904).** Aliasing is periodic, not "beyond a span". A size-n access at a cursor outside the representable
window passes iff (cursor - base) mod 2^(E+14) <= len - n, and otherwise traps 28. Checked against the
compress/decompress model at all 241,984 byte positions out to four spans past this 20,160-byte buffer: 0 mismatches.
- Overruns up to 45,376 bytes past the end trap. Beyond that, a len-wide band in every 64 KiB span aliases: 25% of
  positions over four spans, tending to len/span, about 31%.
- Case 31's cursor is 50,976 bytes past the end and 71,136 past the base: less than one span past the end, and
  5,600 bytes into the next band.

## alias-store (the store twin), result on tree A (board lane, lane f6da1c3aa9e2)

Every pre-registered value holds:
- **alias +0x10100:** decoded child+0x10000; before=165; AS_STORED; AS_READBACK value=60. The store overwrote the
  marker 64 KiB away, with no trap.
- **outside +0x16000:** the same shifted decode; before=90; cause 28 at the cursor; no AS_STORED.
- **past:** cause 28. **inb:** reads back 60.

So stores alias exactly as loads do, under the same condition. An overrun that meets it is a silent write to another
object. Filed by the board lane as ISSUES R-60 (dev d338e44c52d3).
