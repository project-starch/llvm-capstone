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
