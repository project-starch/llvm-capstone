# Case 31 (fz10) diagnostic: the board reading (board lane records/diag31.result-lines.txt, lane c6cccea70ab8)

Tree A, kit bkit7-rev2xD, --only 31 with both controls. Every pre-registered value holds:
- `DIAG31 adata=0x3f93f61800:0x3f93f666c0 cursor=0x3f93f72de0 past_end=50976 q_decoded=0x3f93f71800:0x3f93f766c0`
- A second line: cursor 0x3f93f75be0, past_end 62752, the same q_decoded.
- Base and end are each shifted by exactly +0x10000, and every cursor lies inside the shifted bounds and outside the
  true ones.
- No fault: the case completes with exit 0, and the controls read 25 and 28.

alias-load (lane alias-load) separates "checked against the shifted bounds" from "not checked": checked.

**Correction to the pre-registrations' wording ("a far overrun aliases", 2026-10-20, from the board lane's claim-auditor,
lane c10210a7c904).** Aliasing is periodic, not "beyond a span". A size-n access at a cursor outside the representable
window passes iff (cursor - base) mod 2^(E+14) <= len - n, and otherwise traps 28. Checked against the
compress/decompress model at all 241,984 byte positions out to four spans past this 20,160-byte buffer: 0 mismatches.
- Overruns up to 45,376 bytes past the end trap. Beyond that, a len-wide band in every 64 KiB span aliases: 25% of
  positions over four spans, tending to len/span, about 31%.
- Case 31's cursor is 50,976 bytes past the end and 71,136 past the base: less than one span past the end, and
  5,600 bytes into the next band.
