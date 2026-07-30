# Current recommended next step

## 2026-07-30 — C-14: a global of EXACTLY 16 BYTES wedges the domain on silicon

### The evidence, sorted by global SIZE

| result | rung | global sizes |
|--------|------|--------------|
| PASS | beebs_primer1, bigwin | 4 |
| PASS | gptl | 13 |
| PASS | gpsz, gpcp | 256 |
| PASS | gppv | 512 |
| PASS | gpbg | 2400 |
| **FAIL** | gpn1use0 | **16** |
| **FAIL** | gpn2use0, gpn2use1, gpn2 | **16, 16** |
| **FAIL** | gpn4, gpn8, bigmany | **16 x N** |
| wrong value | gpstress | 4, 256, 2400, 256, 13, 512 |

**Every failing rung's globals are exactly 16 bytes and nothing else. No passing rung
has one.** 13 rungs, no exceptions.

`gpstress` is the odd one out and is probably a SECOND, separate defect: it has no
16-byte global, and it returns a wrong value rather than wedging.

### THIS SUPERSEDES THE "MORE THAN ONE GLOBAL" READING FROM EARLIER TODAY

Count looked like a perfect split for most of the day, and it was wrong. It correlated
only because every multi-global rung happened to be built from 16-byte arrays. Two
results broke it:

  * `gpn1use0` -- **count=1** -- FAILED, with beebs_primer1 passing in the same session.
    It is byte-for-byte gpn2use0 minus one never-accessed global.
  * `gpsz` -- ONE static global, 64 elements, the SAME register-indexed
    store-then-load shape as gpn1use0 -- PASSES.

Anything written earlier about "the carve loop's second iteration" is void. The loop is
fine; `INTERP_BUILD_LIMIT=1` on gpn2use0 (2-entry table, one iteration) still failed,
which was the first sign the count story was wrong.

### FIVE HYPOTHESES ARE DEAD (keep them dead)

1. *Descriptor record order != cap-table index order.* Both emitters walk `M.globals()`
   with the same filter (`CapstoneAsmPrinter.cpp:857, 938`) and `getGpCaptableIndex`
   assigns indices in that order (`CapstoneISelDAGToDAG.cpp:134-138`).
2. *`ldc rd, 16(gp)` mis-decoded.* RTL uses the standard sign-extended 12-bit immediate
   added raw to the cursor, same address for the bounds check and the access, trap on
   misalignment -- identical to QEMU (`decoder.sv:1300-1315, 1767-1770`;
   `capstone_dyn_unit.anvil:296-297, 318-328`).
3. *Unrepresentable capability bases.* `split` sets cursor == base, selecting the
   cursorless branch, where the base is exact at any alignment (R-11).
4. *Coarse capability tag granularity.* `DcacheLineWidth` is 128 bits so one line IS one
   capability, one `cap_tag_q` bit per line (`wt_dcache_mem.sv:134-136, 409-421`); QEMU
   buckets per 16 bytes (`cap_mem_map.c:5-19`). Both exactly 16-byte granular. Also
   refuted empirically: it predicted gpn2use1 would pass, and it failed.
5. *The documented register-indexed-load fault* (`history/27-07-2026_17-05-00`) as the
   mechanism. Its trigger is stores to more than one location; `gpsz` does 64 and passes.

### IN FLIGHT: the direct test

`gpw2 / gpw4 / gpw8 / gpw16` -- identical kernel shape, only the array size changes
(8 / 16 / 32 / 64 bytes), all count=1, all slot-0-only, all QEMU-gated (3983810698 /
1463068797 / 671377293 / 2928574773). Control `beebs_primer1` in the same session.

**Prediction on the record: gpw4 (16 B) FAILS, gpw2 / gpw8 / gpw16 PASS.**

  * Confirmed -> C-14 is "a 16-byte global", the correlation becomes a demonstration,
    and the next question is the mechanism (see below).
  * Refuted -> the size table is a coincidence across 13 rungs and the real variable is
    something all the 16-byte rungs share that gpw4 does not.

### WHY 16 MIGHT BE SPECIAL (mechanism, NOT yet demonstrated)

16 bytes is exactly one capability and, on this RTL, exactly one dcache line. The line
carries a single `cap_tag_q` bit, and bank 1 of the line holds capability METADATA when
the line is tagged and ordinary data when it is not
(`wt_dcache_mem.sv:151-161, 216-221`). A 16-byte global's storage capability spans
exactly one such line, so scalar stores into it and the tag bookkeeping for "this line
holds a capability" occupy the same granule.

Note the glue rounds every global up to at least 16 bytes
(`stor = max(align_up(size,16), 16)`), so a 4-byte global ALSO gets a 16-byte storage
capability and still passes -- the difference must be in which bytes the domain actually
touches, not in the capability's length. `gptl` (13 bytes) passing is the awkward data
point for any pure "bank 1 is touched" story, so do not commit to that mechanism yet.

### IF CONFIRMED, THE LIKELY FIX

Pad every gp-captable global's storage to more than 16 bytes in the glue's carve
(`stor = max(align_up(size,16), 32)`), or have the compiler over-align/pad 16-byte
globals. Both are cheap. Validate on gpw4 first, then SQLite -- which has **37 globals of
exactly 16 bytes**, so it is expected to fail on this mechanism and to be unblocked by
the same fix.

### PROCESS RULES (each cost a session)

1. **ONE board session at a time.** Concurrent runners power-cycle each other mid-load
   and produce a bootrom loop that looks exactly like corrupt firmware.
2. `pgrep -f 'fpga_driver/run_'` **matches its own command line**. Use `grep -E 'run[_]'`.
3. **Do not `rm -f` the board lock** before a run -- it defeats the launcher's flock.
4. Never rebuild an artifact a live session depends on.
5. Compare artifacts by CONTENT, never size.
6. A UART capture is an ACCUMULATING buffer and can hold several unrelated sessions.
7. **Verify a diagnostic knob actually reached the binary** before trusting its result
   (the INTERP_BUILD_LIMIT clamp was checked in the FPGA disassembly first).
8. Always put a known-good control in the SAME session as the experiment.
