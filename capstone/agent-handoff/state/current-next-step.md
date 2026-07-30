# Current recommended next step

## 2026-07-30 — C-14: silicon-only domain wedge, narrowed but NOT root-caused

### What is solid (all measured today, every run with a control in the same session)

| rung | shape | silicon |
|------|-------|---------|
| beebs_primer1 | scalar global | PASS 582955588 (5 sessions) |
| gpsz | 1 static, 64 elems, 256 B | PASS 607423941 |
| gpcp | 1 static, 256 B | PASS 23404485 |
| gpw2 | 1 static, 2 elems, 8 B | ran (value unchecked, oracle was broken) |
| **gpw16b** | **1 static, 16 elems, 64 B** | **PASS 583391941** |
| **gpw16** | **1 static, 16 elems, 64 B** | **FAIL** |
| gpw4, gpw8 | 4 / 8 elems | FAIL |
| gpn1use0 | 1 global, 4 elems | FAIL |
| gpn2, gpn4, gpn2use0, gpn2use1 | 2-4 globals | FAIL |
| gpstress | 6 mixed | wrong value, does not wedge |
| SQLite | 1059 globals | FAIL (wedges in create/entry) |

**gpw16b vs gpw16 is the one solid one-variable result**: same size, same element count,
same access shape, same session. gpw16's store loop reuses the address register as the
loop counter --

    cincoffset a4, a3, a4   ; a4 becomes a CAPABILITY (&g[i])
    sw    a6, 0(a4)
    movc  a4, a6            ; a live SCALAR written over that same register
    bne   a6, a5, back      ; next iteration consumes a4 as an integer

-- and gpw16b (gpsz's value expression, `i*3+7` instead of `i+1`) keeps the index in its
own integer register and passes. The compiler emits the reuse form only when the stored
value is derived from the index.

Plausible mechanism: CVA6 keeps capability metadata in a SEPARATE shadow register file,
so a register can carry capability state that a scalar write does not clear. QEMU keeps
one unified value per register, which is why every one of these rungs is QEMU-green.

### WHY THIS IS NOT YET THE ROOT CAUSE

`check-movc-reuse.py` scores **5 of 8** against known board outcomes. It misses `gpn2`
and `gpn4` (both FAIL with no reuse detected) and false-positives on `gpw2` (has the
pattern, did not wedge). So either something else is also in play, or the reuse is one
surface of a broader hazard. Do not adopt it as the cause on the strength of the pair.

### TWO HYPOTHESES WERE RETRACTED TODAY — do not re-derive either

1. **"More than one global fails."** Held for most of the day and looked exact. Killed by
   `gpn1use0` (count=1, FAILS) and `gpsz` (count=1, 64 register-indexed elements,
   PASSES). Count correlated only because the multi-global rungs were all 16-byte arrays.
2. **"A global of exactly 16 bytes fails."** Killed by the size sweep: gpw8 (32 B) and
   gpw16 (64 B) also fail, while gpsz (256 B) passes. The size table that motivated it
   had mixed today's runs with historical ones.

**Method lesson behind both:** a correlation across rungs measured in DIFFERENT sessions
is not evidence. Re-measure the baseline in the same session before building on a split.
(gpsz/gpcp were re-verified today and are genuinely passing, so the baseline is real.)

### DEAD, with evidence (keep them dead)

* Descriptor record order vs cap-table index order — same enumeration in both emitters
  (`CapstoneAsmPrinter.cpp:857,938`; `CapstoneISelDAGToDAG.cpp:134-138`).
* `ldc rd, imm(gp)` mis-decoded — RTL matches QEMU exactly
  (`decoder.sv:1300-1315,1767-1770`; `capstone_dyn_unit.anvil:296-297,318-328`).
* Unrepresentable capability bases — `split` sets cursor == base, the cursorless branch,
  base exact at any alignment (R-11).
* Coarse capability tag granularity — one dcache line IS one capability
  (`DcacheLineWidth`=128, per-line `cap_tag_q`, `wt_dcache_mem.sv:134-136,409-421`);
  QEMU buckets per 16 B (`cap_mem_map.c:5-19`). Also refuted empirically: it predicted
  gpn2use1 would pass, and it failed.
* The documented register-indexed-load fault (`history/27-07-2026_17-05-00`) as the
  mechanism — its trigger is stores to more than one location; gpsz does 64 and passes.
* The glue's carve loop — `INTERP_BUILD_LIMIT=1` (2-entry table, one iteration) still
  fails, and gpn1use0 fails at count=1.

### NEXT

1. **Explain gpn2/gpn4.** They fail without the reuse pattern. Diff `gpn2`'s domain_main
   against `gpw16b`'s and find what else differs. This is free — no board.
2. **Re-run gpw2 with a working oracle.** Its host printf was generated as `printf("%%u")`
   so every gpw oracle file held the literal text `%u`; gpw2's "pass" means only that it
   did not wedge. Fixed in-tree, needs one rerun.
3. **If the reuse pattern survives (1),** the fix is compiler-side: stop reusing a
   register that holds a live capability as a scalar loop counter. Verify with
   `check-movc-reuse.py`, then re-run gpw16, gpn2, and SQLite.
4. **Do not touch the paper's SQLite claim yet.** It currently says SQLite has not run on
   the board, which remains accurate.

### PROCESS RULES (each cost a session)

1. ONE board session at a time; concurrent runners produce a bootrom loop that looks
   exactly like corrupt firmware.
2. `pgrep -f 'fpga_driver/run_'` matches its own command line — use `grep -E 'run[_]'`.
3. Do not `rm -f` the board lock before a run; it defeats the launcher's flock.
4. Never rebuild an artifact a live session depends on.
5. Compare artifacts by CONTENT, never size.
6. A UART capture is an ACCUMULATING buffer and can hold several unrelated sessions.
7. Verify a diagnostic knob actually reached the binary before trusting its result.
8. Always put a known-good control in the SAME session as the experiment.
9. Check a generated oracle is a NUMBER before believing a "pass".
