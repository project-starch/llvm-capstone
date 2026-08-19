# S-10 — a scrubbed capability still reads back LIVE while the store is in the write buffer

**A plain store that clears a capability's tag is not visible to a load of that capability until
the store drains.** For the whole time the store sits in the write buffer, `ldc` returns the
capability as if it had never been scrubbed. Measured on silicon at **3837 of 3840 — 99.92%**.

> **Sibling issues, so a reader with the wrong symptom is redirected immediately.**
> `S07-capability-untagged-on-reload/` is the tag-reordering defect and is **FIXED** by forbidding
> granule co-residency. **This issue is NOT that defect and is NOT repaired by that fix**, which is
> an allocation-time check between two write-buffer *entries*; this residual needs only **one**
> entry, so the check never fires and a load never consults it. `S09-.../` is the dropped-scrub
> consequence of S-07 and is also fixed.

## The defect

    stc  G, cap      drains to L1, cap_tag_q[G>>4] = 1
    sd   x, G+8      ONE plain entry, word 1, STILL RESIDENT in the write buffer
    ldc  G           granule-aligned, so it compares WORD 0, misses the word-1 entry,
                     and falls through to the STALE cap_tag_q  ->  returns a LIVE CAPABILITY

Mechanism in `wt_dcache_mem.sv:280`, `:301`, `:337`.

## Measured on silicon

Bitstream `caplifive_s07fix.bit`. Two arms differing by **exactly one thing** — the delay between
the scrub and the check. Verified in the disassembly that the only instruction-level difference is
the drain loop. 3840 slots per arm.

| arm | sequence | live capability after its own scrub |
|---|---|---|
| `wr6` | scrub, then type-query **immediately** | **3837 / 3840 (99.92%)** |
| `wr7` | identical, **+ 300-iteration drain** | **0 / 3840** |

**Probe control, in the same run, on both arms:** the delayed type-query loop reported `NOT_CAP`
**3840/3840**. So the query is proven to fire on this bitstream in this binary, which is what makes
`wr7 = 0` a real negative rather than a dead instrument.

The probe is the **type query** (`lcc` field 1, which returns 7 for NOT_CAP *without raising*), not
a scalar readback: the defect is `ldc` returning a live capability, so the capability is what must
be interrogated.

## It is PRE-EXISTING — measured, not inferred

Same binaries, pre-fix RTL (`a3dbae618`) in a worktree against the fixed model:

| test | pre-fix `a3dbae618` | fixed |
|---|---|---|
| `s07-wbuf-forward-residual` | 9 exc / 9234 cyc | 9 exc / 9234 cyc |
| `s07-wbuf-forward-residual-ctl` | 17 exc / 26361 cyc | 17 exc / 26361 cyc |

Identical: **the fix neither repairs nor worsens it.**

**Model-identity control**, because identical numbers are also what running the same model twice
produces: `s07-wbuf-tag-reorder` in that worktree gives **4 exceptions at 9150 cycles** — the
pre-fix signature — against **1 at 9138** on the fixed model, plus zero occurrences of
`gran_conflict` in the worktree source. So the worktree really is the pre-fix RTL.

## Severity — transient, but not an edge case

The window closes completely once the entry drains (`wr7` = 0/3840). But **while the entry is
resident the failure is very nearly certain**, at 99.92%. Short-lived is not the same as unlikely.

**Operational consequence:** a program cannot trust an immediate re-read to confirm it has
destroyed a capability. It can trust the destruction once the buffer has drained.

Contrast with S-07's dropped scrub, which was **persistent** — there the capability survived
indefinitely. This one is bounded by drain time.

## What this folder does not establish

Whether any real workload reads back a capability quickly enough after scrubbing it to observe the
window. The measurement is a directed test built to hit it deliberately; no naturally-occurring
instance has been identified.

Do not quote the Verilator figure (8 of 16 legs) as a rate — the trap handler drains the buffer
between legs and resets the phase, which is what made that pattern alternate. The silicon number
is the magnitude.
