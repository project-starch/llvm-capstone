# A lost tag on a plain stc/ldc spill pair — two hypotheses proposed and both withdrawn

**Date:** 2026-08-23
**Status:** OPEN. The mechanism is not identified. What this note records is what has been RULED
OUT and why, plus the one question the eliminations narrow it to.

## The observation (board lane, on `caplifive_s10fix_80843404c.bit`)

A SQLite wedge localised to a single instruction:

```
0x104910  sqlite3WhereCodeOneLoopStart      (fault 0x8C in)
  104950: stc            a2, 0x0(a0)        ; a0 = s0-0x70, spill pWInfo
  ... 9 unrelated stores, none in the granule ...
  104998: ldc            a4, 0x0(a0)        ; reload, SAME address, 18 instructions later
  10499c: cincoffsetimm  a4, a4, 0xb0       ; TRAPS
```

`mcause 25 UNEXPECTED_OPERAND` (not 29), `mepc 0x82cf499c`, `privM=1`, `rev_node_head 630`. So the
reloaded value has **no tag** — data plausible, tag missing. QEMU runs the identical path tagged.

Trigger is two SQL queries over an EMPTY table differing by one line: `SELECT t1.a FROM t1`
returns and matches QEMU bit-for-bit; `SELECT t1.a FROM t1, t1 AS y` wedges. The self-join changes
register pressure and therefore spill layout — a mechanism-shaped difference, not a workload-shaped
one.

## Ruled OUT, with the evidence

**S-06** — in this bitstream. `25035c4c0` is an ancestor of `80843404c`, and the whole `core/`
delta from `f231b5af0` to `80843404c` is one file, `wt_dcache_mem.sv`.

**AMO invariant I-4** — confined to atomics, and the wrong polarity. `wt_axi_adapter.sv` deliberately
omits `ATOMIC_REQ` from `needs_tag` so that an AMO leaves a tag SET over scalar data. That
resurrects a tag; this fault loses one. A plain spill pair is not an AMO.

**Rev-node pool exhaustion** — `rev_node_head 630` against a 1021 limit.

**S-10b, the store-buffer word-versus-granule hazard mismatch — MY HYPOTHESIS, WITHDRAWN.** It
needs the load and a pending store to disagree at `[11:3]`, i.e. to share a granule at different
words. The board lane checked statically in both frames and found **no scalar store in that
granule at all**, and the faulting pair is at the SAME address, so a word compare matches. The
premise fails.

**S-10's own `gran_clr` — MY SECOND HYPOTHESIS, ALSO WITHDRAWN.** This looked strong because
S-10's safety comment names the symptom exactly: *"a plain store and a later `stc` could both be
resident, and 'any ctag=0 clears' would discard a tag the program had just written."* Two findings
killed it:

* a capability store is **one** write-buffer entry, not two — the same comment says *"it is
  granule-aligned, so it hits `wbuffer_hit_oh` at word 0"* — so there is no second word entry
  carrying `ctag=0` for the `stc` to trip over;
* the `is_cap`-sticky / `ctag`-overwrite asymmetry that looked like a defect is deliberate and
  documented (`wt_dcache_wbuffer.sv:743-750`): *"the tag is LAST-WRITER-WINS … exactly the
  architectural result of sw-after-stc"* — and it needs a plain store into the granule, which is
  not there.

**Both of my candidates ended up depending on the same granule-sharing plain store that does not
exist.** Proposing two and withdrawing both on static reading is the cheap outcome; the expensive
one would have been a board session for either.

## What the eliminations narrow it to

The tag is sourced from the write buffer or from L1 — **never from the store buffer**:

```
rd_ctag_o = wbuffer_gran_clr ? 1'b0
          : (|wbuffer_be)    ? wbuffer_data_i[wbuffer_hit_idx].ctag   <- WRITE buffer
                             : rd_ctag;                               <- L1 tag array
```

So while an `stc` is still in the **store** buffer — upstream of the write buffer — `wbuffer_be`
is 0 and the tag comes from an L1 array the store has not reached. Stale tag, while the data path
has its own interlock. That is "data correct, tag stale", which is the observed polarity.

The thing that should prevent it is `page_offset_matches_o` stalling the load, and at the same
address it ought to match at `[11:3]` on word 0. **So the open question is narrow: why did that
stall not happen, or why did it not cover the tag?**

## Next step, and what NOT to do

**A directed Verilator test of the minimal shape**, A/B'd across S-10 present and absent. RVFI
shows the returned tag directly rather than inferring it from a trap, and it needs neither the
board nor the `rootfs.ext2` lock, so it costs no one anything and cannot collide.

**NOT the padding experiment.** If the mechanism is residency of any kind, padding closes the
window and returns "it is residency" — which is already suspected, does not distinguish *which*
residency, and costs a board session. The trace distinguishes; the padding does not.

---

# RESOLVED: it is a NULL pointer in software. No tag was lost, and no hypothesis in this note was right.

**Date:** 2026-08-23. Settled by four latched values read together, from apertures that were on the
bitstream the whole time and had never been sampled.

```
sw=255  TRAP LOG {seen, mcause[6:0]}  0x99   -> trap_seen = 1, mcause = 25
        trap mepc  (LATCHED)          0x0000000082cf499c   <- MATCHES the faulting instruction
        trap tval  (LATCHED)          0x0000000000000000
```

The staleness guard passes: `trap_seen` is set and `mepc` is exactly the `cincoffsetimm` at
`0x10499C`, so the latch belongs to **this** fault rather than an earlier one. Without that check a
bare `tval` would have been unreadable — the same trap that made `TRAP LOG 0x89` a stale kernel
ecall.

**`tval` carries the rs1 CURSOR.** A capability that had lost its tag would still carry its address
bits and read pointer-like. **Zero means the value is genuinely zero, not a de-tagged pointer.**
`pWInfo` is NULL, and `cincoffsetimm a4, a4, 0xb0` is `&pWInfo->sWC` — on a conventional machine
that computes an offset from NULL and hurts nobody until the load; on Capstone it traps at the
offset computation.

**So: a null dereference in software. No lost tag, no store buffer, no hardware defect in this
fault.**

## The premise that was wrong, and it was available all evening

Everything above was built on "mcause 25 = UNEXPECTED_OPERAND = the operand is NOT_CAP". **On this
bitstream 25 has TWO live producers:**

```
ex_stage.sv:479       64'd24 + exception_code    ordinal 1 = UNEXPECTED_OPERAND
commit_stage.sv:226   64'd25  // INVALID_CAPABILITY (23 + 2)   <- base 23, the PC-capability check
```

Both verified in the flown commit `80843404c`, not merely in-tree. And
`core/anvil_build/capstone_unit.anvilh:298-300` states it outright:

> *"commit_stage.sv:205-228 (the PC-capability check) uses base 23 instead of 24 and so emits a
> DIFFERENT name for the same number. That block disagrees with both encoders and with
> riscv_pkg.sv and looks like an off-by-one in its own right."*

**That block was read in this same session** — it is where `ILLEGAL_OPERAND_VALUE = 30` was taken
from for the S-11 SEAL test, and the line immediately above the note was quoted at the time. The
fact needed was three lines away and was not applied.

A second argument in the same chain was **vacuous rather than wrong**: "mcause 25 not 29, so a tag
failure rather than a bounds failure". `capstone_flu_unit.anvil:57-90` gives `CINCOFFSETIMM` **no
bounds arm at all**, so 29 was never reachable and excluding it excluded nothing — an exclusion
that could not have come out the other way.

## The rule that already covers this, and was not applied

CLAUDE.md, verbatim: **"Ask what the instrument cannot distinguish — before the claim goes out, not
after."** `mcause 25` cannot distinguish UNEXPECTED_OPERAND from INVALID_CAPABILITY on this
bitstream. **No new rule is needed; the existing one was not applied to a cause code, only to
matchers and counters.** Worth recording that the failure was in scope of an existing rule rather
than in a gap.

## What dies with it

* **S-10b as the explanation** — already withdrawn on the address evidence.
* **S-10's `gran_clr`** — already withdrawn on the one-entry reading.
* **The capacity chain** — nine distinct granules against `WtDcacheWbufDepth = 8`, one over. It
  explained the 6537 non-firing pairs without special pleading and had a threshold prediction,
  which made it the best-supported of the three. **It is still not what happened.** A hypothesis
  can fit every constraint you have collected and be about a phenomenon that is not occurring.

The counting work survives as a fact about the frame even though the hypothesis does not: that
window really does need nine write-buffer entries against a depth of eight. If a genuine
capacity-related tag loss is ever chased, the shape is already measured.

## Still open, and not ours to close

Whether a silicon defect produced the NULL upstream. `sqlite3WhereBegin` allocates `pWInfo` via
`sqlite3DbMallocRawNN` and checks `db->mallocFailed`, so NULL should not reach the loop, and the
heap is 256 KiB on both silicon and QEMU — so the obvious configuration difference is already ruled
out. The board lane is converting the hang into a returning answer with an entry-point NULL check
and the `oom=/malloc=/free=` counters. **A non-zero `oom` on silicon with an identical heap under
QEMU would put it back on the hardware.**
