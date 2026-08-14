# For the RTL side: a capability loses its TAG in memory, and it is not S-06

Written 2026-08-14 by the software side. One question, with everything measured that bears on it.
**This is a different defect from S-06** and should not be folded into it: S-06 is plain data losing
its high 64 bits (mcause 29 downstream, or silent corruption); this is a genuine capability coming
back **untagged** (mcause 25, UNEXPECTED_OPERAND).

---

## The observation

Bitstream `caplifive_12august.bit`. In our own `memcpy`, byte tail loop:

```
memcpy+0x2a8:
    lhu           a0, 0x24(a0)     ; a SCALAR load off a pointer      -- succeeds
    cincoffsetimm a2, s0, -0x60
    ldc           a2, 0x0(a2)      ; reload the DEST pointer from its stack slot -- succeeds
    cincoffset    a1, a2, a1       <== mcause 25: a2 is NOT_CAP
    sb            a0, 0x0(a1)
```

Latched trap state: `sw=255 = 0x99` → seen=1, mcause **25**.

## What the disassembly already rules out

Every instruction in `memcpy` that touches the granule at `s0-0x60`:

```
153af4  stc    <- the spill: the capability is written
153b18  ld     <- a PLAIN 8-byte load of its low half
153bdc  ldc
153c28  ldc
153d64  ldc    <- the faulting reload
```

**Zero plain stores.** So:

* it is **not** correct tag-clearing on a partial overwrite — nothing overwrites the granule;
* it is **not** the write-buffer `.user` clobber (`wt_dcache_wbuffer.sv:602` writing `.user`
  unconditionally whole-word while `.data` is byte-gated) — that needs a coalescing plain STORE to
  the same word, and there is none.

## Five shapes measured on silicon, all sound

Each is a ladder rung with a positive control shown to fire (a deliberately wrong operand drives the
result to 0/­collapses the mask), so these are real negatives, not untested gates.

| rung | question | result |
|---|---|---|
| `s06spill` | does a spilled capability come back TAGGED? | 65535 — yes |
| `s06bnds` | ...with its BOUNDS intact? | 65535 — yes |
| `s06wr` | ...surviving byte stores written THROUGH it? | 65535 — yes |
| `s06pld` | ...surviving a scalar load of its own granule? | 65535 — yes |
| `EVICT` (in the SQLite domain, 256 KiB heap) | ...surviving a full cache eviction and refill? | type and cursor unchanged |

The eviction arm matters because it exercises the AXI-level tag memory's refill path
(`wt_axi_adapter.sv`, one byte per 16B granule, the 4-state FSM around `:403`/`:731`).

## The question

**What else, in the RTL, can cause a tagged capability in memory to be reloaded untagged, given that
nothing writes to its granule in between?**

Two specific asks, both cheap for someone already in these files:

1. **An R-20 analogue on another register.** R-20 was an issue-stage forwarding bug specific to
   **x10** — a load's consumer saw an `STC`'s own base cursor instead of the loaded value — found in
   RTL simulation and since fixed (`f623c48a1`, an ancestor of the checked-out HEAD, so it should be
   in this bitstream). **Our fault is on `a2`.** Has `issue_read_operands.sv` been audited for the
   same class on registers other than x10? R-20's own README says it did not isolate which of two
   RTL sites was responsible, only that changing one cured it — so a sibling site is plausible.

2. **Capability TYPE.** Every rung above spills a pointer to a static array, which is **NONLIN**.
   `stc` writes cnull back into rs2 for LINEAR/UNINIT/SEALED (`capstone_dyn_unit.anvil:458-461`), and
   `beebs_freestanding_string.c` already carries a `BEEBS_STRING_LINEAR_SAFE` knob because linearity
   has bitten these primitives before. **Is there a path where a LINEAR (or UNINIT) capability
   round-trips through memory and comes back untagged where a NONLIN one would not?** This is the one
   simple axis our rungs do not cover, and we can build a rung for it if you tell us the shape worth
   testing.

## What would settle it from your side

An RTL simulation of the sequence above — `stc` to a stack slot, a plain `ld` of its low half, then
`ldc` of the same slot — with the shadow tag `cap_tag_q` and the AXI tag byte instrumented, for both
a NONLIN and a LINEAR source capability. If the tag survives in sim for both, the trigger involves
something our static reading has not reached and we should look at the surrounding SQLite context
rather than the primitive.

## Do not conflate

QEMU cannot reproduce any of this: it is instruction-atomic, with no cache, no write buffer and no
eviction, and it keeps a full-precision bounds side-table for tagged loads (`cap_mem_map.h`). Its
silence is structural, not evidence.

Full trail: `history/14-08-2026_02-30-00_sqlite-wedge-is-out-of-bounds-on-Mem.md`.
Related but separate: `ref/S06-WORKAROUNDS-TO-REVERT.md`.
