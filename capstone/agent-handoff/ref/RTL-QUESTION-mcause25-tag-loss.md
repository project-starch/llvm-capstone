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

## It is not statement-specific

The workload bisect put the fault in extended phase 2->3, whose statement is `CREATE INDEX
idx_amount ON nums(amount)` — the obvious suspect, since building an index btree is the heaviest
thing there. **Refuted.** The source carries a matched control (`CAPSTONE_EXT_SKIP_INDEX`) that
replaces that statement with `SELECT count(*) FROM nums`, reaching it through the same
exec_ok/prepare/step machinery. One boot, control returning:

| arm | phase 3 statement | result |
|---|---|---|
| `E2` | stops before it | RETURNED |
| `E3S` | `SELECT count(*)` instead of CREATE INDEX | **WEDGED** |
| `E3N` | `CREATE INDEX` | wedges (measured separately) |

`E3S`'s latched trap is mcause 25 at `memcpy+0x2a8` — **the identical instruction** as the
CREATE INDEX arm:

```
    cincoffsetimm a2, s0, -0x60
    ldc           a2, 0x0(a2)
    cincoffset    a1, a2, a1     <== mcause 25
```

So the trigger is not the index build and not that SQL statement; it is `memcpy` being reached in
whatever state the workload is in by then. A ladder that merely stopped before CREATE INDEX would
have concluded the opposite, which is why the control exists.

## Already eliminated — please do not re-run these

From `ref/ISSUES.md` (dates as recorded there), so this handoff does not send anyone over old
ground:

* **Pool exhaustion** — the rev-node pool holds 65536; observed heads at the wedges were ~250-600.
* **Rev-node tag loss zeroing `valid`** — refuted by rung `s06rev` (returns 11, both arms, control
  green): `valid` sits in `data_rdata`, not in `ruser`, so zeroing `ruser` cannot clear it.
  `s06rev` also covers evict-and-refill of a capability round-tripped through memory WITH the
  validity queries that `ldc`/`stc` perform.
* **The entire revocation-validity family, arithmetically** — those sites raise
  `INVALID_CAPABILITY` = mcause **26**, and the wedge is **25**.
* **The fixup's store pattern** — `s06sfix` returns 2048 at 64 KB scale.

One region fact from that work is still open and unrelated to us: the rev-node pool at
`[0xBFF0_0000, 0xC000_0000)` is cacheable but excluded from the shadow-tag write, so an evicted
rev-node line loses its top 30 bits, i.e. part of `depth`. That is not the mcause-25 mechanism, but
a corrupted depth would affect revocation-tree walks and may deserve its own look.

## IT IS NOT memcpy-SPECIFIC, and it is NOT DETERMINISTIC (measured 2026-08-14, later)

Two boots after the above was written, both facts changed the shape of this report:

**A PASS -> FAIL FLIP on the SAME BINARY.** Arm `G6` (basic workload, stop after the row loop)
returned `rc=3` with all three rows in one boot and **WEDGED** in another. Same file, verified by
hash `f93a9188a9a4433c` in both (kept via BAKE_KEEP, not rebuilt). So the basic workload passing is
NOT reproducible, and any claim resting on a single passing boot -- including one I made -- is
unsafe.

**A THIRD capability, in a completely different place.** `G6`'s flip wedged in the domain's own
output writer, nothing to do with SQLite:

```
output_text+0xdc:
    ld          a2, 0x0(a4)      ; the payload length
    sd          a3, 0x0(a4)
    cincoffset  a1, a1, a2       <== mcause 25 -- a1, the SHARED-REGION PAYLOAD capability, is NOT_CAP
    sb          a0, 0x0(a1)
```

So three distinct capabilities have now been observed coming back untagged, in three unrelated
functions: a `memcpy` stack slot, the shared-region payload capability in `output_text`, and the
lookaside pointer in `sqlite3DbMallocRawNN+0xd8` (`ldc a0, 0x2a0(a0)`, which is where a full run
wedges). The common factor is *a capability read back from memory*, not any particular caller.

**Please read the whole report in that light**: the memcpy disassembly below is still the most
precisely characterised instance, but framing this as a memcpy bug would be wrong, and a fix
validated only against memcpy would not close it.

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
