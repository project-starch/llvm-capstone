# S-07 — a capability read back from memory comes back UNTAGGED, sporadically

**Status: OPEN. Silicon defect, not root-caused. Software workarounds do not address it.**

Bitstream `caplifive_12august.bit`. All measurements below are on that bitstream; a reflash
invalidates them.

**This is NOT S-06, and merging the two would be a mistake.** S-06 is *plain, untagged* data losing
its high 64 bits on an `ldc`/`stc` round trip — it corrupts data and raises nothing. S-07 is a
*genuine capability* coming back from memory with **no tag**, so the next instruction that requires
one raises **mcause 25 (UNEXPECTED_OPERAND)**. Different symptom, different cause code, and a fix
for one should not be assumed to touch the other. Sibling issues: `S06-untagged-ldc-stc-high-half/`,
`S01-image-perturbation-hang/`, `R20-stc-rs1-cursor-forward-x10/`.

---

## The signature

A capability is stored to memory, read back with `ldc`, and the reloaded value is NOT_CAP. The next
`cincoffset` (or dereference) raises mcause 25. Three instances, in three unrelated functions, none
of which share a caller:

**1. In our `memcpy`'s byte tail loop** — the most precisely characterised instance:

```
memcpy+0x2a8:
    lhu           a0, 0x24(a0)     ; a SCALAR load off the pointer      -- SUCCEEDS
    cincoffsetimm a2, s0, -0x60
    ldc           a2, 0x0(a2)      ; reload the dest pointer from its stack slot -- SUCCEEDS
    cincoffset    a1, a2, a1       <== mcause 25: a2 is NOT_CAP
    sb            a0, 0x0(a1)
```

**2. In the domain's own output writer** — nothing to do with SQLite:

```
output_text+0xdc:
    ld          a2, 0x0(a4)        ; the payload length
    sd          a3, 0x0(a4)
    cincoffset  a1, a1, a2         <== mcause 25: a1, the SHARED-REGION PAYLOAD capability, is NOT_CAP
    sb          a0, 0x0(a1)
```

**3. In SQLite's allocator** — where a full workload run dies:

```
sqlite3DbMallocRawNN+0xd8:
    ldc  a0, 0x2a0(a0)             ; db->lookaside.pSmallFree; mcause 25
```

The common factor is **a capability read back from memory**, not any particular caller, primitive,
or data structure.

## It is SPORADIC, and that is part of the signature

The same binary (`G6.dom`, sha256 `f93a9188a9a4433c`, kept across boots and **not** rebuilt —
verified by hashing the initramfs cpio members, not the staging directory) both passes and wedges.
Measured deliberately, 2026-08-14: a control domain then eight repetitions per boot, three boots,
all three controls passing.

| source | genuine executions | passed | **wedged** | entry stalls (excluded) |
|---|---|---|---|---|
| earlier record | 6 | 5 | 1 | 1 |
| boot 1 | 2 | 1 | 1 | 0 |
| boot 2 | 4 | 4 | 0 | 1 |
| boot 3 | 1 | 0 | 1 | 0 |
| **total** | **13** | **10** | **3** | **2** |

**p(wedge) ≈ 3/13 ≈ 23% per execution.** An R-16 entry stall is excluded from both numerator and
denominator — an image that never entered says nothing about the code in it, so counting one as a
failure would be wrong. Each boot stops at its first failure, so these are censored run-lengths,
not 8+8+8 independent trials.

**ALL THREE WEDGES ARE AT THE SAME INSTRUCTION**, `output_text+0xdc`. Not most of them — all,
across three boots. Boot 1 latched `mepc = 0x839416a8` and boot 3 `mepc = 0x835416a8`: different
4 MiB physical placements (two independent `__get_free_pages` allocations), both decoding to
domain VA `0x1516a8`.

So for a given image **the site is fixed and only the firing is sporadic**. This is the single most
useful thing in this folder: it names one `ldc`/`cincoffset` pair to look at rather than a class of
construct. The three instances listed above came from three different builds, which is consistent —
the site moves with the image, not between runs of one image.

One thing this does NOT show, because the overstatement is close by: the low 22 bits of the two
`mepc` values are identical, so every cache set index is the same in both. A **set-dependent**
mechanism is not excluded by this data.

Any experiment on this defect needs repetition: a single passing boot proves nothing, and a single
wedge does not establish a deterministic trigger.

## What has been EXCLUDED, with positive controls that fire

Four ladder rungs, on this silicon, each returning `0xFFFF` — all sixteen slots intact:

| rung | question | selftest build | gives |
|---|---|---|---|
| `s06spill` | does a spilled capability come back TAGGED? | `-DS06SPILL_SELFTEST` | 0 |
| `s06bnds` | ...with its BOUNDS intact? | `-DS06BNDS_SELFTEST` | 0 |
| `s06wr` | ...surviving byte stores written THROUGH it? | `-DS06WR_SELFTEST` | 0 |
| `s06pld` | ...surviving a scalar load of its own granule? | `-DS06PLD_SELFTEST` | 0 |

**Every rung carries a positive control and every one has been shown to fire**, because `0xFFFF`
from a query that cannot return anything else is not a measurement. The selftest build feeds the
same LCC query a value that is not a capability and requires the mask to collapse to 0; all four
do. The controls are exercised under QEMU, whose LCC field-1 is total with the same encoding
(`capstone-qemu/target/riscv/op_helper.c:713-716` returns 7 for an untagged operand), and the
control sits behind an `#ifdef` so the clean build is byte-identical to the one measured on
silicon.

`s06spill`'s control was added on 2026-08-14, after its silicon run — it had shipped without one
while the three rungs written after it all had one. Its 65535 stands (same bytes), but until that
date it was an unproven instrument, and this table said otherwise.

Plus, in the SQLite domain (which owns a 256 KiB heap a rung cannot): a capability held live across
a walk touching **every line of that heap** comes back with type and cursor unchanged — so a plain
evict-and-refill does not lose the tag either.

**Ruled out from disassembly, without a boot**: every instruction in `memcpy` touching the faulting
granule is `stc`, one plain `ld`, and three `ldc` — **zero plain stores**. So this is neither
correct tag-clearing on a partial overwrite, nor the write-buffer `.user` clobber
(`wt_dcache_wbuffer.sv:602` writes `.user` unconditionally whole-word while `.data` is byte-gated),
which requires a coalescing plain STORE to the same word.

**Ruled out previously — please do not re-run these** (recorded in `agent-handoff/ref/ISSUES.md`):

* **Rev-node pool exhaustion** — the pool holds 65536; the heads observed at wedges were ~250-600.
* **Rev-node tag loss zeroing `valid`** — refuted by rung `s06rev` (returns 11, both arms, control
  green). `valid` sits in `data_rdata`, not in `ruser`, so zeroing `ruser` cannot clear it. That
  rung also covers evict-and-refill of a capability round-tripped through memory **with** the
  validity queries `ldc`/`stc` perform.
* **The entire revocation-validity family, arithmetically** — those sites raise
  `INVALID_CAPABILITY` = mcause **26**, and this is **25**.
* **The S-06 fixup's store pattern** — `s06sfix` returns 2048 at 64 KB scale.
* **That it is specific to the `CREATE INDEX` statement** — refuted with a matched control that
  substitutes `SELECT count(*)` and wedges at the *identical* instruction. Table in
  `board/fault-sites.md`.

## Not reproducible under QEMU, structurally

QEMU is instruction-atomic with no cache, no write buffer and no eviction, and keeps a
full-precision bounds side-table for tagged loads (`cap_mem_map.h`). Its silence is not evidence.

## Two questions for the hardware side

1. **An R-20 analogue on another register.** R-20 was an issue-stage forwarding bug specific to
   **x10**, found in simulation and since fixed (`f623c48a1`, an ancestor of the current HEAD, so it
   should be in this bitstream). Instance 1 above faults on **`a2`**, instance 2 on **`a1`**. Has
   `issue_read_operands.sv` been audited for the same class on other registers? R-20's own README
   states it did not isolate which of two RTL sites was responsible, only that changing one cured it.
2. **Capability TYPE.** Every rung above spills a pointer to a static array — **NONLIN**. `stc`
   writes cnull back into rs2 for LINEAR/UNINIT/SEALED (`capstone_dyn_unit.anvil:458-461`), and
   `beebs_freestanding_string.c` already carries a `BEEBS_STRING_LINEAR_SAFE` knob because linearity
   has bitten these primitives before. Can a LINEAR or UNINIT capability round-trip through memory
   and come back untagged where a NONLIN one does not? We can build a rung for it given the shape
   worth testing.

## What would settle it

An RTL simulation of the instance-1 sequence — `stc` to a stack slot, a plain `ld` of its low half,
then `ldc` of the same slot — with the shadow tag `cap_tag_q` and the AXI tag byte instrumented, for
both a NONLIN and a LINEAR source. Note it is sporadic on the board, so a single clean simulation
does not exonerate the path; the interesting output is whether the tag can *ever* be dropped there.

## Impact

The **basic** workload — CREATE / INSERT / SELECT returning all three rows / finalize — runs to
completion on silicon roughly **77%** of the time, and wedges at `output_text+0xdc` the rest. The
**full** workload wedges at instance 3. So SQLite does execute on this hardware; what it does not
do is execute reliably, and the failure is in the domain's own output writer rather than in the
database engine.

## Files

* `board/` — the latched trap state and the decoded fault sites for each instance.
* `src/` — the four exclusion rungs (`s06spill`, `s06bnds`, `s06wr`, `s06pld`), each self-checking
  with a `*_SELFTEST` build that must return 0.
* `run.sh` — rebuilds and stages the exclusion rungs and prints what each should return.

Full investigation trail:
`agent-handoff/history/14-08-2026_02-30-00_sqlite-wedge-is-out-of-bounds-on-Mem.md`.

**This folder is the whole report.** An earlier draft of the same material lived in
`agent-handoff/ref/RTL-QUESTION-mcause25-tag-loss.md`; it was deleted rather than kept in sync,
because two documents for one issue is precisely how a live page ends up contradicting itself.
