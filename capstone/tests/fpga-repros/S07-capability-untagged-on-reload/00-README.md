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

The same binary (`G6.dom`, sha256 `f93a9188a9a4433c`, kept across boots and **not** rebuilt) both
passes and wedges:

| boot | outcome |
|---|---|
| earlier | returned `rc=3`, three rows |
| later | **WEDGED** in `output_text+0xdc` (instance 2 above) |
| later still | returned `rc=3` ×4 consecutively, byte-identical `obs=0x5A6E0603`; a 5th attempt hit an R-16 **entry stall**, which carries no verdict |

So roughly **5 of 6 genuine executions succeed** and one wedges mid-run. Any experiment on this
defect needs repetition; a single passing boot proves nothing, and a single wedge does not
establish a deterministic trigger.

## What has been EXCLUDED, with positive controls that fire

Four ladder rungs, on this silicon, each returning `0xFFFF` (all sixteen slots intact) and each
shown able to report failure when fed a deliberately wrong operand:

| rung | question |
|---|---|
| `s06spill` | does a spilled capability come back TAGGED? |
| `s06bnds` | ...with its BOUNDS intact? |
| `s06wr` | ...surviving byte stores written THROUGH it? |
| `s06pld` | ...surviving a scalar load of its own granule? |

Plus, in the SQLite domain (which owns a 256 KiB heap a rung cannot): a capability held live across
a walk touching **every line of that heap** comes back with type and cursor unchanged — so a plain
evict-and-refill does not lose the tag either.

**Ruled out from disassembly, without a boot**: every instruction in `memcpy` touching the faulting
granule is `stc`, one plain `ld`, and three `ldc` — **zero plain stores**. So this is neither
correct tag-clearing on a partial overwrite, nor the write-buffer `.user` clobber
(`wt_dcache_wbuffer.sv:602` writes `.user` unconditionally whole-word while `.data` is byte-gated),
which requires a coalescing plain STORE to the same word.

**Ruled out previously** (see `agent-handoff/ref/ISSUES.md`): rev-node pool exhaustion; rev-node tag
loss zeroing `valid` (rung `s06rev`, returns 11 — `valid` lives in `data_rdata`, not `ruser`); and
the entire revocation-validity family **arithmetically**, since those sites raise
`INVALID_CAPABILITY` = mcause **26** while this is **25**.

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

SQLite does not run to completion on silicon. The basic workload (CREATE / INSERT / SELECT
returning all three rows / finalize) completes, mostly; the full workload wedges at instance 3.

## Files

* `board/` — the latched trap state and the decoded fault sites for each instance.
* `src/` — the four exclusion rungs (`s06spill`, `s06bnds`, `s06wr`, `s06pld`), each self-checking
  with a `*_SELFTEST` build that must return 0.
* `run.sh` — rebuilds and stages the exclusion rungs and prints what each should return.

Full investigation trail:
`agent-handoff/history/14-08-2026_02-30-00_sqlite-wedge-is-out-of-bounds-on-Mem.md`.
Handoff question: `agent-handoff/ref/RTL-QUESTION-mcause25-tag-loss.md`.
