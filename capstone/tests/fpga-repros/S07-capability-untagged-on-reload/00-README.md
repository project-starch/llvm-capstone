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

A capability is stored to memory, read back with `ldc`, and the value that comes back is NOT_CAP —
the next capability consumer raises mcause 25. Three instances, in three unrelated functions, none
of which share a caller.

> ### CORRECTION, 2026-08-14 — read before instances 1 and 2
>
> **For instances 1 and 2 the measurement does not establish which operand was wrong.** Both fault at
> a `cincoffset`, and that guard has TWO arms:
>
> ```
> core/anvil_build/capstone_flu_unit.anvil:29-31
> func CINCOFFSET(data){
>     if((data.cap_rs1.metadata.cap_type==cap_type_t::NOT_CAP)||(data.cap_rs2.metadata.cap_type!=cap_type_t::NOT_CAP)){
>         call raise_exception(data.trans_id,ex_code::UNEXPECTED_OPERAND)
> ```
>
> mcause 25 is raised if **rs1 is NOT_CAP _or_ if rs2 is anything other than NOT_CAP**. In both
> instances rs2 is an integer produced by a plain `ld` one to four instructions earlier. So "the
> reloaded capability lost its tag" and "the integer offset gained one" are indistinguishable in the
> data we have.
>
> **Instance 3 is unambiguous** and anchors the thesis: it faults *at* the `ldc`, whose guard is
> rs1-only (`capstone_dyn_unit.anvil:327-330`). So at least one genuine "a register that should hold
> a capability is NOT_CAP" event is established — by one instance, not three.
>
> Sentences below reading `<== mcause 25: aN is NOT_CAP` for instances 1 and 2 are therefore an
> INTERPRETATION that was stated as a measurement. The discriminating query is cheap and is in
> "What would settle it".



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
| earlier record (its one wedge was at `output_text+0xdc`, the same instruction) — **one of these 6 is not present in any surviving transcript; a recount from raw logs gives 5, i.e. 12 genuine overall rather than 13** | 6 | 5 | 1 | 1 |
| boot 1 | 2 | 1 | 1 | 0 |
| boot 2 | 4 | 4 | 0 | 1 |
| boot 3 | 1 | 0 | 1 | 0 |
| **total** | **13** | **10** | **3** | **2** |

**The "entry stalls" column is MISLABELLED and it is not R-16.** Those arms stop far earlier, at
`SQ: id=5` with `RGNO:0000E00C` / `RGNN:00000020` — deterministic **monitor region-pool exhaustion**
(32 regions) during setup, before the domain is entered. The signature is identical in every boot in
both measurement windows, so the exclusion is symmetric and cannot bias the comparison — but it does
mean **every boot is structurally capped at about 4 genuine `G6` executions**, which is the real
reason accumulating samples is slow.

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

**"Isn't that just the hottest loop?"** — the first fair objection, and no. `output_text` writes
the domain's output one byte per iteration, so it looks like a hot loop, but per execution it
writes only ~278 characters (3 result rows plus 15 `SQ:` markers) — on the order of 2 000
instructions, against a basic SQLite workload of at least a hundred thousand. That is under ~2% of
the run. Three independent wedges all landing inside a ≲2% region is p ≈ 10⁻⁵ under a uniform
fault; the concentration is real, not a sampling artifact of instruction frequency.

The same objection, answered the other way: if the trigger were something time-based rather than
site-based — an interrupt landing between the `ldc` and its consumer, say, with the domain context
save/restore losing a tag — the wedges would scatter across the workload in proportion to execution
time. They do not.

Any experiment on this defect needs repetition: a single passing boot proves nothing, and a single
wedge does not establish a deterministic trigger.

> ### THE RATE IS A PROPERTY OF THE WHOLE IMAGE, NOT OF `G6.dom` — added 2026-08-14
>
> **If you build only this domain and run it, you may see nothing at all.** Two independent
> measurements the same day show the defect responds to things outside the domain binary:
>
> * Adding ~85 instructions to `output_text` (an in-place probe) turned a working `CREATE` into
>   `rc=11` (malformed schema) — a completely different and much earlier failure. Verified as a
>   matched pair in ONE boot: the uninstrumented binary printed its three rows twice while the
>   instrumented one failed twice.
> * **THE DEFECT MAY HAVE STOPPED REPRODUCING — but this is NOT established, and an earlier version
>   of this note overstated it.** Since the rate was measured, `G6.dom` (byte-identical throughout)
>   has wedged 0 times in 14 further genuine executions. On the **like-for-like** comparison — the
>   same initramfs that is the only configuration ever observed to wedge — it is **0 in 8**, which
>   is **Fisher exact p = 0.26: no evidence of a change at all**. Across all images it is 0 in 14,
>   p = 0.098, still not significant.
>
>   The previously published "0 in 25, p = 0.0015, Fisher 0.034" is **WITHDRAWN**. It pooled in 11
>   executions of a *patched* binary built specifically under a hypothesis that predicts it will not
>   wedge; those arms are predicted not to wedge by both live explanations, so they cannot
>   discriminate between them, and pooling them to reach significance was circular.
>
>   A live alternative that the data does not exclude: **burstiness**. All three wedges fall inside
>   a single 17:27-18:40 window, and one boot *inside* that window was itself 0-in-4.
>
> An earlier version of this note attributed the suppression to three unrelated domains having
> been added to the initramfs. **That is RETRACTED.** Removing them again and rebuilding to a
> byte-size-identical cpio, with all 14 original domains byte-identical, did NOT bring the wedge
> back: 0 in 8 on the restored image. Image composition is therefore not the explanation, and
> physical placement is NOT promoted by this evidence — the earlier paragraph claiming so was
> written before the restoration test and was wrong.
>
> **What this means for reproduction.** The 23% figure is what was measured in one window on
> 2026-08-14. The defect is not reproducing now, on the same binary and an equivalent image, and
> the cause of the change is unidentified. Do not treat 23% as a rate you can rely on seeing.
> Candidates not yet separated: the several firmware rebuilds in between (content-identical monitor,
> relinked), some board-state or thermal effect after a long session, or genuine clustering that
> makes 3/13 a less stable estimate than it looked.

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

**Ruled out from disassembly, without a boot — but the argument differs PER INSTANCE, and the folder
previously gave only the instance-1 form:**

* **Instance 1 (`memcpy`)**: every instruction touching the faulting granule is `stc`, one plain `ld`
  and three `ldc` — **zero plain stores**. Neither correct tag-clearing on a partial overwrite, nor
  the write-buffer `.user` clobber (`wt_dcache_wbuffer.sv:602` writes `.user` unconditionally
  whole-word while `.data` is byte-gated), which needs a coalescing plain STORE to the same word.
* **Instance 2 (`output_text`, the thrice-measured site)**: this loop DOES execute a plain
  `sd a3, 0x0(a4)` on every iteration, so the "zero plain stores" argument does not apply here at all.
  The exclusion still holds, for a different reason: the write-buffer hit compares the full 64-bit
  word address (`wt_dcache_mem.sv:276`, `wt_dcache_wbuffer.sv:444`), and the scalar at `s0-0x48` is a
  different word *and* a different 16-byte granule from the capability at `s0-0x40`.

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

1. **An R-20 analogue on another register — WE HAVE NOW LARGELY ANSWERED THIS OURSELVES; it is here
   so you do not re-derive it.** `f623c48a1` is an ancestor of every candidate synthesis tree and was
   never reverted. R-20's signature is incompatible with S-07 anyway: it was x10-specific, silent, and
   trapped nothing. The entire hand-written core contains exactly three register-literal special
   cases, all CAPENTER/x10-x11 (`issue_read_operands.sv:573`, `scoreboard.sv:236-238`,
   `decoder.sv:1287`) — none names x11 as an operand and none names x12 at all, while our instances
   fault on `a1`(x11) and `a2`(x12). We no longer think this is the mechanism.

   Two workload facts that close whole branches, measured by disassembling the domain (327 860
   instructions): it contains **zero** `amo*`/`lr.*`/`sc.*` and **zero** hardware `mul`/`div`/`rem`
   (soft routines instead). Any hypothesis resting on the atomic path or the multiplier is dead
   without a boot.
2. **Capability TYPE.** Every rung above spills a pointer to a static array — **NONLIN**. `stc`
   writes cnull back into rs2 for LINEAR/UNINIT/SEALED (`capstone_dyn_unit.anvil:458-461`), and
   `beebs_freestanding_string.c` already carries a `BEEBS_STRING_LINEAR_SAFE` knob because linearity
   has bitten these primitives before. Can a LINEAR or UNINIT capability round-trip through memory
   and come back untagged where a NONLIN one does not? We can build a rung for it given the shape
   worth testing.

## What would settle it

**Two experiments, and the first one alone is not enough.**

1. *The memory path.* An RTL simulation of the instance-1 sequence — `stc` to a stack slot, a plain
   `ld` of its low half, then `ldc` of the same slot — with the shadow tag `cap_tag_q` and the AXI tag
   byte instrumented, for both a NONLIN and a LINEAR source.

2. *The register-delivery path — please do not skip this one.* An `ldc`'s metadata reaches the
   register file only via the CAP_WB port (`cva6.sv:1379-1380`, `:1401-1408`). If a response is
   bypassed to LOAD_WB instead, that port carries no capability (`scoreboard.sv:320-324` ties
   `wb[1..3].cap_data` to `'0`) and the scoreboard erases the entry's `cap_result`
   (`scoreboard.sv:242-246`); commit then writes metadata `'0` (`commit_stage.sv:279`) into the
   metadata regfile under the **plain GPR** write enable (`issue_read_operands.sv:1578`
   `we_pack[i] = we_gpr_i[i]`). That produces a NOT_CAP register with a correct cursor, having never
   touched memory. Instrument the CAP_WB/LOAD_WB routing of the `ldc`'s `trans_id`
   (`ex_stage.sv:933`).

**A clean result from experiment 1 excludes nothing about experiment 2's path** — and it would read as
exoneration, which is why both are listed. It is sporadic on the board, so one clean simulation does
not exonerate either path; the question is whether it can *ever* happen.

**We are also running a board experiment that discriminates these directly**: query the type of BOTH
`cincoffset` operands at the failing site, and on a lost tag re-`ldc` the same address. If the retry
comes back TAGGED, memory was never wrong and the fault is in register delivery.

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
