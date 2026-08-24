# Plan — the SQLite `sqlite3WhereCodeOneLoopStart` NOT_CAP wedge

**Status: NOT root-caused by this investigation.** A tight localization, a large set of
controlled exclusions, and a strong candidate mechanism that is *already documented elsewhere*
and does not cleanly fit.

## 1. What is ESTABLISHED (each by its own control, on silicon)

| Fact | How |
|---|---|
| Faults at `sqlite3WhereCodeOneLoopStart+0x8c`, `cincoffsetimm a4,a4,0xb0` right after `ldc a4,0x0(a0)` | latched `mepc` mapped through per-arm `DBAS`, disassembled in 3 binaries, 7 boots |
| The operand's `cap_type == NOT_CAP` | `mcause 25` from the FLU; `cincoffsetimm` is the IMMEDIATE form whose guard is a single `NOT_CAP` test, verified by disassembly |
| Producer is the FLU, **not** commit_stage | `tval != mepc` across 5 boots; the three log registers latch from one event (`cva6.sv:1126-1136`) |
| Writeback-port displacement EXCLUDED | switch 204 = `0x00` halted, with the 220 selftest firing in-boot; 7 boots. Ports 0/4 carry cap data, port 3 is FPU-only, ports 1/2 watched |
| Arm POSITION is not the variable | un-probed wedges at arm 2 AND arm 3; probed completes at both |
| The INPUT is a variable | `q_one` completes, `q_two` wedges — same binary, same arm |
| Instrumentation removes the fault | probed completes, un-probed wedges — same input, same arm |
| **Delay alone is NOT the variable** | `loop3` (~10 dynamic instrs) completes where `pad10` (~10) wedges; a 66x delay sweep with alignment pinned shows no change |
| Reproducer is 588 bytes | `SELECT t1.a FROM t1, t1 AS y` vs `SELECT t1.a FROM t1`, empty table |

## 2. What is EXCLUDED

VDBE execution; setup statements; data volume; `ORDER BY`; rev-node exhaustion (head 630 of
65532); codegen live-range overlap; S-06; the AMO residual; miscompiled bounds check (the LDC
guard wraps every relational); writeback-port displacement.

## 3. What is DISFAVOURED but NOT excluded

* **Spill-side breakage** — the total type query reads healthy right after the spill, but only in
  runs the probe perturbed into completing. Testable via the last-wins STC recorder or a memory
  marker (below).
* **Delay/drain latency** — survives as an observation, not as a cause.

## 4. THE LIKELY ANSWER, and why it is not yet the answer

**S-07 is already root-caused** (`fpga-repros/S07-capability-untagged-on-reload/00-README.md:54`):
the write buffer hits at 64-bit WORD granularity so a granule's halves take separate entries,
each entry writes the WHOLE granule's tag on drain, and drain order is `rr_arb_tree`, not program
order. An older plain store to `G+8` drains after a younger `stc` to `G+0` and clears its tag.
Arms: `wb1` (plain `G+8`; `stc G`) 1107/16384 lost; `wb3` (same + 64 draining stores) 0.

**`wb1` vs `wb3` IS my `pad10` vs `pad600`.** Same mechanism shape, rediscovered expensively.

**Why it does not close:** S-07 requires a plain store *into the subject granule*. Mine has none.
Subject granule `[s0-0x70, s0-0x60)`; the nine stores in the window are at `s0-0x74`, `-0x5d0`,
`-0x5b0`, `-0x90`, `-0x98`, `-0x5a0`, `-0x10c`, `-0x110`, `-0x120`. Sites differ too: S-07 faults
in `sqlite3OsRead` at `0x2a83c` from a `sqlite3JournalOpen` memset.

## 5. PLAN, in dependency order

**P0 — settle identity before spending anything.** (in flight)
* Auditor attacking my granule arithmetic and store enumeration, including **stores made by any
  callee invoked between spill and reload**, which I never considered.
* RTL lane: is the S-07 fix in `caplifive_s10fix_80843404c.bit`? Can the reorder clear a granule's
  tag *without* a same-granule plain store (adjacent granules, `wr_idx` aliasing)? Is
  `sqlite3WhereCodeOneLoopStart` a known second site?
* **If it is S-07 or a known residual (S-09/S-10): STOP.** Record the new site in that folder,
  close this, and hand the reproducer over. Do not open a new issue.

**P1 — if distinct, get the spill-side fork in a FAULTING run.** Needs no new RTL.
* The STC recorder is **last-wins** and survives a wedge. Compare its `paddr` against the subject
  slot (now reported as `slotaddr` from a completing arm). On any wedge where they match,
  `stc_ctag` answers spill-vs-reload directly, unperturbed.
* If it never matches, use a **memory marker**: total `lcc` on the register being spilled, `sd`
  the answer into the shared region, read it over JTAG while halted. Memory survives a wedge —
  only reporting dies. **Control required:** a marker-only arm that must still wedge.
* Use a **light** probe: query the register about to be spilled. The current width probe forces an
  extra `ldc`, the operation under suspicion, which is likely why it removes the fault.

**P2 — placement sweep, expected to be a proxy.** Pads of 0/1/2/3/4 nops shift the faulting
instruction 4 bytes each at near-constant delay. A tight boundary is a real constraint on any
mechanism even when it is not the mechanism. Record as proxy, not cause.

**P3 — blocked on the project lead, surfaced not lobbied.** The s07 LDC recorder is one-shot with
no clear and is consumed by boot software before any domain runs, so it is unusable here. A clear
mirroring `dom_switch_log_clear` is purely additive and would make it work. Competes for a
synthesis slot with the latent dom-switcher defect; that priority is not mine.

## 6. Do NOT do

* Do not open an `fpga-repros` folder until P0 says this is distinct. One issue per folder, and a
  duplicate of S-07 would be worse than nothing.
* Do not build more sim arms of the delay/contention shape — the peer lane measured peak
  write-buffer occupancy at **1** against depth 8, so that environment cannot hold the condition.
* Do not read switch 208 without the granule-address attribution check; bit 7 alone passes a false
  positive.

---

# UPDATE — RTL lane answers. Identity largely settled; two of my findings weakened.

## P0 answers

* **The S-07 fix IS in the flown bitstream, and so is S-10.** `git log 80843404c` carries
  `5c5f4e3a7 S-07 FIX: forbid granule co-residency in the write buffer` and `4fee13b2d S-10 FIX`
  under `3d3ed1502 Merge S-10 fix for synthesis validation`. **So this wedges WITH both fixes
  present**, on an image that is not `fpga-testing-dev` head.
* **My different-granule exclusion is SOUND** — two independent reasons. The fix's condition
  requires same granule (`gran_conflict[k] = valid[k] & gran_eq[k] & word_ne[k] & (...)`), so a
  different-granule store never enters the co-residency path. And the tag write is **way**-
  qualified, not merely index-qualified: `wt_dcache_mem.sv:520` sits inside
  `if (wr_req_i[j] & wr_ack_o)`. On aliasing: `wr_idx = paddr[11:4]`, so collision needs
  congruence mod 4096, and my nine stores span ~1376 bytes — impossible among them.
  **The only remaining hole is the callee question**, which the auditor is on.
* **FOUR sites are already recorded**, and one is in my source file: `sqlite3OsRead+0x4c`,
  `pagerFreeMapHdrs+0x4c`, `sqlite3BackupRestart+0x5c`, and **`whereLoopOutputAdjust+0x200`,
  which the folder calls "the purest instance"** — `where.c`, same file as
  `sqlite3WhereCodeOneLoopStart`. **My site is most likely a FIFTH instance of a known family.**

## Two of my recorded findings are now WEAKER

**1. "The probe is a variable" — weakened, possibly void.** Folder section 3 measured three
instrumented builds:

    S7C  instrumented sqlite3OsRead     -> wedged at pagerFreeMapHdrs+0x4c
    S7P  instrumented pagerFreeMapHdrs  -> wedged at sqlite3OsRead+0x4c
    S7B  instrumented BOTH              -> wedged at sqlite3BackupRestart+0x5c

> *"Each build dies at whichever vulnerable `ldc` pair the previous one had covered. **A software
> probe can therefore never be the thing that fires** — the uncovered site always kills the run
> first."*

So `sqrtw` completing is **equally consistent with the death having moved to a site that did not
fire in that run** as with the probe fixing anything. It did complete with fully correct output,
which is not the same as S7C/S7P/S7B (they wedged elsewhere) — but that distinction is not
strong enough to carry the claim.

**2. The N is too small on the completing side, because the phenomenon is SPORADIC.** The folder
measures **p(wedge) = 25% per execution**, with the site fixed per image and only the firing
sporadic. My un-probed arm wedged **7/7**, which is far more deterministic than 25% and suggests
my site's rate is high. But `sqrtw` completed **2/2**, and at p=25% two consecutive passes occur
56% of the time. **Two completions is weak evidence of "fixed".** Any future claim that an arm
"does not wedge" needs repetitions, not one boot.

## Revised first question, per the RTL lane: what is NEW against the folder?

Provisional, pending the auditor:

* the **588-byte reproducer** (`q_one` vs `q_two`, empty table) — far smaller than opening a DB;
* the **arm-position exclusion** (wedges at arm 2 AND arm 3; completes at both);
* the **pad ladder** with a constant-code-size loop instrument, verified by instruction address;
* the **s07 LDC-recorder instrument defect** — one-shot, no clear, consumed by boot software
  before any domain runs, proven by a pre-run `0xb8` baseline;
* the **liveness-vs-attribution distinction** that fell out of it;
* the **producer discrimination** (`tval != mepc` excludes commit_stage) and the finding that the
  FLU tval path has no fired positive control;
* the **DRAM-over-JTAG escape** — see below.

## The one escape the folder does NOT close

Section 3 concludes *"a wedging run has no reporting channel at all"*. That is true **as written**,
because it concerns `output_text` and the retval, both of which need the domain to RETURN. It is
**not** true of DRAM read over JTAG while halted, which is not a software channel and is not
addressed there. **That remains the only way to get an observation out of a faulting run — and
the folder's own conclusion that no software probe can fire is precisely why it is worth doing.**

## Also pre-empted: the placement sweep

Folder section 2 **withdraws** "the site wanders": five of six recoverable wedges were
`sqlite3OsRead+0x4c` in five different builds with five different link addresses, and reading
that as wandering was an error that an auditor also missed. My case differs — same build,
different pads — so the sweep is not simply re-treading it, **but the failure mode is identical
in shape: treating a link-address difference as an address effect.** Design the arms against that
section, not against my own table.

---

# UPDATE 2 — auditor verdict. One row of mine downgraded, one new leading candidate.

## Strengthened

* **The granule arithmetic is now PROVEN, not assumed.** `stc` alignment is hardware-enforced:
  `capstone_dyn_unit.anvil:418` raises `STORE_ADDRESS_MISALIGNED` on `(rs1 & 15) != 0`. The
  subject `stc` executes without it, so `s0 ≡ 0 (mod 16)` and the granule is `[s0-0x70, s0-0x60)`.
  It matters because the granule is **sandwiched**: `sw` at `s0-0x74` sits immediately below and
  `stc` at `s0-0x60` immediately above. A one-unit boundary error either way would have put a
  store inside.
* **The callee gap is CLOSED.** No `jalr`/`jal` between the subject `stc` and the reload (first
  call is after the fault); 1210 branch targets extracted from the whole function, **zero** land
  in the window. Parser positive-controlled before the empty result was believed.

## REFUTED — my stated reason for distinctness

"A plain store into the granule is S-07's **required** trigger" is **wrong**. It is demonstrated
only for the `sqlite3JournalOpen` -> `sqlite3OsRead` pathway. For `pagerFreeMapHdrs`,
`sqlite3BackupRestart` and `whereLoopOutputAdjust` the folder never enumerates one. **The
folder's actual invariant is a DATAFLOW SHAPE**, `00-README.md:433-435`: the value produced by
the preceding `ldc` arrived NOT_CAP — "the back-to-back dependent capability-load pair".

So "no same-granule plain store" separates this from the **memset pathway**, not from S-07. It
does not distinguish us from three of S-07's own four recorded sites.

## DOWNGRADED — a contradiction in this plan's own ESTABLISHED table

The table listed `cap_type == NOT_CAP` and "producer is the FLU" as established, while the same
document records that the **FLU tval path has never been shown to fire**. Both rows lean on that.

Partly defensible and partly not, precisely:

* **"Not commit_stage" is reasonable.** `commit_stage.sv:604` sets `tval = commit_instr.pc`, the
  latch demonstrably records a NON-zero tval (mcause 15 case), and `tval != mepc` therefore does
  argue against that producer.
* **"The value was NULL" is NOT established**, and neither is "a de-tagged capability". Those
  remain undiscriminated, because separating them needs the FLU tval assignment to be live.

**So `cap_type == NOT_CAP` moves from ESTABLISHED to WELL-SUPPORTED**: it follows from
`mcause 25` at an unambiguous `cincoffsetimm` *given* the FLU producer, and the producer
determination rests on `tval != mepc` rather than on a fired instrument.

## CORRECTION to the auditor

The report states the arm-position confound is "unbroken, 5 for 5". **It was broken.** Boot #26:
`[stages] <-- TEST 2/3 .../sqrt.dom:--slt .../q_two.test NO RETURN within 400s` — the un-probed
build wedging at **arm 2**. The auditor read the retraction section without the later refutation.
Position is not the variable.

## NEW LEADING CANDIDATE — the S-10 fix that is IN this bitstream

The 62-line delta between the S-07-fix-validated build and the flown `80843404c` is **exactly the
S-10 fix in `wt_dcache_mem.sv`**, and it is a **read-path** change that existed in no earlier
bitstream:

    assign wbuffer_gran_clr_oh[k] = wbuffer_gran_oh[k] & ~wbuffer_data_i[k].ctag;
    // S-10: the granule-scoped clear is applied to BOTH legs, and it DOMINATES.

Its own commit subject is **"S-10 FIX: works in simulation, and costs a combinational loop — NOT
ready to merge"**, its comment records `UNOPTFLAT 39 -> 40 on wt_dcache.rd_ctag`, no later commit
in range resolves the loop, and the bitstream has **never closed timing** (WNS −10.629 ns).

**A dominating tag-clearing override, carrying a known combinational loop, on a timing-failing
bitstream, produces EXACTLY the observed polarity — data correct, tag zero — if it ever fires
spuriously.** And S-10b (`c867dfcbb`) is *not* in this bitstream.

This displaces granule co-residency, which is fixed here and micro-arm validated
(`wb1` 1107 -> 0, and `wt_dcache_wbuffer.sv` byte-identical between the validated and flown
builds).

## The defensible sentence today

> A reproducible mcause-25 wedge at `sqlite3WhereCodeOneLoopStart+0x8c` on
> `caplifive_s10fix_80843404c.bit`. S-07's granule-co-residency mechanism is excluded at this slot
> on two independent grounds. Whether a tag was lost at all is undetermined: the FLU `tval`
> instrument has never been shown to fire, and `mcause 25` has two producers on this bitstream.

## Revised next steps

1. **Fire the tval instrument** — `li a0,0xBEEF ; cincoffsetimm a0,a0,8` must trap mcause 25 with
   `tval == 0xBEEF` AND `mepc` at that instruction. Until it fires, "lost tag" and "NULL" are
   both void. **This is now step 1, ahead of everything else.**
2. **Repetitions, not single boots.** p(wedge) is 25% at the measured S-07 site; two clean runs
   prove nothing.
3. Only if (1) fires non-zero does "a capability lost its tag" become a fact — and then the target
   is the **read-side** S-10 override above, not co-residency.

## Still do NOT

Open a folder. P3 is unproven — a folder asserting a distinct silicon defect would be a claim
ahead of its evidence. Equally, do **not** file it as a fifth S-07 instance: the mechanism is
excluded at this slot, and the family-signature match is a shape, not an attribution.

---

# UPDATE 3 — the candidate sharpens; two of my numbers corrected; the A/B image EXISTS

## Corrections to what this plan recorded

* **The combinational-loop premise is WRONG.** `UNOPTFLAT 39 -> 40` belongs to a **rejected**
  variant, not the shipped fix: `wt_dcache_mem.sv:390-399` records that adding a fourth
  `rd_ctag_src_o` code cost the loop, *"so the fix changes `rd_ctag_o` only"*. The commit subject
  I quoted (`4fee13b2d`, "NOT ready to merge") predates that resolution. Empirically the bitstream
  **exists**, so `DRC LUTLP-1` passed — with S-10b as positive control that the check fires, since
  its real loop did block bitgen.
* **The WNS figure was the wrong build's.** `-10.629 ns` is the **CONTROL** (`39b21639d`, 96727
  failing endpoints). The **FLOWN** image `80843404c` is **worse: -16.400 ns**, 102769 endpoints.
* **The S-10 cone is NOT the failing cone.** In the flown build's 2.9 MB forensics `rd_ctag`,
  `gran_clr` and `wbuffer_gran` appear **zero times**, against 26 mentions of `wt_dcache_mem` — so
  the module is covered and the fix's nets are simply not on failing paths. All 102769 failing
  endpoints launch from one register, `dom_switcher/cur_idx_q_reg[3]`, which transitions only on
  a domain switch.

## The candidate, restated in the form that can be attacked

S-10 replaces the tag read with a **bare OR** whose soundness is **conditional on S-07's stall
being complete**. Its own comment:

> *"A plain-store entry and a capability entry can no longer co-reside in one granule: the S-07
> fix refuses to allocate the second one... Without the S-07 stall this reduction would be wrong:
> a plain store and a later `stc` could both be resident, and 'any ctag=0 clears' would discard a
> tag the program had just written."*

**If ANY path lets a plain entry and a capability entry co-reside in one granule** — an allocation
route bypassing `gran_hazard`, a merge rather than an allocate, an entry re-validated mid-drain —
**S-10 discards a tag the program just wrote.** That is exactly the observed polarity (data
correct, tag zero), needs no loop and no timing violation, and exists **only in this bitstream**.

## What my code analysis says about that mechanism — and the one route it cannot cover

Within `sqlite3WhereCodeOneLoopStart`, checked in the artifact:

* **After** the spill: nine stores, none in `[s0-0x70, s0-0x60)`.
* **Before** the spill: the only PLAIN store is `sd ra, 0x7e0(sp)` -> **`s0-0x10`**; every other
  pre-spill store is an `stc`. Nothing plain touches the subject granule.
* The **previous invocation** of this same function uses the SAME frame, and its only plain store
  near the slot is `sw` at `s0-0x74`, granule `[s0-0x80, s0-0x70)` — adjacent, not the subject's.

**NOT COVERED, and it is the route the mechanism needs:** a plain store to that physical address
from **any earlier function at the same stack depth**, still resident in the buffer. That is not
statically enumerable from one function, and my exclusion should never have been stated without
this boundary.

## The A/B EXISTS, and its caveat

`/tmp/capstone/_bitstream/synth-39b21639d-exit0.tar.gz` -> `work-fpga/ariane_xilinx.bit`.
`39b21639d` has the **S-07 fix PRESENT and the S-10 fix ABSENT** — the exact single-variable image.

**Caveat that must be reported with any result:** both images fail timing, so the comparison is
not confounded by one closing and one not — but they fail by **different margins**, `-10.629` vs
`-16.400`. A difference in outcome could be **5.8 ns of margin rather than S-10's logic.** It is
much the cleanest experiment available and it is still not single-variable.

**A reflash is the project lead's call. Surfaced, not assumed.**

---

# UPDATE 4 — both named mechanisms require a store this window does not contain

## The RTL lane closed both routes I proposed, and weakened its own hypothesis

* **Draining entries are covered.** `wt_dcache_wbuffer.sv:621-623`: an entry keeps `valid` while
  `txblock` is set, and `gran_conflict` consults `valid[k]`, so a mid-drain entry still blocks.
* **The merge route is deliberate and architecturally correct**, not a bypass (`:625-628`):
  same-word requests take `wr_ptr = hit_ptr` and merge, where the S-06 P4 rules (sticky `is_cap`,
  last-writer-wins `ctag`) give the right answer — a plain store overwriting part of a granule
  **should** clear its tag.

**So co-residency looks genuinely blocked, and the "S-10 is sound only if the S-07 stall is
complete" mechanism REQUIRES co-residency. That hypothesis is weaker, not stronger** — surfaced
by its own author rather than defended.

## A real documented window with the exact polarity

`wt_dcache_wbuffer.sv:612-619` — the **issue/return desync**, deliberately permitted:

> *"a same-word merge into an entry whose transaction has already issued (`.ctag` written
> unconditionally, no txblock guard, and `wr_ptr = hit_ptr` keeps hitting the entry because
> `valid` survives txblock). **`stc G` then `sd x0, G+0` gives DRAM ctag=1 and L1 ctag=0.**
> It CONVERGES ... so it is a bounded window, and it is PRE-EXISTING rather than introduced here."*

**L1 reads tag 0 with correct data, transiently.** Correct polarity, and *intermittent by
construction*, which fits a ~25% rate far better than anything deterministic.

## But it needs a store this window does not have either

The desync requires a **plain store to the subject WORD after the `stc`**. Enumerated in the
artifact: the nine stores between the subject `stc` and the reload are at `s0-0x74`, `-0x5d0`,
`-0x5b0`, `-0x90`, `-0x98`, `-0x5a0`, `-0x10c`, `-0x110`, `-0x120`. **None is in
`[s0-0x70, s0-0x68)`**, and there is no call in the window through which a callee could add one.

**So BOTH named mechanisms — S-10-conditional co-residency and the pre-existing desync — require
a plain store into the subject granule or word that this window does not contain.** Either the
mechanism is a third thing, or the store comes from outside this function's control flow: the
trap handler, the monitor, or a domain switch. That last class is the one route neither of us can
enumerate statically, and it is now the common requirement of both candidates.

## The A/B is now DISCRIMINATING, which it was not before

| hypothesis | prediction on `39b21639d` (S-07 fix present, S-10 absent) |
|---|---|
| S-10 fix implicated | fault **ABSENT** |
| pre-existing issue/return desync | fault **PRESENT** |

Two named hypotheses with **opposite** predictions, so either result is informative. Still not
single-variable (`-10.629` vs `-16.400` WNS), but a **present/absent** split is much harder to
explain by 5.8 ns of margin than a rate change would be.

**Run it enough times to see the RATE, not just the outcome.** At ~25% a single non-wedge on the
control image is not absence.

Note also: **"only in this bitstream" is not something this investigation established.** No
earlier bitstream has been tested with this reproducer. It is a property of the S-10 candidate as
argued, not a measurement.

## The general rule, since it caught three things tonight

`pgrep -f` matching its own shell; a file-size check that could not separate two instruments; a
grep taking `eth_rxck` instead of the CPU clock. All three are **"the query returned something,
so it answered the question."**

The defence that actually worked, three times, is **a second reading that has to agree**: the
selftest against the wedge read, the granule address against the valid bit, and `-16.400` against
`+4.907`. **None was caught by making the first query more careful.**

---

# UPDATE 5 — THE FAULT IS SPORADIC. Several pad-ladder conclusions rest on N=1.

Boot #32, control passed, and **`sqpad10` COMPLETED** — the same binary that wedged in boots #29
and #31.

    sqpad10   boot 29  WEDGED
    sqpad10   boot 31  WEDGED
    sqpad10   boot 32  COMPLETED

**Two wedges in three runs.** The fault is sporadic at this site, not deterministic, which the
S-07 folder already measured for its own site (p ≈ 25%) and which this investigation had not
applied to its own arms.

## What that costs

**Every "this arm completes" result in the pad ladder is N=1 or N=2**, and a completion is now
known not to mean absence:

| arm | observations | claim it supported |
|---|---|---|
| `pad600` | 1 completion | "600 nops fix it" -> **unsupported at N=1** |
| `loop3` | 1 completion | "the loop pad never wedges" -> **unsupported at N=1** |
| `loop200` | 1 completion | same |
| `sqrtw` | 3 completions | "the probe removes the fault" -> weak, and already qualified |

So **"delay is the variable" and "the probe is a variable" are both further weakened**, and the
`loop3`-vs-`pad10` pair that killed the delay reading is itself two single observations. The
direction of the evidence has not changed, but its weight has.

**The un-probed `sqrt` arm remains the strongest single fact**: wedged in boots 21, 22, 24, 25 and
26 — 5 for 5 — which at p=0.25 would be p≈0.001. Its rate is genuinely high. Everything else needs
repetitions.

## Method change, applying from here

**No single-boot arm may support a conclusion.** Minimum 5 repetitions for a "completes" claim,
and outcomes reported as rates, not verdicts. This is the peer lane's warning applied to my own
ladder rather than only to theirs.

## Shadow-tag formula CORRECTED before it was used

`ariane_pkg.sv:592` comments the tag address as `CAP_TAG_MEM_BASE + (data_paddr >> 4)`. Taken
absolutely that puts tags **above DRAM top** and would have read garbage. It is **relative to
`MEMORY_BASE`**, and the arithmetic proves it:

    MEMORY_BASE 0x8000_0000,  MEMORY_TOP == CAP_TAG_MEM_BASE == 0xBC2D_2D2D
    0xBC2D2D2D + ((MEMORY_TOP - MEMORY_BASE) >> 4) = 0xBFEF_FFFF
                                                   = exactly CAP_REVNODE_MEM_BASE (0xBFF0_0000)

The shadow region fits its address space **exactly** under the relative form and overflows under
the absolute one. Driver now uses `CAP_TAG_MEM_BASE + ((paddr - MEMORY_BASE) >> 4)` and **refuses**
any address outside `[0xBC2D2D2D, 0xBFF00000)` rather than reading it.

Measured addresses, from boot #32 (`slotaddr = 0x82f91430` at arm 2, `DBAS 0x82C00000`, so the
in-domain offset is `0x391430`):

    arm 2   granule 0x82f91430   tag byte 0xbc5cbe70
    arm 3   granule 0x83391430   tag byte 0xbc60be70

---

# UPDATE 6 — METHOD CHANGED, and the first COUNTED result: the window alone is NOT sufficient

## How S-07 was actually cracked, from the commit history

    25b964349b05  granule apertures read HALTED at the wedge -> two granules exactly adjacent
    b8824114deb8  wbuf: a DIRECTED SILICON TEST, with a control that must fail
    d644cd273039  ROOT CAUSE CONFIRMED ON SILICON

**The decisive instrument was a counting microbenchmark, not an instrumented SQLite.** `wbuf`
runs `WBUF_N 256 x WBUF_REPS 64 = 16384` trials per arm and returns
`0xB0000000 | (corrupt<<12) | lost`, 12-bit counters saturating at 0xFFF. That is a **rate per
boot**. This investigation had been harvesting **one bit per boot** against a sporadic fault,
which is exactly why its ladder kept producing N=1 conclusions.

## Result, boot with k800 control passing (oracle 4)

| rung | retval | control bit | lost | meaning |
|---|---|---|---|---|
| k800 | 4 | — | — | boot VALID |
| wb0 | 0xB0000000 | — | 0 | baseline, no plain store at all |
| wb10 | **0xB1000000** | **1** | **0** | 4 granules in flight |
| wb9 | **0xB1000000** | **1** | **0** | 10 granules in flight (OVER depth 8) |

**16384 trials each, ZERO losses, with the detector PROVEN to fire.** So this is a meaningful
zero, not an unproven one.

### What that eliminates

* **Buffer pressure / granule count alone.** 10 distinct granules against `WtDcacheWbufDepth = 8`
  produced zero loss. The over-by-one reading is dead as a sufficient cause.
* **The two `ctag=0` capability stores.** Present in both arms, zero loss.
* **The bare window shape** — subject `stc`, stores to OTHER granules, reload.

**So the SQLite fault needs something the instruction window does not contain**, which is the
same conclusion the static analysis reached from the other side: both named mechanisms required a
plain store into the subject granule or word that the code demonstrably lacks.

## The in-arm positive control, and why it was necessary

`wb2` — the harness's own positive control — **cannot be QEMU-verified, structurally**: the
native oracle is a stub that always returns `0xB0000000` (`wbuf_host.c:4`), and arm 2's entire
purpose is a divergence only capability silicon exhibits. That is the C13 blocker, and it is the
project lead's call; the gate was left alone.

So arms 9/10/11 carry their own control in **bit 24**, set only when the type query answers 7 for
a known non-capability. Native, QEMU and silicon must all produce it.

**It caught its own first version.** The control initially asked `wbuf_type(&ctl_scalar)` — the
address of a local, which IS a real capability here — so it returned that capability's type
rather than 7, and would have reported "detector dead" on healthy hardware. Fixed to query the
integer VALUE.

## k800 unblocked as a side effect

The preflight blocked the run because no first rung was a verified known-good control. `k800` had
an oracle but **no QEMU pass**. Verified it: `oracle: k800 = 4`, `qemu: PASS`. That was one of the
two decisions the RTL lane was blocked on, and it turned out to be a missing verification rather
than a real failure.

## Next arm: the LOAD-to-STORE-to-LOAD chain (arm 11, QEMU-verified)

The one structural difference between the microbenchmark and the SQLite window:

    wbuf 9/10   register -> stc -> ldc      (`base` stays in a register)
    SQLite      ldc -> stc -> ldc           (pWInfo is RELOADED from the caller's frame, passed
                                             in a2, spilled by the callee, reloaded again)

The capability being spilled is itself the result of a recent load. If a forwarding path carries
a load's result into a store without carrying its tag, arm 11 sees it and arms 9/10 structurally
cannot.

## The counted exclusion set — five arms, 16384 trials each, ALL ZERO with fired controls

Every row ran on a boot whose `k800` control returned its oracle of 4, and every `wb9`-family arm
carries an in-arm positive control (bit 24) proving the detector answers 7 for a non-capability
in the SAME run. So these are **meaningful zeros**.

| arm | what it adds to the window | retval | lost |
|---|---|---|---|
| wb0 | baseline, no plain store at all | `0xB0000000` | 0 |
| wb10 | 4 distinct granules in flight | `0xB1000000` | 0 |
| wb9 | 10 granules (OVER depth 8) + two `ctag=0` `stc` | `0xB1000000` | 0 |
| wb11 | `ldc` -> `stc` -> `ldc` chain (spilled value is itself a load result) | `0xB1000000` | 0 |
| wb12 | subject slot on the monitor-carved STACK, not a global | `0xB1000000` | 0 |
| wb13 | subject `shrink`-DERIVED: narrowed bounds, derived revnode, **type 1 = NONLIN** | `0xB3000000` | 0 |

**Excluded as SUFFICIENT causes:** granule count and write-buffer pressure; the `ctag=0` entry
class; the bare window shape; the load-to-store-to-load chain; the stack region; and the
subject's own derivation and metadata class.

`wb13` also confirms the arms test the RIGHT capability class: its subject reads **type 1
(NONLIN)**, the same as the measured healthy baseline of the SQLite capability that faults. Had
it read LINEAR, every arm above would have been testing a different class.

## Revocation-tree depth: DEAD, retired by RTL reading rather than a boot

The rev-node head reads 630 at the SQLite wedge against a handful in wbuf, which looked like the
one state variable differing by orders of magnitude. It cannot be the mechanism
(`capstone_dyn_unit.anvil:357-362`):

* the node a plain `ldc` consults is **`rs1`'s — the AUTHORITY capability**, not the loaded
  value's, so deep revocation state belonging to the spilled capability is never queried;
* failure raises **`INVALID_CAPABILITY` = mcause 26 AT THE `ldc`**, whereas the observed wedge is
  **mcause 25 at a later `cincoffsetimm`** with the `ldc` having completed;
* `CINCOFFSETIMM` has no revnode check at all — only NOT_CAP -> 25 and UNINIT/SEALED -> 27.

So no instruction in the window can turn revocation state into a NOT_CAP operand.

## Where this leaves it

**The trigger is not reproducible from the instruction window, the memory region, or the
subject's metadata.** What remains untested is SCALE: SQLite runs ~10^5 instructions with a
working set that thrashes the caches, takes timer interrupts, and touches a specific physical
set, where wbuf's loop is ~4 KiB and stays resident.

**The honest status: the root cause is NOT found.** What exists is a sharply bounded search
space, a 588-byte reproducer, and a counted instrument that eliminates a family per boot instead
of producing one bit.

### Two harness rules learned, both worth carrying into any future arm

* **Every arm must leave a valid capability in `wbuf_slots[i]`.** The shared check runs FIELD
  queries, and every selector except 1 RAISES on a NOT_CAP, so an arm that writes only its own
  slot traps and produces no retval.
* **A stack local holding a capability needs explicit `aligned(16)`.** Without it the `stc`
  raises `STORE_ADDRESS_MISALIGNED` and the domain produces **NO OUTPUT AT ALL** — indis-
  tinguishable from a hang, and it cost two QEMU cycles to find.

---

# A/B RESULT: the S-10 fix is EXONERATED. The fault predates it.

Reflashed to `caplifive_s07only_39b21639d.bit` — **S-07 fix PRESENT, S-10 fix ABSENT**, ancestry
verified both directions (`5c5f4e3a7` IS an ancestor, `4fee13b2d` is NOT), sha `f5af588fc676cd9a`,
from the exit-0 archive. Resident name RE-READ after a power cycle rather than trusted from the
flash call, and independently re-checked by the driver's own preflight.

Everything else held constant: same binary `ee9a9a86ed12f06b`, same arms, same firmware image,
same control.

| bitstream | S-10 fix | control | `sqrt.dom --slt q_two.test` |
|---|---|---|---|
| `caplifive_s10fix_80843404c.bit` | present | 7 s OK | **WEDGE** |
| `caplifive_s07only_39b21639d.bit` | **absent** | 7 s OK | **WEDGE** |

Same fault, not a different one:

    mcause  0x99 -> 25                                   identical
    mepc    0x828f4ba0, DBAS 0x82800000 -> VA 0x104ba0
            = sqlite3WhereCodeOneLoopStart + 0x8c        identical source line
    sw=204  0x00 with SELFTEST PASS                      displacement still excluded

**N=1 is SUFFICIENT in this direction, and the asymmetry is the point.** Proving *absence* needs
many repetitions — a single non-wedge at ~25% is not absence. Proving *presence* needs exactly
one: a fault that does not exist cannot occur by chance. So one wedge settles it.

### What this eliminates

* **The S-10 read-path override is NOT the cause.** It was the last named candidate and the only
  mechanism that existed uniquely in the flown bitstream. Gone.
* **"Only in this bitstream" is dead as a property** — it was always an argued claim rather than
  a measurement, and this is the measurement that refutes it.

### What it promotes

**The pre-existing issue/return desync** (`wt_dcache_wbuffer.sv:612-619`) — a same-word merge into
an entry whose transaction has already issued, giving DRAM `ctag=1` and L1 `ctag=0`. It is
deliberately permitted, bounded, converging, and **pre-existing**, which is exactly what a fault
present in both bitstreams requires. It remains the case that it needs a plain store into the
subject word that this window does not contain — so it is promoted, not proven.

### Caveat, stated with the result

The two images fail timing by different margins (**−10.629 vs −16.400 ns WNS**), so the
comparison is not strictly single-variable. It matters far less for this outcome than it would
have for the other: a **present/absent** split could have been argued as 5.8 ns of margin, but
**present on both** cannot — the fault occurring on the image with the *better* margin is not
explicable by margin.

---

# IDENTITY SETTLED: this IS S-07 — the instance its own root cause does NOT explain

A documentation sweep found the match, and it was in the S-07 folder the whole time.

**`S07-capability-untagged-on-reload/00-README.md:1040-1043`, "instance 1" (`memcpy`):**

> *"every instruction touching the faulting granule is `stc`, one plain `ld` and three `ldc` --
> **zero plain stores**. Neither correct tag-clearing on a partial overwrite, nor the write-buffer
> `.user` clobber ... which needs a coalescing plain STORE to the same word."*

Independently in `ref/ISSUES.md:914-923`: mcause 25 at `memcpy+0x2a8`, operand reloaded by
`ldc a2,0x0(a2)` **from the stack slot at `s0-0x60`**, and "**Nothing writes the granule.**"

    memcpy+0x2a8                    sqlite3WhereCodeOneLoopStart+0x8c   (ours)
      cincoffsetimm a2, s0, -0x60     cincoffsetimm a0, s0, -0x70
      ldc           a2, 0x0(a2)       ldc           a4, 0x0(a0)
      cincoffset    a1, a2, a1        cincoffsetimm a4, a4, 0xb0
      -> mcause 25                    -> mcause 25

**Same shape: stack slot, reload, dependent `cincoffset`, mcause 25, no plain store in the
granule.** S-07 generalises it at `00-README.md:565-570` across three independent builds: *"a
capability is loaded, spilled to a stack slot, reloaded, and the immediately dependent `ldc`
raises mcause 25."* Sporadicity matches too (`ISSUES.md:214-217`: the same image passed and wedged
eight minutes apart).

**Why this resolves the identity question that has been open all session.** S-07's root cause --
write-buffer granule co-residency -- **requires a same-granule plain store**, which instance 1
does not have and neither do we. So our fault is not a *counterexample* to S-07; it is the
**already-documented residual** that the S-07 fix could not have cured. That is exactly consistent
with it reproducing on BOTH `caplifive_s10fix_80843404c.bit` and `caplifive_s07only_39b21639d.bit`
with `5c5f4e3a7` present in each.

**Instance 1 was never revisited after the 2026-08-19 root cause** -- no document after that date
mentions it. The residual is documented and never closed.

## What we contribute back to S-07

Its one un-refuted mechanism is the **CAP_WB / LOAD_WB register-delivery path**
(`00-README.md:1104-1113`): an `ldc` response bypassed to LOAD_WB has `cap_result` erased by
`scoreboard.sv:242-246`, giving *"a NOT_CAP register with a correct cursor, having never touched
memory"*. The folder argues it cannot happen, but by **source reading**, and lists it under "what
would settle it".

**Our switch-204 result is the first empirical answer**: `0x00` at every wedge across 8 boots with
the 220 selftest firing in the same boot -- a controlled negative, on both bitstreams. That
narrows S-07's own open list.

## Consequence for where this goes

Per this plan's own P0 rule -- *"if it is S-07 or a known residual: STOP, record the new site in
that folder, do not open a new issue"* -- **the finding belongs in the S-07 folder as a second
site for instance 1's residual**, not in a new `fpga-repros` folder. Still true that no tag has
been shown lost (the FLU `tval` path has never fired), so the site is recorded as a reproduction,
not as a proven mechanism.

---

# THE tval INSTRUMENT FIRES. Every prior `tval == 0` is REAL DATA, and the reading changes.

    [wedge] trap mepc = 0x00000000828233ac   -> DBAS 0x82800000 -> VA 0x333ac
    [wedge] trap tval = 0x000000000000beef

VA `0x333ac` disassembles to **`cincoffsetimm a0, a0, 0x8`** with `a0` holding `0xBEEF` — the
control's own instruction, not an inherited latch. `ex_stage.sv:487` puts the rs1 **cursor** in
`tval` for capability causes, and it did.

**So the FLU `tval` path WORKS on this silicon.** The retraction that called every `tval == 0`
"NO DATA" was correct *at the time* — the instrument genuinely had no fired control — but the
control has now fired, and those zeros are evidence.

## What that establishes, and what it does NOT

`tval` carries the rs1 **cursor**. At every SQLite wedge it read **0**. Therefore:

* **A DE-TAGGED CAPABILITY IS EXCLUDED.** A capability that lost only its tag keeps its address
  bits, so its cursor would be pointer-like and `tval` would be non-zero. It was zero.
* **The operand's VALUE was genuinely zero**, not a pointer stripped of authority.

**This points AWAY from tag loss and TOWARD zero DATA.** Which matters, because the whole S-07
family — including the instance-1 residual this site was just filed under — is a **tag**-loss
phenomenon with the data intact.

**Two mechanisms remain, and `tval` alone does not separate them:**

1. **Software passed a genuine NULL** — the reading originally recorded, then retracted for want
   of a fired instrument. It is now *supported* again, though still not proven.
2. **The load returned zero DATA** — a stale read. Note the peer lane's measured S-10b signature:
   a stale read of a granule an `stc` has written returns a **clean zero**
   (`ld x29, 40(a1)` -> `0x0000000000000000` before the fix, `0xfedcba9876543210` after).

## And S-10b's DEFECT is present in BOTH bitstreams — verified

    c867dfcbb (S-10b fix) ancestor of 80843404c?  NO  -> defect present
    c867dfcbb (S-10b fix) ancestor of 39b21639d?  NO  -> defect present

That is exactly what a fault reproducing on both images requires, and it is the **data** route,
whose signature is a clean zero — matching `tval == 0` rather than the tag-loss family.

**S-10b is now the leading candidate**, displacing the tag-loss reading that has framed this
investigation since the beginning.

## Consequence for the S-07 filing

The site was filed under S-07's instance-1 residual on the strength of an identical *code shape*
(stack spill, reload, dependent `cincoffset`, mcause 25, no same-granule plain store). That shape
match stands. But S-07 is a **tag**-loss family and this measurement says the value was **zero**,
so the filing needs qualifying: same shape, possibly different mechanism. Worth re-reading whether
instance 1 ever had a `tval` reading of its own — if it did and it was pointer-like, the two are
different faults despite the identical shape.

---

# The entry marker perturbs too — and the structural wall, restated precisely

`CAPSTONE_ENTRY_MARK` writes the incoming `pWInfo` value and type into the SHARED REGION at
`base + 0x800`, so it survives a wedge and is read over JTAG while the core is halted. That was
meant to escape the bind that "any observation which survives prevented what it was observing",
because **memory survives a wedge; only REPORTING dies**.

It escapes the *reporting* half. It does not escape the *perturbation* half.

    sqem (entry marker)   completed 3 / 3, wedged 0
    sqpad10 (10 nops)     wedged 2 / 3
    sqrt (un-probed)      wedged 5 / 5 + the A/B baseline

Three consecutive completions at a ~2-in-3 rate is p ~= 0.04 — suggestive, not proof. But it
matches what the S-07 folder recorded from three instrumented builds: **"a software probe can
never be the thing that fires — the uncovered site always kills the run first."**

## Two instrument saves in the same campaign, both of which would have manufactured findings

* **The staleness gate refused a spurious `tval`.** A wedging arm latched `mcause 9` at
  `mepc 0xffffffff800072cc` — a kernel ecall — with `tval = 0x73`. The gate (added after it
  misread a VOID boot) rejected it. Without it: "the operand was 0x73", from another
  process's trap.
* **A "wedge" that never entered.** The same arm read `G/enter: False`, dying in setup at
  `SQ: C/mkregion2` with `RGNN = 18`: **monitor region-pool exhaustion**. Each SLT arm costs
  1 `create_dom` + 2 `create_region`, so TWO SLT arms plus a control exhausts the pool. The driver
  prints `NO RETURN`, which reads exactly like a wedge. **One SLT arm per boot.**

## Where the fault actually stands

**ESTABLISHED**
* Faults at `sqlite3WhereCodeOneLoopStart+0x8c`, `cincoffsetimm` after `ldc` from `s0-0x70`,
  mcause 25, across 3 binaries, 2 bitstreams, ~10 boots.
* **The `tval` instrument WORKS** (fired at `0xBEEF`, `mepc` attributing it to the control's own
  instruction), so every `tval == 0` is real data.
* **TAG LOSS IS EXCLUDED.** `tval` carries the rs1 cursor; a de-tagged capability keeps its
  address bits and would read pointer-like. It read 0 — the VALUE was zero.
* **The S-10 fix is exonerated** (A/B on `39b21639d`), **S-10b is excluded** (a granule-aligned
  `ldc` presents word 0, the word the interlock DOES match; and its polarity is tag *survival*),
  **writeback displacement is excluded** (switch 204, 8 boots, selftest firing),
  **wrong-slot read is excluded** (spill and reload both `a0 = s0-0x70`, straight-line, no call).
* Shape, granule count, buffer pressure, the `ctag=0` class, load->store->load, the stack region
  and derived-subject metadata are all excluded by **five counted 16384-trial arms**.

**THE ONE OPEN QUESTION**, unchanged: was the register **already zero at entry** (caller passed
NULL -> software, ours) or **healthy at entry and zero at the reload** 18 instructions later
(-> silicon)?

**AND IT MAY NOT BE ANSWERABLE BY A SOFTWARE PROBE AT ALL**, because every probe built to answer
it has removed the fault. That is not a failure of any one probe; it is the same wall the S-07
folder hit with three instrumented builds.

## What would actually settle it, and it is not more software

* **An RTL-side trace** of the spill/reload pair with the real workload's state — which needs the
  `axi_delayer` or an equivalent, since the microbenchmark environment provably cannot hold the
  triggering condition (peak write-buffer occupancy 1 against depth 8).
* **Or a clear on the s07 LDC recorder** (`load_unit.sv:766` is one-shot with no clear and is
  consumed by boot software before any domain runs), which would make the existing granule
  apertures usable and answer it with no new probe at all. Purely additive; competes for a
  synthesis slot.
