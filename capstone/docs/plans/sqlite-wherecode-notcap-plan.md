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

---

# CORRECTION: the flown bitstream DOES carry a combinational-loop hazard

Earlier this plan recorded, on the RTL lane's word, that the `UNOPTFLAT 39 -> 40` in
`wt_dcache_mem.sv` belonged to a **rejected** variant and the shipped S-10 fix was clean. **That
was wrong, and they have retracted it.** Measured single-variable, the only `core/` difference
between the two commits being that one file:

    39b21639d  (S-10 ABSENT)   UNOPTFLAT 39   LINT GATE PASS
    80843404c  (S-10 PRESENT)  UNOPTFLAT 40   LINT GATE FAIL
    signal added: cva6.gen_cache_wt.i_cache_subsystem.i_wt_dcache.rd_ctag

One signal added, none removed. `4fee13b2d`'s own subject said so — *"costs a combinational loop
— NOT ready to merge"* — and nothing after it resolves it.

**Do not over-correct either.** It is a *Verilator* warning, not proof of a physical loop, and the
bitstream built, so DRC `LUTLP-1` passed — with S-10b as the positive control that the check does
fire. Buildable and carrying a flagged hazard are both true at once.

**Consequence for this investigation's write-ups:** `caplifive_s10fix_80843404c.bit` — the image
most of this work ran on — carries it. Nothing measured is invalidated, but it is a **SECOND
uncontrolled difference** between the two images, alongside the WNS gap (−10.629 vs −16.400 ns),
and both belong beside any A/B result. The A/B conclusion is unaffected in direction: the fault is
**present on both**, and neither a timing margin nor a lint hazard explains a fault appearing on
the image with the *better* margin and the *cleaner* lint.

# The recorder clear is BUILT and IN SYNTHESIS

`s07-recorder-clear-39b` @ `84ed6eafb`, based on **`39b21639d`** — deliberately not the flown
image, because a failing lint gate cannot separate a new change from an inherited one, and the
project rule is that the gate passes before RTL goes to synthesis. That base also already has an
exit-0 synthesis, which is the best prior for a 32-line additive change.

    lint gate      PASS, UNOPTFLAT 39
    signal set     byte-identical to base (not merely the same count)
    capldc         PASS at 529 cycles

**Switch 160** (bank `3'b101` reg 0), level-sensitive, mirroring `dom_switch_log_clear` in the
same bank. **UART-safe** (`160 & 3 == 0`), so it can be applied mid-run — which is the entire
point. Bank 101 has no read mux, so it costs no readable aperture. The clear sits *after* the
capture arm so it dominates while applied, and it clears the granule address and source **with**
the valid bit, so a stale granule can never sit beside a fresh valid.

**The last-wins reduction was NOT taken**, as requested.

**Branch hazard:** there are two on the remote. Build **`s07-recorder-clear-39b`**. The other,
`s07-recorder-clear` @ `170512416`, is the same change on the failing-gate base and is stranded
because the push hook forbids branch deletion.

**Note the new image will be S-10-ABSENT plus the clear** — two differences from the flown image.
That is fine for this purpose: the A/B already established the fault reproduces on `39b21639d`,
so the arm will still fault there.

---

# The recorder clear is FLASHED and appears to work — but the delivery path defeats it

`caplifive_s07clear_84ed6eafb.bit` flashed, resident, re-read after power cycle. WNS −13.516 ns,
approved by the project lead **with that number stated**, on the argument that the experiment is
**self-gating**: the acceptance criteria fail loudly if timing corrupts the instrument, so a bad
image yields a failed criterion rather than a plausible wrong reading.

## Criterion 1 is UNOBSERVABLE as specified, and that is a specification error

    PRE-RUN baseline    sw=208 = 0xb8   ldc0_valid=1  src=1 (miss refill)
    after asserting 160 sw=208 = 0x98   ldc0_valid=1  src=0 (L1 hit)

**The switches are BOTH the clear input and the read selector.** Reading aperture 208 writes the
switch byte to 208, which **de-asserts 160** — so *the act of reading releases the clear*. It is
level-sensitive (correctly), the recorder re-arms first-wins, and with Linux running an untagged
`ldc` re-takes the slot before the read lands.

So `ldc0_valid=1` with a **fresh `src`** is a re-capture, not a failed clear. `src` changing
1 -> 0 is evidence the clear DID fire — a no-op clear would have left the record untouched.

**Recorded as: the clear probably works; criterion 1 as written cannot observe it.** NOT recorded
as a pass — the acceptance criterion is not being relaxed because its result was inconvenient.

## The same mechanism defeats the intended USE, which is the real problem

* **Switch writes do not complete mid-run.** `set_switch` timed out on `switch_state` at BOTH
  10 s and 30 s while the console streams UART, so there is no clearing between arms.
* **Clearing in the pre-run phase works** — but the clear is released as soon as anything touches
  the switches, and **Linux consumes the slot again** before the domain runs. That is the same
  first-wins consumption the clear was built to solve, moved later in time.

Three options put to the RTL lane: make the clear **edge-triggered** (clear on 0->1, stay
cleared, so reads do not re-arm); **gate the re-arm on domain entry** so the first recorded LDC is
necessarily the domain's; or a **read aperture that does not collide** with 160 (fixes
verification only, not the Linux consumption).

## Also settled this round

* **The switches were NOT left stuck.** A later baseline read `0xb8` again, so the earlier `0x18`
  came from a VOID boot's store page fault, not from a held clear. The "accidental evidence the
  clear works" reading from that byte is WITHDRAWN.
* **The software route is dead, measured.** The entry-marker build has now completed ~4 times and
  wedged 0, against 5/5 for the un-probed binary. Every software probe built this session removes
  the fault, which is exactly why the hardware clear matters.
* **The fault still reproduces** on the new bitstream, and the memory map did not move.

## CRITERION 1: PASSED, on the RTL lane's ruling — and it is a PROOF, not an inference

They made the call I asked them to make, with the reasoning, rather than my relaxing my own
criterion because its result was inconvenient.

`s07_ldc0_src_q` has **exactly two writers**:

    capture arm:  if (ldc_result_back && !data_rtag && !s07_ldc0_valid_q)   <- gated on !valid
    the clear:    if (s07_ldc0_clear_i)                                      <- writes src = 0

So `src` changing 1 -> 0 has only two possible causes and **both require the clear to have
fired**: either the clear wrote it, or a capture wrote it — and a capture cannot occur unless
`s07_ldc0_valid_q` is 0, which nothing but the clear produces outside reset. There is no third
path. Absent a reset between the two readings, **the observed `0xb8` -> `0x98` is a proof.**

**The unobservability was the instrument's design, not the criterion's wording** — their words:
level-sensitive was chosen deliberately to mirror `dom_switch_log_clear`, while knowing
`debug_byte_sel = switches_i[7:5]`, and the two facts were not connected. The criterion was
correct for the instrument as described.

Driver now also reads `paddr` at the same moment: the clear zeroes it and a re-capture writes the
new granule, so a moved paddr is a **second independent witness** from the same read.

## The fix that closes it by construction: gate the capture on `capmode`

A domain runs at M privilege with **capmode ON** (`CAPENTER` sets it, sticky). Linux does not, and
neither does the monitor before `CAPENTER`. So gating the recorder on capmode means **only a load
executed inside a domain can ever take the slot** — the Linux-consumption problem disappears
rather than moving later in time.

Cheap because `capmode_i` is **already an input to `load_store_unit`** (`:196`), currently
forwarded only to `i_pmp_data_if` (`:510`):

    load_unit.sv        one new input port, one AND term on the capture arm
    load_store_unit.sv  one wire to the existing i_load_unit instantiation

Two files; no change to `ex_stage.sv` or `cva6.sv`. Smaller than the clear itself.

**It composes with the clear rather than replacing it.** The clear still scopes between arms
within a boot; what changes is that **its release stops mattering**, because between arms nothing
is in a domain and nothing can re-capture. It also makes criterion 1 observable as originally
written, since a clear during Linux would then stay clear.

**Rejected, with reasons:** edge-triggered (option 1) makes verification possible but leaves Linux
consuming the slot — the actual problem — and inverts the semantics so a missed re-arm silently
yields a stale record, a worse failure mode than a visibly empty one. A non-colliding read
aperture (option 3) fixes the observation and not the thing observed.

**Cost: another synthesis and another reflash — the project lead's call, not ours.**

---

# Scoping the ONE remaining RTL change — auditor findings, and a route that may need none

The project lead's constraint: synthesis is ~90 min plus a reflash, so **this must be the last RTL
update**. RTL synthesis is HELD while both lanes audit the scope.

## The STC gap is CONFIRMED, and capmode would NOT fix it

`store_unit.sv:549-553` has **no** `!s07_stc_valid_q` guard, no clear, no scoping — unlike the LDC
side. And gating it on capmode changes nothing: the subject `stc`, the nine intervening stores and
the reload all execute **inside the same domain**, so capmode is 1 throughout. **Last-wins is the
binding problem and capmode is not a per-store selector.**

**So if the STC record is needed, that is a SECOND RTL change** (first-wins + clear, mirroring the
LDC side). Exactly the batching question the "make this the last one" constraint exists for — and
the reason holding was right rather than shipping the capmode gate alone.

## The no-RTL route: read the frame pointer at the halt

At a wedge the core is **halted**, so the frame pointer is directly readable — no instrumentation,
on the un-probed binary that wedges 5/5, where every in-domain probe REMOVES the fault.

    slot     = s0 - 0x70          (s0 = x8)
    granule  = slot & ~0xF
    tag byte = 0xBC2D2D2D + ((granule - 0x80000000) >> 4)

**`s0` IS a frame pointer — verified in the current binary, not assumed:**

    104a70: movc s0, sp                  <== s0 := sp
    104a74: cincoffsetimm s0, s0, 0x7f0

Not `-fomit-frame-pointer`; `s0` is written from `sp` in the prologue. That is what licenses
computing a slot from a halted register read — "the store is written relative to s0" and "s0 is
the frame pointer" are different claims and only the second licenses it.

**Deriving the address from build metadata was tried and FAILS.** The layout arithmetic is clean —
`code_size + blob + captable + storage + STACK = 4192768` for every build, so the stack END is
fixed at `DBAS + 0x3FFA00` — but measured slot offsets still differ by **64 bytes** between builds
whose only difference was INSIDE the faulting function. Call-chain consumption is not derivable.

## The tag-address formula: the header comment is WRONG, and dangerously so

    ariane_pkg.sv:592        CAP_TAG_MEM_BASE + (data_paddr >> 4)              <- NO subtraction
    wt_axi_adapter.sv:158    TAG_MEM_BASE + ((paddr - DATA_MEM_BASE) >> 4)     <- the actual code

`DATA_MEM_BASE = MEMORY_BASE = 0x8000_0000`. **The subtraction is real.** Dangerous rather than
merely wrong: for a domain paddr the two differ by ~5.7 MB and **both land inside the tag region**,
so the wrong one returns a real byte from an unrelated granule with nothing to flag it. Two
independent derivations agree on the subtracting form — that agreement, not either alone, is what
makes the read safe. The comment fix rides along with a future change; a comment cannot justify a
synthesis, and touching that file would invalidate the resident image's provenance.

## Two caveats on the tag byte, both stated rather than assumed away

* **Drain staleness.** The DRAM tag write is not synchronous with retirement, so a byte read
  before the subject store's write-buffer entry drained would hold the PREVIOUS occupant's tag —
  and a `0` would read as "never tagged" when the truth is "not landed yet". That is a
  confident-and-wrong answer pointing straight at software-NULL. **Almost certainly not biting:**
  the drain is not gated on the core (`miss_req_o = (|dirty) && free_tx_slots`, handshake from the
  memory side), so a wedged core stops issuing while the buffer keeps draining, and seconds pass
  before the halt-and-read. Recorded as an inference from the drain logic, NOT a measurement.
* **Direction residual.** The documented issue/return desync runs the OPPOSITE way (DRAM stale
  HIGH, L1 correct low). The mirror case — DRAM stale LOW after a genuinely tagged store — is
  UNRESOLVED: nothing says it is impossible and the path is direction-symmetric, but nobody has
  traced it. **So a tag byte of 1 is strong; a 0 carries this residual.**

## Independent witness for the ADDRESS, and its asymmetry

`cva6.sv:1352-1355` exposes `s07_ldc0_paddr[19:4]` at switches **205/206**, driven by switches
rather than by any instruction — so it **perturbs nothing**, the property every in-domain probe
lacks. Cross-checked against the `s0`-derived granule in the driver.

**Asymmetric by construction:** the recorder is first-wins and may still hold BOOT SOFTWARE's
capture, so a **MATCH is strong** (two independent routes agreeing on one address) while a
**MISMATCH is INCONCLUSIVE** — it may simply be somebody else's record — and is NOT a refutation
of the `s0` derivation. The driver says so in its own output rather than leaving it to be read
wrongly later.

---

# The RTL change may not be needed at all: the instrument is already on the board

Two audits ran in parallel against the "one last bitstream" scope. Both **refuted the change I was
about to request**, and the second turned up an asset nobody had used.

## The capmode gate is REFUTED — it would have filtered nothing

`core/csr_regfile.sv:295`:

    assign capmode_d = capmode_q | capmode_set_i;   // set by CAPENTER, sticky

An OR. The only write of 0 is reset (`:2994`). **No clear on domain exit, return, trap or wedge** —
and `CAPENTER` executes in the monitor's own init (`sbi_capstone_init.S`, reached from
`sbi_hart.c:1085`, the last statement before entering S-mode), so capmode is 1 for the monitor, for
Linux, and for domains alike. Verified in the flashed image itself, not the working tree:

    80020014: 1a94105b   <unknown>      <- CAPENTER, funct7 0x0d
    80020018: a009       j sbi_capstone_init_cap

The gate closes on an empty set. Worse than useless: it would have made the recorder look fixed
while changing nothing, on a run whose whole point was to be the last one.

**And the thing actually eating the slot is worse than "Linux somewhere":** the monitor's trap entry
does `LDC(gp, sp, -16)`, installed as `ctvec` *after* the CAPENTER, so **every ecall and every timer
tick** issues an untagged-capable LDC with capmode already 1. The window is not "something might";
it is "something will, on a millisecond cadence".

## The STC recorder cannot be fixed by scoping either — and the disassembly says why

An independent read of the actual window in the shipped binary (`sqbase.dom`,
`sqlite3WhereCodeOneLoopStart` at 0x1043b0):

    1043e8  cincoffsetimm a0, s0, -0x70     a0 = s0 - 0x70
    1043f0  stc  a2, 0x0(a0)                <== THE SUBJECT STORE
    1043f8  stc / 1043fc sw / 104404 stc / 104408 stc / 104410 sd
    10441c  stc / 104420 sw / 104428 sw / 104434 stc      <== 9 stores, 5 of them stc
    104438  ldc  a4, 0x0(a0)                <== THE RELOAD
    10443c  cincoffsetimm a4, a4, 0xb0      <== THE FAULT, +0x8c

Last-wins over *any* capability store means the record at the wedge is the store at `0x104434`,
0xb0 bytes away — so `s07_gran_match` is **guaranteed 0** and the verdict byte degenerates to
exactly the useless answer already measured. Capmode does not help: all nine stores are in-domain.

**One incidental result worth more than the instrument it came from.** `s0` is the entry `sp`
(`sp -= 0x7f0`, `s0 = sp + 0x7f0`, both multiples of 16), so `[s0-0x70, s0-0x60)` is a granule, and
the four plain stores target `s0-0x74`, `s0-0x98`, `s0-0x10c`, `s0-0x110` — **all in other
granules**. So the subject granule's tag is *not* legitimately cleared between store and reload.
`stc_clobbered` is excluded as an explanation, and a tag read of that granule is unambiguous on
that axis. That holds independently of which instrument eventually reads it.

## The asset: `core/tracer.sv` logs the tag bit, and it is ALREADY IN THE FLASHED BITSTREAM

    core/tracer.sv:126-131
      end else if (commit_instr_i[i].op inside {LDC, STC}) begin
          group_id = 4'd2;
          payload  = {63'b0, commit_instr_i[i].cap_result.result_metadata[CapTagBit]};

Every LDC and STC commit, with **its PC and its real tag bit**. 256-entry ring; the dump FSM is
clocked independently of the core; output reaches the physical console TX
(`cva6.sv:930` → `ariane_xilinx.sv:808` → `ariane_peripherals_xilinx.sv:316`); and the board server
**parses the format natively** (`/api/trace-start` → `trace_result {text}`), already wrapped by the
driver at `fpga_console.py:694`.

**It is in the bitstream on the board right now.** Checked against the revision the resident
bitstream was built from, not the working tree — the distinction that matters, since the tree is
ahead of what is flashed:

    git show 80843404c:core/cva6.sv    -> :960 `tracer #(`  :966 `) i_tracer (`
    git show 80843404c:core/tracer.sv  -> 436 lines, CapTagBit=64 at :94,
                                          identical group-2 payload at :126-129

Arming is CSR **0x810** (`csr_regfile.sv:213` decl, `:1787` write, `:449` readback), and it is
**not** in the domain-switch context list (`csr_regfile.sv:405-432` covers reg_ids 1-8, 9-25 cpmp,
57-66 — no trace), so it survives CAPENTER. Controls: ring mode `sw[2]`, dump `sw[1]`, TX mux
`sw[0]`.

**Why this is the right instrument and every software probe was the wrong one:** it adds **zero
in-domain instructions**. Every probe built so far perturbs the binary and the fault disappears
(probed builds complete ~4/4, un-probed wedge 5/5) — the single fact that has blocked this
investigation throughout. A commit-stage observer is outside that bind entirely.

## It is UNPROVEN, which on this project means broken

The tracer has never been shown to fire on this silicon. So the next boot is a **positive control
on a non-wedging domain** — not the wedge run. Open, and to be settled by that control rather than
by argument:

* Does the dump FSM complete against a **wedged** hart rather than a merely halted one? It looks
  independently clocked, which is the whole attraction, but that is a reading, not a measurement.
* **Ring polarity.** `overwrite_i=1` keeps the LAST 256 entries; `=0` keeps the FIRST 256 and then
  silently drops (`tracer.sv:163`, `:227`). SQLite issues far more than 256 LDC/STC pairs before
  reaching the fault, so first-256 is guaranteed to miss the window. Last-256 is right **only if
  the wedged core is not spinning through a trap handler that itself issues LDCs** — and the
  monitor's trap entry does exactly that. This is the one that decides whether the route works.
* Enable **group 2 only** — bit 0 and bits 16+ off — or a spinning wedge appends trap frames until
  the ring is nothing but exception records.

## Three corrections made while setting the run up, each caught by a gate rather than by care

**1. The subject binary was wrong, and it invalidated the whole previous boot.** The frame-pointer
route was pointed at `sqem.dom`. `sqem` is a **probed** build and completes on every boot; the
un-probed build that wedges 5/5 is **`sqrt.dom`**. The s0 route's entire value is that it needs no
in-domain probe, so running it against a probed binary tests nothing — and the boot came back
"completed", which reads like a non-firing draw rather than like a mis-aimed experiment. The
window transfers unchanged between the two binaries (verified instruction-by-instruction: both
carry `cincoffsetimm a0, s0, -0x70` at +0x38, the subject `stc` at +0x40, the reload at +0x88 and
the fault at +0x8c), and `movc s0, sp` at +0x0c confirms the frame pointer in `sqrt.dom` itself —
but that is a fact I checked afterwards, not a reason the run was sound.

**2. `SLT-FAIL query line=14 nvalue=0 nexpected=1` is the TEST FILE's bug, not a silicon result.**
Both `q_one.test` and `q_two.test` declare `----\n0`, expecting one value, while querying an
**empty** table, which correctly yields zero rows. Settled by running the native oracle rather than
by reading the file: `slt_native` produces the **byte-identical** failure line and summary for both
tests. So the arm that "failed" actually completed correctly, and the matched pair is measuring
wedge-vs-return, not answer correctness — which is what it was built for.

**3. The resident bitstream was NOT the one assumed.** The preflight guard hard-stopped with
*"resident bitstream is `caplifive_s07clear_84ed6eafb.bit`, expected `caplifive_s10fix_80843404c.bit`"*
— before powering the board, so no boot was spent. The tracer verification had been done against
`80843404c`. Re-checked against the revision that is **actually resident**: `84ed6eafb` carries the
same 436-line `core/tracer.sv`, the same group-2 payload at `:126-129`, the same `CSR_TRACE_ENABLE`
at `csr_regfile.sv:213/449/1787`, and the same switch wiring (`cva6.sv:929-930`, `:970-971`). The
conclusion is unchanged, but it was luck rather than method: **verify against the resident hash,
not against the revision you happen to have been reading.**

Two more caught by the preflight in the same launch: `SQLITE_STAGE_DOMS` splits on **`,`**, so a
space-separated pair is one arm (*"1 domains requested"*), and `FPGA_BITSTREAM` unset means the
resident-silicon guard *cannot fire* — it says so and blocks rather than passing quietly. Three
gates, three real errors, zero board time. That is what these gates are for, and it is worth
noting that none of the three would have announced itself in the results.

---

# The tracer boot: the fault reproduced on the right binary, the instrument did not fire

Boot structurally VALID — control `sqbase.dom` **returned in 7 s**, so this boot carries verdicts.

## The fault, now confirmed on the un-probed subject rather than inferred from a sibling

    [wedge] trap mepc = 0x00000000828f4ba0
    [wedge] trap tval = 0x0000000000000000

`sqrt.dom` loads at DBAS `0x82800000` with ELF VA base `0x10000`, so
`mepc - DBAS + 0x10000 = 0x104ba0` — **exactly `sqlite3WhereCodeOneLoopStart+0x8c`**, the
`cincoffsetimm a4, a4, 0xb0` two instructions after the reload. Same site, same `tval = 0`, on the
binary that actually wedges. Everything previously established about the window transfers, and now
without relying on a probed sibling to supply the disassembly.

## Both dumps were empty, and both causes were bugs in the driver, not in the silicon

**Cause 1 — the dump raced its own output channel.** `sw[0]` muxes the console TX to the tracer and
`sw[1]` triggers the dump, and `set_switch_value` chooses its own bit order (it only avoids the
destructive apertures). When `sw[1]` rose first the dump streamed into a TX that was not yet muxed,
every byte was lost — **and the FSM is ONE-SHOT** (`tracer.sv:330-335`: `dump_enable_i` must fall
and rise again before it will fire), so raising `sw[0]` afterwards produced nothing and the server
waited for a `trace_result` that could never arrive. That is arm 2's timeout exactly. Order is now
forced: `sw[0]` first, then `sw[1]`.

**Cause 2 — the arming readback did not prove what it claimed.** `set $csr2064 = 4` followed
immediately by `p/x $csr2064` returned `0x4`, and that was logged as "verified by readback". But
**GDB caches registers while the target is halted**, so that read can return GDB's own copy whether
or not the value ever reached the CSR. It is precisely the shape this project keeps paying for: a
check that fires, and still cannot separate the two hypotheses on the table. The arming now halts a
second time *after a resume* — which flushes the cache — and reports NOT ARMED if the value did not
survive.

Worth stating plainly: the dump FSM **did** run for arm 1 and reported an empty ring, so the read
path works end to end — server parse included. What is unproven is **capture**.

## What the ring being empty does and does not mean

It does **not** mean the tracer is dead. `sqbase.dom` completed, so thousands of LDC/STC
instructions committed while the mask was supposedly set; an empty ring under those conditions
points at the mask never reaching `trace_enable_i`, which is Cause 2's hypothesis and is now
directly testable.

**The next boot makes the check as easy to fire as possible before making it selective.** Arm 1
runs with groups 0-8 enabled (`0x1FF`) — group 8 is *every other committed instruction*, so if
capture works at all the ring cannot be empty. Only if that fires is the narrow group-2 mask worth
running. Arming a selective mask first and reading its silence as evidence is how an instrument
gets mistaken for a subject.

## Collateral: the frame-pointer read did not happen this boot

`[wedge] gdb mtval read failed (ActionTimeout)`. It ran on the previous boot, so the variable is
something this boot introduced — most likely the tracer dump immediately before it, which toggles
switches and puts the server into capture mode. The two instruments need to be sequenced or
separated, and that is a driver problem rather than a board one.

---

# RTL SIMULATION SETTLES IT: the tracer works, and the board failure was the ARMING PATH

Three board boots armed CSR 0x810 over GDB — once with group 2, once with groups 0-8, where
group 8 is *every other committed instruction* — and the dump FSM reported an empty ring every
time, from domains that retired thousands of LDC/STC. Rather than spend a fourth boot, the
question went to simulation, where it costs ~14 s. **That should have come before the second
boot**, and the rule it illustrates is one this project already has: do not board-debug an
instrument that simulation can exonerate for free.

`verif/tests/custom/capstone/tracer-capture.S` — real pass at **585 cycles**, not a timeout
(checked, because the harness prints SUCCESS at the timeout too):

    TRACER-DBG: trace_enable_i changed to 00000004 at time 484
    TRACER-DBG: CAPTURE port 0 group 2 pc 00000008000014e payload 0000000000000000
    TRACER-DBG: CAPTURE port 0 group 2 pc 000000080000152 payload 0000000000000001
    TRACER-DBG: CAPTURE port 0 group 2 pc 000000080000156 payload 0000000000000000
    TRACER-DBG: CAPTURE port 0 group 2 pc 00000008000015a payload 0000000000000001

Four capability accesses, four captures, correct group, correct PCs. **The payloads vary**, which
matters more than their values — a bit that is always 0 or always 1 would be equally consistent
with a tied-off signal, and this is not that.

**The single difference from the board is the arming route.** In sim the mask is written by an
architectural `csrw 0x810` executed by the core itself. On the board it went in over GDB, and the
readback was taken *at the same halt as the write* — which cannot distinguish the hardware
register from the debugger's copy. An empty ring under an unverified mask was never evidence about
the silicon; it was evidence about the instrument, and it took three boots to say so.

## The fix: arm from the running core, in the HOST, where perturbation is free

    sqlite_host.c
      csrw 0x810, %0        /* mask */
      csrr %0, 0x810        /* and PRINT it: SQ: tracearm=<n> */

Verified in the built artifact rather than in the intent — `edc: 81079073 csrw 0x810,a5`,
`ee0: 810025f3 csrr a1,0x810`, once each.

Three reasons this is the right home, and the third is the one that matters:

* CSR 0x810 has `bits[9:8] == 00`, i.e. **U-mode accessible**, and CVA6 enforces exactly that
  (`privilege_violation` tests `access_priv < csr_addr.csr_decode.priv_lvl`, never true for
  priv_lvl 0). So userspace may write it — no monitor change, no firmware/CAPENTER ordering
  question.
* `trace_enable_q` is cleared only by hardware reset (`csr_regfile.sv:1787` write, `:1111` hold,
  `:3078` update — two assignments, neither a clear), so arming once in the host survives into the
  domain and across arms.
* **It adds ZERO instructions to any domain image.** Every probe added inside a domain for this
  bug has made the fault disappear — probed builds complete ~4/4, the un-probed build wedges 5/5 —
  so an in-domain arm would buy observability by destroying the thing being observed. The host
  runs in Linux userspace before `capenter` and touches no domain binary, so this is the first
  instrument in the whole investigation that is outside that bind on **both** arms, not just the
  control.

The printed readback is the measurement all three GDB boots lacked. From here an empty ring is a
real statement about the tracer; before it, it was not.

## Consequence for the RTL side: no synthesis

The batch that was being scoped — dropping the first-wins term at `load_unit.sv:769`, a
watchpoint-granule filter on the STC capture and its `clobbered` arm, the `ariane_pkg.sv:592`
comment fix — stays **designed and unbuilt**. If the tracer returns both tag bits at the subject
PCs, none of it is needed.

## One observation NOT to build on yet

The four sim captures read **STC payload 0 then LDC payload 1** on the same granule — the inverse
of the S-07 direction. That is most likely an artifact of the test (`CAPCREATE` alone may leave the
register untagged until bounds and permissions are set, so the first store genuinely stores an
untagged value), not a finding. Recorded because it was seen, flagged because it is unexplained,
and explicitly not used as evidence for anything.

---

# The tracer FIRES on silicon — and group 2 can never see the subject window

Host-side arming worked on the first attempt. `SQ: tracearm=4` came back from the running core,
and both dumps returned **256 real entries** with real PCs and real tag bits. The instrument is
alive on the board, the arming is measured rather than assumed, and no RTL change was needed to
get there.

## But every entry is in the MONITOR, and none is in the domain

    arm1 (COMPLETED): 256 entries, 38 distinct PCs, range 0x800200a8-0x80024c2c
    arm2 (WEDGED):    256 entries, 38 distinct PCs, range 0x800200a8-0x80024c2c
    entries in the domain range 0x82800000-0x82c00000:  0

## The near-miss: this is NOT evidence that the wedge is a spin

The obvious reading — a ring full of monitor trap-stack PCs means the wedged core is spinning
through the trap handler — was the fork RTL and I had agreed to decide from exactly this
measurement. **It is wrong, and the check that catches it is comparing against the arm that did
not wedge:**

    PCs only in arm2 (the wedged one): NONE
    PCs only in arm1:                  NONE
    identical PC SET:                  True

The completed arm and the wedged arm produce the *same* 38 PCs at the *same* frequencies. So the
flood is **ordinary monitor trap traffic** — the trap entry's `LDC(gp, sp, -16)` on every timer
tick — scavenging the ring during the interval between the domain stopping and the dump. It
happens identically whether the domain returned or wedged, and therefore says nothing about which.

A proven-firing instrument that still cannot separate the two hypotheses on the table: the exact
shape this project's own rule names, caught this time before it was reported rather than after.

## Consequence: ring polarity is not the problem, and no setting of it helps

* `overwrite` HIGH keeps the newest 256 — monitor traffic *after* the domain stopped.
* `overwrite` LOW keeps the oldest 256 after arming — host and monitor traffic during domain
  *setup*.

The subject window sits in the middle, thousands of LDC/STC commits from either end. **A 256-entry
ring keyed on a group the monitor itself triggers continuously cannot reach it at any polarity.**
That retires the whole overwrite-HIGH-vs-LOW question rather than answering it.

## The route that does work: group 9, the store watchpoint — selective by ADDRESS

`CSR_WATCHPOINT_ADDR` (0x811) is a physical-address register, and group 9 logs the **64-bit value a
committed store wrote** when it touches that address (`tracer.sv:236-252`). Because it fires on one
address rather than on an opcode class, the monitor's trap traffic does not touch it and **the ring
does not flood** — 256 entries is then enormous headroom instead of a hard limit.

It also answers a sharper question than group 2 did. `tval` already established that the reload
*returned* zero; group 9 says what the subject store *wrote*. Wrote a tagged non-zero capability
and the reload returned zero → the value was lost between the two, i.e. silicon. Wrote zero →
software, and the whole S-07 framing for this instance is wrong.

Both CSRs are outside the domain-switch context list, so the host can arm them with zero in-domain
instructions, exactly as it now arms 0x810.

**Two things must be settled before spending a boot on it, and neither needs the board:**

1. **The subject slot's physical address**, which is `s0 - 0x70` and needs `s0` read at the wedge.
   That route is built but has now failed twice with `ActionTimeout`, both times after a tracer
   dump — it needs a halt-timeout fallback rather than more sequencing changes.
2. **Whether a 16-byte `stc` presents on `st_commit_paddr`/`be` such that the watchpoint can see
   it at all.** If it does not, group 9 silently never fires on the subject store and the empty
   result reads as "no store happened". That is an rtl-sim question, answerable in the same harness
   that just proved group-2 capture, and it is a **positive control that must fire before any
   empty group-9 result is believed.**

---

# The subject slot, read at the wedge with no instrumentation at all

    [wedge] gdb CSRs:  mcause=2 mepc=2 mtval=0        <- the known junk; the LATCHED
                                                          apertures are the real values
    [wedge] gdb frame: s0(x8)=0x82b9f3d0  sp=0x82b9e790
    [wedge] trap mepc = 0x00000000828f4ba0             <- +0x8c, the faulting instruction

**The read is genuine, and the check that says so is not the range check.** Any plausible-looking
number passes a range check. This one satisfies an arithmetic identity it could not satisfy by
accident:

    s0 - sp   = 0xc40
    prologue  = 0x7f0 + 0x450 = 0xc40          <-  cincoffsetimm sp, sp, -0x7f0   (+0x00)
                                                   movc s0, sp / +0x7f0           (+0x0c)
                                                   cincoffsetimm sp, sp, -0x450   (+0x14)

So the core is halted **inside `sqlite3WhereCodeOneLoopStart`'s own frame, past its prologue and
before any further call** — which is exactly where `mepc` says the fault is. Two independent
routes, one from the trap latch and one from the register file, agreeing on the same frame.

Secondary checks, all passing: `s0 - DBAS = 0x39f3d0`, inside the 4 MiB allocation; `s0` is
16-byte aligned; the slot sits `0x606a0` below the fixed stack end at `DBAS+0x3FFA00`, a plausible
call depth rather than a boundary value.

    SUBJECT SLOT = s0 - 0x70 = 0x82b9f360        va_offset = 0x39f360

**It is 16-byte aligned, so it is a GRANULE BASE** — which the watchpoint requires and which is not
a coincidence worth glossing over: the comparator is word-granular (`st_commit_paddr[PLEN-1:3]`)
against a capability store's single queue entry, which carries the granule base. An address
anywhere in the upper half of the granule would compare against word 1 while the entry holds word
0, and would **silently never fire**. A capability store must be 16-byte aligned, so the address we
care about and an address the watchpoint can match coincide here — but they are different
statements and only alignment makes them agree.

## Carry the OFFSET, never the physical address

`DBAS` is **not stable across boots or arm positions** — this same boot shows `0x82400000` for arm
1 and `0x82800000` for arm 2. Carrying `0x82b9f360` into a later run and arming it blind would
point at a different domain's allocation, group 9 would fire on nothing, and **an empty group 9
reads as "the subject store never happened"** — i.e. it would manufacture the software-NULL
conclusion that is the very thing under test.

`va_offset = 0x39f360` is the stable quantity: same binary, same query, same call chain. The run
that arms the watchpoint must recompute `slot = DBAS_that_run + va_offset` and **refuse to report a
group-9 result at all if the DBAS it actually observes differs from the one the prediction assumed**.
That turns the silent failure into a loud one, which is the only version worth having.

## Two loose ends from this boot

* `gdb mtval read failed (UnboundLocalError)` fired immediately after the frame line, so the
  shadow-tag read did not run. The frame data was already printed, so nothing was lost, but the
  granule/tag read still needs the fix.
* The GDB CSRs read `mcause=2 mepc=2 mtval=0` — the documented nested-trap collapse. Worth
  restating because it is the reason the latched apertures exist: **the debugger's own view of the
  trap cause at this wedge is junk, and only the hardware latch is trustworthy.** The frame
  registers are a different matter — the monitor saves and restores `s0` across traps, so ordinary
  timer-tick traffic cannot clobber it, and the prologue identity above confirms it did not.

---

# THE DISCRIMINATOR FIRES: software stored a real pointer, the reload returned zero

Store watchpoint armed at the subject slot `0x82b9f360`, group 9 only, arming and address both
written by the running core and read back by it (`SQ: tracearm=0x200`, `SQ: tracewp=`).

    [    19]  WATCHPOINT   PC=0x0000000828f4b54   DATA = 0x0000000082be4cf0
    [    20]  WATCHPOINT   PC=0x0000000828f4b54   DATA = 0x0000000082be4cf0
    Total: 21 entries

`0x828f4b54` is `sqlite3WhereCodeOneLoopStart+0x40` — **the subject `stc a2, 0x0(a0)`**, the spill
whose reload at `+0x88` feeds the faulting `cincoffsetimm` at `+0x8c`.

    stored cursor    0x82be4cf0     (group 9 payload = st_commit_data, the 64-bit value written)
    reloaded cursor  0x00000000     (tval at the trap = the faulting instruction's rs1 cursor)

**Both numbers are the same quantity** — a 64-bit capability cursor — so this is a like-for-like
comparison, not an inference across two different measurements. `0x82be4cf0` is inside the
domain's own allocation (`0x82800000`-`0x82c00000`), i.e. a plausible live pointer, not a sentinel.

**Software wrote a real pointer to that slot. The reload of that slot produced zero.** The value
was lost between the store and the load.

## Why the ring being SHORT is the strongest part of the result

21 entries against a 256-entry capacity — **the ring never wrapped**, so this is the *complete*
list of every committed store to that address for the entire run, not a window onto the last few.

That closes the one remaining way software could still be to blame: if some later code had nulled
the slot between the spill and the reload, there would be an entry at that address with
`DATA = 0` after index 20. **There is none.** The last thing anything wrote to that granule was
`0x82be4cf0`.

## What the two controls in this same boot establish

* **Arm 1 is a negative control that fired.** `sqbase.dom` landed at `DBAS 0x82400000`, so the
  compiled-in address pointed into a different allocation — the guard said so, and the dump came
  back `No entries captured.` A watchpoint that returned entries there would have meant it was not
  address-selective at all.
* **Arm 2's guard passed**: `DBAS 0x82800000` matches the basis the address was derived from, so
  the armed address is in this domain's allocation.

Together: the instrument fires when aimed correctly, stays silent when aimed elsewhere, and the
run that produced the result is the one where it was aimed correctly.

## Status: NOT YET A ROOT CAUSE

This localises the loss to **between the committed store and the load's return**, which is the
memory path, and it retires the software-NULL hypothesis for this instance. It does **not** say
which part of that path, and it does not by itself distinguish a lost tag from a lost value —
though `tval = 0` means the cursor bits themselves came back zero, which a de-tagged capability
would not do.

Going to `claim-auditor` before this enters ISSUES.md or any report.

## Gap 1 closed twice, by two independent methods

The board result compares a group-9 payload against `tval`. That is only like-for-like if the
payload is the stored capability's **cursor** — `st_commit_data` is `[XLEN-1:0]`, 64 bits of a
128-bit store, and if that half were the metadata the whole comparison would be meaningless.

**By measurement** (`verif/tests/custom/capstone/watchpoint-cursor.S`, pass at 551 cycles): store a
capability whose cursor is set to a deliberately distinctive value and see what comes back.

    CAPPRINT  Reg[13]: Cursor: 0000000080003030 | Metadata-> Revnode_id: 2 | Type: 2 | Perm: 7 ...
    TRACER-DBG: CAPTURE port 0 group 9 pc 000000080000162 payload 0000000080003030

Identical. **The payload is the cursor.**

**By source, independently** — the data and metadata travel on deliberately disjoint lanes:

    store_buffer.sv:197   st_commit_data_o = speculative_queue_q[rd_ptr].data
    store_buffer.sv:118   .data = data_i
    store_unit.sv:462     data_i = st_data_q
    store_unit.sv:369     st_data_n = data_align(lsu_ctrl.vaddr[2:0], lsu_ctrl.data)
    store_unit.sv:377     st_user_n = (op inside {STC} || sel_dom_switch) ? lsu_ctrl.user : '0

The metadata rides `user`, a separate register behind its own gate; `st_commit_data_o` is wired to
`data`. So `0x82be4cf0` is the stored cursor and `tval` is the rs1 cursor at the trap — same
quantity, same half, both 64-bit.

## And a precision the exhaustiveness argument needs: it is WORD 0, not "the slot"

The watchpoint compare is `st_commit_paddr[PLEN-1:3]` — word-granular — so a store to
`G+8..G+15` presents word 1's tag against an entry carrying word 0's and **never fires**. Nothing
in the hit condition tests `is_cap` (`cva6.sv:904-906` gates on `lsu_commit_commit_ex`, the address
compare, and `st_commit_be[...]`, and `st_be_n = lsu_ctrl.be` is set for every store), so plain
`sd`/`sw` stores **are** visible — but only to word 0.

So the correct statement of the exhaustiveness argument is:

> No entry with `DATA = 0` after index 20 proves nothing nulled **word 0** of that granule between
> the spill and the reload.

That is the word that matters — the cursor lives in word 0 and `tval` reported the cursor as zero —
so the argument survives intact. It is stated this way because a plain store to the granule's upper
half is genuinely invisible to this instrument, and an unqualified "nothing wrote to the slot"
would be a claim the measurement does not support.

---

# RETRACTION: "the reload returned zero" was never measured

An auditor found the instruction I had not looked at, and it breaks the memory-path half of the
claim. Disassembly of the four instructions before the fault, which the whole trail had skipped:

    +0x7c  cincoffsetimm a5, s0, -0x120
    +0x80  movc  a4, zero          <== the LAST architectural writer of a4 before the reload
    +0x84  stc   a4, 0x0(a5)
    +0x88  ldc   a4, 0x0(a0)       <== the reload
    +0x8c  cincoffsetimm a4, a4, 0xb0     <== the fault, rs1 = a4

**The prior architectural value of `a4` is exactly `{cursor 0, NOT_CAP}`** — byte-for-byte what
the FLU is observed to have received. So a **stale operand delivery** — the load returning
correctly and the consumer being handed the *previous* `a4` — predicts `mcause 25`, `tval 0`, at
`+0x8c`, with memory and the load both entirely correct.

## What `tval` actually measures, and what I claimed it measured

    core/ex_stage.sv:489   tval : ... fu_data_i[0].operand_a
    core/ex_stage.sv:797   cap_rs1 : decompress_cap_tagged(fu_data_i[0].operand_a, ...)

Guard and `tval` come from the same `fu_data_i[0]` in the same cycle. So what is established is
**"the FLU received `{cursor 0, NOT_CAP}` for a4"** — not "the load returned zero". The bracket the
instrument measures is `[store commit → operand delivery]`, which contains writeback and forwarding,
**not only memory**.

I wrote "the reload of that same slot produced zero". That measurement was never taken. **Retracted.**

## This is not a hypothetical alternative — it is a documented defect class on this core

`capstone/tests/fpga-repros/R20-stc-rs1-cursor-forward-x10/00-README.md`, sim- and board-reproduced:
*"memory is right, the load is right, and only the consumer's operand is wrong."* Its fix is in the
resident bitstream (`core/issue_read_operands.sv:568`) but was **x10-specific and empirical**, not a
proven invariant; the general capability-forward path (`issue_read_operands.sv:674-677`) is
register-agnostic.

And the signature I had been treating as a curiosity — **"every software probe removes the fault"** —
is a *scheduling* signature, which is exactly what R-20 showed: cured by one nop on the board, four
in simulation. That fact has been sitting in this document as evidence of nothing since the first
day; it now reads as a positive indication for the forwarding hypothesis.

## What SURVIVES, precisely worded

* A committed store put cursor `0x82be4cf0` into **word 0** of PA `0x82b9f360`. **Holds.**
* **No committed store instruction** wrote word 0 of that granule afterwards. **Holds** — with
  blind spots that must travel with it: AMOs are excluded at `commit_stage.sv:339`, domain-switch
  stores bypass the speculative queue the tap reads, and the tap is upstream of the write buffer and
  D-cache, so R-18/R-19/S-07-class corruption is invisible to it *by construction*.
* **This is not a software NULL.** **Holds** — and it is the load-bearing result, because both
  surviving alternatives are hardware.
* "The loss is in the memory path." **UNPROVEN.** Operand delivery survives as an alternative.

## Two wordings to stop using

1. **"valid non-zero capability."** The group-9 entry is `{group, pc, data[63:0]}` with **no tag
   bit** (`tracer.sv:237-239`). Validity at store time is *unmeasured*. Say **"cursor
   `0x82be4cf0`"**. (The `sw=208` recorder's `stc_ctag=0` is about `paddr 0x82b9f2b0` = `s0-0x120`,
   the `stc` at +0x84 — a **different store**, and reading it as the subject's tag would have been a
   second error.)
2. **"nothing nulled the slot"** → "no committed store instruction wrote word 0", with the blind
   spots above.

## One thing the audit made STRONGER

Every one of the 20 domain-range captures decodes as a store instruction — near-conclusive that the
address filter was armed and working. And **entry 11 is a built-in positive control I had not
noticed**: a plain `sw` captured at that very word with `DATA = 0x0`. So a zero-data scalar store to
word 0 *is* visible to this instrument, which means "no `DATA=0` after index 20" is a real absence
rather than a silent zero.

## The discriminator, and it already exists

**Read the 16 bytes at `0x82b9f360` and its shadow tag byte, over GDB, at the wedge.**

    memory intact (0x82be4cf0, tagged)  -> operand delivery. RETRACT the memory-path claim.
    zeros / compress_cap(NULL)          -> memory path. The claim stands.

The instrument is already written and already aimed; it failed on a driver `UnboundLocalError`
immediately after printing the frame registers. **That one-line bug is the single highest-value fix
in this whole trail** — it was three lines away from settling the fork two boots ago.

---

# THE FORK IS SETTLED: memory is INTACT and TAGGED, the consumer got zero

Read over GDB at the wedge, on the un-probed binary, with nothing added to the domain:

    [wedge] trap mepc     = 0x00000000828f4ba0     (+0x8c, the faulting cincoffsetimm)
    [wedge] trap tval     = 0x0000000000000000     what the CONSUMER received
    [wedge] gdb frame     : s0(x8)=0x82b9f3d0  sp=0x82b9e790
            subject slot  = 0x82b9f360
            granule data  = 0x0000000082be4cf0     <== THE VALUE IS THERE
            shadow tag    = 0x01                   <== AND IT IS TAGGED

Tag address arithmetic checked rather than trusted:
`0xBC2D2D2D + ((0x82b9f360 - 0x80000000) >> 4) = 0xBC58CC63`, which is the address read.

## What this decides

| hypothesis | prediction | result |
|---|---|---|
| **Memory path** — the slot lost the value | granule reads 0 / untagged | **REFUTED** |
| **Delivery** — memory fine, consumer got the wrong operand | granule intact and tagged | **SUPPORTED** |

**The memory-path reading is now dead, and I had published it.** Three independent numbers agree
that memory did its job: the store watchpoint recorded `0x82be4cf0` written; the granule still
holds `0x82be4cf0` at the wedge; and the shadow tag byte reads `1`. Meanwhile `tval` says the
consumer was handed cursor `0`.

**The value was never lost. It was never delivered.**

## Why this is trustworthy, and where it still is not

* `s0 = 0x82b9f3d0` is **bit-identical to the previous firing boot**, so the call depth is
  deterministic and the slot address is not a one-off coincidence.
* The DBAS guard fired correctly on arm 1 again (`0x82400000` vs the `0x82800000` the address was
  derived from) and that arm's group-9 dump came back empty — the negative control still works.
* **The honest limit:** a DRAM read reports *what DRAM holds*, not *what the load saw*. The
  documented issue/return desync runs DRAM-stale-high while L1 is correct-low; the mirror case is
  UNRESOLVED. So this does not by itself exclude the load returning something different from what
  memory holds — it excludes memory having *lost* the value, which is a weaker and different claim
  than "the load path is innocent".

## What remains open, stated as the next question rather than a conclusion

The fault is now bracketed to **[load memory access → consumer operand delivery]**, and the prime
suspect is a stale operand: `movc a4, zero` at +0x80 makes `a4`'s prior architectural value exactly
the `{cursor 0, NOT_CAP}` the FLU received.

**The obvious mechanism for that has already been refuted by measurement** — the wrong-producer
forwarding theory needs two live scoreboard entries sharing an `rd`, and a scoreboard checker with a
proven heartbeat measured **zero** such cycles with peak occupancy 2 of 8, even under a scalar test
built to force backpressure. So the *location* is established and the *mechanism* is not.

Going to an auditor before this enters ISSUES.md.

---

# RETRACTION 2: "memory intact ⇒ the fault is in delivery" is an INVALID INFERENCE

An auditor refuted the headline result. The two measured facts stand; the conclusion drawn from
them does not.

**The inference was:** granule intact and tagged at the wedge ⇒ the memory path did not lose the
value ⇒ the fault is in operand delivery.

**Why it is invalid:** the decision table's "memory path ⇒ granule reads 0/untagged" is only
correct for a **persistent** loss. **Every documented memory/load-path defect on this core is
transient and self-heals** long before a debugger read seconds later — write-buffer residency, the
issue/return desync (`wt_dcache_wbuffer.sv:612-619`), S-10b. In all of them the store *does* reach
DRAM correctly; only the load, at the instant it executed, saw something else. So an intact granule
at T+seconds is predicted by **both** arms and discriminates nothing.

**And the existence proof is unfixed on this exact silicon.** Verified independently rather than
taken from the report:

    git merge-base --is-ancestor c867dfcbb 84ed6eafb   ->  false
    c867dfcbb  S-10b: compare the load/store hazard at GRANULE granularity,
               and a stale-data read disappears

S-10b's own record is a load returning `0x0000000000000000` while memory held the stored value.
That is a load-path fault producing a clean zero with memory intact — precisely what the retracted
claim called impossible. (S-10b as a *named* mechanism is unlikely here: `load_unit.sv:312` presents
`vaddr[11:0]` and `store_buffer.sv:279/287/293` compare `[11:3]`, so a granule-aligned reload and a
granule-aligned spill share `[11:3]` and the interlock would stall. It is cited as the counterexample
to the inference rule, not as the mechanism.)

## Two instrument defects found by EXECUTING the driver, not reading it

**I1 — the granule read was truncated to ONE word.** The pattern
`r"0x[0-9a-fA-F]+.*?:\s+0x[0-9a-fA-F]+"` is non-greedy and ends after the first hex group, so
`x/2gx` reported **word 0 only** and word 1 — the capability's METADATA half — was requested and
never printed. So "the granule is intact" was always a **word-0-only** statement that did not say
so. Both instruments backing "intact" (this and the group-9 watchpoint, `st_commit_paddr[PLEN-1:3]`)
are blind to the same half of the granule. Fixed, and negative-tested against realistic output:
the old pattern captures `0x...82be4cf0`, the new one captures both words.

**I2 — the SAME UnboundLocalError, reintroduced 110 lines below its own fix.** `_rd` is used at
`:2594` and `def`ined at `:2614`; a `def` binds the name as a local of the enclosing function, so
the use raises, and the enclosing `try` swallows it. Silently skipped on **every wedge**:

* the **entry-marker read** — the one measurement that would settle whether `a4` was already zero
  at function entry;
* the **mandatory positive control** for the tag path, whose own in-source comment says *"it is not
  optional: if it reads 0 the read path is unproven and the subject's tag carries no verdict"*;
* the gdb-`mtval` mcause/mepc cross-check.

**The evidence file from the boot that produced the headline result ends at exactly the predicted
truncation point.** I had fixed this identical defect for `_memrd` earlier in the same session and
written a comment about it — then reintroduced it below.

## What survives, worded so it cannot be over-read

* A committed store put cursor `0x82be4cf0` into **word 0** of the slot. **Holds.**
* Word 0 still held it at the wedge, and the granule's shadow tag byte read 1. **Holds** — word 0
  only, and at T+seconds.
* The FLU received cursor 0, and separately `cap_type == NOT_CAP`. **Holds** — and note these are
  **two facts on two lanes** (`tval` reports `operand_a`, the cursor; cause 25 is raised on the
  metadata's `cap_type`), consistent with a whole-zero 128-bit operand but not a single observation.
* **Excluded:** a software NULL; anything having **persistently overwritten** word 0.
* **NOT excluded:** a transient memory-path defect. **What the load returned has never been measured.**

Also standing from the audit: attacks on `tval` semantics and on latch coherence both **failed** —
`ex_stage.sv:489` and the single-cycle `always_ff` at `cva6.sv:1133-1143` hold, and the commit-stage
producer is excluded arithmetically. The tag-address formula is confirmed at the bitstream rev. And
this is **N=1 on a fault the folder itself calls sporadic**.

---

# The load-syncer lead, its checker, and why the first measurement is VOID rather than negative

## The mechanism, verified in source rather than taken from a report

`func LDC` does not read the dcache response directly. It dispatches and rendezvous
(`capstone_dyn_unit.anvil:368`, `:370`, `:372`), and the pairing is done by
`capstone_load_syncer`, which demultiplexes the **shared** load-unit response stream — ordinary
scalar loads included — against the DYN unit's pending capability load, **matching purely by
`trans_id`**. The syncer holds **one** pending identifier and sets it on a new `init` with **no
guard** on `req_set`:

    set cap_trans_id := trans_id;
    set req_set := 1'd1

So a second LDC's `init` overwrites the first's pending id while the first is outstanding. The
first LDC's response then fails the match and is **diverted onto the scalar return path**
(`send lsu_ep.normal_res(msg)`), while `check_load_data` pairs whatever did match against whatever
`cap_msg` it dequeues — taking `rd = msg.cap_result`, a whole `fat_cap_t`, **cursor and metadata
together** (`capstone_unit.anvilh:584`).

**That single coupled substitution is the only shape found that is consistent with `tval == 0`
AND `cap_type == NOT_CAP` simultaneously.** It also fits sporadicity, cure-by-added-instruction
(cycle realignment), and five short directed sims missing it.

**Only LDCs contend.** `send cap_load_ri.init` has exactly one sender —
`capstone_dyn_unit.anvil:326`, inside `func LDC`. The comment at `:555` reading *"There is a STC
request"* is **stale and names the wrong instruction class**; the race needs **two overlapping
LDCs**, not an LDC racing one of the window's five `stc`.

## The checker, and the counter that saved it from being read wrongly

Instrumented in `capstone_load_syncer` — **not** the store syncer, whose register block is
character-identical, so a naive anchor matches both and would have measured an endpoint no LDC
ever touches. Counted separately, on purpose:

* **precondition** — `init` firing while `req_set == 1`;
* **outcome** — the same, with the incoming id differing from the pending one;
* **`init` TOTAL** — so a zero in the first two is attributable.

Outcome-only instrumentation cannot separate *"the race is unreachable"* from *"I did not catch
it"*. The precondition is strictly easier to trigger and is decisive **both** ways.

## First run: ALIVE, and inits = 0. VOID, not negative.

    S12-SYNC: ALIVE (load-syncer checker compiled in)
    S12-SYNC: tick 500  inits=0  init-while-pending=0  clobbers=0

The checker is compiled in and running — the heartbeat says so, which is exactly why it exists.
But **`inits = 0` on a test containing 23 `ldc` instructions**: the DYN unit's `func LDC` never ran,
so `s12-full-window.S` does not exercise the syncer at all. The wiring was checked rather than
assumed (`_dyn_ep_init_valid` ← `_cap_load_le_init_valid`, driven by `send cap_load_ri.init`).

**So this measurement says nothing about the hypothesis.** Without the total-inits counter it would
have read as a clean zero and the lead would have been wrongly dismissed.

**Next step is a POSITIVE CONTROL for the checker**: find or build a case that drives `inits > 0`,
prove the counter moves, and only then read a zero in the precondition column as evidence. Until
that fires, the syncer lead is neither supported nor excluded.

**Caveat that must travel with the file:** `core/capstone_dyn_unit.anvil.sv` is a **generated,
gitignored artifact**. `make -C core/anvil_build` silently removes this checker — hence the ALIVE
line, so absence means "not compiled in" rather than "measured zero".

## The load-syncer lead is DEAD: the DYN unit serialises LDCs, measured

    s12-ldc-overlap.S -- 96 back-to-back LDCs to distinct granules and distinct destinations
    S12-SYNC: tick 1450  inits=96  init-while-pending=0  clobbers=0
    S12-PAR:  tick 1450  dyn-dispatches=99  ldc-inits=96

**96 inits and not one arrived while a request was pending.** The mispair needs a second LDC's
`init` to overwrite the first's pending id while the first is outstanding; that window never
opens. **The mechanism is unreachable, so it cannot be the cause.**

**The zero is meaningful because the counter is proven to move**: the previous test recorded
`inits=2` and this one 96, from the same instrument. That is the positive control the precondition
column needed, and it is why this reads as a measured zero rather than a silent one.

It also confirms by measurement what `ex_stage.sv:902-904` asserts in a comment — *"the dyn unit
serializes (capstone_dyn_ready backpressure => one op in flight)"*. The oracle flagged that as
argued-but-unproven; it is now measured.

## Two instrument failures on the way, both mine, both the same shape

**The report cadence outlived the test.** The first run printed only at tick 500 while the LDCs
under test execute at cycles 529-693 of a 734-cycle test — so every counter was displayed *before*
the window it was measuring and read as `inits=0`. I nearly recorded "the DYN unit never ran".
Cadence is now every 50 ticks, finer than any directed test is short.

**A naive anchor matched both syncers.** `capstone_load_syncer` and `capstone_store_syncer` declare
character-identical register blocks; instrumenting the wrong one would have measured an endpoint no
LDC ever touches. Caught by asserting the anchor count rather than trusting it.

Both are the same failure this investigation keeps producing: **an instrument whose silence has two
meanings.** The fix each time was a counter that makes "not exercised" distinguishable from
"exercised and clean" — the total-inits column here, the heartbeat in the scoreboard checker, the
must-fail arm in the adjacent-granule test.

## Where the hunt stands

Every specific mechanism proposed by either lane is now excluded, each with a quoted line or a
fired control: software NULL, plain S-07 tag loss, the R-20 forwarding class, wrong-producer
scoreboard selection, adjacent-granule scalar stores, write-buffer depth, the domain-switch cnull
pack, and now the load-syncer mispair.

**The open question is unchanged and is the one measurement never taken: what the LOAD RETURNED.**
The next step is the board-side one the auditor proposed — with the switch-160 clear applied before
the arm, read the s07 LDC recorder (`load_unit.sv:769-772`) at sw 208 for `{valid, src}` and sw
205/206 for the granule. `valid=1` with the subject granule says the reload itself returned
untagged and `src` names the leg (L1 hit / miss refill / write-buffer forward). Anything else is
**inconclusive**, not exculpatory: the recorder is first-wins, so an earlier legitimate untagged
`ldc` in the same arm consumes the slot.

## The per-arm clear worked, and the recorder STILL cannot answer the question

    [s07] recorder CLEARED before arm 2 -- its record now belongs to this arm's first untagged LDC
    [wedge] s07 ldc0 granule paddr[19:4] = 0x82280      <- NOT the subject's 0x9f360
    [s07] SELFTEST PASS -- the detector fires on this silicon

The clear executed (logged for both arms) and the selftest fires, so the instrument is working.
The record simply belongs to a **different, earlier untagged LDC inside the same arm** — which is
expected, because `load_unit.sv:774-780` documents that *"an `ldc` over a zeroed stack slot is
LEGITIMATELY untagged"* and SQLite does that constantly.

**So the reading is INCONCLUSIVE, not exculpatory.** It does not say the subject reload returned
tagged; it says the first-wins slot was taken before the subject reload happened. Recorded as such.

**This exhausts what the resident bitstream can measure.** The recorder is first-wins with no
address filter, so on any real workload it is consumed by legitimate untagged loads long before the
instruction under test. No arrangement of clears fixes that: the clear scopes it to an arm, and one
arm contains thousands of untagged loads.

## New data the regex fix bought: the metadata half of the granule

    granule data: 0x82b9f360:  0x0000000082be4cf0   0x0000072ba7462d16
                               word 0 = cursor       word 1 = METADATA, non-zero

Word 1 was requested on every previous boot and silently discarded by the truncating pattern. It is
**non-zero**, so DRAM holds a complete capability-shaped value rather than a cursor beside zeros.
That is consistent with the store having written a real capability and does **not** advance the
load question — but it is the first time the metadata half has been seen at all.

## What is now the gating item, and why a bitstream is finally justified

Every mechanism proposed by either lane is excluded. The single unmeasured fact is **what the load
returned**, and the resident silicon cannot report it. The minimal change that can:

**Drop `&& !s07_ldc0_valid_q` from `load_unit.sv:769`** — restoring the ROLLING capture that
already shipped working in `caplifive_s07debug_18august.bit` before `83a7d061f` reverted it. It is
a **strict fanin reduction**: it removes the register's own output from its own enable, i.e. deletes
a feedback edge rather than adding one. Combined with a freeze at the trap
(`recent_nontrivial_trap_seen_log_q`, already present at `cva6.sv:1015`), the record becomes *"the
LAST untagged LDC before the fault"* — which, two instructions after the reload, is the subject.

Better still if it rides with a **granule filter on `CSR_WATCHPOINT_ADDR` (0x811)**, which is
already a physical-address register, already outside the domain-switch context list, and which this
investigation has already **proven armable from userspace and exact** (the group-9 store watchpoint
fired on the armed granule and stayed silent on the unwatched one, confirmed against the
disassembly).
