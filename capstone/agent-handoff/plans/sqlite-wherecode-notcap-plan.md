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
