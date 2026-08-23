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
