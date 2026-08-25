# S-12 — `mcause 25` at `sqlite3WhereCodeOneLoopStart+0x8c`: the operand is zero, and it is not software

**Sibling issues, so a reader who arrived with the wrong symptom can leave now.** If your symptom is
an **untagged capability surviving a store/reload pair**, that is **S-07**, not this. If it is a
**write-buffer tag forgery**, that is **S-09**; a **write-buffer forwarding residual**, **S-10**; a
**scalar store clobbering capability metadata**, **R-18**; **`movc zero` metadata left in a slot**,
**R-19**. The issue this one is closest to, and which it may yet turn out to BE, is
**R-20 — `stc`/rs1 cursor forwarding** (`R20-stc-rs1-cursor-forward-x10/`). Read that one before
this one.

**This folder is about a fault whose LOCATION AND MECHANISM ARE BOTH UNESTABLISHED.** It is open,
and written to be handed over in that state rather than held until it is tidy. What is settled is
what has been *excluded*, and the exclusions are real even though the localisation is not:

> **A committed store put cursor `0x82be4cf0` into WORD 0 of the slot. That word still held it in
> DRAM at the wedge. The FLU received cursor 0.**
>
> This excludes a **software NULL**, and excludes anything having **persistently overwritten**
> word 0 in DRAM. It does **NOT** locate the fault.

**The shadow-tag half of that evidence is WITHDRAWN as evidence about the load.** The byte read
`0x01`, and that is a true statement about DRAM and a useless one about the `ldc`. The tag the load
consumed comes from a **different storage**: `wt_dcache_mem.sv:143` declares a per-{set,way} L1
array `cap_tag_q`, and `rd_ctag_o` (`:319-338`) is sourced from the write buffer, from refill
`user` bits, or from `cap_tag_hit` — **never from a DRAM tag read**. A GDB read of `0xBC58CC63`
therefore has no path to the bit the load actually saw.

Worse, the documented desync runs in **exactly** the direction that would fool this measurement.
`wt_dcache_wbuffer.sv:604-620`: *"ctag is sampled TWICE — at TX ISSUE for DRAM and at TX RETURN for
L1 … `stc G` then `sd x0, G+0` gives DRAM ctag=1 and L1 ctag=0."* **Stale-high DRAM with correct-low
L1 is a known failure mode of this hardware, and the measurement sits inside it.** The usual
mitigation — that the two converge — does not obviously apply on a **wedged** core, since
convergence needs the entry to re-drain and whether that window is still open at read time is
unmeasured.

**Read that limitation, because a stronger claim was published from these same numbers and had to
be retracted.** The reasoning was: memory intact at the wedge ⇒ the memory path is innocent ⇒ the
fault is in operand delivery. That inference is invalid. **Every documented memory/load-path defect
on this core is transient and self-heals long before a debugger read seconds later** — write-buffer
residency, the issue/return desync (`wt_dcache_wbuffer.sv:612-619`), S-10b. So "the granule is
intact at T+seconds" is predicted by **both** arms of the fork and separates nothing.

The existence proof is on this very silicon: **S-10b (`c867dfcbb`) is ABSENT from the resident
bitstream** — verified, `git merge-base --is-ancestor c867dfcbb 84ed6eafb` returns false — and its
commit records a load returning `0x0000000000000000` while the store's data was in memory. A
load-path fault producing a clean zero with memory intact is exactly the combination the retracted
claim asserted was impossible.

**WHAT THE LOAD RETURNED HAS STILL NEVER BEEN MEASURED.** That is the open question, and both a
memory-path and a delivery-path explanation remain live.

---


> ## ⚠ STATUS, 2026-08-25 — READ BEFORE ANY SECTION BELOW
>
> **Every section of this file headed "THE MINIMAL REPRO REPRODUCES", "CONFIRMED … produces the
> S-12 SIGNATURE", "…still reproduces", "…and still faults", and "IT FAULTS ON ONE ITERATION" is
> RETRACTED.** All of them rest on arm 4, which read a subject slot it never wrote. See
> **"RETRACTION — every arm-4 measurement in this folder is VOID"** at the end of this file.
>
> **Net status: S-12 has NO minimal reproducer.** With the detector proven to fire (arm 2 reports
> `bad == REPS`), the real shape returns clean:
>
> | arm | shape | consumer | reps | result |
> |---|---|---|---|---|
> | 1 | subject store + 9 intervening stores | `lcc` (counts) | 512 | `0xC12A1000`, bad=0 |
> | 4 | **identical** | `cincoffsetimm` (**raises**) | 1 | `0xC12A4000`, no fault |
> | 4 | **identical** | `cincoffsetimm` (**raises**) | 512 | `0xC12A4000`, no fault |
>
> Arms 1 and 4 are a matched pair differing in exactly one thing — whether the consumer counts or
> raises — and both are clean, so the consumer is not the missing ingredient. The 40-line window
> does not reproduce S-12. That is a result, not an absence of one.
>
> **Also corrected:** the recorded finding "the stored value is REVOKE-typed" is **inverted** — it
> is **NONLIN**. Read the scope of that carefully, because it is narrower than it looks: it is
> about **the repro's `v`**, not about SQLite's value. NONLIN means the move-clear does not fire
> *in the repro*. It kills the move-clear account **for S-12** only if SQLite's stored value at
> the fault site is also NONLIN — **and that has never been measured.** Same caveat on "the five
> NONLIN simulations were in the right configuration": right only if production's value is NONLIN.
>
> **That makes SQLite's value type the discriminating unknown**, and the kernel's own line — `v`
> is "any tagged capability; its identity is irrelevant" — is now the weakest assumption in this
> folder, because type is exactly what gates the clear set at `load_unit.sv:225-226`. If SQLite's
> value is in that set, the repro never exercised the mechanism it was built to test, and an arm
> with a matching-type `v` is the first variant that *can* reproduce.

## The fault

A pure-capability SQLite domain running a two-table join wedges with
**`mcause 25` (UNEXPECTED_OPERAND)** at a fixed instruction.

    sqlite3WhereCodeOneLoopStart+0x8c :   cincoffsetimm a4, a4, 0xb0

`mcause 25` from the FLU means the rs1 operand arrived with `cap_type == NOT_CAP`
(`capstone_flu_unit.anvil:57-59` — that is `CINCOFFSETIMM`'s only `UNEXPECTED_OPERAND` guard).

**The producer is the FLU, not the commit stage.** The two producers are distinguished
arithmetically, not by inference: the commit-stage path sets `tval == mepc`
(`commit_stage.sv:225-226`, `:604`), and the observation is `tval = 0` with
`mepc = 0x828f4ba0`. Excluded.

## The instruction window (`sqrt.dom`, fn at VA `0x104b14`)

    +0x38  cincoffsetimm a0, s0, -0x70      a0 = the slot
    +0x40  stc  a2, 0x0(a0)                 <== THE SUBJECT STORE
           ... 9 stores, 5 of them stc, none to word 0 of the subject granule ...
    +0x7c  cincoffsetimm a5, s0, -0x120
    +0x80  movc a4, zero                    <== a4 := {cursor 0, NOT_CAP}   ** READ THIS TWICE **
    +0x84  stc  a4, 0x0(a5)
    +0x88  ldc  a4, 0x0(a0)                 <== the reload, same a0
    +0x8c  cincoffsetimm a4, a4, 0xb0       <== THE FAULT, rs1 = a4

`a0` is never rewritten between +0x38 and +0x8c and there are **zero branches or calls** in that
window, so no callee can have touched the slot.

## What was measured

**1. Software stored a real pointer, and nothing overwrote it.**

Store watchpoint (`CSR 0x811`) armed at the slot's physical address `0x82b9f360`, group 9 enabled
(`CSR 0x810 = 0x200`), both written by a `csrw` in the **host process** — outside the domain, zero
instructions added to any domain image. Full dump in `evidence/group9-watchpoint-dump.txt`:

    [ 19]  WATCHPOINT   PC=0x0000000828f4b54   DATA = 0x0000000082be4cf0
    [ 20]  WATCHPOINT   PC=0x0000000828f4b54   DATA = 0x0000000082be4cf0
    Total: 21 entries

`0x828f4b54` is `+0x40`, the subject store. The payload is the stored **cursor** — confirmed two
independent ways: by measurement (`verif/tests/custom/capstone/watchpoint-cursor.S` stores a
capability with cursor `0x80003030` and the payload reads `0x80003030`), and by source
(`st_commit_data_o` → `.data` → `lsu_ctrl.data`, while metadata rides the disjoint `user` lane,
`store_unit.sv:377`).

**21 entries against a 256-entry ring means it never wrapped** — so that is *every* committed store
to that word for the entire run, not a tail window. There is no later entry with `DATA = 0`.

**2. The consumer received zero.** `tval = 0` at the trap, and `tval` is the delivered rs1 cursor
(`ex_stage.sv:489`, same `fu_data_i[0]` the raising guard reads at `:797`).

**3. `tval = 0` also excludes plain S-07 tag loss.** A post-S-06 untagged `ldc` returns a verbatim
copy, so a de-tagged capability would have arrived as `{0x82be4cf0, NOT_CAP}` and `tval` would read
`0x82be4cf0`. It reads `0`. **The cursor bits themselves are zero, not merely untagged.**

## What this excludes, and what it does NOT

**EXCLUDED: a software NULL pointer.** The last committed store to that word wrote a live in-domain
pointer, and no committed store wrote it afterwards. Both surviving explanations are hardware.

**NOT ESTABLISHED: which hardware.** Two hypotheses remain and the evidence does not separate them:

| | prediction |
|---|---|
| **A. Memory path** — the slot returns zero on reload | matches `tval = 0` |
| **B. Operand delivery** — memory and load both correct, the consumer is handed the previous `a4` | **also** matches `tval = 0` |

**B is not speculative.** `movc a4, zero` at +0x80 makes the prior architectural value of `a4`
*exactly* `{cursor 0, NOT_CAP}` — byte-for-byte what the FLU received. And **R-20 is a
board-reproduced instance of precisely this class on this core**: *"memory is right, the load is
right, and only the consumer's operand is wrong."* R-20's fix is in the resident bitstream
(`issue_read_operands.sv:568`) but was **empirical and register-specific** (x10), not a proven
invariant; the general capability-forward path at `issue_read_operands.sv:674-677` is
register-agnostic.

**A retraction, recorded because it shaped the trail:** this was written up as "the reload returned
zero". That measurement was never taken. The instrument brackets
`[store commit → operand delivery]`, which contains writeback and forwarding, not only memory.

## The signature that points at B, and which sat unread for days

**Every software probe added inside the domain makes the fault disappear** — probed builds complete
~4/4, the un-probed build wedged 5/5 across several bitstreams. That is a **scheduling** signature,
and it is exactly what R-20 showed: cured by one nop on the board, four in simulation. It had been
recorded as a nuisance ("we cannot instrument this") rather than as evidence.

The fault is also **sporadic** — the same un-probed binary returned normally on a later draw — so
absence in any single boot is not evidence.

## The discriminator, for whoever picks this up

**Cheapest, no board:** an RVFI simulation of the exact four-instruction shape
(`movc a4,zero; stc a4,0(a5); ldc a4,0(a0); cincoffsetimm a4,a4,0xb0`) using the R-20 template at
`R20-stc-rs1-cursor-forward-x10/sim/r20-stc-ld-x10.S`. The RVFI trace prints the load's **retired
value** and the consumer's **operand** side by side — which no board instrument can.

**On the board:** read the 16 bytes at the slot and its shadow tag byte over GDB at the wedge.
Memory intact and tagged ⇒ **B**, and the memory-path reading must be dropped. Zeros ⇒ **A**.

## Blind spots that travel with the group-9 result

State these with the measurement, because each makes an *absence* mean less than it looks:

* The watchpoint compare is **word-granular** (`cva6.sv:904-906`, `st_commit_paddr[PLEN-1:3]`), so
  stores to `G+8..G+15` are invisible. The claim is about **word 0** only — which is where the
  cursor lives, so it is the word that matters, but the distinction is real.
* **AMOs are excluded** at `commit_stage.sv:339`.
* **Domain-switch stores bypass the speculative queue** the tap reads.
* The tap is **upstream of the write buffer and D-cache**, so S-07/S-09/S-10-class corruption is
  invisible to it *by construction* — this instrument cannot see the very defects it sits next to.
* Group 9 carries **no tag bit** (`tracer.sv:237-239`), so the stored value's *validity* is
  unmeasured. It is a cursor, not "a valid capability".

## Reproduction

    binary   sqrt.dom, sha ee9a9a86ed12f06b, built by benchmarks/sqlite/build-sqlite-silicon.sh
             (UN-probed: any added in-domain probe removes the fault)
    input    benchmarks/sqlite/slt/q_two.test -- SELECT t1.a FROM t1, t1 AS y over an EMPTY table
             (q_one.test is the matched pair: identical but for the second table reference)
    driver   tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py
             SQLITE_STAGE_DOMS="/test-domains/sqbase.dom,/test-domains/sqrt.dom:--slt /test-domains/q_two.test"
             WEDGE_TRACER=1  (arming is compiled into the host: CAPSTONE_TRACE_ARM / CAPSTONE_TRACE_WP)

Both `.test` files declare `----\n0` against an empty table, so both report one query failure on
**every** platform including native — that is the test file's own authoring bug and not a result.
Confirmed byte-identical against the native oracle.

Run a known-good control first in the same boot: a boot whose control fails carries no verdict, and
the control is also what proved the ring flood is ordinary trap traffic rather than a wedge spin.

---

## The minimal repro does NOT reproduce — with the detector proven to fire

Board run, `caplifive_s07clear_84ed6eafb.bit`:

| rung | retval | oracle | verdict |
|---|---|---|---|
| `k800` (known-good control) | 4 | 4 | **OK** — the boot carries verdicts |
| `s12a2` (must-fail control) | `0xC12A2200` | `0xC12A2200` | **FIRED** |
| `s12a1` (the window) | `0xC12A1000` | — | **bad = 0** |
| `s12a3` (window + one nop) | `0xC12A3000` | — | **bad = 0** |

**Arm 2 matched its oracle exactly**, so the detector demonstrably reports a bad reload, and arm
1's zero is a measured zero rather than a silent one. `0xC12A2200` decodes as arm 2 with
`bad = 512 = S12_REPS` — every iteration's reload came back NOT_CAP, which is *correct*
architecture (a plain store clears the granule tag) and exists only to prove the instrument.

**So the instruction window, executed inside a real capability domain — after `capenter`, through
a cap table, on a monitor-carved stack — does not fault.** That closes the largest fidelity gap
the five bare-metal simulations left open, and closes it negatively.

**This is NOT an exoneration, and the kernel said so before the run.** Two deviations from the
production shape remain, both recorded in `src/s12_kernel.h` in advance:

1. the consumer is `lcc` selector 1 (**total**, does not raise) rather than the production
   `cincoffsetimm` (**raises**) — so the exact consuming instruction differs;
2. this is a tight loop, where the board's window sits at a deterministic depth in a long call
   chain, executed twice.

The pre-written next step therefore stands: **move the consumer back to `cincoffsetimm` and accept
wedging arms.** That trades the rate for one bit per boot, which is why it was not the first
version — but a clean arm 1 with a non-raising consumer cannot rule out a fault that only the
raising consumer exposes.

---

## THE MINIMAL REPRO REPRODUCES — 130 lines instead of all of SQLite

| rung | result | meaning |
|---|---|---|
| `k800` (known-good control) | RETURNED, `retval=4` | the boot carries verdicts |
| `s12a4sent` (SENTINEL) | RETURNED, `retval=0xC12A4E17` | **glue, cap-init, entry and return all work** |
| `s12a4d2` (the window) | **NO RETURN** | the silence is attributable to the **BODY** |
| `s12a1` (same body, non-raising consumer) | RETURNED, `bad=0` | — |
| `s12a2` (must-fail control) | RETURNED at its exact oracle | the detector is proven |

**Three draws of arm 4** — `s12a4`, `s12a4d1`, `s12a4d2`, distinct images with a byte-identical
window — fell silent **3/3**. The sentinel, built from the same source with the body skipped,
returned in 599 cycles.

### Why the sentinel is the load-bearing arm

`SHA5`-last on the `lpc` path is **ambiguous by construction**: that controller emits no
post-entry marker, so it cannot distinguish *"the body wedged"* from *"the domain never ran"* —
`locagg_kernel.h:34-36` says exactly this, and the tree records every lpc-hosted domain dying in
share #1 on 2026-08-06 regardless of content. An earlier version of this note read `SHA5`-last as
an **R-16 entry stall**; that was unsupported and is **retracted**. Two draws falling silent at the
same point was itself evidence against R-16, which is per-image and should have shaken loose on a
redraw.

The sentinel removes the ambiguity: it returns, so entry works.

### The matched pair, and the part that does not yet add up

Arm 1 and arm 4 differ by **exactly one thing** — the consuming instruction:

* arm 1 consumes with `lcc` selector 1 (total, non-raising) → **returns, `bad = 0`**
* arm 4 consumes with `cincoffsetimm a4,a4,0xb0` (raises on NOT_CAP), verified byte-identical to
  the board's faulting instruction (`5b 27 07 0b`) → **no return, 3/3**

**But `bad = 0` in arm 1 means the reload was never NOT_CAP there.** And `CINCOFFSETIMM` raises
*only* on `cap_type == NOT_CAP` — it has no bounds arm (`capstone_flu_unit.anvil:57-70`). So the
two results cannot both be about the reload's value unless **the consumer changes the timing**.

That is not a contradiction so much as the signature this whole investigation has been chasing:
the fault is scheduling-sensitive, and *every added instruction makes it disappear*. `lcc` and
`cincoffsetimm` are different instructions with different timing, so arm 1 is not a control for
arm 4's value — it is a different experiment. **Stated here rather than resolved**, because the
alternative readings are still open:

* arm 4 wedges at the `cincoffsetimm` on a NOT_CAP operand that arm 1's timing never produces;
* arm 4 wedges somewhere else in the body for a reason the non-raising build tolerates;
* arm 4 fails for a reason unrelated to the window.

**The trap cause and `mepc` have NOT been read for arm 4.** Until they are, "it reproduces the
S-12 signature" is unproven — what is proven is that a 130-line domain, with the production
consumer, does not return, while the same domain minus its body does.

## CONFIRMED: the 130-line repro produces the S-12 SIGNATURE, not merely a hang

The prediction was written down **before** the measurement (`git log` proves the ordering): if this
is S-12, `mepc` must read `DBAS + 0x41c` and `mcause` must be 25.

    sw=255 TRAP LOG {seen, mcause[6:0]} = 0x99   ->  seen=1, mcause = 0x19 = 25
    mepc (latched)                      = 0x819e041c
    the wedged arm's own DBAS           = 0x819E0000
    mepc - DBAS                         = 0x41c   ->  VA 0x1041c

`VA 0x1041c` is `cincoffsetimm a4, a4, 0xb0`, encoded `5b 27 07 0b` — **byte-identical to the
instruction that faults in SQLite**, same registers, same displacement.

**Same cause. Same instruction. Same encoding. In 130 lines instead of 2.2 MB of SQLite, and in
599 cycles instead of minutes.**

### What this changes

* **The reproducer is no longer SQLite.** Any future experiment — including the pending bitstream
  — can use `s12a4`, which is small, fast, and whose entire source is in `src/s12_kernel.h`.
* **The sentinel makes it attributable.** `s12a4sent` (same build, body skipped) returns cleanly,
  so this is the body faulting and not an entry stall.
* **It is not sporadic in the same way.** Three distinct draws with a byte-identical window
  wedged 3/3, where the SQLite repro fires roughly 2 in 3. A deterministic repro is a much better
  vehicle than a sporadic one.

### What it still does NOT establish

The **mechanism**. This is the same fault, reproduced small — it is not an explanation of it.
Every named mechanism remains excluded (software NULL, S-07 tag loss, R-20 forwarding class,
wrong-producer scoreboard selection, adjacent-granule scalar stores, write-buffer depth,
domain-switch `cnull`, load-syncer mispair), and **what the load returned is still the unmeasured
fact**. The pending bitstream is still what answers it — this just makes the run cheap.

Also still open, and recorded rather than resolved: arm 1 (same body, non-raising `lcc` consumer)
returns `bad = 0`, so *its* reload was never NOT_CAP, while `CINCOFFSETIMM` raises only on
NOT_CAP. Both cannot be about the reload's value unless the consumer changes the timing — which is
consistent with the scheduling sensitivity seen throughout, but is not itself established.

## The matched-pair vehicle: page-aligned, one slot address, still reproduces

The pending experiment arms **one** watchpoint address and runs both consumers against it — the
raising `cincoffsetimm` arm and the non-raising `lcc` arm — so the loads can be compared with the
consumer as the only variable. That requires both arms to place the slot at the **same** address,
and at 16-byte alignment they did not:

    lcc arm            slot VA = 0x11750
    cincoffsetimm arm  slot VA = 0x11790     <- 64 bytes apart

The bodies differ in size and shift everything after them. One armed address could not have served
both, and arming per-arm would reintroduce the **wrong-allocation failure class** the static array
exists to remove — the one that fired for real on an earlier arm and produced a clean-looking
empty record.

**Fixed by page-aligning `s12_frame`**; 4096 swamps the inter-arm drift. Verified, not assumed:

    s12p1 (lcc)            slot VA = 0x12690
    s12p4 (cincoffsetimm)  slot VA = 0x12690     <- ONE address serves both

### And the aligned build still faults, which had to be checked

This fault is layout-sensitive — every added instruction has made it vanish — and page-aligning
moved everything after the frame. So the aligned build could easily have stopped reproducing,
which would have made it useless as a vehicle:

    k800   RETURNED  retval=4            control valid
    s12p1  RETURNED  retval=0xC12A1000   matches its oracle, bad = 0
    s12p4  NO RETURN                     still wedges

### Constants for the run

    watchpoint paddr  = DBAS + 0x2690        (slot VA 0x12690, 16-byte aligned -> granule base)
    expected mepc     = DBAS + 0x41c         (consumer VA 0x1041c, bytes 5b 27 07 0b)
    expected mcause   = 25 (UNEXPECTED_OPERAND)
    arm 1 returns     = 0xC12A1000           arm 4 returns 0xC12A4000 or wedges

Both derived from the ELF plus DBAS, with **no dependence on call depth** and no scraping from a
previous wedge.

## The vehicle is complete: the domain arms its own watchpoint, and still faults

Page-aligning the frame made the **virtual** slot address identical across arms. The watchpoint
compares a **physical** address, and `DBAS` is per-**ARM**, not per-boot — measured: one boot's
arms took `0x82400000` and `0x82800000`. So a single `csrw 0x811` before the boot covers at most
one arm of the matched pair, and the uncovered arm returns an empty record, which the decision
table reads as *"the load was fine"*.

**The host cannot fix it** — `create_dom` returns only a domain id, and `DBAS` is printed by the
monitor, so userspace never sees it. **The domain does not need it.** Capabilities on this design
carry PHYSICAL addresses (the trap `mepc` is physical; the monitor's `BASE:` trace is physical),
so `lcc` **selector 2** — the cursor query (`capstone_dyn_unit.anvil`, `64'd2 => cap.cursor`) — on
a capability pointing at the slot returns the slot's physical address directly:

    lcc  rd, &s12_frame[0x700-0x70], 2     -> the slot's PHYSICAL address
    csrw 0x811, rd                          -> arm the granule filter
    csrr rd, 0x811                          -> readback into a sink: a failed arm is VISIBLE
    csrw 0x810, 0x200                       -> group 9, so the arming is PROVEN this boot

No `DBAS`, no host, no race with the driver, and **each arm arms its own address by
construction** — which is the safe form of per-arm arming, as distinct from scraping an address
from a previous wedge and reusing it.

### And it still faults, which had to be measured rather than assumed

The arming code shifts layout, and this fault vanishes whenever anything shifts. Page-alignment
survived, but that is evidence about a *different* change and does not transfer:

    k800   RETURNED  retval=4            control valid
    s12w1  RETURNED  retval=0xC12A1000   matches oracle, bad = 0
    s12w4  NO RETURN                     still wedges

Verified in the artifact too: both arms share slot VA `0x12690` and contain exactly one
`csrw 0x811`.

### Build recipe for the pending bitstream run

    arm 1 (non-raising, lcc):    -DS12_ARM=1 -DS12_SELF_ARM_WP=1     expect 0xC12A1000
    arm 4 (production consumer): -DS12_ARM=4 -DS12_SELF_ARM_WP=1     expect a WEDGE
    control:                      k800, which must return 4 first

### A WEDGE IS NOT ENOUGH: it must be at the RIGHT INSTRUCTION

`lcc` selector 2 is **not total** — only selector 1 is (`capstone_dyn_unit.anvil:195` excludes
`zimm != 1` from the NOT_CAP guard). Every other selector RAISES on a NOT_CAP operand, so the
**arming `lcc` can itself wedge the domain during setup**, before the body ever runs. From
outside, that is indistinguishable from the subject fault: both are "no return".

The two candidate sites in the self-arming build:

    arming lcc         VA 0x103bc   DBAS + 0x3bc   db 96 26 08   lcc a3, a3, 0x2
    subject consumer   VA 0x10484   DBAS + 0x484   5b 27 07 0b   cincoffsetimm a4, a4, 0xb0

**PRECONDITION: a wedge whose latched `mepc` is not `DBAS + 0x484` is NOT the subject fault.**
It is an arming failure or something else, and that arm is VOID.

**MEASURED, on the build that will actually run:**

    sw=255 TRAP LOG = 0x99          ->  seen=1, mcause = 25
    mepc            = 0x819e0484
    the arm's DBAS  = 0x819E0000
    mepc - DBAS     = 0x484          ->  THE SUBJECT CONSUMER. The arming lcc did not raise.

### The constant this replaces, and why that matters

An earlier note committed `expected mepc = DBAS + 0x41c`. **That was for the pre-self-arming
build**: adding the arming code shifted the consumer from `+0x41c` to `+0x484`. Using the stale
value as the precondition would have declared a perfectly good wedge VOID and thrown the real
result away — a constant that was true when written, silently false after a change, and
confidently wrong at the point of use. **Re-derive both site addresses from the artifact whenever
the build changes**; they are not properties of the bug.

Both arms carry group 9, including the one that returns cleanly — an empty LDC record means
*"the load was fine"* only where group 9 has fired at the subject store on that same boot.

## Group 9 FIRES in this vehicle — the precondition of the row table, measured

The pending decision table reads an empty LDC record as *"the load was fine"* **only where group 9
has fired at the subject store on the same boot**. Nobody had ever confirmed group 9 fires in this
vehicle at all — and discovering it does not, after a reflash, would have cost the whole cycle.

Measured on the CURRENT bitstream, arm 1 (the arm that returns), self-arming:

    256 entries, ONE distinct PC:   0x819e043c  ->  VA 0x1043c  =  stc a2, 0x0(a3)   the SUBJECT store
    256 entries, ONE distinct DATA: 0x819ff760                                        the stored cursor

Group 9 fires **only** on stores to the armed address, so its firing is itself the proof that the
domain's self-arm computed the correct physical address — no separate check needed, and none
possible to forget. A single PC and a single DATA value across all 256 entries is exactly what the
loop should produce, and rules out the record having been filled by unrelated traffic.

**So the row table is usable**: an empty LDC record will be distinguishable from a mis-armed
filter, which is the property the whole matched-pair reading rests on.

### A driver message that would have inverted this result

The driver announced *"NO `SQ: tracearm=` ... Arming UNMEASURED; treat any dump as
uninterpretable"* — because that marker is printed by the SQLite **host**, and a ladder domain
arms **itself** by design, so the marker is absent for a correct run. Taken at face value it would
have discarded a fully-armed 256-entry dump. Scoped to say the opposite for this path: group 9
firing *is* the proof, and an **empty** group 9 is what carries no verdict.

Same class as the stale `mepc` constant: text that was true where it was written, wrong where it
was read, and pointing toward "nothing to see" in both cases.

## Why the CURRENT bitstream cannot answer this for ANY workload — measured, not argued

It was worth one boot to ask whether the first-wins LDC recorder becomes usable on the small
vehicle. The reasoning was that SQLite issues thousands of legitimately-untagged loads before the
subject and consumes the slot, whereas `s12` has **14 static LDCs**. With the clear applied
immediately before the arm, the first untagged LDC had a real chance of *being* the subject.

**It is not.** Clear confirmed running before both arms; selftest firing, so a zero would be a
controlled negative:

    subject slot paddr        0x819e2690   -> recorder should read granule 0xe2690
    recorder actually read                     0x82280

**And `0x82280` is the same value the SQLite runs recorded — on a domain 6 MB away.** A record
that does not move when the workload moves is not the workload's record.

**The reason generalises, and that is the useful part.** The monitor's trap entry issues
`LDC(gp, sp, -16)` on **every timer tick**, so between the clear and any domain instruction at
least one tick almost certainly lands and takes the first-wins slot. That is independent of how
small the domain is, so **no workload can win this race** — shrinking the vehicle was the wrong
lever.

**This independently justifies the granule filter in the pending bitstream.** The filter is
precisely what makes the recorder immune to the monitor's traps: it is not a convenience, it is
the thing that makes the record attributable at all. Rolling-without-filter would still be
consumed by trap traffic; filtered-without-rolling would still be first-wins. Both halves are
load-bearing, which is worth knowing before anyone considers dropping one under synthesis
pressure.

---

# THE STORED VALUE IS **REVOKE**-TYPED — so the LDC MOVE-CLEAR FIRES, and every earlier sim was blind to it

Measured on the board with a type-probe arm: `retval 0xC12A5200` → `lcc` selector 1 returned **2**,
and that selector reports `cap_type - 1` (`capstone_dyn_unit.anvil`, `64'd1 => cap.metadata.cap_type - 3'd1`),
so the raw type is **3 = `CAP_TYPE_REVOKE`**.

**`REVOKE` is in the LDC move-clear set.** `load_unit.sv:225-226` fires the clear for
`{LINEAR, REVOKE, UNINIT, SEALED, SEALEDRET}` with write permission. So on silicon, **every
iteration's reload also WRITES the source granule** — and the payload it writes is
`store_unit.sv:462-469`:

    .data_i  (load_unit_clear_i ? '0   : st_data_q)      cursor   = 0
    .user_i  (load_unit_clear_i ? '0   : st_user_q)      metadata = 0
    .ctag_i  (load_unit_clear_i ? 1'b0 : st_ctag_q)      tag      = 0

which is **bit-for-bit `create_cnull()`** — and therefore bit-for-bit the operand observed at the
fault, `tval = 0` included. This is the first candidate that produces the exact value as a
*designed payload* rather than as corruption.

## Why this invalidates five earlier "clean" simulation results

**Every one of the five directed sims set `CAP_TYPE_NONLIN`**, which `load_unit.sv:225-226`
explicitly excludes. The clear **never fired in any of them**. Their clean results were not
evidence about this mechanism — they were testing a configuration in which it is structurally
absent, and I read them as exclusions.

That is the same failure as the rest of this investigation, in its most expensive form yet: not a
broken instrument, but a **correct instrument pointed at the wrong configuration**, five times,
with three of them carrying proven-firing positive controls that made them look rigorous.

## What the clear implies about the traffic

With the clear firing, each iteration issues **two** writes to the subject granule — the `stc` and
the reload's clear — into a write buffer that is 8 entries deep and which this kernel already
overflows (≥16 dirty words per iteration). If a merge ever lands the clear *after* the following
iteration's `stc`, the granule reads back as the clear payload: a null.

## Status: NOT established, and one attempt already failed

A directed sim using `CAP_TYPE_LIN` **failed for an unrelated reason** — `MOVC` of a linear-class
capability nulls its source, so copying it around violates linearity and `CINCOFFSET` raised
`UNEXPECTED_OPERAND` on a legitimately-nulled operand. Correct architecture, invalid test.

**And the simple version of this account is already excluded by measurement:** if the domain's own
`v` were being nulled by move semantics, the store watchpoint would have recorded it. It recorded
**256 stores, one PC, one non-zero DATA value** — `v` is stable across every iteration. So whatever
happens, it is not the program nulling its own capability.

The next test must respect linearity: a fresh clear-class capability per iteration, or `REVOKE`
specifically (which is what the board actually has), rather than copying one around.

## MEASURED: the LDC move-clear fires on EVERY iteration (512/512)

Arm 6 reloads the same slot **twice** and asks whether the second reload comes back NOT_CAP. If
the clear fires, the first reload zeroed the granule and the second must find nothing.

    control k800   retval 4, oracle 4      OK -- the boot carries verdicts
    s12c           retval 0xC12A6200       arm 6, count = 512 of 512

**Every reload zeroes its own source granule.** That is correct architecture — a move — and it is
now measured rather than inferred from the type. Combined with the type probe (REVOKE, in the
clear set at `load_unit.sv:225-226`) and the clear's payload (`store_unit.sv:462-469`: cursor 0,
metadata 0, tag 0 = `create_cnull()`), the picture on silicon is:

    per iteration:  stc  writes the capability to the slot
                    ...9 intervening stores to other granules...
                    ldc  reads it back AND writes create_cnull into the SAME granule

So **two writes per iteration target the subject granule, from two different agents**, into a
write buffer that is 8 entries deep and that this kernel drives past capacity every iteration
(>=16 dirty words). The buffer's own comments (`wt_dcache_wbuffer.sv:611-619`) document a
merge-ordering residual of exactly this class, for a different instruction pair.

### The hypothesis this makes concrete — and it is a HYPOTHESIS, not a measurement

If iteration N's **clear** ever merges into the buffer *after* iteration N+1's **store**, then
iteration N+1's reload reads the clear's payload instead of its own store's value — and the
payload is `{cursor 0, NOT_CAP}`, which is the operand observed at the fault, `tval = 0` included.

**What is measured:** the clear fires every iteration; the value is REVOKE-typed; the clear's
payload is bit-for-bit `create_cnull`; the store writes a stable non-zero value 256/256.
**What is NOT measured:** that the two writes ever land out of order. Nobody has observed a
reordering; this is a mechanism that *would* produce the observed value, on hardware where both
writes demonstrably exist.

### Why this reopens the memory-path arm

The delivery account (a stale FLU operand read before the load lands) and this one both predict
`{cursor 0, NOT_CAP}`. They differ in **where** the null comes from: the register file versus the
granule. RTL's pending recorder distinguishes them directly — it reports whether the LOAD returned
untagged. That is now a fork between two *specific* mechanisms rather than between two vague
halves of the pipeline.

Note also what arm 1 already says: with the non-raising consumer, the FIRST reload was never
NOT_CAP across 512 iterations. So on that build no reordering occurred — consistent with the
DYN-serialisation difference, and consistent with the reordering being timing-gated.

## The GRANULE row is excluded — structurally, and without a boot

The reordering hypothesis above needs iteration N's clear to merge into the write buffer *after*
iteration N+1's store. **There is no path for that**, and the reason is not the commit gate alone:

    store_unit.sv:449    .valid_i (store_buffer_valid || load_unit_clear_i)
    store_buffer.sv      monotonic +1 pointers on BOTH queues -- a strict FIFO
    load_unit.sv:707-712 valid_o = 0 while the clear has not been accepted

**The clear uses the SAME store-buffer port as ordinary stores**, and that buffer is a strict
FIFO. The gate guarantees the clear is *enqueued* before the LDC commits; in-order commit puts the
next iteration's `stc` later; one FIFO then delivers them to the write buffer **in program order**.
Same granule and same word means they merge into one entry, where `ctag` is last-writer-wins and
the merge order *is* arrival order. The different-word case is covered by the `gran_hazard` stall
for the same reason, and if the clear has already drained there is no merge at all — but it still
landed first.

All three premises verified at the resident revision rather than taken from the report.

**So the fork narrows to one named mechanism**: the null comes from the **register file**, not the
granule — a stale FLU operand read before the load's writeback lands.

### What that does to the pending bitstream's reading

RTL's recorder reports whether the **load** returned untagged. Under the surviving account it
should come back **EMPTY on the faulting arm** — and that is now a **PREDICTION with a named
mechanism behind it**, not a default.

That distinction matters. An expected empty is the result nobody scrutinises; an empty that one
mechanism predicts while the competing mechanism — which predicted the opposite — has just been
excluded on structure, is a result. **A NON-empty record would mean the structural argument above
is wrong**, and that is the more interesting outcome of the two.

And it raises the stakes on the group-9 arming proof: with empty as the predicted answer, group 9
firing is the only thing separating *"empty because the load was fine"* from *"empty because the
filter never armed"*.

## And the REGISTER-FILE row takes a hit too: the RAW check HELD, with the window proven created

`s12-flu-raw.S` builds the window the surviving account needs — a strided, cache-missing `ldc`
followed **immediately** by an FLU consumer of its destination, every granule pre-seeded with a
real capability so a NOT_CAP operand can only come from the pipeline:

    RAW-DBG: ALIVE
    flu-issues = 131    ldc-pending-cycles = 82    HAZARDS = 0

**The condition was created** — 82 cycles with an LDC outstanding, 131 FLU issues alongside — and
**not once** did an FLU op get acked for issue while an LDC writing its `rs1` was still pending.
The generic RAW machinery stalled it every time.

This is a real negative rather than the earlier void: the previous run reported
`ldc-pending-cycles = 0` because its loads all hit, so nothing was tested. The totals are what make
this zero admissible.

### So both named mechanisms now have negative evidence

    granule (clear reordered after the next store)   EXCLUDED structurally -- one FIFO, program order
    register file (stale FLU operand)                RAW check HELD in sim, window proven created

**The fault is not in doubt** — it is deterministic on silicon, at a known instruction, with a
known cause. What is in doubt is every mechanism either lane has named for it.

**Caveat that keeps the register-file row alive, and it is not a small one:** this is simulation.
The RAW check holding here does not prove it holds on silicon, and the fault has never reproduced
in simulation at all. A hazard that is correct in RTL and marginal after synthesis would look
exactly like this. So the row is **weakened, not excluded** — and I am not recording it as an
exclusion for that reason.

### What this does to the pending bitstream

It makes it **more** valuable, not less. Both candidate mechanisms now predict outcomes that
disagree with a measurement, so the recorder is no longer confirming a favoured story — it is the
only thing that can say where the null comes from **without** a hypothesis in hand. An empty record
is no longer "as predicted": it is one of two informative answers, and the non-empty one would
resurrect the granule row against a structural argument, which would be the more interesting result.

## IT FAULTS ON **ONE** ITERATION — accumulation is dead, and so is the last granule story

Sweeping the loop count with the production consumer, ascending, control first:

    k800    RETURNED  retval=4
    s12r1   NO RETURN                <== S12_REPS = 1

Confirmed the same fault, not a different one — the discipline that the `SHA5` retraction bought:

    TRAP LOG 0x99            ->  seen = 1, mcause = 25
    mepc                     =   0x819e0460
    the arm's own DBAS       =   0x819E0000
    mepc - DBAS              =   0x460      -> the SUBJECT consumer
    (the arming lcc sits at   0x39c, so the two are distinguishable and it was not that)

### What one iteration excludes

* **Accumulation across iterations** — there are no other iterations. Any account needing state
  to build up over repetitions is dead: rev-node consumption, cache-set rotation, write-buffer
  phase advancing over a loop.
* **The move-clear as a source of the null.** With `REPS = 1` there is no *previous* iteration
  whose clear could be read back, and the reload's own clear happens at the reload — it cannot
  precede it. The granule story was already excluded structurally by the FIFO argument; this
  closes it independently, from the other end.
* **Any "low per-iteration probability" reading of the earlier threshold sweep.** That caveat was
  raised before this ran and it was the right one to raise — but a single iteration faulting
  deterministically leaves no room for it.

### What survives

One pass: store a capability to the slot, nine intervening stores, `movc a4, zero`, reload,
consume. The consumer receives `{cursor 0, NOT_CAP}` — and `movc a4, zero`, **in this same pass**,
is the only thing that put exactly that value anywhere.

So the **stale-operand account is now the only named mechanism that works at `REPS = 1`.** It also
remains the one the RAW-hazard simulation says should not happen — `HAZARDS = 0` with the window
proven created. Those two facts are in direct tension, and the tension is the finding: either the
RTL check has a gap the directed test did not construct, or it is marginal after synthesis in a
way simulation cannot show.

### The repro is now smaller again

`-DS12_ARM=4 -DS12_SELF_ARM_WP=1 -DS12_REPS=1` — no loop, roughly twenty instructions of body,
faulting deterministically at a known VA with a known cause. That is the artifact to hand anyone,
and it is what the pending bitstream should be pointed at.

---

# CORRECTION: the subject slot is a COMPILER stack spill, not the kernel's static array

The watchpoint address recorded above — `DBAS + 0x2690`, the slot inside `s12_frame` — is **the
wrong granule**. The fault is not on the array the kernel stores to; it is on a spill the compiler
inserts afterwards.

    0x10450  ldc a4, 0x0(a0)              load from the kernel's frame slot
    0x10454  cincoffsetimm a3, s0, -0xc0  a3 = a COMPILER stack slot
    0x10458  stc a4, 0x0(a3)              <- the REAL spill
    0x1045c  ldc a4, 0x0(a3)              <- the REAL reload
    0x10460  cincoffsetimm a4, a4, 0xb0   <- THE FAULT

**Corrected address, and it is corroborated independently rather than merely recomputed:**

    s0 at the wedge   = 0x819ff670
    subject slot      = s0 - 0xc0 = 0x819FF5B0     (== sp; 16-byte aligned, so a granule base)
    low 20 bits       = 0xFF5B0
    s07 STC recorder  = 0xFF5B0                    <- EXACT MATCH

**The STC recorder had been reporting the subject spill all along and I read past it**, because I
was checking it against the address I expected rather than asking what it was pointing at.

## How this happened, and why it is worse than a slip

For SQLite I had this right — the pair was `stc a2, 0x0(a0)` / `ldc a4, 0x0(a0)` with
`a0 = s0 - 0x70`, plainly a stack slot. When I built the minimal repro I **assumed my own static
array would be the subject**, because I had written the store myself. The compiler still spills
`a4` to its own frame slot, and the fault is on reloading *that*. The kernel's `*slot = v` is a
different, earlier store.

Everything downstream inherited the assumption: the page-alignment work, the `lcc`-selector-2
self-arming, the "one address serves both arms" property, and the constants handed over for the
pending bitstream run. **The repro itself is unaffected** — it faults deterministically at a known
VA with a known cause — but every *measurement aimed at a granule* was aimed at the wrong one.

**And it failed in the expensive direction:** an armed address the fault never touches yields an
empty record, and this folder's own decision table reads empty as *"the load was fine, the fault is
in delivery"*. It would have produced a confident wrong answer rather than a null one.

## Consequences for the earlier results

* **The group-9 measurement (256 stores, one PC, one DATA)** was real but was watching the
  kernel's array store, **not** the subject spill. It still proves the self-arm mechanism works and
  that group 9 fires; it says nothing about the subject granule.
* **The `a4 = 0` reading needs re-examination**, since the reload feeding the fault comes from `a3`
  (the stack slot), not `a0`. The precondition held and the value is real; what it means is not yet
  settled and is deliberately not being claimed here.
* **The corrected address is dynamic** — `s0 - 0xc0` — which reintroduces the hazard the static
  array was chosen to remove. `0xc0` is a fixed offset per build and can be re-derived from the
  artifact; only `s0` needs measuring.

---

# THE NULL PREDATES THE SPILL: the subject is one instruction EARLIER

With the watchpoint finally armed at the real subject slot — computed at runtime from `s0`, and
cross-checked against hardware — group 9 fired **exactly once**, at the spill, writing **zero**.

    0x1045c  ldc a4, 0x0(a0)              <- the null enters HERE
    0x10460  cincoffsetimm a3, s0, -0xb0
    0x10464  stc a4, 0x0(a3)              <- THE SPILL.  group 9: 1 entry, DATA = 0x0
    0x10468  ldc a4, 0x0(a3)              <- reload reads the zero back
    0x1046c  cincoffsetimm a4, a4, 0xb0   <- FAULT (mepc - DBAS = 0x46c, mcause 25)

`a4` was **already zero when the spill executed**. So the spill/reload pair is faithful — it stores
a null and returns the same null — and the whole investigation has been instrumenting **one
instruction too late**. The subject is `ldc a4, 0x0(a0)` at `0x1045c`, or whatever put a null in
the memory it reads.

**The cross-check is what makes this readable:** the s07 STC recorder independently reported
granule `0xFF5C0`, matching the armed slot `s0 - 0xb0`. Hardware agreed the right address was
armed, which is exactly the check that caught the previous wrong granule.

## What this retires

* **The spill/reload pair as the site.** It is doing its job correctly. Every measurement aimed at
  it — including the group-9 arming work and the corrected address — was aimed one step past the
  defect.
* **Both remaining accounts, as stated.** The stale-operand story was about the consumer reading
  around the *reload*; the memory story was about the *reload's* granule. Neither survives the null
  already being in the register before either happens.

## How the address problem was finally solved, since it will recur

The subject slot is a compiler spill at `s0 - OFF`, and **the compiler chooses OFF**: measured
`0xc0`, then `0xa0`, then `0xb0` across three builds, because adding the arming code moves the
frame. A hardcoded physical address is therefore stale the moment anything is rebuilt — and arming
it *is* a rebuild. That is circular.

The fix is to compute it at runtime: `s0` is a capability register, so the domain reads it, offsets
it, and queries its cursor for the physical address. Only `OFF` remains a build constant, and being
an immediate it does not change code size — verified by convergence, `0xb0` on two successive
builds. No DBAS, no host, no measured address.

## Recurrence worth noting

The `mepc` predicted before this run (`DBAS + 0x458`) was taken from the **previous** build's
artifact, not the one that ran (`0x46c`). The same stale-constant mistake as `0x41c`, one day and
one explicit fix later. The precondition still held — `0x46c` **is** the consumer in the build that
ran, confirmed by re-deriving from the correct artifact — but it was confirmed after the fact
rather than predicted, and that is the weaker form.

## Pre-registered reading of the one-level-up measurement (written BEFORE the dump)

The group-9 watchpoint is armed one level up, on the store at `s0 - 0xa0`:

```
0x10440  ldc  a1, 0(a0)     the REAL slot read
0x10448  stc  a1, 0(a0)  <- ARMED HERE
0x1045c  ldc  a4, 0(a0)     reads it back -> ZERO
0x1046c  cincoffsetimm a4   <- FAULTS (mcause 25, UNEXPECTED_OPERAND)
```

**The entry COUNT discriminates, not just the payload.** The armed slot holds a live REVOKE
capability, and the reload at `0x1045c` is exactly the LDC shape that fires the move-clear
(`load_unit.sv:225-226`), whose payload — cursor 0, metadata 0, ctag 0 (`store_unit.sv:462-469`)
— is written back to the **same granule** through the shared store port (`store_unit.sv:449`).
So the move-clear can post a *second* group-9 entry, at the same address, reading zero. Three
outcomes, all registered in advance:

| group 9 | reading |
|---|---|
| 1 entry, payload ≠ 0 | the store wrote a real capability and the **load** returned NOT_CAP — the move-clear never fired because the load never saw a cap type. A genuine load-path defect, and the repro vehicle is sound. |
| 1 entry, payload = 0 | `a1` was **already null** upstream; the subject moves up again to the `ldc` at `0x10440`. |
| **2 entries (≠ 0, then 0)** | the store wrote the cap **and** the load *saw* it (move-clear fired) — yet the consumer still faulted. The null then enters **after** the load, in writeback/forwarding, not in memory and not in the load path. This is the most informative signature available and must not be read as "the store wrote zero", nor discarded as noise. |
| empty | uninterpretable — says nothing about the subject. See the instrument caveat below. |

**If it comes back empty, suspect the instrument before the subject.** This is the first boot
combining `WEDGE_S07_CLEAR_PERARM=1` with `WEDGE_TRACER=1`, and the per-arm clear used to restore
the switches to `0` rather than to the tracer's resting value — parking `sw[2]`, the ring-mode
input, low for every arm after the first clear. Read the driver's own `switch_state` events for
the value in force while arm 2 ran before drawing any conclusion from an empty dump. (Fixed in
the driver as of this run's revision, but the running process loaded the older copy.)

## Two boots lost to a fail-open diagnostic — recorded because the class recurs

Two consecutive boots reported the **control** (`k800`, which returns in ~2 s) as NO-RETURN, which
looks like a board or firmware failure. It was neither. Both logs carry the same line:

```
[s07] early halt control failed (ActionTimeout) -- no verdict, and the run continues
```

An optional pre-run diagnostic halted the core over GDB, timed out, and left it **halted**; the
driver logged "the run continues" and carried on, so every stage then timed out at
`SQLITE_STAGE_TIMEOUT` and read exactly like a wedge. The discriminator that settled it: the
control failed **identically** twice. Its own flake rate is ~1 in 5 but that flake is an entry
stall, which varies — two identical control failures are a variable we introduced, and the diff
of the failing driver log against the last passing one lands on the line above.

Two changes followed: the early-halt control is now **fail-closed** (it forces a resume, then
proves the shell answers, and aborts the boot loudly if it does not), and the discriminator is in
the `board-run` skill. The fail-closed path is **unexercised** — this run disables the diagnostic
outright, since it already answered its question in an earlier boot — so by this project's own
standard it is an unproven gate, not a fixed one, until it has been negative-tested.

## RETRACTION — every arm-4 measurement in this folder is VOID

**Arm 4 read a slot it never wrote.** In the kernel the subject store lives inside

```c
#if   S12_ARM == 0                    *slot = v;
#elif S12_ARM == 1 || S12_ARM == 3    *slot = v;   /* + the 9 intervening stores */
#elif S12_ARM == 2                    *slot = v;   /* + the scalar scribble */
#endif
```

and **arm 4 matches no branch of it**, so no store to the slot is emitted. The reload
`void *back = *slot;` then reads `s12_frame + 0x690`, which the image never writes. **The
conclusion rests on "never written", not on "zero"**: a granule no capability store ever targets
has its tag clear, so the reload is NOT_CAP whatever the bytes hold. An earlier draft said
"zero-initialised BSS", which is not established — `llvm-readelf -l` shows `.bss` as `NOBITS`
inside a `NULL` program header rather than a `LOAD`, and nothing in the startup assembly or the
build script zeroes it. The observed zeros (`tval` 0, group-9 payload 0) are an empirical fact
about these runs, not a loader guarantee. `back` is therefore a genuine NOT_CAP, and the `cincoffsetimm` consumer raising `mcause 25` is
**correct ISA behaviour** — the hardware doing exactly what the spec says.

Verified with the preprocessor, not by reading:

```
clang -E -P -DS12_ARM=N -DS12_REPS=1 -DS12_DRAW=0 -x c s12_kernel.h
  ARM=1  ->  `*slot = v;`  then  `void *back = *slot;`
  ARM=4  ->  no `*slot = v` anywhere;  `void *back = *slot;` reads zeroed BSS
```

and corroborated in the artifact. **State that check over the whole loop body, not "between the
two `ldc`s"** — those are adjacent instructions four bytes apart, so a check for a store between
them spans zero instructions and cannot fire. That is this project's own "a clean result is not
evidence until the check is known to fire" class, committed *inside* the retraction about that
class. The check that does fire: the loop body of `s12g.dom` is `0x1042c`–`0x104a8` and holds
exactly four `stc`, all to the stack frame — `0x10434`→`s0-0x90`, `0x10448`→`s0-0xa0`, `0x10464`
and `0x10470`→`s0-0xb0` — and **none through `a0`**, the slot pointer loaded at `0x1043c`.

**What this voids.** Everything measured on arm 4 — the group-9 zero at the spill `s0-0xb0`, the
group-9 zero one level up at `s0-0xa0`, `a4 = 0x0` at the wedge, and the whole "the null predates
the store, move the subject up one level" chain of reasoning. Each of those was measuring
uninitialised memory. The backwards walk kept finding zeros because there was never a capability
in that slot to lose. **Arms 5 and 6 never write the slot either**, but they are not equally
affected, and the difference matters:

- **Arm 5's type report does not merely survive — it INVERTS, from REVOKE to NONLIN.** Arm 5
  encodes the type into bits 8–11 *of `bad`* and then also runs the counting consumer, because
  the `#if S12_ARM != 4 && S12_ARM != 6` at the reload site admits arm 5. So the type field and
  the NOT_CAP counter share one word, and the observed `retval=3240776192` = `0xC12A5200` has
  **two readings**, confirmed by simulating the loop exactly:

  | reading | yields |
  |---|---|
  | type = 2 (REVOKE), reload clean every iteration | `0x200` |
  | type = 1 (NONLIN), reload NOT_CAP every iteration, REPS=512 | `0x200` |

  Arm 5 never wrote the slot, so its reload *was* NOT_CAP every iteration — which selects the
  second row. Corroborated twice: arm 6 counted `bad = 512` on the same address (512/512
  NOT_CAP, direct measurement), and rebuilding arm 5 **with** the slot written makes the
  encoding unambiguous (`bad = t<<8`, zero low byte) — QEMU then returns `3240775936` =
  `0xC12A5100` = **type 1, NONLIN**.

  **This reverses a downstream conclusion rather than voiding it.** The kernel's own text
  (`s12_kernel.h:243-244`) says that if the type is NONLIN "the clear NEVER fires and that whole
  mechanism is dead". So the LDC move-clear account is dead, and the five earlier "clean"
  simulation results that used NONLIN were in the **right** configuration all along — their
  exclusions are **reinstated**, not invalidated. Any section of this file arguing that those
  five sims used the wrong type is itself retracted.

  Still open: QEMU's type model need not equal silicon's, so a board run of the rebuilt arm 5
  is queued to confirm NONLIN on the FPGA. Two independent lines already agree.
- **Arm 6's conclusion is VOID.** Its design — "reload the same slot twice; if the move-clear
  fires the second must come back null" — is vacuous when both reloads read zeroed BSS. Both
  come back null regardless of whether the clear fires, so the arm cannot distinguish its two
  hypotheses. Note arm 6's own comment cites arm 5's REVOKE result as its premise; that premise
  is sound, the experiment built on it is not.

**What survives.** Arms 0/1/2/3 do write the slot, so their results stand. The original SQLite
S-12 observation stands — it is a real workload fault and owes nothing to this repro.

**Why no existing check caught it.** The artifact was correct. The fault was the right `mcause`
at the right `mepc`, predicted from the disassembly before the run. The repro was deterministic
and reproduced at `REPS=1`. Every one of those reads as a strong result. The only thing that
distinguished it was the **absence** of a store, and nothing was looking for an absence. This is
the "directed tests that come back clean without ever creating the triggering condition" class
from CLAUDE.md, in its inverted form — a test that *fails* without ever creating the condition,
which is harder to spot because failure is what you were hoping for.

**The fix, and the guard.** Arm 4 now takes arm 1's branch. Each writing branch defines
`S12_SLOT_WRITTEN`, and the reload site carries `#if !defined(S12_SLOT_WRITTEN) #error ...` — so
an arm that reads the slot without writing it **fails the build** rather than producing a
beautiful void result. Negative-tested in both directions:

| arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 9 |
|---|---|---|---|---|---|---|---|---|
| slot store emitted | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| guard fires | · | · | · | · | · | **✓** | **✓** | **✓** |

## Silicon confirmation: NONLIN, and the LDC move-clear does not fire

One boot, control returned, both arms run with the subject store present for the first time.

| arm | what it measures | retval | reading |
|---|---|---|---|
| 5 | cap type of `v`, bits 8–11 | `0xC12A5100` | **type 1 = NONLIN** |
| 6 | reload the same slot twice, count NOT_CAP on the second | `0xC12A6000` bad=**0** | **the move-clear does not fire** |

Arm 5's **low byte is 0**, which is the internal consistency check: it means the counting consumer
never saw a NOT_CAP reload, i.e. the hoisted store actually landed and the encoding is unambiguous
again. That is what makes this reading trustworthy where the earlier `0x200` was not.

Silicon agrees with QEMU (`0xC12A5100` both). So the REVOKE→NONLIN inversion is confirmed on the
FPGA, not merely in emulation, and arm 6 — which could not distinguish its two hypotheses while the
slot went unwritten — now measures directly that loading a NONLIN-stored slot leaves it intact.
Consistent with `load_unit.sv:225-226`, where NONLIN is absent from the clear set.

**What this does NOT establish.** Both results are about **the repro's `v`**. They say the
move-clear does not fire *here*. They bear on S-12 only if SQLite's stored value at the fault site
is also NONLIN, and **that is still unmeasured** — it is the discriminating unknown, and the reason
the kernel's "any tagged capability; its identity is irrelevant" is the weakest assumption in this
folder. If SQLite's value is in the clear set {LINEAR, REVOKE, UNINIT, SEALED, SEALEDRET}, this
repro has never exercised the mechanism it was built to test, and an arm with a matching-type `v`
is the first variant that could reproduce.

## The discriminating unknown is answered: SQLite's value is NONLIN too

The open question above — is SQLite's stored value at the fault site in the LDC clear set? — is
**no**. The derivation, with each link re-verified against the primary source rather than taken on
report:

1. The monitor hands the domain `dom_data` as **LINEAR** (`sbi_capstone.c:302-305`; `split_out_cap`
   with `linear=1` *asserts* linearity and only the `!linear` arm calls `__delin`,
   `sbi_capstone.c:275-281`). It becomes the domain's `sp`.
2. The entry glue **delinearizes `sp` exactly once**: `delin(sp)` at
   `start-gp-captable-interp.S:268`. **Verified this is the shipping path, not a diagnostic one** —
   the line sits in the `#else` arm, and `INTERP_FAKE_COUNT` (which selects the other arm) is never
   defined for the SQLite build; it appears only for one BEEBS rung in `ladder-rungs.spec`.
3. `SPLIT` preserves `cap_type` (`capstone_dyn_unit.anvil:113-149` touches only start/end/cursor/
   revnode), and so does `cincoffset`/`cincoffsetimm` (`capstone_flu_unit.anvil:38-40`, `:68-70`
   copy the whole `metadata`). So every cap-table storage capability and every carved pointer
   below `sp` inherits NONLIN.
4. Both branches converge. **Stack** pointers and `s0` are `sp` itself or splits of it. **Heap**
   pointers are not what the premise assumed: this build does **not** use `umm_malloc` (that is the
   rv8 benchmarks, zero references in the SQLite path) — SQLite runs memsys5 over a static `.bss`
   arena, `sqlite_heap` (`sqlite_capstone_domain.c:28`, configured at `:1559`), which is itself a
   carved global reached through cap-table slot 176. So a "malloc'd" pointer is pointer arithmetic
   inside one NONLIN global. No `mrev`/REVOKE path is live: `revoke_on_free_alloc.h` is not
   referenced by the SQLite build.
5. **Empirically corroborated by S-02**: a `delin` on `hostcall_payload` *wedged the board*
   precisely because the capability was already NONLIN and the RTL's DELIN accepts LINEAR only
   (`sqlite_capstone_domain.c:208-238`).

**Consequence — a real exclusion, on two independent legs.** The linear move-clear mechanism is
**not live at the S-12 fault site**, for SQLite or for the repro. Measured on silicon here (arm 6,
`bad=0`) and derived-and-verified for SQLite above. It is likewise exempt from the STC rs2-clear
(`capstone_dyn_unit.anvil:439`, `:458`). Whatever loses the value at the fault site, **it is not
source-granule clearing on `ldc`/`stc`.**

**And it clears the repro of the fidelity charge that was open against it.** The worry was that
`v`'s type made the repro test the wrong thing. It did not — the repro's NONLIN matches
production's NONLIN. So the window *with the right type* is genuinely clean, and the missing
ingredient is elsewhere: cache pressure, interrupt/timer traffic, rev-node churn, or simply the
surrounding code volume. That is where an S-12 reproducer has to come from, and it points at
SQLite-side bisection rather than at another arm of this kernel.

**One caveat on provenance.** The derivation above was produced by an agent that read a *stale*
copy of `s12_kernel.h` and quoted its "the granule is intact AND TAGGED … never lost, it was never
delivered" paragraph as current. That paragraph is **withdrawn** (see the header rewrite). The
derivation does not depend on it, but do not carry that sentence forward from any report quoting it.

**The one measurement that would close this by observation rather than derivation** already exists
as an instrument: `CAPSTONE_ARG_PROBE=sqlite3WhereCodeOneLoopStart`
(`build-sqlite-silicon.sh:1030-1060`), expecting **1** on the incoming `pWInfo`. Note the encoding
trap: `lcc` selector 1 reports `cap_type - 1` with NOT_CAP special-cased to 7, so **1 means NONLIN**
and reading it as LINEAR against the raw `asm_insn.h` numbering is the natural mistake
(`sqlite_capstone_domain.c:217-222`).

## MEASURED: SQLite's pointers at the fault site are NONLIN (type 1)

The derivation above is now closed by observation. With the arg probe repaired (its caller query
used a non-total selector and aborted every run — see below), a QEMU run of the same domain and
the same `q_two.test` case reports, at `sqlite3WhereCodeOneLoopStart` itself:

```
ARGP calls=1 ty1=0000000000000001 ty2=0000000000000001 ra=0000000101ce154c
SLT-SUMMARY records=2 stmt_pass=1 stmt_fail=0 query_pass=0 query_fail=1 completed=1
```

`ty1`/`ty2` are the two incoming pointers. **`lcc` selector 1 reports `cap_type - 1`, with NOT_CAP
special-cased to 7 — so `1` is NONLIN, not LINEAR.** Reading it against the raw `asm_insn.h`
numbering is the natural mistake and is documented at `sqlite_capstone_domain.c:217-222`.

Two independent routes now agree — the five-link derivation (monitor LINEAR → `delin(sp)` →
type-preserving `SPLIT`/`cincoffset`) and this direct measurement. That matters because any one
broken link would have taken the derivation's conclusion with it.

**Caveat:** measured under QEMU, not silicon — a cross-check of the derivation, not a silicon
measurement. The comparable repro measurement did agree across both (arm 5 returned `0xC12A5100`
on QEMU *and* on the FPGA), which is evidence the type models match at least here, but the silicon
reading at this site remains unmeasured.

### The probe defect that had to be fixed first

`CAPSTONE_ARG_PROBE`'s two argument queries are selector 1 (total, answers 7 for an untagged
operand). Its **caller** query was not:

```c
__asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, x1, x2" : "=r"(ap_ra_));   /* lcc ra, selector 2 */
```

Selector 2 is not total: on an untagged operand QEMU trips `assert(rs1_v->tag)`
(`op_helper.c:762`) and **aborts the emulator**; silicon raises. And in this build that is
guaranteed, not incidental — the SQLite silicon domain contains **zero `cjalr`** (verified by
`objdump`), so every call is a plain `jal` and `ra` is always a plain integer. The probe therefore
died immediately after `SQ: G/enter` on every run, never reaching the site it was aimed at, and
would have faulted the domain on the board too. Its header's "non-perturbing — records and falls
through" was false for that half.

Fixed to ask the total selector first and take the cursor only when there is a capability to take
it from. Positive-tested the same day: zero `helper_cslcc` aborts, run completes, summary identical
to the un-probed run and to native.

## CONFIRMED LIVE on silicon, 2026-08-25 — and silicon-only

Two arms, one boot, host and domain staged from the same build:

| arm | markers | verdict |
|---|---|---|
| `sqslt.dom` (no `--slt`) — control | `G/enter` → `H/return` | returned in 7 s, **so the boot carries verdicts** |
| `sqslt.dom --slt q_two.test` | `G/enter`, **no `H/return`** | **entered and WEDGED** |

`G/enter` present with no `H/return` is a real wedge, not an entry stall — that distinction is the
whole reason the arm is readable. The fault site was re-derived from the binary that actually ran:

```
latched trap  mcause=25 (UNEXPECTED_OPERAND)   mepc=0x828f4814   DBAS=0x82800000
  -> VA 0x104814;  sqlite3WhereCodeOneLoopStart at 0x104788  ->  +0x8c

104808:  movc  a4, zero               the zeroed value
10480c:  stc   a4, 0x0(a5)
104810:  ldc   a4, 0x0(a0)            the reload
104814:  cincoffsetimm a4, a4, 0xb0   <- FAULTS
```

Exactly the documented signature. **Read the CSRs with care:** gdb reported `mcause=2 mepc=2`,
which the driver DISCARDED because it disagreed with the latched trap — a later trap had clobbered
them. Taking the gdb values at face value would have produced a completely different, wrong story.

**And it does not reproduce under QEMU.** The *same* domain image running the *same* `q_two.test`
completes in emulation (`G/enter → H/return`, `SLT-SUMMARY … completed=1`) and matches the native
baseline record for record. S-12 is **silicon-specific**, and there is now a reference model to say
so — the Q-01 fix.

### Two arms that were NOT verdicts, recorded so they are not re-run as if they were

* A three-arm boot put `q_two` **third** (domain `id=2`, region ids already `0x0A`). It returned
  nothing, but stopped at `SQ: C/mkregion2` — before `D/mapped`, `F/share2` or `G/enter`. That is
  a hang in **region creation, outside the domain**, while S-12 is inside it and necessarily
  follows `G/enter`. "NO RETURN" alone does not separate the two. Put the subject in **slot 2**,
  matching the historical configuration that wedged.
* An earlier boot used a ladder-style `label:path` entry with `SQLITE_HOST=sqlite_host.user`. The
  label became `argv[1]`, the loader printed `Failed to open the file.`, and the driver hard-stopped
  the arm as a PHANTOM — correctly refusing to read the `SQ: G/enter`/`H/return` that followed,
  which belonged to a domain that was never loaded.

### Where this leaves the mechanism

Open, and now bounded from both sides. The move-clear account is dead (NONLIN, measured and
derived). The 40-line window is clean with a proven-firing detector under both consumers. So the
trigger needs something the window does not have — cache pressure, timer traffic, rev-node churn,
or simply the surrounding code volume — and the next step is SQLite-side bisection between the
single-table case that passes and the self-join that does not. `q_one` and `q_two` differ by
exactly `, t1 AS y`, and both tables are EMPTY, so no data is involved either way.

## Silicon: NONLIN confirmed, and S-12 is PERTURBATION-SENSITIVE

Same domain, same `q_two.test`, same slot (arm 2 of 2), with the repaired arg probe enabled:

```
arm 1  sqprb.dom (no --slt), control   G/enter -> H/return   returned 7s
arm 2  sqprb.dom --slt q_two.test      G/enter -> H/return   returned 6s
       ARGP calls=1 ty1=1 ty2=1
       SLT-SUMMARY records=2 stmt_pass=1 query_pass=0 query_fail=1 completed=1
```

**1. The type is now an FPGA measurement.** `ty1=ty2=1` = NONLIN (selector 1 reports
`cap_type - 1`). Previously derived from source and measured only under QEMU. The linear
move-clear is not live at this site, confirmed on the hardware that actually has the bug.

**2. The probed build does not wedge** — where the un-probed build, in the identical slot, wedged
at `+0x8c` in the boot immediately before. And it completes *correctly*: its summary matches QEMU
and native record for record.

**What that does and does not support.** The two builds differ in **two** respects:

| build | size | `WhereCodeOneLoopStart` |
|---|---|---|
| un-probed (wedges) | 1624152 | `0x104788` |
| probed (completes) | 1624904 | `0x104b48` |

The probe adds instructions to the function **and** moves it 960 bytes. So *"S-12 is sensitive to
code perturbation"* is supported; *"the probe's instructions cure it"* is **not**. This is exactly
the arms-differ-in-more-than-one-respect case, and link address is the named confound.

Perturbation sensitivity is itself a strong hint: it is **R-20's signature** — cured by exactly one
nop — which points at a pipeline/scheduling hazard rather than data or memory corruption. It also
explains why the 40-line repro comes back clean: its instruction context is not production's.

**UNRESOLVED:** only one `ARGP` line appears for a two-table self-join, which should drive two
where-loop levels. Whether the report is first-call-sticky or the function is genuinely entered
once is not established.

**The one-boot discriminator:** inject the probe into `sqlite3WhereMalloc` (`0x102f54`, *before*
the fault site). That moves `WhereCodeOneLoopStart`'s address while leaving its body
**byte-identical**. Wedges → the instructions inside the function are what matter. Completes →
the address/layout shift is, and the instructions are incidental.

## The cure does NOT require touching the fault function's instructions

The discriminator ran: probe injected into `sqlite3WhereMalloc`, which sits *before*
`WhereCodeOneLoopStart` in the layout, so the fault function's address moves while its
instruction stream stays the same. Verified rather than assumed — same 2866 instructions in the
same order, the only body difference being one immediate (`addi a2,a2,-0x3d0` → `-0x3d4`), a
globals-offset adjustment that changes no timing or scheduling.

| build | size | `WhereCodeOneLoopStart` | body | result |
|---|---|---|---|---|
| un-probed | 1624152 | `0x104788` | baseline | **WEDGES** at `+0x8c` |
| probe **in** WhereCode | 1624904 | `0x104b48` | + probe instructions | completes |
| probe in **WhereMalloc** | 1624904 | `0x104c0c` | same stream, 1 immediate | **completes** |

Both probed builds complete, and their `SLT-SUMMARY` matches QEMU and native record for record —
they are running correctly, not failing differently.

**What is established:** the fault survives or vanishes according to the *surrounding layout*, not
according to the content of `WhereCodeOneLoopStart`. Adding instructions inside that function is
not necessary to cure it.

**What is NOT established, and must not be written as though it were:** *which* layout property
matters. Address, image size, and globals offset all moved together in both probed builds, so
"it is the address" is one candidate among three. Isolating it needs a build that shifts the
address while holding size and globals fixed, which the two-pass link does not currently offer as
a knob.

**The N=1 problem, being fixed before anything is built on this.** The wedge itself has been
observed **once**. On a system with known per-image nondeterminism — R-16's entry stall is
explicitly per-image, and the standing rule is to REDRAW rather than retry — a single observation
is not enough to carry the "layout decides it" reading. A repeat of the identical un-probed image
is running. If it wedges again the foundation is sound; if it does not, S-12 may be an
alignment/layout **lottery** of the same class as R-16, and every arm above needs re-reading in
that light.

### N=2: the wedge is DETERMINISTIC, not a lottery

The identical un-probed image was re-run on a fresh boot and wedged again, bit-identically:

| run | control | `--slt q_two.test` | latched trap |
|---|---|---|---|
| 1 | returned 7 s | `G/enter`, no `H/return` | `mcause=25  mepc=0x828f4814` |
| 2 | returned 7 s | `G/enter`, no `H/return` | `mcause=25  mepc=0x828f4814` |

Same PC, same cause. **The R-16-class "alignment lottery" hypothesis is refuted**: this image does
not sometimes wedge, it always does. Together with the probed images completing (two different
probe placements, plus their controls), the tally is:

```
un-probed image          q_two   WEDGED     2/2
probe-in-WhereCode       q_two   completed  1/1
probe-in-WhereMalloc     q_two   completed  1/1
```

So the behaviour is a deterministic function of the image, and **layout selects it**. That is what
licenses the layout reading; before this repeat it rested on a single observation, and "this image
lost a lottery" predicted exactly the same data.

**Two variables are now both known to matter, independently:**

* **The SQL.** In the *same* un-probed image and the same slot position, `q_one` completes and
  `q_two` wedges. The pair differs by exactly `, t1 AS y`, with both tables EMPTY — so it is the
  join's generated code path, not data.
* **The layout.** With the *same* SQL, moving `WhereCodeOneLoopStart` cures it without altering a
  single instruction of that function.

A mechanism has to explain both: a code path that only the join generates, which faults only at
certain placements. Deterministic, PC-sensitive, and not source-granule clearing (NONLIN, measured
on silicon) — that profile points at the front end or at something indexed by address, rather than
at data or memory corruption.

## Isolated: the deciding variable is the INSTRUCTION side, not the data side

`link-gpfree.ld` puts `.text` at `0x10000` and the globals at `0x10000 + GOFF`, so raising `GOFF`
moves the globals region while leaving every `.text` address untouched. A new diagnostic knob
(`SQLITE_GOFF_OVERRIDE`, refused if below the computed value so globals can never overlap `.text`)
builds exactly that image. Verified rather than assumed: **331778 instructions, 4 differing** — all
`auipc` globals-base constants (`0x161`→`0x171`) — and `sqlite3WhereCodeOneLoopStart` still at
`0x104788`.

| `WhereCode` @ | globals @ | `.text` | result |
|---|---|---|---|
| `0x104788` | `0x150000` | baseline | **WEDGE**, 2/2 |
| `0x104788` | `0x160000` | 4 `auipc` differ | **WEDGE**, 1/1 |
| `0x104b48` | `0x150000` | +probe in the function | completes |
| `0x104c0c` | `0x150000` | probe in `WhereMalloc` | completes |

**Globals placement is eliminated by direct experiment.** It was already suspect — `GOFF` is
`0x150000` in *all* earlier builds, wedging and curing alike, because it rounds up to 64 KiB and a
~800-byte probe never crosses the boundary — but now it has been moved deliberately and the fault
survived. Every image with the function at `0x104788` wedges (3/3); every image with it moved
completes (2/2).

## What the fault operand actually implies — stronger than "detagged"

`decompress_cap_tagged` (`core/include/ariane_pkg.sv:766-782`) does **not** zero the metadata word
on an untagged read; it stashes the raw 64 bits into `bound_start`, and the cursor passes through
unchanged:

```systemverilog
end else begin
  return '{ metadata: '{ ..., cap_type: NOT_CAP, ...,
                         bound_start: tagged_metadata[63:0],   // raw stash
                         bound_end: '0 },
            cursor: cursor };
```

So an operand that is bit-for-bit `create_cnull` — which `tval = 0` together with `cap_type ==
NOT_CAP` implies — requires the memory to have been **genuinely all-zero on both halves**, not
merely detagged. **A detagged capability would have delivered a non-zero cursor and `tval` would
not be 0.** That excludes the whole "a good pointer lost its tag" family at this site and demands
either a location that really held zero, or a load that returned a default without reading.

**One tempting instantiation, refuted by arithmetic.** The window stores an all-zero capability two
instructions before the reload (`movc a4, zero; stc a4, 0x0(a5)` → `s0-0x120`), so "the reload was
served that zero from the wrong address" is a bit-exact match for the signature. It does not
survive the addresses: at the wedge `s0 = 0x82b9f480`, so the zero-store lands at `0x82b9f360` and
the reload reads `a0 = 0x82b9f410` (= `s0-0x70`). D$ is 32 KiB / 8-way = 4 KiB per way, so the
index is `paddr[11:4]`: **0x36 vs 0x41**, and the granules (`0x82b9f36` vs `0x82b9f41`) differ too.
No set collision and no write-buffer granule match. (`s0` is corroborated independently: the s07
STC recorder reported the last capability store at granule `0x9f360`, exactly `s0-0x120`.)

## Two exclusions in the older notes that are narrower than they read

* **The writeback-port displacement detector** (`core/scoreboard.sv:335-347`) fires on
  `op == LDC` at the writeback trans_id and never inspects the data. Its own comment describes the
  signature it catches as *"retires with a correct cursor and NOT_CAP metadata"*. **Our fault has
  cursor 0 as well**, so a mechanism that zeroes *both* halves would not trip it. Reading its
  `0x00` as "displacement excluded" is only valid for "wrong port, data preserved".
* **Instruction realignment** (`core/instr_realign.sv:66-115`) branches on `address_i[1]`, which is
  layout-dependent in general — but **excluded for this pair by arithmetic**: `0x104788 & 3 == 0`
  and `0x104c0c & 3 == 0`, and the intervening stream is unchanged, so every instruction's
  alignment parity relative to the function start is preserved. (Not a general exclusion — a shift
  that changed the mod-4 residue would reopen it.)

**Still PC-indexed and not excluded:** the I-cache line phase (`CVA6ConfigIcacheLineWidth = 128`,
16-byte lines; `0x104788 mod 16 = 8` vs `0x104c0c mod 16 = 12`) and the BTB, which indexes on
`vpc_i[PREDICTION_BITS-1:ROW_ADDR_BITS+OFFSET]` with only 8 entries (`core/frontend/btb.sv:66-72`),
so aliasing shifts when a function moves. Neither has a demonstrated path to a corrupted capability
*value* — both are timing/contention knobs, which is how this folder should keep describing them
until a path is shown.

## RETRACTION x2 — the "byte-identical body" claim and the "NONLIN measured at the fault site" claim

Both were caught by an adversarial audit and both are verified against primary source here.

### 1. The fault function was NOT byte-identical between the two builds

Stated in this file and in commits `384a919666bf` / `96288aa211fa` as *"2866 instructions in the
same order, the only body difference one immediate"*. **False.** Measured over the exact symbol
bounds from `llvm-nm --print-size` (`0x104788`+`0x47e0` vs `0x104c0c`+`0x47dc`):

```
slt2 (wedges)    4600 instructions
slt5 (completes) 4599 instructions        <- not even the same COUNT
differing:       1469,  of which 958 differ in mnemonic/register, not just an immediate
```

Register allocation and the gp-table index materialisation changed. **How the error was made:**
the comparison helper's end-of-function regex terminated early (2866 instructions instead of
4600), and it printed only the FIRST difference — which I then reported as though it were the
ONLY one. Counting was one line away and would have caught it.

**Consequence:** *"the cure does not require touching the fault function's instructions"* is
**unearned**. The instructions were touched. The discriminator did not do what it was built to do.

The confound list was also wrong. Between the wedging and curing images `.text` size, every data
section's VA, `.capstone_gp_table` size (+0x90, nine more entries) and the cap-init entry count all
moved, along with register allocation inside the fault function. And **the globals offset was never
a confound at all** — `GOFF` is `0x150000` in every build (it rounds to 64 KiB); the
`-0x3d0`→`-0x3d4` difference is a per-reference offset, not the region base. The GOFF-override
experiment is still valid and still useful (it independently confirms the wedge at N=3 and rules
the globals region out), but it was answering a question that was not open.

### 2. NONLIN was NOT measured at the fault site

The arg probe reads arguments **1 and 2**. The faulting value is argument **3**:

```
1047a4:  movc a6, a0             a6 = arg1 (pParse)     <- ty1
1047a8:  ldc  a0, -0x600(s0)     a0 = arg2 (v)          <- ty2
1047c0:  cincoffsetimm a0, s0, -0x70
1047c8:  stc  a2, 0x0(a0)        arg3 -> s0-0x70        <- THE SUBJECT, never probed
104810:  ldc  a4, 0x0(a0)        reload arg3
104814:  cincoffsetimm a4, a4, 0xb0   FAULTS
```

So `ty1=ty2=1` says two *sibling* arguments are NONLIN. The value that is stored, reloaded and
faults was never measured — and it could not have been, because probing the image cures it.

**What survives:** the RTL half. NONLIN is absent from the clear set (`load_unit.sv:226-229`), so
*if* the value is NONLIN the move-clear cannot fire. The claim must read **"inferred NONLIN by
analogy with two sibling arguments, not measured"**. Re-pointing the probe at argument 3 would not
fix this — it can only ever measure a completing build. Only a non-perturbing hardware instrument
can read the operand on an image that actually wedges.

### 3. Corrections to two other things this folder asserted

* **"The two-pass link offers no such knob" is false.** `build-sqlite-silicon.sh:462` already has
  `CAPSTONE_TEXT_PAD=N` — dead, never-called code at the top of `.text`, no globals, written for
  exactly this purpose ("the null-shift control the audit asked for"). With `.text` at `0x144004`
  and globals pinned at `0x160000` there is ~49 KB of headroom before anything downstream moves.
* **"One subject arm per boot" is false.** Three domains have reached `H/return` in a single boot
  (`slt-board8/10/11.log`, ids 0,1,2), with `SPLB:0000E010` (`CAPSTONE_ERR_SPLIT_EXACT`) hitting on
  the *fourth*. `up6b`'s death at the third is pool-state dependence, already documented in
  `RATE-RULE.md`. **Budget three arms.**
* **"Deterministic" is overclaimed at 2/2.** `RATE-RULE.md` records a background wedge rate of
  p̂ ≈ 0.22, and 2/2 gives only a 95% lower bound of p ≥ 0.224 — indistinguishable from background.
  N=3 puts P(background) at 0.011, which is the cheap meaningful step; "deterministic" by counting
  alone would need n≈14–29. (The 0.22 was measured on a *different* bitstream, `s07debug_18august`,
  not the resident `s07clear_84ed6eafb` — quote it with that caveat.) The **signature** carries more
  weight than the count: two bit-identical `25 / 0x828f4814` readings are far less likely under
  background than two bare wedges, but nothing currently records per-wedge signatures to establish
  that. Worth starting to record.
* **Entry attribution should quote `ENT1`, not `SQ: G/enter`.** The monitor's own rule
  (`sbi_capstone.c:924-926`): `ENT0` then silence = died before the switch; `ENT1` then silence =
  control genuinely left M-mode and the domain owns the wedge; `ENT2` = returned. Both wedges show
  `ENT0`+`ENT1` then silence.

## RETRACTED (3rd time): "layout decides it" is DEAD. Address is not the variable.

The address-only experiment, built properly this time. `CAPSTONE_TEXT_PAD=1120` inserts dead,
never-called nops at the top of `.text` — no globals, no control flow — tuned offline in two
builds so `sqlite3WhereCodeOneLoopStart` lands at **exactly `0x104c0c`**, the address at which the
probe builds completed. Gates run BEFORE the boot, which is what the retracted attempt skipped:

```
instruction count          4600 == 4600
STRUCTURAL diffs in fn     0            (605 immediate/label-only, from pcrel shifts)
.capstone_gp_initdesc      VA+size IDENTICAL
.rodata / .data / .bss     VA+size IDENTICAL
residual                   .gct +0x50, gp_table VA +0x50 (SIZE unchanged -> same indices)
```

**It wedges.** `ENT0=1, ENT1=1`, no `ENT2` — control left M-mode and the domain owns it. Latched
trap `mcause 25`, `mepc 0x828f4c98` = `0x104c0c + 0x8c`, and the instruction there is the same
`cincoffsetimm a4, a4, 0xb0` preceded by the same `ldc a4, 0x0(a0)`.

| image | `WhereCode` @ | globals @ | fault | cause |
|---|---|---|---|---|
| baseline | `0x104788` | `0x150000` | `+0x8c` ×2 | 25 |
| GOFF moved | `0x104788` | `0x160000` | `+0x8c` | 25 |
| **TEXT_PAD** | **`0x104c0c`** | `0x150000` | **`+0x8c`** | **25** |
| probe in WhereCode | `0x104b48` | `0x150000` | — completes | |
| probe in WhereMalloc | `0x104c0c` | `0x150000` | — completes | |

**Same address, opposite outcomes.** `TEXT_PAD` and the WhereMalloc-probe build both put the
function at `0x104c0c`; one wedges and one completes. Address cannot be the variable, and neither
can the globals region. **What is left is the code change the probe introduced** — the register
allocation inside the fault function (958 structurally different instructions), the nine extra
gp-table entries, and the changed cap-init length. The `TEXT_PAD` build has none of those and
faults exactly like the baseline.

So the correct statement is the narrow one: **something about the code generated for this function
(or the cap-table/cap-init geometry around it) decides the fault, and pure placement does not.**

### And a self-inflicted error, recorded because it is the same class

This boot's driver line first read `latched trap (24/...)`, and I began interpreting mcause 24 —
checking whether it was `NO_EXCEPTION` or the `commit_stage` off-by-one at `commit_stage.sv:205-228`.
It was neither. Earlier today I "fixed" the message's hardcoded `25` by interpolating
`_latched & 0x7f` — but `_latched` is the latched **mepc**, not the cause, and `0x828f4c98 & 0x7f`
is 24 by coincidence. The trap log itself read `0x99` → cause `0x19` = **25** all along.

The original hardcoded `25` was wrong for a different reason (it printed 25 even when the latch
held cause 9), so the defect was real — but my fix invented a number out of an address. It now
reads the cause from the trap-log value (`traplog_v & 0x7F`), which is where it was available the
whole time. **A "fix" that produces a plausible number from the wrong variable is worse than the
bug it replaced**, and it very nearly bought a mechanism investigation into an exception code that
was never raised.

## FOUND: `.bss` SIZE decides S-12 — 32 bytes cures it, and the prediction was made first

Tabulating every image built against its outcome, exactly one property separates the two groups:

| build | fn @ | caps | `.bss` VA | **`.bss` size** | outcome |
|---|---|---|---|---|---|
| slt2 | `104788` | 328 | `170bf0` | **`409c0`** | WEDGE ×2 |
| pad2 (TEXT_PAD) | `104c0c` | 328 | `170bf0` | **`409c0`** | WEDGE |
| goff (globals +64K) | `104788` | 328 | `180bf0` | **`409c0`** | WEDGE |
| gp6 (+6 `.data` globals) | `1048e4` | 338 | `170d20` | **`409c0`** | WEDGE |
| slt4 (probe in fn) | `104b48` | 337 | `170cb0` | **`409e0`** | completes |
| slt5 (probe in WhereMalloc) | `104c0c` | 337 | `170cb0` | **`409e0`** | completes |
| **bss32 (+32 B `.bss`)** | `10479c` | 329 | `170c00` | **`409e0`** | **COMPLETES** |

Function address, cap count, `.bss` VA and heap VA each vary *within both groups*, so each is
refuted by its own data. **`.bss` size does not**: `0x409c0` in every wedging build, `0x409e0` in
every completing one. The `0x20` delta is exactly the four `unsigned long` globals the arg probe
adds.

**`gp6` is the control that makes this a mechanism rather than a coincidence.** It appends six
dummy globals *with initialisers*, so they land in `.data` and `.bss` never moves — same extra
globals, wrong section — **and it wedges.**

**And `bss32` was predicted before it was run.** Take the wedging build, append 32 bytes of
uninitialised global, change nothing else. Predicted: completes. It completed —
`ENT2=5117600D`, `SLT-SUMMARY records=2 … completed=1`, matching QEMU and native.

### The mechanism chain, visible in the build's own output

`.bss` size decides how much room `domdata-budget` leaves for the stack:

```
slt2   (WEDGES)     = STACK  2347072
bss32  (COMPLETES)  = STACK  2346992        <- 80 bytes smaller
```

So: **`.bss` size → stack size/base → `s0` → the address of the faulting slot at `s0-0x70`.** That
is a data-address mechanism, and it survives every code-side refutation — address, cap count,
globals region, and the code of the fault function itself (whose instructions are byte-identical
from entry through the fault in both a wedging and a completing build).

### What this retires

Every "layout"/"perturbation-sensitivity" reading in this folder above should be read as **stack
address sensitivity**. The probe builds never cured anything by perturbing *code*; they cured it by
adding four globals to `.bss` and moving the stack. That is why probing the function and probing an
uncalled function worked equally well, and why `TEXT_PAD` at the identical address did not.

### Next: the periodicity ladder

`CAPSTONE_BSS_PAD=N` is now the knob, and it is cheap and code-neutral. A ladder of N
(16/32/48/64/…/4096) maps wedge vs complete against stack displacement. If the pattern is periodic,
the period names the aliasing structure directly — D$ set (32 KiB/8-way → 4 KiB per way, index
`paddr[11:4]`), write-buffer granule, or page. That is a far sharper question than anything
available before today.

### Boot-budget caveat

The third arm of this boot died at `SPLB:0000E010` (`CAPSTONE_ERR_SPLIT_EXACT`) on domain `id=2`,
before `SQ: G/enter` — no verdict, exactly as in the earlier three-arm boot. Three arms is a
gamble, not a guarantee: it is pool-state dependent. **Put the arm that matters second.**

### But the slot's ADDRESS is not the variable — `gp6` moved it and still wedged

| build | `s0` | slot (`a0`) | outcome |
|---|---|---|---|
| slt2 | `0x82b9f480` | `0x82b9f410` | WEDGE |
| pad2 | `0x82b9f480` | `0x82b9f410` | WEDGE |
| goff | `0x82b9f480` | `0x82b9f410` | WEDGE |
| **gp6** | **`0x82b9f350`** | **`0x82b9f2e0`** | **WEDGE** |

`gp6`'s stack sits `0x130` lower and its faulting slot is at a different address — and it wedges
anyway. So the fault is **not** "the slot must land at one particular address", and not a fixed
D-cache index (`paddr[11:4]` = `0x41` for the first three, `0x2e` for gp6) or granule.

This matters because it kills the tempting chain *.bss → stack base → slot address → address-indexed
structure* at its **last** link, using data already in hand. `.bss` size still separates the two
groups across seven images; what it does downstream is not simply "moves the slot somewhere bad".

The first link is independently confirmed — the stack budget tracks the pad — but **not linearly**:

```
pad=0   STACK 2347072
pad=16  STACK 2347008     (16 bytes of .bss costs 64 of stack)
pad=32  STACK 2346992     (32 costs 80)
```

so the carve arithmetic has its own rounding and a stack address must be **measured**, not predicted
from a pad size.
