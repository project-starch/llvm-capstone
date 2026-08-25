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
