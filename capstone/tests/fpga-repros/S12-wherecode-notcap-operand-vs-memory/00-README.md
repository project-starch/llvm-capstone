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
