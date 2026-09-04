# S-12: the proposed fix is REFUTED (it breaks UNINIT), and the mechanism framing was wrong

## 1. The fix is withdrawn — it does not break linearity, it breaks UNINIT

Proposed: for the non-clearing STC case, return the response with the capability-result validity
deasserted so the entry is not a forwarding candidate. **Refuted on three independent grounds.**

**It would deadlock.** There is no per-response validity field; one wire drives both consumers
(`cva6.sv:1534` and `:1546`). Deassert it and `sbe.valid` is never set (`scoreboard.sv:214-224`), so
the STC never commits — the core wedges on the first `stc` of an integer.

**Steelmanned with a new bit, it is ineffective.** `cap_result.valid` is ANDed into only the
rs1-alias disjunct of `rs1_fwd_req` (`issue_read_operands.sv:720-727`); the **rd-match leg the claim
depends on has no such term**. And the metadata mux at `:769-772` degenerates on an rd match — both
arms of the ternary yield `result_metadata`, so the forwarded value is bit-identical either way.

**The safety answer, which is not the one I expected.** Linearity, sealing and revocation are
SAFE: the clearing branch covers LINEAR, REVOKE, UNINIT, SEALED, SEALEDRET and EXIT, and the fix
only touches NONLIN and NOT_CAP. But the same else-branch response carries `cap_rs1`, which for
**UNINIT is the advanced cursor** (`capstone_dyn_unit.anvil:436-450`), and the rs1 writeback at
commit is gated on exactly the bit the fix deasserts (`commit_stage.sv:278-281`). Storing an integer
or a NONLIN capability **through an UNINIT capability — i.e. initialising memory, the whole purpose
of UNINIT** — takes that branch. Under the fix the cursor never advances, every store lands at the
same offset, and the capability never converts. Silently.

The acceptance test that fails: derive UNINIT over 32 bytes, `stc` twice, require the two granules
to hold distinct values and the cursor to read `base+32`. **A linearity control alone would have
passed and certified a fix that breaks UNINIT** — which is exactly why the ISA-invariant rule
demands a criterion that FAILS on the break.

## 2. The mechanism framing was wrong: there are never two arbitrated producers

I described the arbiter handing the consumer the STC's entry over the LDC's. That cannot happen.
Issue is single-issue and in-order (`SuperscalarEn: 0`), and the LDC's own `rd` is a4, so
`gpr_clobber_vld[a4][stc_idx]` holds `stall_waw[LDC]` while the STC is a live claimant — the LDC
cannot even issue. The tree already measures this: `duplicate-live-rd cycles = 0` with the detector
proven live. **The arbiter/ExtPrio leg of my account was superfluous and misleading**, and an RTL
reviewer checking it would have found no arbitration and discarded the whole report.

The window that does exist is a **commit-stall ordering escape**: `commit_stage.sv:323-325` asserts
`we_gpr_o[0]` and `cap_we_o[0]`, then `:341-347` withholds only `commit_ack_o[0]` when the store
buffer is full. `we_gpr=1 / waddr=a4 / ack=0` persists for the whole stall, which clears both
`stall_waw` and `stall_waw_rs1` (`issue_read_operands.sv:1633-1642`) and releases younger
instructions while the STC remains `still_issued & sbe.valid`.

**And the forwarded value equals the stale register-file value.** `{0, NOT_CAP}` is a4's
architectural pre-LDC content, so this is an ordering escape — the consumer issuing before its true
producer produced — not a wrong datum injected by the store.

## 3. Decode aliasing alone does not discriminate — the CLEAN arm has it too

`02-clean-stc-t0.dom` contains `stc a4, -0x5a0(s0)` six instructions ahead of the same
`ldc a4` / `cincoffsetimm a4`. So "STC decodes rd := rs2, therefore a4 has two producers" is equally
true in the **non-wedging** arm. Leg (a) carries nothing on its own; the discriminator is **STC
liveness at the LDC's issue**, not the aliasing.

## 4. I cited a RETRACTED number

I repeatedly quoted "a fence cut the rate from 11/11 to 1/4" as supporting evidence. **That
dose-response was retracted in this folder** (`00-README.md:1495-1515`): the middle rung's single
wedge was an R-16 entry stall the classifier miscounted. There is no gradient. The defensible fence
numbers are 0/4 and 0/3 against 4/4 unmodified in the same boots. Quoting a retracted figure from
the folder I was editing is the kind of error the folder exists to prevent.

## 5. A simpler explanation is live, and my simulations were built blind to it

`00-README.md:1478-1490` already names it: a **wrong-address write-buffer forward** — the null
capability written by the `stc` forwarded to the `ldc` one instruction later. It predicts
`{cursor 0, NOT_CAP}` exactly, needs no new defect (it is the S-09/S-10/R-19/R-20 family already
live on this bitstream, with `wt_dcache_mem.sv:280`'s `wbuffer_hit_oh` word-granular against a
16-byte capability), and is cured by a fence.

The obvious objection — that a data-forward cannot produce the one-byte dissociation, since both
arms store identical bytes — dissolves: the byte changes the STC's scoreboard `rd`, which changes
whether `stall_waw` holds the LDC's **issue**, which changes the LDC's timing relative to the
in-flight store. **The rd aliasing is a timing lever, not a data source.**

My simulation variants deliberately place the STC target far from the LDC's granule to avoid a
write-buffer confound — so they were **built to be blind to this account**, and their negatives say
nothing about it.

**Cheap discriminator, already written in the folder at `:1492-1495`:** move the null-capability
store out of the window (relocate the `Index *pIdx = 0;` initialiser past `pWC = &pWInfo->sWC;`).
Wedge disappears → write-buffer account. Wedge persists → the ordering-escape account.

## 6. The simulation zeros are UNRESOLVED, not supporting

The decisive combination has never run: the mechanism needs a **stalled STC and a pending LDC
together**, and each test supplied one. `stc-ldc-miss.S` issues one store per iteration to a single
hot line — nothing that fills a store buffer — while the sbpressure run stayed at 7.0 cycles/load.
Worse, `_s12e_escape` counts *any* commit `we_gpr` on the consumer's rs1, which is the normal
forwarding case; **no counter anywhere measures `commit_instr_i[0].op == STC && !commit_lsu_ready_i`**.
A zero whose precondition was never shown to occur is not evidence — my own rule, applied against me.

## 7. One finding that is report-ready regardless

`commit_stage.sv` asserts `we_gpr_o[0]` (`:323`) and `cap_we_o[0]` (`:325`) for an instruction whose
`commit_ack_o[0]` it then withholds (`:341-347`). Inert for plain RISC-V stores (`rd = x0`), but this
fork gives STC a real `rd` and LDC has one natively, so both reach that gate: a register-file write,
repeated every stall cycle, for an instruction that has not retired, feeding two stall-clearing
clauses that release younger instructions. Wrong on its own terms, independent of S-12.

## What survives

Legs (b), (c) and (d) are confirmed at source, the one-byte dissociation was independently
re-derived (`cmp -l`: one byte at `0xf580e`, the rs2 field), and every line quoted was verified
present in the bitstream the board evidence came from — `80843404c` versus HEAD differs only by two
`ifndef SYNTHESIS` instrumentation hunks, zero deletions.
