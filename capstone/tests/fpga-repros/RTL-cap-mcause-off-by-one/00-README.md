# SPEC VIOLATION — every capability `mcause` from the DATA path is one code too high, and 25 aliases

> **2026-09-07 (from the 2026-09-05 sweep):** **Not exercised in the 2026-09-05 sweep**; status unchanged (code-level RTL defect, cost is misclassification). Listed here so the folder does not read as silently current. (sweep table: `docs/plans/bug-sweep-2026-09.md`; registry: `docs/ref/ISSUES.md`)

**Status: code-level RTL defect, verified against the reference model. Not reproduced on silicon
as a functional failure — its cost is misclassification, not miscomputation.**

Sibling issues, so a reader who arrived with the wrong symptom is redirected now:
`../S12-wherecode-notcap-operand-vs-memory/` is where the aliasing was noticed and where it bites;
`../S07-capability-untagged-on-reload/` and `../S06-untagged-ldc-stc-high-half/` classify faults by
`mcause` and are affected by the same offset. This folder is only about the cause NUMBERING.

## The defect

The core computes capability exception causes from a 4-bit enum, and it uses **two different
bases** for the same enum depending on where the exception was raised.

    core/ex_stage.sv:348-358      typedef enum logic[3:0] {
                                    NO_EXCEPTION            = 0,
                                    UNEXPECTED_OPERAND      = 1,
                                    INVALID_CAPABILITY      = 2,
                                    UNEXPECTED_CAP_TYPE     = 3,
                                    INSUFFICIENT_PERMISSION = 4,
                                    OUT_OF_BOUNDS           = 5,
                                    ILLEGAL_OPERAND_VALUE   = 6, ... } cap_exception_t;

    DATA path, FLU        core/ex_stage.sv:481    cause = 64'd24 + exception_code[3:0]
    DATA path, DYN        core/cva6.sv:1523       cause = 64'd24 + exception_code[3:0]
    PC-capability path    core/commit_stage.sv:216,219,223,226
                                                  cause = 26 / 27 / 28 / 25, commented "(23 + n)"

The reference model settles which base is right:

    caplifive-system/hw/qemu/target/riscv/cpu_bits.h:693-700
        RISCV_EXCP_UNEXP_OP_TYPE   = 0x18   /* 24 */   <- enum 1, so the base is 23
        RISCV_EXCP_INVALID_CAP     = 0x19   /* 25 */   <- enum 2
        RISCV_EXCP_UNEXP_CAP_TYPE  = 0x1a   /* 26 */   <- enum 3
        RISCV_EXCP_INSUF_CAP_PERMS = 0x1b   /* 27 */   <- enum 4
        RISCV_EXCP_CAP_OOB         = 0x1c   /* 28 */   <- enum 5
        RISCV_EXCP_ILLEGAL_OP_VAL  = 0x1d   /* 29 */   <- enum 6

**Base 23 is correct. `commit_stage.sv` uses it; `ex_stage.sv` and `cva6.sv` use 24.** So every
capability fault raised by the FLU or the DYN unit is reported one code too high:

    raised                     silicon mcause    reference mcause    reads as
    UNEXPECTED_OPERAND              25                24             INVALID_CAP
    INVALID_CAPABILITY              26                25             UNEXP_CAP_TYPE
    UNEXPECTED_CAP_TYPE             27                26             INSUF_CAP_PERMS
    INSUFFICIENT_PERMISSION         28                27             CAP_OOB
    OUT_OF_BOUNDS                   29                28             ILLEGAL_OP_VAL

The misaligned/illegal remaps (enum 7/8/9) are special-cased ahead of the addition on both data
paths and are NOT affected.

## Why it matters more than an off-by-one usually does

**`mcause = 25` has THREE sources that mean different things, and nothing in `mcause` separates
them.** This count has now been wrong twice in this file: an earlier version said two, a later one
said four. Three is the number, and the retraction of the fourth is recorded below because it
changes what this folder claims about the LSU.

    1  core/ex_stage.sv:479,488          FLU     24 + code   tval = fu_data_i[0].operand_a
    2  core/commit_stage.sv:226,604      pc_cap  23 + 2      tval = commit_instr_i[0].pc
    3  core/cva6.sv:1516,1521            DYN     24 + code   tval = capstone_dyn_ftval

**RETRACTED 2026-09-15 — the former source 4.** This file said
`core/load_store_unit.sv:1005-1009` was "the LSU's **bounds** check reporting cause 25" and called
it "the worst of them". Both halves are wrong, and were wrong at the date they were written.
`:1005-1009` is the `always_ff` reset block, not a cause assignment. The actual clauses, read at
`6fe7a49ab` and unchanged since, are:

    load_store_unit.sv:974   NOT_CAP                      64'd24
    load_store_unit.sv:977   not LINEAR/NONLIN            64'd26
    load_store_unit.sv:980   LOAD  && !perm[2]            64'd27
    load_store_unit.sv:983   STORE && !perm[1]            64'd27
    load_store_unit.sv:987   out of bounds                64'd28
    load_store_unit.sv:990   revocation node not valid    64'd25

So the LSU's bounds check reports **28**, which is correct, and the clause that reports 25 is the
**revocation-node validity** check, for which 25 is also correct. The LSU is not a fourth
mis-numbered source; it is numbered right. An out-of-bounds access is NOT indistinguishable from a
revocation failure on this silicon — that sentence is withdrawn.

**And the LSU is evidence, not a defect.** Its literals are base 23 + ordinal throughout, they are
original to 2026-05-10 and were never edited, and `commit_stage.sv:216-226` carries the same
numbering with the arithmetic spelled out in its comments ("23 + 3", "23 + 4", "23 + 5"). Together
with the spec (`capstone-academic-spec/parts/int-except.adoc:21-27`) and QEMU
(`capstone-qemu/target/riscv/cpu_bits.h:693-699`, `0x18..0x1e`) that is four sources on base 23
against the two execute-path encoders on base 24. `docs/ref/ISSUES.md` recorded this under R-24 on
2026-09-10; this file had not caught up.

How the error happened is worth one line, because it is this project's most expensive shape: the
line range was read from a stale offset and the clause identified from the line ABOVE the one that
sets the cause. The check that would have caught it is reading the assignment and its guard
together rather than the line number alone.

Source 3 carries a further hazard for anyone reading `tval`: `capstone_dyn_ftval` is a LATCHED
register, so it reads zero when it was never armed. A `tval == 0` from the DYN path is genuinely
no data, whereas the same value from the FLU path is a reading. The 2026-08-31 positive control
(`tval = 0xBEEF`) proves only the **FLU** path live and says nothing about the DYN latch or the
LSU's `lsu_ea_full`.

For the rest of this document, "the two paths" means sources 1 and 2 — the pair that the S-12
investigation actually has to tell apart, because `mepc` there names a `cincoffsetimm`, which
`core/decoder.sv:1167,1294-1299` shows is FLU-only. A data-path `UNEXPECTED_OPERAND` — an operand that is not a capability — arrives as 25
because of the offset. A PC-capability `INVALID_CAPABILITY` — a revocation-node validity failure on
the fetched instruction — arrives as 25 correctly. These are unrelated defects with unrelated fixes.

The only field that discriminates them is `mtval`:

    core/ex_stage.sv:490       tval = fu_data_i[0].operand_a    (the rs1 cursor; 0 for an integer)
    core/commit_stage.sv:604   tval = commit_instr_i[0].pc      (never zero)

and on this silicon `mtval` has never been shown to carry a non-zero value for a capability cause,
so it is not currently able to make that distinction. A positive control for it is built and
staged; until it reports, any `mcause 25` on this platform is ambiguous between the two.

**Do NOT try to use privilege or address range as a second discriminator.** An earlier version of
this file argued that because the PC-capability check is gated on `priv_lvl_i == PRIV_LVL_M`
(`core/commit_stage.sv:208`), a faulting PC outside the monitor's range could not be a pc_cap
fault. That is wrong twice over. ~~Capability domains on this platform run **in M-mode** (the
documented S-12 wedge state is `MPP=M`), so the gate is satisfied *by* the domain~~ — **that first
reason is WITHDRAWN, see below; domains run in S-mode and the gate is never satisfied by them.**
The second reason stands and is sufficient on its own: the `0x80000000`–`0x80800000` pair at
`:200-201` is a constructed capability's bounds, not a gating range. `tval` is the only
discriminator.

The instruction in bold above is UNCHANGED by the withdrawal. It is if anything stronger: a domain
cannot satisfy the privilege gate at all, so privilege still cannot separate the two paths.

**How it was resolved, kept because the shape recurs.** On 2026-09-15 this file's "domains run in
M-mode" clause was found to contradict
`docs/history/15-09-2026_lsu-capmode-gate-why-domains-cannot-satisfy-it.md`, which argued from
monitor source that domains run in **S-mode** and can never satisfy a `PRIV_LVL_M` gate. The
conflict was flagged here rather than decided, because the two sides rested on different KINDS of
evidence — an inference from an observed wedge state against quoted source — and the folder that
owns the question is R-34's, not this one. It was adjudicated the same day, below.

**ADJUDICATED 2026-09-15 (R-34's lane, as the flag asked): the S-mode reading is right and the "domains run in M-mode" clause is WITHDRAWN.** The wedge evidence was misread rather than wrong: `mstatus.MPP` records *the privilege that was interrupted*, written on trap entry as `mstatus_d.mpp = priv_lvl_q` (`core/csr_regfile.sv:2100`). So `MPP = M` at a wedge says the faulting code was in M-mode — the monitor, which is where a domain's fault lands once the domain's own trap vector is unset (M-1) — and says nothing about the privilege the domain itself ran at. The monitor `mret`s into S-mode, and the entering `mret` clears `MPRV` when `MPP != M` (`:2320`), so a domain cannot reach `ld_st_priv_lvl == PRIV_LVL_M` by any route. The independent confirmation is the E1 matrix: stale plain loads retire inside a domain, which is only possible with the gate unsatisfied. This changes nothing in the paragraph's conclusion — `tval` remains the only discriminator.

## What would fix it

Change the two data-path sites from `64'd24 +` to `64'd23 +`, matching `commit_stage.sv` and the
reference. Nothing else in the core derives a capability cause.

**This changes the value software sees.** The monitor's exception switch and every recorded
classification in this repository were written against the current, shifted numbering, so the fix
and the software that reads it have to move together — which is why this is filed as an observation
with a proposed fix rather than applied.

## Not verified here

* Whether the reference model actually raises `UNEXP_OP_TYPE` for the same conditions the RTL
  raises `UNEXPECTED_OPERAND` — only that the two enumerations disagree by one.
* Whether any recorded verdict in this repository is wrong *because* of the offset. Every
  classification to date was made against silicon and interpreted with the silicon numbering, so
  they are internally consistent; the exposure is to anything comparing silicon against QEMU by
  cause number, and to the `mcause 25` aliasing above.
