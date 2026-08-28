# SPEC VIOLATION — every capability `mcause` from the DATA path is one code too high, and 25 aliases

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

**`mcause = 25` has two sources that mean different things, and nothing in `mcause` separates
them.** A data-path `UNEXPECTED_OPERAND` — an operand that is not a capability — arrives as 25
because of the offset. A PC-capability `INVALID_CAPABILITY` — a revocation-node validity failure on
the fetched instruction — arrives as 25 correctly. These are unrelated defects with unrelated fixes.

The only field that discriminates them is `mtval`:

    core/ex_stage.sv:490       tval = fu_data_i[0].operand_a    (the rs1 cursor; 0 for an integer)
    core/commit_stage.sv:604   tval = commit_instr_i[0].pc      (never zero)

and on this silicon `mtval` has never been shown to carry a non-zero value for a capability cause,
so it is not currently able to make that distinction. A positive control for it is built and
staged; until it reports, any `mcause 25` on this platform is ambiguous between the two.

Note also that the PC-capability check is gated on `priv_lvl_i == PRIV_LVL_M && capmode_i`
(`core/commit_stage.sv:208`), so it can only fire on an M-mode PC — which is a second, independent
discriminator wherever the faulting PC is known to be outside the monitor's address range.

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
