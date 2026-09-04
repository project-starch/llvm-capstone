# The monitor-regen boot hang: the recorded root cause cannot be the cause

**Date:** 2026-07-28
**Touches:** `plans/large-ro-delivery-completion-task-A.md` §1-STATUS v2/v3
**Bottom line:** the large-`.rodata` delivery track — and therefore SQLite on
silicon — is blocked on a monitor boot hang whose cause is **still unlocalised**.
The explanation currently recorded for it is contradicted by the evidence in the
same document. Nothing here fixes the hang; this narrows where to look and closes
off one candidate for good.

## Why this was re-opened

SQLite on silicon needs the large-RO delivery path (`plans/sqlite-on-silicon-scoping.md`
Stage 3+). That path needs one monitor change. The monitor cannot currently be
rebuilt: **every regeneration boot-hangs with zero serial; only the checked-in
prebuilt `fw_jump.elf` (md5 `6724bcb3`) boots.**

Today's rung work made the blocker concrete rather than theoretical. `beebs_ns`
was rejected by the glue generator verbatim:

    2512 B of *initialized* data overflows the 12-bit store offset and is not
    copy-eligible (sym='ns_keys', size%8=4)

So the unrolled `li`/`sd` path caps a **single** initialized global at ~2 KB. That
is a hard ceiling, not a tuning knob, and SQLite's static tables are far past it.
The copy path is the only way up, and the copy path needs the monitor.

## The candidate that was never tried, and is now refuted

`plans/large-ro-delivery-completion-task-A.md` §1-STATUS v3 reports the isolation
was run with capstone-c at `8cda52c` (drifted master) and at `4899cf9`, and
concluded the good monitor "was built by a different compiler state not
reproducible from the current capstone-c".

Neither of those is the pin the firmware repo actually declares.
**`caplifive-system` pins `sw/capstone-c` at `508342a`** ("fix: incorrect linear
array indexing in rvalue", on `bugfix`), and `4899cf9` is merely the merge-base of
that branch and master:

    4899cf9  (merge-base)  fix: incorrect codegen for loading from dyn addresses
      ├─ master : 8cda52c  ANON_IRDAY_NODE_ID for temps in linear addr dyn offset
      └─ bugfix : 3780447  Fixed overly large alignment for gct
                  508342a  fix: incorrect linear array indexing in rvalue   <-- the pin

The bugfix side even carries an *alignment* fix, which reads like a plausible
boot-hang candidate. So this was worth testing.

**Tested without touching any firmware** — built `508342a` in a throwaway git
worktree (the submodule working tree was never checked out; see the standing rule
about destroying uncommitted submodule work) and ran the regen command straight
out of `caplifive-buildroot/Makefile:26` into a temp file:

    capstone-c --abi capstone <wrapper.c> -- -I<capstone-sbi> -D__riscv_xlen=64

Result — `508342a` vs the current tree's `.c.S` (same source, built by `8cda52c`),
**2 differing lines**, and both are the same line:

    5438c5438
    < .align 4
    ---
    > .align 16

That is it. The two compiler versions produce byte-identical monitor asm apart
from one alignment directive, and the direction is *away* from the good monitor
(the good `.c.S` has `.align 16`, like master; `508342a` is the one that changes
it). **Refuted.** The capstone-c version is not the variable, and this cost no
board time and no firmware risk.

### Reconciling with `ref/HOW-TO-LAUNCH-ON-FPGA.md`

That reference already records the opposite conclusion, dated 2026-07-25:

> **Known fix (2026-07-25):** the working firmware is built by `caplifive-system`'s
> pinned `capstone-c` = branch `bugfix`@`508342a` [...] So for any monitor rebuild,
> build with **that** compiler, not our tree's.

The measurement above is not a contradiction of that as stated, but it does bound it.
What was tested here is the **buildroot** monitor (`caplifive-buildroot/components/
opensbi/.../sbi_capstone_dom.c.S`, the QEMU `fw_jump` input), and for that file the
two compilers differ by one `.align` directive. If the 2026-07-25 note is about
**caplifive-system's own** monitor source — a different tree — it may still hold
there; that has not been checked either way.

What can be said without qualification: **for the buildroot monitor, switching to
`508342a` changes nothing that could fix a boot hang**, and the note should not be
read as a fix for *that* rebuild. Anyone acting on it should first confirm which tree
they are rebuilding.

## The recorded root cause cannot produce the observed failure

§1-STATUS v3 attributes the breakage to register allocation: the good monitor uses
`s0–s6` with frame −368, every regen uses `s0–s11` with frame −464. That difference
is real — it reproduces exactly. But attributing the boot hang to it requires the
differing code to *run at boot*, and it does not.

Diffed the good `.c.S` against a fresh regen and attributed every differing line to
its enclosing label:

| lines differing | function |
|---:|---|
| 35 | `_create_domain.7` |
| 31 | `_create_domain.9` |
| 22 | `_create_domain.0` |
| 21 | `create_domain` |
| 21 | `_create_domain.ret` |
| 13 | `_create_domain.6` |
| 13 | `_create_domain.8` |
| 2 | `_cap_env_init.ret` |

The `_cap_env_init.ret` entry is not a code difference: those two lines are the
trailing `.align 16` vs `.align 4` above, plus the line-number shift. `cap_env_init`
itself is byte-identical.

So **100 % of the real codegen differences are inside `create_domain`** — which is
an SBI call handler invoked when userspace asks for a domain, long after boot.
§1-STATUS v2 already recorded the observation that settles it: the rebuilt monitor
hangs *"with ZERO serial (no OpenSBI banner)"*, and *"`create_domain` isn't even
called at boot"*.

A function that never executes cannot hang the boot. v2 and v3 contradict each
other, and v2 has the direct observation. The larger frame is a real codegen
difference and a real toolchain drift, but it is **not** the failure — and a bigger
frame with more callee-saved registers is, on its own, perfectly correct code.

## What this changes

The blocker is not "find the compiler that built the prebuilt". It is **"find what
in the rebuilt firmware fails before the OpenSBI banner"**, and that is a different
and more tractable search — nothing in it depends on archaeology about a lost
toolchain state.

Untried, in rough order of cost:

1. **Is `sbi_capstone_dom.c.S` even implicated?** The regen also regenerates
   `capstone_int_handler.c.S`, and the buildroot link step is a third input. This
   analysis covers only `sbi_capstone_dom.c.S` — no known-good backup of the int
   handler was found, so it is **unexamined**, and it is the obvious next suspect
   precisely because it *is* live early.
2. **Splice, don't rebuild.** Take the known-good `.c.S`, apply only the large-RO
   copy hunk to it, and rebuild. If that boots, the copy is delivered and the regen
   question can stay open indefinitely. This is the cheapest route to unblocking
   SQLite and it sidesteps the hang entirely rather than solving it.
3. **Localise the hang** with the board's gdb (halt, read `pc`) or by bisecting the
   firmware inputs one generated file at a time.

## Hazard found in passing — the checked-in `.c.S` is the BROKEN one

`components/opensbi/lib/sbi/sbi_capstone_dom.c.S` in the working tree is md5
`6dfe662a`, dated Jul 24 17:02 — the `s0–s11` / frame −464 regen. Only
`fw_jump.elf` was restored on 2026-07-24; the generated asm was not. The `.c.S` has
no `%.c.S: %.c` rule that would refresh it from a source edit, so **any buildroot
rebuild from this tree links the broken monitor**, silently, for both lanes.

Known-good copies still exist outside this repo:
`.../llvm-capstone-b/.../opensbi-custom/lib/sbi/sbi_capstone_dom.c.S.orig`
(md5 `b7baff6f`, the one that matches the good monitor's `s0–s6` / −368 prologue)
and a second, older `.orig` (md5 `37d354d5`) that differs from both by ~1,700 lines
and is probably from an earlier source revision — do not use it.

These are in temp/scratch locations and will not survive. **Preserving `b7baff6f`
somewhere durable is worth doing before anything else here**, since it is currently
the only artifact from which the working monitor can be rebuilt.
