# `csdrop` (DROP) implemented in capstone-qemu — the LINEAR / row-11 QEMU unblock

**Date:** 2026-07-09
**Lane:** B-lane (compiler/codegen + emulator)
**Task:** `capstone/agent-handoff/tasks/agentB-002-csdrop-row11-linear.md`
**Submodule commit:** `capstone-qemu` `cf541a1f` → `2e6a67d1` (branch `capstone-bootstrap-b`)

## Why

Stage-2 corpus **row 11 (LINEAR, `go_double_finalize`, double-free)** was deferred
as "blocked intra-domain" (see
`08-07-2026_13-01-23_linear-uninit-rows-blocked-intra-domain.md`, Blocker 2). The
fork's `__builtin_capstone_cap_drop` already lowers to a `drop` mnemonic, but the
emulator had **no `csdrop`**, so the instruction decoded as an *illegal
instruction* rather than executing as a clean capability operation. Implementing
`csdrop` is the QEMU-lane unblock for LINEAR. (Row 14 UNINIT is A's firmware lane
and was explicitly out of scope.)

## Spec semantics (authoritative)

`capstone-spec/parts/cap-man-insn.adoc` `[#drop]` and `insn-list.adoc`:

- Encoding: **Func3 = `001`, Func7 = `0001011`, opcode `1011011`**, single operand
  `rs1` (a capability `C`); `rd`/`rs2` unused. (One below MOVC's `0001010`.)
- Behaviour: DROP **invalidates a capability**.
  - Exception `Unexpected operand type (24)` when `x[rs1]` is not a capability.
  - If `x[rs1].valid == 0` (already invalid): **no-op**.
  - Otherwise set `x[rs1].valid = 0`.
- Note DROP has **no** `Unexpected capability type` exception — it is defined for
  **any** capability type, *unlike* MREV (which asserts `type == LIN`). So DROP is
  **not** restricted to linear caps; linear caps are simply its primary user,
  since a consumed linear handle must leave no usable copy behind.

## How it maps onto this emulator

This QEMU has **no explicit `valid` bit** on a `CapFat`; a capability's spec
"valid" bit is the register **`tag`** (`CapRegVal.tag`). A tagged register is a
live capability; clearing the tag turns it into a plain (non-capability) value,
and any later deref of an untagged register already raises
`RISCV_EXCP_UNEXP_OP_TYPE` (**cause 24**, "Cap mem access requires capability") in
`_helper_access_with_cap` (`op_helper.c:984`). So:

- `helper_csdrop` clears `rs1` to `CAPREGVAL_NULL` (tag = false) when it is a live
  capability, consuming the handle;
- if `rs1` is already untagged it raises `RISCV_EXCP_UNEXP_OP_TYPE` (spec
  "Unexpected operand type"), mirroring the deref-path tag check;
- a subsequent use of the dropped register therefore faults **cleanly** (cause 24)
  instead of executing as an illegal instruction.

## Pieces (all in `capstone-qemu`, additive)

| File | Change |
|---|---|
| `target/riscv/insn32.decode` | `csdrop 0001011 ..... ..... 001 ..... 1011011 @r` |
| `target/riscv/helper.h` | `DEF_HELPER_2(csdrop, void, env, i32)` |
| `target/riscv/insn_trans/trans_capstone.c.inc` | `trans_csdrop` (rs1-only, modeled on `trans_csrevoke`) |
| `target/riscv/op_helper.c` | `helper_csdrop` (spec-faithful, type-agnostic) |

Rebuilt `qemu-system-riscv64` with the standard host toolchain (`CC=/usr/bin/gcc`)
at `-j 80`; the decoder (`decode-insn32.c.inc`) regenerated cleanly.

## Validation (under `qemu-system-riscv64`)

A domain naturally holds ordinary (NONLIN) capabilities to its own globals, and
because `csdrop` is **type-agnostic** the full consume→use→fault cycle is
demonstrable **in-lane** without linear authority:

- **Control** (`csdrop_live`): read a live global cap twice, no drop → **ok**,
  `Called dom retval = 571408478` = `0x220F005E`.
- **Fault** (`csdrop_use_after`): read the live cap (ok), `__builtin_capstone_cap_drop`
  it (`drop a3` in the `-O0` asm), then re-read the dropped register (`lbu a3,
  0(a2)`) → serial shows
  `Cap mem access requires capability: ... rs1 = x12` and
  `domain halted by capability fault: cause = 24`, **no retval**. Register `x12`
  (`a2`) is exactly the dropped cap in the asm. This is a **clean capability
  fault**, not an illegal instruction — the row-11 mechanism.

**Regressions:** canonical `run-smoke.sh` passes; the borrow-revoke UAF (R-shape)
Stage-2 probe passes (`region revoked`). The change is purely additive (a new
opcode that was previously illegal), so existing instructions are untouched.
(Several runs hit a transient boot-timing infra flake — exit 75 / not reaching the
shell prompt — on the loaded machine; re-runs at the default timeout pass. Not
related to the instruction.)

## What row 11 still needs from A-lane (firmware lane)

`csdrop` is the *instruction*. A full **before→after row-11 domain demo** of a
double-free (mint a genuine LINEAR cap → use → drop → double-drop/use → trap)
still needs **intra-domain linear authority**, which is A's sign-off-gated
`my_first_domain/start.S` / firmware change (Blocker 1 in the deferral note;
`start.S` delinearises `sp`/`gp` before `domain_main`, and linearity cannot be
fabricated from NONLIN). Until then, the demonstration above uses an ordinary cap
as the drop subject — which fully exercises the instruction's decode/execute/
consume/clean-fault semantics, but exercises the *runtime trap*, not the
*linear-type exclusivity* (that remains a compile-time/type-system property). The
linear-authority piece is unchanged and remains owned by A.

## Test artifacts

The two probe sources live in the session scratchpad (not committed — they need
A's `runtime-qemu` domain harness, which is A's held `capstone/tests` dir, to
build/run; per this task's "work only in `capstone-qemu`" scope they were not
added to A's authority suite). If desired later, they can be promoted into
`capstone/tests/capstone-authority/domains/` as `csdrop_*` probes with oracle
lines (a `csdrop_use_after` → tag-fault, `csdrop_live` → ok), analogous to the
`subobjfield_*` set — that is a cross-lane addition and would be done with the
same additive discipline used for the C1 v1 probes.
