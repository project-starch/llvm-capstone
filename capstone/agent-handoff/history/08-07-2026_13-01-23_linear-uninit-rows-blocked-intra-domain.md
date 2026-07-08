# LINEAR (row 11) and UNINIT (row 14) are blocked intra-domain — deferred

**Date:** 2026-07-08
**Context:** SQLite-Capstone Stage-2. With the three revocation-family shapes
BORROW-REVOKE (R), HIERARCHICAL-REVOKE (H), and SEALED-CALLBACK (S) all validated on
RTL (15 of 17 in-scope corpus rows), the two remaining shapes were LINEAR (row 11
`go_double_finalize`, double-free) and UNINIT (row 14 `cpython_uninit_connection`,
use-before-init). The corpus plan optimistically framed these as "small,
self-contained, no revoke scaffold." This note records the empirical finding that
**both are blocked in the current toolchain/emulator** as clean intra-domain probes,
and the decision to defer them.

## What was investigated

Both shapes need a genuine capability of the right *type* — a LINEAR (move-only) cap
for double-free, a UNINIT cap for use-before-init. The LLVM fork exposes the full
C-level surface (`clang/include/clang/Basic/BuiltinsCapstone.td`):
`__builtin_capstone_cap_drop` (consume/invalidate a linear cap),
`__builtin_capstone_cap_init` (initialise an uninit cap), `cap_mrev`, `cap_revoke`,
`cap_delin`, `cap_get_tag`, etc. Spikes compiled cleanly and lowered to real ISA
mnemonics (`drop a2`, `init a2, a0, a2`). The **runtime** is where they fail.

## Blocker 1 — UNINIT: no intra-domain source of an uninit cap

`helper_csinit` asserts its input is already `CAP_TYPE_UNINIT`
(`capstone-qemu/target/riscv/op_helper.c:857`). A UNINIT cap arises **only from a
revoke** (`op_helper.c:710` sets the revoked handle's type to `LIN` or `UNINIT`;
the `713`–`718` comment documents UNINIT as "a linear node was revoked, data not
retained", cursor at END — the canonical form `csinit` requires). Revoke needs a REV
handle from `mrev`, and `helper_csmrev` asserts `rs1.type == CAP_TYPE_LIN`
(`op_helper.c:731`). A domain has **no linear authority**: `my_first_domain/start.S`
delinearises `sp`/`gp` before `domain_main`, and linearity cannot be fabricated from
NONLIN. So a domain cannot mint an UNINIT cap. This is the **same wall that suspended
#78 phase-2** (see `06-07-2026_18-00-00_revoke-on-free-step0-linear-authority-finding.md`):
obtaining intra-domain linear authority needs a sign-off-gated `start.S` change, and
that finding also documented a follow-on blocker (`gp` is a ~592-byte small-data
pointer, not an arena-covering capability).

## Blocker 2 — LINEAR: `csdrop` is not implemented in this QEMU

The fork's `__builtin_capstone_cap_drop` lowers to a `drop` instruction, but there is
**no `csdrop`** in the emulator. The implemented Capstone instructions
(`target/riscv/insn_trans/trans_capstone.c.inc`, `insn32.decode`) are: `csmovc,
cscincoffset(imm), csscc, cslcc, csrevoke, csmrev, csshrink, csshrinkto, cssplit,
cstighten, csdelin, csinit, csseal, csccsrrw, csldc, csstc, cscjalr, cscbnz, cscall,
csreturn, cscapenter` (+ debug ops). No drop. So a "consume the linear handle, then
use it → trap" probe cannot execute; it would be an illegal instruction, not a clean
capability fault. Linear *exclusivity* in this stack is therefore a compile-time /
type-system property (linear types leave no second consumable copy), not a runtime
trap we can demonstrate today.

## Feasible-but-costly alternative (not taken)

The **monitor** (capstone-c) does hold linear authority — it mints region caps LINEAR
via `split_out_cap` and already produces/handles UNINIT caps (a `revoke_region`
leaves the retained handle UNINIT; `share_child_region` re-inits with `C_INIT` when
`cap_type == 3`). So UNINIT (row 14) *is* achievable via a new monitor op
(`share_uninit_region`: hand the domain an uninit cap → read faults → then init → ok)
plus a two-domain probe — a `share_child_region`-scale cycle (monitor op + kernel/lib
plumbing + probe + firmware rebuild + the 4-level nested-submodule commit chain).
LINEAR (row 11) has no clean RTL-trap vehicle regardless, because `csdrop` is absent
from the emulator.

## Decision (2026-07-08)

**Defer rows 11 and 14.** Rationale: R/H/S are the paper's *load-bearing* temporal-
safety shapes and are validated on RTL (15/17 in-scope rows); rows 11 and 14 are one
row each and explicitly non-load-bearing. Spending a firmware-op cycle (UNINIT) plus a
sign-off-gated `start.S` change (LINEAR, and even then LINEAR needs `csdrop` support
QEMU lacks) on two single rows is lower-leverage than folding the R/H/S wins into
the paper and doing the API-classification pass (priority 2). The scoped path above is
recorded so either row can be picked up deliberately later.

## Status of the Stage-2 shapes

| Shape | Rows | Status |
|---|---|---|
| BORROW-REVOKE (R) | 3, 13, 18, 19 | validated on RTL |
| HIERARCHICAL-REVOKE (H) | 4, 5, 7, 8, 9, 10, 12 | validated on RTL |
| SEALED-CALLBACK (S) | 1, 2, 6, 16 | validated on RTL |
| LINEAR (L) | 11 | **deferred** — no RTL vehicle (`csdrop` unimplemented); type-level guarantee |
| UNINIT (U) | 14 | **deferred** — needs a monitor op (`share_uninit_region`) or gated `start.S` linear authority |
| N/A | 15, 17 | out of scope (liveness / non-memory-safety) |
