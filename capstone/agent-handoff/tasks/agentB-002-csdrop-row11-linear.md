# Agent-B task 002 — implement `csdrop` in capstone-qemu (the LINEAR / row-11 QEMU unblock)

*Hand this whole file to Agent-B (`claude-b`), clone `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`.*

---

You are **Agent-B** (compiler/codegen + **emulator** lane). Your v1 subobject-bounds
work was merged into canonical `capstone-bootstrap` by Agent-A (fast-forward to
`c4758de`), so canonical and your branch are now equal. Obey `./CLAUDE.md` and
`capstone/agent-handoff/{MULTI-AGENT-WORKFLOW,COORDINATION}.md`.

## Why this task

Two Stage-2 corpus rows are deferred as "blocked intra-domain"
(`capstone/agent-handoff/history/08-07-2026_13-01-23_linear-uninit-rows-blocked-intra-domain.md`).
They split cleanly by lane:

- **Row 11 (LINEAR, `go_double_finalize`, double-free)** — *your* lane. Blocker 2:
  `__builtin_capstone_cap_drop` already lowers to a `drop` mnemonic, but **the
  emulator has no `csdrop`**, so the instruction can't execute — it would be an
  illegal instruction, not a clean capability fault. Implementing `csdrop` is the
  QEMU-lane unblock.
- **Row 14 (UNINIT, `cpython_uninit_connection`)** — **NOT your task.** It needs a
  new *monitor* op (`share_uninit_region`) in the firmware (capstone-c /
  capstone-sbi), which is Agent-A's gated firmware lane. **Do not touch firmware
  submodules and do not attempt row 14.**

## Strict scope (lane rules)

- Work **only** in `capstone/capstone-qemu`. You WILL bump the `capstone-qemu`
  gitlink — that is your lane and is allowed; **log the bump** in COORDINATION's
  submodule-bump log in the same commit.
- **Do NOT** touch `caplifive-buildroot`, `opensbi`, `capstone-sbi`, `capstone-c`,
  or `capstone/my_first_domain/start.S`. In particular, **do not** try to give a
  domain linear authority by editing `start.S` — that is a sign-off-gated firmware
  change owned by Agent-A (see Blocker 1 in the history note). Your deliverable is
  the *instruction*, not the domain's linear-authority source.
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**.

## Steps

1. **Spec first.** Read `capstone/capstone-spec` for the `drop` / `csdrop`
   instruction: encoding, operand form (the note shows `drop a2`), and exact
   semantics — it consumes/invalidates a **linear** capability. Confirm the
   post-drop register state the spec mandates.
2. **Study the existing pattern** in capstone-qemu:
   - `target/riscv/insn32.decode` — instruction formats.
   - `target/riscv/insn_trans/trans_capstone.c.inc` — translation functions.
   - `target/riscv/op_helper.c` — the linear-type helpers to mirror:
     `helper_csinit` (~:857, asserts `CAP_TYPE_UNINIT`), `helper_csrevoke` (~:710,
     sets a handle's type to `LIN`/`UNINIT`), `helper_csmrev` (~:731, asserts
     `rs1.type == CAP_TYPE_LIN`). These show how linear caps and type asserts are
     handled — `csdrop` follows the same shape.
3. **Implement `csdrop`:** add the decode entry, a `trans_*` translator, and
   `helper_csdrop` — assert the input is `CAP_TYPE_LIN`, consume it (set the
   register to the spec's post-drop invalid/null capability), so that a **later use
   of the dropped register faults as a clean capability fault**, not an illegal
   instruction.
4. **Test at the QEMU level.** Add a focused test proving: the instruction decodes
   and executes, consumes the linear cap, and a subsequent use of the dropped
   register produces a clean capability fault. Note: a *domain* can't currently
   mint a linear cap (that's the gated `start.S`/firmware piece owned by A), so an
   end-to-end row-11 *domain* demo is **blocked on A**. Demonstrate `csdrop`
   semantics via whatever linear-authority vehicle is testable in your lane (a
   QEMU unit / hand-built LIN cap in a test harness, or the monitor-minted-linear
   path if reachable without touching firmware source). Document precisely what
   remains gated on A.
5. **No regressions:** existing capstone-qemu instruction tests pass; the R/H/S
   Stage-2 probes are unaffected.

## Coordination + checkpoint

- Update `state/current-state.B.md`, `current-next-step.B.md`, and your
  COORDINATION **Current position** line at the checkpoint.
- Commit small/often on `capstone-bootstrap-b`; log the capstone-qemu bump; push
  the branch at the checkpoint so A can integrate (A merges B→canonical, as with v1).
- **STOP at the checkpoint and report:** the spec semantics you implemented, the
  test result (consume→use→clean fault), and a clear statement of what row 11 still
  needs from A (intra-domain linear authority) for the full before→after demo.

## Not in scope right now

- Subobject-bounds **increment 2** (embedded-struct/scalar fields) stays **PI-gated**
  (container_of policy) — do not start it.
- Row 14 (UNINIT) — A's firmware lane, as above.
