# In-domain capability-fault delivery: Step A experiment + clean-halt (retire the do_interrupt abort hack)

*Status: 2026-07-03. Retires the `riscv_cpu_do_interrupt` abort that turned every
in-domain capability violation into a QEMU SIGABRT (and swallowed the domain's
buffered serial output). See `design/domain-fault-delivery-proposal.md` for the
design/threat framing; this note is the empirical trail + the implemented change.*

## Problem

`target/riscv/cpu_helper.c`, `riscv_cpu_do_interrupt()`, `if(env->cap_mem)` branch
had `assert(env->priv < PRV_C); /* TODO: a hack */`. Because `PRV_C == PRV_M`
(both 3; C-mode = `PRV_M` + `env->cap_mem`), this fired on **every** synchronous
capability fault taken *inside a domain*. Consequences: a violation was an
emulator crash rather than a catchable event, and `abort()` does not flush stdio,
so a domain that faulted mid-run lost its buffered serial output (the "no serial
output" symptom seen in SQLite gaps 8/9).

## Step A experiment — the `ctvec` horizontal-trap path is the wrong channel

Retired the assert alone and let the existing `if (env->priv == PRV_C)`
horizontal-trap path (sets `cepc`/`mcause`, `pc_redirect_to_cap(&env->ctvec)`) run.
Rebuilt QEMU, ran `global_oob` / `forge_inttoptr` / `global_inbounds`.

Result: it does **not** deliver cleanly — control reaches the next line and aborts
on **`assert(env->ctvec.tag)`**. Log:
`Cap mem access OOB: pc=829e0170 … bounds=(829e0220,829e0260)` →
`Assertion 'env->ctvec.tag' failed`.

Root cause: **a domain installs no `ctvec`.** Only the monitor installs one
(`_cap_trap_entry`, which handles host S/U-mode traps: ecall, CPMP-miss
`handle_exception`→`swap_cpmp`, timer). The horizontal-trap path is for the
*monitor* faulting in cap mode, not a domain. Delivering an in-domain fault via
`ctvec` is wrong. Reverted the experiment (submodule kept clean).

## Why return-to-host is not a QEMU-only change

The correct escape channel for a *running domain* is the domain-switch back to the
parent monitor. QEMU already has it for **async** interrupts
(`if (async && env->cap_mem && env->cis && capstone_int_can_take(env))`):
`swap_domain_scoped_regs(SWAP_OUT)`, seal `env->cih` (SEALEDRET) into `ra`, cause
in `a0`, return to `PRV_C`. Its guard `capstone_int_can_take` (`cih` is a live
sealed handler) is exactly "inside a domain with a parent to return to."

But the **monitor has no receiver** for such an escape in this build:
`__domasync` is defined but never used; `DOM_REENTRY_POINT` defaults to the stub
`_dom_reentry: j _dom_reentry`; the domain call/return path is compiler-builtin
(`__domcallsaves` / `__domreturnsaves`, `caller_dom`) and is purely synchronous
(monitor `call_domain` → domain → `return_from_domain` → back). Delivering a
synchronous fault through the async machinery would make the monitor *resume* the
domain (wrong), and there is no fault-return path to unwind to the host launcher.
So true "domain terminates, host regains control and continues" requires new
**monitor** code — a separately-scoped step, flagged in the proposal.

## Implemented (QEMU-only): clean halt with reported cause + preserved output

`cpu_helper.c`, same branch: for the in-domain case (`env->priv == PRV_C` —
precisely what the old assert forbade) print a structured diagnostic and halt
gracefully:

```
[CAPSTONE] domain halted by capability fault: cause = <c>, pc = <pc>, tval = <t>, badaddr = <b>
```
then `fflush(stderr); fflush(stdout); exit(0);`.

- `env->priv < PRV_C` (monitor handling a host trap, `ctvec` valid) is untouched —
  that whole delivery path is byte-for-byte the same.
- `exit()` flushes stdio (unlike `abort()`) → the domain's output now survives; the
  fault is a named clean halt, not a SIGABRT crash.

## Validation gate (all green, pristine QEMU)

- Authority suite **full run: all domains PASS**, `__CAPSTONE_AUTHORITY_SUITE_PASSED__`,
  exit 0. Faulting probes now emit both the `Cap mem access …` diagnostic *and* the
  new `domain halted by capability fault: cause=…` line, then a clean EOF — no
  `Assertion`/`Aborted`/`core dumped` anywhere in the log. The harness classifier
  is unchanged (matches the message; accepts prompt-or-EOF terminal).
- Observed causes: 24 (`RISCV_EXCP_UNEXP_OP_TYPE` = untagged-cap use / tag fault),
  5 (`LOAD_ACCESS` = bounds fault).
- SQLite (base + extended): **PASS**, unchanged — `row name=alpha/beta/gamma`,
  `__CAPSTONE_SQLITE_EXTENDED_PASSED__`, `__CAPSTONE_SQLITE_MEMORY_PASSED__`,
  `QEMU smoke passed`, exit 0; no abort in the serial log. (SQLite exercises no
  deliberate fault, so this confirms the change is inert on the passing path.)

## Follow-up (next, needs approval — touches the monitor TCB)

Return-to-host delivery: (1) QEMU, in the synchronous in-domain branch, escape via
the domain-switch-to-parent path (mirror the async branch) instead of halting;
(2) monitor `sbi_capstone.c` — a fault-return path so `call_domain` /
`__domcallsaves` recognizes a cap-fault return (vs a normal `domreturn`) and
reports+terminates the domain to the host rather than resuming it; (3) harness —
expect a clean host error line + returned prompt, add a `cap_fault_clean_delivery`
probe. Design decision to settle first: QEMU-synthesizes-the-exit vs monitor owns
more of it.
