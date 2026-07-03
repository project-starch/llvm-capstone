# Proposal: clean delivery of in-domain capability faults (retire the `do_interrupt` abort hack)

*Status: PARTIALLY IMPLEMENTED (2026-07-03). The abort hack is retired: an
in-domain capability fault now **halts cleanly with a reported cause and
preserved output** instead of aborting the emulator (QEMU-only change, see
"Implemented" below). The stronger **return-to-host** delivery (domain terminates,
host regains control and continues) remains a proposal — it needs a monitor-side
fault-return path that does not exist yet. This touches the trusted trap path
(QEMU `do_interrupt` + the OpenSBI capstone monitor) and the authority harness.*

## Implemented (2026-07-03): clean halt, output preserved (Option 1, QEMU-only)

`target/riscv/cpu_helper.c`, `riscv_cpu_do_interrupt`, `if(env->cap_mem)` branch:
replaced `assert(env->priv < PRV_C)` with — for the in-domain case
(`env->priv == PRV_C`) — a structured diagnostic
(`[CAPSTONE] domain halted by capability fault: cause=… pc=… tval=… badaddr=…`)
plus `fflush` + `exit(0)`. Rationale and evidence:
- `env->priv == PRV_C` is exactly the case the old assert forbade; a monitor
  host-trap arrives with `priv < PRV_C` and a valid `ctvec`, so the monitor's
  host-trap delivery path below is **byte-for-byte unchanged**.
- `abort()` did not flush stdio, so a mid-run domain's buffered serial output was
  lost (the gaps 8/9 "no serial output" symptom). `exit()` flushes → output now
  survives; the fault is a clean halt with a named cause, not a SIGABRT crash.
- **Step A empirically proved** the horizontal-trap (`ctvec`) path cannot deliver
  this fault: a domain installs no `ctvec`, so it aborts one line later on
  `assert(env->ctvec.tag)`. Hence a clean halt (not `ctvec` redirect) here.
- **Follow-up (not done): return-to-host.** True delivery — domain terminates,
  host domain-launcher regains control with an error and continues — needs a
  monitor fault-return path. Today the monitor has **no receiver**: `__domasync`
  is defined but unused, `DOM_REENTRY_POINT` is a stub (`j _dom_reentry`), and the
  domcall/domreturn ABI is compiler-builtin. See "Corrected Step B" below; that
  is the next, separately-scoped step.

## Problem

When a capability violation fires **inside a domain** (bounds fault, tag fault,
unaligned cap access, `cjalr` on an untagged pointer, …), QEMU does **not** deliver
a trap — it aborts:
```c
// target/riscv/cpu_helper.c, riscv_cpu_do_interrupt(), under `if (env->cap_mem)`
assert(env->priv < PRV_C); /* TODO: a hack */
```
Because **`PRV_C == PRV_M`** (both `3`; C-mode = `PRV_M` + `env->cap_mem==true`),
this assert fires on *every* domain-mode fault. Consequences:
- a violation is an **emulator crash**, not a catchable trap;
- buffered domain output is **lost** (the abort pre-empts the host's read) — this is
  why SQLite gaps 8/9 showed no serial output;
- there is **no clean "the domain was stopped because it violated a capability"**
  event — which is exactly the security property the paper wants to demonstrate.

## What already exists (the intended model)

- The same `if (env->cap_mem)` block **already has** the delivery path: an explicit
  `if (env->priv == PRV_C)` "horizontal trap" branch that sets `cepc` (as a
  capability), `mcause`, and `pc_redirect_to_cap(env, &env->ctvec.val.cap)`, then
  `riscv_cpu_set_mode(PRV_C)`. The `assert` sits *above* it and prevents it running.
- The trusted **OpenSBI capstone monitor sets the trap vector**:
  `sbi_capstone_dom.c` does `C_WRITE_CCSR(ctvec, _cap_trap_entry)` and installs a
  handler stack/seal/code region. So `assert(env->ctvec.tag)` in the delivery path
  should already hold in domain context.
- The monitor's handler (`capstone_int_handler.c.S`) reads `mcause` and can
  `domreturn(rd, rs1, rs2)` (custom insn `0x5b/0x1/0x21`) back to the domain's
  launcher — i.e. the machinery to **report a fault and return to the host** exists.

## The catch — the authority suite depends on the abort

`tests/capstone-authority/run-authority-suite.py` detects a fault by the
`"Cap mem access {OOB,requires capability}"` diagnostic **and explicitly accepts the
QEMU abort (EOF) as the valid terminal state** (its own comment: "a capability fault
inside a domain hits a `riscv_cpu_do_interrupt` assertion, so we accept EOF"). So the
abort is today's de-facto fault signal; changing delivery **must** update this
harness in lockstep or the 25/0 suite breaks.

## Options

1. **Minimal report-and-halt (QEMU only).** Replace the `assert` with a clean
   diagnostic (`cause`, `pc`, `tval`) + a controlled stop (no abort). Low effort, no
   OpenSBI change; makes faults tidy and output-preserving, but is *not* a real
   catchable trap (no return-to-host, no monitor involvement).
2. **Full trap delivery (QEMU + monitor + harness).** Remove the hack so a PRV_C
   fault redirects to `ctvec` → OpenSBI handler reports the cause and `domreturn`s to
   the host with an error, so the host prints "domain faulted (cause=…)" and exits
   cleanly. Requires: (a) verify/extend `capstone_int_handler` for the *synchronous*
   cap-fault causes (today's paths are interrupt/`domcall`-oriented); (b) update the
   authority harness terminal expectation from "message + EOF/abort" to "message +
   clean domain termination / host error line". This is the security-model-correct
   behavior and the paper-worthy one.
3. **Hybrid (recommended).** Do (2), but **keep emitting the existing
   `Cap mem access …` diagnostic** so the harness's message match still works; only
   its *terminal* expectation changes (EOF-abort → clean exit). Falls back to (1) if
   the OpenSBI handler needs more than a small change to handle synchronous causes.

## Recommendation

**Option 3, in two reviewable steps:**
- **Step A (QEMU):** retire `assert(env->priv < PRV_C)`; let the existing PRV_C
  horizontal-trap path run; keep the diagnostic print. Empirically test with a
  deliberate-OOB probe domain: confirm the fault reaches `_cap_trap_entry` and the
  monitor `domreturn`s to the host (vs. looping or hitting `assert(ctvec.tag)`).
- **Step B (monitor + harness):** if Step A shows the handler doesn't cleanly
  terminate on a synchronous cap fault, add that path in `capstone_int_handler`
  (report cause, `domreturn` an error); update `run-authority-suite.py` to the clean
  terminal; add a probe `cap_fault_clean_delivery` asserting a domain OOB is
  **reported and the domain terminates without a QEMU abort**.

## Step A result (2026-07-03) — the ctvec path is the wrong channel

Ran Step A empirically: retired `assert(env->priv < PRV_C)`, rebuilt QEMU, and ran
the deliberate-fault probes (`global_oob`, `forge_inttoptr`) plus an `ok` control
(`global_inbounds`) through the authority harness. Findings:

- The authority classifier still **passes** (it keys on the `Cap mem access …`
  diagnostic + absence of a retval, and `run_domain` already accepts either a
  returned prompt or EOF) — so message matching is not the obstacle.
- **But delivery does not become clean.** Instead of the first assert, control
  now reaches the *next* line and aborts on **`assert(env->ctvec.tag)`**
  (`cpu_helper.c`, horizontal-trap path). Concrete log:
  `Cap mem access OOB: pc=829e0170 … bounds=(829e0220,829e0260)` immediately
  followed by `Assertion 'env->ctvec.tag' failed`.
- **Root cause:** a **domain never installs a `ctvec`.** Only the *monitor*
  installs one (`_cap_trap_entry`, which handles host S/U-mode traps — ecall,
  CPMP-miss `handle_exception` → `swap_cpmp`, timer). So the horizontal-trap
  path in `do_interrupt` — which redirects to `env->ctvec` — is meant for the
  **monitor** faulting in cap mode, **not** for a domain. Delivering an in-domain
  fault through `ctvec` is the wrong channel.

**The correct channel is the existing domain-switch escape.** QEMU already has a
path that escapes a *running domain* back to its parent monitor: the async
cap-interrupt branch (`if (async && env->cap_mem && env->cis &&
capstone_int_can_take(env))`). It: `swap_domain_scoped_regs(… SWAP_OUT)`, seals
`env->cih` as a `SEALEDRET` return cap into `ra` (`gpr[1]`), passes the cause in
`a0` (`gpr[10]`), nulls `cih` (masks further interrupts), and returns to `PRV_C`
(the monitor). `capstone_int_can_take(env)` (`cih.tag && type==SEALED &&
async==SYNC`) is precisely "we are inside a domain with a live parent handler" —
the right discriminator for "escape to monitor" vs. "top-level monitor fault."

### Corrected Step B (supersedes the ctvec-based Step B above)

- **QEMU (`do_interrupt`, `if(env->cap_mem)` synchronous branch):** before the
  `ctvec` redirect, if `capstone_int_can_take(env)` (in a domain), deliver the
  synchronous cap fault via the **same domain-switch-to-parent path** the async
  branch uses (swap out domain regs, seal `cih` into `ra`, pass the cause in
  `a0`, return to `PRV_C`) — *not* through `ctvec`. Keep the diagnostic. Only the
  top-level-monitor case (no `cih`) keeps/needs the `ctvec` path.
- **Monitor (`sbi_capstone.c`):** the domain launcher (`call_domain` /
  `__domcallsaves`) must recognize a **fault-return** (domain came back with a
  cap-violation cause, not a normal `domreturn`) and surface it — report the
  cause and terminate the domain to the host, rather than resuming it. Confirm
  whether the existing `__domcallsaves` return path already distinguishes an
  async/cause re-entry (the async-interrupt return uses this same machinery) or
  whether a small new branch is needed.
- **Harness:** with clean delivery, the terminal becomes a returned shell prompt
  (domain terminated, cause reported) instead of a QEMU abort/EOF; update
  `run-authority-suite.py`'s expectation accordingly and add the
  `cap_fault_clean_delivery` probe.

**Open design decision for review:** whether QEMU should *synthesize* the
domain-exit for a synchronous fault (symmetry with the async path, QEMU-heavy) or
whether the monitor should own more of it. This is a trusted-path (TCB) choice —
hence pausing here for direction before implementing Step B.

## Validation gate

- Authority suite **25/0** preserved under the new detection.
- SQLite (base + extended) still passes; a deliberately-faulting SQLite variant now
  **reports a clean fault** instead of aborting QEMU (and its buffered output is
  visible).
- CoreMark / RV8 / BEEBS unaffected (they do not fault).
- New probe demonstrates: domain capability violation → clean trap + reported cause +
  domain terminated, **no emulator abort**.

## Why propose-first

This edits the **trusted** trap path (QEMU `do_interrupt` + the OpenSBI monitor that
is the TCB) and a **currently-green** harness that is coupled to the present
behavior. A wrong move destabilizes the authority suite (25/0) and the just-completed
SQLite result. Small, security-critical, cross-component → review the direction
before implementing.
