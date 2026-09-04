# Revocation #70 follow-ons (b) clean fault delivery + (a) linear re-share

*2026-07-03. Both remaining #70 follow-ons, done and validated together. The
recording+enforcement halves already landed (`8b6a47f322`). (b) A caught
use-after-revoke inside a domain call left the monitor **spinning** instead of
returning to the caller (`run-revoke-matrix-probe.sh` hung in round 2). (a)
Re-sharing a **linear** (`REV_BORROWED`) borrow after revoke asserted at
`helper_csmrev` once revoke became active — this had turned `run-hostcall-all.sh`
red across a class of probes. This note covers the empirical root-causes (both
corrected earlier assumptions), the fixes, the corrected two-monitor
architecture, and validation.

**Two-monitor architecture (learned the hard way).** There are two separate
Capstone monitors, both built from a `sbi_capstone.c` (distinct nested-submodule
copies):
- **OpenSBI firmware** (`fw_jump.elf`, submodule `components/opensbi`,
  `lib/sbi/capstone-sbi/sbi_capstone.c`): the **M-mode** monitor handling the
  lender's **SBI ecalls** — `region create/share/revoke`, i.e.
  `shared_region_annotated` / `mrev` / `revoke`. This is where the **(a)** linear
  re-share runs.
- **`sbi.dom`** (submodule `package/capstone-sbi-domain/capstone-sbi`, same
  filename): the domain substrate loaded via `create_dom`, whose `_cap_trap_entry`
  handles a **borrower's in-domain fault** (`handle_exception` / `swap_cpmp`).
  This is where **(b)** clean fault delivery runs.
A fix must go in the copy that actually runs for the path; editing the wrong copy
silently no-ops (cost me a full rebuild+test cycle on (a)).

## 1. Empirical root cause — corrected the proposal's premise

The proposal (`design/domain-fault-delivery-proposal.md`, "Corrected Step B")
assumed the round-2 fault was raised **in C-mode (`priv == PRV_C`)** and should
escape via QEMU's async domain-switch path in `do_interrupt`. Running the probe
at HEAD showed otherwise.

The revoke-matrix borrower is an **S-mode** domain (`revoke_matrix_probe.smode.c`,
talks to the monitor by `ecall`). Sequence in round 2 (serial log):
- recording works: `region revoked`;
- the borrower reloads its cached borrow **untagged** (`Cap(0, 0x7, …)` — tag 0,
  the enforcement demote-on-reload);
- the borrower's `slot[0] = STAGE2` store through the now-untagged base is **not**
  a capability-bounds fault but a plain access → a **CPMP miss**, delivered to the
  monitor's `ctvec` handler `_cap_trap_entry → handle_exception` with a
  store-access cause;
- `handle_exception` calls `swap_cpmp(badaddr)`, which searches `regions[]` for a
  region covering `badaddr`, finds none (the region was revoked), and hit
  `capstone_error(CAPSTONE_NO_CPMP_REGION)` = `C_PRINT(0xdeadbeef); C_PRINT(0x2);
  while(1)` — **the spin**.

So the hang was the monitor's `swap_cpmp` treating an unrecoverable domain fault
as if it were a recoverable CPMP miss, then dying in `capstone_error`. This is a
**different channel** from the authority suite's cap-bounds faults, which are
`RISCV_EXCP_CAP_OOB` raised at `priv == PRV_C` and terminated by QEMU's
Step-A clean-halt (`exit(0)`) — never touching the monitor. The two are
independent; enforcement itself was already correct (store dropped).

## 2. Design decision resolved: monitor-owns (not QEMU-synthesize)

The proposal flagged a TCB choice — QEMU synthesizes a domain-exit vs. the monitor
owns termination. The root cause resolves it toward **monitor-owns**, and cleanly:
`return_from_domain()` (`*caller_buf = retval; __domreturnsaves(caller_dom, …)`) is
**already invoked from inside `_cap_trap_entry`** on the normal `DOM_RETURN` ecall
path (`handle_trap_ecall → return_from_domain`, never returns). Calling the same
machinery from the fault path (`handle_exception`, same trap context) reuses a
proven, working unwind — no QEMU/TCB trap-path change needed.

## 3. The fix (monitor, `package/capstone-sbi-domain`)

`capstone-sbi/sbi_capstone.h`: add sentinel
`CAPSTONE_DOMAIN_FAULT_RETVAL 0x0FA017EDu` (the retval the caller sees when a
domain is terminated by a fault).

`capstone-sbi/sbi_capstone.c`: add
```c
static void fault_return_from_domain(unsigned cause) {
    C_PRINT(CAPSTONE_ERR_STARTER); C_PRINT(cause);   /* log the cause */
    return_from_domain(CAPSTONE_DOMAIN_FAULT_RETVAL); /* clean unwind, no return */
}
```
and replace the two spin-to-death sites:
- `swap_cpmp`: when no region covers `badaddr` →
  `fault_return_from_domain(CAPSTONE_NO_CPMP_REGION)` (was
  `capstone_error(NO_CPMP_REGION)`);
- `handle_exception` `default:` (non-access-fault causes, e.g.
  `RISCV_EXCP_INVALID_CAP`) → `fault_return_from_domain(cause)` (was
  `capstone_error(UNKNOWN_EXCP)`).

**Only currently-dead paths change.** Both edited sites previously ended in
`while(1)`; a recoverable CPMP miss (region found) and every normal monitor
operation are untouched. So the change cannot regress anything that presently
passes — it converts two hangs into a clean, reported domain termination.

`sbi.dom` is a runtime-loaded domain (`/test-domains/sbi.dom`), a **separate copy**
from the OpenSBI-firmware monitor (`components/opensbi/.../sbi_capstone.c`). The
probe exercises `sbi.dom`; the firmware copy has the identical two spin sites and
is a documented parallel follow-on (§6).

Build note: rebuild via `make build CAPSTONE_CC_PATH=$(realpath capstone/capstone-c)
A=capstone-sbi-domain-rebuild` from `caplifive-buildroot`. Buildroot sanitizes the
env, so the package's `cargo`-based `sbi.dom.c → sbi.dom.c.S` step can silently
emit an **empty** `.c.S` (undefined refs at link) if `CAPSTONE_CC_PATH` doesn't
reach the package sub-make; generate the `.c.S` with a direct `cargo run -- --abi
capstone` if that happens.

## 4. Validation

- `run-revoke-matrix-probe.sh` — **PASS** (exit 0), both cases 2 (memory-stored
  pointer) and 3 (stc/ldc a separate cap slot): `region revoked` → `round 2
  returned 0xfa017ed` (clean fault delivery) → `use-after-revoke did not update
  lender view (word=0x1111…1111)` (STAGE2 store dropped). No spin/timeout; the
  runner's stale "store landed" success marker was updated to the fixed behavior,
  and its header refreshed.
- **Authority suite — PASS** (`__CAPSTONE_AUTHORITY_SUITE_PASSED__`): all
  bounds-fault probes still detect faults; normal monitor paths (region share,
  CPMP swap on a *found* region, domain call/return) exercised and green. Confirms
  no regression and that authority's channel is independent of the edited paths.
- **Hostcall suite (`run-hostcall-all.sh`) — GREEN after (a) below** (12/12
  probes). Before the (a) fix it failed at `hostcall-file-open-close-probe` with
  `helper_csmrev: Assertion CAP_TYPE_LIN` during its "payload revoked and
  re-shared" step — the linear re-share, exposed once revoke became active
  (`8b6a47f322`) and *not* caught at recording-fix merge time (that fix was
  validated on the revoke-matrix and payload-revoke probes only).

## 7. Follow-on (a): linear (`REV_BORROWED`) re-share after revoke

**Root cause = an ISA gap.** Revoking a linear borrow leaves the retained handle
**UNINIT** (data not retained), and `helper_csrevoke` set its `cursor = base`.
But the only UNINIT→LIN transition, `csinit`, asserts `cursor == end` (the
canonical UNINIT form), and `scc` (set-cursor) refuses UNINIT — so the handle was
stranded in a state no instruction could advance, and `mrev` (which needs LIN)
asserted. This is exactly the runtime-author question that was drafted; the fix
resolves it by experiment (author greenlit experimenting on the check).

**Fix (two parts):**
1. **QEMU** `helper_csrevoke` (`op_helper.c`): position the retained handle's
   cursor by type — LIN → `base` (ready to use), UNINIT → **`end`** (the canonical
   `csinit`-able form). Global (both monitors' `revoke` route through it).
2. **Monitor** `shared_region_annotated` `REV_BORROWED` branch, in **both**
   `sbi_capstone.c` copies (firmware + `sbi.dom`): if the handle came back UNINIT
   (`cap_type(r)==3`), `csinit(→LIN, offset 0)` before `mrev`. A `C_INIT` macro
   (raw `.insn`, funct7 `0x9`) emits `csinit` since no `__init` builtin exists.

`csinit` remains a **required, explicit reclaim step** — I only made its
precondition reachable — so linear-borrow exclusivity is unchanged. This is an
**experimental revocation-semantics choice**; confirm with the runtime author
(the drafted question, held under `/tmp/capstone/`, is now answered by
construction: non-linear re-share already worked; linear re-share now works via
`init`-before-`mrev` with `revoke` leaving `cursor==end`).

## 8. Result — revocation fully end-to-end

Both #70 follow-ons resolved and validated on clean images:
- **record → enforce → clean delivery (b):** a use-after-revoke (or any
  unrecoverable domain fault) **terminates the domain and returns to the caller**
  (sentinel `0x0FA017ED`); the store never lands; no abort, no spin.
- **linear re-share (a):** revoke → `init` → re-`mrev` re-lend now works;
  `run-hostcall-all.sh` is green (12/12).

Validated: `run-revoke-matrix-probe.sh` PASS (both cases), authority suite
`__CAPSTONE_AUTHORITY_SUITE_PASSED__`, `run-hostcall-all.sh` 12/12. QEMU's
`helper_csrevoke` change only affects an explicit `csrevoke` (never issued by
CoreMark/RV8/BEEBS), so the benchmark suites are unaffected by construction.

**Change locations (nested submodules):**
- `capstone-qemu` (`capstone-bootstrap`): `target/riscv/op_helper.c`
  (`helper_csrevoke` UNINIT→cursor=end).
- `package/capstone-sbi-domain/capstone-sbi` (`sbi.dom`): `sbi_capstone.c`
  (`fault_return_from_domain` + `C_INIT` + `csinit` re-share) + `sbi_capstone.h`
  (`CAPSTONE_DOMAIN_FAULT_RETVAL`).
- `components/opensbi` (firmware): `sbi_capstone.c` (`C_INIT` + `csinit` re-share).
- parent: `tests/runtime-qemu/run-revoke-matrix-probe.sh` (markers/header) + this
  note.

## 9. Residual follow-ons

- **Firmware fault-delivery is intentionally NOT unified.** `sbi.dom` and the
  OpenSBI firmware are two checkouts of the *same* `capstone-sbi` repo, so I first
  tried putting the (b) `fault_return_from_domain` fix in both. It regressed the
  hostcall suite: the **firmware's** `swap_cpmp`/`handle_exception` also service
  **host S-mode** traps (the OS kernel's CPMP misses on shared regions), where
  there is no in-progress domain call — so `return_from_domain(caller_dom)` has no
  valid target. The (b) fix is therefore **`sbi.dom`-only** (its faults are always
  a borrower inside a domain call); the firmware keeps `capstone_error` on those
  paths. The **csinit re-share (a) fix goes in both** (both do the lender's
  `mrev`). Net: the two `capstone-sbi` checkouts hold *different* content and get
  *different* commits — firmware = csinit-only, `sbi.dom` = csinit + fault_return.
  A proper firmware (b) path would need to distinguish "in a domain call" from
  "host S-mode trap" before returning — future work if a firmware-mediated domain
  fault ever needs clean delivery (not exercised by current probes).
- **Author confirmation**: the (a) revocation-semantics choice (revoke leaves
  UNINIT at `cursor==end`; `init`-before-`mrev` re-lend) is experimental — worth a
  one-line confirmation with the runtime author, though it is now validated by the
  hostcall suite as oracle.
- **Domain un-resumability** after a (b) fault-termination (saved S-mode context
  left as-is) — harmless for probes; a clean design would mark it un-resumable.
- **Domain un-resumability**: a fault-terminated domain's saved S-mode context is
  left as-is; a subsequent call to the same `dom_id` would resume the faulted
  context. Harmless for the probes (they exit), but a clean design would mark a
  faulted domain un-resumable. Minor.
