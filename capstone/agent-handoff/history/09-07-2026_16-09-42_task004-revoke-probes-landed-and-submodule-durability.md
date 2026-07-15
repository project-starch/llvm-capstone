# Task 004 — revoke probes landed in-submodule + csdrop commit durability

**Date:** 2026-07-09
**Lane:** compiler/codegen + emulator
**Branch:** `capstone-bootstrap-b`
**Scope:** `capstone/capstone-qemu` submodule + handoff docs. No firmware touched.

Closes two loose ends from tasks 002 (`csdrop`) and 003 (revoke memory-alias
sweep, outcome **a**): the csdrop submodule commit's durability, and landing the
task-003 probes into the repo so the validation is reproducible.

## Starting state reconciled

This clone was found checked out on canonical `capstone-bootstrap` (not `-b`), and
the branch histories had been rewritten by the integrator/human:

- Superproject `capstone-bootstrap-b` tip `66d912dd` (task 003), parent `a0fa5ef9`
  (task 002 csdrop code, gitlink `cf541a1f`→`2e6a67d1`). Already on
  `origin/capstone-bootstrap-b` (`project-starch/llvm-capstone`).
- Superproject `capstone-bootstrap` (canonical) contains only the csdrop **task
  prompt** file (`c321ac08`), gitlink still `cf541a1f` — i.e. csdrop is **not yet
  integrated** into canonical.

Switched this clone back to `capstone-bootstrap-b` and aligned the submodule to
`2e6a67d1` before working.

## Step 1 — durability of the csdrop submodule commit

The submodule's only remote is `origin = project-starch/capstone-qemu` (the
`.gitmodules` URL). Evidence it is **already durable there**:

```
git -C capstone/capstone-qemu reflog show origin/capstone-bootstrap-b
  2e6a67d112 ...@{0}: update by push
```

A remote-tracking ref reaches that value only after a successful push/fetch of it;
`update by push` means `2e6a67d1` **was pushed to project-starch/capstone-qemu**
from this clone (by the operator, who holds org write access — they push the
superproject the same way). So A can `git submodule update` and resolve the
gitlink from the submodule's own `origin`.

**Credential reality in this session:** the non-interactive shell has **no** push
creds — `GITHUB_TOKEN` invalid, SSH to github blocked, no credential helper, HTTPS
push prompts for a username and fails. So *I* cannot push; the operator must, as
before. The NEW probe commit (step 2) therefore needs one operator push of the
submodule branch (see checkpoint report for the exact command). I did **not** fake
a push or alter `.gitmodules`/`origin`.

## Step 2 — probes landed in the submodule's own test tree

New dir `capstone/capstone-qemu/tests/capstone-revoke-probes/`:

- `csrevoke_probe.h` — the firmware-free LINEAR mint via `csdebuggencap`
  (`.insn r 0x5b, 0x1, 0x40`).
- `revoke_mem_alias.c` (KEY), `revoke_reg_alias.c`, `revoke_unrelated_ok.c`,
  `revoke_mem_control.c` — the four task-003 probes, verbatim.
- `run-revoke-probes.sh` — driver that **reuses** the sibling
  `capstone/tests/runtime-qemu` harness (does not re-implement it); locates it via
  `CAPSTONE_REPO_ROOT` (default 4 dirs up).
- `README.md` — mechanism, reproduction table, and the provenance constraint.

They live in the submodule (not `capstone/tests/`, the firmware lane) so the probe sources
travel with the emulator they exercise.

## Reproduction (binary `2e6a67d1`, one boot per probe)

| Probe | Assertion | Observed |
|---|---|---|
| `revoke_mem_alias` **(key)** | fault | `Cap mem access requires capability`, **cause 24** — memory-resident alias reloads untagged |
| `revoke_reg_alias` | fault | `Cap mem access on revoked capability`, **cause 25** — live cap's rev-node invalidated |
| `revoke_unrelated_ok` | retval `571670579` | `0x22130033` — no over-broad sweep |
| `revoke_mem_control` | retval `571736158` | `0x2214005E` — isolates `REVOKE` as the cause |

The two fault codes expose the two enforcement points of lazy revocation (live
per-deref `capstone_cap_revoked` check → cause 25; reload-time untag in
`helper_reg_set_cap_compressed` after `rev_node_id` is restored from the
compressed cap → cause 24). Both trap ⇒ the sweep provably reaches a memory-
resident copy. This refines the task-003 note, which recorded only cause 24 for
both (the reg-alias path yields the more precise cause-25 message on this binary).

## Harness interaction learned (documented in the driver + README)

A `csdebuggencap`-minted domain has **no lender frame**. When it faults the
monitor halts it but there is no lender to receive a fault sentinel, so the guest
does not return to the shell prompt — the prompt-based smoke harness hangs and
exits non-zero *by design*. The driver therefore asserts fault probes by grepping
the serial log for the monitor's fault line (ignoring the harness exit code) and
treats a boot that reaches `Created domain ID` with **no** fault line as a sweep
regression. OK probes keep the normal marker/exit-code path. Also: a faulting
domain poisons later domain creation in the same boot, so each probe runs in its
own guest.

## Provenance constraint (design input for A's firmware/monitor lane)

A region reached through an independent SBI `REGION_QUERY` mapping is **not** a
tracked descendant of the revocable cap, so `REVOKE` will not touch it (the earlier
borrow-revoke "NO-TRAP-GAP"). For the single-domain BORROW-REVOKE rows (3/13/18/19)
to bite, the region must be delivered to the borrower **through the tracked linear
cap** (the `MREV` ancestor), not re-derived from an SBI query. This is the
constraint the `start.S`/monitor linear-authority work must satisfy.

## Gitlink / integration state

- Submodule: probe commit `e0cd45de` added on top of `2e6a67d1` (see COORDINATION
  submodule-bump log).
- Superproject `capstone-bootstrap-b`: gitlink bumped `2e6a67d1`→`e0cd45de`.
- Operator push still required for both the submodule branch (durability) and the
  superproject branch (see checkpoint report).
