# HostCall follow-up and current blockers

Timestamp: 2026-05-08T11-21-40Z

This note is intended for a future LLM/session.
It preserves:
1. the substance of the immediately previous assistant answer,
2. the current blocker analysis after attempting the next minimal runtime step.

---

## A. Substance of the previous assistant answer

The previous answer established the following:

- The attempted full `HC_WRITE_STDOUT` HostCall v0 proof was **not validated**.
- The useful validated outcome from that turn was **not** the HostCall PoC itself, but:
  - `capstone/tests/runtime-qemu/run-domain-smoke.py` now supports `--guest-command`,
  - the existing `run-smoke.sh` baseline still passes,
  - the handoff notes were updated to reflect the narrower blocker.
- The conclusion from that turn was:
  - do **not** claim the full HostCall RPC round trip is working yet,
  - instead first prove a much narrower fact: can an `sbi.dom + .smode` payload mutate one host-shared region in a way visible to the guest userspace helper?

That answer also concluded that the next minimal step should be:

> one shared region, one `.smode` payload, a sentinel write, and host-side verification that the sentinel became visible after `call_dom()`.

---

## B. What was done next

That exact narrower probe was implemented and executed.

Files added for the probe:
- `capstone/tests/runtime-qemu/shared-region-probe/shared_region_probe.h`
- `capstone/tests/runtime-qemu/shared-region-probe/shared_region_probe_guest.c`
- `capstone/tests/runtime-qemu/shared-region-probe/shared_region_probe.smode.c`
- `capstone/tests/runtime-qemu/build-shared-region-probe.sh`
- `capstone/tests/runtime-qemu/run-shared-region-probe.sh`

The guest helper does:
- `create_dom("/test-domains/sbi.dom", <custom .smode>)`
- `create_region(4096)`
- `map_region()`
- initialize `word[0] = 0`
- `share_region(dom_id, region_id)`
- `call_dom()` twice
- check whether the shared word changed to stage-1 and stage-2 sentinels.

The `.smode` payload does:
- query `REGION_COUNT`
- query `REGION_QUERY(BASE)` for the last region
- write sentinel stage 1 to `region[0]`
- `ecall`
- write sentinel stage 2 to `region[0]`
- `ecall`

---

## C. Observed result

The probe **failed**.

Observed guest-visible output:

- `shared-region-probe: created domain ID = 0`
- `shared-region-probe: created shared region ID = 0`
- `shared-region-probe: initial word = 0x0000000000000000`
- `shared-region-probe: region shared`
- `shared-region-probe: call 1 retval = 0`
- `shared-region-probe: word after call 1 = 0x0000000000000000`
- `shared-region-probe: stage 1 sentinel mismatch (observed=0x0000000000000000)`

So the host-visible mapped region remained unchanged.

---

## D. Current blockers / problems discovered

### 1. `call_dom()` scalar return is not a reliable status channel in this path
Evidence:
- even the existing `/sbi-dom.user` + `/test-domains/sbi.smode` path returns `0` on repeated calls.
- therefore we cannot currently build the HostCall ABI around the assumption:
  - `.smode` does `DOM_RETURN(x)`
  - guest userspace helper reliably observes `x` via `call_dom()`.

### 2. Shared-region mutation is not yet visible back to the guest helper
Evidence:
- the new one-region sentinel probe leaves the host-visible mapped word unchanged.

This means at least one of the following is false in the current `sbi.dom + .smode` path:
- the custom `.smode` code actually executes as assumed,
- `share_region()` installs region state in the structure later consulted by S-mode `SBI_EXT_CAPSTONE_REGION_QUERY`,
- the returned region capability/base is actually usable for writes in this wrapper path,
- or the guest helper’s mapped region is not the same memory that the `.smode` path sees.

### 3. The blocker is now narrower and more concrete than “HostCall PoC failed”
The immediate problem is specifically:

> why does the `sbi.dom + .smode` shared-region sentinel probe leave host-visible shared memory unchanged?

That is the blocker to solve before going back to full shared-memory RPC.

---

## E. Likely resolution strategy

Recommended next diagnosis path:

1. Compare `sbi.dom` / `capstone-sbi` wrapper behavior against a known in-tree S-mode region-sharing consumer, especially:
   - `capstone/caplifive-buildroot/package/capstone-nested-enclave/capstone_split/sdom/miniweb_backend.smode.c`
   - `capstone/caplifive-buildroot/package/capstone-null-blk/capstone_split/sdom/nullb_split.smode.c`
2. Determine which of these hypotheses is true:
   - the `.smode` payload is not executing as expected,
   - `share_region()` into `sbi.dom` does not populate state later used by S-mode `REGION_QUERY`,
   - the queried capability is not writable/usable in this wrapper path,
   - the host-side mapping is not coherently observing the same region contents.
3. Add the smallest possible targeted diagnostic for one hypothesis at a time.

Most likely next diagnostics:
- inspect `sbi.dom` generated/runtime path vs working split-S-domain region-sharing paths,
- prove whether `.smode` sees `REGION_COUNT >= 2` in this path,
- prove whether the custom `.smode` code runs at all in a way externally observable.

---

## F. What should be committed vs not committed

Safe to commit:
- generalized QEMU harness support for `--guest-command` in `run-domain-smoke.py`,
- handoff updates reflecting the new blocker,
- handoff reorganization into `current/` and `history/`.

Not ideal to commit as a validated milestone yet:
- the new shared-region sentinel probe as if it were a passing regression test,
- any claim that HostCall v0 shared-memory RPC is working.

If committing probe files at all, do it only as an explicitly investigative/WIP change.

