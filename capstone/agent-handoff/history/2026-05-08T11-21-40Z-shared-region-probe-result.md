# Shared-region sentinel probe result

Timestamp: 2026-05-08T11-21-40Z

## Goal

Execute the narrowest next runtime step after the failed HostCall v0 attempt:

> prove that an `sbi.dom + .smode` payload can mutate one host-shared region in a way visible to the guest userspace helper.

## Implemented probe

Host side:
- create one region
- map it
- initialize `word[0] = 0`
- share it into `sbi.dom`
- call the domain twice
- inspect `word[0]` after each call

S-mode side:
- find the last shared region through `REGION_COUNT` / `REGION_QUERY(BASE)`
- write `0x1111111111111111`
- `ecall`
- write `0x2222222222222222`
- `ecall`

## Files used

Probe sources:
- `capstone/tests/runtime-qemu/shared-region-probe/shared_region_probe.h`
- `capstone/tests/runtime-qemu/shared-region-probe/shared_region_probe_guest.c`
- `capstone/tests/runtime-qemu/shared-region-probe/shared_region_probe.smode.c`
- `capstone/tests/runtime-qemu/build-shared-region-probe.sh`
- `capstone/tests/runtime-qemu/run-shared-region-probe.sh`

## Commands used

Build log:
- `/tmp/capstone/build-shared-region-probe.txt`

Runtime wrapper log:
- `/tmp/capstone/run-shared-region-probe-wrapper.txt`

Full QEMU serial log:
- `/tmp/capstone/capstone-runtime-qemu-shared-region-probe.log`

## Observed output

Key observed lines:

- `shared-region-probe: created domain ID = 0`
- `shared-region-probe: created shared region ID = 0`
- `shared-region-probe: initial word = 0x0000000000000000`
- `shared-region-probe: region shared`
- `shared-region-probe: call 1 retval = 0`
- `shared-region-probe: word after call 1 = 0x0000000000000000`
- `shared-region-probe: stage 1 sentinel mismatch (observed=0x0000000000000000)`

## Verdict

The probe **failed**.

At least in the currently tested `sbi.dom + custom .smode` path, the host-visible
mapped region did not reflect the sentinel write expected from the `.smode` payload.

## Practical consequence

Do **not** resume the full `HC_WRITE_STDOUT` shared-memory RPC PoC yet.

The next smallest step is now diagnostic:
- determine why the sentinel probe did not become visible,
- likely by comparing the `sbi.dom` wrapper path with known working region-sharing
  S-mode consumers in the tree (`miniweb`, `nullb_split`),
- and proving whether `.smode` execution and `REGION_QUERY` state behave as assumed.

