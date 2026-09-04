# First HostCall stdout probe validated

Timestamp: 13-05-2026 14:10:47 +0800

## Summary

A new minimal split host/service runtime proof was added and validated under the
existing QEMU harness.

This step intentionally stayed inside the top-level repository and did not require
new changes in child repositories.

## What was added

New runtime probe files:

- `capstone/tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h`
- `capstone/tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe_guest.c`
- `capstone/tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.smode.c`
- `capstone/tests/runtime-qemu/build-hostcall-stdout-probe.sh`
- `capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh`

## What the proof does

The new probe validates a tiny two-round HostCall-style request/response flow:

1. the guest helper creates a domain using `/test-domains/sbi.dom` plus a custom `.smode` payload,
2. the helper creates one shared metadata region and one shared payload region,
3. the `.smode` payload writes a fixed string into the payload region,
4. the payload writes a fixed-width `hostcall_v0` request into the metadata region with opcode `HC_V0_OP_WRITE_STDOUT`,
5. the first `call_dom()` returns `HC_V0_RET_PENDING`,
6. the helper prints the payload from ordinary Linux userspace and writes the response back into metadata,
7. the second `call_dom()` returns `HC_V0_RET_DONE`,
8. the metadata ends in `HC_V0_PHASE_DONE`.

## Validation performed

The new build wrapper was executed successfully.

The new runtime wrapper was executed successfully twice.

Observed success markers included:

- `hostcall-stdout-probe: first call retval = 1`
- `hostcall-v0 payload from domain`
- `hostcall-stdout-probe: second call retval = 0`
- `hostcall-stdout-probe: success`

## What changed in the handoff bundle

The handoff files were updated so future sessions know that:

- every completed step must be tested at the relevant layer before it is treated as done,
- after a coherent validated change set, exact commit command(s) and proposed commit message(s) should be reported when a commit is appropriate,
- the validated runtime baseline now includes the first HostCall stdout proof,
- the next recommended micro-step is to keep the metadata region shared but tighten the payload region to a directional borrowed handoff.

