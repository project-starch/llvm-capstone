# Second-`PENDING` diagnostics: metadata-only path works; repeated borrowed payload reuse reproduces `helper_csmrev`

Timestamp: 2026-05-14T20:05:00Z

## Summary

Two narrower diagnostics were added and run after the earlier failed multi-op file-object attempt.

The result is now more precise:

- a minimal metadata-only probe shows that one domain invocation can return `PENDING` twice in the current environment,
- but a second diagnostic that reuses/re-shares the same borrowed output payload across that next round reproduces the QEMU assertion in `helper_csmrev`.

So the current blocker is **not** simply "multiple `PENDING` returns are unsupported".
It is narrower and currently appears tied to repeated borrowed-output payload reuse/re-share across rounds.

## Diagnostic 1: metadata-only second `PENDING`

Files added:

- `capstone/tests/runtime-qemu/hostcall-second-pending-probe/hostcall_second_pending_probe_guest.c`
- `capstone/tests/runtime-qemu/hostcall-second-pending-probe/hostcall_second_pending_probe.smode.c`
- `capstone/tests/runtime-qemu/build-hostcall-second-pending-probe.sh`
- `capstone/tests/runtime-qemu/run-hostcall-second-pending-probe.sh`

Observed result:

- first `call_dom()` returned `PENDING`,
- helper responded,
- second `call_dom()` returned `PENDING` again,
- helper observed the second-stage request,
- third `call_dom()` returned `DONE`.

Key serial markers:

```text
hostcall-second-pending-probe: first call retval = 1
hostcall-second-pending-probe: second call retval = 1
hostcall-second-pending-probe: second pending observed
hostcall-second-pending-probe: third call retval = 0
hostcall-second-pending-probe: success
__HOSTCALL_SECOND_PENDING_OK__
```

## Diagnostic 2: second `PENDING` with payload reuse/re-share

Files added:

- `capstone/tests/runtime-qemu/hostcall-second-pending-payload-probe/hostcall_second_pending_payload_probe_guest.c`
- `capstone/tests/runtime-qemu/hostcall-second-pending-payload-probe/hostcall_second_pending_payload_probe.smode.c`
- `capstone/tests/runtime-qemu/build-hostcall-second-pending-payload-probe.sh`
- `capstone/tests/runtime-qemu/run-hostcall-second-pending-payload-probe.sh`

Observed result:

- stage 1 request with borrowed output payload returned `PENDING`,
- helper snapped the request successfully,
- helper printed that it was about to re-share the payload for round 2,
- the next step reproduced:

```text
qemu-system-riscv64: ../target/riscv/op_helper.c:700: helper_csmrev: Assertion `rs1_v->val.cap.type == CAP_TYPE_LIN' failed.
```

Key serial markers:

```text
hostcall-second-pending-payload-probe: first call retval = 1
hostcall-second-pending-payload-probe: snapped stage1 request{phase=1 opcode=6 offset=0 length=14}
hostcall-second-pending-payload-probe: about to re-share payload for round 2
qemu-system-riscv64: ../target/riscv/op_helper.c:700: helper_csmrev: Assertion `rs1_v->val.cap.type == CAP_TYPE_LIN' failed.
```

This localizes the current limitation more tightly than the earlier file-object attempt did.

## Current interpretation

The best current hypothesis is:

- metadata-only multi-round control flow can work,
- the failing edge is connected to repeating the borrowed output payload share/use discipline on the next round,
- the bug/limitation likely sits below the guest helper logic itself, on the runtime/OpenSBI/QEMU capability-state path.

## Follow-up authoritative answer

The runtime/QEMU author later confirmed the intended rule:

- if a region is already borrow-shared, it must be revoked before anything else is
  done to it, including borrow-sharing it again,
- the current "borrowed output" behavior is part of the SBI C-mode abstraction
  rather than evidence that repeated borrowed re-share without revoke should work.

So this diagnostic remains useful, but the interpretation tightened further:

- the metadata-only probe demonstrates that multi-`PENDING` control flow itself is
  not the issue,
- the payload-reuse probe demonstrates the required borrow-share lifecycle rule by
  showing what happens when the helper omits the required revoke step.

## Useful question for the runtime/QEMU author

The refined question is now:

> metadata-only second-`PENDING` works, but re-sharing the same borrowed output payload for the next round reproduces `helper_csmrev`; is repeated borrowed output payload re-share supposed to be supported by the current runtime/QEMU path, and if so, which capability-state invariant is being violated around that re-share/return path?

## Revalidated working baseline after adding these diagnostics

```bash
cd /home/alexey/dev/llvm-capstone
bash capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh
```


