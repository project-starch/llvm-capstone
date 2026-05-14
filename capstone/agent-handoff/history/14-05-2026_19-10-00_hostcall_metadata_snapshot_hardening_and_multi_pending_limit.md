# HostCall metadata snapshot hardening validated; multi-`PENDING` handle-path attempt hit a QEMU/runtime limit

Timestamp: 2026-05-14T19:10:00Z

## Summary

This step did two things:

1. hardened the currently validated HostCall helper proofs against metadata/payload TOCTOU-style misuse by snapshotting shared request state immediately after the first `call_dom()` return, and
2. attempted the next architectural slice (a handle-based multi-op file-object path), but removed that experimental code after it hit a reproducible QEMU/runtime assertion and therefore did **not** qualify as validated baseline.

## What was successfully validated

The following helper-side probes now snapshot the round-1 request before host-side service:

- `capstone/tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe_guest.c`
- `capstone/tests/runtime-qemu/hostcall-filewrite-probe/hostcall_filewrite_probe_guest.c`
- `capstone/tests/runtime-qemu/hostcall-fileread-probe/hostcall_fileread_probe_guest.c`

Practical meaning:

- the metadata region may remain `INOUT + SHARED`,
- but the helper now copies the request fields it intends to trust into a local snapshot,
- and for output-style requests it also snapshots the borrowed payload bytes before performing host work,
- so host-side service no longer depends on repeated reads of mutable shared state during the service window.

This is not a full security proof, but it is a concrete hardening step and a better discipline for future service growth.

## What was attempted but not promoted to baseline

An experimental handle-based file-object path was tried:

- `FILE_OPEN`
- helper response
- second domain `PENDING` for the next file operation

Observed result:

- after the first successful `FILE_OPEN` service round, the next attempted re-entry hit:

```text
qemu-system-riscv64: ../target/riscv/op_helper.c:700: helper_csmrev: Assertion `rs1_v->val.cap.type == CAP_TYPE_LIN' failed.
```

This happened during the attempted continuation beyond the already validated two-round shape.

The experimental files were removed from the tree so the repository only keeps validated or intentionally documented paths.

## Updated conclusion

The current validated HostCall baseline remains:

- one `PENDING` return,
- helper-side service from a snapped request,
- one completion return.

The stable file-service subset still makes architectural sense, but future sessions should first determine whether multiple successive `PENDING` returns from one domain invocation are actually supported by the current runtime/QEMU path.

## Revalidated commands

```bash
cd /home/alexey/dev/llvm-capstone
bash capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh
bash capstone/tests/runtime-qemu/run-shared-region-probe.sh
```

