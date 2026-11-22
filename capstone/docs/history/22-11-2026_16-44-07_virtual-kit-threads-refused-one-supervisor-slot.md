# The virtual kit's pthread/tls libc-test failures: one supervisor slot refuses the child context (2026-11-22)

Four libc-tests fail the same way on the kit `perf/devheap/kit-devq`, both before and after delegate-pre-tls
(dev 7046cda1bdb6): `tls_local_exec`, `pthread_tsd`, `pthread_robust` and `cancel-points`. Each faults in join or
cancel on a NULL or garbage `pthread_t`. The libc-test runner (`run-libc-test-delegated.py`) keeps only the fault line,
so the earlier run could not see `pthread_create`'s return code.

## Prior art

- The programme's inventory (`docs/plans/2026-11-20-post-submission-programme.md` section 1.4, on spec-silicon) says
  the native route refuses threads. The virtual profile does not take that route. `runtime/virtual/pthread.c`
  implements `__clone` through `__capstone_virtual_context_create`. The recorded virtual pthread passes ran on
  `x-capstone-supervisor-slots=32` with a module told 32 (`runtime/virtual/results/pthread-matched-250/`,
  `registry-order/`, `exec/`).
- `capstone_vm.c` refuses a thread create when the live frames would exceed `supervisor_slots`. The default is 1,
  the vp2 value, and the error is -EBUSY with a must-fire line: "thread create refused, N supervisor slot(s) in use".
  The vp2 board's tier-12 boots print the same line (`tests/rtl-smoke/vp2-kit-2026-10-17/board-tier12/results/`,
  r12 and r13).
- Q-32 (ISSUES.md) requires the module's `supervisor_slots` to equal QEMU's `x-capstone-supervisor-slots`. The
  check is `slot-match-gate.py`.

## Arm 1: the kit as it runs (1 slot / 1 slot)

Kit `kit-devq` (QEMU wrapper -> `spec/kit-profq`; module `capstone_vm.ko` af7484e5e39019a6; QEMU binary
5b5563063547e5ac). The images are the TIP's from the delegate-pre-tls validation, plus one probe. The probe `pt.dom`
(6e3316518150d358) calls `pthread_create`, prints rc and errno, then joins. The module's load line reads
`supervisor_slots=1`. The -cpu string sets no slot property (QEMU's default is 1).

| image | sha256 (16) | reading |
|---|---|---|
| pt-probe | 6e3316518150d358 | `PT create rc=11 (Resource temporarily unavailable) errno=16 t=0`; exit 10 |
| lt-tls_local_exec | 5b9825899f393250 | `tls_local_exec.c:54: [thread 1]: pthread_create(&td, 0, check, 0) == 0 failed.`, then cause 24 at address 0x40 |
| lt-pthread_tsd | 1177592f9819a589 | `pthread_tsd.c:37: pthread_create(...) failed: Resource temporarily unavailable`, then cause 24 at a garbage address |
| lt-pthread_cond | 4f5a560acc1128e2 | `pthread_cond.c:45: pthread_create(...) failed: Resource temporarily unavailable`, then hangs until the 120 s timeout |

The new guest dmesg lines include four "capstone_vm: thread create refused, 1 supervisor slot in use", one per image.
The probe's launcher stats count one `service3_thread_create` service. The chain of readings:
- the module returns -EBUSY (errno 16 in the probe);
- musl's `pthread_create` maps any clone failure to EAGAIN (rc 11);
- the tests then join or cancel the unset handle, which faults.

So on this kit "pthread never worked" is the expected reading. The cause is not the native-route refusal: it is a
one-slot configuration that admits no second live context. The landed `delegate.c` is not involved.

## Arm 2, pre-registered: the same boot with 32 slots on both sides

One variable changes: `--cpu-prop x-capstone-supervisor-slots=32 --module-param supervisor_slots=32`. The images,
kit and run order are the same. `slot-match-gate.py --applied-log` runs after the boot.

Predicted reading if the slot count is the whole cause:
- the module's load line reads `supervisor_slots=32`, and the slot gate passes;
- no "thread create refused" line appears;
- pt-probe prints `PT create rc=0`, `PT thread ran arg=7` and `PT join rc=0 ret=42`, and exits 0;
- lt-tls_local_exec, lt-pthread_tsd and lt-pthread_cond exit 0.

Any of the three tests failing past `pthread_create` is a second defect, and it would be reported as one.
The positive control for the refusal line is arm 1, where it fired four times.

Not run: the same tests on the physical or delegated SDK. This kit boots the virtual profile, and the physical
route refuses threads by design (the inventory above), so that arm cannot tell the hypotheses apart.
