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

## Arm 2: the same boot with 32 slots on both sides

One variable changed: `--cpu-prop x-capstone-supervisor-slots=32 --module-param supervisor_slots=32`. The images, kit
and run order were the same. The prediction was pushed before the boot, in lane commit f13293001250. It held on
every item:
- the module's load line reads `supervisor_slots=32`, and the slot gate printed `SLOT-GATE OK: module
  supervisor_slots=32 (set), platform 32` on the -cpu string the kit wrapper actually applied;
- the guest dmesg gained 250 lines across the run and none of them is a refusal;
- the same dmesg filter found the four refusal lines in arm 1, so it can fire;
- pt-probe printed `PT create rc=0 ... t=0x3f84322ea0`, `PT thread ran arg=7` and `PT join rc=0 ret=42`, and exited 0;
- lt-tls_local_exec, lt-pthread_tsd and lt-pthread_cond each exited 0.

The launcher stats show the thread services ran. pthread_cond made 7 `service3_thread_create` calls and 7
`service9_pthread_exit` calls; each of the other three images made one of each.

The kit's slot count is therefore the whole cause for these three tests. The kit is configured as vp2 (one slot),
where a second live context does not exist, so on this kit the tests measure the configuration, not the runtime.
A suite that runs pthread tests needs a kit booted with matched slots above 1 (Q-32).

## Arm 3: the other two failing tests at 32 slots; the prediction FAILED

`pthread_robust` (3e19fa94112e4d06) faulted at `pthread_timedjoin_np+0x188` with cause 28, which was not explained.
`cancel-points` (46fa1dd4d615351c) faulted at `pthread_cancel+0xe4`, address 0x44. Arm 2 did not include them. The
prediction was pushed before the boot, in lane commit f689f2d3e4db: both exit 0 with no refusal line. Both exited 1
on the same 32-slot boot (slot gate OK, `supervisor_slots=32`, no refusal line among 217 new dmesg lines). pt-probe
passed again as the control. Neither test faults any more. Both now fail on a check of their own, past
`pthread_create`, on two separate limitations of the virtual profile. Neither is the slot count.

- **pthread_robust: PI mutexes.** The test prints `pthread_robust.c:43: pthread_mutexattr_setprotocol(&mtx_a,
  PTHREAD_PRIO_INHERIT) failed: (pshared==0, pi==1) Function not implemented`, and the same for pshared==1. The
  non-PI cases print nothing. The chain, read from source and not traced:
  - musl 1.2.5's `pthread_mutexattr_setprotocol` probes `FUTEX_LOCK_PI` once and returns its error;
  - both of the virtual profile's futex filters admit only WAIT, WAKE, WAIT_BITSET and WAKE_BITSET, and return
    -ENOSYS for anything else (`runtime/virtual/pthread.c:175`, `runtime/virtual/exec.c:1218-1220`).

  The delegated runtime implements PI futexes (`ports/musl-capstone/runtime/delegate.c:776-860`); the virtual
  profile does not. The earlier cause-28 fault (the one-slot validation run) does not recur at 32 slots. That it
  followed from a refused create is inferred: that run kept no stdout.
- **cancel-points: thread signals.** Seven cases print `pthread_cancel.c:137: pthread_cancel(td) failed: No such
  process`. musl's `pthread_cancel` signals the thread with tkill. The shared delegate service
  (`runtime/linux/delegate-service.c:701-709`) hands tkill to `host->tkill` when the launcher sets one. When it
  does not, it returns -ESRCH for every tid except the process's own. The physical launcher sets it
  (`runtime/linux/exec.c:490, :805`, `context_tkill`). A `git grep -i tkill` over `runtime/virtual` and
  `runtime/host` finds nothing. Virtual thread ids are 0x400000 + id, so every cancel of a virtual thread reads
  ESRCH. The delegated runtime's tkill translation is B8 in `docs/plans/delegation-threads.md`.

Both are features the virtual launcher lacks at this revision, not defects of the tests or the compiler. They belong
to the runtime lane. Neither is new with delegate-pre-tls: the same images faulted the same way on base and tip.

## What this closes

- tls_local_exec, pthread_tsd and pthread_cond: the cause is the kit's one supervisor slot. They pass with matched 32
  slots.
- pthread_robust's PI cases and cancel-points: past the slot refusal, the virtual profile has no PI futexes and no
  tkill translation.
- The landed `delegate.c` (dev 7046cda1bdb6) is implicated in none of these.

Not run: the same tests on the physical or delegated SDK. This kit boots the virtual profile, and the physical
route refuses threads by design (the inventory above), so that arm cannot tell the hypotheses apart.
