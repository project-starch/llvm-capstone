# Delegated threads: one Linux thread per protected context

Status: PROBE A CASES PASS (A2 refuted then closed by the P0 sealed-return fix, 2026-09-30), PROBE B NATIVE PHASE PASSES,
PROBE B DOMAIN PHASE PASSES: a transport per context (T1, B7), parking through the launcher
(T2, B6 and B12), the runtime locks (T3, Q6, B9 and B14), musl's own threads on minted contexts
(T4, Q4, B10 and B11), signals per context with cancellation (B8), fifteen threads per application
(T5) and a thread's name and CPU set. libc-test's thread group passes seven of nine
(`pthread_cancel`'s asynchronous case waits for the doorbell, `sem_open` for file-backed mappings),
and GLib's `GCond` passes. CPython's earlier gate-pass claim was premature: subprocess creation
errors prevented child assertions from running. The review qualification below records the
subprocess-enabled results and explicit exclusions. Open: asynchronous delivery (the doorbell). Branch
`delegation-threads`, stacked on `delegation-signals` (c460e8c). The contracts below are what
Probe A and Probe B test.

Done so far:
- The domain half on the unchanged platform: context arena, mint, thread entry, exit and
  re-entry, revoke/remint, entered by a nested unsupervised CALL (record
  `results/20260929-context-probe-domain.json`).
- Through the monitor, on the context-slot branches of capstone-qemu
  (`qemu/context-slots-on-pin` ee9c93777f: DEAD status, dead-slot removal), capstone-sbi
  (0451a1a: slots with generations, lent descriptors, ADOPT,
  FORGET; 3f9efa9: the descriptor's offer slot is cleared after the seal is taken out) and
  caplifive-buildroot (11c024a: driver ids and ioctls; f9b2408: the firmware build refuses a
  capstone-c miscompile this work hit), and the launcher's CONTEXT requests here. Passing,
  with the signal contract and the application gate unchanged (record
  `results/20260929-context-probe-monitor.json`): A1, A3, A4, A5, A6, A8 (DEAD, FORGET
  once, STALE), A11, A13 without the foreign-owner request, A14, and A15 with its control on
  the base platform (there the kept result slot still writes).
- A12 (Q3 answered below): privilege changes and non-C seals are refused under supervision; this
  closed an escape into user mode that A12 found first.
- A2: at the first entry instruction only ra (sealed return), gp (exactly the main context's) and
  a1 (the 64-byte write-only descriptor loan) are tagged, and cscratch is exactly the start block.
  An independent review (2026-09-30) first refuted A2: a context could load the caller's saved
  state -- the monitor's saved pc, ctvec and cscratch -- through the sealed-return capability ra,
  because capstone-qemu checked no type or window on the data-access path. Fixed in capstone-qemu
  (P0): a sealed-return operand may reach only the general-purpose slots, the window
  [base + 3*CLEN, base + 64*CLEN); the first three capability slots fault. The ra-slot0/16/32 modes
  (fault at the ldc) and ra-gp (offset 48, no fault) are the controls, with the pre-fix binary as
  the positive control. A store past the loan faults; the loan's write-only permission is not
  enforced on capstone-qemu (Q1, ISSUES Q-14).
  A follow-up in capstone-qemu 674cdab03c closes a 64-bit overflow in that window check:
  `addr + size` could wrap to zero for an address near `UINT64_MAX` and pass both upper-bound
  comparisons. The same flaw in the general `cap_in_bounds` check is fixed too. Both checks now
  compare offsets and remaining bytes without adding to `addr`. The QEMU unit test covers both
  window edges, the capability end and wraparound in both checks; compiled against the pre-fix
  expressions it fails all three of its subtests, against the fix it passes them. On the new binary the
  context probe passed 36/36, the signal contract 26/26 and the application gate exited 0;
  see `results/20260930-sealed-return-wrap.json`.
- A10: dead registrations are retired on shortage in the monitor, and the driver drops a record
  when the monitor reissues its slot. 128 create/exit/revoke cycles never forgetting, and five
  applications claiming 40 of the 32 slots at once, all succeed with their live contexts intact;
  both fail on the previous monitor.
- A9: generations stop at 0x7fffffff, so every id is a positive long (the driver takes a negative
  SBI value for an error, and ids reach the domain as long); a slot that had the last generation
  is never assigned again, and a cached application block moves off it. A test firmware starts
  every slot near the end (CAPSTONE_TEST_GEN_PRESET): contexts and relaunches of one application
  run through the last generations and on to other slots. On the previous monitor (limit
  0xffffffff, no move) the first adoption at generation 0x80000000 fails with EIO, and the cached
  block can no longer be launched at all (ENOSPC).
- A13's foreign-owner request: an application naming another application's context or first
  context in STEP gets EPERM and in FORGET ESTALE (the driver's owner check), and the owner then
  still steps and forgets that context.
- A7: a paused context's area is revoked, the collector releases its seal's node, and a new seal
  is minted with that node and those bounds (capstone-qemu's test-only node instruments show
  both); the new context enters its own entry. With the collector's dead-slot removal disabled
  the same sequence resumes the dead context's continuation instead, which faults on its
  untagged stack: the removal is what prevents the ABA.
- P1 (review 2026-09-30): a THREAD-mode context's launcher thread blocked signals only after it
  started; a caught signal in the window between pthread_create and that block could race the
  single-producer signal trampoline. The launcher now blocks every signal around the create so the
  child starts blocked. The deterministic race probe is Probe B's B8 (signals during a stepping
  context), domain phase.
- Probe B, native phase: the launcher's parking queue (`runtime/linux/park.c`) passes B1 to B5,
  B12 and B13a to B13d with forced interleavings, each check shown to fire against a seeded
  defect (record `results/20260930-park-native.json`). B6 to B11 and B14 need the domain runtime:
  per-context transport and the WAIT/WAKE/REQUEUE requests.
- Probe B, domain phase, T1: a transport per context (Q2, first part; "A transport per context"
  below). A THREAD context's delegated calls are served by its own launcher thread through its own
  entry block and exchange region, so B7 holds: a context blocked in `read` on an empty pipe leaves
  the other running (it computed for 100 ms meanwhile) until that one writes. `thread-probe`
  passes 15 of 15 modes: transport, B7, reservation and its lifetime, 28 sequential contexts in one
  area, seven contexts at once each checking its own pipe traffic, a child preempted while the first
  context makes rounds, 100000 rounds from one context, `exit()` and a fault in a further context
  ending the process, a REGISTER context without transport, the signal requests a further context
  may not make yet, signal handlers only in the first context, SIGPIPE from a further context (by
  default the end of the process, EPIPE when ignored), and exec in place from a further context.
  Probe A (35/35 and the explicit A7, A9 and ctl-wfi), the signal contract
  (26/26) and the application gate, now with perl and mruby relinked on this runtime, pass
  unchanged (record `results/20260930-transport.json`).
- Found by T1, fixed in the monitor (capstone-sbi dd812db): every call's loan takes a revocation
  node, and nothing collected them unless supervised code ran short itself. One process's 65536th
  call (the VM's `CAPSTONE_REV_NODES`) halted the monitor at `loan_begin` with cause 30, with or
  without threads; no earlier run had made that many calls. The monitor now collects before a loan
  when fewer than 1024 nodes are free. 100000 rounds took 9126 ms and 9069 ms in two runs, every
  10000-round block between 904 and 935 ms.
- Found by T1: the launcher's region capabilities can cover more than was asked for (the driver
  hands out a larger cached block), and the runtime had sized the exchange region by the
  capability's bounds. After an application with seven transports, the signal contract's
  `retry-partial` sent a write of 0x7fff0 bytes against a 0x40000-byte exchange region and got
  EFAULT. The runtime now cuts
  its transports by the sizes the image declared.
- Probe B, domain phase, T2: parking through the launcher (Q5 answered below). futex WAIT, WAKE
  and REQUEUE from any context are served by the launcher's park queue; the lock word stays in the
  domain. `thread-probe` adds ten modes (25 of 25 pass): the value check, a timeout that keeps its
  budget, WAKE selecting a parked waiter in another context, REQUEUE as musl's condition variables
  issue it, B6 and B12 with a waiter held across four quanta in the window between its compare and
  its WAIT while the other context releases and WAKEs or REQUEUEs with nobody queued (the WAIT
  answers RECHECK at once), B6's control with the generation read after that window instead (the
  wake is lost: its 500 ms deadline expires), two contexts counting 800 increments under a futex
  mutex with 796 parked waits, and a signal while the first context is parked, answered as Linux
  answers the same futex call natively (untimed under SA_RESTART: handler at once, the wait goes on
  to the wake; untimed without SA_RESTART and timed: EINTR). The promptness check fired against a
  launcher whose park sleep bypassed the signal stub (handler only at the wake, 533 ms). Record
  `results/20260930-park-domain.json`.
- Probe B, domain phase, T3: the runtime locks (Q6 answered below) and each context's thread
  identity (Q2, second part). `thread-probe` adds nine modes (34 of 34 pass): distinct identities
  (the pid, then 2^22 upwards), 12000 lines from eight contexts into one FILE with none malformed or
  out of order, stdio calls that retake a FILE lock (`puts`, `perror`, `fclose`, `putc` inside
  `flockfile`) from minted contexts, eight contexts on the level0 heap and on the Sublet heap (1500
  rounds each, no corrupted block), B9 (a linearizable exchange chain of 40000 distinct
  capabilities, a CAS cursor advanced exactly 40000 cells, a scalar counter exact, the atomics lock
  found taken 16 times), B14 (a signal arriving while the first context holds lock A and waits for
  lock B runs its handler once, at lock depth 0, by the release of A), 24 concurrent spawns each
  with its own exit status, 1600 concurrent anonymous mappings, and three contexts at once each
  creating a context of its own. Each lock check fired against a runtime seeded with the defect it
  is about: the depth rule off (the handler ran at depth 1), the heap lock off (a fault), the
  atomics lock off (duplicated exchanges, the cursor 34779 of 40000), every minted context with
  thread identity 0 (21873 lines for 12000, 21130 malformed), identities from 0x40000000 (the
  nested stdio calls waited for themselves). Record `results/20260930-runtime-locks.json`.
- An independent review of T3 (2026-09-30) found the identity range defect above: the first
  version counted from 0x40000000, which is musl's `MAYBE_WAITERS` bit, so a minted context
  waited for itself in every nested FILE lock and recursive mutexes broke; the probes then passed
  because none retook a lock. It also found builds outside `capstone_configure_application` that
  compile runtime files one by one (libc-test, the stdio/file/write probes, the capability-atomics
  QEMU test, the native mapping test) without the new `lock.c`; a further context reaching the
  first context's signal state through `longjmp`; and a lock depth that could go negative. All are
  fixed; stdio-nested, the Sublet image, spawn-concurrent, mmap-concurrent and nested-create were
  added for the gaps it named.
- Probe B, domain phase, T4: musl's threads on minted contexts (Q4 answered below). musl's
  `pthread_create`, `join`, `detach` and `exit` run unchanged; the runtime supplies `__clone`, a
  thread's end and its reaping (`context.c`). `__clone` mints a context whose area is only a seal
  region and a start block (1280 bytes of the context arena), entered on the stack and thread
  pointer musl built in its own mapping, and runs it on a launcher thread. A thread ends with
  `SYS_exit`: CONTEXT_EXITING names the word to wake, then the context stores 0 into its
  CLONE_CHILD_CLEARTID word (musl's thread-list lock) and 1 into its completion word, in that
  order and after its last use of the stack and TLS, and returns EXITED; the launcher frees the
  transport and then wakes the word once, as Linux does. `pthread-probe` passes 12 of 12: 60
  rounds of seven threads held together by a barrier, each joined with its value and its own
  identity, in seven records (areas reused); B10, a joiner parked on the thread-list lock while
  the child holds 40 ms between its announcement and its clear, released by the launcher's wake
  (41 to 46 ms); B11, 300 detached threads, at most six at once, in one record, their mappings
  back on the heap; a reservation that waits for an ending context's transport (a thread made 47
  ms later, while an eighth at once is refused EAGAIN); `pthread_exit` in the main thread while
  another runs (the application ends 0 with the other's output); `exit` in a thread (status 5);
  thread-specific data with destructors; a condition variable over 2000 turns; `clone` for a child
  process refused; priority-inheritance mutexes (a timed lock that expires, one whose deadline is
  the end of time and waits); a thread on a stack the application gave it, told its own stack by
  `pthread_getattr_np`; three threads made after the main thread left. Each check fired against a
  seeded defect: no
  clear (the join never returns), no reaping (EAGAIN at round 7, and at detached thread 27), the
  main thread's exit as `exit_group` (status 0 without the other thread's line), a launcher without
  the wake (the join never returns), a launcher whose reservation does not wait (EAGAIN after 14
  ms), the PI deadline computed without saturation (the far deadline expired after 1 ms), musl
  without patch 0005 (the user stack faults, cause 24). A first version of the reservation mode could not fire: a musl thread holds the thread-list
  lock until its clear, so the creator waited in `pthread_create` for that lock, not in the
  reservation; the mode now ends a context of the runtime's own interface, which holds no lock.
  Record `results/20260930-pthreads.json`.
- Found by T4 and fixed (the first four shown failing before, by the probe or libc-test):
  `libc.page_size` was never set (a domain has no auxv), so `PAGE_SIZE`, `sysconf(_SC_PAGESIZE)`
  and every `pthread_create`'s mapping size were 0; `pthread_cond_t` put `_c_head` over `_c_clock`
  and `_c_tail` 32 bytes past the object at 16-byte pointers (C-65), the defect e2c9ad3 worked around in
  GLib (musl patch 0004 lays it out pointers first; a signal faulted in the probe before);
  `get_robust_list` answered ENOSYS, so `pthread_mutexattr_setrobust` refused and an orphaned
  robust mutex blocked for good; PI futexes answered ENOSYS, so `setprotocol` refused and a PI lock
  slept for good. Added before any run: `mprotect`, which answered ENOSYS (reported unserved) for
  musl's thread stacks, now answers 0 for read-write pages inside one mapping, which states what is
  true; guard pages are not enforced, the mapping's capability bounds are. The TLS blocks are now musl's own layout (`__copy_tls`, tp page-aligned), one
  layout for every context. The application SDK declares seven contexts and a 64 KiB context
  arena by default; it declared none, so no SDK application could make a thread. The start cost of
  seven contexts is not measurable: 20 starts of the same program took 10.63 s with them and 10.60
  to 10.63 s without (after a first round of 11.31 s).
- An independent review of T4 (2026-09-30) found two defects. The main thread's `SYS_exit` took a
  musl lock after musl's `pthread_exit` had set `need_locks` to -1, and a preemption between that
  lock's load and store could switch musl's locking off for good while two threads ran; the path
  now takes no lock (an atomic count of the threads `__clone` made replaces the scan of the
  records). `pthread_attr_t` kept the stack address as a long, so `pthread_attr_setstack` built an
  untagged stack and `pthread_getattr_np` returned one; musl patch 0005 keeps it as a pointer. It
  also found `SYS_exit` served only after the transport check, the PI deadline arithmetic
  overflowing, and stale comments; all are fixed. Its two further risks are B8's: after the main
  thread's `pthread_exit` no signal handler runs any more, and a minted thread cannot be signalled
  (`raise`, `abort`, `pthread_kill`), and `sched_*` given a thread's identity name no Linux thread.
  The race itself has no probe; `main-exit-more` makes threads and uses the heap after the main
  thread left.
- Open after T4: the first context's `pthread_getattr_np` derives the stack from `libc.auxv`, which
  a domain lacks; level0 has no `aligned_alloc` or `posix_memalign` (musl's allocator object then
  collides with it at link time); `sem_open` needs file mappings.
- libc-test on this runtime (delegated runner, `results/20260930-pthreads.json`): 56 PASS, 3 FAIL,
  1 FAULT, 5 NOBUILD, 12 EXCLUDED of 77, against 46, 4, 4, 3 and 20 on 2026-09-29, none worse.
  `pthread_cond`, `pthread_mutex`, `pthread_mutex_pi`, `pthread_robust`, `pthread_tsd`, `sem_init`,
  `tls_init` and `tls_local_exec` pass; `popen` and `setjmp` pass too, which this run does not
  attribute. `pthread_cancel` and `pthread_cancel-points` do not build (musl's cancellation points,
  `__syscall_cp_asm`), and cancelling a thread needs its signal (B8). `sem_open` stays excluded, for
  a reason that is not threads: it maps a /dev/shm file MAP_SHARED, and a domain maps no files.
- Probe B, domain phase, B8: signals per context and cancellation ("Signals per context" below).
  `pthread-probe` adds 14 modes (26 of 26 pass): `pthread_kill` into a thread blocked in `read`
  (its handler runs there, the read answers EINTR); a signal sent to the process runs in the one
  thread that does not block it; `raise` in a thread; `sigwait` in a thread; `pthread_cancel` of a
  thread blocked in `sem_wait` and in `read` (cleanup run, `PTHREAD_CANCELED`), with cancellation
  disabled until the thread enables it, and with the cancellation handler installed before the
  first thread existed; `abort` in a thread (SIGABRT); a signal after the main thread's
  `pthread_exit`, which now runs in the thread that is left; a signal to a thread parked in a futex
  wait whose handler makes a delegated call (the wait goes on to its wake); a sigreturn mask left
  by a handler during `sigsuspend`; `setuid` answering at once while another thread computes; a
  handler switched 600 times while its signal is taken. `thread-probe`'s two signal modes now
  check the new contract (a further context's own handler and mask; a signal sent to the process
  runs in the context that does not block it, 20 of 20). Each check fired against a seeded defect:
  a launcher without the `tkill` mapping (four modes fail), one whose context thread keeps its
  creation mask (the context that never sets one takes nothing, 0 of 20), one that does not apply
  a further context's `rt_sigprocmask` (routing and the main-exit signal fail), one without the
  glibc settle below (SIGSEGV in glibc's setxid handler), a runtime without the cancellation
  accounting (both cancel modes hang), one with the earlier sigreturn mask, one with musl's own
  `__synccall` (hangs). Two first controls did not fire, each for a reason named in the record:
  musl sets every new thread's mask itself, and the first `setuid` mode's computing loop read a
  clock, which enters the dispatcher. The probes found two runtime defects before any review: a
  cancellation delivered at the SIGPOLL round a cancellation point makes first was shown
  `__cp_end` and hung the thread, and a RETRY round that was not issued again made a cancelled
  `read` return 0 (end of file). libc-test: 57 PASS, 3 FAIL, 1 FAULT, 3 NOBUILD, 13 EXCLUDED of 77;
  `pthread_cancel-points` passes, `pthread_cancel` is excluded (its first case cancels a thread in
  `for (;;);`, the doorbell's). Of the libc-test regression tests about signals and threads, 11 of
  16 pass: `sigaltstack`, `sigreturn`, `sigprocmask-internal`, the cancellation and robust-mutex
  regressions among them; `raise-race` and `pthread_exit-dtor` fork, `pthread_atfork-errno-clobber`
  forks, `flockfile-list` defines its own `malloc` (the heap does not yield to one), and
  `pthread_cond-smasher` runs ten threads at once, more than seven (T5). A round costs about 1 to
  3 percent more: 100000 rounds took 9304 to 9463 ms in four runs, against 9189 and 9241 ms at T4.
  Record `results/20260930-signals-per-context.json`.
- T5, more contexts per application: each application's threads each hold one of its invocation
  descriptors, so 8 descriptors allowed seven threads, while musl's condition-variable regression
  and CPython's `test_various_ops` start ten at once. The monitor (capstone-sbi 4674ab6) carves
  16 descriptors per application off the top of its data region (1024 bytes, the driver's copy of
  `process-abi.h` matching) and keeps them in four pool parts of 128 capabilities, since
  capstone-c limits a global to 2048 bytes; the runtime allows `CONTEXTS` up to 15 and the SDK
  declares 15. The 32 slots stay the board's, shared by all applications (monitor unification),
  so QEMU's supervisor table is unchanged. `pthread-probe` now runs at 15: 60 rounds of fifteen
  threads joined (900 threads, fifteen records), a sixteenth refused with EAGAIN, the reservation
  waiting 48 ms for an ending transport; all 26 modes pass. On the previous firmware and driver
  the same image fails at its eighth thread at once (EAGAIN). `pthread_cond-smasher` passes (the
  regression selection 12 of 16); libc-test, the context probe with A7, A9 and ctl-wfi, the signal
  contract and the application gate as before. 100000 rounds took 9395 ms. Record
  `results/20260930-contexts-per-application.json`.
- A thread's name and CPU set: `pthread_setname_np` and `pthread_getname_np` on the calling thread
  (musl's `prctl(PR_SET_NAME/PR_GET_NAME)`) are the runtime request `THREAD_NAME`, and
  `sched_getaffinity` and `sched_setaffinity` of the calling thread (pid 0 or its own tid) are
  delegated. The launcher thread that serves a context runs them on itself, so a name shows in
  `/proc/<pid>/task/*/comm` and the CPU set is that thread's. Naming or asking about another thread is refused: a minted
  thread's tid, from `0x400000`, names no Linux task (ENOENT), and the launcher answers ESRCH for
  any pid but 0. musl patch 0006 keeps the name a pointer: `prctl` read its arguments as
  `unsigned long` and the thread-name calls passed `(unsigned long)name`, so the first request
  faulted (cause 24); GLib names every thread it creates, so its thread tests faulted too.
  `pthread-probe` adds `thread-name` and `affinity` (28 of 28 pass); on a launcher without the
  request and the call, both fail. Found by GLib's `thread6` (a thread's name read back empty)
  and `thread7` (it sets a thread's CPU set), and by CPython's `ThreadPoolExecutor()`, whose
  default worker count asks for the CPU set.
- Signal/read probe review (2026-09-30): `kill-thread` no longer guesses that the
  worker has entered `read` after sleeping 40 ms. It names the worker, observes that Linux
  thread blocked in the empty-pipe `read` through `/proc/self/task`, then signals it.
  `kill-thread-early` forces one delivery before the read, then verifies a second delivery
  interrupts the observed read. Error cleanup releases the worker and writes a byte before
  joining. Native ASan/UBSan CTest passes 44/44, including the deliberately missing second
  signal returning failure in 3.01 s; the guest pthread probe passes 29/29. Record
  `runtime/tests/application/results/20260930-kill-thread-review.json`. This removes a known
  test race; the earlier intermittent timeout's exact schedule was not recorded, so its
  individual cause remains unproven. This is synchronous delivery, not the doorbell.
- Initial CPython evidence, before subprocess patch 0015: CPython 3.13.7 with C11 thread-locals
  (patch 0006 and its define are gone) and patch 0016, which keeps the join handle and the raw
  mutex's waiter link as pointers (without it the first `Thread.join` faults in `pthread_join`,
  and a contended `_PyRawMutex` at
  `waiter->next`). In the guest `test_threading` runs all 212 tests: 28 errors, all fork (26
  through `subprocess`, 1 `os.fork`) or sixteen threads at once, 13 skipped; `test_thread`
  (without `test_forkinthread`, which blocks for good when fork is refused), `test_threading_local`
  and `test_queue` pass; the thread pool fails only on a process pool and a fork. Found on the
  way: C-75, a compiler defect (`compiler/cap-init-alias`): a pointer slot whose initializer
  names an alias was never tagged, so musl's `fork()` faulted on its table of lock pointers once a
  second thread existed; libc-test's `raise-race` had met it since T4. Record
  `ports/cpython/interpreter/results/threads-2026-09-30.json`.
- CPython review qualification with patch 0015: subprocess now goes through the launcher's
  `posix_spawn`, while fork remains ENOSYS. `host/run-thread-gate.py` checks refusal beside
  four live threads, then enables upstream's explicit no-fork skips before loading tests.
  Configure disables unsupported epoll so `selectors` does not emit an UNSERVED diagnostic
  on child stderr. The rerun exposed level0 retaining the full 32 KiB allocation after
  CPython shrank a short-read buffer: capturing a recursive child's output exhausted the
  parent at 128 MiB and 192 MiB. Level0 now releases and coalesces the unused tail. The native
  control fails at allocation 31 before the fix. Afterwards, native and domain controls retain
  4,096 buffers, preserve pointers through regrowth and recover a 900 KiB free block.
  With the fix, all five CPython suites pass at 64 MiB: 448 tests, 38 reported skips, no
  errors or failures. Subprocess smoke passes 21/21; of the 26 earlier creation errors,
  18 now pass, seven skip actual fork and one skips the root-only rlimit check. Native
  ASan/UBSan CTest passes 45/45 and the rebuilt pthread probe passes 29/29. Record
  `ports/cpython/interpreter/results/thread-review-2026-09-30.json` preserves the failed
  runs and every explicit exclusion. This qualifies the supported threading profile, not
  asynchronous delivery, fork or the omitted optional/resource-dependent tests.
- An independent review of B8 (2026-09-30) found five defects, all fixed: a delivery read the
  shared handler table unlocked while another context could change it (the action is now copied
  under a leaf lock); glibc's `sigaddset` refuses signals 32 and 33, so the trampoline could not
  block musl's SIGCANCEL (bits are set directly); the first design kept a signal that reached a
  launcher thread before it attached by requeueing it, which the seccomp filter and the kernel
  refuse (such a signal now waits in a small per-thread list that attach moves into the ring);
  the creator read the new thread's record after the thread could have freed it (the handshake
  now uses the creator's own frame); and a cancelled `close` returned 0 without cancelling (a
  cancelled cancellation point now calls musl's `__cancel` itself, as `__cp_cancel` does). Fixed
  from its risks: glibc installs its own handler on 33 at its first `pthread_create` (the launcher
  now has that happen at start and restores the process's dispositions), the trampoline read the
  shared flags twice, a round status could outlive an abandoned cancellation point, a
  `sigsuspend` handler was shown the wait's mask, and `__synccall` waited for every thread to make
  a call (it now runs its function once in the caller: set*id is not served, and one hart orders
  memory for every context). Named, not fixed: a SIGACTION changes the kernel's disposition just
  before the generation the trampoline stamps, and the domain keeps one previous installation, so
  in that window an event can run the old handler or be dropped; and a signal sent to the process
  that one context has taken stays with that context, where Linux could hand it to another thread
  that `sigwait`s. The spawn helper and exec in place now carry signals 32 and 33 in the masks
  they hand on, and exec in place keeps every signal blocked until the new launcher sets the
  caller's mask, so a pending signal survives it.
- An independent review of T1 (2026-09-30) found five defects. Fixed, each with a probe mode that
  failed before and passes after: a further context ran one of the first context's pending signal
  events, which then stayed blocked (now only the context with the handover block touches signal
  state); exec in place from a further context started the new image with every signal blocked
  (the thread now takes the application's mask just before `execve`); SIGPIPE from a further
  context's write stayed pending on its blocked thread (now forwarded to the process). Fixed by
  review: exit, fault and exec in place take one lock, so an end that has begun wins. Left for Q6:
  posix_spawn and execve pack into one static block in the domain, so two contexts spawning at once
  mix requests, like malloc and stdio, which further contexts still run without locks.
- A13, changed with T1: a THREAD context has one stepper, its own launcher thread, which serves its
  transport; a STEP the application asks for is refused (EINVAL). Before, the application's steps
  and the thread's interleaved; with a transport per context, a step from another thread would hand
  one of the context's requests to the wrong thread. `two-steppers` now checks the refusal. The
  driver's serialisation of concurrent STEPs is still there, but no launcher path exercises it.
- Pins: capstone-qemu 674cdab03c (`qemu/context-slots-on-pin`, on ac2837aa0e, the head of
  `qemu/supervisor-switch-cost`); caplifive-buildroot 8fd1ea1 (`modcapstone/context-slots`, the
  driver), whose components/opensbi is cf344cf (caplifive-opensbi `wrapper/context-slots`) at
  capstone-sbi 4674ab6 (`monitor/context-slots`). The wrapper and monitor commits are on the
  forks only (the `runtime-fork` remotes). `qemu/context-slots` (d220ab6ee9, on 22aec7ee0f with
  its own copies of the two switch-cost commits) is the frozen predecessor of
  `qemu/context-slots-on-pin`: its commits are patch-identical to ee9c93777f..a53ac18e3d, it lacks
  674cdab03c, and nothing pins or extends it. Upstream pull requests: none until the thread
  runtime and its gates pass (decided 2026-09-30).

## Scope

- **Platform:** capstone-qemu with the supervisor extension (`CSUPERVISE`, protected
  continuations, the 5 ms quantum). The extension is "a VM platform extension, not an
  implementation claim about the existing FPGA instruction set" (`capstone_supervisor.c`, header).
  Nothing in this plan is a claim about the RTL.
- **One hart.** `capstone-vm` boots `-smp 1` (`runtime/host/capstone_vm/cli.py`), and the
  supervisor's collector refuses to run with a second CPU (`helper_cssupervisor_gc`). The probes
  assert one online CPU at start. SMP is a later goal with its own contract: migration of
  continuations, capability moves and revocation across harts, the memory model. Nothing below
  precludes it.
- **One Linux thread per context.** Linux schedules, blocks and routes signals. The monitor does
  not know threads.
- **Out of scope:** SMP; the ISA's own preemption path (an interrupt enters a handler domain with an
  asynchronously sealed context). That path fits this design: a handler domain can save the context,
  hand back an opaque continuation and leave the choice to Linux, so it needs platform work but no
  scheduler in the monitor. Also out of scope: the bell (asynchronous entry into a running context,
  a separate operation from adoption); `FUTEX_CMP_REQUEUE`, `FUTEX_WAKE_OP`, PI and robust futexes
  (they answer ENOSYS visibly); asynchronous cancellation; `fork` in a threaded process (already
  ENOSYS); sockets. Plain `FUTEX_REQUEUE` is in scope: musl's condition variables issue it
  (`unlock_requeue` in `pthread_cond_timedwait.c`, zero wakes and one requeue).

## Responsibility

| Layer | Knows | Does not know |
|---|---|---|
| Linux | when each launcher thread runs; blocking; signal routing; its tids | domain memory; any capability |
| Launcher | context id of each of its threads; per-context transport; the park queue | domain memory beyond the regions it is lent |
| Driver and monitor | context slots, their owner, application memory | threads, tids, scheduling, pthread |
| QEMU supervisor | one protected continuation per slot | what a context is used for |
| Domain runtime | the contexts it minted, their areas, pthread state, lock words | Linux's scheduling decisions |

What the design claims: Linux cannot create capability authority, cannot inject a context, and
cannot resume a context whose seal the domain has revoked. Linux decides when and whether a context
runs, and its answers may be wrong. The runtime treats every external answer so that it never breaks
a protected ownership or lifetime rule: memory is reused only after the revoke of a handle the
domain holds, never on the strength of a Linux answer.

What the design does not claim:
- **Isolation between threads.** Contexts of one application trust each other. A bounded stack or
  TLS capability stops an access outside its bounds through that capability. It does not isolate a
  context from a sibling that was given overlapping authority.
- **Rust's `Send`.** Moving a linear capability between contexts transfers exclusive ownership of
  that memory. It proves no type invariant, no thread affinity and no correct synchronisation, and
  a non-linear capability is not Rust's `&T` or `Arc<T>`.

## Terms and identities

- **Application:** the domain image and the memory the driver allocated for it; one per
  `capstone-exec` process. Owns the heap and the globals.
- **Context:** one executable state with its own continuation, stack and TLS block. The main
  context is the one `create_domain` builds; every further context is minted by the runtime.
- **Thread area:** one block of application memory per minted context, taken with a retained
  parent handle. Its children are the seal region, the stack and the TLS block (which holds musl's
  `struct pthread`). One revoke of the parent handle ends all of them.
- **Seal:** the linear sealed capability over the seal region.
- **Context id:** slot index and generation, `(k, g)`. Every driver and monitor operation that
  names a context carries both.
- **Linux thread:** the launcher thread that steps one context and executes its delegated calls.

## What the code does today

These facts are what the contracts below are written against.

- `helper_csseal` requires a linear, RW, aligned capability of at least 528 bytes
  (`CAP_SEALED_SIZE_MIN`). A synchronous seal is not a register image: CALL and RETURN exchange only
  pc, `ctvec`, `cscratch` and the control words (`swap_c_effective_regs`). Layout: pc at 0, `ctvec`
  at 16, `cscratch` at 32, `mstatus | priv << 38` at 48, then `mideleg`, `medeleg`, `mip`, `mie` at
  56, 64, 72, 80. `create_domain` writes `(3 << 38) | (2 << 34)` at 48 and zero after it.
- Every CALL delivers a fresh sealed-return capability in `ra`, and RETURN consumes it
  (`helper_cscall`, `helper_csreturn`). The runtime keeps it in the global `__capstone_dom_ret`, or
  with fault recovery in the block behind `cscratch` (`start-musl.S`). `__capstone_yield` builds a
  256-byte frame on the stack before it saves anything. The first entry (`test`) runs the
  capability-global initializers and then `domain_main`.
- The supervisor keeps at most 32 continuations per hart (`SUPERVISOR_SLOTS`) and finds a slot by
  `same_authority`, which compares the revocation node id and the bounds only. It writes the
  caller's continuation into the seal region on every resume (`capstone_supervisor_call`). A
  supervised switch drops the LR reservation (`restore_state`: `load_res = -1`).
- On resume the supervisor restores the callee's snapshot, including the capabilities the callee
  received as arguments at the first entry of that call; the arguments of the resuming CALL are not
  delivered (`capstone_supervisor_call`). The monitor lends the result slot per call, derived with
  `csshrinkto` and `cstighten`, which create no revocation node of their own, and does not revoke
  it after the call (`supervised_call`).
- Domain code runs at `PRV_C`, which is `PRV_M` on capstone-qemu (`cpu_bits.h`). While a supervised
  call is active, `riscv_csrrw_check` refuses every CSR above user level (`csr.c`). The `mcause` and
  `mtval` writes in `start-musl.S` belong to the branch without `CAPSTONE_DOMAIN_FAULT_RECOVERY`;
  applications build with it (`Application.cmake`).
- `CSUPERVISE` and the supervised CALL raise an illegal-instruction exception on an invalid seal,
  also in the forget form with a null buffer (`helper_cssupervise`, `capstone_supervisor_call`). The
  monitor issues both without a check (`supervised_invoke`). Its QEMU exception handler treats
  every trap other than an access fault as a domain fault (`handle_exception` →
  `fault_return_from_domain`), which is not built for a trap raised in the monitor itself. What
  happens then is untested.
- The collector clears stale tags in memory, in registers and in paused continuations, and pins
  the pc node of each paused continuation. Before it releases invalid nodes, it does not look at a
  slot's own seal identity (`helper_cssupervisor_gc`). On capstone-qemu a capability loaded after
  its node was revoked comes back untagged (ISSUES Q-11).
- The monitor holds one seal per slot (`domains[]`, 32) and keeps memory per slot:
  `managed_destroy_domain` first forgets the continuation, then reclaims (`sbi_capstone.c`). The
  driver binds owner, id and memory in one `process_block` (32 domains, `process.c`) and serialises
  its monitor calls on one mutex (`capstone_lock`).
- Single-thread assumptions in the domain runtime:
  - the delegation state (`dl_entry`, `dl_exchange`, `dl_capacity`, `dl_used`, `dl_status`) is
    static (`delegate.c`);
  - `tls.c` has one static `struct pthread` and answers `set_tid_address` with 1;
  - capability-width atomics are unsynchronised library calls that the file itself calls "WRONG
    for any build that can run two threads" (`atomic_libcalls.c`, C-54);
  - `level0.c` and `sublet_heap.c` take no lock.

## Context lifecycle

The normal join path is:

`MINTED → OFFERED → ADOPTED → (RUNNING ⇄ PAUSED)* → EXITED → REVOKED → REMOVED`

Revocation may also terminate an offered, adopted or paused context, for example during creation
rollback or a negative probe; it does not require `EXITED`. `EXITED` is application completion,
whereas `DEAD` means that the executable authority is invalid. `FORGET` removes a registration and
its continuation; it neither proves application completion nor returns the private thread area to
the allocator. Registration removal and area reclamation are separate operations.

1. **Mint (domain).** The runtime:
   - takes a thread area from application memory with `sublet_take_linear` and keeps the parent
     handle in application-owned bookkeeping outside that area;
   - splits the area into seal region, stack and TLS, and writes the TLS image and `struct pthread`;
   - reserves a start/recovery block inside the thread area, outside the seal region and the
     usable stack, containing stack, tp, gp, start function, argument and completion reference;
   - writes the seal region: pc = the thread entry label, `ctvec` = the runtime's trap vector,
     `cscratch` = the start block, control words exactly as `create_domain` writes them;
   - seals it.
2. **Hand-off (domain to monitor).** On every call the monitor lends each context its invocation
   descriptor (Q1), which replaces today's bare result slot (`supervised_call`) and holds the
   result word and a 16-byte capability slot. The runtime moves the seal into that slot and makes
   the delegated request `CONTEXT_CREATE`, whose argument is only a ticket. The seal never passes
   through memory that Linux can write, and the launcher never sees it.
3. **Adopt (launcher, driver, monitor).** `ADOPT(parent (k, g), ticket)` moves the seal out of the
   parent's hand-off slot into a free context slot and returns `(k', g')`.
   - The new slot's owner is the parent's owner. Only code running in a context of the same
     application can write the parent's hand-off slot.
   - Only one offer may be outstanding in a parent's hand-off slot. The ticket identifies that
     offer; it conveys no capability authority. Repeated or delayed requests must not consume a
     later offer from the same parent. The exact register/descriptor layout and ticket validation
     are question Q1.
   - The launcher then creates the Linux thread. If that fails, it calls `FORGET(k', g')` and
     answers the request with an error; the runtime revokes the area. Failure before adoption
     must also clear or retire the offered seal before the mailbox is reused. Neither failure
     path blindly replays a consumed offer.
   - Stack, TLS, transport and protected runtime bookkeeping must be ready before the child's
     first STEP. A Linux thread may start before the creator receives its reply. Any required
     startup acknowledgement must account for that ordering (Q2).
4. **Run.** The new Linux thread loops `STEP(k', g')`. The first entry lands on the thread entry
   label, which:
   - loads sp, tp, gp and the start function from the block behind `cscratch`;
   - keeps this context's sealed-return capability and recovery block there;
   - never runs the capability-global initializers;
   - calls the start function.

   Preemption and resume belong to the supervisor.
5. **Exit:** see "Exit and join".
6. **Revoke (domain):** the joiner or the reaper revokes the parent handle.
7. **Remove (supervisor and monitor):** by the rules below.

### Supervisor and monitor rules

These are generic context-management rules. None of them mentions threads.

- **One registration, one entry at a time.** ADOPT moves the seal out of the hand-off slot, so a
  seal is registered at most once. The supervisor has one `active` call per hart, and the driver
  serialises its monitor calls, so a slot is never entered while it is already running. Both
  properties use the ISA's move semantics, the supervisor and driver serialization; case A13
  verifies that the implementation preserves them, including the empty mailbox after adoption.
- **A slot whose seal is invalid is dead.** Arm and resume answer the status `DEAD` instead of
  raising, and the monitor returns it to the driver as a STEP result. This is an expected lifetime
  outcome, not a domain fault. Wrong-owner or malformed control operations retain their separate
  validation; accepting a revoked handle does not bypass those checks.
- **Dead slots are removed before their node can be reused.** The collector removes every slot
  whose seal node is invalid, together with its saved continuation and any `armed` reference, before
  it releases nodes. It also runs when a slot is requested and none is free, so dead slots cannot
  exhaust the 32 even if Linux never forgets them. The supervisor continuation table, monitor
  `domains[]` registrations and driver context records are distinct tables: collecting a
  supervisor slot alone does not free the other two. Monitor/driver cleanup must retire their
  matching dead registrations, without reclaiming application memory or prematurely freeing
  transport still used by a launcher thread. Slot-pressure tests cover all three tables.
  As built: the monitor retires a minted registration whose copy of the seal no longer reads as
  sealed (a revoked seal reloads untagged), or whose context has ended with a valid seal (a STEP
  faulted, and the supervisor never arms that continuation again, or was refused), when an
  adoption finds no descriptor for its application, or no slot, and when a new application block
  finds no slot. A slot with an outstanding offer is kept until the offer is adopted: retiring it
  would discard the offered seal, and an adoption would then register nothing. That case needs a
  minted parent, which today's launcher never adopts from, so it rests on review, not on a probe.
  A new block takes a free slot, not the next index. The driver drops every record of a slot the monitor has just
  given a new generation, so its table never holds more records than the monitor has slots.
- **FORGET is idempotent and cannot be redirected.**
  - A tagged seal that is invalid still names exactly its old slot, because its node is not reused
    before the collector has untagged every copy. FORGET removes that slot.
  - An untagged value names nothing, and FORGET returns without effect.
  - No slot is ever found from an untagged scalar representation. This rule relies on the complete
    stale-tag sweep and dead-supervisor-slot removal preceding node reuse.
- **Integer ids carry a generation.** `STEP`, `FORGET` and `ADOPT` take `(k, g)`; a stale `g` is
  refused with `STALE`. Generations never silently wrap: retire an exhausted slot instead of
  reissuing an old identity. `STEP` on a still-registered revoked seal returns `DEAD`; after
  retirement the old id can return `STALE`. Repeated FORGET of a retired id has no effect and may
  report `STALE`; it must not act on a replacement. This protects the launcher from its own late
  requests. Within one owner this is robustness, since Linux may end its own contexts at will.
  Cross-owner validation remains required. See Q1 for encoding and status mapping.
- **Application memory is not context memory.** Contexts have no memory of their own in the
  monitor's or the driver's books. Application memory is reclaimed only when the application is
  destroyed, after every context slot of it has been removed. Destroying one context never reclaims
  memory.
- **Control words (open point).** A minted seal carries its own `mstatus` and privilege,
  `mideleg`, `medeleg`, `mip` and `mie`, and the monitor cannot read a sealed region to check them.
  The existence of an unsupervised seal/call path does not establish that arbitrary control words
  are safe under supervision. The contract distinguishes the first entry, which loads the seal's
  control words, from a resume, which restores the supervisor's snapshot while the seal region
  holds caller state; a template check at arm time covers only the first. A12 must verify continued
  capability confinement, bounded return to the owner and intact caller state, or a clean refusal,
  for the tested alternatives. Merely recording a changed privilege or a crash is not a passing
  oracle. The supported control-word contract and its enforcement point are Q3, to resolve before
  accepting Probe A.

The process ABI changes in place (`process-abi.h`: driver and monitor are pinned together, no
feature probe).

## Exit and join

Three events are distinct:

1. the thread function has finished (cleanup handlers, TLS destructors, return value stored);
2. the context can never be resumed;
3. the Linux thread has ended.

Only (2) licenses reuse of the thread area, and the domain establishes it itself, by revoking.

**Completion record.** It lives in musl's `struct pthread` inside the TLS block and holds the
return value, the detach state and `done`. The joiner copies what it needs before it revokes.
Join and detached reaping have exactly one claimant, chosen in protected application state; neither
may traverse a freed pthread block. A detach/reaper list entry needed after revocation lives outside
the thread area and must be installed before `done` is published.

**Exit path.** musl's `pthread_exit` runs cleanup and destructors, stores the result, then calls
`__capstone_context_exit`. That routine is non-returning assembly and:

1. loads into registers everything the final switch needs: the current sealed-return capability
   and the result-slot capability;
2. stores `done = 1` with release semantics;
3. touches no stack, TLS or `struct pthread` after that store, calls no libc function and never
   enters the syscall dispatcher, so no signal is delivered in the domain;
4. writes `EXITED` into the result slot and returns, with the reentry pc set to its own re-entry
   label.

**Re-entry.** Every later entry into an exited context lands on that label, uses the `ra` and
result slot delivered by that same entry, and returns `EXITED` again after a bounded number of
instructions, without stack, TLS or `struct pthread`. The switch itself still writes the seal
region, and a preemption on this path remains possible. The path does not wait for a quantum.

**Join.** The joiner waits, by the parking protocol below, until it reads `done == 1` with acquire
semantics. It then copies the result, revokes the parent handle, and returns the area to the
allocator. A detached thread cannot revoke its own area: the runtime keeps exited detached contexts
on a list, and a later `pthread_create` or join can reap them. The publication of `done` is not
itself a wake operation: a joiner already asleep needs a notification through the park service.
Q4 fixes that notification path, the detached-reaping trigger and treatment of the main context.
Probe A may inspect completion directly; it does not thereby qualify a blocking `pthread_join`.

**Why one revoke is enough on one hart.** The ISA switch writes the seal region on every exit, and
the supervisor writes it on every resume. After the revoke, no switch into the context can happen:
the monitor's copy of the seal is invalid, and the supervisor answers `DEAD`. On one hart no switch
out of the context can be in progress at the same time. "Prevent resumption, then reclaim" is
therefore one step. On SMP it is not, and that belongs to the later contract.

**Linux thread end** matters only for the launcher's own state. The host stack, park-queue entries
and per-context transport mappings are freed after the Linux thread has ended and its last STEP has
returned, and after outstanding wakers, signal publication and other launcher references have
released them. `EXITED` or `DEAD` ends the normal STEP loop; repeated entry into an exited context
is a defensive property tested by A4, not the launcher's normal exit policy.

## Parking

Linux cannot see domain memory, so the lock word cannot be handed to `futex` itself. The domain
keeps the authoritative lock state, as native pthreads do. Linux only puts threads to sleep and
wakes them.

**Shared state.** A table of generation words lives in a region shared by every context and the
launcher: the application's process-wide META region, not the per-context blocks.
- Each generation word is an aligned atomic `uint64_t`. Only the launcher writes it, under its
  park mutex, with release semantics; the domain reads it with acquire semantics.
- The key is the domain address of the lock word, as an integer. The launcher never dereferences
  it; the bucket is a hash of the key.
- **No wrap.** Equality does not prevent ABA: a waiter delayed across a full counter cycle could
  see its old generation while its condition has changed. The initial implementation saturates at
  `UINT64_MAX`. On reaching saturation, the launcher completes the requested key's selected waits
  as WOKEN and notifies all remaining bucket records as RECHECK; future WAITs in that bucket return
  RECHECK without sleeping. The bucket stays saturated for the application lifetime. This costs
  repeated checks under contention but cannot lose a wake through wrap. Resetting an epoch requires
  a separate quiescence protocol and is not part of these probes.

**Launcher state.**
- one park mutex;
- for each launcher thread, one wait record `{key, notified, state, outcome}`; `notified` is an
  aligned atomic 32-bit word in Linux-accessible memory, used as that thread's private futex word;
- per bucket, a queue of wait records that holds the full key.

Record states are `IDLE → QUEUED → NOTIFIED → IDLE` or `IDLE → QUEUED → ABORTED → IDLE`.
Queue membership and final state changes are protected by the park mutex. A generation mismatch or
saturation returns RECHECK without registering a wait. The mutex is never held while sleeping or
while returning control to a domain handler.

```
domain WAIT(key):
  g = load_acquire(gen[bucket(key)])
  re-check predicate / retry acquisition
  return only if operation succeeds
  request WAIT(key, g)
  on return: re-check (loop)

domain WAKE(key, n):
  change the protected state (atomic RMW for a mutex)
  request WAKE(key, n)

launcher WAIT(key, g):
  lock
    if saturated or gen[b] != g: unlock; return RECHECK
    initialize record with key, notified = 0, state = QUEUED
    enqueue(b, record)
  unlock
  while load_acquire(notified) == 0 and no abort:
    futex_wait(&notified, 0, remaining deadline)
  lock
    if notified: result = record.outcome
    else: dequeue; mark ABORTED; result TIMEOUT or EINTR
    mark IDLE after all waker accesses are complete
  unlock

launcher WAKE(key, n):
  lock
    advance gen[b] with saturation, release
    select up to n QUEUED records with this key; count them
    complete each selected record as WOKEN
    if saturated: complete every remaining bucket record as RECHECK
  unlock
  return count

launcher REQUEUE(src, dst, nwake, nmove):
  lock
    advance gen[b(src)] with saturation, release
    select up to nwake QUEUED records with key src; complete each as WOKEN
    select up to nmove further QUEUED records with key src; for each:
      if b(dst) is saturated: complete as RECHECK
      else: dequeue from b(src); key = dst; enqueue in b(dst)
    if b(src) saturated: complete every remaining b(src) record as RECHECK
  unlock
  return woken + moved + released at a saturated dst (selected records only; Q5)

complete(record, outcome), with mutex held:
  dequeue; record.outcome = outcome; mark NOTIFIED
  store_release(record.notified, 1)
  futex_wake(&record.notified, 1)
```

- **A WAKE between enqueue and sleep is not lost.** `notified` is already set, and `futex_wait`'s
  value check refuses to sleep.
- **Notification wins.** Under the mutex, a notified record becomes WOKEN; only a record without a
  notification is aborted and dequeued. The saturation flush returns its recorded RECHECK outcome
  instead; it does not consume a selected wake for another key.
  - This is Linux's own order: a waiter already dequeued by a waker returns success before timeout
    or pending signals are considered (`kernel/futex/waitwake.c`).
  - The pending signal stays deliverable.
  - The guarantee covers the completed wait and its cleaned-up record. What a handler does
    afterwards, a `longjmp` for example, is the business of the libc operation.
- **Every wait completes exactly once.** A record is reused only after the waker is finished with
  it; in the first version the waker calls `futex_wake` under the mutex.
- **Collisions** only make a WAIT return early, because the queue keeps full keys and WAKE selects
  by key. Saturation is the explicit broadcast exception. RECHECK and WOKEN are both hints to
  re-examine the protected predicate; neither grants a mutex or proves completion.
- **WAKE result:** count selected records, not the sum of native `futex_wake` return values. A
  selected thread may not yet have entered the kernel, so its native wake count can be zero even
  though the stored notification completes its wait. Saturation RECHECK notifications do not
  contribute to this count, which remains at most `n` for the requested key.
- **REQUEUE advances the source generation.** Without that, a waiter that read the source
  generation and checked its condition before the releaser's change, and enqueues only after a
  REQUEUE that found nobody queued, would sleep through an event that has already happened.
  Generation advance, selection, re-keying and queue membership change under the one park mutex.
  A moved record keeps its waiter's deadline, and timeout or EINTR cleanup dequeues it from
  whichever queue holds it. A REQUEUE is never replayed after it has changed any record.
- **Timeouts and errors:** preserve the original deadline and clock across spurious wakeups,
  signal continuation and retries; do not start a fresh relative timeout each time. Every error
  after enqueue removes or completes the record under the mutex before returning. The API mapping
  of RECHECK, WOKEN, timeout and signal continuation is Q5.
- **External futex ABI and internal deadline are separate.** musl's `__timedwait_cp` converts its
  absolute deadline into a relative timeout for `FUTEX_WAIT`. An external `FUTEX_WAIT` keeps that
  relative meaning. The domain runtime turns it once, at syscall entry, into an absolute
  `CLOCK_MONOTONIC` deadline, which then survives RECHECK, spurious wakeups and signal
  continuation. `FUTEX_WAIT_BITSET` is the launcher's native backend for that deadline.
- **Every direct futex caller goes through this protocol.** For `FUTEX_WAIT(addr, val)` the
  domain runtime reads the generation first, then compares `*addr` with `val` in the domain, and
  only then requests WAIT. Linux never reads `addr`.

**Relation to signals.** WAIT, WAKE and REQUEUE are compound operations in the sense of
`delegation-signals.md`, with their own continuation points:
- a WAIT's record is completed before a domain handler runs or a new WAIT starts;
- a WAKE or REQUEUE that has published a notification or moved a record is reported as done,
  never as `RETRY`;
- internal mutex operations, notification stores and futex wakes are launcher housekeeping;
  they must complete even when the signal ring is nonempty. A partially published wake is not
  abandoned by the delegated-syscall stub.

**Memory order in the domain.** The generation load stays before the re-check, and the change of
the lock state stays before the WAKE request.
- The compiler barrier sits at the domain-side transition: the delegation stub is `asm volatile`
  with a `memory` clobber. A barrier in the launcher's native stub cannot order the domain's
  accesses.
- The hardware argument is for one hart and coherent ordinary shared RAM. CALL/RETURN and trap
  transitions are not asserted to be general memory fences. Record the generated atomics and
  compiler barriers used by the two sides and test visibility through the actual shared mapping.
  The SMP argument is deferred.

## Runtime state: per context and process-wide

| Per context | Process-wide |
|---|---|
| sealed-return capability (today `__capstone_dom_ret`), recovery block behind `cscratch` | heap, and its lock |
| delegation transport: wire entry, exchange region, `dl_*` state (becomes `__thread`) | generation table |
| signal mask, receive ring, handover block | signal dispositions (the `sigaction` table) |
| errno and `struct pthread` (already reached through tp) | capability-global initialization, once |

Before general libc code may run in a second context:
- capability-width atomic calls use a shared lock implemented with supported scalar atomics; the
  lock path must not recursively call the generic atomic library, allocate, or deliver a domain
  signal handler while holding that lock. Report the actual lock-free status and preserve pointer
  tags and each operation's ordering requirements;
- allocator locking and musl's `need_locks` are enabled before the child can enter libc;
- shared runtime structures get documented synchronization and lock ordering. The one-hart
  restriction does not protect an operation that is preempted halfway through;
- while a context holds any runtime-internal lock, no domain handler runs and no reaping happens,
  including inside a yield taken while waiting for a further lock. Otherwise allocator lock held,
  atomics lock contended, delegated `sched_yield`, handler, `malloc` re-requests the held lock (Q6);
- per-context TLS and transport are installed before the first delegated call. Merely changing
  declarations to `__thread` does not construct a child's TLS or initialize its region pointers.

Probe A may initially use preallocated areas, explicit scalar atomics and assembly without a
general concurrent allocator or pthread implementation. It must not be described as qualifying
unrestricted multi-threaded libc. Capability-pointer atomics and linear-capability hand-off are
separate contracts; ordinary C atomic loads do not acquire permission to copy a linear capability.

Preemption on one hart is enough to break an unlocked read-modify-write.

## Probe A: context and lifetime

A domain program with one mode per case, driven from the host like `signal-contract.dom`. Each case
names the state it starts from, the point it forces, and the condition it checks.

Points are forced deterministically:
- a register-only delay loop longer than one quantum on the exit path (the supervisor's preemption
  event PC must fall inside the named loop's label range; a counter alone cannot locate it);
- an explicit call to the supervisor's collector;
- slot or node exhaustion by construction.

Inspect the exit assembly to establish the absence of stack/TLS accesses after `done`; the delay
loop exercises a chosen interleaving, not every possible instruction boundary. Faulting negative
tests use separate executions and an expected cause/location, rather than accepting any crash.

| Case | Start | Forced point | Postcondition |
|---|---|---|---|
| A1 mint and enter | main context running | adopt one minted seal | the second context runs its function with its own sp and tp; both see one global counter; the initializers ran once |
| A2 authority at entry | minted seal with an explicit allowed-authority inventory | first thread-entry instruction, then bootstrap complete | registers and reachable capabilities grant only the intended shared image/start-block authority and bounded return/result/mailbox grants; a separate negative access outside those grants faults as expected; no monitor-private authority leaks |
| A3 preemption | second context in a long loop | at least two quanta | resumes with its register state intact (checksum) |
| A4 re-entry after exit | second context exited | probe-only child handles revoke stack, TLS and start/recovery block while retaining the seal; repeated STEP, including preemption in the exit tail | `EXITED` each time using fresh entry grants; no access to revoked children; the normal runtime still uses one parent revoke |
| A5 join and reuse | exited, `done` set | joiner copies, revokes, reallocates the area, writes a pattern | pattern intact; the joined result is the one the thread returned |
| A6 preempted after `done` | exit path between release store and return | joiner revokes and reallocates; the launcher STEPs the old context | DEAD if still registered, STALE if already retired; no re-entry; pattern intact |
| A7 capability ABA | context A paused | revoke A's area, run the collector, re-mint in the same area until the new seal has A's node and bounds | STEP of the new context enters its own thread entry, not A's continuation |
| A8 late STEP and FORGET | context revoked | separate sequences before and after GC: STEP and repeated FORGET, including a tagged-invalid operand where available and an untagged one | DEAD before registration retirement, STALE afterwards; no exception in the monitor; repeated removal has no effect; a live sibling still runs |
| A9 id ABA | slot k reused with a new generation | late STEP and FORGET with old g; generation preset near exhaustion | stale requests have no effect on the replacement; an exhausted generation never wraps into an old id |
| A10 slot exhaustion | Linux never forgets | four times as many create/exit/revoke cycles as the largest registration table | supervisor, monitor and driver slots are all reusable; every create succeeds at bounded live occupancy; a live context keeps running |
| A11 adoption rollback | one creation in progress | fail before consumption, after adoption, and at Linux-thread creation; replay an old ticket after a new offer | no new child executes after a reported pre-start failure; the old ticket cannot consume the new offer; area reclaimed by revoke; no registration leak |
| A12 control words | separately minted seals with alternative privilege/status/mask values, including `mie = 0`; running main and sibling contexts | first supervised entry and a quantum expiry; CSR reads and writes above user level in both running contexts; `mret` and `sret`; a context calling a minted seal directly (nested, unsupervised CALL) | the Q3 contract is enforced: clean refusal or confined execution with protected caller state and bounded return; an unexplained crash or escape fails |
| A13 single registration and entry | one context adopted | ADOPT again from the consumed offer; two launcher threads STEP the same `(k, g)` | duplicate adoption refused; STEPs serialize into one continuous execution; a controlled foreign-owner request is refused |
| A14 loan across preemption | context stores its descriptor capability, then computes past a quantum | resume, then write through the stored copy | the write lands; the loan outlived the preemption |
| A15 loan after its end | context stores its descriptor capability; the call returns cooperatively | separate runs: write through the stored copy (a) during the next call of the same context, (b) after the slot is reissued to another owner | no authority in both runs: fault at the expected location, the current descriptor unchanged; the same check first against today's result slot |

## Probe B: parking

First as a native program with ordinary Linux threads, using test hooks in the park code that stop
a thread at a named point. Then in a domain, where it adds the memory-order, blocking, signal and
atomics cases. B1 to B5, B12 and B13a to B13d are native, each B13 variant a separate run; B6 to
B11 and B14 need Probe A and the relevant runtime integration.
The native phase tests queue mechanics; it does not qualify the domain-side memory-order argument.

| Case | Start | Forced point | Postcondition |
|---|---|---|---|
| B1 wake before sleep | waiter enqueued | waiter stopped after unlock, before `futex_wait`; waker runs | WOKEN without sleeping |
| B2 collision | waiters on keys A and B in one unsaturated bucket (table of one bucket), both asleep | WAKE(A, 1) | only A is selected and completes as WOKEN; an unrelated native spurious wake of B is rechecked internally |
| B3 abort against selection | at least two waiters queued on one key | force both mutex orderings: selection before abort, and abort before selection, for timeout and EINTR | selection first gives WOKEN; abort first removes the record so WAKE can select the next waiter; every wait completes once |
| B4 record reuse | waiter selected | hold the waker at its last record access; attempt completion and a subsequent wait; inject a native spurious wake | record cannot be reset while the waker owns it; after completion the same record can wait again; no lost notification |
| B5 saturation | generation preset to UINT64_MAX - 1, with waiters on colliding keys and a delayed pre-enqueue waiter | WAKE reaches UINT64_MAX; repeat WAIT and WAKE | selected matching records get WOKEN, other queued records get RECHECK; no wait is stranded; new/delayed WAITs return RECHECK; no wrap/reset or wake count above n |
| B6 domain order | two contexts on one lock | preemption between generation load and re-check, and between state change and WAKE | no lost wake; counter exact |
| B7 blocking | context 1 in a delegated `read` on an empty pipe | context 2 runs and writes | context 2 runs meanwhile; the read returns its data |
| B8 signal continuation | WAIT queued, or WAKE or REQUEUE partially publishing | signal to the relevant Linux thread at each named point; handler makes a delegated call | record cleaned before the handler runs; notification takes precedence; a partially committed WAKE or REQUEUE completes once without replay; original timeout budget retained |
| B9 atomics under preemption | two contexts using scalar counters and capability-valued load/store/exchange/CAS | preemption in scalar LR/SC loops and while the generic atomic lock is held | scalar counts and pointer tags/bounds/identity correct; holder can resume; no recursion through the generic atomic lock |
| B10 blocking join | joiner already parked on completion | child publishes done then exits; separate run preempts child after done before exit | notification eventually releases joiner under a progressing Linux schedule; result acquired from protected state before revoke; no C/stack/TLS work after done |
| B11 creation and reaping | runtime can create siblings | child runs before create reply; detached completions and concurrent reaping; main context exits while a sibling lives | Q2/Q4 startup and lifetime rules hold; one claimant per area; no premature reuse; resource counts return to the documented bound |
| B12 requeue before enqueue | waiter has read the source generation and checked its condition | stopped there; releaser changes the condition and issues REQUEUE with nobody queued; waiter resumes into WAIT | WAIT returns RECHECK; no sleep through the event |
| B13a requeue, then wake | waiters queued on src, colliding keys queued in src's and dst's buckets | REQUEUE(src, dst, 0, 1), then WAKE(dst, 1) | only a src waiter moves; WAKE(dst) completes it as WOKEN; the colliding waiters stay queued; REQUEUE returns 1 |
| B13b requeue, then timeout | as B13a | REQUEUE(src, dst, 0, 1); the moved waiter's deadline expires | TIMEOUT; the record is dequeued from dst's queue; no record in two queues |
| B13c requeue, then EINTR | as B13a | REQUEUE(src, dst, 0, 1); a signal interrupts the moved waiter | EINTR, or the signal plan's continuation; dequeued from dst's queue; the deadline is kept for a continuation |
| B13d requeue into a saturated bucket | as B13a, dst's bucket saturated | REQUEUE(src, dst, 0, 1) | the selected src waiter completes as RECHECK and is counted; no record is enqueued in dst; REQUEUE returns 1 |
| B14 no handler under internal locks | context 1 holds the allocator lock and waits for the generic-atomics lock, which context 2 holds; a signal with a `malloc`-calling handler is pending for context 1 | context 1 spins or yields while waiting | context 1's handler runs only after it holds no runtime-internal lock; no deadlock; no reaping meanwhile |

Native abort tests use interrupted wait outcomes; delivery through the domain signal ring is checked
by B8. Returning WOKEN does not suppress a signal accepted for domain delivery. Full pthread gates
also verify allocator contention and the selected mutex/condition-variable semantics; a passing
parking probe alone is not full futex or POSIX-thread conformance.

## Questions to resolve during the probes

The questions below do not reopen the one-hart, Linux-scheduled architecture. Record each answer
and its test evidence here before accepting the affected gate. Implementation may proceed on
independent probe cases while an answer is pending.

### Q1. What exactly is the adoption ABI? (before accepting Probe A)

Which register or bounded descriptor carries the result slot, the 16-byte hand-off slot and the
offer ticket? How does the monitor associate the ticket with the offer, reject a stale request
after mailbox reuse, and return or retire an offer when no slot is available? Specify id widths,
generation exhaustion, owner binding and the driver-visible DEAD/STALE/empty-offer statuses.

Recommended starting point: a bounded per-context invocation descriptor, one outstanding offer,
and no implicit retry after consumption. Preserve the distinction between the supervisor's
capability-based forget and the monitor's integer-id FORGET. Confirm mailbox authority cannot
survive slot reassignment into another application's descriptor.

- **Descriptor.** One monitor-owned block per context slot: result word, status, offer ticket, and
  a 16-byte capability slot for the offered seal. It is lent as one capability derived under its
  own revocation handle, which `csshrinkto` and `cstighten` alone do not provide. The loan is
  write-only, but capstone-qemu checks tag, revocation and bounds on a data access and no
  permission (`op_helper.c` `_helper_access_with_cap`), so on this platform its authority is its
  64 bytes: A2 reads through it without a fault (ISSUES Q-14).
- **Loan lifetime.** A loan covers one logical call, including every preemption and resume of it:
  a resume restores the callee's snapshot with the old descriptor capability and delivers no new
  one, so a revoke at a preemption escape would break a regular continuation. The loan is revoked
  at cooperative return or when the continuation is finally given up. Before the descriptor serves
  another call or owner, every older derivation is invalid, and an offered seal has been adopted
  or preserved before the descriptor memory is reinitialised. This per-call revocation is the
  contract of the first version, and A15 tests it. It costs one revocation node per round, which
  Probe A records; the supervisor collects by itself only when supervised code runs short, so the
  monitor collects before a loan when fewer than 1024 nodes are free (without that, T1 found, one
  process's 65536th call halted the monitor with cause 30). Revoking only before the slot is reissued to another generation or owner would
  still keep owners apart, but it is a later contract change that has to amend A15, not an
  implementation option. A14 and A15 test both sides; the existing result slot is examined first.
- **Ticket.** A counter the domain writes next to the offered seal. ADOPT consumes the offer only
  when ticket and slot match. It answers `STALE` for an old ticket, `EMPTY` for no offer, and
  `FULL` when no context slot is free, leaving the offer in place. After `FULL`, the domain revokes
  the thread area, and the monitor clears the now invalid offer at the next offer or call.
  - **One offer at a time (P2, decided 2026-09-30).** A first, still-valid offer is kept: while one
    is outstanding a second offer of the same context is dropped and its ticket is `STALE`, so the
    `FULL` the domain saw stays repeatable. Once the first offer's seal is revoked (it reads
    untagged) the next offer replaces it. `offer-keep` and `offer-replace` test the two cases; the
    pre-fix monitor overwrote the first offer and fails `offer-keep`.
- **FORGET by id.** On capstone-qemu a revoked seal reloads untagged (Q-11). The monitor therefore
  cannot rely on the tagged-invalid case for its own copy: its FORGET works on `(k, g)`, and the
  supervisor slot is removed by the collector.

### Q2. When may the child start, and what does musl use as its tid? (before runtime integration)

What must be published before the first child STEP, and which state is already valid if the child
runs before the parent's create reply? Will `t->tid` contain a Linux tid, or will mutex ownership
use a protected runtime identity with an explicit translation for `tkill`/`tgkill`? Document how
untrusted or reused Linux tids affect recursive mutex ownership and signal targeting without
confusing protected context identities. Native launcher TLS is separate from domain TLS.

Recommended starting point:
- **Transport before mint.** The creator requests the child's transport region in a separate
  round. The launcher creates it and shares it with the creator, and the creator places it in the
  child's start block before sealing. Everything the first STEP needs then exists, whenever the
  Linux thread starts. The launcher releases the region after the Linux thread has ended.

**Answer, first part: the transport (2026-09-30, T1; evidence `results/20260930-transport.json`).**
- **Declared, not negotiated.** The image declares how many contexts besides the first may run at
  once with a transport of their own: `CONTEXTS` (0 to 7; the monitor lends each application 8
  descriptors), a field of the application descriptor. The launcher grants 1 + CONTEXTS transports
  as two regions shared at launch: META blocks of 16 KiB (entry at 0, signal handover at 4096) and
  exchange slices of the declared `EXCHANGE_BYTES` (a multiple of 4096), transport i at i times the
  block size in both. Transport 0 is the first context's, byte for byte where it was. No region is
  shared later: a share reaches only an application's first context, and only between calls.
- **Reserved before the request.** `CONTEXT_RESERVE` returns a free transport index; the creator
  writes it into the child's start block (`WORD_TRANSPORT`) and then asks `CONTEXT_CREATE(ticket,
  THREAD, index)`. The launcher binds the transport to the adopted context before it starts the
  thread, so the child's first entry already has it, whenever that is. A request consumes the
  reservation whatever its outcome. The launcher frees a transport when the context's thread has
  stepped it for the last time and forgotten it; until then RESERVE may answer EAGAIN.
- **Installed at entry.** The entry glue calls `__capstone_context_run(start block)`, which
  installs the transport (bounded capabilities to its entry block and exchange slice, cut from the
  first context's region capabilities by the declared sizes) into the context's TLS block and then
  calls the application's function. The delegation state (`dl_*`) is `__thread`.
- **Signals** were the first context's in T1 (a further context's signal requests answered
  ENOSYS, its thread blocked every signal, its SIGPIPE was forwarded to the process). B8 made them
  every context's own; see "Signals per context" below.
- **What ends with a context and what with the process.** The context's thread stops at the first
  step that is neither a preemption nor a request (EXITED in every probe mode; DEAD, STALE and
  REFUSED take the same path), forgets the context and frees its transport. `exit()` or a fault in
  any context ends the process, as in Linux; nothing is unmapped first, since other threads may
  still be serving. Exec in place is served from any context as from the first (not yet run with
  more than one context).
- **Open in Q2** after T1: the translations of a thread identity to a Linux tid (`tkill`,
  `tgkill`, `pthread_kill`, `sched_*`, `SIGEV_THREAD_ID`), and the start of a child before its
  creator's reply as seen by musl. B8 answers `tkill` (and with it `raise` and `pthread_kill`);
  `tgkill`, `rt_tgsigqueueinfo`, `sched_*` with a thread and `SIGEV_THREAD_ID` are not on the wire
  and stay reported as unserved, so `PTHREAD_EXPLICIT_SCHED` answers ENOSYS. musl's start of a
  child before its creator's reply is T4's (the child waits for the thread-list lock).

**Answer, second part: the thread identity (2026-09-30, T3; evidence
`results/20260930-runtime-locks.json`).** The first context's identity is the pid, as Linux gives
a process's first thread. A minted context gets one from a process-wide counter from 2^22 to
0x3ffffffe: above Linux's pids (below `PID_MAX_LIMIT`, 2^22), and inside the 30 bits musl keeps in
a lock word (bit 30 is `MAYBE_WAITERS`, 0x3fffffff a marker). It is written into the context's
`struct pthread` before the seal is offered and never given to another context in the process's
life, so a recursive lock (musl's `__lockfile`, a recursive mutex) never takes a new context for
an old owner; when the range runs out, minting fails.
`gettid` and `set_tid_address` answer it without a round. Linux tids are not visible to the domain.
Since B8, `tkill` reaches the launcher thread serving the context it names (the pid: the first
context's); another identity answers ESRCH.
- **One identity inside the domain.** `t->tid` and every tid the domain sees is a protected runtime
  identity, never reused while its `struct pthread` is live. Linux reuses tids after thread exit
  even without malice. The launcher translates runtime identity to Linux tid for its own contexts
  only, and answers ESRCH for anything else. The translation applies consistently to `gettid`,
  `set_tid_address`, `raise`, `pthread_kill`, `tkill`, `tgkill`, `rt_tgsigqueueinfo`, `sched_*`
  calls that take a tid, and `timer_create` with `SIGEV_THREAD_ID`. Linux tids remain visible only
  in host artefacts such as `/proc/<pid>/task`, a named deviation.

### Q3. Which sealed control states are admitted? (before accepting Probe A)

Which privilege/status/mask combinations preserve capability confinement and the supervisor's
return guarantee? Where is that enforced when the monitor cannot inspect a sealed region? The
supported runtime uses the established entry template, but that alone is not validation of all
seals ADOPT can receive. Fix the allowed outcomes for A12 before running it; do not turn whatever
QEMU happens to do into the contract.

Recommended starting point:
- Treat the first entry and the resume separately. At first entry, the supervisor, the only party
  besides the domain that can read a sealed region, compares the control words with the
  `create_domain` template and refuses a mismatch with a clean status. A resume restores the
  supervisor's own snapshot and needs no template.
- During a supervised call, `riscv_csrrw_check` already refuses every CSR above user level. A12
  confirms this for the main and a sibling context rather than assuming it.
- `mret`, `sret` and a direct, unsupervised CALL of a minted seal from inside a supervised context
  are not covered by either check. A12 establishes what they do before the contract names them.

**Answer (2026-09-29; evidence: A12 in `results/20260929-context-probe-monitor.json`).**
Before the change, A12 showed two escapes: `mret` in a supervised context was carried out
(domains run at `PRV_C`, which is `PRV_M`), and a seal minted with user privilege was entered
with it. Both times the context left C-mode for user mode on the owner's page table and faulted
at its first fetch there (cause 12) only because the target address was not mapped in the
launcher; outside C-mode the quantum is not polled either (read from `translate.c`, not
measured). The contract, now enforced by the supervisor (capstone-qemu `qemu/context-slots-on-pin`):
- A supervised context runs in C-mode only. `mret` and `sret` raise an illegal-instruction fault;
  a fresh supervised entry into a seal whose saved privilege is not C answers REFUSED (step
  event 5) and is not entered; a nested CALL into such a seal faults. A resume restores the
  supervisor's own snapshot and is not checked.
- CSRs above user level stay refused (`riscv_csrrw_check`), shown for a machine CSR.
- `mie`, `mideleg`, `medeleg` and `mip` in a seal are admitted: with a seal enabling machine
  interrupts the context still ran confined and was preempted 2096 times.
- `wfi` is a no-op on this platform and returns.

### Q4. Who notifies joiners and reclaims detached or main contexts? (before B10/B11)

Which event invokes WAKE on the completion key after `done` is published? The exit assembly cannot
call the ordinary dispatcher. Recommended starting point: retain the completion key in the
launcher's creation record and notify on EXITED/DEAD; this is only a wake hint, while the joiner
still checks protected `done`. Specify the case where the joiner observes done and revokes before
the child reaches its final exit return.

Who reaps detached contexts when there is no later create or join, and how is the single reaper
claim synchronized? Can the main context's stack/TLS use the same arena discipline, or are they
retained until application destruction? Distinguish main-thread `pthread_exit` from process exit,
which terminates the application. Until these are answered, the probes do not promise prompt
detached-resource reclamation or complete main-thread pthread semantics.

Recommended starting point:
- **Completion wake:** as above. The launcher wakes the completion key from its creation record
  on `EXITED` and on `DEAD`. If the joiner already saw `done` and revoked, the wake is redundant
  and harmless.
- **Reaper list.** Each node lives outside every thread area and holds the parent handle and a
  state word. A context claims a node by CAS on that word, so each area has exactly one claimant.
  Iteration and unlinking run under the runtime's reaper lock and read only nodes, never the
  thread area, so a revoked area is never touched. A node is freed only after it has been
  unlinked under that lock.
- **Reaping points:** `pthread_create`, `pthread_join`, and a bounded amount at each dispatcher
  entry, never while a runtime-internal lock is held (Q6). Reclamation then progresses as long as
  any context makes delegated calls. The leak is bounded by the number of exited, unreaped
  detached contexts.
- **Main context.** Its stack and TLS were built by the monitor, not taken from the arena, so they
  are retained until the application is destroyed. `pthread_exit` in the main thread is a context
  exit; the application ends with status 0 when its last context has exited. `exit` and
  `exit_group` end the application at once.

**Answer (2026-09-30, T4; evidence `results/20260930-pthreads.json`).**
- **The joiner's word is musl's.** musl's `pthread_exit` stores `DT_EXITED` and wakes the joiner
  itself; its join then waits on the thread-list lock, which the ending thread holds from its
  unlink to its end and which Linux releases through CLONE_CHILD_CLEARTID. That clear word is the
  completion key: the context names it in CONTEXT_EXITING (its last request), stores 0 into it
  after its last use of the stack and TLS, then publishes `done` and returns EXITED; the launcher
  frees the transport and wakes the key once on EXITED and on DEAD. A joiner that saw the word
  clear before the final return goes on and may free musl's mapping at once; the wake is then
  redundant. A reservation that finds no transport free waits while one is ending.
- **Reaping.** One record per area, outside every area, under one lock (`clones_lock`, before the
  arena, map and heap locks). The reaper runs at every `__clone`, revokes each area whose `done`
  is 1 and keeps it for the next thread, and gives a detached thread's mapping (recorded by
  `__unmapself`) back to the heap. Join frees a joinable thread's mapping itself (musl). The bound:
  a finished, unreaped thread keeps one area and at most one mapping until the next thread is made;
  records never exceed the contexts alive at once plus those between their clear and their `done`
  when a thread is made (seven in 60 rounds of seven, one for 300 detached threads). The reaper
  revokes only contexts that published `done`, which never park again, so no parked record
  survives a reaped context (Q5's open point).
- **Main context.** `SYS_exit` in the first context clears and wakes its word itself (its stack and
  TLS are never freed) and then waits for good, its signals as musl left them (all blocked), while
  another thread lives; the last thread's `pthread_exit` finds itself alone and calls `exit(0)`, as
  on Linux. Alone, the first context's `SYS_exit` ends the application with its status. `exit`
  and `exit_group` end the application from any context. Only `exit_group` crosses to the
  launcher; a thread's end is the domain's.

### Q5. What is the park API's timeout and signal contract? (before accepting Probe B)

Specify deadline representation, clock, allowed counts, result codes and the mapping to libc's
futex-facing operations. Which unnotified interruption finishes as EINTR, and which becomes a
signal-plan RETRY? A selected wake finishes as WOKEN; a saturation flush finishes as RECHECK.
All continuations preserve the original deadline and finish queue cleanup before domain signal
delivery. State the initially supported futex operation subset rather than silently
approximating requeue or PI operations.

Recommended starting point:
- **Supported operations:** `FUTEX_WAIT`, `FUTEX_WAKE` and `FUTEX_REQUEUE`, with or without
  `FUTEX_PRIVATE_FLAG`. Domain memory is never shared with another process, so both forms are
  private. Everything else answers ENOSYS and stays visible in the unserved report.
- **Timeouts:** an external `FUTEX_WAIT` keeps its relative timeout. The domain runtime converts it
  once, at syscall entry, into an absolute `CLOCK_MONOTONIC` deadline. The park service's WAIT
  takes only that absolute deadline, and the launcher waits with `FUTEX_WAIT_BITSET`. An external
  `FUTEX_WAIT_BITSET` is not supported at first.
- **Results:** WOKEN maps to 0, TIMEOUT to `-ETIMEDOUT`, an interruption without notification to
  `-EINTR` (or to the signal plan's continuation where it applies), and RECHECK to 0. RECHECK is
  like a spurious wakeup, which every futex caller must tolerate.
- **Counts:** WAKE returns the number of selected records, at most `n`. REQUEUE takes the wake and
  move counts as Linux passes them and returns woken plus moved records, as Linux does: in
  `kernel/futex/requeue.c`, `futex_requeue` returns one shared `task_count` ("the number of tasks
  requeued or woken"), also without PI. The manual page's "woken" understates this, and musl
  ignores the value. A selected move candidate whose target bucket is saturated is released with
  RECHECK and counted once, so the result is at most `nwake + nmove`. Records completed only because
  a bucket is saturated, but never selected, count for neither WAKE nor REQUEUE.

**Answer (2026-09-30, T2; evidence `results/20260930-park-domain.json`).**
- **Served:** `FUTEX_WAIT`, `FUTEX_WAKE`, `FUTEX_REQUEUE`, with or without `FUTEX_PRIVATE_FLAG`,
  with Linux's argument checks (a misaligned word and negative REQUEUE counts are EINVAL; WAKE
  asked for none or fewer wakes one). Everything else (PI, `WAKE_OP`, `CMP_REQUEUE`, `WAIT_BITSET`,
  and any operation with the realtime clock flag) answers ENOSYS and is reported as unserved.
- **Wire:** `PARK_WAIT(key, gen, deadline)`, `PARK_WAKE(key, n)`, `PARK_REQUEUE(src, dst, nwake,
  nmove)`; the key is the word's address; the park table (256 generation words, 4 KiB) follows the
  last META block, and both sides compute the bucket with `capstone_park_bucket_of`.
- **Domain order:** generation load with acquire, compare of the word, request; a changed word
  answers EAGAIN without a round. A relative timeout becomes one absolute `CLOCK_MONOTONIC` deadline
  at entry, kept across every round. The domain's clock is the launch record's, extrapolated with
  `rdtime`, and the launcher sleeps on the kernel's: the deadline can be off by the microseconds
  between the two reads at launch and by any clock slew since, a named deviation.
- **Results:** WOKEN and RECHECK map to 0, TIMEOUT to `-ETIMEDOUT`, an interruption to `-EINTR`.
  On the first context's thread the park sleep goes through the signal stub: an untimed wait
  interrupted under SA_RESTART ends as a RETRY round, its record already out of the queue; the
  handler runs, and the domain compares the word again before it waits again, as Linux restarts the
  call. A timed wait interrupted in its sleep answers EINTR even under SA_RESTART, as Linux does (it
  restarts a timed futex wait through a restart block, which a handler turns into EINTR; checked
  natively). A signal accepted before the sleep begins ends any wait as RETRY, deadline kept, as a
  signal before the call would. (Before B8 this held on the first context's thread only; a
  further context's thread blocked every signal and was never interrupted; since B8 every
  context's thread takes its own signals.)
- **Open for Q4 and B11:** a context revoked while its thread is parked leaves its record queued,
  and a WAKE can select that record instead of a live waiter. Reaping a parked context has to abort
  its wait first. (T4's reaper revokes only contexts that published completion, which never park
  again; an application revoking a live context through the context interface still can.)
- **Counts:** WAKE returns the records it selected; REQUEUE woken plus moved.

### Q6. How are the first runtime locks bootstrapped? (before concurrent libc)

Which scalar atomic primitive protects generic capability atomics and allocator metadata without
depending on those same services? Specify lock ordering, when signal handlers may run, and how
the holder makes progress after preemption on one hart. Name the shared runtime tables covered by
the audit and verify that pointer atomics preserve tags as well as values. Probe A's preallocated
assembly harness does not discharge this requirement.

First part, settled with T2: every application is built with the A extension
(`PORT_C11_ATOMICS=ON`, required by `capstone_configure_application`, as musl already is), so the
runtime and the application have scalar atomics.

Recommended starting point:
- **Primitive:** a spinlock on an aligned 4-byte word with scalar LR/SC. It is correct under
  preemption because a supervised switch drops the reservation (`load_res = -1`).
- **Waiting:** first spin and rely on the existing 5 ms preemption, which enters no dispatcher.
  A bounded yield path is added only if spinning proves too costly, and it must defer domain
  handlers and reaping.
- **Lock depth:** each context keeps a count of runtime-internal locks it holds. While the count is
  non-zero, the dispatcher runs no domain handler and no reaper, including during any yield taken
  while waiting for a further lock. B14 tests this.
- **Scope of the rule.** A handler that calls `malloc` while the interrupted code is inside
  `malloc` is the application's undefined behaviour, natively as well. The rule protects the
  runtime's own paths, reaping and any async-signal-safe function whose implementation here takes
  an internal lock, and it keeps a pending handler from running inside a lock chain the
  application never saw.
- **Ordering:** the generic-atomics lock is a leaf lock; the allocator lock may be held while
  taking it, never the reverse.

**Answer (2026-09-30, T3; evidence `results/20260930-runtime-locks.json`).**
- **Two kinds** (`capstone/lock.h`): `capstone_lock` is musl's `__lock`, a futex lock through T2's
  parking, a no-op while the application has one context; `capstone_spin_lock` is the leaf spin
  lock on a scalar word. Neither allocates or calls the generic atomics.
- **Covered state:** the heap (level0 and the Sublet heap, one lock each over their tables, public
  entries taking it once), the mmap/shm tables, the context arena, the static posix_spawn and
  execve request blocks (one lock for both), and the unserved-call notes (spin); the
  capability-width atomics (spin). musl's
  own locks (stdio's FILE locks, its internal `__lock` users) are switched on at the first THREAD
  context exactly as musl's first `pthread_create` does: FILE lock words leave -1,
  `libc.threaded`, `threads_minus_1` and `need_locks` are set. The count only rises for now, since
  a context's end is not seen by the runtime yet (Q4): correct, only slower.
- **Order:** maps, then heap, then atomics; the spawn and arena locks take no other.
- **No handler under a lock:** each context counts the `capstone_lock`s it holds, whether or not
  musl really took them (it does not while the application has one context), from the moment the
  first context has a TLS block; a round's delivery is skipped while the count is not zero, and
  the last release delivers what waited. A spin lock makes no round while held, so no handler can
  run under it and it is not counted.
- **Not yet:** `__synccall` (setuid and relatives) signals every thread in musl's list and needs
  signals per context (B8); a REGISTER context holding a runtime lock runs again only when its
  stepper steps it. (Thread-specific data came with T4: musl's `pthread_create` gives each thread
  its array.)
- **Progress on one hart:** a spinner spins until the scheduler preempts it and the holder runs;
  a `capstone_lock` waiter parks.
- **Measured cost:** none measurable on a single context's rounds: 100000 rounds took 9134 ms with
  T2's probe image and 9239 ms with T3's on this runtime, against 9069 to 9152 ms on the T1 and T2
  runtimes. (A T3 image from before the review's fixes took 9863 ms; the final one does not
  reproduce that, and the difference is not explained.)

### Signals per context (B8)

**Answer (2026-09-30, B8; evidence `results/20260930-signals-per-context.json`).** Linux keeps the
dispositions per process and the mask, the pending signals and the alternate stack per thread;
so does this runtime, one context per thread.
- **Launcher.** One disposition table per process, shared by every context's state; each
  context keeps its ring, in-flight set, backpressure, logical mask and handover block (in its
  transport's META block). Each context thread's kernel mask is its context's logical mask with
  the in-flight and backpressure signals added, set on that thread, so Linux itself picks the
  thread for a signal sent to the process, and the trampoline records into the ring of the context
  its thread serves (`__thread` state, attached before the thread's first step; a signal that
  reaches a launcher thread before it attached is kept pending, on that thread if Linux sent it
  there and on the process otherwise). A new context's mask is its creator's, as a new thread's
  is. The masks and the actions go to the kernel directly: glibc filters its own internal signals
  (32 and 33) out of every mask and refuses them in `sigaction`, and a domain's musl uses them
  (its timer and cancel signals); the launcher uses neither glibc feature behind them.
- **Requests.** Every signal request is every context's own (only HELLO stays the first
  context's). SIGACTION changes the shared table under the owner's lock. `tkill` names a context
  by its thread identity (CONTEXT_CREATE now carries it) and becomes `tgkill` to the launcher
  thread that serves it, sent under the service lock that thread takes to leave the table, so
  the Linux tid named is never one Linux gave to another thread. A spawned child and an exec in
  place start with the calling context's mask. The SIGPIPE forwarding of T1 is gone: the writing
  thread's own mask and the process's disposition decide.
- **Domain.** The handler table is shared (changed under a lock, request and table together);
  the mask mirror, the events (on the heap, 256 per context, taken when the transport is
  installed), the alternate stack and the delivery state are the context's. A context that ends
  gives its events back, freed by the reaper for a musl thread (which may take no musl lock at its
  end) and at once otherwise. A handler's sigreturn mask is honoured: the bits it changed in
  `uc_sigmask` are applied over the mask from before the delivery.
- **Cancellation.** musl's `cancel_handler` moves the interrupted pc from
  `[__cp_begin, __cp_end)` to `__cp_cancel`. A delegated call is rounds, not instructions, so
  `__syscall_cp_asm` is C: it marks the context as inside a cancellation point, a handler that
  runs before the call's own round has its result (at the SIGPOLL round before it, or at a RETRY
  round) is shown `__cp_begin`, one after it `__cp_end`, and a handler that moved it to
  `__cp_cancel` ends the call with EINTR without issuing it again, on which musl's
  `__syscall_cp_c` cancels. The three symbols are consecutive bytes nothing executes. A RETRY
  round that is not issued again now answers EINTR everywhere (it answered 0 in the general call
  and in readv/writev, which a cancelled `read` returned as end of file).
- **Delivery is synchronous.** A context takes its signals at its next delegated call; one that
  computes without calls takes them later, and asynchronous cancellation of such a thread does not
  happen (libc-test's `pthread_cancel` starts with exactly that, `for (;;);`). That needs the
  doorbell of delegation-signals.md ("The bell, later"), which this branch does not build.

## Relation to the Capstone paper

The paper (https://arxiv.org/abs/2302.13863) already covers three things:
- **Several physical threads.** Its formal model (Section 4.1) gives each physical thread its own
  register file; at each step the machine picks one thread and executes one instruction on it.
- **An untrusted preemptive scheduler.** This is one of its motivating examples (Section 2.2).
- **Linear capabilities as Rust-like ownership** (Sections 2.2 and 2.3).

None of these is a contribution of this work. What this work adds is the concrete integration with
Linux scheduling: protected continuations, revocation, and the reuse of identities. Cases A7 and A9
exist only because an implementation recycles node ids and slot ids.

The SMP question left open is how a real multicore implements the atomic capability operations and
the global revocation the model assumes, under its actual memory model.

## Delivery

One branch per repository, pinned together; each submodule change gets its own upstream pull
request, opened once the runtime's gates below pass.

- **capstone-qemu:** the supervisor rules (`DEAD` status, dead-slot removal in the collector and on
  slot shortage, forget of an invalid seal) and the first-entry control-word check (Q3).
- **capstone-sbi:** the per-context descriptor and its revocable loan, `ADOPT`, generation ids,
  `DEAD`/`STALE`/`EMPTY`/`FULL`, and application memory kept apart from context slots.
- **caplifive-buildroot (driver):** context ids apart from process blocks, and the `ADOPT` ioctl.
- **llvm-capstone:** mint, thread entry, exit path, per-context state, locks, the park protocol in
  launcher and libc, and the two probe programs.

After both probes, `delegation-threads-runtime` builds musl's `__clone`, and with it
`pthread_create`, `join`, `detach` and `exit`, on mint and adopt. The minimal integration needed by
B6 to B11 is not yet a complete pthread implementation. Resolve Q2, Q4 and Q6 before treating that
runtime as generally thread-capable. Gates:
- the libc-test thread group leaves the excluded set (seven of the nine pass after B8;
  `pthread_cancel` waits for the doorbell, `sem_open` for file mappings, not threads);
- GLib's `GCond` in the tshark deps (the `pthread_cond_t` size fix, e2c9ad3; the layout itself is
  fixed by musl patch 0004): pass 2026-09-30, glib-0008 is gone and GLib's own `cond` test passes
  in a domain, with `thread`, `rec-mutex` and `asyncqueue`; `mutex`, `once` and `rwlock` stop only
  at their hundred-thread tests (`runtime/tests/application/results/20260930-glib-threads.json`);
- CPython's basic `threading` tests: the subprocess-enabled review run passes all five modules
  (448 tests, 38 reported skips) after the level0 shrinking-realloc fix. Use
  `ports/cpython/interpreter/results/thread-review-2026-09-30.json` for the explicit exclusions;
  subprocess errors in the original `threads-2026-09-30.json` were blocked tests, not passes.
