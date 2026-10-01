# Delegated threads: the computing model

This document describes how a Capstone application runs several threads on the delegated
runtime:
- which component does what;
- how a thread is created, run, blocked, signalled and ended;
- what Linux decides, and what it cannot do;
- where the model stops.

It describes the system as built. [The plan](../plans/delegation-threads.md) holds:
- the contracts this rests on;
- the probe cases that test them;
- the record of every run, including the defects found on the way.

Section 13 maps each property to its evidence.

The platform is capstone-qemu with its supervisor extension, on one hart. Nothing here is a claim
about the FPGA or the RTL (section 11).

## 1. The model in brief

Each thread of an application is a protected *context* in the domain. Each context is driven by an
ordinary Linux thread of the launcher, `capstone-exec`. The Linux thread *steps* its context: it
enters the domain through the driver and the monitor. The domain runs until one of these happens:
- it needs a service from Linux (a delegated system call);
- it ends;
- it faults;
- its quantum expires.

The step then returns to the Linux thread, which serves the call with Linux's own semantics and
steps again.

Linux therefore schedules, blocks and routes signals per thread, as it does for native threads.
The monitor only ever handles sealed contexts; it knows nothing about threads.

Two limits follow from how a step is built:
- Linux can switch between contexts only at step boundaries.
- At most one context executes domain code at any moment, because the driver serialises steps on
  one mutex and the platform has one hart.

The model gives POSIX thread *concurrency*, not *parallelism*.

## 2. Components and responsibilities

```
 Linux user space   capstone-exec: one Linux process per application
 +----------------------------------------------------------------------------------+
 |  main thread         context thread 1     context thread 2      spawner helper    |
 |  steps context 0     steps context 1      steps context 2       (posix_spawn)     |
 |  serves its calls    serves its calls     serves its calls                        |
 |  ...... park queue . signal state per context . one transport per context ....... |
 +--------+--------------------+--------------------+--------------------------------+
          | STEP ioctl         | STEP ioctl         | STEP ioctl      (libcapstone)
 Linux    v                    v                    v
 kernel  +----------------------------------------------------------------------------+
         | modcapstone, /dev/capstone: context ids (slot, generation), owner checks,  |
         | one mutex around every ioctl                                               |
         +--------------------------------------+-------------------------------------+
                                                | SBI ecall
 firmware +-------------------------------------v-------------------------------------+
 (M-mode) | capstone-sbi monitor: 32 context slots with generations, invocation       |
          | descriptors lent per call, offers, ADOPT / STEP / FORGET                  |
          +-------------------------------------+-------------------------------------+
                                                | supervised CALL
 capstone- +------------------------------------v-------------------------------------+
 qemu      | supervisor (CSUPERVISE): one protected continuation per slot, the 5 ms    |
           | quantum, removal of dead slots before their nodes are reused              |
           +------------------------------------+-------------------------------------+
                                                | first entry or resume
 domain    +------------------------------------v-------------------------------------+
 (C-mode)  | application + musl + runtime: contexts, each with stack, TLS and          |
           | struct pthread; runtime locks; park client; signal delivery; delegation  |
           +--------------------------------------------------------------------------+
```

| Layer | Knows | Does not know |
|---|---|---|
| Linux | when each launcher thread runs; what it waits for; where a signal goes; Linux tids | domain memory outside the regions the launcher was lent; any capability |
| Launcher (`runtime/linux/`) | which context each of its threads steps; each context's transport; the park queue; each context's signal state | the seals; domain memory outside its regions |
| Driver (modcapstone) | context ids, their owner (the open `/dev/capstone` file), application memory | threads, tids, scheduling |
| Monitor (capstone-sbi) | context slots, generations, descriptors, offers | what a context is used for |
| QEMU supervisor | one protected continuation per slot; the quantum | threads, owners |
| Domain runtime (`ports/musl-capstone/runtime/`) and musl | the contexts it minted, their areas, pthread state, lock words, thread identities | Linux's scheduling decisions; Linux tids |

Every layer below the launcher is thread-agnostic. "Thread" exists in exactly two places:
- Linux, which schedules the launcher's threads;
- the domain, which runs musl's pthreads on its contexts.

The layers between them move sealed contexts and nothing else.

## 3. Terms

- **Application.** One domain image and the memory the driver allocated for it; one per
  `capstone-exec` process. It owns the heap and the globals.
- **Context.** One executable state with its own continuation, stack and TLS block:
  - the *first context* is the one the monitor's `create_domain` builds;
  - every further context is *minted* by the domain runtime.
- **Thread area.** The block of application memory a minted context is built in, taken from the
  context arena with a parent handle the runtime keeps outside the area. Its layout:
  - the runtime's own contexts: seal region (1024 bytes), start block (256 bytes), TLS block and
    stack;
  - a musl thread: only the seal region and the start block (1280 bytes), because musl builds the
    stack and TLS in a mapping of its own.
  
  Revoking the parent handle ends every capability derived from the area.
- **Seal.** The sealed capability over the seal region: the context's entry authority.
- **Context id.** `(generation << 32) | slot`. `STEP`, `ADOPT` and `FORGET` name a context by both.
  Generations are never reissued and end at 0x7fffffff.
- **Invocation descriptor.** A 64-byte block the monitor lends a context for one call, across that
  call's preemptions: result word at 0, offer ticket at 16, offered seal at 32. Each application
  has 16.
- **Offer and ticket.** A context hands a minted seal to the monitor by moving it into its
  descriptor with a ticket, a counter of that context. The ticket names the offer; it carries no
  authority.
- **Transport.** A context's private channel to its launcher thread:
  - a 16 KiB META block (the request entry at 0, the signal handover block at 4096);
  - an exchange slice for buffers (256 KiB by default).
  
  Transport *i* sits at index *i* of two regions shared at launch. Transport 0 is the first
  context's.
- **Step.** One `STEP` ioctl: enter or resume a context, run it, return one event.
- **Round.** One delegated call: the domain writes a request into its transport and yields, and
  its launcher thread serves the request and steps it again.
- **Thread identity.** What the domain sees as a tid:
  - the first context's is the pid;
  - a minted context's comes from a counter from 0x400000 (2^22) to 0x3ffffffe, never reused
    within the process.
  
  Linux tids are not visible to the domain.
- **Park key.** A futex word's domain address, as an integer. The launcher never dereferences it.

## 4. Running a context

### 4.1 One step

```
 launcher thread      driver                  monitor                  supervisor / domain
 ---------------      ------                  -------                  -------------------
 STEP(id) ioctl ----> take the mutex
                      check the owner
                      ecall STEP -----------> a new call: lend a
                                              descriptor (after a
                                              preemption: keep the loan)
                                              supervised CALL -------> first entry, or resume
                                                                       of the saved continuation
                                                                       ...domain runs...
                                                                       yields a request, exits,
                                                                       faults, or the quantum
                                                                       expires
                                              <----------------------- event
                                              not a preemption:
                                              take an offered seal out,
                                              then end the loan
                      <--------------------- event, result, cause, pc
                      release the mutex
 <------------------- cond_resched()
```

The step is synchronous. From the ecall until the step returns, the hart runs the monitor and the
domain, and Linux does not run on it.

What the step returns:

| Event | Meaning | The launcher thread then |
|---|---|---|
| `RETURNED`, request in the entry | the context yielded a delegated call | serves the call on itself, then steps again |
| `RETURNED`, result `EXITED` | the context ended | leaves its loop (section 9.2) |
| `PREEMPTED` | the supervisor stopped the context and keeps its continuation: the quantum expired, the revocation nodes ran short, or an interrupt the seal enabled arrived | steps again at once |
| `FAULT` | a capability or other fault in the context | writes the fault record and ends the process (SIGSEGV) |
| `DEAD` | the context's seal is invalid (revoked) | leaves its loop |
| `STALE` | the id's generation is no longer registered | leaves its loop |
| `REFUSED` | the seal's saved privilege is not C-mode; it was not entered | leaves its loop |

For the first context, `DEAD`, `STALE` and `REFUSED` end the process: the application's first
context is gone. A STEP ioctl that fails for any reason but `EINTR` also ends the process (status
125):
- the driver has no record of the id (`EPERM`);
- the monitor could not reserve revocation nodes for a loan.

The loop of a context thread (`context_thread` in `runtime/linux/exec.c`), in outline:

```
attach to the context's signal state; tell the creator
loop:
    STEP(id)                        EINTR: the step never entered; try again
    PREEMPTED          -> continue
    FAULT              -> end the process
    DEAD/STALE/REFUSED -> break
    EXITED             -> break
    serve the request on this thread (delegate-service.c)
    exec in place, or exit, if the request asked for it
detach signals; FORGET(id); free the transport; wake the exit key once
```

The first context runs the same loop on the launcher's main thread.

### 4.2 A round

```
 domain (context n)                          launcher thread n
 ------------------                          -----------------
 read(fd, buf, len)
   pack the request into transport n
   (buffers -> exchange slice n)
   __capstone_yield() ---- step returns ---> validate the request against the shape table
                                             run read() itself, on this thread
                                             (may block here: only this context waits)
                                             copy the result into slice n
                                             publish pending signals for context n
   <----------------- STEP(id) resumes ----- step again
   take signals, copy data back
   deliver handlers (between rounds)
 return
```

The system call runs on the Linux thread that serves the context, outside any step and outside the
driver's mutex. That is what makes blocking per context (section 6). Each context has its own
transport, so the calls of two contexts never share an entry block or an exchange slice.

A round costs about 0.09 ms on capstone-qemu. One context making 100000 rounds took between
9.07 and 9.55 s across the thread steps' records, from T1 to the thread-name step. The signals per
context (B8) added 1 to 3 percent.

## 5. Scheduling: who decides what

### 5.1 Linux decides

Linux schedules the launcher's threads like any other threads:
- which one runs next;
- how long it waits;
- whether it sleeps in a system call or in the park queue;
- which thread a process-directed signal goes to.

A context's scheduling state is its Linux thread's:

| The context is... | Its Linux thread is... |
|---|---|
| running domain code | inside a STEP ioctl, holding the driver mutex |
| runnable | waiting for the mutex (asleep in the kernel), or on Linux's run queue |
| blocked in a delegated call | blocked in that system call (`wait4` is a 1 ms poll) |
| waiting on a futex | asleep in the park queue |
| between rounds | running launcher code |

Nothing below Linux makes a scheduling decision:
- the monitor runs whichever context it is asked to step;
- the supervisor only enforces the quantum.

### 5.2 Linux decides at step boundaries

Linux cannot preempt a context in the middle of a step, because the step is one synchronous ecall
(4.1). It regains the hart when the step returns, which happens at the latest when the quantum
expires:
- **The quantum.** 5 ms of QEMU's virtual clock (`QUANTUM_NS` in
  `target/riscv/capstone_supervisor.c`).
- **How it fires.** Every translated block that runs in C-mode checks the quantum flag at its start
  (`riscv_tr_tb_start` in `translate.c`). Once the timer has fired, the supervisor saves the
  continuation and returns `PREEMPTED`.
- **After the step.** The monitor returns to the driver, which releases its mutex and calls
  `cond_resched()`. The launcher thread steps again, and competes for the mutex like everyone else.

A context's domain code therefore runs for at most one quantum per step. The step itself can take
longer: before a loan the monitor collects revocation nodes when fewer than 1024 are free, and that
time is not the context's. Linux's switching granularity between domain contexts is the step, not
the kernel tick.

### 5.3 One step at a time

Two things make domain execution strictly serial:
- **The driver's mutex.** `device_ioctl` in modcapstone takes one mutex, `capstone_lock`, around
  every ioctl, `STEP` included, for the whole of the step. The name is unrelated to the domain
  runtime's `capstone_lock()` function.
- **One hart.** `capstone-vm` boots `-smp 1`, and the supervisor's collector refuses to run when a
  second CPU exists (`CPU_NEXT(first_cpu)` in `capstone_supervisor.c`).

At any moment at most one context executes domain code. Every other context is in Linux: waiting
for the mutex, serving or blocked in a call, asleep in the park queue, or runnable.

The mutex is taken interruptibly. A signal to a thread waiting for it ends the ioctl with `EINTR`
before the domain is entered, and the launcher steps again after the signal has been recorded.

### 5.4 A timeline

An illustrative schedule, not to scale. Context A computes for a while and then writes to a pipe.
Context B reads from that pipe while it is still empty.

```
 domain code on the hart:  B | A ....... | A ....... | A | B
 thread B:  step B; serve read(): blocked in Linux ....... | data in; step B
 thread A:  waits | step A    | step A    | step A; serve write()
 steps end: B yields read; A preempted; A preempted; A yields write; B yields again
```

1. Thread B steps B. B yields a `read`, and thread B calls `read()` on the empty pipe. It blocks in
   Linux, holding nothing.
2. Thread A steps A. A computes; each step ends at the quantum, and thread A steps again.
3. A yields a `write`. Thread A calls `write()` on the pipe, and Linux wakes thread B.
4. Thread B copies the data into B's exchange slice and steps B, and B's `read` returns.

B's read blocks only B's Linux thread. A computes meanwhile, one quantum per step. Probe case B7 is
this schedule: the reader stays blocked while the other context computes for 100 ms, until that
context writes.

### 5.5 What the model gives, and what it does not

- **Concurrency.** A blocked context never stops the others: blocking happens in its Linux thread,
  outside the step.
- **Progress for computing contexts.** A context that never makes a call still yields the hart
  after each quantum. Probe A3 preempts a context across quanta and checks its register state.
- **No parallelism.** Two contexts never execute domain code at the same time, on this platform or
  under this driver.
- **Fairness and latency.** These are whatever Linux's scheduler and the kernel mutex give, and no
  bound is claimed:
  - the kernel mutex is not FIFO;
  - a preempted thread asks for the next step at once;
  - a step can include the monitor's node collection.
  
  Nothing beyond the probes' schedules is measured.
- **Scheduling calls:**
  - `sched_yield` is delegated and runs on the serving thread: it yields that Linux thread.
  - `nanosleep` and `clock_nanosleep` sleep the serving thread.
  - `sched_getaffinity` and `sched_setaffinity` act on the calling thread only. The runtime turns
    pid 0 or the caller's own tid into 0 and answers `ESRCH` for another thread's identity
    without a round; the launcher refuses any pid but 0 with `EPERM`.
  - `getpriority` and `setpriority` are delegated for `PRIO_PROCESS` and the task itself (0 or
    its pid), and so are `sched_get_priority_max`, `sched_get_priority_min` and
    `sched_rr_get_interval` (the plain rows, `runtime/applications.md`). They run on the serving
    thread, and Linux keeps a nice value per thread: 0 names the calling context's serving
    thread, the pid the first context's. That a nice value set from one context changes only its
    own thread is Linux's rule and was not measured here.
  - Policy calls (`sched_setscheduler`, `sched_setparam`) are not delegated. They answer `ENOSYS`
    and appear in the unserved report.
  - Every context starts on a launcher thread of the same priority, and no priority crosses a
    lock. The runtime's priority-inheritance mutexes therefore inherit nothing (6.2).
- **Thread names.** `pthread_setname_np` names the serving Linux thread, so the name shows in
  `/proc/<pid>/task/*/comm`. Naming another minted thread is refused: its tid names no Linux task
  (`ENOENT`). Naming the first context from another thread (its tid is the pid) is untested.

## 6. Blocking and waking

### 6.1 Delegated calls block their own context only

A call that blocks in Linux blocks the Linux thread that serves it. It holds no lock the other
contexts need:
- the driver mutex was released when the step returned;
- the launcher's shared locks are taken only around short bookkeeping (a spawn, the child list,
  the dispositions), and `wait4` and a park wait release theirs before they sleep.

Every other context keeps being stepped. This is why each context has a transport of its own, and
why a thread is not multiplexed onto another context's Linux thread.

### 6.2 Futexes through the park queue

Linux cannot see domain memory, so a domain lock word cannot be given to `futex`. The domain keeps
the authoritative lock state, as native pthreads do. The launcher's park queue
(`runtime/linux/park.c`) only puts launcher threads to sleep and wakes them, by key.

The shared state:
- a table of 256 generation words, one per bucket, in the META region after the last transport;
- only the launcher writes a generation word, under its park mutex, with release semantics;
- the domain reads it with acquire semantics.

```
 domain: futex(word, WAIT, val, timeout)        launcher thread: PARK_WAIT(key, gen, deadline)
   gen = load_acquire(table[bucket(word)])        lock park mutex
   if *word != val: return EAGAIN                 if bucket saturated or table[b] != gen:
   request PARK_WAIT(key, gen, deadline) ------>      unlock; return RECHECK
                                                  queue this thread's record (full key)
                                                  unlock
                                                  sleep on the record's own word
                                                  (FUTEX_WAIT_BITSET, absolute deadline)
                                                  lock; completed as WOKEN, or dequeue on
                                                  timeout or signal; unlock
   <-------------------------------------------- WOKEN / RECHECK / ETIMEDOUT / EINTR
   WOKEN or RECHECK: return 0 (the caller re-checks, as after any wakeup)

 domain: futex(word, WAKE, n)                   launcher thread: PARK_WAKE(key, n)
   (the lock state already changed)               lock; advance table[b], release;
   request PARK_WAKE(key, n) ------------------>  complete up to n records with this key as
                                                  WOKEN; unlock; return how many
```

- **No lost wake.** A WAKE between the domain's compare and its WAIT has advanced the bucket's
  generation, so the WAIT answers RECHECK instead of sleeping. A WAKE between enqueue and sleep has
  already set the record's word, so the sleep does not start. A generation saturates at
  `UINT64_MAX` instead of wrapping; a saturated bucket answers every WAIT with RECHECK.
- **Keys.** Collisions only make a WAIT return early, since queues keep full keys and WAKE selects
  by key.
- **Served operations:**
  - `FUTEX_WAIT`, `FUTEX_WAKE` and `FUTEX_REQUEUE`, private or not, with Linux's argument checks.
    `REQUEUE` advances the source generation, so a waiter that checked its condition before the
    requeue does not sleep through it.
  - `FUTEX_LOCK_PI` and `FUTEX_UNLOCK_PI` are served in the domain over the same park calls. The
    word holds the owner's thread identity with `FUTEX_WAITERS` and `FUTEX_OWNER_DIED`, as on
    Linux. An unlock frees the word and wakes one waiter to take it, and no priority is inherited.
  - Every other operation answers `ENOSYS` and is reported as unserved, among them `WAKE_OP`,
    `CMP_REQUEUE`, `WAIT_BITSET`, `WAKE_BITSET`, `TRYLOCK_PI`, `LOCK_PI2`, the requeue-PI pair, and
    any operation with the realtime-clock flag.
- **Timeouts.** A relative timeout becomes one absolute `CLOCK_MONOTONIC` deadline at entry and
  survives the signal restarts within the call. A WAIT that ends in WOKEN or RECHECK returns 0, like
  a spurious wakeup; `FUTEX_LOCK_PI` keeps waiting with the same deadline. The domain's clock is
  the launch record's, extrapolated with `rdtime`, and the launcher sleeps on the kernel's. The
  deadline can be off by the difference between the two, a named deviation.
- **Signals.** A signal to a thread parked in `FUTEX_WAIT` behaves as Linux's futex call does:
  - untimed under `SA_RESTART`: the handler runs, and the wait goes on to its wake;
  - timed, or without `SA_RESTART`: `EINTR`.
  
  `FUTEX_LOCK_PI` goes on waiting after a handler, as Linux restarts that call.

### 6.3 Waiting inside the runtime

The runtime has two kinds of internal lock (`capstone/lock.h`):
- **`capstone_lock`.** musl's `__lock`, a futex lock that parks through 6.2. It costs nothing while
  the application has one context.
- **`capstone_spin_lock`.** A leaf lock on a scalar word; a waiter spins. On one hart a spinner
  spins until its quantum ends, and the holder releases the lock when Linux next steps it.

## 7. State and synchronisation in the domain

| Per context | Process-wide |
|---|---|
| continuation, stack, TLS block, `struct pthread`, thread identity | globals and heap |
| transport: entry block, exchange slice, the `dl_*` delegation state (`__thread`) | park generation table |
| signal mask, event ring, handover block, alternate stack | signal dispositions (the `sigaction` table) |
| `errno`, locale, cancellation state (behind `tp`) | capability-global initialisation, run once by the first context |

- **TLS.** C11 thread-locals in musl's own layout, one layout for every context. `tp` is part of
  the suspended computation, so `__capstone_yield` saves it with the callee-saved registers.
- **Atomics.** Every application builds with the A extension, so scalar atomics are the
  hardware's (AMOs and LR/SC). A supervised switch drops the LR reservation, so an LR/SC loop
  preempted in the middle retries.
  Capability-width atomics run under a spin lock: pointer tags and bounds survive, and they are not
  lock-free.
- **Runtime locks.** A lock covers each of:
  - the heap (level0 and the Sublet heap);
  - the mmap and shared-memory tables;
  - the context arena;
  - the static `posix_spawn`/`execve` request block;
  - the signal handler table.
  
  The order is maps, then heap, then atomics; the spawn and arena locks take no other lock. The
  lock over musl threads' records (`clones_lock`) comes before the arena, map and heap locks.
- **No handler under a runtime lock.** Each context counts the `capstone_lock`s it holds. While the
  count is not zero, no domain signal handler runs in that context. What arrived meanwhile runs at
  the last release. A spin lock makes no round while held, so no handler can run under it. Reaping
  runs only in `__clone`, under the records' lock.
- **musl's own locks.** They switch on at the first thread, as musl's first `pthread_create` does:
  FILE locks leave -1, and `libc.threaded` and `need_locks` are set. musl switches `need_locks` off
  again when its thread count falls back to one, as it does on Linux. A thread ending in
  `SYS_exit` therefore takes no musl lock (9.2).
- **Memory order.** The argument is for one hart and coherent shared memory. The compiler barrier
  sits at the domain's side of every transition (the delegation stub is `asm volatile` with a
  `memory` clobber). The SMP argument is deferred.

## 8. Signals

The runtime keeps Linux's division, with one context per thread:
- dispositions are per process;
- mask, pending signals and the alternate stack are per thread.

- **Launcher.**
  - One disposition table serves the process.
  - Each context has its own ring, in-flight set and logical mask, in the state of the launcher
    thread that serves it, and its own handover block, in its transport's META block.
  - Each context thread's kernel mask is its context's mask plus the signals in flight or held
    back, so Linux itself picks the thread for a process-directed signal.
  - A new context starts with its creator's mask. Its Linux thread starts with every signal blocked
    (glibc unblocks its own two in each new thread) and attaches to its context's state before the
    first step. A signal that reaches it before then is kept, blocked, and moved into the
    context's ring at attach.
- **Delivery is synchronous.** A context takes its signals at its next round, after the call's data
  is back:
  - a context that computes without calls takes them later;
  - one that never makes a call never takes them.
  - Asynchronous delivery (the "bell": entering a running context) is not built.
- **Directed signals.** `tkill`, `raise` and `pthread_kill` name a thread identity. The launcher
  sends `tgkill` to the Linux thread that serves that context, under the lock a thread takes to
  leave the table, so the Linux tid is never one Linux has given to another thread. An identity
  that no context holds answers `ESRCH`.
- **Cancellation.** Cancellation is deferred, through musl's cancellation points: a delegated call
  is rounds, not instructions. `__syscall_cp_asm` marks the context as inside a cancellation point,
  and a cancelled call ends with `EINTR` without being issued again. Asynchronous cancellation of a
  thread that makes no calls does not happen.
- **`__synccall`.** Behind `setuid` and its relatives, it runs its function once, in the caller.
  None of those calls is served (`ENOSYS`), and a served one would have to act on each Linux thread.

## 9. Creation, end and reclamation

### 9.1 Creating a thread

```
 domain (creator)                   launcher (creator's thread)       driver / monitor
 ----------------                   ---------------------------       ----------------
 pthread_create (musl): stack, TLS, struct pthread in its own mapping
 __clone:
   take an area (reuse a reaped one, or the arena)
   mint: split seal region and start block; write sp, tp, entry, argument;
         seal (pc = the context entry label, control words as create_domain's)
   CONTEXT_RESERVE ---------------> a free transport index
                                    (waits for one that is ending)
   write the index into the start block
   offer: move the seal and a ticket into the descriptor this call was lent
   CONTEXT_CREATE(ticket, THREAD,
                  transport, tid) -> ADOPT(creator's id, ticket) ----> take the offered seal
                                                                       into a free slot of the
                                                                       same application
                                    <-------------------------------- new id (slot, generation)
                                    make a Linux thread, every signal blocked;
                                    wait until it attached to the context's signals
   <------------------------------- new id
 return the tid                     new thread: STEP(new id) ---------> first entry: the entry
                                                                       glue, then
                                                                       __capstone_context_run:
                                                                       install the transport,
                                                                       call musl's start
```

- **The seal stays out of Linux's reach.** It passes only through the descriptor the monitor lent
  to the creator, never through memory Linux can write, and the launcher never sees it. ADOPT
  moves it, so it is registered at most once.
- **Failures take everything back.** If ADOPT fails, the domain revokes the area (by the code;
  no probe forces it). If the Linux thread cannot be made, the launcher calls FORGET on the new id
  (probe A11). No context runs after a reported failure.
- **The transport comes first.** It is reserved before the request, so the child can start
  (whenever Linux runs its thread) before the creator has its answer.

### 9.2 Ending a thread

A musl thread ends in `SYS_exit` (`__capstone_thread_exit`, then the non-returning assembly
`__capstone_context_exit_clear`). This path takes no lock, only atomics. In order:

0. If this is the last musl thread and the first context has already ended, the application ends
   now with the first context's status. Otherwise the context gives up its signal events.
1. `CONTEXT_EXITING(key)`: the launcher learns that the context is ending and which word to wake.
2. The clear word (musl's thread-list lock, `CLONE_CHILD_CLEARTID`) becomes 0. From here a joiner
   may free musl's mapping, stack and TLS included.
3. The completion word becomes 1. From here the area may be revoked. Everything the final switch
   needs is already in registers: nothing touches the stack, TLS or start block after this store,
   and no signal is delivered.
4. The context returns `EXITED`. The launcher thread leaves its loop, forgets the context, frees
   the transport, and wakes the key once, as Linux does after clearing the word.

Any later entry into an exited context lands on a re-entry label that answers `EXITED` again,
without stack, TLS or start block.

### 9.3 Join, reaping and the first context

- **Join.** musl's own `pthread_join` waits for the thread's exit and then for the thread-list
  lock, which the ending thread's clear word releases (9.2, step 2).
- **Reaping.** One record per area, outside every area, holds the area's handle. At every `__clone`
  the reaper revokes each area whose completion word is 1 and keeps it for the next thread. It also
  returns a detached thread's mapping to the heap. A finished, unreaped thread costs one area and
  at most one mapping until the next thread is made.
- **The first context.** Its stack is the monitor's and its TLS the runtime's first allocation;
  neither is ever freed.
  - `pthread_exit` in the main thread clears and wakes its word and then waits for good while
    another thread lives.
  - The last thread's `pthread_exit` finds itself alone and calls `exit(0)`, as on Linux.
- **Process end.** `exit`, `exit_group` or a fault in any context ends the whole process.

### 9.4 Stale identities and revocation

- **Generations.** A context id carries a generation, and a stale id never reaches a replacement:
  - the driver refuses an id it holds no record of (`STEP`: `EPERM`; `FORGET`: `ESTALE`);
  - the monitor answers `STALE` for a generation it has retired.
  
  A revoked seal steps `DEAD`.
- **Retirement.** Dead registrations are retired on shortage, so Linux never has to FORGET.
- **Capability ABA.** The supervisor's collector removes a dead slot before its revocation node can
  be reused. A new seal with the old node and bounds therefore enters its own entry, never the old
  continuation (probe A7).
- **Why one revoke suffices.** On one hart, revoking a context's area both prevents its resumption
  and allows reclamation. Every switch writes the seal region, and after the revoke no switch into
  the context can happen. No switch out of it can be in progress at the same time. On SMP this is
  not true, and it belongs to the later contract.

### 9.5 Two ways to run a minted context

- **`THREAD`** (every musl thread): the launcher makes a Linux thread that steps the context until
  it ends. A `CONTEXT_STEP` from the application naming such a context is refused (`EINVAL`), so its
  own thread is its only stepper. One window is not closed: between ADOPT and the thread's
  registration in the launcher, a `CONTEXT_STEP` naming the new id would pass the check. Only a
  context of the same application can send it, and nothing in the runtime does.
- **`REGISTER`** (the runtime's context interface, used by the probes): the context is only
  registered. The application steps it with `CONTEXT_STEP` and gets the event back. That step runs
  on the requesting context's Linux thread, and the registered context has no transport of its own.

## 10. Authority: what Linux can and cannot do

Linux decides when and whether a context runs, and its answers may be wrong. Linux cannot:
- create capability authority or inject a context: a seal reaches the monitor only through a
  descriptor lent to a context of the same application;
- resume a context whose seal the domain has revoked (`DEAD`);
- reach a context's older generation through a stale id (`STALE`).

Applications are kept apart from each other by the driver, which is part of Linux: a launcher can
step, adopt from or forget only its own contexts. The owner check is in the driver, while the
monitor checks slot, generation and ticket only. This separates launcher processes; it is no
protection against the kernel itself. Probe A13 shows it for `STEP` and `FORGET`.

A context cannot:
- read the monitor's saved state through its return capability: a sealed-return capability
  reaches only the caller's general-purpose slots (probe A2);
- leave C-mode: `mret` and `sret` fault under supervision, and a seal with another privilege is
  `REFUSED` (probe A12).

The domain never frees memory on the strength of a Linux answer, only after revoking a handle it
holds.

What the model does not claim:
- **Isolation between threads.** Contexts of one application trust each other. A bounded capability
  stops an access outside its bounds; it does not separate a context from a sibling given
  overlapping authority.
- **Rust's `Send`.** Moving a linear capability between contexts transfers exclusive ownership of
  that memory. It proves no type invariant and no correct synchronisation.
- **Permission enforcement on data accesses.** capstone-qemu checks a data access's tag,
  revocation and bounds, not its permission bits (ISSUES Q-14). The descriptor loan's write-only
  permission is therefore not enforced there.

## 11. Limits

| Resource | Limit | At the limit |
|---|---|---|
| threads per application besides the first | at most 15 (`CONTEXTS`; 16 descriptors per application). The SDK build declares 15; `capstone_add_application` defaults to 0 | `pthread_create` answers `EAGAIN`; a reservation first waits for a context that is ending |
| context slots, all applications together | 32 monitor slots, 32 supervisor continuations per hart | dead registrations are retired; then ADOPT answers `ENOSPC`, and `pthread_create` fails |
| thread identities per process | 0x400000 to 0x3ffffffe, never reused | minting fails, and `pthread_create` fails |
| generations per slot | up to 0x7fffffff | the slot retires; it is never reused |
| context arena (`CONTEXT_BYTES`) | 64 KiB in the SDK build, 0 by default; a musl thread takes 1280 bytes of it, kept for the next thread after reaping | minting fails, and `pthread_create` fails |
| exchange slice per transport | 256 KiB by default (`EXCHANGE_BYTES`) | bounds what one round of that context carries, as without threads |
| park buckets | 256 | collisions only cost early returns |

Not supported, and why:
- **Parallelism and SMP.** One hart; the supervisor's collector refuses a second CPU. SMP needs its
  own contract: continuations moving between harts, revocation across harts, and the memory model.
- **`fork`.** `ENOSYS` by design: a domain cannot be duplicated by Linux. `posix_spawn` and exec in
  place work from any context through the launcher.
- **Asynchronous signal delivery and asynchronous cancellation.** Both need the bell (asynchronous
  entry into a running context), which is not built. libc-test's `pthread_cancel` waits for it.
- **Futex operations.** Beyond those listed in 6.2: `ENOSYS`, visible in the unserved report.
- **`sem_open`.** It needs a file mapped `MAP_SHARED`, and a domain maps no files.
- **Priority scheduling.** Policy calls are not delegated, a nice value is per serving thread
  (5.5), and PI mutexes inherit nothing.
- **Another thread's affinity or name.** `ESRCH` from the runtime (`EPERM` from the launcher for
  a raw pid) and `ENOENT`.
- **The platform.** The supervisor extension is "a VM platform extension, not an implementation
  claim about the existing FPGA instruction set" (`capstone_supervisor.c`). The quantum, the
  continuations and the dead-slot collection are capstone-qemu's.

## 12. Why this design

- **Linux keeps what Linux already does well.** Scheduling, blocking, timeouts and signal routing
  are Linux's, through ordinary threads. The domain gets Linux's semantics for them, not an
  imitation. There is no scheduler in the monitor, the supervisor or the domain.
- **The trusted layers stay small and thread-agnostic.** The monitor and the supervisor manage
  sealed contexts, generations and loans. Nothing in them depends on how the domain uses contexts,
  so the same rules serve threads, probes and any later use.
- **Authority never passes through Linux.** Seals move from the domain to the monitor through a
  lent descriptor. Tickets and ids name things; they grant nothing.
- **One Linux thread per context.** The alternative, several contexts stepped by one Linux thread,
  would block every context behind one blocking call, or need a second scheduler to avoid that.
- **One version of each ABI.** The process ABI changed in place, with the driver and the monitor
  pinned together, and no feature probe.

## 13. Evidence

Most probe cases have a control, a run that fails without the mechanism; the plan and the records
say which, and some cases (A1, A3, B7) have none. The records are under
`runtime/tests/application/results/` unless noted.

| Property | Probe | Record |
|---|---|---|
| a minted context runs with its own stack and TLS; initializers run once | A1 | `20260929-context-probe-monitor.json` |
| authority at entry; the sealed-return window | A2 | `20260929-context-probe-monitor.json`, `20260930-sealed-return-wrap.json` |
| preemption keeps a context's state | A3 | `20260929-context-probe-monitor.json` |
| exit, re-entry after exit, join and reuse | A4, A5, A6 | `20260929-context-probe-monitor.json` |
| capability ABA prevented by the collector | A7 | `20260929-context-probe-monitor.json` |
| generations never wrap; stale ids | A8, A9 | `20260929-context-probe-monitor.json` |
| dead registrations retired | A10 | `20260929-context-probe-monitor.json` |
| creation rolled back when the Linux thread cannot be made | A11 | `20260929-context-probe-monitor.json` |
| C-mode only | A12 | `20260929-context-probe-monitor.json` |
| one registration; owner checks for `STEP` and `FORGET` | A13 | `20260929-context-probe-monitor.json` |
| one stepper per `THREAD` context (the refusal) | A13 | `20260930-transport.json` |
| a blocked context leaves the others running | B7 | `20260930-transport.json` |
| park queue, no lost wake, requeue, signals | B1-B6, B12, B13 | `20260930-park-native.json`, `20260930-park-domain.json` |
| runtime locks, atomics under preemption, no handler under a lock | B9, B14 | `20260930-runtime-locks.json` |
| musl threads: create, join, detach, exit, reaping | B10, B11 | `20260930-pthreads.json` |
| signals per context, cancellation | B8 | `20260930-signals-per-context.json`, `20260930-kill-thread-review.json` |
| fifteen threads per application | T5 | `20260930-contexts-per-application.json` |
| thread names and CPU sets | | `20260930-thread-names-cpu-sets.json` |
| GLib's thread tests, and those that fail | gate | `20260930-glib-threads.json` |
| CPython's threading tests | gate | `ports/cpython/app/results/thread-review-2026-09-30.json` |

## 14. Code map

| Part | Where |
|---|---|
| context threads, the step loop, context requests | `runtime/linux/exec.c` |
| serving a round, the park requests, the seccomp filter | `runtime/linux/delegate-service.c` |
| the park queue | `runtime/linux/park.c`, `park.h` |
| signal state per context, the trampoline | `runtime/linux/signals.c` |
| wire ABI: requests, transports, park table | `runtime/include/capstone/delegate.h`, `runtime/common/delegate.c` |
| mint, `__clone`, thread end, reaping | `ports/musl-capstone/runtime/context.c` |
| entry glue, exit assembly, seal, offer, yield | `ports/musl-capstone/runtime/start-musl.S` |
| rounds, futexes, the transport per context | `ports/musl-capstone/runtime/delegate.c` |
| runtime locks | `ports/musl-capstone/runtime/lock.c`, `runtime/include/capstone/lock.h` |
| signal delivery and cancellation in the domain | `ports/musl-capstone/runtime/signals.c` |
| driver: ids, owner checks, ADOPT, FORGET, the mutex | caplifive-buildroot `package/modcapstone/module/process.c`, `capstone.c` |
| monitor: slots, loans, offers, STEP, retirement | capstone-sbi `sbi_capstone.c` (`context_step`, `loan_begin`, `loan_end`, `context_adopt`, `context_retire`) |
| process ABI shared by driver and monitor | `process-abi.h` (identical copies) |
| supervisor: continuations, quantum, collector | capstone-qemu `target/riscv/capstone_supervisor.c`, `translate.c` |
| probes | `runtime/tests/application/context-probe.c`, `thread-probe.c`, `pthread-probe.c`, `pthread-kill-probe.c` |
