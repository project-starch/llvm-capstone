# Delegated threads: one Linux thread per protected context

Status: PLAN, 2026-09-29. Branch `delegation-threads`, stacked on `delegation-signals` (90eab46).
Nothing here is implemented. The contracts below are what Probe A and Probe B test; the runtime
branch that builds `pthread_create` on them is written after both probes pass.

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
  a separate operation from adoption); futex
  requeue, PI and robust mutexes; asynchronous cancellation; `fork` in a threaded process (already
  ENOSYS); sockets.

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
  - capability-width atomics are lock-free library calls that the file itself calls "WRONG for any
    build that can run two threads" (`atomic_libcalls.c`, C-54);
  - `level0.c` and `sublet_heap.c` take no lock.

## Context lifecycle

`MINTED → ADOPTED → (RUNNING ⇄ PAUSED)* → EXITED → REVOKED → REMOVED`

1. **Mint (domain).** The runtime:
   - takes a thread area from application memory with `sublet_take_linear` and keeps the parent
     handle;
   - splits the area into seal region, stack and TLS, and writes the TLS image and `struct pthread`;
   - builds a start block (stack, tp, gp, start function, argument, completion record);
   - writes the seal region: pc = the thread entry label, `ctvec` = the runtime's trap vector,
     `cscratch` = the start block, control words exactly as `create_domain` writes them;
   - seals it.
2. **Hand-off (domain to monitor).** On every CALL the monitor lends each context a 16-byte
   capability slot, next to the result slot it already lends (`supervised_call`). The runtime moves
   the seal into that slot and makes the delegated request `CONTEXT_CREATE`, whose argument is only
   a ticket. The seal never passes through memory that Linux can write, and the launcher never sees
   it.
3. **Adopt (launcher, driver, monitor).** `ADOPT(parent (k, g))` moves the seal out of the parent's
   hand-off slot into a free context slot and returns `(k', g')`.
   - The new slot's owner is the parent's owner. Only code running in a context of the same
     application can write the parent's hand-off slot.
   - The launcher then creates the Linux thread. If that fails, it calls `FORGET(k', g')` and
     answers the request with an error; the runtime revokes the area.
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
  properties come from the ISA and the supervisor, not from new monitor code; case A13 shows them.
- **A slot whose seal is invalid is dead.** Arm and resume answer the status `DEAD` instead of
  raising, and the monitor returns it to the driver as a STEP result.
- **Dead slots are removed before their node can be reused.** The collector removes every slot
  whose seal node is invalid, together with its saved continuation and any `armed` reference, before
  it releases nodes. It also runs when a slot is requested and none is free, so dead slots cannot
  exhaust the 32 even if Linux never forgets them.
- **FORGET is idempotent and cannot be redirected.**
  - A tagged seal that is invalid still names exactly its old slot, because its node is not reused
    before the collector has untagged every copy. FORGET removes that slot.
  - An untagged value names nothing, and FORGET returns without effect.
  - No slot is ever found from bits.
- **Integer ids carry a generation.** `STEP`, `FORGET` and `ADOPT` take `(k, g)`; a stale `g` is
  refused with `STALE`. This protects the launcher from its own late requests. It is not a security
  boundary, since Linux may end its own contexts at will. The owner check stays as it is.
- **Application memory is not context memory.** Contexts have no memory of their own in the
  monitor's or the driver's books. Application memory is reclaimed only when the application is
  destroyed, after every context slot of it has been removed. Destroying one context never reclaims
  memory.
- **Control words (open point).** A minted seal carries its own `mstatus` and privilege,
  `mideleg`, `medeleg`, `mip` and `mie`, and the monitor cannot read a sealed region to check them.
  Any domain can already seal and call such a region itself, so this is not new authority. What is
  new is that the monitor enters it under supervision. Case A12 records what a first supervised entry
  does with values that differ from `create_domain`'s; the rule is written from that record.

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
on a list, and a later `pthread_create` or join reaps them.

**Why one revoke is enough on one hart.** The ISA switch writes the seal region on every exit, and
the supervisor writes it on every resume. After the revoke, no switch into the context can happen:
the monitor's copy of the seal is invalid, and the supervisor answers `DEAD`. On one hart no switch
out of the context can be in progress at the same time. "Prevent resumption, then reclaim" is
therefore one step. On SMP it is not, and that belongs to the later contract.

**Linux thread end** matters only for the launcher's own state. The host stack, park-queue entries
and per-context transport mappings are freed after the Linux thread has ended and its last STEP has
returned.

## Parking

Linux cannot see domain memory, so the lock word cannot be handed to `futex` itself. The domain
keeps the authoritative lock state, as native pthreads do. Linux only puts threads to sleep and
wakes them.

**Shared state.** A table of generation words lives in a region shared by every context and the
launcher: the application's process-wide META region, not the per-context blocks.
- Each generation word is 64 bits and aligned. Only the launcher writes it, under its park mutex,
  with release semantics; the domain reads it with acquire semantics.
- The key is the domain address of the lock word, as an integer. The launcher never dereferences
  it; the bucket is a hash of the key.
- Generations are compared for equality only, so the wrap from 2^64−1 to 0 is an ordinary
  increment.

**Launcher state.**
- one park mutex;
- for each launcher thread, one wait record `{key, notified, state}`; `notified` is that thread's
  private futex word;
- per bucket, a queue of wait records that holds the full key.

```
domain WAIT(key):                        domain WAKE(key, n):
  g = load_acquire(gen[bucket(key)])       change the lock state (atomic RMW)
  re-check the lock state; done if free    request WAKE(key, n)
  request WAIT(key, g)
  on return: re-check (loop)

launcher WAIT(key, g):                   launcher WAKE(key, n):
  lock                                     lock
    if gen[b] != g: unlock, return           store_release(gen[b], gen[b] + 1)
    record = {key, notified = 0}             take up to n records with this key
    enqueue(b, record)                       for each: notified = 1; futex_wake(&notified)
  unlock                                   unlock
  futex_wait(&notified, 0) until notified or abort
  lock
    if notified: result WOKEN
    else: dequeue; result TIMEOUT or EINTR
  unlock
```

- **A WAKE between enqueue and sleep is not lost.** `notified` is already set, and `futex_wait`'s
  value check refuses to sleep.
- **Notification wins.** Under the mutex, a notified record becomes WOKEN; only a record without a
  notification is aborted and dequeued.
  - This is Linux's own order: a waiter already dequeued by a waker returns success before timeout
    or pending signals are considered (`kernel/futex/waitwake.c`).
  - The pending signal stays deliverable.
  - The guarantee covers the completed wait and its cleaned-up record. What a handler does
    afterwards, a `longjmp` for example, is the business of the libc operation.
- **Every wait completes exactly once.** A record is reused only after the waker is finished with
  it; in the first version the waker calls `futex_wake` under the mutex.
- **Collisions** only make a WAIT return early, because the queue keeps full keys and WAKE selects
  by key.

**Relation to signals.** WAIT and WAKE are compound operations in the sense of
`delegation-signals.md`, with their own continuation points:
- a WAIT's record is completed before a domain handler runs or a new WAIT starts;
- a WAKE that has published notifications is reported as done, never as `RETRY`.

**Memory order in the domain.** The generation load stays before the re-check, and the change of
the lock state stays before the WAKE request.
- The compiler barrier sits at the domain-side transition: the delegation stub is `asm volatile`
  with a `memory` clobber. A barrier in the launcher's native stub cannot order the domain's
  accesses.
- The hardware order is argued separately. On one hart it is program order plus the trap at the
  transition. The SMP argument is deferred.

## Runtime state: per context and process-wide

| Per context | Process-wide |
|---|---|
| sealed-return capability (today `__capstone_dom_ret`), recovery block behind `cscratch` | heap, and its lock |
| delegation transport: wire entry, exchange region, `dl_*` state (becomes `__thread`) | generation table |
| signal mask, receive ring, handover block | signal dispositions (the `sigaction` table) |
| errno and `struct pthread` (already reached through tp) | capability-global initialization, once |

Before a second context may exist:
- `atomic_libcalls.c` takes a lock;
- the allocators take a lock once musl's `need_locks` is set;
- every other process-wide structure the runtime owns gets a lock.

Preemption on one hart is enough to break an unlocked read-modify-write.

## Probe A: context and lifetime

A domain program with one mode per case, driven from the host like `signal-contract.dom`. Each case
names the state it starts from, the point it forces, and the condition it checks.

Points are forced deterministically:
- a register-only delay loop longer than one quantum on the exit path (the supervisor's preemption
  counter confirms the preemption landed there);
- an explicit call to the supervisor's collector;
- slot or node exhaustion by construction.

| Case | Start | Forced point | Postcondition |
|---|---|---|---|
| A1 mint and enter | main context running | adopt one minted seal | the second context runs its function with its own sp and tp; both see one global counter; the initializers ran once |
| A2 authority at entry | minted seal | first instruction of the thread entry | no register holds a tag except the delivered `ra`, the result and hand-off slots, and what the start block provides |
| A3 preemption | second context in a long loop | at least two quanta | resumes with its register state intact (checksum) |
| A4 re-entry after exit | second context exited | STEP again, after its stack and TLS children were revoked | `EXITED` every time, no fault |
| A5 join and reuse | exited, `done` set | joiner copies, revokes, reallocates the area, writes a pattern | pattern intact; the joined result is the one the thread returned |
| A6 preempted after `done` | exit path between release store and return | joiner revokes and reallocates; the launcher STEPs the old context | `DEAD`; pattern intact |
| A7 capability ABA | context A paused | revoke A's area, run the collector, re-mint in the same area until the new seal has A's node and bounds | STEP of the new context enters its own thread entry, not A's continuation |
| A8 late STEP and FORGET | context revoked, not yet collected | STEP, FORGET, FORGET | `DEAD`, removed, then idempotent; the monitor survives and a live context still runs |
| A9 id ABA | slot k reused with a new generation | late `FORGET(k, old g)` | `STALE`; the new context unaffected |
| A10 slot exhaustion | Linux never forgets | four times as many create/exit/revoke cycles as slots | every create succeeds; a live context keeps running |
| A11 adoption rollback | adopt done | Linux thread creation fails (launcher test switch) | slot removed, area revocable, nothing leaked |
| A12 control words | minted seal with a different privilege and `mie = 0` | first supervised entry | behaviour recorded; the control-word rule is written from it |
| A13 single registration and entry | one context adopted | ADOPT again from the same hand-off slot; two launcher threads STEP the same `(k, g)` | the second ADOPT is refused (slot empty); the STEPs serialise, and the context's step counter shows one continuous execution |

## Probe B: parking

First as a native program with ordinary Linux threads, using test hooks in the park code that stop
a thread at a named point. Then in a domain, where it adds the memory-order, blocking, signal and
atomics cases. B7 to B9 need Probe A.

| Case | Start | Forced point | Postcondition |
|---|---|---|---|
| B1 wake before sleep | waiter enqueued | waiter stopped after unlock, before `futex_wait`; waker runs | WOKEN without sleeping |
| B2 collision | waiters on keys A and B in one bucket (table of one bucket), both asleep | WAKE(A, 1) | the A waiter wakes; the B waiter stays asleep |
| B3 abort against selection | waiter selected and notified | a timeout or a signal reaches it before it retakes the mutex | WOKEN; the signal is still pending and then delivered |
| B4 record reuse | waiter woken | the same thread waits again while the waker still holds the mutex | no stray and no lost wake (counts) |
| B5 wraparound | generation preset to 2^64 − 2 | WAIT and WAKE across the wrap | no lost wake, no false sleep |
| B6 domain order | two contexts on one lock | preemption between generation load and re-check, and between state change and WAKE | no lost wake; counter exact |
| B7 blocking | context 1 in a delegated `read` on an empty pipe | context 2 runs and writes | context 2 runs meanwhile; the read returns its data |
| B8 signal during WAIT | context waiting | signal to its Linux thread | the wait completes once, the handler runs after it, WAKE is never `RETRY` |
| B9 atomics under preemption | two contexts | preemption inside 8-byte LR/SC loops and capability-width atomics | final counters exact |

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
request.

- **capstone-qemu:** the supervisor rules (`DEAD` status, dead-slot removal in the collector and on
  slot shortage, forget of an invalid seal).
- **capstone-sbi:** the hand-off slot, `ADOPT`, generation ids, `DEAD`/`STALE`, and application
  memory kept apart from context slots.
- **caplifive-buildroot (driver):** context ids apart from process blocks, and the `ADOPT` ioctl.
- **llvm-capstone:** mint, thread entry, exit path, per-context state, locks, the park protocol in
  launcher and libc, and the two probe programs.

After both probes, `delegation-threads-runtime` builds musl's `__clone`, and with it
`pthread_create`, `join`, `detach` and `exit`, on mint and adopt. The tid question is decided there:
musl uses `t->tid` for mutex ownership and for `tkill`, and the choice is between a runtime-assigned
id routed by the launcher and the Linux tid taken as an answer. Gates:
- the libc-test thread group leaves the excluded set;
- GLib's `GCond` in the tshark deps (the `pthread_cond_t` size fix, e2c9ad3);
- CPython's basic `threading` tests.
