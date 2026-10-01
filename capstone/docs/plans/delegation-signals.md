# Delegated signals: synchronous delivery, Linux keeps the state

Status: IMPLEMENTED, 2026-09-29, on `delegation-signals` (over
`delegation-launch-cost`). The contract in the last section passes 26/26 in the
guest, and the application gate, the binfmt contract and Perl `t/base` pass on
the same images; the record is
`runtime/tests/application/results/20260929-signal-contract.json`, the
user-facing summary is in `runtime/applications.md` under Signals. The wire
(96-byte entry with a round status, 16 KiB META region with the handover block
at 4 KiB, the recorded-sequence word) retains its event offsets; two inherited
state words are appended after the handover array. What this
branch does not do is listed under Deviations; the bell is still later.

## Responsibility

Linux decides every kernel operation and its native signal handling:
dispositions (`SIG_DFL`, `SIG_IGN`, caught), the blocked mask, the pending set,
queueing of realtime signals, default actions, whether an interrupted system
call is restarted or fails with `EINTR`, inheritance by children, process
groups, and the reset of caught signals on `exec`.

The runtime is responsible for one thing Linux cannot do: carrying a signal
the kernel has already delivered to the task into a handler that lives in the
domain, with the states that carrying needs, and with the deviations from
Linux named explicitly. The monitor knows nothing about signals.

The domain holds only what is a domain address: the handler functions, their
flags and `sa_mask`, and the alternate stack.

## The one state Linux does not have

When Linux delivers a caught signal it removes it from the pending set and runs
the handler at once. Here the handler runs later, in the domain. Between those
two moments the event is owned by the runtime, in these states:

1. **accepted**: the launcher's native trampoline ran; the kernel has forgotten
   the signal. The trampoline recorded the event into a private ring.
2. **published**: at the end of a round the launcher moved the event into the
   handover block in the META region, which stays unchanged until the domain
   acknowledges it.
3. **running**: the libc has entered the domain handler.
4. **done**: the libc acknowledged the event with a runtime request; the
   launcher frees its domain slot and removes its deferred signal from the
   in-flight set.

Every event carries a sequence number. Publish and acknowledge are by sequence
number, so no event is delivered twice and no block is released early.

## The launcher

**Dispositions.** `rt_sigaction` is delegated, with the handler pointer replaced
by a class: `SIG_DFL` and `SIG_IGN` are installed in the kernel as they are; a
caught signal gets the trampoline, installed with `SA_SIGINFO`, with the
domain's `SA_RESTART`, `SA_RESETHAND`, `SA_NOCLDSTOP` and `SA_NOCLDWAIT` flags
mirrored, and with `sa_mask` = all signals so recording is atomic. Linux then
decides restart against `EINTR` by its own rules, and resets a `SA_RESETHAND`
disposition itself at acceptance. The old action is answered from the libc's
table, not from the kernel.

**Mask.** `rt_sigprocmask` is delegated. The kernel's blocked mask is the
domain's mask, plus what the runtime adds below (physical mask).

**Trampoline.** Async-signal-safe, does four things: appends
`{seq, signo, siginfo (128 bytes), uc_sigmask, pc}` to the private ring (single
producer, single consumer, lock-free); bumps the recorded-sequence word in the
shared META region; adds the signal to `uc_sigmask` in the frame when the
domain's action has no `SA_NODEFER`, so the signal stays blocked in the kernel
until the domain handler is done, as Linux blocks it during the handler; and
redirects the interrupted program counter when it lies inside the syscall stub
(below).

**Stub.** Every delegated system call the launcher executes on the domain's
behalf goes through one assembly routine with marks `begin`, `ecall`, `after`.
Inside the protected range, before `ecall`, the stub checks the ring and skips
the call when events are waiting. The trampoline sets the frame's pc to the
stub's `retry` exit when pc is in `[begin, ecall]`: that covers a signal that
arrived after the ring check, a signal that arrived before the call, and a
restart the kernel prepared (RISC-V rewinds `epc` by 4 and restores `a0` for
`ERESTARTSYS` with `SA_RESTART` and for `ERESTARTNOINTR`). The launcher then
reports the round as `RETRY`: no result, no output data. A completed call
reports its result, `-EINTR` included, and whatever events were accepted.

`RETRY` means "no final result for the application yet", not "never entered":
a kernel-prepared restart has entered the call and Linux decided it be
repeated. Only completed calls copy output data back.

**Compound operations** are not covered by the stub rule and get their own
continuation points: `wait_child`'s polling loop (`wait4(WNOHANG)` plus a 1 ms
`nanosleep`) checks the ring each iteration and answers `RETRY` when every
accepted signal's action has `SA_RESTART`, `-EINTR` otherwise; the spawn
protocol never repeats a request once it has been sent to the helper and
finishes the reply receive on `EINTR`. The launcher's own housekeeping calls
(publishing, masks, acknowledgement) never use the stub and are never held
back by a non-empty ring.

**Physical mask.** The kernel mask the launcher applies is
`logical ∪ inflight ∪ backpressure`: the domain's logical mask as delegated,
the signals of accepted-but-not-done events without `SA_NODEFER`, and the
currently caught signals while the ring is nearly full. The trampoline adds
backpressure to its return frame at the threshold, before another event can
fill the ring. Every logical change goes
through the launcher, which reapplies the union; acknowledgement removes a
signal from `inflight` and reapplies.

**Step ioctl.** `EINTR` from the driver's interruptible lock means the domain
was not entered: retry.

**Helper.** The spawner helper is cloned with exit signal 0, so its own death
produces no `SIGCHLD` for the launcher and nothing has to be filtered. It
resets every disposition to default at start and ignores the terminal signals
itself, so a Ctrl-C the application survives does not kill its spawner. Each
spawn request carries the launcher's currently ignored set, the request's
`POSIX_SPAWN_SETSIGDEF` set and `SETSIGMASK` mask; the helper applies ignored,
then defaults, then the mask, and `exec` carries the rest, as Linux does.
Children are cloned with `CLONE_PARENT | SIGCHLD`, so their `SIGCHLD` reaches
the launcher.

## The libc

**Order in a round.** Save result, status and published events; copy output
data back for a completed call; release the transport; deliver; on `RETRY`
marshal the call again from the caller's buffers and repeat. Handlers never
run while the exchange region still holds a call's data.

**Which events run, under which mask.** Two classes:

- An event accepted while the launcher was inside a wait with a temporary mask
  (`rt_sigsuspend`, `ppoll`, `pselect6`, `epoll_pwait` with a mask; pc at or
  before the stub's `after` mark) is bound to that wait. It runs before the
  wait's result is returned, whatever the restored mask says, under
  `temporary ∪ sa_mask ∪ {sig}` (`{sig}` unless `SA_NODEFER`); afterwards the
  wait's original mask is in force. This is what Linux does: the frame carries
  the original mask (`sigmask_to_save` picks `saved_sigmask`), the handler runs
  under the temporary one, the original returns on `sigreturn`.
- Any other event is runnable when its signal is not blocked by the current
  logical mask, in acceptance order, under `current ∪ sa_mask ∪ {sig}`. That
  is the base Linux uses (`current->blocked` at delivery), and it keeps a
  handler that blocked itself blocked while a nested handler runs.

Nested delivery happens at the end of a handler's own rounds, exactly as Linux
delivers at the next return to user mode; a signal blocked by the running
handler's mask waits. The management steps of delivery (take events, confirm a
mask change, update the table, enter the handler, acknowledge) are one
guarded transition each; general delivery does not restart inside them, and a
running application handler does allow nested delivery.

**Installation generations.** Each `rt_sigaction` bumps the signal's
generation; an event records the generation at acceptance. `sigaction` first
runs the runnable accepted events of that signal with the handler they were
accepted for, then installs the new one. `SA_RESETHAND` resets the libc table
only when the generation is unchanged; the kernel side was reset at acceptance.

**Acknowledgement.** After the handler and the restoring mask round, the libc
acknowledges the event's sequence number. Every published event holds one of
the libc's 256 slots until that acknowledgement; the launcher does not publish
beyond those slots. Events without `SA_NODEFER` also leave the in-flight mask.

**Hint.** The recorded-sequence word in the META region is compared with the
consumed sequence at every entry into the libc's syscall dispatcher, local
answers included; when it moved, the libc makes a fetch round. The latency
promise is therefore "the next entry into the syscall dispatcher": a pure
computation, or a `strlen`, never notices. That gap is the bell (below).

**Realtime signals** are queued in the ring in arrival order with their
`siginfo`; standard signals coalesce keeping the first event. When the ring is
nearly full the launcher blocks caught signals in the physical mask so the
kernel queues realtime signals (up to `RLIMIT_SIGPENDING`, its own limit) and
unblocks when the domain drains. A spare ring record retains the original
`siginfo` if one event reaches the full-ring boundary. No event is dropped by
the runtime.

**`sigaltstack`** is implemented in the libc: queries, `SS_ONSTACK`, `SS_DISABLE`
and `EPERM` while on the stack, handlers with `SA_ONSTACK` run on it through a
small stack-switching entry, a nested handler already on the stack does not
reset to its top, and `siglongjmp` out of a handler leaves the stack state
consistent. CPython's `faulthandler` refuses to enable without it.

**`sigevent`** for `timer_create`: `sival_ptr` is a capability; the domain keeps
the value and hands Linux an opaque token, resolved back at delivery.
`SIGEV_THREAD` does not exist without threads. `siginfo_t` is translated field
by field, for handlers and for `rt_sigtimedwait`.

**`raise()`** is musl's: block application signals, `tkill`, restore the mask.
`tkill` joins the table, restricted to the task's own tid.

## Deviations, named

- Synchronous only: a domain that neither yields nor enters the syscall
  dispatcher does not run a handler. Default actions still apply at once,
  because they are the kernel's.
- A second instance of a signal without `SA_NODEFER` cannot trigger the default
  action before the first handler ran: the signal stays blocked in the physical
  mask. What can differ from Linux is the moment the handler runs.
- `ucontext` in a domain handler carries no register image.
- Backpressure blocks caught-signal delivery to the trampoline while the
  ring is nearly full; the kernel keeps them, so nothing is lost, but ordering across
  standard and realtime signals during backpressure differs from Linux.
- `ITIMER_VIRTUAL` and `ITIMER_PROF`: how Linux accounts the domain's time
  inside the step ioctl is measured before either is claimed. `ITIMER_REAL`
  and `alarm` have no such question.
- Domain faults stay fatal, whatever the domain's `SIGSEGV` action; an
  explicit `raise(SIGSEGV)` is an ordinary caught signal.
- Not in this branch: `timer_create` and `sigevent` (`getitimer`/`setitimer`
  are delegated), the `epoll_pwait` mask (`rt_sigsuspend`, `ppoll` and, since
  `delegation-runtime-rows`, `pselect6` classify a wait), `tgkill`, `signalfd`. `ITIMER_VIRTUAL` and
  `ITIMER_PROF` are delegated as they are; how Linux accounts the domain's
  time inside the step ioctl is not measured.
- Conformity worth naming: `rt_sigprocmask` writes `sigsetsize` (8) bytes of
  old mask, exactly as the kernel does. musl's own `sigaction` relies on it
  with an `unsigned long[1]` around a `SIGABRT` installation; the first
  implementation wrote a whole `sigset_t` there and Perl died at exit of it.

## Design points, settled by the contract tests

1. Mask transitions: the two classes above suffice. `sigsuspend`, `ppoll`,
   `nest`, `self`, `self-nodefer` and `self-defer` pass with an event carrying
   one mask (the wait's temporary mask for a wait event, nothing for the
   rest); the restore mask is the libc's mirror at entry. What the tests did
   force: the mirror moves, and the table moves, before delivery decides what
   is runnable, because the round that unblocks or reinstalls a signal is the
   round that publishes it.
2. Compound operations: `wait_child` answers `RETRY` when every accepted
   signal's action has `SA_RESTART` and `-EINTR` otherwise (`wait-restart`,
   `wait-eintr`); the spawn protocol finishes its reply after the request went
   out (`spawn-interrupted`). Vector I/O needs no continuation of its own: it
   goes through the stub and `retry-partial` passes.

The wire keeps the 96-byte entry and the handover event offsets. The launcher
appends inherited mask and ignored-set words after the 64 event slots in the
16 KiB META region; older images still read their events at the same offsets.

## Contract tests: the definition of done

A domain program `signal-contract.dom` with one mode per case, driven from the
host like `application-contract.dom`, each with its oracle:

| Case | Oracle |
|---|---|
| signal before a blocking `read` on an empty pipe | handler runs before `read` blocks; `read` then returns the data written by the handler's own `write` |
| handler that calls `write()` | the outer `read`'s data is intact; the handler's bytes arrive |
| `sigsuspend` and `ppoll` with a temporary mask | the original mask blocks the waking signal; its handler runs anyway before the call returns; the mask after the call is the original |
| nesting A -> B without `SA_NODEFER` | A stays blocked while B runs; A does not re-enter |
| self-delivery, with and without `SA_NODEFER` | `raise` runs the handler before returning; with `SA_NODEFER` a nested `raise` re-enters, without it it does not |
| `RETRY` against partial transfer | a `write` interrupted after a partial transfer returns the partial count, no byte twice; a `read` restarted by `SA_RESTART` returns once |
| `waitpid(-1)` with `SA_RESTART`, spawn interrupted after the request | the wait completes with the child's status after the handler; exactly one child exists |
| `waitpid(pid)` with `SA_RESTART` | the handler runs while the child is alive, then the wait returns its status |
| `SA_RESETHAND` twice, then reinstall | second instance takes the default action or is blocked, per the deviation list; a reinstalled handler is the one that runs |
| ring overflow with realtime signals | no event lost, count equals sent |
| 400 realtime signals with `SA_NODEFER` | all 400 handlers run, including when the domain event queue fills; the native launcher test checks every queued payload |
| recorded-sequence hint | a signal accepted during computation runs at the next local syscall |
| `sigaltstack` with a nested handler and `siglongjmp` | `SS_ONSTACK` reported inside, stack pointer inside the alternate stack, no reset to the top for the nested one, state consistent after the jump |
| `siglongjmp` followed by a deep stack frame | the abandoned regular-stack handler is acknowledged; its signal can run again |
| `sigtimedwait` for `SIGCHLD` | `si_pid` and the child's exit status are translated into the domain's `siginfo_t` |
| inherited mask and ignored disposition in a second domain | `sigaction`, `sigsetjmp` restoration and a native grandchild observe the inherited state |
| `SIG_IGN` inheritance after the helper started | a child spawned after `signal(SIGPIPE, SIG_IGN)` ignores it; one spawned with `SETSIGDEF` does not |
| a caught `SIGABRT` | musl's `sigaction` blocks all signals around it with an 8-byte old-set buffer; the handler runs once and the process survives |

Result, 2026-09-29: 26/26 modes pass. Two modes were wrong as first written and
looped forever on Linux as well (each handler raised the other without end;
the alternate-stack mode re-entered its own `sigsetjmp`); both now cross-raise
once. The five modes in the second half
of the table (the positive-PID wait, the 400-signal burst, the deep-stack jump,
`sigtimedwait` output, inherited state) came from a review of the first
implementation and found the fixes the launcher and libc sections describe:
positive-PID waits poll, every published event holds a domain slot until
`SIGDONE`, backpressure is set before the ring fills, `longjmp` acknowledges the
handlers below its saved stack pointer, `sigtimedwait` output is translated
before delivery, and startup mirrors the inherited mask and ignored
dispositions. The existing gates on the same images: the four contract fault
modes unchanged, the application gate 21 PASS lines, the binfmt contract, Perl
`t/base` 9/9 in 23 s, the native suite 25/25. libc-test with the SDK built from this commit: 48 PASS, 4 FAIL, 2 FAULT, 3 NOBUILD, 20 EXCLUDED of 77; `popen` (SIGUSR1 from its child) and `setjmp` (six `sigprocmask` calls that were no-ops) pass now, `tls_local_exec` fails instead of faulting, the rest is unchanged: `clocale_mbfuncs` faults at startup as before, `mntent`, `strptime` and `strtold` fail as before, the TLS tests do not build.

## The bell, later

Asynchronous delivery into a computing domain is a generic doorbell: the
owner task asks the monitor to interrupt domain X, one bit and no meaning;
the supervisor lands the next resume on the domain's interrupt entry with the
interrupted context saved; a libc entry there reads the handover block. The
monitor learns nothing about Linux. It is the same primitive threads need,
and it is not this branch.
