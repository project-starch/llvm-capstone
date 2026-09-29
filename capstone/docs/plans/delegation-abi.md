# Delegated syscalls and the task model

Status: PLAN, 2026-09-29. Branch `delegation-abi`, from `dev`. Nothing here is
implemented yet; every claim of behaviour below is a target, and every number a
gate to be measured, not a result.

## Goal

A Capstone application is the user half of one Linux task. Every operating-system
service it uses is the Linux syscall itself, executed by the task that owns the
domain, under that task's credentials, descriptor table, working directory and
seccomp filter. The runtime keeps no OS state in the domain. The monitor learns
nothing about processes.

What this replaces: the HostCall v0 opcode list, the domain-side file table,
file positions, pipes, working directory and timers, the stdio special case,
the 4 KiB payload region and the bounce buffer. What it keeps: the CALL/REGION_SHARE
entry ABI, the resumable yield, managed ownership tied to the open device file,
and the supervised step with its typed events.

The measure of success is not a new feature list. It is that the ports stop
inventing conventions: no absolute `argv[0]` because `getcwd` is missing, no
standard library packed into a zip because directory reads cost a round each,
no `ac_cv_func_mmap=no`, no thread-locals rewritten as globals. The gates are
the existing ones: musl's libc-test in a domain, today 43 of 77 with 22 excluded
for missing processes, threads and sockets; Perl `t/base`, today 8 of 9; and
the persistent-guest application gate with its 1,008 mixed starts.

## The wire ABI

One versioned request block in the metadata region, laid out as an io_uring
submission entry so that a later kernel-side or io_uring transport changes
nothing above it:

| Field | Meaning |
|---|---|
| `version` | 2 |
| `count` | entries in this batch, 1 for now |
| `nr` | Linux syscall number, RV64 table |
| `args[6]` | integers, or offsets into the exchange region for pointer arguments |
| `result` | the syscall's return value, negative errno on failure |
| `flags` | which arguments are exchange offsets, for the launcher's bounds check |
| `pending` | signal mask written by the launcher on every return |

Pointer arguments never cross as capabilities and never as domain addresses.
The libc copies buffers into the **exchange region** and passes offsets; the
launcher validates each offset and length against the region and hands the
kernel its own mapping address. This is the copy every kernel makes at the
user boundary, once per syscall, and it keeps domain memory invisible to Linux.
An identity mapping of the domain block into the task is an optimization for a
later branch, not part of this ABI.

The exchange region size is declared in the application descriptor, like the
heap. Requests larger than the region are chunked in the libc, as reads and
writes are chunked today.

Structures that contain pointers are marshalled in the libc, because our
pointers are 128 bits wide and the kernel's are 64: `iovec` for the vector
forms, `msghdr` when sockets arrive, argv and envp for spawn. Everything else
is integers and byte buffers and passes through. The reference for every
syscall's semantics is the Linux manual page; this document describes only the
transport, the marshalling and the exceptions.

## The three exception groups

Everything not listed here is delegated. The list is closed; adding to it is
an ABI change.

**Memory.** `mmap`, `munmap`, `mremap`, `brk`, `mprotect`, `madvise`. Domain
memory is capability memory and Linux cannot grant authority into it. Anonymous
`mmap` is served by the domain allocator. File `mmap` is ENOSYS. A later branch
turns anonymous `mmap` into a region grant through the monitor, delivered at
the yield's resume label; the driver already resumes shares across preemption.

**Processes.** `clone`, `fork`, `vfork`, `execve`, `exit_group`. `fork` without
`exec` is ENOSYS by design and stays visible in the unserved report. `exit_group`
is delegated as is: the launcher ends the process with the status. The rest
become the spawn service below.

**Signals.** `rt_sigaction` keeps a table in the domain, `rt_sigprocmask` a mask
in the launcher, `rt_sigreturn` is unused. Delivery is synchronous: the launcher
receives the Linux signal, records it in `pending`, and the libc runs the
installed handler before returning from the yield. This covers SIGCHLD, SIGALRM,
SIGPIPE and Ctrl-C for any program that makes syscalls, which is every
interpreter in the study. Asynchronous delivery into a domain that never yields
is a later branch and needs the sibling-context primitive.

## The task model

- **Spawn.** `posix_spawn` in the libc becomes one request: path, argv, envp,
  cwd, file actions. The launcher forks, applies the file actions, and execs.
  A Capstone image with an ABI v1 descriptor runs under `capstone-exec`; a
  native Linux program runs directly. The child is a real process with a real
  PID. The launcher records only children it started.
- **Wait and kill.** `wait4` and `kill` are delegated, restricted to recorded
  children. `capstone-job` already records waitpid status; it stays.
- **Exec in place.** `execve` ends the step loop, closes the device, which
  destroys the domain, reopens it, creates the new domain and continues with the
  same PID and descriptors.
- **Pipes and descriptors.** `pipe2` is a Linux pipe. `dup`, `dup3`, `fcntl`
  including `F_SETFL`, `ppoll` and `lseek` are delegated, so shared offsets,
  non-blocking mode and real waiting come from Linux. The domain-side pipe
  queue and file table are deleted.
- **Threads.** Not in this branch. The design is one launcher thread per sibling
  context; the primitive is monitor work.

Port patches: Perl's `my_popen` and `do_exec`, and CPython's
`_posixsubprocess.fork_exec`, are routed to `posix_spawn`. Both interpreters
already have such a path for Windows.

## Policy

The launcher installs a seccomp filter derived from its allowlist before the
first step. The set of syscalls a domain can reach is then enforced by the
kernel, not by runtime code. The default profile is the delegated set above
minus sockets; a port that needs more says so in its descriptor and the
launcher refuses anything outside the profile.

## Faults

The supervised step already returns cause, PC and address. The launcher prints
them, together with the image hash and load base, on stderr before raising
SIGSEGV, and writes the same record next to the `capstone-job` result. A
symbolizer on the host maps PC to a line using the retained debug image. A
fault record without a symbol is a bug in this plan, not an acceptable output.

## Measurement

`--stats` gains counters for delegated calls, bytes through the exchange
region and cycles per round, read from `rdcycle` in the launcher around each
step. The first commit that runs an application records cycles per delegated
syscall on QEMU with `icount` next to a native process making the same call.
No optimization lands before that number exists.

## Delivery, one branch per step

Each step stacks on the previous one, is gated, and lands squashed.

1. **`delegation-abi`**: this document, the request block header, the
   marshalling table as a header, native unit tests for pack, unpack and bounds
   validation.
2. **`delegation-libc`**: the musl dispatcher becomes a stub plus the exception
   table; HostCall v0 stays behind `CAPSTONE_APPLICATION_RUNTIME` for legacy
   probes. Gate: libc-test file, time, directory and descriptor groups at or
   above today's count.
3. **`delegation-launcher`**: generic dispatcher, seccomp profile, fault record.
   Gate: the application contract program, `perl.dom` and `mruby.dom` through
   the persistent-guest gate; `capstone-exec --stats` shows the new counters.
4. **`delegation-spawn`**: spawn of native programs, wait, host pipes, exec in
   place. Gate: Perl `t/base` 9 of 9; the libc-test process group leaves the
   excluded set.
5. **`delegation-memory`**: region grant at the resume label, chunk allocator,
   heap declared as initial size. Gate: CPython built without `ac_cv_func_mmap=no`.
6. **`delegation-signals`**: synchronous delivery. Gate: the libc-test signal
   group; CPython's `signal` tests that do not need a second process.

Out of scope for the whole stack: threads, sockets, asynchronous signals, the
identity mapping, io_uring, any monitor change beyond the grant, any ISA change.

## What the existing applications become

**SQLite** keeps its two entry styles. The standalone domain with memsys5 and
the two-region host protocol is unchanged and stays the reference for the
boundary benchmark and the revocation probes. The `speedtest1` and SLT builds
that link musl move to the delegated path and gain a real VFS: `xOpen` opens a
file, `xRead` and `xWrite` are `pread` and `pwrite`, `xSync` is `fsync`,
`xAccess` is `faccessat`. The in-memory smoke stays exactly as it is.

**CPython** loses patch 0006 once threads exist and nothing before; the other
thirteen patches are about pointer width and stay. Its build drops
`ac_cv_func_mmap=no` in step 5. The standard library can be a directory on the
share instead of a zip. `subprocess` works for native children in step 4,
`signal` in step 6. The test suite becomes runnable through the upstream
runner; what fails then is a port result, not a runtime gap.

**Perl** needs one patch fewer, the spawn path replaces the private
`my_popen` workaround, and `t/base` completes. The rest of the upstream suite
becomes the next port target.

**mruby** changes nothing in source. It gains `system` and file I/O with real
semantics.

**PostgreSQL** keeps the single-user backend for the study. The postmaster
becomes possible with `EXEC_BACKEND` after step 4, and its self-pipe latch is
correct by construction on a Linux pipe with `O_NONBLOCK`.

**FFmpeg and tshark** keep their app ports; their file service calls map one to
one onto delegated `openat`, `pread`, `pwrite` and `close`.

The legacy probes under `tests/runtime-qemu` and the FPGA gates keep HostCall
v0. They are evidence for silicon claims and are not migrated by this plan.
