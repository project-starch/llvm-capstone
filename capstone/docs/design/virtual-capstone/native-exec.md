# Experimental native domain exec

Branch `runtime/module-binfmt` implements the M4 experiment. The module's
`dom_binfmt=1` parameter registers a root-only loader; the default remains off.
This is not a qualified replacement for the launcher. See the
[contracts and measurements](../../../runtime/virtual/tests/native-dom/README.md).
The scalar-memory probe confirms a security blocker; keep the experiment off.

## Exec and transport

The loader validates the existing virtual application ELF and constructs the
domain's image, stack, startup block, metadata and fixed exchange grants.
It returns from Linux exec into a scalar trampoline, so Linux completes exec
and retains its ordinary wait, stop, termination and task teardown paths.
There is no glibc launcher, spawner, dummy thread or newly installed seccomp
filter. Existing Linux credentials and inherited filters still apply.

The trampoline contains no libc and only a step loop, a validated protection
bridge, a generic prepared syscall bridge, completion ioctls and exit. Its
assembly source is `runtime/virtual/module/trampoline.S`; the loader maps the
corresponding words read/execute. The native context fd, scalar stack,
trampoline and bounce mappings receive no domain capability grants.

The module snapshots the delegate entry and uses the same wire-shape validator,
result-byte rules, descriptor positions and fcntl/ioctl command policy as the
launcher. It alone chooses the syscall and constructs native pointer arguments.
Strings occupy separate private slices so overlapping output arguments cannot
change the pathname seen by Linux. Vector descriptors are snapshotted and rebuilt
with bounded offsets. Output copies into the exchange and metadata go through
`touch()`, the implementation of CV_TOUCH, before writing.

The transport lifetime hold is structural: its grants cannot retire, only the
owning Linux task may step the context, and a pending syscall must complete
before domain re-entry. Read/write remain in the module and release the VM lock
while sleeping. Other admitted I/O uses private bounce pages. The domain SDK
still performs capability-checked copies between application objects and exchange.

## Closed syscall set

`dom_forwarded()` admits the following ordinary calls. Every row is in the
shared delegated-shape table and therefore in the launcher seccomp allowlist
constructed by `capstone_delegate_seccomp_allowlist()`. The launcher invokes
that filter from `exec.c`; the implementation lives in `linux/delegate-service.c`.

```
getcwd dup dup3 fcntl ioctl mkdirat unlinkat ftruncate faccessat symlinkat
sync_file_range flock fchmodat chdir openat close pipe2 getdents64 lseek
pread64 pwrite64 readv writev preadv pwritev readlinkat newfstatat fstat
fsync fdatasync utimensat renameat2 clock_gettime gettimeofday times getpid
getppid getuid geteuid getgid getegid gettid umask uname sysinfo prlimit64
getrandom sched_yield getpgid getsid linkat mknodat statfs fstatfs statx
truncate fallocate fchdir fchmod fchown fchownat faccessat2 sendfile
copy_file_range readahead fadvise64 sync syncfs memfd_create clock_getres
getgroups getrusage getpriority setpriority sched_getaffinity
sched_setaffinity sched_get_priority_max sched_get_priority_min
sched_rr_get_interval getresuid getresgid rt_sigprocmask socket socketpair
bind listen connect sendto setsockopt shutdown epoll_create1 epoll_ctl
eventfd2 timerfd_create timerfd_settime timerfd_gettime
```

Special wire calls FCNTL_LOCK and IOCTL_BUF translate to command-restricted
`fcntl` and `ioctl`. SIGACTION translates only default/ignored dispositions to
`rt_sigaction`; a caught disposition returns ENOSYS. The trampoline's own calls
are `ioctl`, `mprotect` and `exit_group`, also present in the launcher allowlist.
The module additionally serves read/write and exit, thread-self, anonymous MAP,
whole-grant UNMAP, PROTECT and node reservation. HELLO and SIGPOLL complete
without a Linux syscall. WAIT does not create another execution context.

Clone/thread creation, fork, exec/spawn, job control, nested-message I/O,
pointer-retaining calls, signal-aware waits and caught-signal handover are not
implemented. Unsupported calls fail with ENOSYS. Inherited masks and ignored
dispositions are published at startup; ordinary fatal/stop signals return through
Linux. This is not caught-signal/restart parity with the launcher.

## Security qualification boundary

The syscall-number subset establishes only a necessary condition for removing
seccomp. A p12 probe confirms that delegated file I/O can reach scalar host memory:
`scalar-memory.c` locates the trampoline with pread on `/proc/self/maps`, opens
`/proc/self/mem`, then reads and writes back the same instruction word. It reports
`base_found=1 opened=1 read=4 write=4`, `probe_rc=1`, and clean module unload.
The initial stdio version could not read maps because the module's read path
requires `read_iter`; positioned I/O uses the scalar syscall bridge and succeeds.

A domain cannot directly issue scalar Linux syscalls or derive capabilities for
the private trampoline, but this file-mediated authority bypass invalidates the
assumption that only trusted code can modify the scalar bridge. Arbitrary syscall
execution was not attempted. The resource policy must close this path before
the absence of seccomp can be called equivalent. A pathname blacklist alone is
insufficient: aliases, device files and inherited descriptors must obey the same
policy. The launcher's file policy also needs review; its syscall-number filter
does not itself prevent process-memory I/O.

Do not claim production isolation or complete syscall-policy equivalence from
the functional SPEC results. Signal handover, capability-aware ptrace, process
creation, raw brk authority, the resource-policy audit and full gate parity remain
qualification requirements.
