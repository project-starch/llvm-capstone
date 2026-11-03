# Experimental native domain exec

Branch `runtime/module-binfmt` implements the M4 experiment. The module's
`dom_binfmt=1` parameter registers a root-only loader; the default remains off.
This is not a qualified replacement for the launcher. See the
[contracts and measurements](../../../runtime/virtual/tests/native-dom/README.md).
Round 3 adds a module resource boundary. Keep the experiment off pending full
compatibility and gate qualification; the original exploit remains a negative control.

## Exec and transport

The loader validates the existing virtual application ELF and constructs the
domain's image, stack, startup block, metadata and fixed exchange grants.
It returns from Linux exec into a scalar trampoline, so Linux completes exec
and retains its ordinary wait, stop, termination and task teardown paths.
There is no glibc launcher, spawner, dummy thread or newly installed seccomp
filter. VFS operations use the current Linux credentials. Inherited filters
constrain actual scalar syscalls and ioctls; kernel services behind an admitted
ioctl use the module policy, not per-virtual-call seccomp emulation.

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

## Resource boundary and design choice

The module validates every request before the scalar trampoline runs it. The
trampoline stays 152 bytes and has no new filter. A syscall-number seccomp filter
would still permit the demonstrated procfs `pread`/`pwrite` attack; it cannot
inspect pathname contents or file identity. A trampoline policy would duplicate
the existing wire validator and need more trusted scalar code. The module can
hold VFS objects and validate them before side effects, so it is the chosen broker.
This is a narrower ABI, not equivalence with the launcher's complete API.

The initial cwd is an explicit directory grant. It must be a directory on tmpfs,
ext4 or 9p and cannot be `/`. Relative paths start there or at a directory opened
through that grant. Absolute paths must have that initial cwd prefix. Each
component is opened as `O_PATH|O_NOFOLLOW`; intermediates also require
`O_DIRECTORY`. `..`, symlinks, magic links, mount transitions and all final types
except ordinary files/directories are refused. Linux 6.1 does not export an
in-kernel `openat2` entry point to modules. This stricter component walk supplies
the intended beneath/no-magic-link/no-mount-crossing constraints without checking
one name and later reopening the original pathname. Symlinks within the root are
also unsupported. A renamed pinned directory remains the same granted object.
The Linux kernel, VFS and filesystem server are trusted; a hostile host process
already able to ptrace this task is outside this boundary.

Existing files are opened from the checked, pinned dentry after inode_permission
and append-only/directory checks; O_PATH and dentry_open alone do not perform
Linux may_open. The ordinary open still performs file LSM hooks, leases and
write-access acquisition. A UID-0, capability-empty exec probe verifies DAC
refusal, with a failing control on the pre-check broker. Creation forces
`O_EXCL` so a racing device or symlink cannot be opened. Truncation follows inode
validation. Stat calls use a private temporary O_PATH fd and `AT_EMPTY_PATH`;
the fd is closed before re-entry. Out-of-grant metadata lookups return ENOENT
without exposing whether a host path exists; opens still return EACCES. Unlink locks the pinned parent and removes one
leaf without following it. Other path operations return ENOSYS. Open flags are
explicitly restricted, including refusal of O_PATH and O_TMPFILE requests.

A per-context bitmap grants at most 1,024 fd numbers. Only safe inherited stdio
(ordinary files on the admitted filesystems, anonymous pipes, `/dev/null` or
`/dev/console`) is
initially granted. All other inherited fds, including directories, sockets,
process-memory files and device handles, are inaccessible. New descriptors come
only from the broker; close revokes the grant before scalar dispatch. Thread
creation, fd duplication, descriptor receipt and exec are unsupported, so no
concurrent task can replace a granted number. Exec unshares the Linux fd table;
the loader rejects LSM_UNSAFE_SHARE so an external CLONE_FS peer cannot share
cwd/umask with the new domain. A native clone/exec negative control verifies
this refusal. The private context fd is never
admitted. The caller explicitly grants the selected stdio files; the directory
policy is not a sandbox against an overbroad or sensitive file chosen by that
trusted caller.

## Closed syscall set

`dom_forwarded()` is the exhaustive ordinary call allowlist. Unknown numbers
return ENOSYS. All remaining ordinary rows are in the shared wire-shape table:

```
getcwd fcntl ioctl unlinkat ftruncate sync_file_range flock openat close getdents64
lseek pread64 pwrite64 readv writev preadv pwritev newfstatat fstat fsync
fdatasync clock_gettime gettimeofday times getpid getppid getuid geteuid getgid getegid
gettid umask uname sysinfo prlimit64 getrandom sched_yield getpgid getsid fstatfs
statx fallocate sendfile copy_file_range readahead fadvise64 clock_getres getgroups getrusage getpriority
sched_getaffinity sched_get_priority_max sched_get_priority_min sched_rr_get_interval getresuid getresgid rt_sigprocmask
```

Open and unlink execute entirely in the module. Read/write also stay in the
module and check the same fd grants. FCNTL_LOCK admits only the shared lock
commands. Ordinary fcntl permits GETFD, SETFD and GETFL; ioctl permits only FIOCLEX
and FIONCLEX. Buffered ioctl is refused. Prlimit is query-only and self-only;
scheduling/priority queries retain their existing self constraints. Umask only
changes this task's future file-creation mode. Identity, time, system and process
metadata queries remain deliberately available. File I/O, truncation, allocation,
sync, advice and locks affect only granted files.

Special SIGACTION accepts only default/ignored dispositions. Signal-mask updates
use a checked private buffer. No native callback address is accepted. The module
also serves exit, thread-self, anonymous MAP, whole-grant UNMAP, PROTECT and node
reservation. MAP chooses a fresh range; PROTECT/UNMAP require owned dynamic
arenas and cannot name the image, fixed transport or scalar bridge. The
trampoline's own ioctl/mprotect/exit_group calls receive only module-prepared
arguments. This is not semantic equivalence to arbitrary inherited filters
written for direct Linux file syscalls; those filters must account for the
module ioctl API.

## Authority audit

| Vector | Boundary and tested scope |
|---|---|
| `/proc/self/mem`, maps, pagemap | No ambient path access; procfs is not an admitted filesystem, and mounts cannot be crossed. Named paths and a proc mount within the grant are probed. |
| `/proc/self/fd`, cwd, root, exe; thread-self and pid aliases | Magic links and all other symlinks are refused before opening. Absolute paths outside the cwd grant fail. Symlink aliases and an inherited proc directory fd are probed. Numeric pid paths follow the same rule. |
| Inherited proc/device fds; fd reuse and vector I/O | Only safe stdio is granted. Every I/O fd is checked, including scalar/vector, sendfile/copy-file-range positions and module read/write. Probes include unsafe fd 0, fd 9, directory fd 8, read/write/pread/pwrite/readv/writev/fstat and closed-fd reuse. |
| `process_vm_readv/writev`, ptrace | Absent from the allowlist; raw-wire refusals are tested. Capability-aware external ptrace remains unresolved. |
| Raw mmap, mprotect, munmap, mremap, brk, madvise | Absent from the allowlist. Raw-wire refusals and the dedicated PROTECT and UNMAP services against the actual scalar trampoline address are tested. Existing contracts exercise owned map/unmap and invalid protect ranges. |
| prctl, personality, execve/execveat | Absent, with raw-wire refusal probes. No name-space, credentials, memory-policy, executable or native handler changes are exposed. |
| io_uring, userfaultfd, perf_event_open, bpf | All entry points absent; raw-wire probes. No retained userspace pointers or kernel execution interface admitted. |
| Arbitrary-fd ioctl and hidden pointers | Untrusted descriptors fail the grant check; only the two fd close-on-exec commands are admitted. Scalar and buffered TIOCSTI probes are refused. No device-specific ioctl dispatch is exposed. |
| Signals to self or others | kill/tkill/tgkill, timers, sigreturn, altstack and raw sigaction are absent; probes cover them. F_SETOWN/F_SETFL O_ASYNC are refused. Caught handlers are unsupported. External Linux stop/continue/termination remain covered by contracts. |
| fork/clone/clone3, threads, descriptor passing | Absent, with raw-wire clone/clone3/socket/recvmsg probes and a libc fork refusal contract. No shared-fd-table races are allowed. |
| General path calls, symlinks, hard links, mounts, devices | Only brokered open/unlink/stat are admitted. Parent traversal, symlink aliases, a nested proc mount, memory/null device nodes, link/symlink/mknod/rename/chdir/fchdir and raw openat2 are probed. Mount/namespace, open-by-handle and pidfd_getfd probes are also refused; unknown syscalls default to ENOSYS. |
| Ordinary output pointers and nested arguments | Shared shape validation, private bounce memory and rebuilt iovecs; strings have independent slices. Overlapping pathname/stat-output wire probe and malformed lengths/flags remain in the contracts. Nested messages and arbitrary pointer-bearing commands are refused. |

The checked boundary prevents the demonstrated domain-to-scalar memory channel
under these trust assumptions. It does not establish full Linux API compatibility,
complete signal/restart handover, production isolation from a privileged host, or
nine-gate parity. Functional and cost qualification is recorded separately in the
[native test README](../../../runtime/virtual/tests/native-dom/README.md).
