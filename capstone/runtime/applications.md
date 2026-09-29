# Applications in a persistent Linux guest

Boot once, enter the ordinary Linux shell, and run compatible Capstone programs:

```sh
capstone-exec /mnt/host/perl.dom -e 'print "hello\n"'
printf 'input\n' | capstone-exec /mnt/host/perl.dom -e 'print uc(<STDIN>)'
capstone-exec /mnt/host/mruby.dom -e 'puts 6 * 7'
capstone-exec --stats
```

The launcher is a Linux process. `main(argc, argv)` receives argv, environment
and cwd before constructors. Its three standard descriptors are inherited from
Linux. Normal exit, SIGINT/SIGTERM/SIGKILL, a domain fault and failed creation all
release the process's ownership. A capability fault produces real SIGSEGV;
normal exit 139 remains a normal exit. The parent shell and VM survive, including
when the domain destroys its stack, clears its trap vector or never yields.

This implementation targets **one-hart Capstone QEMU**, with the matching
supervised CALL extension, monitor and driver on `domain-process-runtime`.
The pinned submodule revisions are part of the implementation: the QEMU,
Buildroot, OpenSBI and monitor changes are pull requests against each
repository's integration branch, and `.gitmodules` keeps the upstream URLs.
The monitor sits on the upstream reclaim fill and M-6 fix; the
[2026-09-28 gate](tests/application/results/20260928-monitor-rebased.json)
passes on it with output identical to the earlier monitor base. Existing FPGA hardware does not implement this extension; no FPGA result is
claimed here.

## Build and install the guest tools

The Buildroot external package `BR2_PACKAGE_CAPSTONE_RUNTIME` installs
`capstone-exec`, the small `capstone-job` waitpid collector, the driver and
Dropbear. Set `BR2_PACKAGE_CAPSTONE_RUNTIME_SOURCE` to this LLVM checkout.
`S40capstone` loads the driver, selects the process API and optionally mounts the
9p host share. The serial Linux shell works without the host CLI or SSH.
See the package's README in `capstone/caplifive-buildroot/package/capstone-runtime`.

For an incremental launcher build against an existing Buildroot toolchain:

```sh
source capstone/tests/capstone-test-env.sh
cmake -S capstone/runtime/exec -B "$CAPSTONE_TMP_ROOT/application-linux" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/capstone/ports/common/cmake/toolchains/linux-guest.cmake" \
  -DCAPSTONE_LIBCAPSTONE_DIR="$CAPSTONE_RUNTIME_BUILDROOT/package/modcapstone/userspace/lib"
cmake --build "$CAPSTONE_TMP_ROOT/application-linux"
```

`CAPSTONE_BUILDROOT_DIR` supplies the compiler under `build/host/bin`;
`CAPSTONE_RUNTIME_BUILDROOT` names the matching driver source checkout.
The kernel must support CMA and 9p/virtio for this development configuration.

## One application build interface

The runtime contains no application-name dispatch. CMake projects add
`capstone/runtime` and use:

```cmake
add_executable(program main.c)
capstone_configure_application(program
  DATA_BYTES 4194304 STACK_BYTES 1048576 ARENA_BYTES 4194304)
```

Select `ports/common/cmake/toolchains/capstone-domain.cmake`, prepared musl
headers/archive, `PORT_HEADER_PROVIDER=musl` and `PORT_C11_ATOMICS=ON`. Enable
C and ASM. `HEAP sublet HEAP_LOG 22` selects the revoking allocator and declares
its transferred grant automatically. Default `level0` uses a static arena.
These allocators retain their existing, different protection semantics.

For Make/configure or another upstream build system, build the same runtime as
an SDK and give the generated driver to the upstream build as `CC`:

```sh
source capstone/tests/capstone-test-env.sh
cmake -S capstone/runtime/application -B "$CAPSTONE_TMP_ROOT/application-sdk" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/capstone/ports/common/cmake/toolchains/capstone-domain.cmake" \
  -DPORT_HEADER_PROVIDER=musl -DPORT_C11_ATOMICS=ON \
  -DPORT_MUSL_ROOT="$MUSL_SOURCE" -DCAPSTONE_MUSL_ARCHIVE="$MUSL_ARCHIVE" \
  -DCAPSTONE_APPLICATION_SDK=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$CAPSTONE_TMP_ROOT/application-sdk"
export CC="$CAPSTONE_TMP_ROOT/application-sdk/capstone-cc"
"$CC" main.c libprogram.a -o program.dom
```

`sdk.json` records the compiler, headers, linker script, runtime, libc and
compiler builtins. The driver preserves source/library order and normal
compile/preprocessor modes; unknown options and unresolved symbols fail.
Configuration checks the compiler's direct-call capability codegen and refuses
the old C-46 backend. `capstone-cc --check-toolchain` repeats that check when an
existing SDK is reused; source freshness checks alone cannot establish what an
older compiler binary actually emits.
It does not claim C++ runtime, dynamic-linker or arbitrary GCC-driver support.
The standalone CMake project defaults to the currently verified `-O1` runtime;
compiler regression work may override that explicitly. The imported upstream
objects retain their own optimization settings.

Alternatively, set `CAPSTONE_APPLICATION_INPUTS` to a semicolon-separated list
of ordinary C sources, objects and archives and `CAPSTONE_APPLICATION_NAME` to
an output basename. Omit `CAPSTONE_APPLICATION_SDK`; the output is `NAME.dom`.
Exclude old port entry adapters and runtime objects. Resource settings are
`CAPSTONE_APPLICATION_{DATA,STACK,ARENA}_BYTES`, `CAPSTONE_APPLICATION_HEAP` and
`CAPSTONE_APPLICATION_HEAP_LOG`.

The [shared port build and run interface](../ports/common/application/README.md)
covers Perl, mruby, CPython, PostgreSQL, SQLite, FFmpeg and tshark. Application
recipes use this SDK; private argv/env files and application HostCall launchers
are retired. Historical hardware and allocator probe targets remain separate.

## Host session and commands

Install the host Python package once:

```sh
python3 -m venv "$CAPSTONE_TMP_ROOT/application-tools"
"$CAPSTONE_TMP_ROOT/application-tools/bin/pip" install ./capstone/runtime/host
export PATH="$CAPSTONE_TMP_ROOT/application-tools/bin:$PATH"
capstone-vm --state "$CAPSTONE_TMP_ROOT/dev-vm" up \
  --qemu "$CAPSTONE_QEMU_BINARY" \
  --kernel "$CAPSTONE_BUILDROOT_DIR/build/images/Image" \
  --firmware "$CAPSTONE_BUILDROOT_DIR/build/images/fw_jump.elf" \
  --rootfs "$CAPSTONE_BUILDROOT_DIR/build/images/rootfs.ext2" \
  --share "$APPLICATION_SHARE"
capstone-vm --state "$CAPSTONE_TMP_ROOT/dev-vm" shell
```

QEMU needs user networking (`--enable-slirp`). Defaults are one hart, 8 GiB RAM,
512 MiB CMA, a 384 MiB retained process-storage limit, `CAPSTONE_GP_NONLIN=1`
and 65,536 revocation nodes. `--cma-mib` and `--process-cache-mib` configure
the two memory limits; the mixed application-port matrix uses 1024 and 768.
They are recorded and preserved across explicit restarts. Explicit supported
emulator environment settings are recorded in the session identity and passed
to QEMU on every boot or restart. The rootfs
runs as a disposable snapshot; files on the host share remain persistent.

The installed rootfs already supplies the tools. For development, optional
`--launcher`, `--module` and `--ssh-server` override installed components.
`--launcher` also takes `capstone-job` from the same build directory unless
`--job-helper` is supplied. The SSH override is a Dropbear multi-call binary.

```sh
capstone-vm --state "$CAPSTONE_TMP_ROOT/dev-vm" run \
  --cwd /mnt/host -e EXAMPLE='value with spaces' /mnt/host/program.dom '' 'two words'
capstone-vm --state "$CAPSTONE_TMP_ROOT/dev-vm" run \
  --result result.json /mnt/host/program.dom
capstone-vm --state "$CAPSTONE_TMP_ROOT/dev-vm" exec uname -a
capstone-vm --state "$CAPSTONE_TMP_ROOT/dev-vm" status
capstone-vm --state "$CAPSTONE_TMP_ROOT/dev-vm" restart
capstone-vm --state "$CAPSTONE_TMP_ROOT/dev-vm" down
```

`run` uses ordinary pipes over SSH; `shell` uses a terminal. Arguments are quoted
once, preserving empty strings, quotes and newlines. `--env` is guest environment,
not an implicit export of host secrets. Ctrl-C, SIGTERM and SIGHUP on the host
CLI are forwarded to that job, with a random environment token checked before
signalling its PID. Guest cancellation is confirmed even when the output pipe
is full. No application is implicitly retried after a connection failure.

`capstone-job` is a short-lived C parent, not a daemon. It records actual waitpid
status separately from application streams. For example, `--result` writes
`{"version":1,"kind":"exit","value":139}` or
`{"version":1,"kind":"signal","value":11}`. Both give shell status 139;
only the latter produces a signal diagnostic. Missing completion metadata is
reported as unknown completion/transport failure. Use `exec` for unwrapped Linux
commands; its SSH numeric status alone cannot establish signal provenance.

`up` reuses only a matching session, including hashes of its platform binaries.
A mismatch is an error; it never kills another run. `restart` explicitly stops
the guest and boots the recorded paths/settings, picking up rebuilt binaries.
`down` is explicit termination. The CLI keeps private state, loopback-only SSH,
a private client key and a pinned guest host key. QEMU retains the repository's
common QEMU lock for its lifetime. Diagnostics go to `console.log`/`qemu.log`;
they are not parsed to determine application success.

## Ownership, reuse and limits

Each managed open file owns its domains and region grants. Duplication, fork
and VMAs retain that file; the final reference triggers destruction in the
kernel even after SIGKILL. Managed handles cannot access another owner's IDs.
The old globally addressed driver API and the managed API are mutually exclusive
for a module lifetime. Installed application guests select the managed API.

The trusted monitor keeps revocation roots above domain storage and transferred
heaps. Destruction first stops re-entry, then revokes descendants, scrubs memory,
clears CPMP associations and returns extents to the reusable pool. Linux mappings
prevent linear transfer, and transferred regions cannot subsequently be mapped.
Applications cannot replace the trusted interrupt path or enable debug authority
fabrication. The VM periodically returns a protected continuation to Linux so
pending signals can terminate even code with no HostCalls.

The pool retains physical memory because the monitor still owns the original
carved authority. **`live_bytes=0` does not mean the cached RAM was returned to
the Linux page allocator.** `--stats` distinguishes live resources, cached bytes,
poisoned blocks, revocation high-water/live/retired nodes and cumulative allocations.
The default retained-storage limit is 384 MiB (`process_cache_bytes` module
parameter); storage shape and concurrency can raise the cache high-water mark.
There are 32 managed domain slots and 96 monitor region slots shared with monitor
bookkeeping. Capacity failures are ordinary errors. A cache pins the module;
changing the platform or its pool budget requires an explicit VM restart.

Before invalid node IDs are reused, QEMU clears stale tags in memory, registers
and paused continuations. Invalid saved PCC identities stay pinned. Applications
approaching the node limit are suspended at the allocation instruction, collected
through the trusted monitor context, and resumed to retry that instruction. This
works within a continuing process without application or allocator changes. The
emulator unwinds the interrupted helper after collection instead of continuing
with its temporary capability copies. The sweep also covers other paused
applications.

The 256-node emergency reserve remains unavailable to applications. If collection
cannot recover space because nodes are still valid or pinned, the application
receives the resource fault and cleanup still allows the next launch. This is a
one-hart QEMU software tag sweep; it does not establish FPGA reclamation support
or hardware collection cost. Increasing `CAPSTONE_REV_NODES` only changes the
capacity; it is no longer necessary for the six previously failing mruby repeats.

The musl port's existing syscall coverage still applies: launching a Linux
process does not add target fork, exec, threads, dynamic loading or full POSIX
fd semantics inside a domain. Standard-stream read/write/EOF/close/stat/query
fcntl use Linux objects; other file operations retain the existing host service.
The launcher uses its Linux filesystem authority and is not a filesystem sandbox.
No claim of a complete hostile-code or QEMU security audit is made.

## Delegated syscalls (application ABI v2)

Every application built by the SDK speaks ABI v2: every Linux service is the Linux syscall itself, run by the launcher
task. The domain fills a 96-byte entry at the start of the 16 KiB META region,
copies pointer arguments into the exchange region as offsets, and yields; the launcher
validates the entry against the shape table, runs `syscall()` under its own
credentials, descriptors and working directory, and writes the result back.
The wire ABI, the shape table and the closed exception groups are in
`include/capstone/delegate.h`; the design is
[docs/plans/delegation-abi.md](../docs/plans/delegation-abi.md).

What crosses: files, directories, descriptors, time, identity, limits,
`getrandom`, `wait4`, `kill` confined to the task, its children and its parent,
`exit_group`. What does not:
memory (`mmap` is the domain allocator's, file `mmap` is ENOSYS), processes
(`clone` and `fork` are ENOSYS; image exec uses the process service below), and
threads. Signals cross: the kernel keeps dispositions, mask, pending set and
restart decisions, and a caught signal runs its domain handler at the domain's
next round (see Signals below). `ioctl` crosses for the terminal requests a
libc uses and nothing else, each with its buffer size known to the libc and
its number on the launcher's allowlist: the window size, the `termios` set,
`FIONREAD` and `FIONBIO`, the pseudo-terminal pair (`TIOCSPTLCK`,
`TIOCGPTN`, so `openpty`, `posix_openpt`, `unlockpt` and `ptsname` work) and
the foreground process group (`TIOCGPGRP`, `TIOCSPGRP`). A request is an
unsigned int to the kernel; musl passes it as an int, so both sides mask it
before looking it up. The application contract's `pty` mode opens a pair
through musl's `openpty`, passes bytes through it and reads the foreground
group; the native edge test covers the entry from the launcher's side. The
unserved report at exit lists what did not cross.

The image declares the exchange region with `EXCHANGE_BYTES` (default 256 KiB,
`CAPSTONE_APPLICATION_EXCHANGE_BYTES` for the SDK project); larger buffers are
chunked, so a big read or write is a short one. A v2 image's descriptor is 48
bytes. `capstone-exec` rejects v1 images with exit 126. The SDK rejects
`CAPSTONE_APPLICATION_DELEGATE=OFF`; rebuild old applications.
`CAPSTONE_APPLICATION_GRANT_BYTES` declares shared backing for existing inner
allocators, including a Sublet outer heap where selected.

The launcher installs a seccomp filter from the same shape table before the
first step: the delegated numbers plus its own, everything else answers
ENOSYS. Installation failure aborts launch; `CAPSTONE_EXEC_NO_SECCOMP=1`
disables it explicitly for debugging. The dispatcher separately validates
command-dependent pointers and excludes launcher-private descriptors.
`CAPSTONE_DELEGATE_STATS=1` prints rounds, syscalls, refused entries, bytes
through the exchange region and `rdtime` ticks at exit.

A domain fault produces a record with cause, PC, address, the runtime address
of `domain_main`, the code bounds and the sealed image's SHA-256, written to the file `CAPSTONE_FAULT_RECORD`
names and to stderr only when stderr is a terminal or
`CAPSTONE_EXEC_DIAGNOSTICS=1` is set; application streams carry application
bytes only. `capstone-vm run` collects the record into its result JSON and
prints it. `python3 -m capstone_vm.symbolize --image IMAGE "RECORD"` maps the PC
to `function+offset` from the symbol table, and to a line when the image has
debug information. A mismatching image hash is rejected; older records without
a hash remain readable.

Measured on 2026-09-29 in the QEMU guest, `rdtime` at its 10 MHz rate, one hart:

| Loop of `getpid` | Ticks per call |
|---|---|
| Delegated round, `delegate-bench.dom`, 10,000 calls | 1,049 |
| Native process, same counter, 10,000 calls | 8.2 |

This run did not use `icount`; it is a wall-time observation affected by host
scheduling, not a hardware cost or completion of the planned per-step cycle
measurement. It predates the review corrections below.

The step ioctl behind every round used to make ten SBI ecalls: a feature probe,
the STEP, and eight QUERY calls fetching result, cause, pc and address as
32-bit halves. With `caplifive-buildroot` 201a8d4 (monitor `capstone-sbi`
02d9d47) STEP returns the whole event in one ecall, in a1..a5. Measured on
2026-09-29, same guest, same host, base and new booted back to back, median
of five runs of 10,000 calls and two of 100,000
([record](tests/application/results/20260929-step-one-ecall.json)):

| Step protocol | Ticks per round, 10,000 calls | 100,000 calls |
|---|---|---|
| Ten ecalls per step | 1,047 | 1,048 |
| One ecall per step | 893 | 885 |
| One ecall, no S-mode swap around the CALL | 527 | 519 |
| plus QEMU: TLB flush only when translation state changes | 351 | 345 |
| plus QEMU: quantum timer instead of a clock read per block | 303 | 262 |

Still wall time without `icount`, with another guest running on the host. The
fault records of the four contract fault modes are byte-identical on both
platforms; the application gate and the binfmt contract pass on the new one.

The third row is `caplifive-buildroot` c3507a5 (monitor `capstone-sbi`
a810177): the monitor's supervised invoke uses `__domcall`, so the compiler
no longer swaps the sixteen CPMP CCSRs and nine S-mode CSRs out and back
around every step. The supervisor snapshots and restores that state itself,
and each CPMP write is a full TLB flush in QEMU
([record](tests/application/results/20260929-no-smode-swap.json)).

The last two rows are emulator changes, `capstone-qemu` ac2837aa: the
supervisor's `restore_state` flushed the TLB on every switch although
satp, the CPMP registers and the mstatus translation bits never change across
a supervised switch, and every C-mode translation block began with a helper
that read the virtual clock to enforce the 5 ms quantum, 771 times per
round. The flush is now conditional and the quantum is a timer with an inline
flag test. Both rows are QEMU-only savings and say nothing about hardware; the
control for them is an unmodified build of the previous pin at 512 / 509
([record](tests/application/results/20260929-qemu-switch-cost.json)).

The number of rounds is the other half of the cost. The libc now answers
identity and the two clocks from the launch record the task writes at start
(`launch.h`: pid, ppid, the ids, and the clocks paired with `rdtime`), answers
musl's thread setup itself, and no longer opens a `fcntl` round after every
`O_CLOEXEC` open; stdio buffers are 8 KiB and the `getdents64` buffer 32 KiB
(`ports/musl-capstone/musl-patches/`). Measured 2026-09-29 with
`CAPSTONE_DELEGATE_STATS=1`: `perl -e 'print ...'` 43 -> 26 rounds, `mruby -e
'puts 1'` 6 -> 4, the getpid benchmark 10,003 -> 2. Perl `t/base` takes 35 s
either way: the nine launches, not the rounds, are what remains of that figure
([record](tests/application/results/20260929-libc-rounds.json)).

A launch, measured in the guest with `CAPSTONE_DELEGATE_STATS=1` (the
launcher prints `launch ticks` per stage), perl.dom warm on the 9p share:

| Stage | Before | After |
|---|---|---|
| image read over 9p into a memfd | 1.1 s | 0.5 s |
| SHA-256 of the image | 2.0 s | 0 (computed only for a fault record) |
| domain: loader copy and `DOM_CREATE` | 1.9 s | 0.5 s (copy the file-backed 14.5 MB, not the 82 MB memsz; zero fresh blocks only) |
| regions, heap included | 4.0 s | 4.2 s |
| total before the first instruction | 9.0 s | 5.2 s |
| wall in the guest, `perl -e 1` | 12.0 s | 6.9 s |

`capstone-vm` keeps one SSH connection per VM (`ControlMaster`), so a
command costs 0.1 to 0.3 s of host time instead of 0.5 to 1.1 s. Perl
`t/base` through `prove` from the host: 35 s -> 23 s. What remains of a
launch is the heap region: the monitor's reclaim fills a released region with
zero capabilities granule by granule and does so again when the region is
prepared for the next owner, 4 s for the 64 MiB Perl heap in QEMU
([record](tests/application/results/20260929-launch-cost.json)).

### Processes

`posix_spawn`, and with it `posix_spawnp`, `popen` and `system`, cross as one
request: the path, argv, environment, working directory and file actions in a
block in the exchange region. The launcher hands it to its **spawner**, a child
it forked before installing the seccomp filter, so the programs it starts are
not filtered. The spawner forks each child as the launcher's own child, puts
the launcher's descriptors at their numbers with their close-on-exec flags,
applies the actions, and execs: a native program directly, a Capstone image
through the launcher. Each request carries the current cwd and umask; only
application descriptors are inherited. The helper closes its inherited
descriptors and dies if the launcher dies. File actions cannot overwrite the
exec-error channel; descriptor overflow is an error rather than truncation.
`wait4` selects recorded children, including for `waitpid(-1)`, and retains
stopped/continued children. `kill` accepts this task, a recorded child or the
task that spawned it (a child domain may signal its parent), not
process-group targets; a pid nobody has answers ESRCH, any other process
EPERM. A blocking any-child wait polls recorded PIDs at
1 ms intervals to avoid reaping the helper or unrelated children. `execve` of a Capstone image replaces the task
through the launcher's own binary, keeping pid, descriptors, argv[0], the
unfiltered helper and the recorded children. The replacement image is validated
and sealed before exec; failure returns errno to the current application; `execve` of a
native program answers ENOSYS, because the filtered task cannot become one.
`fork` without `exec` stays ENOSYS by design.

A shell in the guest starts a Capstone image with `binfmt_misc`. Buildroot
`227fdfa` enables `CONFIG_BINFMT_MISC=y` for QEMU. The VM setup mounts the
filesystem before registering the ELF machine 259 handler, passes literal
magic escapes to the kernel, and fails on a registration error. The `P` flag
preserves the original argv[0]; the launcher distinguishes this invocation
using the kernel's `AT_FLAGS_PRESERVE_ARGV0` auxiliary-vector bit. Kernels
without support remain usable through the explicit launcher and report that
direct image execution is unavailable.

Perl rebuilt on this runtime with patch 0008 now passes all nine `t/base`
files and 493 assertions, including the shell-launched image in `term.t`;
see [the result](../ports/perl/musl/results/2026-09-29/base-tests-binfmt.txt).

musl's libc-test runs against this runtime with
`ports/musl-capstone/libc-test/run-libc-test-delegated.py`, every functional
test built by the SDK driver and run through `capstone-vm run` in one boot.
On 2026-09-29, before the spawn branch: 43 PASS, 6 FAIL, 1 FAULT, 5 NOBUILD,
22 EXCLUDED of 77, the pass count of the HostCall v0 runtime's last run.
`fscanf` passes now that `pipe2` is a Linux pipe; `clocale_mbfuncs` faults
with cause 5 on a locale-table walk, the class the quarantine list records for
`mbc`. The five NOBUILD are the thread-local tests the compiler cannot lower.
With the spawn branch, `popen` and `spawn` leave the excluded set: 44 PASS,
6 FAIL, 2 FAULT, 5 NOBUILD, 20 EXCLUDED. `spawn` passes; `popen`'s child
sends SIGUSR1 to the task, which has no handler installed for the domain yet,
so the task dies of it: that is the signals branch. The remaining exclusions
are threads, sockets, SysV IPC, dynamic loading, `vfork` itself, `wordexp`
and the `fcntl` test's forked child.

### Signals (2026-09-29)

Linux owns every signal decision; the runtime carries a caught signal into the
domain. `rt_sigaction` is delegated with the handler replaced by a class
(default, ignore, caught): a caught signal installs the launcher's trampoline
with `SA_SIGINFO`, the domain's `SA_RESTART`, `SA_RESETHAND`, `SA_NOCLDSTOP`
and `SA_NOCLDWAIT` mirrored, and every signal masked while it records. The
trampoline appends `{seq, signo, siginfo, mask}` to a lock-free ring, bumps a
recorded-sequence word in the META region, keeps the signal blocked in the
kernel until the domain handler is done (unless `SA_NODEFER`), and when the
interrupted pc lies inside the launcher's syscall stub before its `ecall`
redirects it to a `retry` exit, so the round reports `RETRY` with no result and
no output and the libc marshals the call again after the handler. A completed
call reports its result, `-EINTR` included; the kernel's own restart decision
(`SA_RESTART`) lands in the same stub range. `wait4` polling and the spawn
protocol have their own continuation points. The kernel mask the launcher
applies is the domain's mask plus the in-flight signals plus caught-signal
backpressure while the ring is nearly full; nothing is dropped.

The libc runs accepted events after every round, in acceptance order, under
`current ∪ sa_mask ∪ {sig}`; an event accepted inside `rt_sigsuspend` or a
masked `ppoll` runs under that call's temporary mask before the call returns.
Nested delivery happens at the end of a handler's own rounds. `sigaction` on a
signal with runnable events runs them under the old installation first.
`sigaltstack`, `sigsetjmp` with a saved mask and `siglongjmp` are the libc's;
a handler abandoned through `siglongjmp` is acknowledged when the jump is
seen. The recorded-sequence word is checked at every entry into the syscall
dispatcher, so a signal accepted during pure computation runs at the next
libc call, not before: delivery is synchronous. A domain fault stays fatal
whatever the `SIGSEGV` action, `ucontext` carries no register image, and
`timer_create` is not served yet.

`tests/application/signal-contract.c` is the definition of done, one mode per
case of the plan, driven by `run-signals.py`:
26/26 modes pass in the guest (self delivery with and without `SA_NODEFER`, a
signal before and during a blocking read with restart and with `EINTR`, a
handler that writes, `sigsuspend`, `ppoll`, nesting A -> B, `RETRY` after a
partial write, `waitpid(-1)` restarted and interrupted, a spawn interrupted
after its request, `SA_RESETHAND` dying and reinstalled, a realtime queue
through the ring, the hint, `sigaltstack` with a nested `siglongjmp`,
`SIG_IGN` inheritance with `SETSIGDEF`, and a caught `SIGABRT` through musl's
own sigaction path, plus positive-PID wait timing, a 400-signal `SA_NODEFER`
burst, a deep-stack `siglongjmp`, translated `sigtimedwait` status, and
inherited state in a second domain). The native suite (25/25) checks all 400
queued payloads and the spare overflow record. The application gate (21 PASS
lines), the binfmt contract and Perl `t/base` (9 files, 493 tests, 23 s) pass
on the same images; `perl -e` makes 30 rounds instead of 26, the signal
calls Perl makes that the previous runtime answered locally as no-ops.
libc-test with the SDK built from this commit: 48 PASS, 4 FAIL, 2 FAULT, 3 NOBUILD, 20 EXCLUDED of 77; `popen` (SIGUSR1 from its child) and `setjmp` (six `sigprocmask` calls that were no-ops) pass now, `tls_local_exec` fails instead of faulting, the rest is unchanged: `clocale_mbfuncs` faults at startup as before, `mntent`, `strptime` and `strtold` fail as before, the TLS tests do not build.
Record: [20260929-signal-contract.json](tests/application/results/20260929-signal-contract.json).

### Review verification (2026-09-29)

The [checked result](tests/application/results/20260929-delegation-review.json)
records the launcher/image identities and individual libc-test verdicts.
Thirteen additional native cases cover output tails, raw-pointer rejection,
nested iovec offsets, RV64 stat bounds, private descriptors, wait/kill scope,
current cwd/umask, stale inherited descriptors, error-pipe collisions and
explicit FD overflow. All thirteen fail against the original dispatcher/spawner implementation;
all now pass alongside the existing eight native tests with ASan/UBSan.
The host tests pass 17/17 and the runner cancellation tests 2/2.

Fresh guest contracts pass scalar/vector I/O, current process state, image
spawn with a custom argv[0], PATH independent of child envp, exec with existing
children, spawn after exec, exec with closed standard descriptors, and recovery
from a rejected image. The deliberate fault is SIGSEGV and resolves to
`main+0x95c` after verifying the sealed image hash. The unchanged v1 application
gate passes 108 mixed starts in the same boot with stable retained resources.
At that review revision both SDK variants built. The subsequent port migration
removes v1 application support; this historical result remains tied to its hashes.

The fresh delegated libc-test result is **45 PASS, 5 FAIL, 2 FAULT, 5 NOBUILD,
20 EXCLUDED** (77 total). `utime` gains its pass because futimens now carries
utimensat's nullable path. The remaining failing tests are `mntent`, `setjmp`,
`sscanf_long`, `strptime`, and `strtold`; the two signal exits remain
`clocale_mbfuncs` (SIGSEGV) and `popen` (SIGUSR1). A runner timeout now asks the
CLI to cancel and reap the guest and stops the suite if cleanup cannot be
confirmed. This initial review run did not add Perl coverage. The follow-up
below closes its binfmt_misc gate; signal delivery and region-grant memory
remain open. Step 4 is implemented in part, not accepted against all its
original gates.

### Shell execution qualification (2026-09-29)

The [checked result](tests/application/results/20260929-delegation-binfmt.json)
records a kernel built from clean pinned Linux `830b3c68c1fb`, the expanded
Buildroot `227fdfa` QEMU configuration, and the rebuilt Perl/launcher hashes.
The earlier failed snapshot boot is reproducible (`E2BIG` starting init).
That snapshot contained local address-tag changes in `pgtable.h` and `uaccess.h`;
the clean pinned sources boot with the same configuration including binfmt_misc.
The installed known-good Image and the snapshot build-tree Image also had
different hashes; a snapshot directory alone did not identify the booted kernel.

Four new host tests cover mount/registration, all 20 bytes of the ELF match,
native-ELF rejection, stale registration replacement, explicit errors, and
the older-kernel fallback. All 21 host tests pass. Direct guest exec preserves
custom argv[0] (a failing control before the `P`/auxv fix), exit status,
I/O, spawn and in-place exec, including closed standard descriptors and failed
exec recovery. The complete v1 gate passes another 108 starts with stable
retained resources; delegated libc-test remains 45/77 PASS. Fresh Perl `t/base`
passes 9/9 files and all 493 assertions. These are QEMU results, not FPGA results.

Run the direct-exec regression against a provisioned guest with the new kernel:

```sh
python3 capstone/runtime/tests/application/run-binfmt.py --state "$VM_STATE" \
  --image /mnt/host/delegate-contract.dom
```

## Verification

```sh
source capstone/tests/capstone-test-env.sh
cmake -S capstone/runtime/exec -B "$CAPSTONE_TMP_ROOT/application-native" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS='-Wall -Wextra -fsanitize=address,undefined'
cmake --build "$CAPSTONE_TMP_ROOT/application-native"
ctest --test-dir "$CAPSTONE_TMP_ROOT/application-native" --output-on-failure
PYTHONPATH=capstone/runtime/host python3 -m unittest discover -s capstone/runtime/host/tests
```

Build `runtime/tests/application` using the application toolchain to obtain
`application-contract.dom`. The Linux build also supplies `application-supervisor`
and `application-process-test`. Stage those and `perl.dom`, `mruby.dom` on the
existing share, then:

```sh
python3 capstone/runtime/tests/application/run.py \
  --state "$CAPSTONE_TMP_ROOT/dev-vm" --repeat 200 --report acceptance.json
```

For the transferred heap/node-reuse/exhaustion controls, build the same
`contract.c` through `runtime/application` with `CAPSTONE_APPLICATION_HEAP=sublet`,
`CAPSTONE_APPLICATION_HEAP_LOG=20`, name `contract-sublet`, then additionally pass
`--sublet-image /mnt/host/contract-sublet.dom` to the gate. All tests use the
existing boot. The checked report includes platform hashes, before/after counters
and the unchanged Linux boot ID. Two 200,000-cycle churn cases check continued
execution, preservation of live data and rejection of an old reference after
identity reuse. A separate case keeps creating valid ancestors to verify genuine
exhaustion and recovery. Allocation-progress checks reject faults that happen
before the intended threshold. Upstream test failures remain port results;
see [Perl's actual tested subset and limitations](../ports/perl/musl/README.md).

The [2026-09-26 acceptance result](tests/application/results/20260926-qemu-rebased.json)
records 1,008 mixed starts after node exhaustion, with stable pool/node/tag counts.
The [2026-09-27 node-reuse results](tests/application/results/20260927-node-reuse/README.md)
add in-process collection and repeat the lifecycle gate at the same 65,536-node
capacity. They retain the old failing control and separate application reruns.
Four native ASan/UBSan tests and twelve Python tests pass. A subsequent common
gate uses fresh SDK-built Perl and mruby. The [complete Perl `t/base` run](../ports/perl/musl/results/2026-09-26/base-tests-rebased-qemu.txt)
has eight passing files and one failing file (unsupported target subprocess
creation). Legacy CoreMark/shared-region/basic HostCalls pass; null_blk
and borrowed-region open/close failures reproduce on the old platform as well.
