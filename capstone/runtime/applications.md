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
`S40capstone` loads the driver, checks its process ABI and optionally mounts the
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

Perl's build recipe now uses this SDK; its private compiler wrapper, entry
adapter and VM runner were removed. Perl and mruby objects were also linked
through the identical SDK driver and executed successfully. Other ports can
adopt either build interface while retaining their upstream patches and build
recipes. Historical hardware gates and loaders for older ABIs remain explicit.

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
640 MiB CMA, `CAPSTONE_GP_NONLIN=1` and 65,536 revocation nodes. Explicit supported
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

An application built with the default `CAPSTONE_APPLICATION_DELEGATE=ON` speaks
ABI v2: every Linux service is the Linux syscall itself, run by the launcher
task. The domain fills an 88-byte entry in the entry region, copies pointer
arguments into the exchange region as offsets, and yields; the launcher
validates the entry against the shape table, runs `syscall()` under its own
credentials, descriptors and working directory, and writes the result back.
The wire ABI, the shape table and the closed exception groups are in
`include/capstone/delegate.h`; the design is
[docs/plans/delegation-abi.md](../docs/plans/delegation-abi.md).

What crosses: files, directories, descriptors, time, identity, limits,
`getrandom`, `wait4`, `kill` confined to the task, `exit_group`. What does not:
memory (`mmap` is the domain allocator's, file `mmap` is ENOSYS), processes
(`clone` and `fork` are ENOSYS; image exec uses the process service below), and
signals (`rt_sigaction` and `rt_sigprocmask` are accepted and recorded as
no-ops until the signals branch). The unserved report at exit lists both.

The image declares the exchange region with `EXCHANGE_BYTES` (default 256 KiB,
`CAPSTONE_APPLICATION_EXCHANGE_BYTES` for the SDK project); larger buffers are
chunked, so a big read or write is a short one. A v2 image's descriptor is 48
bytes; `capstone-exec` accepts v1 and v2 images and keeps HostCall v0 for the
former. Building with `CAPSTONE_APPLICATION_DELEGATE=OFF` produces a v1 image.

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

That ratio is the emulator's: each round crosses U, S and M mode twice and
QEMU flushes its TLB on every supervised switch. This run did not use `icount`; it is a wall-time observation affected by host
scheduling, not a hardware cost or completion of the planned per-step cycle
measurement. It predates the review corrections below.

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
stopped/continued children. `kill` accepts this task or a recorded child,
not process-group targets. A blocking any-child wait polls recorded PIDs at
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
Both v1 and v2 SDK builds succeed.

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

The libc heap qualification runs the heap cases of `contract.c` once on the
`HEAP=sublet` image and once on a `HEAP=level0` image of the same source,
which is the control: every `fault-*` case must be a SIGSEGV on the first and
must complete on the second, every `heap-*` case must complete on both.

```sh
python3 capstone/runtime/tests/application/run-heap.py \
  --state "$CAPSTONE_TMP_ROOT/dev-vm" \
  --sublet-image /mnt/host/contract-sublet.dom --control-image /mnt/host/application-contract.dom \
  --sublet-elf <build>/contract-sublet.dom --control-elf <build>/application-contract.dom \
  --platform <kernel> <firmware> <rootfs> <qemu> <launcher> --report heap.json
```

It needs the emulator the tree pins (in-process node reuse): on the base
emulator the 200,000-cycle churn case exhausts the node pool after about
65,000 allocations, on any image. The
[2026-09-30 record](tests/application/results/20260930-heap-qualification.json)
is the first run; the plan is
[capstone-heap-protection.md](../docs/plans/capstone-heap-protection.md).

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
