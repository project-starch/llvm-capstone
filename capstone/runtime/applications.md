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
The pinned submodule revisions are part of the implementation. The Buildroot
chain uses published OpenSBI/monitor forks (recorded in `.gitmodules`) because
this lane lacks upstream write access. Recursive checkout fetches the tested
commits. Existing FPGA hardware does not implement this extension; no FPGA
result is claimed here.

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
cannot consume the emergency reserve required by monitor cleanup; node exhaustion
faults the application and reclamation makes the next launch possible. This is a
one-hart VM implementation, not a silicon reclamation or performance claim.

The musl port's existing syscall coverage still applies: launching a Linux
process does not add target fork, exec, threads, dynamic loading or full POSIX
fd semantics inside a domain. Standard-stream read/write/EOF/close/stat/query
fcntl use Linux objects; other file operations retain the existing host service.
The launcher uses its Linux filesystem authority and is not a filesystem sandbox.
No claim of a complete hostile-code or QEMU security audit is made.

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

For the transferred heap/stale-reference/exhaustion controls, build the same
`contract.c` through `runtime/application` with `CAPSTONE_APPLICATION_HEAP=sublet`,
`CAPSTONE_APPLICATION_HEAP_LOG=20`, name `contract-sublet`, then additionally pass
`--sublet-image /mnt/host/contract-sublet.dom` to the gate. All tests use the
existing boot. The checked report includes platform hashes, before/after counters
and the unchanged Linux boot ID. Upstream test failures remain port results;
see [Perl's actual tested subset and limitations](../ports/perl/musl/README.md).

The [2026-09-26 acceptance result](tests/application/results/20260926-qemu.json)
records 1,008 mixed starts after node exhaustion, with stable pool/node/tag counts.
Four native ASan/UBSan tests and twelve Python tests pass. A subsequent common
gate uses fresh SDK-built Perl and mruby. The [complete Perl `t/base` run](../ports/perl/musl/results/2026-09-26/base-tests-fixed.txt)
has eight passing files and one failing file (unsupported target subprocess
creation). Legacy CoreMark/shared-region/basic HostCalls pass; null_blk
and borrowed-region open/close failures reproduce on the old platform as well.
