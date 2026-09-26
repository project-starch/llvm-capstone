# Applications in a persistent Linux guest

`capstone-exec PROGRAM [ARG...]` runs a musl Capstone application as a Linux
child process. It passes argv, environment and cwd before constructors, uses
the launcher's stdin/stdout/stderr, and returns the application's exit status.
A cooperating application's capability fault terminates the launcher with
actual `SIGSEGV`. The parent Linux shell remains usable.

The shared CRT calls ordinary `main(argc, argv)`. No application names or
argument files occur in the launcher. The startup wire format is versioned,
bounded and pointer-free; capabilities for argv and environ are constructed
inside the domain. The loader validates an immutable, sealed snapshot of the
ELF before creating a domain. Images must declare application ABI v1.

## Current scope

This is the bounded application milestone of the
[domain process plan](../docs/plans/domain-process-runtime.md). It requires the
QEMU domain trap-delivery change (`77d69353b7`) and libcapstone commit `8ba1b7f`
(`capstone_set_verbose`, `capstone_call`, `capstone_share`). Legacy
domains retain their existing entry paths and loader diagnostics.

**Domain destruction and trusted interruption are still missing.** Closing the
launcher releases its Linux mappings and file handles, but the current driver
and monitor retain the domain memory and region/domain slots. Runs within one
boot remain limited by those resources. `timeout` or Ctrl-C cannot reliably
interrupt a domain that never yields. Do not use this as an unattended,
unbounded test service. No FPGA fault-recovery claim is made.

The existing musl syscall coverage remains in force: externally launching a
program does not add target fork, exec, threads or full POSIX descriptor
semantics. Standard descriptor reads/writes, EOF, fstat, close and query-only
fcntl use Linux objects; target pipes and higher file descriptors retain the
existing musl host service behavior.

## Build the guest launcher

Select an existing compiler build, prepared Buildroot output and the matching
Buildroot source checkout. Keep builds outside the repository.

```sh
source capstone/tests/capstone-test-env.sh
cmake -S capstone/runtime/exec -B "$CAPSTONE_TMP_ROOT/application-linux" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/capstone/ports/common/cmake/toolchains/linux-guest.cmake" \
  -DCAPSTONE_LIBCAPSTONE_DIR="$CAPSTONE_RUNTIME_BUILDROOT/package/modcapstone/userspace/lib"
cmake --build "$CAPSTONE_TMP_ROOT/application-linux"
```

`CAPSTONE_BUILDROOT_DIR` supplies the cross compiler under `build/host/bin`.
`CAPSTONE_RUNTIME_BUILDROOT` above identifies the source checkout with the new
library APIs; these can be different directories. A matching driver with CMA
domain allocation is needed for large interpreters.

## Build an application

Existing CMake ports can add `capstone/runtime` and call:

```cmake
add_executable(program main.c)
capstone_configure_application(program
  DATA_BYTES 4194304 STACK_BYTES 1048576 ARENA_BYTES 4194304)
```

Select the existing `capstone-domain.cmake` toolchain,
`PORT_HEADER_PROVIDER=musl`, `PORT_C11_ATOMICS=ON`, `PORT_MUSL_ROOT` and
`CAPSTONE_MUSL_ARCHIVE`. Enable C and ASM in the project. Data includes globals
and stack; the function adds the recovery reservation. Arena size configures
the existing level0 allocator. Other allocator policies remain future work.

For a port already built into objects and archives, the standalone project
links those against the same CRT:

```sh
source capstone/tests/capstone-test-env.sh
cmake -S capstone/runtime/application -B "$CAPSTONE_TMP_ROOT/application-domain" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/capstone/ports/common/cmake/toolchains/capstone-domain.cmake" \
  -DPORT_HEADER_PROVIDER=musl -DPORT_C11_ATOMICS=ON \
  -DPORT_MUSL_ROOT="$MUSL_SOURCE" -DCAPSTONE_MUSL_ARCHIVE="$MUSL_ARCHIVE" \
  -DCAPSTONE_APPLICATION_NAME=program \
  "-DCAPSTONE_APPLICATION_INPUTS=$PROGRAM_MAIN_OBJECT;$PROGRAM_ARCHIVE" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE=-O1
cmake --build "$CAPSTONE_TMP_ROOT/application-domain"
```

Inputs can contain ordinary C sources, objects and archives, in link order.
Include the application's main object and libraries, excluding its old
`domain_entry`, heap and port runtime objects. The output is `program.dom`.
The defaults for `CAPSTONE_APPLICATION_{DATA,STACK,ARENA}_BYTES` can be overridden.
This was exercised with the existing Perl 5.36.3 and mruby application objects.
Their source preparation and target patches remain the ports' responsibility.

## Start one VM and reuse it

Install the single host package into a virtual environment:

```sh
python3 -m venv "$CAPSTONE_TMP_ROOT/application-tools"
"$CAPSTONE_TMP_ROOT/application-tools/bin/pip" install ./capstone/runtime/host
export PATH="$CAPSTONE_TMP_ROOT/application-tools/bin:$PATH"
```

The QEMU build must enable user networking (`--enable-slirp`). Supply a guest
Dropbear multi-call binary containing `dropbear` and `dropbearkey`, built by
the same Buildroot toolchain. This development provisioning path installs
tools into the guest's disposable rootfs snapshot; it does not rebuild or edit
the base image. Native Buildroot package integration remains a later milestone.

```sh
export CAPSTONE_GP_NONLIN=1 CAPSTONE_REV_NODES=8388608
capstone-vm --state "$CAPSTONE_TMP_ROOT/dev-vm" up \
  --qemu "$CAPSTONE_QEMU_BINARY" \
  --kernel "$CAPSTONE_BUILDROOT_DIR/build/images/Image" \
  --firmware "$CAPSTONE_BUILDROOT_DIR/build/images/fw_jump.elf" \
  --rootfs "$CAPSTONE_BUILDROOT_DIR/build/images/rootfs.ext2" \
  --share "$APPLICATION_SHARE" \
  --launcher "$CAPSTONE_TMP_ROOT/application-linux/capstone-exec" \
  --module "$CAPSTONE_MODULE" --ssh-server "$DROPBEAR_MULTI"

capstone-vm --state "$CAPSTONE_TMP_ROOT/dev-vm" shell
```

Inside the ordinary Linux shell:

```sh
capstone-exec /mnt/host/perl.dom -e 'print "hello\n"'
printf 'input\n' | capstone-exec /mnt/host/perl.dom -e 'print uc(<STDIN>)'
capstone-exec /mnt/host/mruby.dom -e 'puts 6 * 7'
```

From another host terminal, `capstone-vm … run /mnt/host/program.dom ARGS…`
invokes the same launcher; `exec COMMAND ARGS…` invokes any Linux program.
Arguments are quoted once using POSIX shell quoting, including empty arguments
and newlines. SSH carries the streams and status. SSH alone cannot distinguish
exit 139 from signal death: use a Linux waitpid harness when that distinction
matters, as the acceptance test does.

`status` checks QMP; `down` stops the guest and discards its temporary disk
changes. The shared host directory keeps its changes. A second `up` reuses the
VM only when paths, binary hashes, memory, share and Capstone emulator options
match. Configuration mismatch requires explicit `down`; commands never restart
a VM or retry an application implicitly. Changing files in the development
share is allowed. Rebuild application images to a temporary filename and rename
them into place, so concurrent starts see a complete image.

State directories are private. SSH listens only on host loopback, uses a private
per-session client key and pins the host key obtained through provisioning.
QEMU keeps the repository's common QEMU lock for its lifetime. Console and
emulator diagnostics go to `console.log` and `qemu.log` in the state directory,
separate from application output. Leaving the CLI or SSH session leaves the VM
running until `down`.

## Verification

Native validation, including malformed startup/ELF inputs, immutable images and
pipe/descriptor behavior:

```sh
source capstone/tests/capstone-test-env.sh
cmake -S capstone/runtime/exec -B "$CAPSTONE_TMP_ROOT/application-native" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS='-Wall -Wextra -fsanitize=address,undefined'
cmake --build "$CAPSTONE_TMP_ROOT/application-native"
ctest --test-dir "$CAPSTONE_TMP_ROOT/application-native" --output-on-failure
PYTHONPATH=capstone/runtime/host python3 -m unittest discover -s capstone/runtime/host/tests
```

Build `runtime/tests/application` with the application toolchain/settings above
to obtain `application-contract.dom`. The launcher build also provides
`application-supervisor`. Stage these and `perl.dom`, `mruby.dom` on the share.
On an already running guest:

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/runtime/tests/application/run.py --state "$CAPSTONE_TMP_ROOT/dev-vm"
```

The gate checks a healthy run, a fault after HostCalls, a fault with destroyed
SP/GP, normal exit 139 and a healthy final run with actual waitpid results.
It then exercises Perl arguments and stdin, mruby execution, and an unchanged
Linux boot ID across all commands. It never boots a VM itself. These tests do
not replace upstream application suites or establish domain resource reuse.
