# Linux trap libc

The virtual libc now starts from the exec stack and calls musl's
`__libc_start_main`. Its syscall boundary moves six arguments with MOVC,
puts the Linux number in a7 and executes ecall. Mapping results return through
a pointer-typed entry. No exchange region is used by this path.

Build on a host with the Capstone LLVM tools:

```sh
source capstone/tests/capstone-test-env.sh
bash capstone/ports/musl-capstone/build-trap-smoke.sh "$CAPSTONE_TMP_ROOT/trap-smoke"
```

Set `CAPSTONE_LLVM_BIN` before sourcing the environment when the tools are
outside the default build. An existing libc archive can be supplied with
`TRAP_LIBC_ARCHIVE`. The script otherwise calls `build-musl-capstone.sh`.
The shared CMake SDK builds the same runtime. Virtual SDKs now declare
application ABI 3. Physical SDKs retain ABI 2.

`smoke.dom` writes `ok`, maps a page, writes and reads through the returned
reference, unmaps it and returns 7. `malloc-smoke.dom` allocates, accesses
and frees an object through mallocng. These programs have been linked and
inspected, but have not been run. Execution needs the kernel's trap loader.

The script retains disassemblies and checks the eleven Linux numbers used by
read, write, openat, close, mmap, munmap, mprotect, exit, exit_group, getpid and
clock_gettime. It also checks the six MOVC instructions, the ecall, the musl
startup symbols, absence of launcher stubs and absence of the image handshake.

For the negative control, extract `runtime/virtual/start.S` and `vm-abi.h`
from baseline `ac80a6cdf11` into a scratch directory. Set `TRAP_CONTROL_START`
to that copy of `start.S`. The script compiles it and requires its
`.capstone_virtual` section to contain `CV_IMAGE_MAGIC`. The new linked images
must have no such section.

## Verified on 14 November 2026

On p13, under `~/asr-work/s5l`:

- `build-musl-capstone.sh` compiled 1,349 of 1,349 files with zero failures.
- Both static smoke programs linked with no undefined symbols.
- All eleven number checks and the tag-preserving boundary check passed.
- The baseline control contained `43504f4e 56564d37`, the image magic in memory.
- The CMake SDK built, and its generated `capstone-cc` linked `smoke.c`.

The tools were the existing LLVM installation at `~/lanes/gatefix/llvm/bin`.
Clang reported version `22.0.0git`. Its installation has no `build.ninja`, so
the environment's toolchain freshness check could not verify source freshness.
The successful builds establish compatibility with that installed toolchain.

## Loader and syscall contract

The kernel supplies tagged sp and gp. gp covers the image, including its data
and TLS template. No grants arrive in a0 to a3. The startup code does not read
those registers as launch arguments.

The stack layout follows the 16-byte pointer ABI. The concrete definition is
in `runtime/include/capstone/linux-start.h`:

| Item | Layout |
| --- | --- |
| argc | Low 8 bytes of the first 16-byte stack cell |
| argv | 16-byte tagged pointer cells, followed by a zero cell |
| envp | 16-byte tagged pointer cells, followed by a zero cell |
| auxv | 32-byte entries, type at offset 0 and value at offset 16 |

Scalar auxv values occupy the low word of the value cell. Pointer values,
including AT_RANDOM and AT_EXECFN, retain their tags. AT_NULL terminates auxv.
AT_PAGESZ must be 4096. Startup preserves AT_RANDOM for the stack protector and
mallocng. TLS is initialized from the static image's linker symbols.
The legacy `getauxval` interface remains scalar and does not return references.

Syscall entry receives arguments in a0 to a5 and the number in a7. The kernel
must preserve their tags in the task's saved slots. A successful mmap returns
a tagged depth-1 block in a0. Errors and scalar results return untagged.
munmap and mprotect receive the original reference, never its integer address.
The mallocng mapping index retains the depth-1 block, so whole-block munmap
can revoke it. Object references remain children of that block.

## Scope relative to plan v5

The explicit task instruction replaces the plan's number-in-a6 and selector-in-a7
convention with ordinary Linux numbering in a7. Compiler-generated pointer-global
initializers remain necessary for this toolchain. They run before libc startup
without consuming launcher grants or performing an image handshake.

Mallocng already obtains its metadata through mappings. Its payload mappings
now use real mmap and munmap too. The trap syscall wrapper refuses brk.
Thread creation now uses tagged clone arguments in the step 8 slice. Signal
registration remains unsupported. The trap build
excludes the launcher thread and signal transports. Epoll operations and the
signal operations that need opaque-reference translation return ENOSYS until
the side tables from plan v5 are implemented. Execve also remains deferred.
The kernel's syscall binder, nested-pointer loaders, mapping enforcement and
runtime fault controls are integration work. No kernel files changed here.

## Thread build slice

The thread runtime issues Linux clone (220) with CLONE_VM and CLONE_THREAD.
The child stack carries tagged entry and argument cells. Clone passes the
stack in a1, parent tid in a2, TLS in a3 and clear-child-tid in a4. The kernel
must restore child sp and tp from these references and return integer zero
in the child's a0 slot. The parent receives the Linux tid.

Musl owns TLS, pthread objects, join and mutex state. Its futex calls retain
the original references through the ordinary syscall wrapper. Signal masks
use rt_sigprocmask, which needs no opaque-reference conversion. Caught
signals and cancellation remain deferred. Thread exit uses SYS_exit (93).
The assembly __unmapself performs munmap then exit without another stack
or TLS access.

On p13 in `~/asr-work/s8t`, all 1,349 libc files compiled. The two-thread
`threads-smoke.dom` linked without undefined symbols and passed the clone
disassembly checks. The existing write/mapping and malloc smokes still link.
The three images have not run against the integrated kernel.
See the [kernel thread contract and controls](../../../../kernel/tests/threads/README.md).
