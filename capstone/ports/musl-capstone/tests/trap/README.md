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
and frees an object through mallocng. The write/mapping smoke now runs on
the integrated kernel as recorded below. The malloc smoke was rebuilt and
checked here, but was not rerun in this qualification.

The script retains disassemblies and checks fourteen Linux syscall numbers,
including rt_sigaction, rt_sigprocmask and sigaltstack. It also checks the six MOVC instructions, the ecall, the musl
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
Thread creation uses tagged clone arguments in the step 8 slice. Caught
signal delivery remains unsupported. The trap build
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

On p13 in `~/asr-work/s8t`, all 1,349 libc files rebuilt. On integrated kernel
`2443072aa6b`, `threads-smoke.dom` now exits zero. Trap startup must retain
`__init_tp`'s `libc.can_do_threads` result. Resetting it to zero caused
`pthread_create` to return ENOSYS before clone. The `threads-disabled.dom`
control restores that reset and exits one. `futex-stale.dom` confirms EFAULT
for stale reads and wake-op writes, with successful fresh-reference controls
at the same address. The link and clone disassembly checks also pass.
See the [kernel thread contract and controls](../../../../kernel/tests/threads/README.md).

## Jump buffers and signal registration

The trap build keeps ra, sp and s0 to s11 in fourteen tagged STC/LDC slots.
`sigsetjmp` saves the mask with Linux rt_sigprocmask (135 in a7), and
`siglongjmp` restores it when savemask was nonzero. `longjmp` does not change
the mask. Neither jump path calls the launcher save or reap hooks.
The SDK defines CAPSTONE_RUNTIME_TRAP for assembly as well as C.

Registration uses ordinary Linux rt_sigaction and sigaltstack calls. Their
structures contain 8-byte addresses, so the trap dispatcher translates the
libc's 16-byte pointer fields before calling the kernel. RISC-V's native
sigaction layout has no restorer field. Handler addresses use ordinary jalr
in this ABI. A thread-local reference preserves each thread's registered alternate stack
when it is queried.
No launcher signal metadata or signals.c object is linked.
Signal masks go directly through rt_sigprocmask without translation.
The kernel must refuse caught delivery and kill the task until tagged signal
frames are implemented. These changes do not implement caught delivery.

`setjmp-smoke.dom` jumps across two frames after clearing a reference in s2.
It checks s2 with LCC and reads the referenced integer after both longjmp and
siglongjmp. It also checks the zero-to-one return rule, saved-mask restoration
and the savemask-zero case. `signal-smoke.dom` installs and queries SIGINT,
uses signal(), blocks and restores the mask, and registers, queries and
restores an alternate stack. It never delivers a signal.

Both programs link and pass disassembly checks on p13. The rebuilt CMake SDK
also links the sigsetjmp probe. To reproduce the link-failure control, extract
runtime/setjmp.S from bfb7cd559bb and set TRAP_CONTROL_SETJMP to that file.
The build requires that control to fail on both missing launcher hooks.
Artifacts are under ~/asr-work/s5l/signal-smoke and sdk-setjmp.dom.

## Integrated execution on 14 November 2026

Branch `trap/s5-libc-int` combines the jump and signal changes with base
`2443072aa6b`, including its exec stack contract and thread runtime. On p13,
the existing LLVM installation rebuilt all 1,349 libc files with zero failures.
All five smoke images linked and passed disassembly checks, including the
thread clone checks. The startup-magic and missing-launcher-hook controls
still passed. The installed toolchain's source freshness remains unverified.

The guest used these inputs:

- Kernel: `~/asr-work/eval/kernel/Image-2443072aa6b`.
- QEMU: `~/asr-work/q1b/bin/qemu-1b`, one hart, sstc off.
- Firmware: `~/asr-work/del-fw/out/f-keep/fw_jump.elf`.
- Kernel parameter: `sublet.dom_binfmt=1`.
- Initramfs: the busybox base at `~/asr-work/s5k/base-initramfs.cpio.gz`,
  packed with the repository's `kernel/tests/libc-smoke/run.sh`. It uses the
  extraction and repacking procedure from `kernel/tests/first-program/run.sh`.

| Program | Output | Exit |
| --- | --- | --- |
| `setjmp-smoke.dom` | `setjmp ok` | 0 |
| `signal-smoke.dom` | `signal ok` | 0 |
| `smoke.dom` | `ok` | 7 |

The jump probe checks the restored reference's tag with LCC before reading
through it. The signal probe checks registration and masks without delivering
a signal. The existing integration start contract and thread sources are
retained. The alternate-stack reference is now thread-local because the
combined runtime supports threads. Thread execution and caught signal
delivery remain outside this qualification.

Two short execution controls use the same libc archive and kernel. Replacing
s2's jump-buffer STC stores with SD makes the jump probe exit 1. Linking the
signal probe with the integration base's dispatcher, which refuses signal
registration, makes it exit 1. Ordinary busybox commands still exit 0 after
both positive and control runs. Neither run reports a kernel panic or a
Sublet fault.

Build output is in `~/asr-work/s5l/build-integrated.log`. Binaries,
disassemblies and serial logs are in `~/asr-work/s5l/integrated`, with
`integrated.log` for the positive run and `controls.log` for the controls.
The boot wrapper printed EOF because its pattern did not match doubled
carriage returns. The verdicts above were checked directly in the serial
logs. The base initramfs also lacked bare head and sed commands used by the
wrapper's final dmesg filter. The complete serial logs were checked for faults.
The earlier loader rejection remains recorded in `~/asr-work/s5l/signal-run`.

The syscall boundary assumes the step-3 register slots preserve references
across ordinary ecalls. Native nested signal structures require the
translations above, not a capability-layout kernel sigaction structure.
No additional plan-v5 interface change is introduced.

## CDERIVE failure in mallocng

When the lifetime table has no free slot and the kernel cannot grow it, the
kernel sets NOGROW and the resumed CDERIVE writes an untagged 0. The libc
treats that result like its own out-of-memory case. malloc and aligned_alloc
give the raw slot back with `__malloc_unissue`, which is mallocng's free
without a lifetime to end, and return NULL with ENOMEM. realloc creates the
new object before it ends the old one. An in-place resize derives the new
object first and returns NULL if that fails, leaving the old object live. A
resize that would move a mapping goes through malloc, memcpy and free. The
internal `__heap_remapping` is gone.

`malloc-fail.c` checks this with a libc compiled with
`-DCAPSTONE_HEAP_DERIVE_INJECTION=1`, where the test makes the next derive
fail. Set `TRAP_LIBC_INJECTION_ARCHIVE` to that archive and the build links
`mf-1.dom` to `mf-6.dom`. They cover a small malloc, slot and mapping leaks over
repeated failures, and realloc in place, on a mapping, and with a move.
`stale-free.dom` reads an object after free and must die with cause 25.

On 14 November 2026, on p13 under `~/asr-work/libc-fail` with kernel
`Image-0133b9e21cd`, all six cases exited 0. stale-free faulted with cause 25,
and smoke, malloc-smoke, threads-smoke and setjmp-smoke passed. In the control,
the libc of `0133b9e21cd` was built with the same injection hook. It returned
NULL with errno 0 and leaked slots (a 262 KB address span) and mappings (3,152
pages). Each of the three failed reallocs ended the old object, so the next
read of that object faulted with cause 25.
