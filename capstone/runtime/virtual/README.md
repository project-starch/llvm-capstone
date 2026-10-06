# Virtual C applications on Linux

This adapter runs Capstone applications in **C mode with user virtual
addresses**. The existing Linux kernel schedules the launcher, owns its page
tables and supplies its memory and system calls. A loadable module connects
Linux to the QEMU execution interface. No additional kernel-core or firmware
patch is required for the recorded platform.

The [application gate](result.json) runs a normal C program, the explicit
same-`mm` thread contract, SQLite's shell and the mruby interpreter. The
existing ports build through the common SDK; applications must be rebuilt for
its virtual profile. An ELF marker prevents the loader from accepting an image
built for the physical calling convention.

## Execution and memory

```text
Linux launcher: ELF loader + existing delegated services
        | ioctl enter / resume / register / retire
loadable module: owning mm, pinned frame, private lifetime table
        | CSRUNV                         ^ service / fault / quantum
virtual C application: normal main, capability libc, growing allocator
        | bounds + permissions + live node
        | owning process's page tables, user PTE permissions
Linux pages, physically scattered; physical capability tags
```

The application retains a complete execution context across service calls.
A service reply resumes after ECALL; a resolved page fault retries the exact
faulting instruction. Scalar replies replace a0 with an untagged value. New
mapping replies transfer a linear capability through a consumed frame slot.
Quanta return to Linux even if the application never calls the runtime.

Virtual threads use the same Linux `mm`, page tables and lifetime table. Each
thread has its own QEMU supervisor slot, PCC, registers, stack and optional TLS
pointer. `capstone_virtual_thread_create` supplies an explicit start frame;
the trusted module supplies `satp` and `srevroot`, forces protected-U options,
and one Linux worker task drives each slot. `capstone_virtual_thread_exit`
discards only the calling slot and returns its frame mapping to the trusted
launcher. `capstone_virtual_thread_join` waits for that exit protocol before
the caller retires the child stack or TLS mapping. The workers share
the process transport under a wire lock; fork, shared tagged pages and SMP
remain out of scope.

The loader registers image, stack, startup and exchange mappings. Startup
pages are pinned. Growing RW arenas initially have absent PTEs: the module
asks Linux's GUP interface to resolve the first touch, then retains a pin
until retirement. It does not supply a pager or change VMA permissions to
make a denied access succeed. TLS uses the common capability libc setup.

Each launch owns a 65,536-slot lifetime table including the invalid header
identity, occupying 1 MiB. Linux keeps the arena's private ancestor as a
scalar identity. Whole-arena retirement walks and invalidates its descendants
and the ancestor before clearing physical tags and unpinning pages. Reusing
the same VA creates a fresh live identity. When the table is pressured, the
trusted collector clears stale tags in registered pages and saved contexts,
then returns unpinned identities to the table free list; dead PCC identities
remain pinned because a PCC has no tag to clear. Teardown discards the context
and clears tags before freeing its table and saved frame.

The allocator uses existing Sublet operations for object lifetimes. It grows
through Linux mappings and supports `malloc`, `calloc`, `realloc`, alignment
and whole-free-arena `malloc_trim`. It checks the remaining node budget
before allocation. Blocks are aligned powers of two, minimum 256 bytes. The
returned capability is bounded to the request; a request of 4 KiB or more is
rounded up to the representable grain, at most 1/512 of its size.
`malloc_usable_size` reports that bounded extent, not the block.
Anonymous RW `mmap` and whole-arena `munmap` use the same grant/retire path.
The collector is only enabled after a complete namespace sweep. If all
remaining identities are live or pinned, the context ends with a resource
fault (cause 30) instead of guessing that a stale capability is gone.

The compiler profile combines `-capstone-gp-free` calls within PCC with
`-capstone-image-gp`: a representable readable image capability supplies
globals and anonymous constant pools. It permits those pools only for this
profile; the bounded capability-table ABI retains its existing refusal.
Virtual C cannot fabricate a missing gp. The shared setjmp and signal-stack
assembly follows the profile's scalar return-address convention.

## Build and run

Use a compiler built from this branch, including `clang`, `lld`, `llvm-ar`
and the usual Capstone test tools. All generated inputs and outputs belong
outside the source tree. The kernel build must match the Image being booted.

```sh
export CAPSTONE_LLVM_BUILD_DIR=/path/to/llvm-build
source capstone/tests/capstone-test-env.sh
export KERNEL_BUILD=/path/to/prepared-linux-build
export CROSS_COMPILE=/path/to/riscv64-linux-gnu-
out="$CAPSTONE_TMP_ROOT/virtual-runtime"
export MUSL_CACHE_ROOT="$out/musl-source"
musl=$(bash capstone/ports/musl-capstone/prepare-musl-capstone.sh | tail -1)
bash capstone/runtime/virtual/build-sdk.sh "$out" "$musl"
bash capstone/runtime/virtual/build-adapter.sh "$out/adapter"
"$out/sdk/capstone-cc" -O1 capstone/runtime/virtual/contract.c -o "$out/contract.dom"

# Existing application ports, using this SDK:
bash capstone/ports/sqlite/app/prepare-sources.sh "$out/sqlite-source"
bash capstone/ports/sqlite/app/build-domain.sh "$out/sdk" "$out/sqlite-source" "$out/sqlite"
MRBD_ROOT="$out/mruby" MRBD_SDK="$out/sdk" MRBD_PIN=4.0.0-rc2 \
  bash capstone/ports/mruby/app/build-mruby-domain.sh

python3 capstone/runtime/virtual/run.py \
  --qemu capstone/capstone-qemu/build/qemu-system-riscv64 \
  --images /path/to/images --adapter "$out/adapter" \
  --application "$out/contract.dom" --sqlite "$out/sqlite/sqlite3.dom" \
  --mruby "$out/mruby/src/mruby/build/capstone/bin/mruby" --record "$out/result.json"
```

The runner uses one hart and
`rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true`.
Inside that guest, load `capstone_vm.ko` and invoke
`capstone-vexec PROGRAM.dom [ARG...]`. The device is root-only. Set
`CAPSTONE_VM_TRACE=1` to report delegated syscall numbers and results.
The launcher reports node use, pinned pages, service/copy counters, and guest
monotonic launch/elapsed times. These QEMU times are not processor benchmarks.

## Qualification and limits

The gate requires argv/environment, TLS, longjmp, file operations including
capability-valued lock arguments, 1.5 MiB of heap growth and release, actual
Linux demand faults, two independent processes, a child virtual context sharing
the process `mm`, termination of a non-yielding program, and module unload.
Stale/free, bounds and retired-then-reused-VA
accesses must fail. SQLite writes a database and another process reopens it;
mruby performs arithmetic and file I/O. The `--omit-application` negative
control must fail, even though the shell and other applications still run.

The recycling evidence uses the existing application contract
(`runtime/tests/application/contract.c`), built with this SDK and run through
`capstone_vm --profile virtual`: `churn` performs 200,000 malloc/free cycles
beside a held block, `fault-reused` then dereferences the first freed pointer
(cause 24), and `fault-exhaust` keeps creating unrevoked ancestors until the
namespace is exhausted (cause 30 after a collection that frees nothing). The
guest supervisor of `runtime/tests/application` drives each mode, and the
launcher's `--stats` counters read before and after it give the allocations,
collections and reclaimed identities of that mode; the
[qualification record](qualification.json) keeps them with the command, as
well as the process, signal and socket fixtures run on the same images.

The [instruction gate](../../capstone-qemu/tests/virtual-capstone-runtime/README.md)
also checks consuming LDC/STC fault retries, linear replies, immutable context
roots and restoration of the physical caller. The
[qualification record](qualification.json) includes compiler, regression and
negative-control results. Existing physical applications remain a regression
gate; this adapter does not replace their monitor or CALL/RETURN path.

This is the first QEMU application profile: one hart, private anonymous
mappings, 256 MiB per registered region, 1 GiB
aggregate registered VA, 512 mappings, and a bounded recyclable node table.
Fork, shared tagged mappings, file-backed
mmap, partial unmapping, swap, migration, application register editing by
signals and POSIX thread synchronization need further contracts. Anonymous
`MAP_SHARED` and SysV segments are process-local compatibility for nested
ports, since there is no fork; every other mapping form fails explicitly.

Pages stay pinned after first touch. A page that Linux populated without an
application fault is pinned by the next retirement or collection, so a
teardown does not clear its tags itself; Linux clears them with ordinary
stores when it zeroes the page for reuse. The initial ELF image has the
existing combined code/data layout. Representability padding above 4 KiB
requests is accessible within its own block. Recycling requires the
registered-page and saved-context sweep; a namespace full of live or pinned
identities reports resource exhaustion.
Compact-bounds conformance of every existing compiler/helper path
remains separate work: the inherited QEMU data-tag implementation still
retains extra bounds metadata. The gates establish the stated behavior in
QEMU; they do not establish an RTL implementation or a complete memory-safety
proof for all programs.
