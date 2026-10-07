# Virtual C applications on Linux

Current review integration: [pthread checks](results/review-r3-pthreads.json)
pass 9/9 with freshly built libc, SDK and adapter. The
[v2 compatibility gate](results/review-r3-v2-compatibility.json) passes 39/39,
with its two long recycling cases explicitly omitted. Earlier result files
below retain the identities of the historical runs they describe.


Review integration: checked-in qualification JSON files identify historical
binaries by hash. They do not qualify this extracted stack on current dev.
The [integrated R1 result](results/review-r1.json) passes 22/22 checks, including
SQLite/mruby relinked against the fresh SDK; its build scope is explicit.
The adapter uses trusted Linux and one hart. POSIX pthreads are introduced in
a dependent change; this stage supports the explicit virtual-context API.


This adapter runs Capstone applications in **C mode with user virtual
addresses**. The existing Linux kernel schedules the launcher, owns its page
tables and supplies its memory and system calls. A loadable module connects
Linux to the QEMU execution interface. No additional kernel-core or firmware
patch is required for the recorded platform.

The [application gate](result.json) runs a normal C program, the explicit
same-`mm` thread contract, SQLite's shell and the mruby interpreter. The
existing ports build through the common SDK; applications must be rebuilt for
its virtual profile. An ELF marker prevents the loader from accepting an image
built for the physical calling convention. VM-service version 3 adds pthread
services and accepts version-2 images, whose mapping ABI is unchanged.
Version-1 virtual images are rejected. Older launchers reject version-3 images;
rebuild the SDK and application together for the pthread profile.

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

The virtual musl bridge also supplies `pthread_create`, join, TLS, mutexes and
condition variables. musl owns thread objects, stack layout and join protocol;
the runtime replaces clone, futex transport and final thread exit. Each POSIX
worker installs its own META/exchange pair, bounce buffer, logical signal mask
and event ring. Linux executes blocking I/O without holding the process service
lock, and futexes use the original registered VA rather than an exchange copy.
The child clears musl's exit word only after its virtual CPU and saved frame
have stopped, so join can safely retire its stack. Process exit and faults stop
all Linux workers before mappings are released by the adapter's fd teardown.

The [pthread gate](pthread-result.json) checks separate TLS, tagged join results,
mutex/condition synchronization, timeout, independent blocking I/O, private
epoll event conversion and 64
successive joined lifetimes, plus preempted I/O through a shared-TLS explicit
context. Wire ownership uses the trusted context ID, not the TLS address.
Compile `pthread-contract.c` with the SDK and `-I capstone/runtime/include`,
then run `run-pthreads.py` with adapter, application, QEMU, images and a fresh
work directory. Cancellation, arbitrary `pthread_kill`, main-thread-only
`pthread_exit` and full POSIX signal/register semantics are not qualified by
this gate. The processor interface and kernel module are unchanged by this
pthread bridge.

The [version-2 compatibility regression](pthread-compatibility-result.json)
passes the existing 46 checks, including spatial/temporal faults, shared-heap
threads, SQLite/mruby/Perl and both 200,000-allocation recycling cases. It was
run after the blocking-I/O/exit changes; the final context-identity and
worker-fault-record additions and epoll/message conversion fixes are covered by the new
pthread/server gates. The [epoll controls](pthread-epoll-controls.json) insert
the same sleep after syscall copy-back in both variants: the shared-buffer
mutation fails the event-identity assertion, while the private-buffer version
passes. An ordinary old-buffer run can pass; the controlled interleaving is
what exposes the race.
The [host message controls](pthread-host-controls.json) additionally block one
delegated receive while another completes with different buffer offsets. The
shared-view/iovec mutation corrupts the first copy-back; private per-call
descriptors pass. Build the native test from
`runtime/tests/application/delegate-thread-test.c`, `linux/delegate-service.c`,
`linux/signals.c`, `linux/spawner.c`, `linux/park.c`, `common/delegate.c`, `common/msghdr.c` and
`common/spawn.c`, with `-I runtime/include -lpthread`.

The loader registers image, stack, startup and exchange mappings. Startup
pages are pinned. Growing private anonymous mappings initially have absent PTEs: the module
asks Linux's GUP interface to resolve the first touch, then retains a pin
until retirement. A permitted write fault after `mprotect` is resolved by Linux
and must retain the same physical backing; a denied VMA access is never fixed
by granting more permissions. It does not supply a pager or change VMA permissions to
make a denied access succeed. TLS uses the common capability libc setup.

Each launch owns a growing lifetime table with 16-byte records. Its physical
root stays fixed; a 5:9:9-bit directory maps the high 23 bits of a node ID to
a 4-KiB record page, and the low eight bits select one of its 256 slots.
Only base-page allocations are required. The initial four record pages plus
root and directory pages occupy 28 KiB; IDs 0 and 1 remain reserved.
Linux keeps the arena's private ancestor as a
scalar identity. Whole-arena retirement walks and invalidates its descendants
and the ancestor before clearing physical tags and unpinning pages. Reusing
the same VA creates a fresh live identity. When the table is pressured, the
trusted collector clears stale tags in resident registered pages and saved contexts,
then returns unpinned identities to the table free list; dead PCC identities
remain pinned because a PCC has no tag to clear. Teardown discards the context
and clears tags before freeing all table pages and its saved frame.

`node_initial_pages` (default 4), `node_batch_pages` (16) and `node_max_pages`
(0 means the 31-bit ID limit) count record pages and are read-only module
parameters; directory pages are accounted separately in `node_bytes`. Existing free
IDs are used first. Pressure can collect a substantial retired population or
append initialized pages, then retry the original instruction. At a quota or
allocation failure, a final collection may still recover a smaller retired
population. Published node pages remain allocated until namespace teardown;
this version does not compact records or return individual empty pages.
The root's retired count tracks invalid, not-yet-reclaimed records; it does
not count capability references. A malformed count refuses collection.

The normal collection threshold is the maximum of the requested deficit,
one growth batch, a quarter of table capacity and the number of registered
virtual pages (including holes). This amortizes the full VMA inspection over
retirements; a small live heap beside a large sparse reservation must not
trigger a sweep every few hundred allocations. Quota/allocation failure
still permits a smaller last-chance sweep.

The same `ensure_nodes` path supplies CSMINT, instruction pressure and the
SDK's `CV_SERVICE_NODES` budget request. The SDK checks the instruction's
256-slot cleanup reserve too. This request supplies capacity, not an exclusive
reservation against other threads. Failed budget requests return `ENOMEM`;
an instruction which still cannot obtain a node produces resource cause 30.
The module checks `urevavail` before its first mint, so a processor that does
not understand the paged format refuses the open instead of taking a mint
fault in the kernel. Flat tables remain supported by QEMU.

Collection requests use linked 4-KiB lists of physical pages. QEMU validates
the complete list and every issued node page before changing tags. All C
contexts in the namespace remain stopped through the whole sweep. A small
set of saved PCC IDs replaces the host bitmap indexed by the node high-water
mark. `node_capacity`, `node_bytes` (including directories) and `node_growths`
report actual resources instead of the former fixed 1-MiB charge.

The allocator uses existing Sublet operations for object lifetimes. It grows
through Linux mappings and supports `malloc`, `calloc`, `realloc`, alignment
and whole-free-arena `malloc_trim`. It checks the remaining node budget
before allocation and asks the adapter for more capacity when needed.
Blocks are aligned powers of two, minimum 256 bytes. The
returned capability is bounded to the request; a request of 4 KiB or more is
rounded up to the representable grain, at most 1/512 of its size.
`malloc_usable_size` reports that bounded extent, not the block.
Private anonymous `mmap`, page-range `mprotect` and whole-mapping `munmap`
use the same VM service. `mmap` supports `PROT_NONE` and R/W/X combinations;
its capability carries maximum anonymous-mapping rights while PTEs enforce
current protection. `mprotect` never increases an explicitly restricted
capability's own rights. Management operations use the registered virtual
range without reading through the pointer, so `PROT_NONE` can be retired.
Public lengths are rounded to pages. Internal power-of-two backing padding
stays `PROT_NONE`, cannot be exposed by `mprotect`, and is retired with the
whole mapping. Heap grants retain linear ownership; public mmap grants are
explicitly delinearised. Both belong to one mm, shared by its C threads.
The virtual SDK bounds metadata at 65,536 block records and 256 arena records by default.
It starts with at most 256 block records and 32 arena records, then adds metadata
slabs through the common VM service without recursing into malloc. Metadata has
separate mapping lifetimes and survives payload `malloc_trim`. A scalar atomic
mutex protects allocator state across thread switches and suspended mapping
calls; contended callers return to Linux through the WAIT service. The
uncontended malloc/free path performs no lock-service calls. Larger applications may set `CAPSTONE_APPLICATION_VIRTUAL_BLOCKS` and
`CAPSTONE_APPLICATION_VIRTUAL_ARENAS` when building the SDK.
The collector is only enabled after a complete namespace sweep. If growth
is unavailable and remaining identities are live or pinned, the context ends with a resource
fault (cause 30) instead of guessing that a stale capability is gone.

The compiler profile combines `-capstone-gp-free` calls within PCC with
`-capstone-image-gp`: a representable readable image capability supplies
globals and anonymous constant pools. It permits those pools only for this
profile; the bounded capability-table ABI retains its existing refusal.
Virtual C cannot fabricate a missing gp. The shared setjmp and signal-stack
assembly follows the profile's scalar return-address convention.

## VM service contract

`vm-abi.h` is shared by startup assembly, libc and the trusted launcher;
`vm.h` is the private libc grant interface. MAP takes bytes, alignment, Linux
protection, maximum capability rights and a mapping kind (heap, application,
metadata). It returns a consumed linear grant or a scalar negative errno.
UNMAP removes an exact whole visible mapping. PROTECT operates on a page range
inside one registered payload mapping. WAIT yields to Linux without holding
the transport lock. Metadata mappings are not exposed to public protection or
retirement services. New images carry `CPONVVM3`; the launcher also accepts
`CPONVVM2` images. Application ABI-v2's
ordinary delegated system-service format remains independent of this marker.

Linux, the adapter and the runtime belong to the TCB. Linearity concerns
virtual authority in one address-space instance; threads do not get separate
ownership namespaces. The private-anonymous/pinned-page restrictions are
integration limits, not a claim of global physical ownership. Kernel page
faults, VMA policy and physical allocation remain Linux's responsibility.
The adapter supplies only the capability lifecycle and continuation glue.

Collection and retirement inspect missing versus resident PTEs under Linux's
mm lock. Trusted `FOLL_NOFAULT` inspection includes resident RO/PROT_NONE
storage without materializing holes; huge/swapped or unexpected shared backing
fails closed. The already qualified physical tag rules clear stale tags before
IDs are recycled. This remains a QEMU collector, not an RTL reclamation design.

Executable mappings can supply PCC to an explicit virtual-thread context.
The gp-free application's ordinary scalar-return call convention retains its
image PCC. General JIT calls between separately bounded code mappings need a
capability call/return ABI and are not claimed by the executable-context test.

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
"$out/sdk/capstone-cc" -O1 -Icapstone/runtime/include capstone/runtime/virtual/contract.c -o "$out/contract.dom"

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

To exercise node growth and reuse, build `node-contract.c` with this SDK:

```sh
"$out/sdk/capstone-cc" -O1 -Icapstone/runtime/include \
  capstone/runtime/virtual/node-contract.c -o "$out/node.dom"
python3 capstone/runtime/virtual/run-nodes.py \
  --qemu /path/to/new-qemu-system-riscv64 --images /path/to/images \
  --adapter "$out/adapter" --application "$out/node.dom" \
  --work "$out/node-gate"
```

This gate keeps 70,000 structural nodes live, retires them and repeats. It
checks live capabilities across growth, stale denial after reuse, a small
module quota with both an `ENOMEM` service reply and an instruction resource
fault, continued authority after refused growth, and complete teardown.
It measures node capacity and all directory/record bytes separately from the
allocator's block-record count. Rebuild the runtime SDK and relink applications
to use its proactive node-budget service; old SDKs can return `ENOMEM` from
their fixed-budget check before an instruction requests node growth.

## Qualification and limits

The shared application source recipes support a virtual profile for CPython,
PostgreSQL, FFmpeg and tshark; see the
[build instructions](../../ports/common/application/README.md#virtual-source-builds).
The [port result](app-ports-result.json) passes 16/16 normal and 17/17
inner-allocator checks against native output oracles, and 32/32 configured
FFmpeg/tshark safety fixtures with their original expectations. Omitting the
CPython image fails its checks while the other ports complete.
The virtual heap also implements the internal linear-block interface used by
FFmpeg pools and tshark wmem chunks. It loans a size-class block while keeping
a senior revocation handle; returning the block retires derived lifetimes
before reuse. Its scalar-base return operation belongs to the trusted nested
allocator ABI. Public `free(pointer)` continues checking the capability and
rejects loan blocks. Heap counters report actual operations; buddy merges are
zero because this allocator uses size classes rather than a buddy tree.
The [extended runtime result](app-ports-security-result.json) passes 46 checks,
adding linear-block reuse and an exact-site stale-descendant fault to the
existing VM, thread, recycling and SQLite/mruby/Perl checks.

The [libc VM-service v2 result](libc-vm-result.json) passes 42 checks on the
recorded QEMU/Linux platform, including SQLite, mruby and Perl. It covers
page protections, guard pages, requested-length bounds, failed-realloc
preservation, executable child contexts, shared-heap allocation across
preemption, and rejection of old virtual images. Two 200,000-allocation
cases exercise recycling with an untouched 64-MiB reservation and stale tags
on a `PROT_NONE` page. The positive case reclaims 194,982 identities in three
collections without populating unused pages; the negative case faults at the
stale load after restoring read permission. Input and source hashes are in
the result. See [the qualification record](libc-vm-qualification.json) for
mutation controls and the processor regression gate.

The gate requires argv/environment, TLS, longjmp, file operations including
capability-valued lock arguments, 1.5 MiB of heap growth and release, actual
Linux demand faults, two independent processes, a child virtual context sharing
the process `mm`, termination of a non-yielding program, and module unload.
Stale/free, bounds and retired-then-reused-VA
accesses must fail. SQLite writes a database and another process reopens it;
mruby performs arithmetic and file I/O. The `--omit-application` negative
control must fail, even though the shell and other applications still run.

The historical fixed-table recycling evidence uses the existing application contract
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
For the growing-table profile, `run-nodes.py` sets `node_max_pages` explicitly
to exercise bounded exhaustion instead of relying on the old pool size.

The [instruction gate](../../capstone-qemu/tests/virtual-capstone-runtime/README.md)
also checks consuming LDC/STC fault retries, linear replies, immutable context
roots and restoration of the physical caller. The
[qualification record](qualification.json) includes compiler, regression and
negative-control results. Existing physical applications remain a regression
gate; this adapter does not replace their monitor or CALL/RETURN path.

This is the first QEMU application profile: one hart, private anonymous
mappings, 256 MiB per registered region, 1 GiB
aggregate registered VA, 512 mappings, and a recyclable node table bounded by
RAM, an optional module quota and the 31-bit ID field. The separate allocator
block/arena limits are unchanged by node-table growth.
Fork, shared tagged mappings, file-backed
mmap, partial unmapping, swap, migration, application register editing by
signals need further contracts. The qualified pthread subset is described
above. Address hints/fixed mappings and `mremap` remain unsupported. Partial unmap
requires a separate lifetime contract: a live wide capability must not acquire
a replacement mapping placed into its former hole. Anonymous
`MAP_SHARED` and SysV segments are process-local compatibility for nested
ports, since there is no fork; every other mapping form fails explicitly.

Pages stay pinned after first touch. A page that Linux populated without an
application fault is pinned by the next retirement or collection, so a
teardown does not clear its tags itself; Linux clears them with ordinary
stores when it zeroes the page for reuse. The initial ELF image has the
existing combined code/data layout. Representability padding above 4 KiB
requests is accessible within its own block. Recycling requires the
registered-page and saved-context sweep; a namespace full of live or pinned
identities reports resource exhaustion when growth is unavailable.
Compact-bounds conformance of every existing compiler/helper path
remains separate work: the inherited QEMU data-tag implementation still
retains extra bounds metadata. The gates establish the stated behavior in
QEMU; they do not establish an RTL implementation or a complete memory-safety
proof for all programs.
