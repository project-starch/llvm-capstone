# Local mallocng in virtual Capstone

Application code and musl, including mallocng, are compiled for Capstone and
execute in the same virtual C context. `malloc` and `free` are ordinary calls.
Allocation policy belongs to musl: size classes, group counts, slot selection,
offset cycling, retention and the in-place/moving realloc decisions remain
upstream's. The launcher does not allocate application objects.

```text
Capstone application
        | ordinary call
Capstone-compiled musl mallocng
        | choose slot                  | needs OS backing
        v                              v
local SPLIT / MREV / DELIN / REVOKE     mmap / munmap / mremap / mprotect
        |                              | VM boundary
bounded object capability             trusted adapter -> Linux
```

Node growth/collection, page faults, scheduling and contended waits can also
cross the boundary. They are resource and execution slow paths, not an
unconditional allocation service. The 200,000-lifetime stress test holds size-class groups live to exercise
identity recycling rather than predominantly repeated mapping teardown.
The directed fast-path test holds a free
slot in a live group and enough node capacity; its malloc/free loop must
produce no service transitions. A control deliberately inserts a WAIT
service, which the oracle must detect. A workload that repeatedly causes
musl to release and recreate whole groups legitimately invokes Linux.

## Representation without a new allocation policy

The port keeps `UNIT=16`, `IB=4`, the size-class tables and the mmap threshold.
A group header still occupies 16 bytes. Its metadata reference is a scalar
identifier; allocator authority comes from the private metadata index, never
from converting that identifier into a capability.

A mapping is partitioned into a header and the exact slots selected by
mallocng. Each slot has aligned linear ownership and an out-of-line record
holding its revocation handle. A private 4,096-bucket index chains these
records; bucket count does not limit object count, and chains grow with the
live population. `enframe` retains musl's offset calculation.
The four prefix bytes immediately before a slot are represented in that
record; prefixes at nonzero offsets remain inside the slot. This prevents a
slot's allocator metadata access from crossing into its neighbour's linear
ownership. Footer checks remain inside the owning slot.

```text
mapping ancestor (retained by the kernel)
  +-- group header
  +-- slot owner / allocator handle
  |     +-- current non-linear lifetime
  |            +-- internal full-slot pointer (allocator only)
  |            +-- bounded public pointer and its aliases
  +-- next slot owner
```

The application receives only the requested object bounds. The allocator's
full-slot pointer permits its private prefix/footer operations. `free`
validates the supplied capability before looking up internal authority,
clears payload tags and revokes the lifetime before the upstream backend
publishes the slot as free. A successful in-place realloc rotates the
lifetime; a failed realloc preserves the original. Moving realloc uses the
existing Capstone memcpy, which copies non-linear tags and consumes linear
ones. Explicit Sublet loans remain linear; reclaiming their descendants
retains the UNINIT initialization rule.

The capability records enlarge `struct meta`; upstream's existing metadata
allocator grows according to that representation's `sizeof`. This is a real
metadata cost, separate from payload policy. There is no 65,536-object array
or replacement buddy allocator. The hash index has a fixed number of buckets
but unlimited linked entries within available metadata/resources. It does
not choose allocation addresses. A shared lock covers publication and the
metadata index across same-address-space threads.

The current process ABI does not expose the launcher's `brk` to application
libc. mallocng therefore uses its own mmap fallback for metadata. Static
images do not donate dynamic-loader tails. Neither restriction introduces a
replacement allocator. `malloc_trim` is a compatibility no-op: musl retains
its normal group-release decisions. A zero-byte allocation has one byte of
capability extent for the validity check, with nominal size zero.

`check-mallocng-policy.py` compares decision functions, tables, constants and
selection blocks against the verified upstream archive. It is a focused
source guard, not a proof of identical execution. Mutation controls change
UNIT, group-count selection and the in-place realloc condition. Runtime
qualification separately checks behavior, lifetimes and the service boundary.

## VM remapping

Large realloc follows upstream's mremap branch. Heap mappings use their
page-rounded requested length, without the old power-of-two padding.
The native launcher invokes Linux `mremap(MREMAP_MAYMOVE)`; it does not run
mallocng. The module's BEGIN/END transaction:

1. Validates the old arena, pins existing resident pages, reserves node and
   page-array capacity, and parks all protected contexts in this namespace.
2. Lets Linux perform mremap. On failure, cancellation leaves old lifetimes
   live and resumes the namespace.
3. On success, retires the old ancestor, preserves tags on retained physical
   frames, clears tags on truncated frames, updates the registered mapping,
   and returns a fresh linear grant.
4. Lets Capstone libc reconstruct the group and publish its new lifetime.

The transaction is only a VM slow path. There are no kernel object records,
heap BEGIN/COMMIT calls or native allocation callbacks. Linux core and firmware
are unchanged. Public application `mremap`, partial unmap, file-backed mappings,
swap/migration and multi-hart execution are not added by this allocator port.

## Processor storage contract

ABI v5 requires QEMU's explicit `x-capstone-exact-bounds=true` profile. The
launcher checks its advertised bit before starting the image. Version 2/3
images remain accepted; the withdrawn native-heap version 4 is rejected.

The profile makes the existing physical shadow bounds authoritative during
minting and context-frame restoration as well as capability spill/reload.
Thus a spill or VM reply cannot widen an object into adjacent allocator
metadata. This is a QEMU prototype contract with additional shadow metadata,
not a one-bit-tag encoding or an RTL implementation. No allocator-specific
instruction is added. The physical monitor's default processor profile is
unchanged.

## Reproduce

Build the virtual SDK and adapter using the main runtime README. Preparation
applies the representation patch; the virtual survey requires all 1,361 C
translation units to compile, including the six mallocng files. The physical
survey retains its original layout and failure control.

```sh
"$out/sdk/capstone-cc" -Icapstone/runtime/virtual \
  capstone/runtime/virtual/malloc-contract.c -o "$out/malloc.dom"
python3 capstone/runtime/virtual/run-malloc.py \
  --adapter "$out/adapter" --application "$out/malloc.dom" \
  --native /path/to/native-musl-malloc-contract \
  --qemu /path/to/qemu-system-riscv64 --images /path/to/images \
  --work "$out/malloc-gate"
```

Build the same contract with a native RV64 musl toolchain for the reference
argument. The native allocator is used only as a test control. Runtime test
records identify the exact application, launcher, module, emulator and images;
old native-heap or buddy-allocator results are not qualification of this port.
