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
local CDERIVE / CREVOKE                mmap / munmap / mremap / mprotect
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
musl to release and recreate whole groups invokes Linux only when the
runtime's kept mappings ([Recycled group mappings](#recycled-group-mappings))
cannot supply one of the needed length.

## Representation without a new allocation policy

The port keeps `UNIT=16`, `IB=4`, the size-class tables and the mmap threshold.
A group header still occupies 16 bytes. Its metadata reference is a scalar
identifier; allocator authority comes from the private metadata index, never
from converting that identifier into a capability.

A mapping grant is delinearized once and supplies a managed group lifetime.
Each slot retains a narrowed copy of that group authority; it needs neither
a linear partition nor its own parent node. The out-of-line slot record holds
that parent and the current issued child. A private 1,024-bucket index chains
these records; bucket count does not limit object count, and chains grow with
the live population. `enframe` retains musl's offset calculation.
The four prefix bytes immediately before a slot are represented in that
record; prefixes at nonzero offsets remain inside the slot. This prevents a
slot's allocator metadata access from crossing its bounds. Footer checks
remain inside the owning slot. The index is consulted once per call that
receives a pointer from the program (`free`, `realloc`, `malloc_usable_size`,
the nested-allocator trio): the record found there is passed on to mallocng's
`free` and `realloc`, whose prefix accesses take it as an argument
(`cap_prefix(p, s)`), and `enframe` passes the record of the slot it takes.
An allocation consults the index not at all: `enframe` keys the record as its
last step and the runtime publishes the object from that record. Non-overlap follows mallocng's slot geometry;
the managed lifetime instructions do not enforce non-overlap.

```text
mapping ancestor (retained by the kernel)
  +-- managed group (allocator only; also covers the header)
        +-- current public object and its aliases
        +-- another current object
        +-- nested mallocng group
              +-- its current objects
```

The application receives only the requested object bounds. The allocator's
bounded group copies permit its private prefix/footer operations. `free`
validates the supplied capability before looking up internal authority and
revokes the lifetime before the upstream backend publishes the slot as free.
It leaves the payload's bytes and tags in place unless the SDK is built with
`CAPSTONE_HEAP_SCRUB_FREE=ON` ([Measurement switches](#measurement-switches)). A successful in-place realloc rotates the
lifetime; a failed realloc preserves the original. Moving realloc uses the
existing Capstone memcpy, which copies non-linear tags and consumes linear
ones. Inner allocators use ordinary malloc/free and derive their own children
from the allocation. There is no separate linear heap loan API. Freeing the
outer object invalidates its descendants. Linear grants remain at the VM
boundary, where mapping ownership is transferred.

The capability records are not part of `struct meta`: a meta (112 bytes,
upstream's fields plus three) points to its group's records, one per slot,
which `heap.c` hands out as arrays of 1, 2, 4, 8, 16 or 32 records, the
smallest that holds the group's slot count, from chunks it maps for them
(16 KiB, doubling to 256 KiB), reused through a free list per size. Upstream's
metadata allocator still sizes meta areas by `sizeof(struct meta)`, so a page
holds dozens of metas; with 32 records embedded a meta was 3,152 bytes and a
page held one, whatever the group's slot count. The records are a real
metadata cost, separate from payload policy. The managed slot record keeps the
same four capability fields as the previous representation, and the old
linear tail field is removed from `struct meta`. There is no 65,536-object array
or replacement buddy allocator. The hash index has a fixed number of buckets
but unlimited linked entries within available metadata/resources. It does
not choose allocation addresses. A shared lock covers publication and the
metadata index across same-address-space threads.

The current process ABI does not expose the launcher's `brk` to application
libc. mallocng therefore uses its own mmap fallback for metadata. Static
images do not donate dynamic-loader tails. Neither restriction introduces a
replacement allocator. musl retains its normal group-release decisions;
`malloc_trim` only unmaps the runtime's kept group mappings. A zero-byte allocation has one byte of
capability extent for the validity check, with nominal size zero.

`calloc` skips its `memset` when the object's slot was clean at this take:
a slot of a fresh mapping (or of a nested group inside such a slot) that no
object has used, or a reused mapping's slot, zeroed at that take. This is
upstream's `__malloc_allzerop` (which knows only fresh single-slot and
individually mapped groups) extended to every slot: no capability the
program holds covers a slot it was not issued from, so nothing has written
one nobody took. `CAPSTONE_HEAP_OBJECT_BOUNDS=OFF`, where the program holds
the group's authority, never skips. A slot reused within its group is zeroed
by the `memset` as before. `free_meta` clears the group's slot records
before returning them and then the meta, so a group always binds records
that are all zero.

`check-mallocng-policy.py` compares decision functions, tables, constants and
selection blocks against the verified upstream archive. It is a focused
source guard, not a proof of identical execution. Mutation controls change
UNIT, group-count selection and the in-place realloc condition. Runtime
qualification separately checks behavior, lifetimes and the service boundary.

## Measurement switches

Two CMake options of `runtime/application` build the arms of the spatial-versus-temporal
comparison; `sdk.json` records them as `heap_lifetimes` and `heap_scrub_free`, and
`build-sdk.sh` forwards them (`-DCAPSTONE_HEAP_LIFETIMES=OFF`). The defaults, lifetimes
ON and scrub OFF, are the heap described above.

- `CAPSTONE_HEAP_LIFETIMES=OFF`: a public object, and a nested group, is a SHRINK copy of
  the slot's authority over the same bytes instead of a CDERIVE child. Bounds and rights
  are those of the child; what is missing is the lifetime of its own, so `free` revokes
  nothing and a stale capability stays usable until its mapping is retired (`munmap`,
  `mremap`). The lifetime per mapping remains, as does the node reserve request. This is
  the virtual counterpart of level0 without `CAPSTONE_LEVEL0_LIFETIMES`, the paper's
  "spatial" arm: public heap object lifetimes off, mapping and thread retirement unchanged.
- `CAPSTONE_HEAP_SCRUB_FREE` (default OFF): `free` leaves the payload's bytes and tags in
  place (`calloc` still zeroes), so residual data and capabilities in a reused slot are the
  next owner's to read. A residual capability to a freed object still faults on use, because
  its lifetime ended; one to a live object stays usable, which is outside what lifetimes
  protect. A mapping that `free` unmaps is cleared of tags by the module's retirement
  either way, and a kept mapping's slots are zeroed at their first reuse. `CAPSTONE_HEAP_SCRUB_FREE=ON` zeroes the object's current extent before the
  CREVOKE, the hygiene arm of the comparison. It does not reach the tail an in-place
  `realloc` shrink gave up, so it does not make every reused slot clean.

`run-malloc.py --sdk DIR` reads the SDK's switches and expects, without lifetimes, exactly
these readings from the anchored denial cases: `bounds` still faults with cause 28,
`remap-stale` still with cause 25 (the module retired the old mapping), `stale-free` and
`interior-free` with cause 2 at `cap_malloc_invalid_free` (no slot record), and `stale`,
`inplace-stale`, `reuse-stale` and `reuse-free` survive and say `MALLOC_SURVIVED`. The
same harness on an SDK with lifetimes must still read the lifetimes table, which is what shows
the survive checks can fail.

## Recycled group mappings

When mallocng releases an empty mapped group (its own `okay_to_free` decision,
unchanged), the runtime keeps the mapping registered and resident and hands it
to the next group of the same mapping length, so neither the release nor the
reuse is a service. Every mapped group's lifetime is a CDERIVE child of a root
derived once per mapping and held only in `heap.c`'s private state, never in a
meta, a slot record or a return value:

```text
mapping grant (kernel) +-- root (heap.c private state, lives as long as the mapping)
                             +-- group lifetime (one per group) -- slots -- objects, nested groups
```

Releasing the group CREVOKEs its lifetime from the root, which ends every
object and nested group of it at once, with or without per-object lifetimes;
the next group gets a fresh lifetime and freshly derived slot parents. Each
slot of a reused mapping is zeroed at its first take, before any capability
over it is issued, so no tag of the old group reaches the new owner; with
`CAPSTONE_HEAP_OBJECT_BOUNDS=OFF`, where the program holds the group's
authority, the whole mapping is zeroed before the new group is bound. At most
4 mappings of one length, 32 in all and 1 MiB are kept, none above 256 KiB;
a release beyond a bound is unmapped as before, and large (`>= MMAP_THRESHOLD`)
or remapped mappings always are; `malloc_trim` unmaps all kept mappings. Kept
mappings stay registered arenas: the module's collection sweeps them and its
teardown retires them at exit. `CAPSTONE_HEAP_RECYCLE_TRACE=1` prints a line
per reuse and per kept release.

The trust boundary is the program's own pointer flow: a pointer obtained from
malloc and kept past free cannot reach a later object, including across
mapping reuse. Code that reads allocator state on purpose -- through the
image-wide gp capability into libc's statics, or through the SDK-internal
entry points (`__capstone_malloc_find` and the other `__capstone_malloc_*`
calls musl and some ports make) -- is outside it, as it already was before
recycling; such code can reach the roots in `heap.c`'s statics, though no
entry point returns one. Moving allocator state out of gp's bounds is a
separate item. `run-malloc.py`'s
`recycle-*` cases qualify this under every heap configuration (with object
bounds off they are that arm's whole gate) and fail if the mapping was not
reused: stale use after reuse (`recycle-stale`, `-free`, `-geometry`,
`-nested`, `-aligned`, `-realloc`), tags in the reused object (`recycle-tags`)
and, with object bounds off, in the header, untaken slots and slack
(`recycle-header`, `-slack`), no internal record exposes a capability beyond
the group's own (`recycle-root`), and the bounds plus a concurrent
`malloc_trim` (`recycle-limits`). `calloc-clean` reads every byte and granule
of callocs on a fresh slot, a slot reused after an object filled it with
bytes and capabilities, and a recycled mapping's slot, and requires the
skip exactly where the slot was clean. `calloc-shrink` (a calloc reusing the
slot of an object that was filled, shrunk in place and freed),
`calloc-geometry` (a single-slot calloc on a kept mapping a multi-slot
group's dirtied object used) and `meta-reuse` (slot-record arrays reused by
groups of many -> few -> many slots: every record of the array past the
meta's `last_idx` is all zero bytes) read the same properties on the paths
around it.

## Shared backing for small groups

The group-mapping trace motivates sharing backing for groups with at least
three slots and at most 16 KiB of page-rounded storage. The measured PHP ary2
workload creates 15,000 groups of twenty 192-byte slots: each occupies a
4 KiB mapping although its header and slots need only 3,856 bytes. This is
a mapping-service cost as well as page padding. Single-slot groups include
page slack in their stride; removing that slack would be a separate change
to their representation, not a consequence of sharing backing.

The runtime carves these groups from 64 KiB aligned regions. Each region holds
equal-sized extents: header plus slots, rounded to a 16-byte granule and then to exactly
representable bounds. Extents smaller than 1 KiB retain dedicated backing,
bounding the region occupancy bitmap at 64 bits. The descriptor and root stay
in private runtime state; each group receives a separately bounded CDERIVE
child. Neither mallocng's meta nor a slot record receives the region root.
The region lookup uses the group's address; available regions are indexed by
extent size. No program-supplied pointer becomes allocator authority.

`CAPSTONE_HEAP_COARSE_PAGES=ON` is an experimental, default-OFF SDK option.
It rounds eligible extents up to 4 KiB before selecting a region bin. Nearby
group geometries can then reuse the same page-aligned extent, and consecutive
extents never share a page. The rounded extent must still fit the mapping
length requested by mallocng. Eligibility, individual group lifetimes, dirty
bits and scrubbing are unchanged. `sdk.json` records `heap_coarse_pages`; this
is a placement experiment, not a change to the 128-bit capability ABI.

Release revokes the group before making its extent available. A persistent
dirty bit marks reuse even when the new group has different slot geometry.
Each reused slot is scrubbed before its first take, using the existing slot
record flags; with object bounds disabled, the entire extent is scrubbed
before binding slots. The slot arrays and pointer index retain their existing
ownership and lookup rules. Single-slot groups, including every mremap
candidate, keep dedicated mappings and the existing recycling cache.

Empty regions share the cache's 1 MiB retention budget. `malloc_trim` returns
all empty regions through the retiring VM unmap path; partially occupied
regions are never unmapped or advised away. A failed region unmap leaves its
descriptor registered so trim can retry. All region and cache operations use
the existing heap lock. The mallocng representation patches and allocation
policy are unchanged.

Fewer mappings need not mean fewer resident pages. In particular, the measured
PHP shape above fits sixteen extents in 64 KiB: seventeen need 65,552 bytes.
The region's 3,840-byte tail therefore consumes the page padding saved between
those sixteen groups. Other shapes can pack more densely, but region tails,
partially occupied regions and private descriptors must be included when
assessing page savings. The current region size bounds bookkeeping and reuse;
it does not promise the ideal packing bound from a group-padding census.

The malloc gate includes `coarse-reuse`, `coarse-stale` and `coarse-bounds` in
all three heap configurations. These call the internal group interface in a
single thread, so per-object revocation cannot hide a missing group revoke.
Twenty groups must occupy two 64 KiB regions with 3,360-byte extents; siblings
must share a page and survive release, partial-region trim and reuse. Reuse
changes nineteen 176-byte slots into eleven 304-byte slots at the same address.
Every returned slot must contain zero bytes and no tags, with a stored-tag
positive control; bounds-off heaps must also clear the whole extent before
any slot is taken. Stale and out-of-bounds stores must fault with causes 25
and 28 respectively at the labelled store, with exit 139. Missing or duplicate
process records fail the oracle. Partial-region trim must release no region;
final trim must release exactly two.

With `heap_coarse_pages=true`, the same gate requires 4,096-byte extents and
sixteen groups in the first region; it rejects the default compact geometry.
Both modes still require two mappings for twenty groups. The test selects
the expected geometry from the SDK, independently of the observed layout.

The same contract on the dev control without coarse regions reports
`coarse_backing=dedicated`: twenty 4 KiB mappings. It checks group revocation,
geometry-changing cache reuse, scrubbing and final trim there; shared-page
placement and partial-region trim do not apply. The runner selects the expected
layout from the linked heap's coarse-region symbol, so a shared heap falling
back to dedicated mappings cannot silently pass the sharing checks.

## VM remapping

Large realloc follows upstream's mremap branch. Heap mappings use their
page-rounded requested length, without the old power-of-two padding. One
mapping holds at most `CV_MAX_REGION_BYTES` (2 GiB) and a process's mappings
`CV_MAX_BYTES` (4 GiB), from `vm-abi.h`. A malloc needs `IB + UNIT` bytes above
its payload, so the largest payload is a little under 2 GiB. Pages are backed and
pinned as they are first touched, so only what a program touches counts against
guest RAM; with `CAPSTONE_HEAP_SCRUB_FREE=ON`, `free` clears the whole payload and so touches every page of it. QEMU's exact-bounds metadata for a page that has held a capability
also costs host memory, about 6 KiB per 4 KiB page.
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

ABI v7 runs on QEMU's default profile (option iii). The
launcher accepts this version only; an image of an earlier version must be
rebuilt.

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
"$out/sdk/capstone-cc" -Icapstone/runtime/virtual -Icapstone/runtime/include \
  capstone/runtime/virtual/malloc-contract.c -o "$out/malloc.dom"
python3 capstone/runtime/virtual/run-malloc.py \
  --adapter "$out/adapter" --application "$out/malloc.dom" \
  --native /path/to/native-musl-malloc-contract \
  --qemu /path/to/qemu-system-riscv64 --images /path/to/images \
  --sdk "$out/sdk" --work "$out/malloc-gate"
```

Build the same contract with a native RV64 musl toolchain for the reference
argument. The native allocator is used only as a test control. Runtime test
records identify the exact application, launcher, module, emulator and images;
old native-heap or buddy-allocator results are not qualification of this port.
