# Minimal protected musl mallocng

The virtual profile uses musl 1.2.5 mallocng inside the application. The port
keeps musl's slot selection, queues, offset cycling, bounce tracking and group
release decisions. It does not maintain a second object allocator or object
registry. `runtime/virtual/malloc-mapping.c` only supplies mapping authority.
The former `heap.c`, slot/lifetime registry, recycle policy and widening APIs
are removed. Free does not scrub payloads; calloc retains normal C zeroing.

## Porting changes

Patch `0009-virtual-mallocng-capabilities.patch` contains the representation
and lifetime changes:

- A group header occupies 32 bytes instead of 16 because it contains a
  capability pointer. Allocation units remain 16 bytes. Power-of-two edge
  size classes lose one unit and the mmap threshold becomes 131036 bytes;
  the size-class calculation accounts for the extra header unit.
- Metadata is obtained through mmap instead of integer-valued brk. Pointer
  arithmetic retains authority, including metadata-page and group lookups.
  Loader donation is disabled for this static profile.
- malloc obtains its raw slot from musl and derives one object child. Requests
  of 4096 bytes or more use musl's aligned-allocation path with representable
  length/alignment. That path calls the raw allocator directly, so it does not
  issue and revoke a temporary object child.
- free validates the supplied capability, recovers mapping authority for musl
  metadata, revokes the object child and follows ordinary mallocng free.
- Successful realloc replaces the old lifetime. In-place and mapping-resize
  paths also check representable alignment; otherwise musl allocates, copies
  with tags preserved and frees. Failure leaves the original object live.

Patch `0010` retains the existing metadata-area protection batching.
`check-mallocng-policy.py` verifies the upstream archive hash, applies these
reviewed patches in a temporary tree and compares every mallocng source file.
It also checks unchanged allocation/release policy functions against upstream.
The negative tests reject an altered policy, a removed revoke and an added
registry source file. This is a guard against unreviewed source changes, not
an assertion that capability layout or representability are identical to RV64.

## Mapping authority and nested allocators

`heap_sublet.h` supplies direct `heap_issue` (CDERIVE), `heap_end` (CREVOKE, the
object alone) and `heap_renew` (CRENEW).
The mapping layer retains a three-level Sv39 page index. Each occupied page
names its mapping-wide parent; there are no entries for individual allocations
or mallocng groups. Lookup is a fixed sequence of indexed loads, not a search.
Table pages are retained for process lifetime; mapping entries are cleared
on unmap. Creating/removing mappings serializes index updates.

`heap_access` first probes the supplied capability for liveness and, when
object bounds are enabled, checks that its cursor is at its base. Only then
does it recover mapping authority for trusted allocator metadata access.
An integer address alone cannot authorize free. Nested allocators must retain
their own parent; there is no API that recovers an object's parent from an
object registry. Allocation/free still require musl's normal metadata work
and this mapping lookup in addition to the lifetime instructions.

PHP retains its original malloc capability in one pointer-sized prefix
immediately before its unchanged 48-byte Zend header. The payload is a direct
child of that parent. Large allocations add padding for representable payload
bounds. Cache reuse issues a fresh child; realloc failure preserves the old
child. The obsolete virtual hierarchical adapter is removed; virtual PHP uses
`PHP_REALPHP_ALLOC=nested`. The separate physical PHP port is unaffected.

This is the trusted allocator model: mapping authority is accessible to the
allocator inside the application's context. It does not isolate an adversarial
allocator or arbitrary application code with access to those internals.

## Measurement switches

Build via `runtime/virtual/build-sdk.sh OUT PREPARED_MUSL [CMAKE_OPTIONS...]`.
Use a fresh `MUSL_CACHE_ROOT` when replacing a tree prepared with the retired
patch. The preparer rejects a tree to which the new patches do not apply.
The SDK script compiles both libc and runtime with the same switches:

| Lifetimes | Object bounds | Behavior |
|---|---|---|
| ON | ON | Default: bounded child per object; revoke on free |
| OFF | ON | Bounds only; stale aliases can survive slot reuse |
| OFF | OFF | Mapping authority at the object's cursor |

Lifetimes ON with bounds OFF is rejected. `CAPSTONE_HEAP_SCRUB_FREE=ON` is
rejected. A policy-specific symbol makes mismatched libc/runtime archives fail
to link. Changing only a runtime CMake flag against an old libc is unsupported.
Direct nested derive/revoke primitives are independent of these outer-heap
comparison switches. PHP cache retirement uses them directly, but final block
release and moving realloc rely on the default outer malloc lifetime. Those
operations lose their temporal guarantee when outer lifetimes are disabled.

Concurrent libc callers must use pthreads. Public raw context creation is
rejected with ENOTSUP before allocating a frame or starting a worker. Raw virtual contexts sharing TLS do
not initialize musl's thread bookkeeping or turn on its allocator locks. The
heap concurrency contract therefore uses pthread_create/join; the raw-context
transport tests remain separate. The old heap's unconditional private locks
had hidden this distinction.

## Qualification

The [minimal-port qualification](results/minimal-mallocng/qualification.json)
records input hashes, functional and denial cases, policy controls, PHP and
SPEC test-workload results. The tests use both compressed and exact bounds;
compressed representability remains enforced by allocation geometry in both.
The runtime test's `dense-churn` mode is a fixed-table silicon exhaustion
experiment, not a success test on QEMU with dynamic node growth.

The compiler and Linux/QEMU platform are reused, identified inputs. This is
not a new LLVM build, RTL qualification, full SPEC campaign or full application
port sweep. Historical [local-mallocng results](results/local-mallocng/README.md)
qualify the previous implementation and must not be relabelled as this port.
