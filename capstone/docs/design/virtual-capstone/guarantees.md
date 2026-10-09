# Guarantees, limits and evidence

[Guide](README.md) · Previous: [Development](development.md)

The intended guarantee is capability memory safety **within one address
space**, assuming a trusted OS, adapter and correct lifetime management.
The current evidence is a set of QEMU implementations and bounded tests.
It is not a complete memory-safety proof for arbitrary programs, nor an
RTL/FPGA result.

## What each mechanism protects

| Mechanism | Protection | Condition or limit |
|---|---|---|
| Tag and type checks | Scalar bits cannot manufacture a usable capability | Trusted minting and tag-preserving transfers remain part of the contract |
| Bounds and rights | Access stays within the authority carried by the pointer | Authority must have been narrowed to the intended object; ABI v5 uses exact shadow bounds in QEMU |
| User PTE permissions | VM protection restricts permitted accesses further | `mprotect` does not revoke an object's lifetime |
| Lifetime revocation | Old aliases cannot access a retired object even at a reused VA | Allocator must revoke before reuse; nested objects need nested lifetimes |
| Consuming transfers | Moving linear authority does not duplicate its source | All fault checks precede mutation and use the same translation |
| Namespace sweep | Recycled IDs do not resurrect stale tagged pointers | All contexts must stop and every tag-bearing storage location must be accounted for |
| PCC checks | Execution stays within live executable authority, including cached code | Compiler call convention and code-mapping contracts still apply |

For example, a stale pointer to an outer `malloc` block is invalidated by
its free path. A stale pointer to a slab item can remain usable if the slab
allocator reuses the item without creating/revoking an item lifetime.
Virtual memory alone does not repair that missing allocator boundary.

Linux can inspect memory, change tables, mint roots and destroy namespaces.
This profile intentionally trusts those actions. It offers no confidentiality
or integrity against a malicious kernel or trusted adapter. Physical tag
identity prevents divergent tags through aliases, but the adapter still
controls whether aliases or independent overlapping grants may exist.

## Current integration boundary

| Supported by the recorded profile | Requires further contracts or qualification |
|---|---|
| One hart; Linux-scheduled same-mm contexts and the tested musl pthread subset | Multi-hart execution, cross-core revocation and context migration |
| Private anonymous backing, demand faults, resident-page pinning | Swap, physical-page migration and general shared/file-backed tagged mappings |
| Whole-mapping unmap and page-range protection changes | Partial unmap, fixed replacement, address hints and public `mremap` (mallocng uses an internal VM remap path) |
| Independent launches with separate namespaces | Protected `fork` and capability transfer between address spaces |
| Executable child contexts | General JIT calls/returns across separately bounded code mappings |
| Recorded delegated signal and exit behavior | Full POSIX cancellation, arbitrary signal delivery and signal register editing |

Anonymous `MAP_SHARED` and SysV compatibility accepted by the runtime remain
process-local backing in this no-fork profile. They do not establish
cross-process tagged sharing.

The node-growth extension replaces the fixed 65,536-slot / 1-MiB table with
base-page-backed tables bounded by RAM, an optional quota and the 31-bit ID
field. It retains the separate limits of 32 supervisor slots on the hart, at most
32768 registered mappings, 1 GiB per registered region and 4 GiB aggregate
registered VA. Local mallocng removes the fixed allocator block/arena arrays;
its metadata grows through musl mappings and remains subject to the adapter
quotas above. Table pages can be physically scattered. Allocator capacity requests
can return `ENOMEM`; if collection and growth cannot supply enough IDs,
node pressure terminates with resource cause 30;
it is not universally converted into recoverable `malloc` failure.

## Encoding work still matters

QEMU uses a different 128-bit field split from deployed RTL and keeps extra
uncompressed bounds in its physical tag shadow. The opt-in exact-bounds
profile makes those bounds authoritative during minting and context restore
as well as normal spill/reload. Local mallocng requires this profile to retain
upstream slot geometry and expose only requested object bounds. The default
profile continues to reject inexact CSMINT descriptors. A future tag-bit-only implementation
must not widen authority during a round trip. Reconcile the formats and test
representability before claiming ISA/RTL equivalence or freezing the format.

Other remaining interface decisions include permanent opcode/CSR allocations,
portable context storage and resource-failure behavior. A second OS adapter
would test the intended integration boundary. These are engineering tasks
separate from the demonstrated Linux application workloads.

## Reference revisions

| Reference | Revision used by this guide |
|---|---|
| QEMU through collection | [9bf9c1f28653][qemu] |
| Compiler/runtime through pthreads | [518a5b805aa7][runtime] |
| Memcached application leaf | [1354aee956cd][memcached] |
| Cross-repository evidence manifest | [e03131496304][manifest] |

Implementation references are pinned so links remain meaningful while PR
bases and landing commits change. The source code is the authority for what
this prototype does. In particular, the earlier spec patch's blanket
kernel-private-frame description must be read with the implemented
[user-mapped thread startup protocol](runtime.md#threads-share-memory-and-lifetimes).
The draft ISA amendment has not yet been published in the academic-spec
repository; a [transferable patch][spec-patch] is available.

## Recorded checks

These results were recorded during extraction of the review stack. Writing
this guide did not rerun the compiler, processor or guest test suites.
The [manifest][manifest] carries exact source/binary identities and build scope.

| Area | Recorded result | Interpretation |
|---|---|---|
| Compiler | 7/7 focused lit tests | Existing compiler matched the tested source; no fresh full LLVM build |
| QEMU | 33/33 virtual, 60/60 foundation, 69/69 access, 3/3 physical bounds | Includes context, authority, fault and instruction checks; not full physical runtime parity |
| Lifetime model | 8,949 prefixes for each of host and guest backends | Actual list algorithm compared with an independent forest; finite exploration |
| Linux adapter | 22/22 | Includes bounds/stale faults, same-VA reuse, process isolation and SQLite/mruby workloads |
| VM/heap | 42/42 | Includes two 200,000-allocation recycling cases |
| Pthreads | 9/9 plus 39/39 short v2 compatibility | Long recycling cases omitted from this later short compatibility run |
| Memcached | 73/73; 20 applicable outer-heap fixtures | Four workers; expected surviving nested-lifetime gaps remain explicit |

The safety fixtures include expected returns for known gaps. “All predictions
passed” therefore does not mean every tested bug was prevented. Memcached's
slab/cache lifetime and bipbuffer logical-reuse gaps remain; other ports need
their documented inner-allocator configurations.

### Growing node-table qualification

The core virtual stack is now on `dev`. The node-growth review branch has its
own [qualification manifest](../../../runtime/virtual/results/node-growth/qualification.json),
covering the paged processor paths, growth/reuse and quota refusals, same-mm
pthreads, and the complete runtime gate with SQLite, mruby and Perl smoke.
The manifest distinguishes new QEMU builds from reused source-matched runtime,
compiler and application artifacts. Node exhaustion and failed collection
remain fail-closed, and published pages are retained until namespace teardown.

This does not requalify every application port, the full upstream suites or
all nested-allocator bug corpora. Full physical runtime/thread regression,
the inherited epoll/message mutation matrix, multi-hart and RTL remain outside
this node-growth qualification. Historical runs retain their value for their
recorded revisions; they do not qualify a future merged tree automatically.

[qemu]: https://github.com/project-starch/capstone-qemu/tree/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9
[runtime]: https://github.com/project-starch/llvm-capstone/tree/518a5b805aa70c222ef0496c151b2845b4ab8dae
[memcached]: https://github.com/project-starch/llvm-capstone/tree/1354aee956cd79905a1eba3055aac00dae9f3ebb
[manifest]: https://github.com/project-starch/llvm-capstone/blob/e0313149630450c1a906cc3506b317add64266e0/capstone/docs/plans/virtual-capstone-review-results.json
[spec-patch]: https://github.com/project-starch/llvm-capstone/blob/e0313149630450c1a906cc3506b317add64266e0/capstone/docs/plans/virtual-capstone-isa.patch
