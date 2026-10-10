# Runtime terminology glossary

This note is a compact reference for the terms used in the current split host /
domain runtime work, and in section 6 for the allocator work (nested allocators,
Sublet, the ports and the bug corpora). It is grouped by topic so future notes can
link here instead of redefining terms ad hoc.

## 1. Execution layers and actors

### Developer machine
The real workstation where the repository is edited and where QEMU is launched.

### Guest
The whole virtual machine booted by QEMU:

- OpenSBI,
- Linux kernel,
- Buildroot root filesystem,
- guest Linux userspace,
- `capstone.ko`,
- `/dev/capstone`.

### Guest runtime world
The active execution world inside that guest VM. In practice this means the
firmware, kernel, device interfaces, guest userspace helpers, and the domain
runtime path taken together.

### Host
In the current split-runtime notes, “host” usually means the ordinary Linux
userspace helper running **inside the guest image**, not the developer's physical
workstation.

### Helper
A guest-side Linux userspace program that bridges ordinary Linux services and the
Capstone domain runtime. A helper can create domains, create/share/map regions,
call domains, inspect requests, and write responses back.

### Domain
An isolated Capstone payload executed through the Capstone runtime path rather
than as a normal Linux process ABI.

### `sbi.dom`
A reusable domain-side substrate installed into the guest under `/test-domains/`.
It provides the Capstone C-domain / reentry scaffolding used by split `.smode`
experiments.

### `smode` / `.smode`
S-mode (Supervisor mode) companion payload code used in the split-domain path.
The `.smode` suffix is a local naming convention for such payloads.

## 2. Region and memory-sharing terms

### Region
A runtime-managed memory object identified by a `region_id`.

In the current implementation, the allocation path is:

1. the helper calls `create_region(len)` from guest userspace,
2. `libcapstone` sends `IOCTL_REGION_CREATE` to `/dev/capstone`,
3. the kernel module allocates guest pages with `__get_free_pages(...)`,
4. the kernel passes the physical address to OpenSBI via `SBI_EXT_CAPSTONE_REGION_CREATE`,
5. OpenSBI records the region and returns a `region_id`.

So the region is not allocated by `malloc()` in the helper and not allocated on the
developer machine. It is guest memory allocated by the guest kernel on behalf of
the helper request.

### Helper mapping of a region
`map_region(region_id, len)` maps those same guest pages into the helper's Linux
virtual address space via `mmap()` on `/dev/capstone`.

So when notes say that a region “maps into the helper virtual address space”, that
means the helper receives a userspace mapping of the already-created guest pages.
It does **not** mean the helper is the ultimate allocator of a separate copy.

### Shared region
A region that has been shared with a domain. After sharing, the helper-side mapping
and the domain-side capability refer to the same underlying guest memory, subject
to permission and revoke rules.

### `shared_region_annotated(...)`
The helper uses `shared_region_annotated(dom_id, region_id, perm, rev)` to share a
previously created region **with the specified domain**.

In simple terms:

- helper side: already has a Linux mapping of the region,
- domain side: receives access to that same region through the Capstone runtime,
- runtime: applies the chosen permission (`IN`, `OUT`, `INOUT`, ...) and revoke
  policy (`SHARED`, `BORROWED`, ...).

So the data is shared between the helper and the domain, not between two unrelated
Linux processes.

### Metadata region
The shared region that stores the fixed-width protocol header such as `phase`,
`opcode`, `offset`, `length`, `result`, and `error`.

### Payload region
The shared region that stores the actual request bytes or response bytes.

## 3. Ownership and permission terms

### Ownership discipline
The explicit rule for which side is allowed to write, read, retain, or stop using a
shared buffer at each protocol step.

### Disciplined protocol
A protocol whose state transitions and buffer usage follow a narrow, explicit set of
rules, instead of letting both sides read/write everything all the time.

### Permissive sharing
A broad sharing mode that gives both sides more freedom than they strictly need.

Example: `INOUT + SHARED` for a payload buffer means both sides can keep accessing
the same buffer across rounds. That is convenient for bring-up, but looser than the
current stdout proof's tighter payload model.

### Stricter sharing
A more constrained sharing mode that gives each side only the access it actually
needs.

For example, the current stdout payload is written by the domain and only consumed by
the helper, so it now uses a one-direction borrowed buffer instead of broad shared
read/write access.

### Borrowed region / borrowed handoff
A region shared with post-return revoke enabled. The receiving side gets temporary
access for one step/round rather than indefinite shared access.

### One-direction borrowed handoff
A borrowed sharing pattern where data should flow in only one direction:

- producer side writes or provides the data,
- consumer side reads/consumes it after control returns,
- the runtime revokes that temporary access when the round completes.

This is the intended meaning behind phrases such as “payload becomes one-direction
borrowed”.

### `SHARED` vs `BORROWED`
- `REV_SHARED`: post-return revoke disabled; the region remains broadly shared.
- `REV_BORROWED`: post-return revoke enabled; access is intended to be temporary.

### `IN`, `OUT`, `INOUT`
Local shorthand for the permission annotation passed to
`shared_region_annotated(...)`:

- `IN`: the domain receives read-like access,
- `OUT`: the domain receives write-like access,
- `INOUT`: the domain receives both.

The exact capability mechanics live in OpenSBI, but the practical design intent is
least privilege.

## 4. Control-transfer and protocol terms

### `call_dom()`
A helper-side userspace API that asks the runtime to enter a domain and returns
when the domain executes `DOM_RETURN(...)`.

### `DOM_RETURN(...)`
The domain-side handoff back to the helper. It returns control plus a small scalar
status code such as DONE, PENDING, or ERROR.

### Requested service
The host-side operation encoded in shared metadata, for example
`HC_V0_OP_WRITE_STDOUT`.

Important nuance: this should usually mean a **coarse host service**, not a perfect
1:1 mirror of one libc symbol or one Linux syscall.

For example, a single HostCall service may be implemented inside the helper with
multiple ordinary Linux calls such as `open(...)`, `write(...)`, and `close(...)`.

### Two-round protocol
A synchronous request/response sequence:

1. helper enters the domain,
2. domain publishes a request and returns,
3. helper performs the requested service,
4. helper enters the domain again,
5. domain validates the response and finishes.

This is not busy-wait polling. It is a pair of explicit control transfers.

### HostCall v0
The bare-domain transport this section describes: a metadata region and a payload
region, a two-round protocol, coarse services such as `HC_V0_OP_WRITE_STDOUT`
(`runtime/include/capstone/hostcall.h`). The S-mode wire probes
(`tests/runtime-qemu/hostcall-*-probe`) and the FPGA gates use it. It is not an
application ABI: the musl runtime's HostCall v0 mode, which emulated files, pipes,
the working directory and path operations inside the domain, was removed on
2026-09-30.

### Delegated runtime (application ABI v2)
The only application runtime. A musl application's Linux calls cross one at a time:
the domain fills an entry block, copies pointer arguments into an exchange region and
yields, and the launcher's task (`capstone-exec`) runs the call as Linux. An image
without the v2 descriptor is refused. See `runtime/applications.md` and
`plans/delegation-abi.md`.

### HostCall proof
A proof that a domain can request a host-side service through the shared-memory
protocol, return control, let the helper perform the service, and then validate the
response on re-entry.

Direction reminder:

- the domain still initiates the request,
- the helper performs the host-side work,
- the helper does not “ask the domain to do Linux work for it”.

### Tighten the HostCall proof
Keep the same basic host/service flow, but make the ownership and permission model
stricter so the proof is closer to the intended long-term ABI.

In the current workspace this specifically meant:

- metadata stayed `INOUT + SHARED`,
- payload moved from broad shared access to `OUT + BORROWED`,
- the same stdout wrapper was then revalidated successfully.

## 5. Validation and planning terms

### Probe
A narrow diagnostic or proof-of-correctness experiment used to validate one exact
runtime or ABI hypothesis.

### Proof
A probe that has been run successfully enough times to support a concrete
engineering claim. In this workspace a “proof” is narrower than a full feature;
it proves one specific contract.

### Revalidate it
Re-run the same wrapper/proof after an ABI or permission change to confirm that the
intended contract still works in the live runtime.

### Next small host service
The next narrowly scoped host-side operation after stdout, for example a very small
buffered write-like or file-related service, added only after the current proof is
stable.

### `WRITE_GUEST_TMPFILE`
The current second proof opcode meaning “helper, take these payload bytes and commit
them into a fixed tmp file inside guest Linux”.

The name does **not** mean that the domain is exposing guest Linux syscalls of its
own. It only records where the helper-side file exists.

### Reverse-direction payload proof
A proof where the request is still domain-initiated, but the helper supplies payload
bytes back to the domain as the response body.

This is the natural read-like counterpart to the already validated output-like proofs
where the domain supplies payload bytes to the helper.

### Runtime/ABI shaping
The phase where the project is still deciding and validating the exact runtime
contracts, ownership rules, and boundary-crossing protocol rather than treating
those interfaces as frozen.

### Compatibility-oriented hosted mode
A future hosted Linux mode intentionally shaped to stay close to the existing
RISC-V Linux userspace ABI, in order to reuse more of the current kernel/libc/sysroot
stack.

### Native Capstone Linux ABI
A future hosted Linux mode with a genuinely Capstone-specific userspace ABI. This
would require a coherent agreement across compiler ABI, pointer model, loader, crt,
libc, syscall ABI, kernel user ABI handling, and related Linux runtime surfaces.


## 6. Allocator terms

The allocator work (Sublet, the ports under `ports/`, the corpora under
`bug-corpora/`) uses the six words below, each with one meaning. The programs we port
use the same words for other things, so **an upstream name always carries its
program**: aset chunk, pymalloc block, pymalloc pool, wmem block, memcached page,
APR memnode. A bare *block*, *object* or *pool* means the term defined here.

### System allocator
The allocator that `malloc` and `free` resolve to: glibc on Linux, CheriBSD's libc,
and in a Capstone domain one of the two heaps in `ports/musl-capstone/runtime/`:

- the **first-fit heap** (`level0.c`): first fit over one static arena, per-object
  bounds unless built with `-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0`, no revocation;
- the **Sublet heap** (`sublet_heap.c`): a buddy heap over a granted region,
  per-object bounds, and every free revokes.

The paper and older notes call the system allocator *level 0*. In code,
`level0` names the first-fit heap only (`level0.c`, `CAPSTONE_LEVEL0_*`, the
`*_HEAP=level0` knob values), not the system allocator in general.

### Nested allocator
An allocator that gets its memory from below (the system allocator, another nested
allocator, or the kernel or the host directly) and hands out pieces of it to the
program. PostgreSQL's memory contexts, pymalloc, APR's pools and bucket allocator,
nginx pools, memcached's slabs, FFmpeg's buffer pools, wmem, the Perl SV arenas,
mruby's and MicroPython's GC heaps, SQLite's memsys5 and lookaside, ggml contexts.
Where the memory comes from does not decide it: pymalloc is a nested allocator
whether its arenas come from `mmap` or from `malloc`.

Not *custom allocator* (CMASan's term; cite it as theirs), *inner allocator*,
*sub-allocator* or *manager*.

### Block
What a nested allocator gets from below in one request: one `malloc`, one `mmap`,
one piece of a granted region. The aset block, the pymalloc arena, the wmem block,
the memcached page, the APR memnode and the mruby heap page are blocks in this sense.
Not *malloc block* or *backing allocation*.

### Object
What a nested allocator hands to the program: an aset chunk, a pymalloc block, a wmem
chunk, a memcached item, an mruby RVALUE, an SV head. When an overflow matters, say
which boundary it crosses: the object (the bytes requested), the allocator's rounding
of it (aset chunk, pymalloc size class), or the block.

### Pool
A group of objects the program releases together with one call: a PostgreSQL memory
context (reset, delete), an APR or nginx pool (clear, destroy), a wmem allocator
(`wmem_free_all`), an AVBufferPool (uninit), a ggml context (reset, free). A nested
allocator without such a call has no pools; pymalloc's are size-class pages, so
write *pymalloc pool*.

### Release
The end of an object's lifetime. *Free* is the call; *release* is the event. An
object is released **one by one** (its own free, or a garbage collector's sweep) or
**with its pool** (reset, clear, destroy).

A release is **invisible** when it never reaches the system allocator. Invisible
releases are what the project protects: the system allocator, and every defense
built on it (ASan, MTE, CheriBSD's revocation), never sees them.

Not *inner release*, *bulk free*, or *the allocator hides it*.

### Arm
One configuration a workload or a corpus case runs under. Not *variant*, *protection
mode*, *heap mode* or *adapter mode*; an adapter's numeric knob stays *mode 0/1/2*.

An arm that changes only the system allocator is named `sysalloc-<protection>`:

| Arm | System allocator | FFmpeg, memcached, tshark | mruby, Perl, `build.py --heap` | heap qualification |
|---|---|---|---|---|
| `sysalloc-none` | first-fit heap, no per-object bounds | `level0` | — | `control` |
| `sysalloc-bounds` | first-fit heap, per-object bounds | `shrink` | `level0` | `level0` |
| `sysalloc-sublet` | Sublet heap | `sublet` | `sublet` | `sublet` |

The knob values in the middle columns are today's code. The same value means
different arms in different ports (`level0` is `sysalloc-none` in one column and
`sysalloc-bounds` in the next), which is why results name the arm, not the knob.
Per-object bounds became the first-fit heap's default with PR #170 (on `dev`
2026-10-01); before that, `level0` had no per-object bounds in any port, so a
`level0` result recorded earlier is `sysalloc-none`.

An arm that changes a nested allocator keeps its corpus name: `spatial` (per-object
bounds, no revocation), `sublet` (the allocator's Sublet port), and, in PostgreSQL's corpus only since
2026-10-10, `poisoncap-spatial` and `poisoncap-protected`. `cheribsd-revocation` is CheriBSD's system allocator with
revocation on and the nested allocator unchanged. An arm that changes both says
so: mruby's `sublet-gc` and tshark's `chunks` are `sysalloc-sublet` plus that
program's Sublet port (mruby's GC object slots, wmem's block allocator).

### Revocation capability, alias
The paper's names for what the Sublet API calls a handle (`sublet_handle`) and for
the copyable capability `sublet_take` returns, which the program receives. Not
*token* or *lease* in prose; the API identifiers keep their names.
