# Runtime terminology glossary

This note is a compact reference for the terms used in the current split host /
domain runtime work. It is grouped by topic so future notes can link here instead
of redefining terms ad hoc.

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

### Two-round protocol
A synchronous request/response sequence:

1. helper enters the domain,
2. domain publishes a request and returns,
3. helper performs the requested service,
4. helper enters the domain again,
5. domain validates the response and finishes.

This is not busy-wait polling. It is a pair of explicit control transfers.

### HostCall proof
A proof that a domain can request a host-side service through the shared-memory
protocol, return control, let the helper perform the service, and then validate the
response on re-entry.

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

