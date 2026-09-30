# Protect libc malloc/free with the existing Sublet heap

Status: PLAN, 2026-09-30. The allocator and build selection already exist.
This plan adds no implementation or test result. Its first milestone is to
verify ordinary libc heap protection with the smallest necessary changes.

## 1. Goal and contract

Application code continues to call `malloc` and `free` normally. Each
allocation has bounded, copyable capability pointers. After `free` completes,
all aliases of that allocation are invalid, including when its address is
reused. A stale `free` must not revoke a newer allocation at that address.

Use the alias path of Sublet. The allocator retains a revocation handle and
is trusted: revoking non-linear aliases can return readable authority. This
is protection against stale application pointers, not confidentiality from a
malicious allocator. Objects carved inside an application's own pool share
that pool's lifetime unless the application adds its own protection; they
are outside this milestone.

For small allocations the bounds are byte-exact. Larger allocations may
require representable padding, contained within their owned buddy block.
Do not claim that every access past the requested size faults when it falls
inside that padding.

## 2. Reuse the existing implementation

[sublet_heap.c](../../ports/musl-capstone/runtime/sublet_heap.c) already
provides the required path:

- `malloc` takes a buddy block, creates the allocation's revocation handle,
  and returns an alias narrowed to its representable size.
- `free` checks access through the supplied pointer before using its address
  to find the block, scrubs through that pointer, revokes the allocation,
  and returns the block to the buddy. Metadata stays outside the object.
- `calloc` uses this heap and zeroes its result. `realloc` allocates, copies
  with capability-preserving `memmove`, and frees only after allocation
  succeeds. Keep these paths and the libc internal aliases consistent.
- `free(NULL)` remains a no-op; `malloc(0)` keeps its current one-byte policy.

[Application.cmake](../../runtime/cmake/Application.cmake) already selects
this source with `HEAP sublet` and a configured `HEAP_LOG`. Use that selection
for the qualification target and retain `HEAP level0` as a test control.
Check the linked allocation symbols so public calls and musl's internal calls
reach the selected heap. Application source changes are not required.

Keep the buddy layout, current backing pool, metadata tables and
[Sublet primitives](../../sublet/sublet.h). Change runtime code only to fix
a concrete failure of the contract above. This milestone does not require
new size classes, a pagemap, a mallocng port, compiler changes or translated
memory. It qualifies the current single-threaded use; concurrent allocation
requires a separate synchronization contract.

## 3. Focused acceptance checks

Reuse existing fixtures where they cover these cases. Faulting accesses run
in separate attempts from their positive controls; a timeout or unrelated
fault does not count as a successful rejection.

| Check | Required result and control |
|---|---|
| Ordinary allocation | Read/write the first and last requested bytes of a small allocation, then free it successfully. |
| Bounds | An access just beyond that small allocation faults; the last byte succeeds. For larger allocations, check the representable bound and separation from neighbours. |
| Free and address reuse | Keep an alias, free the allocation, obtain the same address again, and show that the old alias faults while the new pointer works. Confirm the address reuse. |
| Stale free | A second free, including after address reuse, faults before revoking another allocation's handle. A valid free succeeds. |
| Adjacent live allocation | Freeing one block leaves its live neighbour readable and unchanged. |
| Companion APIs | Check `free(NULL)`, `malloc(0)`, zeroed `calloc`, and `realloc` content preservation, including tagged pointers. An allocation failure in `realloc` leaves the original usable. |
| Integration | One existing application smoke completes with the protected heap; verify its linked heap selection. |

Record the actual runtime and hardware or emulator versions. QEMU's exact
bounds side table is not evidence for compressed store/reload behavior;
check that property against the encoder and RTL before making a hardware
claim. No application-wide benchmark campaign is required for this milestone.

## 4. Next action and completion

1. Select the existing Sublet heap for the qualification target and verify
   symbol resolution.
2. Run the focused checks, adding only missing contract cases.
3. Fix only demonstrated gaps and update misleading comments alongside the
   verified results. Two are known already: the heap's header comment still
   says a stale capability's load retires on the RTL and that the node pool
   is 65,532 per boot, both overtaken (ISSUES R-35 and R-45; the emulator's
   in-process node reuse under
   `runtime/tests/application/results/20260927-node-reuse/`); and the bounds
   model (`docs/design/capability-bounds-model.md`) writes the compression
   exponent with `ceil` where the encoder (`cap_compress.c:39-43`) and
   `sh_narrow` use the highest set bit, `floor`. Stop when the contract and
   application smoke pass on the named target.

The current minimum buddy block is 256 bytes. Accept that memory cost for
this first milestone. It is a layout choice, not a requirement imposed by
capability compression. Splits depend on free-list state; every allocation
does not necessarily create exactly two nodes. If memory use later prevents
a required workload from running, measure it and choose a separate change
then. Slabs and size-class tuning are not prerequisites for malloc/free
protection.

## 5. Deferred: size classes below the atom

Recorded so the reasoning is not redone when memory use forces the change.

- The waste is the atom: `CAPSTONE_SUBLET_ATOM_LOG` is 8, so a 24-byte
  object takes a 256-byte block. Interpreter heaps are mostly below that.
- The representable construction below the atom is a slab of exactly one
  page inside the buddy, carved lazily into slots of one stride. Bounds are
  byte-exact below 4096 bytes, so every stored intermediate of the carve is
  representable without a further rule. Larger slabs would tie strides to
  the grain `2^(E+3)`, `E = floor(log2 len) - 12`.
- Fixed strides over a larger region do not help: carving `[0, 48)` from a
  48 KiB region leaves `[48, 49152)`, whose grain is 64 bytes and whose
  stored encoding is `[0, 49152)`, over the object just handed out. Every
  stored intermediate must be representable, not only the finished slots.
- musl's mallocng, once it builds for a 16-byte pointer, is the unmodified
  control of the level0 class, not a safe allocator: it revokes nothing on
  `free` and reads its metadata from the bytes before the pointer. Giving it
  the discipline would be a new allocator with mallocng's size classes.
- Any slab code needs a native harness around the emulator's
  `cap_compress.c` run over the construction sequence before measurement,
  because QEMU's exact side table hides the rounding. A first-fit carve is
  the harness's positive control.

## 6. Pointers

This is phase 2 of the bounded-heap work
(`docs/design/heap-temporal-safety-revoke-on-free-proposal.md`, suspended
2026-07-06) as the Sublet primitives made it: the existing
`sublet_heap.c` is the implementation that proposal lacked. `govslot`
(tag `2-reference-model`) is a different mechanism, lifetimes for
capability-oblivious Linux processes; its executable-model method is worth
borrowing for §5, its mechanism is not.
