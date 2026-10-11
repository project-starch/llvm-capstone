# Protect libc malloc/free with Capstone

Status: QEMU qualification completed, 2026-09-30, within the scope below.
The allocator and build selection already existed; this milestone adds
contract tests and checks their evidence without changing allocator behavior.

## 1. Goal and contract

Application code continues to call `malloc` and `free` normally. Each
allocation has bounded, copyable capability pointers. After `free` completes,
all aliases of that allocation are invalid, including when its address is
reused. A stale `free` must not revoke a newer allocation at that address.

Use Capstone's non-linear capabilities for ordinary, copyable C pointers.
The allocator retains a revocation handle and is trusted: revoking
non-linear aliases can return readable authority. This
is protection against stale application pointers, not confidentiality from a
malicious allocator. Objects carved inside an application's own pool share
that pool's lifetime unless the application adds its own protection; they
are outside this milestone.

For small allocations the bounds are byte-exact. Larger allocations may
require representable padding, contained within their owned buddy block.
Do not claim that every access past the requested size faults when it falls
inside that padding.

## 2. Reuse the existing implementation

The [existing heap implementation](../../ports/musl-capstone/runtime/sublet_heap.c) already
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
this source with the current option `HEAP sublet` and a configured `HEAP_LOG`.
The implementation names remain unchanged. Use that selection
for the qualification target and retain `HEAP level0` as a test control.
Check the linked allocation symbols so public calls and musl's internal calls
reach the selected heap. Application source changes are not required.

Keep the buddy layout, current backing pool, metadata tables and
[capability operations](../../sublet/sublet.h). Change runtime code only to fix
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

1. Select the existing Capstone-protected heap for the qualification target and verify
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

### Result, 2026-09-30

The milestone ran on QEMU
([checked record](../../runtime/tests/application/results/20260930-heap-qualification-on-dev.json),
runner `runtime/tests/application/run-heap.py`). Both images built from this
tree's `contract.c`, one with `HEAP sublet`, one with `HEAP level0`; the
symbol check matches all six allocation entry points, including their ELF
addresses and sizes, to their input objects in each image's linker map.
Guest image hashes must match the checked host ELFs.

| Case | `HEAP=sublet` | `HEAP=level0`, the control |
|---|---|---|
| fault-stale, fault-reused, fault-bounds, fault-bounds-large, fault-double-free, fault-double-free-reused | SIGSEGV at the intended byte probe, PASS | operation survives, sentinel exit 90; supervisor FAIL as required |
| healthy, churn, heap-bounds, heap-neighbour, heap-companion | PASS | PASS |
| the supervisor's own application sequence | PASS | PASS |

The added cases cover the table of §3: the byte past a 24-byte and a
5000-byte allocation, a double free, a double free after the address was
reissued, a live neighbour across a free and the freed address returning,
and `free(NULL)`, `malloc(0)`, zeroed `calloc`, `realloc` keeping bytes and a
stored capability, and a failed `realloc` leaving the original usable. No gap
was demonstrated, so no runtime code changed; the two comments named in
step 3 were corrected.

The runner requires the pre-operation marker, exact output and wait status,
and a launcher fault record identifying the checked ELF. Bounds failures
must be QEMU load access faults (cause 5); stale pointers must fail the tag
or node check (24 or 25). Both are checked at the exact load instruction;
stale `free` must stop at its initial byte probe, before allocator metadata
is accessed. Every unprotected control must print its survival marker and
exit 90, so allocation failure or failure to reuse the address cannot pass.
The protected churn and stale-after-churn attempts each allocated 200,064
nodes; the runner requires at least 200,000. Cleanup counters return to zero.

This record supersedes the
[initial run](../../runtime/tests/application/results/20260930-heap-qualification.json).
Its runner accepted any control FAIL, any protected SIGSEGV, and a matching
object filename in the build directory. Those were insufficient oracles.
Sixteen host regression tests now reject setup errors, unrelated faults,
insufficient progress and incorrect link provenance, with passing controls;
the existing 21 host tests also pass. These are test-instrument fixes, with
no change to the allocator or to the qualification's platform scope.

One instrument finding: on the emulator without in-process node reuse
(capstone-qemu 6550f194) the existing churn case faults with cause 30,
INSUF_RESOURCES, after 65,228 node allocations, on this image and on the
2026-09-27 image alike. The qualification requires the emulator the tree
pins, 22aec7ee, and the record names it.

Re-run after merging dev 884d9434 into the lane, on the platform dev pins
(capstone-qemu ac2837aa, the module of caplifive-buildroot d60365e3, the
capstone-sbi a810177 firmware, launcher and images from the merged tree with
the intcap-capable toolchain the SDK now requires): every check passes
unchanged, churn allocates 200,064 nodes. That record is the current one;
the two earlier records of the pre-merge platform are kept and marked
superseded.

Not covered: silicon, where the compressed-store rounding and the R-35/R-45
fixes would have to be exercised; threads; the ports' own allocators.

### Result, 2026-10-01: the default level0 heap as a third arm

Since #170 the `level0` heap that applications link bounds each allocation, yet
no gate ran it: the runner's two arms were Sublet and the unprotected control.
The delegated threads stack then let level0's `realloc` shrink a block in place
and release the tail, which the next allocation may take, and neither change
had been tested with the other. The runner now qualifies three images
([record](../../runtime/tests/application/results/20261001-heap-three-arms.json),
on the threads stack with dev's #169 and #170 merged):

| Case | `sublet` | `level0` | `control` |
|---|---|---|---|
| fault-bounds, fault-bounds-large, fault-realloc-shrink | SIGSEGV, cause 5, at the probe | SIGSEGV, cause 5, at the probe | survives, exit 90 |
| fault-stale, fault-reused | SIGSEGV, cause 24, at the probe | survives, exit 90 | survives, exit 90 |
| fault-double-free, fault-double-free-reused | SIGSEGV, cause 24, at the probe in `sh_free` | survives, exit 90 | survives, exit 90 |
| healthy, churn, heap-bounds, heap-neighbour, heap-companion, heap-realloc-shrink | PASS | PASS | PASS |

`fault-realloc-shrink` shrinks a 4096-byte block to 24 bytes and reads one byte
past the new end; `heap-realloc-shrink` checks that a shrink keeps the bytes and
a stored capability, and that an allocation after it leaves the shrunk block
intact. The check fires: a level0 whose in-place shrink keeps the old pointer
fails the gate at `level0 fault-realloc-shrink`, the read surviving to exit 90.
On the stack, `free` takes the heap lock and frees in `sh_free`, whose first act
is the stale-pointer probe, so the runner takes the probe from there.

Not covered here: silicon, and the CPython and GLib threading gates on this head.

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
  `free` and reads its metadata from the bytes before the pointer. Adding
  per-allocation capability bounds and revocation would require changing its
  metadata and allocation paths.
- Any slab code needs a native harness around the emulator's
  `cap_compress.c` run over the construction sequence before measurement,
  because QEMU's exact side table hides the rounding. A first-fit carve is
  the harness's positive control.
