# Mapping blocks in the Linux trap route

Patches 0040 and 0041 implement the mapping portion of plan v5 step 6. Apply
`capstone/kernel/patches/series.s6-mappings` after the base series. The fragment
is separate so the trap and handle streams can integrate their patches without
renumbering this work.

Each block is one depth-1 child of R. The existing table code creates R and
uses one CDERIVE per block. The new `arch/riscv/mm/sublet-blocks.c` reads each
request from the kernel reference slot named by its handle. A request must
carry a tagged, live reference at depth 1 whose bounds cover the requested
pages inside one block. A renewed reference works because the check reads its
current lifetime record. No stored grant reference authorizes a later request.

A partial munmap clears the removed pages in the block's page state. The
block continues to reserve its entire address range, including holes. When a
request removes its last mapped pages, the kernel executes CREVOKE on the
request's reference before Linux removes the pages. The check and CREVOKE run
with interrupts and preemption disabled. The instruction has an exception-table
fixup. A fault refuses the request before page state or page tables change.
The unmap hook runs after Linux's fallible VMA preparation.

MAP_FIXED and growth in place refill pages only if the reference's record has
no deeper successor in the lifetime list. The kernel checks this again after
mapping work that can schedule. A refused refill removes the new mapping and
keeps the range reserved. If Linux cannot prepare that unmap, the kernel
removes access from the overlapping VMAs and quarantines the block until exit.
A merged neighbouring VMA can also lose access on that allocation-failure path.
Replacement first removes the old pages without
ending the block. If allocation then fails, those holes remain reserved. A
failed replacement of all pages can therefore leave an empty live block until
exit. This is a safe leak, not a new mapping at a live block's address.

A moving mremap reserves and derives its new block before moving page tables.
Linux calls the reference check before removing the old mapping. A refused
CREVOKE uses Linux's existing page-table rollback. A move of the block's last
pages revokes its lifetime before the old range becomes available. A partial
move leaves the old block and its remaining pages live. A fixed destination
must be free of both VMAs and reserved blocks. It cannot evict another block
without that block's reference.

The image, main stack, startup and shared blocks are kernel-owned. Their
munmap, mremap, mprotect and madvise requests fail. Runtime mappings cannot
become executable. Linux still owns VMAs, page tables, private anonymous
backing and the existing page-tag operations.

## Interfaces for the other streams

`include/linux/sublet-mappings.h` owns these declarations:

- `sublet_handle_ref(handle, &ref)` returns the address of a tagged 16-byte
  kernel slot. It must not sleep. The calling syscall owns the slot until
  return. Its weak default returns EFAULT.
- `sublet_mapping_return(ref, address)` copies the reference with its tag into
  the saved a0 slot and sets its cursor to `address`. It must not fail or sleep.
  The syscall return path must preserve that reference when the scalar result
  equals the cursor. The symbol is optional at link time. Without it, mmap and
  mremap return EOPNOTSUPP before changing a mapping.
- The loader publishes its `struct sublet_context *` in
  `mm->context.capstone_namespace` only after creating the initial grants.
  Until publication, its ordinary Linux mapping calls bypass these hooks.

The runtime mapping entries take `mmap_write_lock` first. Table allocation
then takes `sublet_lock`. The loader's grant path uses the existing reverse
order only before its context is published. The trap stream must route
supervisor capability faults through the exception-table lookup and preserve
reference tags across traps. The binder must reject other address-changing
calls, including brk and remap_file_pages, until they have equivalent hooks.

The patches retain the base series' two-argument `sublet_grant` interface.
The block structure moves from `sublet-table.c` to
`sublet-blocks-internal.h`. Changes by the loader stream to grant output slots
must be carried into this definition and its wrapper during integration.

## Scope and verification

The runtime path accepts private anonymous mappings. File mappings, shared
mappings, huge pages, userfaultfd mappings and MREMAP_DONTUNMAP are refused.
Madvise accepts MADV_DONTNEED and MADV_FREE only. Its protected path takes the
write lock and rejects userfaultfd before Linux can drop that lock. These
limits preserve the adapter's private-page policy. They do not establish full
Linux syscall coverage.

The host state test passes with a negative control that removes the depth-1
check and must fail. The Linux 6.1 Image builds on the private p13 copy with the
base series and this fragment. The build emits no warnings. The copied tree
is under 3 GB. These results do not qualify trap-mode execution of the mapping
syscalls. That requires the handle, return-slot and trap streams together.
