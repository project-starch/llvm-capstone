# Kernel patches for the trap route

The kernel that runs protected programs as ordinary Linux tasks (plan v5,
[2026-11-13-kernel-trap-entry.md](../../docs/plans/2026-11-13-kernel-trap-entry.md))
lives in `arch/riscv` of the native lane. This directory keeps what that kernel
takes from this repository. The Linux adapter (module, launcher and runners) was
deleted on 2026-11-14. Its sources and evidence stay reachable through the tag
`pre-trap-route-2026-11-14`.

## linux-6.1-unpinned.patch: tags follow physical pages

A complete patch against upstream Linux 6.1 (commit `830b3c68c1fb`). It adds
`arch/riscv/mm/captag.c` and hooks it into `page.h`, `pgtable.h` and the mm
Makefile:

- `copy_highpage` and `copy_user_highpage` move the copied page's capability
  tags with a raw granule loop when the source page's sticky flag is set, and
  redo the pass after a firmware timer interrupt;
- `arch_free_page` clears the tags of every flagged page and checks an empty
  acknowledgement before the page allocator reuses it;
- `arch_prepare_to_swap` refuses a page that may hold tags;
- `clear_page` and `clear_user_page` go through one zero primitive, with an
  optional bulk `cbo.zero` path (`captag_bulk=on`, off by default);
- `captag=off` disables the hooks for negative tests.

The patch enables these hooks only when the device tree has a
`capstone,unpinned-v1` node (QEMU publishes it) with `id-bits` 30, 55 or 63, exact-bounds
0 or 1, the base page size and a 16-byte granule, and refuses KSM and
transparent huge pages. It still exports `capstone_unpinned_config` for the
deleted module; plan v5 step 6 decides whether the built-in code keeps it.

```sh
# In an otherwise unmodified Linux 6.1 tree:
git apply capstone/kernel/patches/linux-6.1-unpinned.patch
```

The adapter's README at the tag records the patch's qualification (bulk probe,
firmware-interrupt window, lifecycle double).

## Headers the kernel's Sublet code needs

The deleted adapter build copied two headers next to the module sources. The
kernel build of plan v5 step 6 needs the same two, and no script copies them
any more. Copy both into the kernel's Sublet directory from the revisions the
kernel is built against:

- `capstone/capstone-qemu/target/riscv/cap_rev_table_abi.h` (the lifetime table
  layout and the growth protocol, from the QEMU that runs the kernel);
- `capstone/runtime/include/capstone/representable.h` (the representability
  rule, shared with the allocator and the libc, which include it directly).

## Kept tests that plan v5 ports

`capstone/runtime/virtual/root-contract.c` is the only test of the kernel-owned
flag (the image, main stack, shared region and startup block refuse munmap,
mprotect and madvise) and of the CREVOKE that ends a block when its last mapped
page goes. Its runner was deleted with the adapter. Plan v5 step 6 ports the
program to the trap route.
