# M2: the delivery transport for translated mappings

Status: PLAN, 2026-09-30, lane `delegation-memory-qemu` (parent), with
`sbi/mapping-transport` (caplifive-sbi, from a810177d, the parent's pinned
monitor), `buildroot/mapping-transport` (caplifive-buildroot, from d60365e3)
and `opensbi/mapping-transport` (caplifive-opensbi, from 853197a1). It
connects the Stage-1 instructions of [M1](mapping-qemu-stage1.md) to the
delegated runtime: a domain's anonymous `mmap` becomes a mapping the monitor
creates and populates, delivered into the domain's context, and `munmap`
becomes detach, destroy and reclamation. Gates are the physical plan's
[M1 and M2 lists](delegation-memory.md#8-implementation-order-and-acceptance-gates),
read against translated memory.

## 1. The runtime today, as read on 2026-09-30

| Fact | Where |
|---|---|
| A syscall row is a 96-byte entry at offset 0 of the 16 KiB META region; buffers travel through the exchange region as offsets; the libc's `dl_round` writes the entry and yields with `domreturn` to `.Lyield_resume` | `runtime/include/capstone/delegate.h:19-41`, `ports/musl-capstone/runtime/delegate.c:142-153`, `start-musl.S:297-341` |
| The launcher services rows in `capstone_delegate_serve`; MEMORY-group numbers are refused with ENOSYS by `capstone_delegate_validate`, and `mmap`/`munmap` are libc overrides backed by `malloc`, so they never reach a row | `runtime/linux/delegate-service.c:482-539`, `runtime/common/delegate.c:270-300`, `ports/musl-capstone/runtime/mmap_shm_level0.c`, `libc_overrides.list:5` |
| The domain resumes through the monitor's `__domcall(d, DPI_CALL, result)` after STEP: a0 = 0, a1 = a capability to `supervised_results[id]`, ra = a fresh sealed return; the resume path overwrites x1-x5, s0-s11, a0, a1 and leaves a2-a7 and t1-t6 untouched until it returns into C | monitor `supervised_call` (764-782), `start-musl.S:343-383` |
| The recovery context (256 bytes, in cscratch) uses slots 0, 16, 32, 48, 64 and 80; 96 onwards are free | `start-musl.S:78-112`, `:343-354` |
| The monitor keeps per-domain state in arrays indexed by domain slot (32): `domains[]` (the SEALED handle), `managed_domain_root[]` (a `__rev` array), and per region (96): `regions[]`, `managed_region_root[]`, `managed_region_owned[]` | monitor 307-336, 684-690 |
| Managed regions come from `dma_alloc_pages` in the module (contiguous, CMA above 4 MiB) and `SBI PROCESS_REGION_CREATE`; the monitor carves them with `split_out_cap` and keeps an MREV root; reclamation is `managed_reset_region`: revoke, scrub with `stc`, `csinit`, re-MREV; regions return to the process cache, never to Linux, until the file closes | module `capstone.c:340-435`, monitor 704-722, 1953-1985 |
| A domain's sealed context is 0x600 bytes; the M1 delivery slot at 0x3B0 and its register index at 0x3C0 fit | monitor 293-294, 1206-1277 |
| capstone-c: arguments in a0.., return in a0, `__linear`/`__rev`/`__dom` types, `__split`, `__mrev`, `__revoke`, `__delin`, `__seal`, inline asm with `r` constraints only, globals at most 2048 bytes | monitor throughout, `capstone_target.h` |
| The process ABI numbers 0x21-0x2a are taken; an unpinned branch takes 0x2b and ioctls 14 and 15 | module `include/process-abi.h`, `capstone.h:92-127` |
| Firmware builds standalone: capstone-c turns `sbi_capstone_dom.c` into assembly, OpenSBI links it with the Buildroot cross toolchain; the module builds out of tree against `build-qemu/build/linux-custom`; `capstone-exec` is a CMake target over `runtime/linux`; domains are applications built with `capstone_configure_application`; the gate boots the images with `run-domain-smoke.py` | Buildroot `Makefile:186-195`, `package/modcapstone/module/Makefile`, `runtime/exec/CMakeLists.txt`, `runtime/cmake/Application.cmake` |

The firmware built this way from the M2 monitor (`sbi/mapping-transport`
7c6ba0b5) boots the gate images and passes the SQLite memory gate.

Correction, 2026-09-30: an earlier version of this paragraph claimed the same
for the unchanged monitor, "verified before any change". That run had booted
the pinned snapshot firmware, not the rebuilt one: `run-domain-smoke.py` reads
`CAPSTONE_BUILDROOT_DIR/build/images/fw_jump.elf`, and the rebuilt file had
been copied into a sibling `images/` directory that nothing reads. The claim
was unverified and is withdrawn. The gate directory now carries its own
`build/images` with the lane's firmware and links to the snapshot's kernel and
root filesystem; a positive control with a truncated `fw_jump.elf` at that path
fails the gate (exit 75, no serial output), so the gate does boot what it is
given.

## 2. Protocol

**Rows.** Two special numbers after the signal rows:

| Row | Arguments | Result |
|---|---|---|
| `CAPSTONE_NR_MAP_GRANT` | `len` (bytes, page multiple), `prot` (R or RW) | the mapping's binding word (`id | gen << 12`), or `-errno`; the mapping capability arrives in a2 at resume |
| `CAPSTONE_NR_MAP_RELEASE` | the binding word | 0 or `-errno` |

**Delivery.** CREATE names a2 (x12) as the delivery register. `.Lyield_resume`
stores a2 into recovery slot 96 before anything else runs; `dl_round`'s caller
takes it with `__capstone_map_take()`, which loads the slot, clears it, and
checks tag, linear type, binding equal to the row's result, base and length;
anything else is "no delivery" and the row fails with EIO. The launcher and the
monitor never hold the mapping capability: CREATE writes it into the context.

**Launcher.** `capstone_delegate_serve` handles the two numbers: for GRANT it
rounds `len` to pages, computes the chunk size (frames plus one leaf table per
256 pages plus the root page), creates a managed region of that size through
the module (`REGION_CREATE`), then calls the new `IOCTL_MAP_GRANT {dom_id,
region_id, len, prot}`; the result is the binding word or an error, and on
error the region is reset. For RELEASE it calls `IOCTL_MAP_RELEASE {dom_id,
binding}`.

**Module.** Two ioctls (20 and 21) forwarding to two SBI functions (0x30 and
0x31) with the same arguments; the module records which region backs which
mapping so that `process_release` resets it like any other region.

**Monitor.**
- `map_grant(dom, region, len, prot)`: checks the region is managed, owned and
  large enough; splits the chunk into the root page, the leaf table pages and
  the frames in that order; allocates a logical range by the E6 rule (size the
  next power of two above `len`, base a multiple of twice that, from a bump
  allocator over the logical region that never reuses a range within a boot);
  executes `csmapcreate` with the domain's SEALED handle and delivery register
  12, then `csmappopulate` for every frame, supplying a leaf page whenever the
  path has none; keeps the detach handle, the region id, the range and the
  domain in a mapping table of 32 entries; returns the binding word read with
  `lcc 8` from the detach handle. The chunk's MREV root, kept by
  `managed_create_region`, is the revocation-from-above handle for the whole
  mapping.
- `map_release(dom, binding)`: finds the entry, `csmapdetach`, `csmapdestroy`,
  then reclaims the chunk with the existing `managed_reset_region` path
  (revoke the root, scrub, re-arm), and frees the entry.
- The instructions and their fixed-register operands live in assembly
  functions in `sbi_capstone.S`, called from C with the C convention so that
  a0-a7 carry exactly what the instruction reads; the SEALED handle is written
  back through a capability pointer because passing it moves it.

**Libc.** `mmap(NULL, len, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)` issues
GRANT and returns the delivered capability's cursor; other flags keep today's
behaviour (`MAP_FIXED` EINVAL, files ENODEV, hugetlb ENOMEM); `munmap` of a
whole mapping issues RELEASE and drops the record; partial `munmap` is EINVAL
as today. The System V shm emulation stays `malloc`-backed. A mapping table of
32 entries, as today.

## 3. Deviations from the design, recorded

Frames come from one contiguous chunk per mapping, not scattered pages; the
scattered case is exercised by M1's bare-metal tests. UNMAP is not used: a
release detaches, destroys and revokes the chunk from above, which returns the
whole chunk as UNINIT for the existing scrub. Logical ranges are never reused
within a boot. One mapping spans at most 256 MiB (M1's two-level tables) and
one region (the module's CMA limit applies).

## 4. Order and gates

1. Plan (this document).
2. Monitor: mapping table, allocator, `map_grant`, `map_release`, SBI 0x30
   and 0x31, assembly wrappers; firmware builds; the SQLite gate still passes.
3. Module: the two ioctls; the module loads from the share directory in the
   gate image.
4. Libc: row numbers and shapes, the a2 stash, `__capstone_map_take`, the
   `mmap`/`munmap` overrides; the application core rebuilds.
5. Launcher: the two row services; `capstone-exec` rebuilds and runs from the
   share directory.
6. Contract test `mapping-contract.c` as an application: acquire, write and
   read both endpoints and every page, zero fill, release, then use a saved
   alias and expect the fault the runtime reports; acquire again and see fresh
   zeroed pages; a second live mapping survives the first's release; error
   paths: length 0, unaligned protection, exhaustion of the mapping table,
   release of an unknown binding, release twice; a `munmap` of a partial
   range refused. Then the SQLite memory gate and the application gate on the
   new firmware, module and launcher; pins bumped on the parent lane.

Not in M2: growth of a mapping, shared mappings, file mappings, delivery to a
preempted domain (the paused-resume path), and any change to the allocators
(M3).
