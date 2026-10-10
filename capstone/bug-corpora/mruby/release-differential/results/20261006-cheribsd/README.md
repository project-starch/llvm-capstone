# Results, 2026-10-06 -- CheriBSD purecap, revocation at its default

The same 23 cases, the same source pin, on CheriBSD 15.0-CURRENT riscv64-purecap
with libc revocation as the platform ships it: on, sweeps batched and
asynchronous (`runtime_revocation_default=1`, `runtime_revocation_async=1`,
`every_free_default=0`). The sysctl is read back inside the guest before any
case runs; any value but 1 aborts the run rather than scoring it.

| arm | reports | of |
|---|---:|---:|
| `sysalloc-bounds` (our baseline) | 16 | 23 |
| `sublet-gc` | 23 | 23 |
| **`cheribsd-revocation`** | **14** | 23 |

Controls: the interpreter evaluates (`puts 6*7` → 42) and 40- and 500-frame
recursion both survive. The recursion control matters here: under revocation the
old VM stack is dead after `stack_extend_alloc`, which is what app patch 0010
fixes, and without it this arm would die at ~40 frames before reaching any case.

## The 14 are a strict subset of our 16

Nothing this arm reports is missed by `sysalloc-bounds`. Two cases go the other
way, and for different reasons.

**`13_4a386f80e` -- structural, and no configuration reaches it.** `pack.c`
computes `num_lines = (slen + count - 1) / count`, which overflows for a huge
`count`, so `buffer_size` comes out too small, `str_len_ensure` reserves that
much, and the encoder writes past it. Nothing is freed, so there is nothing to
revoke or quarantine. It is a plain overflow of a live buffer, and CheriBSD's
allocator bounds each allocation to its **size class**, not to the request:

| request | capability length, CheriBSD | capability length, our heap |
|---:|---:|---:|
| 7 | 16 | 7 |
| 20 | 32 | 20 |
| 100 | 112 | 100 |
| 1000 | 1024 | 1000 |
| 4097 | 5120 | 4097 |

Both measured, same program, same sizes. A 1000-byte length is exactly
representable in a 128-bit capability, so this is allocator policy and not
compression: `malloc_usable_size` returns the bin size there and programs are
entitled to use it, so narrowing to the request would break them. `level0`
narrows to the request (`l0_narrow`, `level0.c:93-97`) and faults with cause 7.

**Caveat for silicon:** our column is QEMU. A stored capability is compressed on
the FPGA, and a region whose base or end is not a multiple of its length's
granule reloads wider -- the reason `sublet_heap.c` is a buddy allocator. This
catch is sound under QEMU and open on silicon, most of all for the larger sizes.
It is not yet evidence of a general property of the baseline.

**`17_bef45e223` -- configuration, not structure.** The fix's own comment says
Ruby can replace `keys` "moving the buffer as well as the length", so the old
element buffer is freed and the held pointer is stale. The block goes into
CheriBSD's quarantine, and because the sweep is batched the stale capability is
still valid when it is used: the case completes and prints a wrong answer. With
`_RUNTIME_REVOCATION_EVERY_FREE_ENABLE=1` and a synchronous sweep it becomes
SIGPROT, which is what proves the block was quarantined and merely unswept.
That configuration is not this arm -- it is far slower, and the arm is the
platform's default on purpose. Recorded here because it separates "CheriBSD
cannot" from "CheriBSD does not by default".

## What it does not catch

None of the six cases `sublet-gc` adds over our baseline. Three of them
(`04`, `05`, `08`) reuse a GC object slot inside an `mrb_heap_page` the system
allocator still holds, so no `free` ever reaches it; the other three
(`03`, `09`, `10`) are stale pointers into blocks whose lifetime has not ended,
so there is no free to revoke on.

Five of its 14 are outside this study's scope -- four NULL dereferences and one
type confusion. In scope it reports 9, against 16 for `sysalloc-bounds` and 22
for `sublet-gc`.

## Reproducing

    CHERI_SDK=... CHERI_SYSROOT=... bash capstone/ports/mruby/cheribsd/build.sh

then the guest runner with the image and rootfs from the same SDK. The binary's
sha256 is in `inputs.json`; mruby needs three things here that the domain build
does not, and `build.sh` says which and why.
