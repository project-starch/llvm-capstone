# memcached/allocator-repros on the virtual profile, 2026-11-11

Both virtual arms, re-measured after patch 0006 changed its design and the processor model changed
its lifetime instructions:

* `CREVOKE` takes the object alone. `slabs_free` ends the caller's chunk with `CREVOKE(chunk)`.
* `CRENEW` is new. `cache_free` renews the lifetime of the allocation `malloc` returned, and
  `cache.c` derives nothing. The slot in front of each cache object is gone.

| arm | configuration | cases | tally | against `../2026-10-11-virtual` |
|---|---|---|---|---|
| `virtual-nested-pools` | `virtual-memcached-sublet` (`-DMCP_SUBLET=ON`) | 9 | CAUGHT 8, NO-READING 1 | every verdict equal |
| `virtual-malloc` | `virtual-memcached-stock` (`-DMCP_STOCK_MALLOC=ON`) | 9 | CAUGHT 1, MISSED 8 | every verdict equal |

Controls, each in the arm's own boot: `bounds-malloc` and `uaf-malloc` fault on both arms (causes 28
and 25). `chunk-free-90` faults on `virtual-nested-pools` (cause 25 at `read_probe`) and completes on
`virtual-malloc`, as `tools/arms.json` declares.

The fault causes of `virtual-nested-pools` equal the earlier bundle's case by case: 25 at
`read_probe` for 00 to 04, 28 at `write_probe` for 06, 28 at `read_probe` for 07, 28 in
`mc_case_body` for 08.

Case 05 stays NO-READING (`control-failed`) for the reason `corpus.json` records: the case's setup
writes its sentinel into byte 0 of the neighbouring live item, which under patch 0006 holds that
item's generation, so the item's later `slabs_free` faults in the fixed run (cause 24). One detail
changed with the instruction. The one-operand `CREVOKE(chunk)` in `slabs_take_back` succeeds, and the
fault is the next statement's `lcc` on the untagged generation (`slabs.c:74`). The two-operand form
faulted at the revoke itself. The buggy run of the same image faults at `write_probe` with cause 28.

## Platform

The kit is the tree's virtual platform on the model with the one-operand `CREVOKE` and `CRENEW`;
`inputs.json` of each bundle carries every file hash. Two settings are not in `inputs.json`:

* CPU properties: `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`,
  so the `-cpu` line reads
  `rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,x-capstone-test-csrs=true,x-capstone-exact-bounds=true`.
  Exact bounds is the profile the 2026-10-11 bundles ran (then `run-staged.py`'s default).
* Module: loaded without parameters (`unpinned=0`, `supervisor_slots=1`).

Under the model's default compact bounds the nested arm gives no reading: the first
`split_slab_page_into_freelist` faults with cause 29 at the `CDERIVE` of the page's generation,
because a generation of `size * perslab` bytes is not representable for most classes. The derive has
the same arguments as before the conversion.

## Build

    CAPSTONE_SDK=<corpus SDK> shared/build-cases.sh capstone-application <cases> -DMCP_SUBLET=ON -DMCP_STOCK_MALLOC=OFF
    CAPSTONE_SDK=<corpus SDK> controls/virtual/shared/build-cases.sh capstone-application <controls> (same options)
    tools/run-virtual-cases.py --corpus . --arm virtual-nested-pools --prebuilt <cases> --prebuilt-controls <controls> ...

and the same with `-DMCP_SUBLET=OFF -DMCP_STOCK_MALLOC=ON` for `virtual-malloc`. The SDK is a corpus
SDK (`-fno-builtin`). Lifetime instructions in the nested build's allocator objects
(`ports/common/application/lifetime-instructions.py`): `slabs.c.o` CDERIVE 2, CREVOKE 1;
`cache.c.o` CRENEW 1. The stock build's objects carry none.
