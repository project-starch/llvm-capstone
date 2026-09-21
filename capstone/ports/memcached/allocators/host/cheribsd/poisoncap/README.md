# memcached allocators on PoisonCap

Experimental protection for the extracted `slabs.c` and `cache.c`, on the
already reconstructed
[PoisonCap platform](../../../../../ffmpeg/buffer-pool/host/cheribsd/poisoncap/README.md).
Its pinned LLVM, QEMU and CheriBSD sources are reused without additional
platform patches. The server, its threads and the page mover are outside this
port, exactly as on the other targets; the allocators are single-threaded and
trusted.

## What is different from the stock CheriBSD build

The [stock build](../README.md) takes every slab page and cache object from
the platform's own `malloc`, so its libc revocation is asked the question at
the level where it lives -- and is never asked, because neither allocator
calls `free()` on the path where a freed unit is reused. This build replaces
that level: one `mmap`'d arena carries `CHERI_PERM_POISON` and
`CHERI_PERM_SW_VMEM`, that capability is the adapter's and stays in its
records, and every alias handed to memcached is bounded to its unit and
stripped of both permissions.

`src/cheribsd/poisoncap-authority.c` is the whole of it: ~160 lines under the
**same ledger the Capstone domain runs** (`src/shared/leases.c`, unchanged).
The ledger owns the page map, the chunk and object states and the counters;
only the authority beneath it differs, so the two systems measure one
bookkeeping and two mechanisms.

| mode | what a release does | what the corpus requires |
|---|---|---|
| 0 | nothing: bounded leases, no poison, no sweep | the sequence completes, and the adapter reports `sweeps=0` |
| 1 | `cpoison` on each 16-byte granule of the unit, one synchronous `cheri_revoke`, `cclearpoison`, then a `memset` | the stale access faults with `SIGPROT` `si_code=PROT_CHERI_TAG` at the corpus's labelled load |

The `memset` is not hygiene. `cclearpoison` resets the granule's access state
but leaves the poison capability stored in the payload, and a later sweep
would read that as a freshly issued lease. Both allocators write their
free-list links into the unit immediately after the release returns, so the
adapter hands back a fresh alias and upstream rebuilds the links through it.

Bounds are set with `cheri_setboundsexact` and never widened: a capability
longer than its unit would put a live neighbour's base inside the granules
this one poisons, and the sweep would take the neighbour with it. A size the
128-bit format cannot express exactly is refused (code 535) instead of being
rounded up, so a class beyond the representability boundary fails loudly
rather than silently loosening the property this arm measures. The classes the
corpus and the example use are all below it.

## Build and run

```sh
source capstone/tests/capstone-test-env.sh
export CHERI_SDK=/tmp/capstone/poisoncap-work/sdk
export CHERI_SYSROOT=/tmp/capstone/poisoncap-work/output/rootfs-riscv64-purecap
PC=capstone/ports/memcached/allocators/host/cheribsd/poisoncap
bash "$PC/build.sh" /tmp/capstone/poisoncap-memcached/build

python3 capstone/ports/common/host/cheribsd/run.py /tmp/capstone/memcached-poisoncap-1 \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" \
  --image /tmp/capstone/poisoncap-work/output/cheribsd-riscv64-purecap.img \
  --build memcached=/tmp/capstone/poisoncap-memcached/build \
  --disable-default-revocation
```

`allocator-example` defaults to **mode 1** in this build -- the protected arm
is what the platform is here for -- and prints its counters beside the
ordinary report, so the registered example is itself a protected run. An
explicit `0` or `1` argument overrides it.

Guest libc automatic revocation is off and the flag also clears the guest
default before SSH starts, which is the published platform's documented
workaround for its VM-locking failure. The adapter's own sweeps are
unaffected: they are explicit `cheri_revoke` calls, not libc's quarantine.
This is not whole-process temporal protection.

The [memcached corpus](../../../../../../bug-corpora/memcached/allocator-repros/runners/poisoncap/README.md)
runs its cases through this build, in both modes, behind the platform's own
instruction controls.
