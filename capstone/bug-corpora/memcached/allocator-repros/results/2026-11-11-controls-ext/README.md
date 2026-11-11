# memcached cache.c object cache: parent, sibling and successor controls, 2026-11-11

The controls of [`../2026-10-12-controls-ext`](../2026-10-12-controls-ext/README.md), run again
after patch 0006 changed `cache.c`. The pre-registration is unchanged:
[`docs/plans/2026-10-12-nested-lifetime-controls-extended.md`](../../../../../docs/plans/2026-10-12-nested-lifetime-controls-extended.md).

The protected arm is `virtual-nested-pools` (`-DMCP_SUBLET=ON`, `virtual-memcached-sublet`), the
release arm is `virtual-malloc` (`-DMCP_STOCK_MALLOC=ON`, `virtual-memcached-stock`).

**Every prediction held: 8 runs, one boot per arm.** The stale access completed on release and read
C's data. On protected it faulted at the labelled probe with cause 25. The 2026-10-12 record ran
three boots per arm; this one ran one, because the boots queued behind a shared lock.

## What changed under the controls

On 2026-10-12 `cache_create` reserved one capability in front of each object, `cache_alloc` handed
out a `CDERIVE` child of the object's lifetime, and `cache_free` revoked that child. Now an object
is the allocation `malloc` returned, with upstream's layout. `cache_alloc` hands out the pointer on
the free list. `cache_free` renews the object's lifetime through the caller's pointer (`CRENEW`)
and files the renewed pointer on the list. So the list head names the object's new lifetime, and
every copy of the caller's pointer names the one that ended.

## Results

| control | arm | outcome | A | B / C | what the program printed |
|---|---|---|---|---|---|
| 1 parent, `list-head` | protected | completed | 0x3fbf1f3440 | reissued at A | after `cache_free(A)` the list head has A's address and A's bounds `[A, A + 256)`, tag 1; the link reads 0; the head reads the owner's 0xa1, writes 0x7e, and the next `cache_alloc` returns A's storage holding 0x7e |
| 2 sibling | protected | completed; B intact | 0x3f9c9ff440 | B 0x3f9c9ff580 | A `[A, A + 256)`, B `[B, B + 256)`; B reads 0xb2 after A's free and 0xb5 after a rewrite |
| 3 successor, `through-c` | protected | completed; C = A after 1 allocation | 0x3fa3d53440 | C 0x3fa3d53440 | C `[A, A + 256)`; C reads back 0xc3 |
| 3 **stale** | protected | **fault cause=25 pc=0x3fbaf8de44**, `read_probe`, `shared/corpus.h:81:10` (link pc 0x1de44); faulting address 0x3fbb073450 = A + 16 | 0x3fbb073440 | C 0x3fbb073440 | A and C `[A, A + 256)` |
| 1 parent, `list-head` | release | completed | 0x3f8d3b4070 | reissued at A | the list head has A's address and bounds `[A, A + 256)`; same reads and writes as protected |
| 2 sibling | release | completed; B intact | 0x3f9949f070 | B 0x3f9949f1b0 | A `[A, A + 256)`, B `[B, B + 256)` |
| 3 successor, `through-c` | release | completed; C = A after 1 allocation | 0x3faec9f070 | C 0x3faec9f070 | C `[A, A + 256)` |
| 3 **stale** | release | **completed**; read 0xc3, C's data | 0x3f82b9f070 | C 0x3f82b9f070 | A and C `[A, A + 256)` |

On both arms every object is taken from the cache's free list (`freecurr` drops by one, `total` does not
change), and its capability is `[object, object + 256)` with the cursor at the object: the bounds
of the allocation, which the renewal keeps. The 256 bytes are `cache_create`'s `bufsize`. The slot
of the 2026-10-12 layout is gone.

Each boot ran the virtual heap's own controls first: `uaf-malloc` faulted with cause 25 and
`bounds-malloc` with cause 28, on both arms.

## Prediction verdicts

| prediction (pre-registration) | protected | release |
|---|---|---|
| control 1: completes; the read returns what the allocator left there | holds | holds |
| control 2: completes; B intact | holds | holds |
| control 3: C's start equals A's; access through C completes | holds | holds |
| stale read through A: protected rejects with cause 25 at the probe; release completes and reads C's data | holds (cause 25 at `read_probe`) | holds (read 0xc3) |

## Provenance

`results.json` carries the runner's record of each boot, every image hash, the
hash of patch 0006 and of each control file, and the platform's file hashes.

* **Platform and options.** As [`../2026-11-11-controls`](../2026-11-11-controls/README.md): QEMU
  `2d5a7107...`, launcher `0f8e0f90...`, module `e47bca62...`;
  `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`, `-smp 1`,
  module loaded without parameters.
* **Lifetime instructions** in the protected images outside libc's heap: `CRENEW` in
  `capstone_cap_renew`, `CDERIVE` in `capstone_cap_derive` and `CREVOKE` in `capstone_cap_release`
  (the port library is a Debug build, so the helpers stay functions; the last two are
  `slabs.c`'s, which the images also link). The release images carry none.
