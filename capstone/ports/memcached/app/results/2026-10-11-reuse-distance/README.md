# memcached 1.6.45: reuse distance at the nested allocators, stock against protected (2026-10-11)

This is the memcached row of the pre-registration
[`docs/plans/2026-10-11-nested-reuse-distance.md`](../../../../../docs/plans/2026-10-11-nested-reuse-distance.md)
(pushed as 16d2171a3edc). The pre-registration says it comes first and is not edited. The data is in
[`reuse.json`](reuse.json). Bucket k is the observer's `bins[k-1]`, so it holds distances d with
2^(k-1) <= d < 2^k. Empty buckets are omitted.

## Hooks

The hooks are one patch,
[`experiments/study/reuse-hooks/memcached/memcached-1.6.45-reuse-gap-hooks.patch`](../../../../../experiments/study/reuse-hooks/memcached/memcached-1.6.45-reuse-gap-hooks.patch),
plus the header [`mc-reuse-gap.h`](../../../../../experiments/study/reuse-hooks/memcached/mc-reuse-gap.h).
Everything sits behind `-DEXP_REUSE_GAP`.

Patch 0006 does nothing without `-DMC_CAPSTONE_SUBLET`, and both arms are built from one tree:
upstream, plus patches 0001-0006, plus the hook patch. So **one** hook patch serves both arms, and the
hooked lines are the same text in both. The build checks that `slabs.c`, `cache.c`, `slabs_mover.c`
and `memcached.h` are byte-identical between the arms. The arms differ only in that one define.

There is one observer per boundary, held in a static object in the file that owns the boundary. Each
has a static table of 2^15 slots (power of two), with no heap use and no capability retained. A
hand-out calls `reuse_gap_attempt` and then `reuse_gap_issue(start, size)`. A take-back calls
`reuse_gap_release(start)`. The start is the allocator's own pointer, read as an integer address.

| Boundary | Site (file:function, same in both arms) | Address recorded: stock | Address recorded: protected |
|---|---|---|---|
| slabs, hand-out | `slabs.c:do_slabs_alloc`, in `if (ret)`; size `p->size` (the chunk size) | `ret`, the chunk taken off the free list | `ret`, the CDERIVE child of the page generation, at the chunk's address |
| slabs, free | `slabs.c:do_slabs_free`, after `it = (item *)ptr` (also the header of a chunked item) | the caller's item | `slabs_take_back`'s generation-derived pointer at the same address |
| slabs, free of chained chunks | `slabs.c:do_slabs_free_chunked`, each chunk in the loop | the chunk | the chunk after `slabs_take_back` |
| slabs, not a release | `slabs.c:do_slabs_newslab` and `slabs_finalize_page_move` set `mc_rg_splitting` around `split_slab_page_into_freelist`. That function files never-issued chunks through `do_slabs_free`, so those calls are not releases. | | |
| cache, hand-out | `cache.c:do_cache_alloc`, before `return object` when `object != NULL`; size `cache->bufsize` | `ret`, the buffer (the same as the object under NDEBUG) | `ret`, the object's lifetime: the buffer start, with the object 8 bytes in. `bufsize` includes the reserved slot. |
| cache, free | `cache.c:do_cache_free`, after the NDEBUG block and before the limit check | `ptr`, the buffer | `ptr` after `ptr = life`, which is the buffer start |

Every slab hook runs under `slabs_lock`. Workers call `do_cache_*` on their own caches without a lock,
and one observer covers every `cache_t`, so the cache observer has its own mutex. Each boundary's first
event registers an **atexit** handler. On SIGTERM, memcached's `sig_handler` ends the event loop and
`main` returns, which runs the handler. The workload reaches that path. The handler writes one line
per boundary to the server's stderr:
`REUSE-GAP <boundary> arm=<stock|protected> attempts= issues= releases= reuses= distinct_starts= error= capacity= b1= ... b32=`.
`attempts` counts successful hand-outs only. That is the pre-registration's event index.

## Builds

[`build-reuse.sh`](../../../../../experiments/study/reuse-hooks/memcached/build-reuse.sh) does the
build without editing the port's scripts:
- it runs `deps/build-libevent.sh`;
- it gets the SDK from `deps/env.sh`;
- each arm uses the configure and make of `host/build-sublet.sh`, with
  `CPPFLAGS="-I runtime/include -I <hooks> -I experiments/study -DEXP_REUSE_GAP"`, and the protected arm
  adds `-DMC_CAPSTONE_SUBLET`.

The build has four gates:
- **INSN:** the protected image carries 14 CDERIVE and 4 CREVOKE, and the stock image carries none.
- **HOOK:** both images contain the report string.
- **LINK:** clean.
- **SOURCE:** the hooked files are identical in both arms.

| Arm | Image sha256 | Bytes |
|---|---|---|
| stock | 248b29be3658b74d25b96a716d5918a16f89420b2fdab0f5fc71c41316bc6ce1 | 4,137,144 |
| protected | a3b7459290215f71484bb90b195a4dea367897727318ef3b6cb4bb8a2df07c52 | 4,140,584 |

**Source:** `git archive 16d2171a3edc capstone`, the uncommitted files in
`experiments/study/reuse-hooks/memcached/`, and llvm-capstone 9bc6f9adf1c9 `compiler-rt/lib/builtins`.
These were committed into a scratch git repository on the build host. The toolchain is the kit's.

**Build time:** 329 s for libevent and both servers. libevent's native `make verify` had one failure,
`util/monotonic_prc_fallback`. It passed 6 of 6 reruns, which the port's rule accepts.
`capstone-test-env.sh` warned that it could not check the kit toolchain's freshness (there is no
`build.ninja`).

## Platform and workload

The platform is that of the pre-registration:

| Component | sha256 |
|---|---|
| QEMU | af37cc3269d1… |
| launcher `capstone-vexec` | 3975314a7a6d… |
| module | a1c6cb6b79a9… |
| `capstone-job` | fb6192b86410… |

The images are `Image` e833dfb1…, `fw_jump.elf` 44cd6cbc… and `rootfs.ext2` 62049800…. The guest
has one hart, and memcached runs four workers.

The workload is the protocol oracle unchanged: `host/mc-harness/mc-harness.c`, source sha256
1e569a8b…. The guest uses the kit's static RISC-V build of that source (sha256 63cf0442…), because
the build host has no RISC-V gcc. The harness starts
`capstone-job … --user 65534:65534 -- capstone-vexec memcached.dom -l 127.0.0.1 -p 21299 -U 0 -m 64 -t 4`
as `host/run-virtual.py` does. It then runs its four phases:
1. one connection through every command family;
2. values from 1 B to 900 KB, plus one value that is too large;
3. eight concurrent connections, each pipelining 40 set/get pairs with an incr on every fifth key;
4. stats.

Finally the harness stops the server with TERM.

[`run-reuse.py`](../../../../../experiments/study/reuse-hooks/memcached/run-reuse.py) boots one arm
and runs **three servers in turn**. There was one boot per arm. Both boots ran in parallel, each with
its own QEMU lock and a fresh work directory:

```sh
CAPSTONE_QEMU_LOCK=<own lock> python3 capstone/experiments/study/reuse-hooks/memcached/run-reuse.py \
  --arm stock|protected --image <build>/<arm>/memcached.dom --work <fresh> \
  --qemu $K/kit-sublet/qemu-system-riscv64 --images $K/kit-sublet/images --adapter $K/kit-sublet/adapter \
  --job $K/adapter-sublet/capstone-job --harness <mc-harness-riscv-static> \
  --run-staged capstone/runtime/virtual/run-staged.py --reps 3
python3 capstone/experiments/study/reuse-hooks/memcached/assemble-reuse.py stock/result.json protected/result.json build.log reuse.json
```

**Boot wall time:** 53.5 s (stock) and 54.1 s (protected).

**Sanity checks:**
- All six servers exited 0 on TERM, with `capstone-job` recording `exit 0`, and printed nothing else
  on stderr.
- `error=0` at both boundaries in every run.
- The normalised protocol transcript is the same in all six runs (sha256 ee1522c8e0ec…).

## Results

`reuse.json` has repetition 1 at the top level of each arm, and all three under `repetitions`.

| Boundary | Arm | Issues | Releases | Reuses | Distinct starts | Buckets (rep 1) | Median bucket |
|---|---|---|---|---|---|---|---|
| slabs | stock | 352 | 20 | 20 | 332 | 1:13 2:2 3:1 4:3 5:1 | 1 |
| slabs | protected | 352 | 20 | 20 | 332 | 1:13 2:2 3:1 4:3 5:1 | 1 |
| cache | stock | 131 | 127 | 111 | 20 | 1:79 2:18 3:12 4:1 7:1 | 1 |
| cache | protected | 131 | 127 | 111 | 20 | 1:81 2:15 3:13 4:1 7:1 | 1 |

**slabs:** all three repetitions of both arms are identical. Issues, releases, reuses, distinct
starts and every bucket match. The share of reuses whose bucket differs between the arms is **0**.

**cache:** issues, releases, reuses and distinct starts are identical in all six runs. The buckets
vary between repetitions *within* an arm. Phase 3's eight connections are served by four workers, and
the order in which the workers interleave their hand-outs on the one cache index is not fixed. The
counts were:

| Arm | Rep | Buckets |
|---|---|---|
| stock | 1 | 1:79 2:18 3:12 4:1 7:1 |
| stock | 2 | 1:81 2:16 3:12 4:1 7:1 |
| stock | 3 | 1:82 2:15 3:12 4:1 7:1 |
| protected | 1 | 1:81 2:15 3:13 4:1 7:1 |
| protected | 2 | 1:77 2:21 3:11 4:1 7:1 |
| protected | 3 | 1:79 2:20 3:10 4:1 7:1 |

The share of reuses whose bucket differs between the arms is 0.027 (3 of 111) for repetition 1. Over
all nine stock × protected pairs it is between 0.009 and 0.054. Within one arm it is between 0.009
and 0.027 (stock) and between 0.018 and 0.054 (protected). The median bucket is 1 in every run.

## Verdicts

1. **Prediction 1, equal histograms for memcached:**
   - **slabs: holds**, bucket for bucket, in every repetition.
   - **cache: holds under the variation clause.** The arms vary alike: the per-bucket ranges over
     three repetitions overlap in every bucket, and all counts other than the buckets are equal.
   - A single run per arm would *not* have shown equality bucket for bucket at this boundary. The
     difference between the arms is the same size as the difference between repetitions of one arm.
   - The pre-registration listed only Perl and CPython as varying. memcached's cache.c boundary
     varies too, because of the oracle's concurrent phase. That is a finding of this row.
2. **Prediction 2** concerns SQLite and does not apply.
3. **Prediction 3, the figure:** the paper script draws it from `reuse.json`, and this row draws
   nothing. From the data, the slabs curves are identical and the cache curves differ by at most the
   run-to-run spread.

## Deviations and limits

- **Three servers per arm:** the pre-registration says one run suffices for a deterministic program.
  A short native trial of the hooked server, run before the QEMU runs, showed the cache buckets
  varying. So each boot ran three servers. The top-level numbers are repetition 1.
- **Free-list splits are not releases:** `do_slabs_free` also files fresh, never-issued chunks
  (`split_slab_page_into_freelist`). Those calls are excluded by a flag set around the two call sites.
- **Slab mover:** chunks the mover rescues, or reclaims with a page move, are never passed to a free.
  The observer would flag an overlapping issue (error 2) if such an address were handed out again.
  No run reported an error.
- **Cache size field:** the cache hook records the buffer that the cache keeps. Its recorded size is
  8 bytes larger in the protected arm, because of the lifetime slot. The size does not enter any
  distance.
- **No safety fixtures:** these images were not run through the port's safety fixtures. The
  instrumented builds are measurement images only.
