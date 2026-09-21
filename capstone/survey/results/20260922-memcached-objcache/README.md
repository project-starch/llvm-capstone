# memcached's two allocators have nothing in common but the process

`memcached-objcache` recorded on 1.6.45, which is already the revision
`capstone/ports/memcached/allocators` pins, so this row needs no
re-recording. The slab level was recorded in the same run and is kept
beside it, because the contrast is the finding.

## The two levels, churn rung, repetition 1

| | slab classes | object cache |
|---|---:|---:|
| allocations | 100,000 | 23,511 |
| reuses | 80,000 | 23,499 |
| before backing release | **100.0 %** | **100.0 %** |
| distinct addresses | 20,000 | **12** |
| most objects at one address | 5 | **11,507** |

Both reuse before anything goes back to malloc, and there the likeness
ends. The slab level spreads its objects over twenty thousand addresses,
five to an address. The object cache concentrates twenty-three thousand
objects onto twelve, more than eleven thousand at one of them.

A survey with one row per program shows the first shape and misses the
second. That is the whole argument for a row per allocator, and it is the
allocator the slab wiring's own comment named as not measured.

## Repetitions

| rep | objcache allocations | reuses | distinct | slab allocations |
|---|---:|---:|---:|---:|
| 1 | 23,511 | 23,499 | 12 | 100,000 |
| 2 | 23,214 | 23,202 | 12 | 100,000 |
| 3 | 23,216 | 23,204 | 12 | 100,000 |

The object cache varies slightly because its population follows connection
and IO timing rather than the request count. The slab level is identical
across all three, which is what a per-request allocator should be.

## Where the instrument sits

`hook-objcache.py` in the manuscript's `experiments/a1/memcached/` places
it in `cache.c` the way `hook.py` places the slab one in `slabs.c`:
anchored on text, refusing a missing anchor, a doubled anchor or a second
run. Two sites. An object handed out by `do_cache_alloc`, whichever branch
produced it, and the death that never reaches malloc, the push back onto
the cache's STAILQ. The other branch of that same `if` calls `free` and is
a level-0 release, which the interposed malloc family already records.

## Provenance

| | |
|---|---|
| memcached | 1.6.45, pinned in `experiments/a1/memcached/sources.sha256` |
| libevent | 2.1.12-stable, built static by the same Makefile |
| workload | `client.py --rung churn`, server `-m 1024 -t 4 -o slab_automove=0` |
| wiring | `mchook-objcache.inc`, level name `objcache`, `A1_NO_BULK` |
| companion pass | `experiments/results/W2/20260921T214756Z-memcached`, three arms, three repetitions |
| compiler | gcc 13.3.0 |
| raw | `raw/`, hashed in `raw/SHA256SUMS` |

The shipped arm is the oracle and every row passed its checks.

## Limits

One rung. `cache_destroy` frees the list one object at a time and ends no
live object, so there is no bulk release to count and the level runs with
`A1_NO_BULK`, as the slab level does.
