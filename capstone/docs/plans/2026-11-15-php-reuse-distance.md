# Reuse distance at PHP's Zend allocator, stock against protected: pre-registration, 2026-11-15

Committed before any instrumented build or run. A reading that differs from what is written here
is a finding; this file is not edited afterwards.

## The question

`2026-10-11-nested-reuse-distance.md` measured reuse distance at the nested allocators of eight
programs, stock against protected. This adds PHP, whose bug corpora already run on the trap route
with its lifetime patch: does that patch change when PHP's Zend allocator reuses an address?

## Definition (the report's, unchanged)

Reuse distance counts allocations between the release of an address and its next allocation, at
one allocator boundary. Distances go into power-of-two buckets by bit length: bucket k holds
distances d with 2^(k-1) <= d < 2^k. The observer is `experiments/study/reuse-gap-observer.h`
(integer-only, no capability retained), driven through `reuse-hooks/php/reuse-gap-hook.h`, the
header the Perl and FFmpeg hooks use. Its `bins[b]` holds bit length b + 1, so bucket k =
bins[k-1]. The event index is the allocator's own hand-outs (`reuse_gap_attempt` once per
hand-out), and a reuse is a hand-out at a start address that was released before. The run is
rejected if the observer's `error` is nonzero.

## The program and its allocator boundary

PHP 5.0.0, the source snapshot of the corpora (`ports/php/app/prepare-sources.sh`: the release
tarball by SHA-256 with the two tree patches), built by the common entry
(`ports/common/application/build-virtual.sh php`, real PHP, 145 translation units).

**PHP 5.0.0 has no Zend MM.** The Zend memory manager (`zend_mm_alloc_small`,
`zend_mm_free_small`, its bins and runs) arrived in PHP 5.2; this release does not contain it
(`ZEND_MM` is not defined). Text about this measurement must not name the Zend MM. The nested
allocator measured here is the small-block cache of PHP 5.0.0's `Zend/zend_alloc.c`
(`AG(cache)`). `_emalloc` rounds a request to `REAL_SIZE(n)` (a multiple of 8) and computes
`cache_index = REAL_SIZE(n) >> 3`. Below `MAX_CACHED_MEMORY` (11, so up to 80 bytes) it hands out
the block last parked in that bin of `AG(cache)` (256 entries per bin, last in first out), and
takes a new block from malloc only when the bin is empty. `_efree` parks such a block in its bin,
and frees it to malloc only when the bin is full. Larger blocks go to and from malloc directly.
These are the hand-out and release points the lifetime patch instruments (its
`ZEND_CAP_PAYLOAD` at both `_emalloc` returns, `ZEND_CAP_RECLAIM` in `_efree`).

Two boundaries, by the size class Zend computes:

| Boundary | Blocks | Hand-out | Release |
|---|---|---|---|
| `zend-small` (the small bins) | `cache_index` < 11 | `_emalloc` from `AG(cache)`, else from malloc | `_efree`: parked, or freed to malloc past a full bin |
| `zend-large` (second boundary) | all others | `_emalloc` from malloc | `_efree`, freed to malloc |

The address recorded is the block header (`zend_mem_header *`), the pointer Zend itself works
with, in both arms; never the caller's payload pointer. A successful `_erealloc` that keeps the
header's address and boundary is a resize (no hand-out); any other outcome releases the old
header at its boundary and hands out the new one at its boundary. Request shutdown frees every
block PHP never freed: each is a release there, so a later hand-out at that address is a reuse.
Each boundary also reports, beside the observer, `drops` (blocks freed to malloc), `fresh`
(hand-outs of a block newly taken from malloc), `cache_hits`, `moves`, `resizes` and
`shutdown_releases`.

## Workload

Zend/bench.php, PHP's own benchmark. The PHP 5.0.0 snapshot does not ship it: the file first
appears in PHP 5.1.0. The workload is therefore bench.php from the PHP 5.1.0 release tarball
(`php-5.1.0.tar.gz`, SHA-256 3eab94beeac8d1d4f64a79190b78989324025d068d90501e0dbea0698a5d2f41;
`Zend/bench.php`, SHA-256 cf2587610279004af239024a2d57af0aaf1f4d66b573b3c59f7b5d8fe7a71053),
which uses only functions PHP 5.0.0 has. It runs as `php -n bench.php` from a staged copy.

The full file is too long under QEMU: on the uninstrumented protected image it had not finished
after 140 minutes. The pinned setting cuts each function's work to about a tenth
(`reuse-hooks/php/make-bench.py`, which checks both input hashes and the result's,
SHA-256 6b9fc8f9ba15cb7a0972408b8e69724787085e5a50fc59898b49d0c52b934606):

| Function | bench.php | Pinned |
|---|---|---|
| simple, simplecall, simpleucall, simpleudcall | 1,000,000 iterations | 100,000 |
| mandel | 1,000 iterations per point | 100 |
| mandel2 | 5,000 iterations per point | 500 |
| ackermann | Ack(3, 7) | Ack(3, 5) |
| ary, ary2 | 50,000 | 5,000 |
| ary3 | 2,000 elements, 1,000 passes | 2,000 elements, 100 passes |
| fibo | 30 | 25 |
| hash | 50,000 | 5,000 |
| hash2 | 500 (n^2 work) | 158 |
| heapsort | 20,000 | 2,000 |
| matrix | 20 products of 30 x 30 | 2 |
| nestedloop | 12 (n^6 work) | 8 |
| sieve | 30 passes | 3 |
| strcat | 200,000 | 20,000 |

On the same image this setting ran in 67 minutes (bench.php's own total: 4,014 s guest time),
with 3,442,407 emalloc payloads issued by the nested layer.

## Arms

From one source snapshot and one SDK per arm, built by `reuse-hooks/php/build.sh`:

- **stock**: PHP's own `zend_alloc.c` over virtual mallocng (`CAPSTONE_VIRTUAL_LIFETIMES=0`).
- **protected**: the lifetime patch the corpora's Sublet arm uses,
  `ports/php/zend-alloc/sublet/patches/5.0.0/0001-zend-alloc-sublet-hooks.patch`, run as the
  object arm (`PHP_CAP_MODE=6`, the configuration `php-real-object`, virtual-nested-pools): every
  emalloc payload a child of its block, ended at every efree.
- **exact**, third arm at no build cost: the same protected image with `PHP_CAP_MODE=7`
  (`php-real-exact`), each child bounded to the request instead of `REAL_SIZE(n)`.

Both images carry the same hook patch (`reuse-hooks/php/zend-alloc-reuse-gap.patch`) behind one
define, `EXP_REUSE_GAP`. The hooks sit away from the lifetime patch's seven substitutions, whose
patch applies to the hooked file at fuzz 0; `build.sh` checks that and fails otherwise.

## Platform

The trap route of run-3: kernel Image-0133b9e21cd, qemu-1b, firmware f-keep, the libc and
lifetime patches of native-kernel-new-qemu 7ba3f427f9 (trap/eval), kit `kit-run-3`. Results go to
`~/asr-work/eval/run-5/reuse/php` as `trap_platform.py run` bundles (`results.json`, the format
of the Perl, FFmpeg and PostgreSQL rows of run-3), one boot per run.

## Repetitions

Two runs per arm (stock-1, stock-2, protected-1, protected-2, exact-1, exact-2), each its own boot,
at most eight QEMU instances at once. bench.php prints timings, so string lengths and the
allocations behind them can differ between runs; the second run of each arm measures that noise.

## Does the patch change sizes the way memsys5 does?

Checked in the source before writing the predictions. In SQLite the patch changed what memsys5
reports to SQLite, and SQLite sized its own requests from that, so the request sequence at the
boundary changed. Here it does not: the lifetime patch adds no statement to PHP's logic and
leaves `REAL_SIZE`, `cache_index`, `MAX_CACHED_MEMORY`, `MAX_CACHED_ENTRIES` and the bin order
alone, so PHP makes the same `_emalloc`, `_efree` and `_erealloc` calls in both arms, and
`AG(cache)` hits and misses the same way.

The patch does change what Zend asks malloc for, below the boundary:

1. **Block size.** Stock asks for the header (48 bytes) plus `REAL_SIZE(n)`. The protected layer
   (`zend_cap_block`) rounds to a 16-byte grain, takes at least 64 bytes, and adds an 8-byte
   prefix for the parent reference, plus alignment slack for payloads of 4 KiB or more:
   emalloc(8) is a 56-byte malloc in stock and a 72-byte one protected, emalloc(80) 128 against
   136. mallocng can then place fresh blocks from other size classes.
2. **Realloc.** Stock `_erealloc` calls realloc, which may keep the block in place. The protected
   layer always takes a fresh block, copies and frees the old one.

So a hand-out from `AG(cache)` returns the same block in both arms, while a fresh block or a
reallocated one can land elsewhere. The histograms are therefore not predicted equal.

## Predictions

1. **Same request stream.** At each boundary, `cache_hits` is equal in both arms, and so is
   `fresh - moves` (the emalloc misses) and `shutdown_releases`. The protected arm has
   `resizes = 0`, and its `moves` equal the stock arm's `moves + resizes`. These hold run for run
   between runs that print timings of the same widths; between other runs, the arms' ranges
   overlap.
2. **Histograms differ, centre stays.** Stock and protected histograms are not identical at
   either boundary. The median bucket of the reuses is equal at each boundary. The share of
   reuses whose bucket differs between the arms (`reusediff`) is reported per boundary.
3. **Exact equals protected.** The exact arm (`PHP_CAP_MODE=7`) equals the protected arm in every
   count and every bucket at both boundaries, run for run between runs that print timings of the
   same widths: both use the same block geometry, and the child's bound reaches no allocation.
4. **Clean runs.** The observer's `error` is 0 in every run, and bench.php's output is the same
   in every run apart from its timing column.
5. **Repetitions.** Where the two runs of an arm differ, the arms differ alike (overlapping
   ranges per bucket). The stock against protected `reusediff` exceeds the noise floor (stock-1
   against stock-2) at the small boundary, since prediction 2 expects real differences there.

For each arm and boundary the report gives issues, releases, reuses, distinct start addresses,
error, and the `reusediff` share against stock: the sum over buckets k of
|a_k/N_a - b_k/N_b| / 2, as in the October records. The median bucket is the first bucket at
which the cumulative count reaches half the reuses. Per boundary the report gives `reusediff` for
stock-1 against stock-2 (the noise floor), protected-1 against protected-2, and stock against
protected (repetition 1 against 1, and the range over all four pairs); exact likewise.
`reuse-hooks/php/collect.py` reads the bundles and computes these.
