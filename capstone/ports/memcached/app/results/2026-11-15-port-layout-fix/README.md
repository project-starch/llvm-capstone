# memcached 1.6.45: patch 0006 with upstream's item layout, run-6 (2026-11-15)

The pre-registration is
[`docs/plans/2026-11-15-memcached-port-layout-fix.md`](../../../../../docs/plans/2026-11-15-memcached-port-layout-fix.md),
committed as 187c25e9eb9 before any run. The change is 07f60dc6f7f on `fix/memcached-port-layout`. The
measured workload is run-5's
([`2026-11-15-established-reuse`](../2026-11-15-established-reuse/README.md)). The run records are in
`~/asr-work/eval/run-6/reuse/memcached-<arm>-<rep>/` on the build host, each with its `provenance.txt`.
[`reuse.json`](reuse.json) holds the pre-registered two repetitions per arm, assembled by
`experiments/study/reuse-hooks/memcached/assemble-reuse.py`.
[`reuse-with-diagnostic-rep3.json`](reuse-with-diagnostic-rep3.json) adds a third repetition per arm,
run after the first two as a diagnostic (Verdict 5).

## What changed

Patch 0006 no longer adds `sublet_parent` to `item` and `item_chunk`. Each chunk is a CDERIVE child of
its page, the reference the class keeps in `slab_list`. `slabs_free` ends the caller's chunk and finds
the page by address range over `slab_list`. A page move renews the page (CRENEW), which ends every
chunk the page handed out. With `-L` each page is a child of the preallocated block.
`do_slabs_alloc` and `do_cache_alloc` clear the free-list link of the chunk or object they hand out.
The allocators port's shim (`adapted/mc_slabs_shim.h`) drops the field as well. The reuse-gap hooks
patch follows the new context with the same observation points: after `unifdef -UMC_CAPSTONE_SUBLET`
the stock sources of `slabs.c`, `cache.c`, `slabs_mover.c` and `memcached.h` equal run-5's.

## Unit test (`tests/sublet-layout/`)

One trap-route boot, both builds through the allocators port's Sublet build. All nine steps read as
predicted ([`sublet-layout-check.txt`](sublet-layout-check.txt)).

| step | old patch (5060011844e) | new patch |
|---|---|---|
| `scan` | 3 LEAK: slab chunk offset 0 (the generation, bounds 1 MiB), offset 16 (the free-list link, page bounds), cache object offset 0 (the next free object), exit 1 | no LEAK, exit 0 |
| `stale` | cause 25 at the read after `slabs_free` | cause 25 |
| `bounds` | cause 28 one byte past the chunk | cause 28 |
| `move` | the other page's chunk, an earlier page's chunk and a chunk of the class that took the moved page read back, then the moved page's chunk faults with cause 25 | the same |
| `move-prealloc` | not run | the same, with every page cut from one block |

The old build prints `sizeof(item)=96`, class 1 chunks of 144 bytes. The new build prints 80 and
128, as stock.

## Static layout

`build-reuse.sh`'s LAYOUT gate compiles `layout-probe.c` in both arm trees with make's flags. Both arms:
`sizeof(item)=80 offsetof(item,data)=72 sizeof(item_chunk)=80 offsetof(item_chunk,data)=66`.

## Runs

The new stock image (6ddd173e904143a8) differs from run-5's (f272d96abda0...): the application runtime
of 5060011844e differs from run-5's tree (libc fe8f030e... is the same). So stock ran again.

| run | boot s | load s | transcript | job | slabs issues / releases / reuses / distinct | cache issues / releases / reuses / distinct | error | evictions (3, 8, 13, 19) |
|---|---|---|---|---|---|---|---|---|
| stock-1 | 1,045.0 | 996.0 | ee1522c8e0ec | exit 0 | 60,352 / 47,984 / 47,984 / 12,368 | 61,770 / 61,766 / 61,750 / 20 | 0, 0 | 31,047 (3,692, 7,734, 10,060, 9,561) |
| stock-2 | 1,024.2 | 974.9 | ee1522c8e0ec | exit 0 | 60,352 / 47,985 / 47,984 / 12,368 | 61,766 / 61,762 / 61,746 / 20 | 0, 0 | 31,013 (3,672, 7,698, 10,056, 9,587) |
| protected-1 | 1,078.6 | 1,029.0 | ee1522c8e0ec | exit 0 | 60,352 / 47,984 / 47,984 / 12,368 | 61,794 / 61,790 / 61,774 / 20 | 0, 0 | 30,987 (3,679, 7,722, 10,075, 9,511) |
| protected-2 | 1,055.0 | 1,006.0 | ee1522c8e0ec | exit 0 | 60,352 / 47,984 / 47,984 / 12,368 | 61,773 / 61,769 / 61,753 / 20 | 0, 0 | 30,987 (3,632, 7,730, 10,056, 9,569) |
| stock-3 (diagnostic) | 1,040.2 | 989.2 | ee1522c8e0ec | exit 0 | 60,352 / 47,987 / 47,984 / 12,368 | 61,772 / 61,768 / 61,752 / 20 | 0, 0 | 31,096 (3,536, 7,772, 10,104, 9,684) |
| protected-3 (diagnostic) | 1,047.3 | 998.4 | ee1522c8e0ec | exit 0 | 60,352 / 47,986 / 47,984 / 12,368 | 61,775 / 61,771 / 61,755 / 20 | 0, 0 | 30,938 (3,552, 7,567, 10,090, 9,729) |

Every load completed with processed = 240,000. No run printed anything on stderr but the observer lines.

## Dynamic layout

In all six runs every class reads as run-5 stock in chunk size, chunks per page and pages:

| class | 1 | 3 | 8 | 10 | 13 | 19 | 20 | 29 | 31 | 35 | 36 | 37 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chunk bytes | 128 | 208 | 704 | 1,104 | 2,192 | 8,448 | 10,560 | 78,848 | 123,200 | 300,800 | 376,000 | 524,288 |
| chunks per page | 8,192 | 5,041 | 1,489 | 949 | 478 | 124 | 99 | 13 | 8 | 3 | 2 | 2 |
| pages | 1 | 1 | 2 | 1 | 4 | 17 | 1 | 1 | 1 | 1 | 1 | 1 |

`total_malloced` reads 33,554,432 in every run. Run-5's protected arm read 144 bytes and 7,281 chunks
for class 1, and 240 bytes and 4,369 chunks for class 3.

## Share of reuses in another bucket (reusediff)

Pre-registered repetitions 1 and 2:

| boundary | within stock | within protected | stock x protected (4 pairs) | run-5 stock x protected |
|---|---|---|---|---|
| slabs | 0.0039 | 0.0033 | 0.0012, 0.0023, 0.0028, 0.0054 | 0.0056 to 0.0084 |
| cache | 0.0088 | 0.1323 | 0.0014, 0.0102, 0.1222, 0.1309 | 0.0010 to 0.0097 |

With the diagnostic repetition 3 (pairs with protected-1 apart):

| boundary | within stock (3) | within protected (3) | stock x protected, without protected-1 (6) | stock x protected-1 (3) |
|---|---|---|---|---|
| slabs | 0.0016 to 0.0039 | 0.0008 to 0.0036 | 0.0010 to 0.0029 | 0.0023 to 0.0054 |
| cache | 0.0088 to 0.0177 | 0.0033, and 0.13 with protected-1 | 0.0014 to 0.0110 | 0.122 to 0.140 |

Slab bucket 1 holds 91.36 and 91.67 % of the reuses in stock and 91.13 and 91.46 % in protected
(run-5: stock 91.33 to 91.52 %, old protected 92.08 to 92.16 %).

## Verdicts

1. **Output: holds.**
2. **Static layout: holds.**
3. **Dynamic layout: holds.** Every class, `total_malloced`, issues, distinct starts (12,368) and reuses
   (47,984) equal stock. Evictions lie inside the stock range widened by its width (stock runs of run-5
   and run-6: 31,013 to 31,216 in total, protected 30,987 and 30,987), and so does every class.
4. **Slab distribution: fails for one pair of four.** The floor is the largest within-arm value of
   run-5 and run-6, 0.0039 (run-6 stock). Three pairs lie below it. stock-2 x protected-1 reads 0.0054.
   The run-5 gap is gone: run-5's pairs read 0.0056 to 0.0084 against a floor of 0.0030, with bucket 1
   apart by 0.6 to 0.8 points. Now bucket 1 of the two arms overlaps, and the deterministic counts are
   equal.
5. **Cache distribution: holds only formally.** protected-1 reads 0.13 against every other run (bucket 1
   65.6 % instead of 77.8 to 79.5 %, bucket 3 9.2 % instead of 2.7 to 3.6 %). protected-2 reads 0.0014
   and 0.0102 against the two stock runs, inside run-5's spread. The largest within-arm value is the
   protected-1 x protected-2 pair itself, so the prediction passes on the outlier's own value. That is
   not evidence.
6. **Unit test: holds.**
7. **Corpus: holds.** See below.

**Diagnostic repetition 3.** After Verdict 5 one more server per arm ran with the same arguments
(`run-reuse.py` as `run-established.sh` calls it, both boots in parallel). protected-3 reads like
stock at the cache boundary (0.0045, 0.0110 and 0.0068 against stock-1 to stock-3, 0.0033 against
protected-2). protected-1 is one outlier run, at both boundaries. Its slab pairs include the two
largest cross values (0.0054 and 0.0038). Without it every slab cross pair (0.0010 to 0.0029) lies
below the largest within-arm value of each arm (0.0039 stock, 0.0036 protected). The median slab cross pair is 0.0023, the median within-arm pair of
run-6 0.0029. protected-1 is the boot whose load started slowest (2,901 commands after 10 s, against
3,525 in protected-2), while two corpus boots and other streams' boots shared the host. Its cause
is not established. The repetition does not change the pre-registered verdicts.

Evictions read 30,938 to 30,987 in the three protected runs and 31,013 to 31,216 in the six stock
runs of run-5 and run-6: about 0.1 to 0.9 % fewer, with every protected run below every stock run.
The protected server also ran 1 to 3 % slower (load 998 to 1,029 s against 975 to 996 s in run-6).
memcached's LRU maintainer thread wakes on time, so a slower server gives it more turns per command.
That may explain the shift. These runs do not test it. In protected-3 class 8 evicted 7,567 items, 21
below the widened stock range (7,588 to 7,864). Run-5's old layout evicted 8 % more.

## Corpus

`memcached/allocator-repros`, arm `virtual-nested-pools`, is the only corpus built with patch 0006
(`shared/build-cases.sh capstone-application` with `MCP_SUBLET=ON`). No trap-route run of it existed.
Both builds ran on the run-6 kit (`tools/run-virtual-cases.py --prebuilt`, corpus SDK sdk.json
9800ab3f...), and both read the same for every control: bounds-malloc, uaf-malloc and chunk-free-90
fault. Records: `~/asr-work/eval/run-6/corpora/memcached-allocator-repros-virtual-nested-pools/` (new)
and `~/asr-work/fix-memcached/results/corpus-old-patch/` (old). The verdict lines are
[`corpus-new-verdicts.jsonl`](corpus-new-verdicts.jsonl) and [`corpus-old-verdicts.jsonl`](corpus-old-verdicts.jsonl).

| case | old patch | new patch |
|---|---|---|
| 00 to 04 | CAUGHT, cause 25 at `read_probe` | the same |
| 05 | NO-READING: the fixed run faults with cause 24 | CAUGHT, cause 28 at `write_probe` |
| 06 | CAUGHT, cause 28 at `write_probe` | the same |
| 07 | CAUGHT, cause 28 at `read_probe` | the same |
| 08 | CAUGHT, cause 28 in `mc_case_body` (declared site) | the same |

Case 05 changed, as predicted. Its setup writes a sentinel into byte 0 of the neighbouring live item.
Under the old patch that byte belonged to the item's parent capability, so the fixed run's `slabs_free`
handed CREVOKE an untagged operand (cause 24). The new patch keeps no reference in the item, the fixed
run prints VERDICT FIXED, and the buggy run's one-byte overflow faults at the chunk bound.

The app's safety fixtures (`host/build-sublet.sh`, `memcached-safety.dom`) also build with patch 0006.
Their runner needs the module-based route, so they did not run. The lifetime controls under
`controls/lifetime/` (91 to 93) describe the old layout: control 91's `generation` variant loads a
generation from byte 0 of a chunk, which the new patch no longer stores.

## Limits

- `slabs_free` searches its class's page list, so its cost grows with the class's pages (at most 17 here).
- Restart mode (`-e`) stays unsupported.
- The allocators clear only their own links. What a chunk's previous owner wrote into it stays, as
  upstream leaves it, so a reissued item can still hold that owner's references, for example its
  hash-chain link to another item. Clearing reissued memory is the separate initialization question.
- The measured tree is 07f60dc6f7f. The commit after it changes one paragraph of patch 0006's
  description and no hunk.
- The two repetitions per arm are the pre-registered minimum. The cache boundary depends on how four
  workers on one hart interleave, which the host's load changes.
