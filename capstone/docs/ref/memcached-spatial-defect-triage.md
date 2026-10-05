# memcached: triage for SPATIAL defects, and why class B is empty here for a structural reason

Mirror of `wireshark-spatial-defect-triage.md` for memcached. The temporal hunt filtered this class
out — `bug-corpora/memcached/allocator-repros/README.md:52-57` *"filtered on temporal-safety
vocabulary (55 hits)"* — so it had never been triaged. See the retraction in
`spatial-vs-temporal-three-programs.md:20`.

**VERDICT SUPERSEDED 2026-10-05: three spatial cases are now built and measured.** The original verdict below — 0 class-B, structural — was reached with a WORDING filter, and the shape search this file itself prescribed found three. See §6. The structural argument in §2 survives in part and is corrected there. The spatial defects memcached fixes are in its *metadata arrays*
and its *protocol buffers*, not in item data inside a slab page. One candidate is more interesting
than its class suggests — see §4.

## The instrument

| | |
|---|---|
| clone | `/tmp/capstone/upstream/memcached`, 2,900 commits |
| pin | tag **`1.6.45`**, which **is upstream head** — `memcached-allocator-defect-triage.md:15-17`: *"commits after the pin: 0"* |
| population | whole history to the pin = **2,349** commits |
| filter 1 | spatial wording — **20 survive**. Integer and refcount overflow excluded: memcached case `03` is exactly that trap, an *integer* overflow driving an early free, i.e. temporal |
| filter 2 | which bound does the overflow cross, read from the allocation site |
| filter 3 | the pin **is** head, so no candidate can be live and every case here is a **fix-reversal** — which is what all five existing memcached corpus cases already are. This makes memcached a **control for filter 3**; see §5 |
| script | `bug-corpora/tools/spatial-triage.py --program memcached` |

## The classes, as they came out

| class | count | what they are |
|---|---:|---|
| **A** — crosses the `malloc` bound | 3 | `shrink`, `sublet` and CHERI already fault. A tie row |
| **STACK** — a stack array | 4 | no allocator involved; out of class |
| **GLOBAL** — a static array index | 4 | the slab/LRU metadata arrays. See §3 |
| **B** — inside a slab page or a `cache.c` object | **0** | see §2 |
| unresolved | 9 | allocation site more than one alias hop from the hunk. **Unresolved, not zero** |

The script itself reports `? 13 / A 3 / STACK 4`; the four GLOBAL ones are inside its 13 `?`,
adjudicated by hand in §3, which leaves 9 genuinely unresolved.

## 1. The class-A defects, for the tie row

- `ddee3e2` — `authfile.c:44` `auth_data = calloc(1, sb.st_size);` and the parser then reads
  `auth_cur[x]` past it. The fix is `calloc(1, sb.st_size + 1)`. A plain heap overread: per-object
  bounds already cover it, which is why it is a control and not a contribution.
- `11b5f9b` — `proxy_lua.c:1403`, a `realloc`'d `proxy_hook_tagged` array.
- `d5d9ff0` — `items.c:270-271`, two plain `malloc`s for a histogram and a 2 MiB buffer.
  **Cited by hash only: this commit's subject names an outside contributor, and a person's name
  must not enter a committed file here.**

## 2. Why class B is structurally empty, not merely unfound

memcached's nested allocators are `slabs.c` (items carved from 1 MiB pages) and `cache.c` (an
object cache). For a spatial defect to be class B, something must write past an **item's data**
while staying inside its slab page. That requires the item's own length accounting to be wrong —
and memcached sizes an item exactly from the key and value lengths at `do_item_alloc`, with the
protocol parser validating those lengths *before* allocation.

So the length bugs land one layer out, in the parser's own `malloc`'d or stack buffers, which is
precisely where the class-A and STACK candidates sit: **7** of the 20 candidates have a subject
beginning `proxy:`, and each overflows a buffer the proxy allocated, never an item.

**This is a claim about the 2,349 commits searched, not a proof.** What would settle it: a search
for fixes that change an `item_make_header` / `ITEM_*` size computation, which is a *shape* search
rather than a wording search and is the obvious next instrument — the same move
`ffmpeg-pool-consumer-defects.md` made when wording failed on the temporal side.

## 3. The four slab/LRU candidates are GLOBAL, not B — verified at the pin

All four index a **static global array**, with no heap allocation anywhere in the diff (checked: 0
lines adding `malloc`/`calloc`/`do_item_alloc`/`slabs_alloc`/`cache_alloc` in any of them):

- `slabs.c:37` at the pin — `static slabclass_t slabclass[MAX_NUMBER_OF_SLAB_CLASSES];`
- `items.c:50-54` at the pin — `static item *heads[LARGEST_ID];`, `static item *tails[LARGEST_ID];`,
  `static itemstats_t itemstats[LARGEST_ID];`, `static unsigned int sizes[LARGEST_ID];`,
  `static uint64_t sizes_bytes[LARGEST_ID];`

| candidate | the off-by-one |
|---|---|
| `08474d8` | `if (cur > MAX_NUMBER_OF_SLAB_CLASSES)` → `>=`, in the slab picker loop |
| `a2fc8e9` | the `MAX_NUMBER_OF_SLAB_CLASSES` boundary throughout, plus redefining it as `(63 + 1)` |
| `a3fcd40` | `if (sid > POWER_LARGEST)` → `>=`, in the LRU crawler |
| `fa51ad8` | `s_cls->slabs--` moved; a counter ordering bug rather than an index overflow |

Our `level0` arm already faults on a global out-of-bounds — `mcapp` fixture 7 `global_oob` is
registered `FAULT oob` on *every* arm. So these carry no nested-allocator contribution.

## 4. The one finding worth keeping: `a3fcd40` may be a real instance of the `global_merged` gap

`items.c:50-54` declares **five same-shaped `static` arrays back to back**. That is exactly the
shape fixture `global_merged` exists to probe — tshark fx9 / FFmpeg fx10, where the compiler merges
adjacent statics into one `.L_MergedGlobals` and per-object global bounds consequently **do not
hold**. That fixture is registered `RETURN` on *every* arm including `chunks`: **no arm catches it.**

So `a3fcd40`, an off-by-one indexing `heads[]`/`tails[]` past `POWER_LARGEST`, would on a merged
layout read or write into the *neighbouring array* rather than out of bounds at all — a real
upstream defect whose detectability depends on the compiler's merging decision, not on the
allocator.

**This is a hypothesis with a cheap test, not a result.** It needs: (a) checking whether our build
actually merges those five arrays (read `.L_MergedGlobals` in the port's object file, the same way
FFmpeg fixture 8 established the merge in the first place), and (b) if merged, whether the
off-by-one's reachable index lands inside the merged group. Until both are done this is not a case,
and it must not be counted as one. Recorded here so it is not re-derived.

## 5. memcached measures filter 3's FALSE-POSITIVE rate, and it is not zero

This program is the one place where liveness has a ground truth independent of any probe: **the pin
is upstream head**, so *no* candidate can be live. Run against that, filter 3 reported:

| filter 3 said | count | truth |
|---|---:|---|
| `FIX-IN-PIN` | 12 | correct |
| `UNRESOLVED` | 4 | correct, and honestly refusing |
| **`DEFECT-LIVE`** | **4** | **wrong — all four are false positives** |

The mechanism: filter 3 looks for the fix's own added lines in the pinned file, and those lines can
have been *further modified* after the fix, so they are no longer present verbatim even though the
fix is. 4 of 20 is a ~20% false-live rate on this population.

**Consequence for the Wireshark results, stated plainly:** `1d8acb21ab` and `d24613c461` are
reported `DEFECT-LIVE` by the same filter, so that verdict alone would not be trustworthy. Both were
therefore confirmed **by reading the pinned source directly** — for `1d8acb21ab` the vulnerable
`[i + 6]` read is quoted from `v4.6.8:packet-solaredge.c:1029` and the fix's marker
`payload_length -= 6` has 0 occurrences in the pinned file. That hand check is what the liveness
claim rests on, not filter 3's label. Any future candidate needs the same treatment before the word
"live" is written down.

This is also why a `DEFECT-LIVE` from this script is a **candidate**, never a result.

## What this instrument cannot see

- **A defect never fixed upstream.** The population is fixes.
- **A defect whose fix does not word itself spatially.** 9 of 20 candidates also came back with an
  unresolved allocation site, and an unresolved site is **not** class A.
- **Item-size accounting bugs**, which is the one place class B could live here — those need the
  shape search in §2, not a wording search.
- **The proxy subsystem's reachability.** `#1308`'s `raw_line()` underflow was rejected by upstream's
  own adjudication as *"reachable only by a privileged user writing a configuration that would never
  work"* (`allocator-repros/README.md:79`); that rejection stands and is about reachability, not
  class.


## 6. SUPERSEDED: the shape search this file prescribed found three cases (2026-10-05)

§2 said the next instrument was *"a search for fixes that change an `item_make_header` / `ITEM_*`
size computation, which is a **shape** search rather than a wording search"*. Run, it gives **157**
commits touching an item-size computation, and **three** of them are reducible spatial defects — now cases **5, 6 and 7** of
`bug-corpora/memcached/allocator-repros`, measured 8/8 natively with the five temporal rows as a
regression control (`results/20261005-native-spatial/`).

| case | upstream | the crossing | leaves the chunk? |
|---|---|---|---|
| **5** | `2d61f18` | three bytes of a two-byte terminator, one byte past the item's data | **yes** — the case sizes the item to the measured chunk stride |
| **6** | `78eb770` | four bytes of flags into suffix space that does not exist, over the value | **no** — a sub-object crossing inside the chunk |
| **7** | `ecdb011` | an unterminated key read forward past the key field | no fixed extent |

**All three have subjects that say "corruption", not "overflow"** — which is exactly why the wording
filter missed them, and is the measured cost of a wording filter on top of the one already recorded
for the temporal hunt.

**What §2's structural argument got right and wrong.** Right: an item is sized exactly from the key
and value lengths, so there is no *generic* item-data overflow. Wrong: it concluded no class-B
defect could live here. Two of the three are size-computation slips at the item's own field
boundaries — case 5 a hard-coded copy length, case 6 a field with no space allocated — which the
argument did not consider. Case 6 is the more interesting of the two: it stays **inside** the chunk,
and the slab port's bound *is* the chunk, so a chunk-granular bound cannot see it. That is the same
sub-object shape FFmpeg's `subobject-repros` corpus is built around.

The Capstone arms are **declared predictions, not measurements** — case 5 predicts a fault, case 6 a
completion, case 7 states that it depends on how far the scan runs. No domain build was made.

### The measured result lines (this corpus keeps no committed bundle, by its own `.gitignore`)

`runners/run-native.sh` exit **0** over all eight cases — the five temporal rows ran in the same
pass as a regression control. The three spatial rows:

```
    05_2d61f18_item_data_one_past VERDICT FIXED the two-byte copy ends exactly at the item's data end
    05_2d61f18_item_data_one_past VERDICT DEFECT-REPRODUCED the three-byte copy of a two-byte terminator wrote one byte past the item's data, into the next chunk of the slab page
    06_78eb770_suffix_write_no_space VERDICT FIXED the guard skipped the copy when no suffix space was allocated
    06_78eb770_suffix_write_no_space VERDICT DEFECT-REPRODUCED the four-byte flags copy overwrote the value's storage, because with nsuffix = 0 the suffix field has no room of its own
    07_ecdb011_unterminated_key_read VERDICT FIXED the bounded copy terminated at nkey, so the read stopped there
    07_ecdb011_unterminated_key_read VERDICT DEFECT-REPRODUCED the formatter read past the key, because nothing terminated it
```

The Capstone arms are declared predictions, not measurements, and point different ways on purpose:
case 5 predicts a fault (its crossing leaves the chunk), case 6 a completion (it does not), case 7
states that it depends on how far the scan runs. No domain build was made.

### A note on how to cite this search's numbers

**157 is a population, not a candidate count, and there is no intermediate number to quote.** A
regex over the diffs flagged 89 commits as "changing size arithmetic on item storage", and that
figure is *not* trustworthy: it matched logger and cachedump lines such as
`memcpy(le->key, ITEM_key(it), it->nkey)` that have nothing to do with an item's size. **Five
commits were read individually** — the ones whose subjects carry a spatial or corruption word — and
three of those became cases. The other 152 were **not** read one by one.

Quoting 89 as "candidates" would be the same mistake this tree already records on the FFmpeg side,
where *"overflow 2,415, out of array 1,062"* was measured in aggregate and never read case by case.
An aggregate is evidence that a population is large, never that its members were triaged.
