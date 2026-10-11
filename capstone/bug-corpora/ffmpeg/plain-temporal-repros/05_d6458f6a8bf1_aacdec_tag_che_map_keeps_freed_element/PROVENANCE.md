# d6458f6a8bf1 — avcodec/aacdec: Fix heap-use-after-free in USAC decoding

## The defect

A `ChannelElement` is reachable from two tables: `ac->che[type][id]`, which owns it, and `ac->tag_che_map[][]`, which caches it for tag lookups. `che_configure` frees it with `av_freep(&ac->che[type][id])` — clearing only the owning pointer — so the cached entry in `tag_che_map` still names the freed object and a later lookup returns it. The fix nulls the matching cache entries first.

## Upstream defect

- **Fix:** `d6458f6a8bf1`, *"avcodec/aacdec: Fix heap-use-after-free in USAC decoding"*, `libavcodec/aac/aacdec.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin clears the cache entries. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
        if (ac->che[type][id]) {
            ac->proc.sbr_ctx_close(ac->che[type][id]);
        }
        av_freep(&ac->che[type][id]);
```

## The fix

```c
            for (int i = 0; i < FF_ARRAY_ELEMS(ac->tag_che_map); i++) {
                for (int j = 0; j < MAX_ELEM_ID; j++) {
                    if (ac->tag_che_map[i][j] == ac->che[type][id])
                        ac->tag_che_map[i][j] = NULL;
                }
            }
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no AAC stream, no SBR, no element tags. Two table slots point at one allocation; the free clears one and the lookup through the other is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
