# 0f605245cf3f — Fix buffer-overrun when logging key to delete in binary protocol.

## The defect

The verbose branch printed the binary-protocol key with a `"%s"` conversion. The key is `binary_get_key(c)` — `c->rcurr - keylen`, a region inside the connection read buffer — delimited by `nkey` and by no terminator, so the conversion scans past the key and out of the allocation.

## Upstream defect

- **Fix:** `0f605245cf3f`, *"Fix buffer-overrun when logging key to delete in binary protocol."*, `memcached.c`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** The pin carries the counted loop.

## The vulnerable code, quoted from the fix's parent

```c
    if (settings.verbose > 1) {
        fprintf(stderr, "Deleting %s\n", key);
    }
```

## The fix

```c
    if (settings.verbose > 1) {
        int ii;
        fprintf(stderr, "Deleting ");
        for (ii = 0; ii < nkey; ++ii) {
            fprintf(stderr, "%c", key[ii]);
        }
        fprintf(stderr, "\n");
    }
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is — neither the slab allocator nor `cache.c` is involved.

**Reduced:** no server, no connection, no protocol parse. The key is a bare allocation filled with non-zero bytes and the conversion's scan is reduced to the loop that leaves it, so the crossing is attributable to the labelled probe rather than to stdio. The real trigger also needs `-vv`; the reduction supplies the branch directly.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific trigger chosen here.
