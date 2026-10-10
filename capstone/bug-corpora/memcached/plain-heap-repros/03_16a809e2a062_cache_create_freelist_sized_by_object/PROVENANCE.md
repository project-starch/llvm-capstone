# 16a809e2a062 — Issue 161 incorrect allocation in cache_create

## The defect

`cache_create` allocates the object cache's freelist — an array of `void *` — but sized it by `bufsize`, the size of the object being cached, while `ret->freetotal = initial_pool_size` promises 64 slots regardless. For any cached object smaller than a pointer the array is short by that ratio, and `do_cache_free`'s store at `ptr[63]` is far outside it.

## Upstream defect

- **Fix:** `16a809e2a062`, *"Issue 161 incorrect allocation in cache_create"*, `cache.c`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** The construct was removed rather than fixed: 1.6.45's freelist is a STAILQ threaded through the free objects, with no pool array.

## The vulnerable code, quoted from the fix's parent

```c
    cache_t* ret = calloc(1, sizeof(cache_t));
    char* nm = strdup(name);
    void** ptr = calloc(initial_pool_size, bufsize);
```

## The fix

```c
    void** ptr = calloc(initial_pool_size, sizeof(void*));
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is — neither the slab allocator nor `cache.c` is involved.

**Reduced:** no server, no connection, no protocol parse. The array is allocated by each arm's own expression and the freelist store is reduced to the labelled probe at the first byte past. Upstream's own regression test for this fix is `cache_bulkalloc(1)` in testapp.c, which is the same trigger: a one-byte cached object.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific trigger chosen here.
