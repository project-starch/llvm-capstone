# 0d4901071c74 — restart: fix potential double free

## The defect

`restart_get_kv()` frees the previous line at entry and re-publishes a new one only on the `RESTART_OK` path. Every other return — `RESTART_NOTAG`, `RESTART_BADLINE`, `RESTART_DONE` — leaves `c->line` pointing at the freed buffer, so the next call's entry free releases the same allocation again. The reader calls this function in a loop.

## Upstream defect

- **Fix:** `0d4901071c74`, *"restart: fix potential double free"*, `restart.c`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** The pin clears the field.

## The vulnerable code, quoted from the fix's parent

```c
    if (c->line != NULL) {
        free(c->line);
    }

    if (getline(&line, &len, c->f) != -1) {
```

## The fix

```c
    if (c->line != NULL) {
        free(c->line);
        c->line = NULL;
    }
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— neither `slabs.c` nor `cache.c` is in the path.

**Reduced:** no metadata file and no restart. One allocation stands for the line buffer, and the next call's entry free is reduced to a READ through the stale field -- a real second free aborts in glibc, which is rc=134 and not a verdict.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is conditional on revocation sweep timing. Nor upstream reachability: cases 01 and
02 are reached when a log watcher's socket closes while its buffer is full, which the reduction
supplies directly.
