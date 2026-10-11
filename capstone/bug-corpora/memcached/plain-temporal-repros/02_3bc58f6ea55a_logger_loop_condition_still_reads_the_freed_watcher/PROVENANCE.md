# 3bc58f6ea55a — logger: closed worker fix again

## The defect

The residual of the previous fix. Guarding only the **write** left every subsequent **read** of `w` in the loop unguarded: with `failed_flush` never set, the `while` condition `!w->failed_flush && (bipbuf_request(w->buf, ...)) == NULL` is re-evaluated against freed memory — and cannot terminate. The second fix skips the rest of the iteration instead of setting a flag.

## Upstream defect

- **Fix:** `3bc58f6ea55a`, *"logger: closed worker fix again"*, `logger.c`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** The pin carries the `continue`.

## The vulnerable code, quoted from the fix's parent

```c
            if (logger_thread_poll_watchers(0, x) <= 0) {
                L_DEBUG(...);
                // Oddity; poll_watchers can free *w, recheck it.
                if (watchers[x] != NULL) {
                    w->failed_flush = true;
                }
```

## The fix

```c
                // Oddity; poll_watchers can free *w, recheck it.
                if (watchers[x] == NULL) {
                    continue;
                }
                w->failed_flush = true;
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— neither `slabs.c` nor `cache.c` is in the path.

**Reduced:** no logger thread and no bipbuf. The loop is reduced to the condition that re-reads the watcher, and that read is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is conditional on revocation sweep timing. Nor upstream reachability: cases 01 and
02 are reached when a log watcher's socket closes while its buffer is full, which the reduction
supplies directly.
