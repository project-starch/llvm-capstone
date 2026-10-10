# e7793811f8c8 — logger: fix use-after-free of closed watcher

## The defect

When a watcher's buffer is full, `logger_thread_write_entry()` polls the watchers to drain them. That poll can discover the socket is gone and call `logger_thread_close_watcher()`, which nulls the global slot `watchers[w->id]` and then frees the watcher — after which the function writes `w->failed_flush` into the freed struct through its own local `w`.

## Upstream defect

- **Fix:** `e7793811f8c8`, *"logger: fix use-after-free of closed watcher"*, `logger.c`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** Superseded by the next case's fix.

## The vulnerable code, quoted from the fix's parent

```c
        while (!w->failed_flush &&
                (skip_scr = (char *) bipbuf_request(w->buf, scratch_len + 128)) == NULL) {
            if (logger_thread_poll_watchers(0, x) <= 0) {
                L_DEBUG(...);
                w->failed_flush = true;
            }
        }
```

## The fix

```c
                // Oddity; poll_watchers can free *w, recheck it.
                if (watchers[x] != NULL) {
                    w->failed_flush = true;
                }
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— neither `slabs.c` nor `cache.c` is in the path.

**Reduced:** no logger thread, no sockets, no bipbuf. One allocation stands for the watcher, the global slot is a single variable, and the flag store is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is conditional on revocation sweep timing. Nor upstream reachability: cases 01 and
02 are reached when a log watcher's socket closes while its buffer is full, which the reduction
supplies directly.
