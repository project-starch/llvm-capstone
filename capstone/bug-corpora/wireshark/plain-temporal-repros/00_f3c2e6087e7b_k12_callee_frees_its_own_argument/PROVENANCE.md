# f3c2e6087e7b — More double-free fixes - destroy_k12_file_data() frees its argument, so calling g_free() on that argument after calling destroy_k12_file_data() is always an error.

## The defect

`destroy_k12_file_data()` releases the `k12_t` struct it is handed, not merely its contents — its last statement is `g_free(fd)`. Several error paths in `k12_open()` called it and then called `g_free(file_data)` on the same pointer, releasing one allocation twice.

## Upstream defect

- **Fix:** `f3c2e6087e7b`, *"More double-free fixes - destroy_k12_file_data() frees its argument, so calling g_free() on that argument after calling destroy_k12_file_data() is always an error."*, `wiretap/k12.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The caller's free is deleted at the pin.

## The vulnerable code, quoted from the fix's parent

```c
            if (file_seek(wth->fh, offset, SEEK_SET, err) == -1) {
                destroy_k12_file_data(file_data);
                g_free(file_data);
                return -1;
            }
```

## The fix

```c
            if (file_seek(wth->fh, offset, SEEK_SET, err) == -1) {
                destroy_k12_file_data(file_data);
                return -1;
            }
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— this code is outside `epan`'s wmem scopes.

**Reduced:** no K12 file and no wiretap state. The helper is reduced to the free it performs, and the caller's second release is reduced to a READ through the stale pointer -- a real second free aborts in glibc, which is rc=134 and not a verdict, and the dangling pointer is the same defect either way.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing. Nor upstream reachability of
the specific sequence chosen here.
