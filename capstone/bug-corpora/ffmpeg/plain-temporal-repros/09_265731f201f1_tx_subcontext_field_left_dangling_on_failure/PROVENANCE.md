# 265731f201f1 — lavu/tx: reset subcontext pointer if initialization fails

## The defect

When sub-transform initialisation fails, `ff_tx_init_subtx` released the subcontext with `av_free(sub)` — a **local** copy of `s->sub`. The owning field kept the released address, so teardown dereferenced and freed it a second time. The fix frees through the field itself, which also clears it.

## Upstream defect

- **Fix:** `265731f201f1`, *"lavu/tx: reset subcontext pointer if initialization fails"*, `libavutil/tx.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin frees through the field. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    av_free(sub);

end:
    av_free(cd_matches);
```

## The fix

```c
    av_freep(&s->sub);

end:
    av_free(cd_matches);
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no transform context and no sub-transforms. One allocation, one owning field, one local copy; the teardown's use of the field is the labelled probe rather than a second free, which would abort in glibc instead of reporting.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
