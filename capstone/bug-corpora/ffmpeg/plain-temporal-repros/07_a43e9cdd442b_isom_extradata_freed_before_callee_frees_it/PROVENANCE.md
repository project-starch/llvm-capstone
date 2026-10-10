# a43e9cdd442b — avformat/isom: don't free extradata before calling ff_get_extradata

## The defect

The MPEG-4 descriptor reader freed `st->codecpar->extradata` and left the field pointing at the released buffer. The very next line calls `ff_get_extradata`, which releases the old extradata itself — a second free of the same allocation. The fix removes the caller's free, since the callee already owns that step.

## Upstream defect

- **Fix:** `a43e9cdd442b`, *"avformat/isom: don't free extradata before calling ff_get_extradata"*, `libavformat/isom.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The caller's free is deleted at the pin. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
        if (!len || (uint64_t)len > (1<<30))
            return AVERROR_INVALIDDATA;
        av_free(st->codecpar->extradata);
        if ((ret = ff_get_extradata(fc, st->codecpar, pb, len)) < 0)
            return ret;
```

## The fix

```c
        if (!len || (uint64_t)len > (1<<30))
            return AVERROR_INVALIDDATA;
        if ((ret = ff_get_extradata(fc, st->codecpar, pb, len)) < 0)
            return ret;
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no MP4 descriptors and no codec parameters. The field is a bare pointer and the callee's release is reduced to a read through the stale field at the labelled probe, so the case reports a verdict instead of aborting in glibc the way a real second free would.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
