# dc87758775e2 — avio: fix potential crashes when combining ffio_ensure_seekback + checksums

## The defect

`ffio_ensure_seekback` installs a larger IO buffer and frees the old one. `s->checksum_ptr` is a long-lived pointer **into** the old buffer, saved so the checksum can be updated later, and it was not rebased — so the later `update_checksum` reads freed storage. The fix converts it to an offset across the replacement.

## Upstream defect

- **Fix:** `dc87758775e2`, *"avio: fix potential crashes when combining ffio_ensure_seekback + checksums"*, `libavformat/aviobuf.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin saves and restores the offset. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    int filled = s->buf_end - s->buffer;

    buf_size += s->buf_ptr - s->buffer + max_buffer_size;
    ...
    s->buf_end = buffer + (s->buf_end - s->buffer);
    s->buffer = buffer;
    s->buffer_size = buf_size;
    return 0;
```

## The fix

```c
    ptrdiff_t checksum_ptr_offset = s->checksum_ptr ? s->checksum_ptr - s->buffer : -1;
    ...
    if (checksum_ptr_offset >= 0)
        s->checksum_ptr = s->buffer + checksum_ptr_offset;
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no AVIOContext, no protocol, no checksum function. The buffer is replaced by a real allocate-and-free pair and the checksum update is reduced to one read through the stale field at the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
