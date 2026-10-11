# 4b2248594c7f — avcodec/jpeg2000dec: clear pointer which become stale in get_ppt

## The defect

`get_ppt` grows `tile->packed_headers` with `av_realloc`. `tile->packed_headers_stream` is a `GetByteContext` whose pointers are **into** that buffer, and it was left holding the old base — so a later `s->g = tile->packed_headers_stream` reads freed storage while passing the reader's own bounds checks. The fix zeroes the context so it must be re-created.

## Upstream defect

- **Fix:** `4b2248594c7f`, *"avcodec/jpeg2000dec: clear pointer which become stale in get_ppt"*, `libavcodec/jpeg2000dec.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin zeroes the context. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
        tile->packed_headers = new;
    } else
        return AVERROR(ENOMEM);
    memcpy(tile->packed_headers + tile->packed_headers_size,
           s->g.buffer, n - 3);
```

## The fix

```c
    memset(&tile->packed_headers_stream, 0, sizeof(tile->packed_headers_stream));
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no JPEG 2000 codestream and no tiles. The buffer is grown by a real realloc forced to move, the reader context is reduced to its base pointer, and the read through it is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
