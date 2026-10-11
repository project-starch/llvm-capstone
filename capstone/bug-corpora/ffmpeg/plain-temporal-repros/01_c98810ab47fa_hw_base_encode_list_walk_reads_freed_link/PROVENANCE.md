# c98810ab47fa — avcodec/hw_base_encode: fix use after free on close

## The defect

The close path walks the picture list with `for (pic = ctx->pic_start; pic; pic = pic->next)` and frees each node in the body. The increment then reads `pic->next` **out of the node it has just freed**, so every node after the first is reached through freed storage. The fix latches the next pointer before the free.

## Upstream defect

- **Fix:** `c98810ab47fa`, *"avcodec/hw_base_encode: fix use after free on close"*, `libavcodec/hw_base_encode.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin latches the link first. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    FFHWBaseEncodePicture *pic;

    for (pic = ctx->pic_start; pic; pic = pic->next)
        base_encode_pic_free(pic);
```

## The fix

```c
    for (FFHWBaseEncodePicture *pic = ctx->pic_start, *next_pic = pic; pic; pic = next_pic) {
        next_pic = pic->next;
        base_encode_pic_free(pic);
    }
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no encoder, no hardware context, no pictures. Two nodes are allocated and linked, and the `next` field is the allocation's first word; the walk is reduced to the increment that reads it after the free.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
