# e8714f6f93d1 — avcodec/h264: Clear delayed_pic on deallocation

## The defect

`h->DPB` is **one** `av_mallocz_array` block of pictures. `h->delayed_pic[]` is a reorder array holding **interior** pointers, `&h->DPB[i]`. `ff_h264_free_tables` frees the whole block with `av_freep(&h->DPB)` without clearing the reorder array, so a later `h->delayed_pic[i]->reference = 0` **writes** into freed storage. The fix zeroes the reorder array first.

## Upstream defect

- **Fix:** `e8714f6f93d1`, *"avcodec/h264: Clear delayed_pic on deallocation"*, `libavcodec/h264.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin zeroes the reorder array before the free. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    if (free_rbsp && h->DPB) {
        for (i = 0; i < H264_MAX_PICTURE_COUNT; i++)
            ff_h264_unref_picture(h, &h->DPB[i]);
        av_freep(&h->DPB);
```

## The fix

```c
        memset(h->delayed_pic, 0, sizeof(h->delayed_pic));
        av_freep(&h->DPB);
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no H.264 stream, no picture reordering. One block stands for the DPB, one interior pointer for a reorder entry, and the `reference = 0` store is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
