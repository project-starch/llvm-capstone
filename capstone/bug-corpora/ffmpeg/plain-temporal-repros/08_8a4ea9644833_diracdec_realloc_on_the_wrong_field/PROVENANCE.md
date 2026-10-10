# 8a4ea9644833 — diracdec: use correct buffer for slice_params_buf realloc

## The defect

`decode_lowdelay` grows the slice-parameter buffer, but passed `s->thread_buf` as the block to realloc while storing the result in `s->slice_params_buf`. `realloc` releases `s->thread_buf`'s block, so that field is left naming freed storage while a different field owns the new block — one allocation with two owners, one of them stale.

## Upstream defect

- **Fix:** `8a4ea9644833`, *"diracdec: use correct buffer for slice_params_buf realloc"*, `libavcodec/diracdec.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin reallocs the right field. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    if (s->slice_params_num_buf != (s->num_x * s->num_y)) {
        s->slice_params_buf = av_realloc_f(s->thread_buf, s->num_x * s->num_y, sizeof(DiracSlice));
```

## The fix

```c
        s->slice_params_buf = av_realloc_f(s->slice_params_buf, s->num_x * s->num_y, sizeof(DiracSlice));
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no Dirac stream and no slices. Two fields stand for the two buffers, the realloc is real and forced to move, and the use of the stale field is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
