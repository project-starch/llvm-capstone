# ba28222a14ab — avcodec/ratecontrol: Fix double free on error

## The defect

`ff_rate_control_uninit` releases the rate-control expression with `av_expr_free`, which does **not** clear its argument, and the field was left set. The mpeg encoders declare `FF_CODEC_CAP_INIT_CLEANUP`, so an init failure runs this uninit and `ff_mpv_encode_end` runs it again — releasing the same object twice. The fix makes the teardown idempotent.

## Upstream defect

- **Fix:** `ba28222a14ab`, *"avcodec/ratecontrol: Fix double free on error"*, `libavcodec/ratecontrol.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin clears the field. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    av_expr_free(rcc->rc_eq_eval);
    av_freep(&rcc->entry);
```

## The fix

```c
    av_expr_free(rcc->rc_eq_eval);
    rcc->rc_eq_eval = NULL;
    av_freep(&rcc->entry);
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no encoder and no rate control. One allocation and one field; the second teardown's use of the field is the labelled probe rather than a second release.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
