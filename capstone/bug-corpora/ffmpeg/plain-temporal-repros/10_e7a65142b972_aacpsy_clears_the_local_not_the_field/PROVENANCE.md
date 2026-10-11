# e7a65142b972 — avcodec/aacpsy: Clear the correct pointer

## The defect

On the allocation-failure path `psy_3gpp_init` called `av_freep(&pctx)` — releasing the private context and clearing the **local** variable. `ctx->model_priv_data`, which owns the object, kept the released address, so `psy_3gpp_end` frees it a second time. `av_freep` did its job; it was pointed at the wrong pointer.

## Upstream defect

- **Fix:** `e7a65142b972`, *"avcodec/aacpsy: Clear the correct pointer"*, `libavcodec/aacpsy.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin clears the field. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    pctx->ch = av_mallocz_array(ctx->avctx->channels, sizeof(AacPsyChannel));
    if (!pctx->ch) {
        av_freep(&pctx);
        return AVERROR(ENOMEM);
    }
```

## The fix

```c
        av_freep(&ctx->model_priv_data);
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no psy model and no channels. One allocation, one owning field, one local; the teardown's use of the field is the labelled probe rather than a second free.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
