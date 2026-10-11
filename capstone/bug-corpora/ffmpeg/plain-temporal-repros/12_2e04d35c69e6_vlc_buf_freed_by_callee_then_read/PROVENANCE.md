# 2e04d35c69e6 — avcodec/vlc: Attempt to free buf after use in ff_vlc_init_multi_from_lengths

## The defect

`ff_vlc_init_multi_from_lengths` allocates a scratch `VLCcode` table and passes it to `vlc_common_end`, which frees it when it differs from the caller's on-stack fallback. The next line then hands the same buffer to `vlc_multi_gen`, reading freed storage. The fix passes `buf` as the fallback so the callee does not release it, and frees it in the caller after the second call.

## Upstream defect

- **Fix:** `2e04d35c69e6`, *"avcodec/vlc: Attempt to free buf after use in ff_vlc_init_multi_from_lengths"*, `libavcodec/vlc.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin frees in the caller, after the use. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    ret = vlc_common_end(vlc, nb_bits, j, buf, flags, localbuf);
    if (ret < 0)
        goto fail;
    return vlc_multi_gen(multi->table, vlc, nb_elems, j, nb_bits, buf, logctx);
```

## The fix

```c
    ret = vlc_common_end(vlc, nb_bits, j, buf, flags, buf);
    if (ret < 0)
        goto fail;
    ret = vlc_multi_gen(multi->table, vlc, nb_elems, j, nb_bits, buf, logctx);
    if (buf != localbuf)
        av_free(buf);
    return ret;
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no VLC tables and no code lengths. The scratch buffer is a bare allocation, the callee's conditional release is reduced to the free itself, and vlc_multi_gen's use of it is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
