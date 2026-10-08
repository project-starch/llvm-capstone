# 716d2a47c565 — swscale/ops_dispatch: fix use-after-free when adding opaque passes

## The defect

`compile_single` frees the pass struct `p` on the path that hands its compiled parts onward, and then stores `comp->backend->flags` into the output. `comp` is an **interior pointer** at `&p->comp`, so that read goes through the freed allocation. The fix reads the field from the local copy `c` instead, which is not part of the freed block.

## Upstream defect

- **Fix:** `716d2a47c565`, *"swscale/ops_dispatch: fix use-after-free when adding opaque passes"*, `libswscale/ops_dispatch.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin reads from the local copy. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
        if (ret >= 0) {
            (*output)->backend = comp->backend->flags;
            ff_sws_pass_link_output(*output, link);
        }
```

## The fix

```c
            (*output)->backend = c.backend->flags;
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no swscale graph, no passes, no backends. The struct is a bare allocation whose first word stands for the `backend` field, and the interior alias is taken at a fixed offset.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
