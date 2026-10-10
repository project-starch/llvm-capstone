# ca1c1f29ce47 — avformat/img2enc: Check split planes packet size

## The defect

In split-planes mode the muxer writes `ysize`, `usize`, `usize` (and optionally another `ysize`) bytes at successive offsets into `pkt->data`, with the sizes derived from `par->width/height` rather than from `pkt->size`. A packet smaller than the geometry implies is read past its end.

## Upstream defect

- **Fix:** `ca1c1f29ce47`, *"avformat/img2enc: Check split planes packet size"*, `libavformat/img2enc.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin carries the guard. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
        if ((ret = write_and_close(s, &pb[0], pkt->data                , ysize)) < 0 ||
            (ret = write_and_close(s, &pb[1], pkt->data + ysize        , usize)) < 0 ||
            (ret = write_and_close(s, &pb[2], pkt->data + ysize + usize, usize)) < 0)
```

## The fix

```c
        if (ysize + 2*usize + (desc->nb_components > 3) * ysize > pkt->size) {
            ret = AVERROR(EINVAL);
            goto fail;
        }
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no muxer, no image writer, no pixel descriptors. The packet is a bare allocation and the three plane reads are byte loops, with the first crossing byte probed.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
