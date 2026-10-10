# 3fbb9560821b — avfilter/afir: bound the crossfades by the samples of the input frame

## The defect

The crossfade paths copy and read a fixed `min_part_size` samples from the input frame at `offset`, bounding nothing against the frame's own `nb_samples`. When the final partition of a frame is shorter than `min_part_size`, the `memcpy` and the fade loops read past the end of the input audio plane.

## Upstream defect

- **Fix:** `3fbb9560821b`, *"avfilter/afir: bound the crossfades by the samples of the input frame"*, `libavfilter/afir_template.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin clamps the length. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
            memcpy(dst, in, sizeof(ftype) * min_part_size);
```

## The fix

```c
    const int nb_samples = FFMIN(min_part_size, s->in->nb_samples - offset);
    ...
            memcpy(dst, in, sizeof(ftype) * nb_samples);
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no FIR filter, no partitions, no audio frames. The plane is a bare allocation of the frame's samples and the copy length is each arm's own expression, with the first crossing sample probed.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
