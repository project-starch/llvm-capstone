# 989444060d5f — avfilter/vf_lut3d: compute size2 after the 3DLUTSIZE directive

## The defect

`parse_dat` sets `size = 33` and immediately derives `size2 = size * size`. The optional `3DLUTSIZE n` line that follows can lower `size`, and `allocate_3dlut` then allocates `n^3` entries — but the write index still uses the stale `size2`. A declared size of 2 gives an 8-entry array indexed at 1089.

## Upstream defect

- **Fix:** `989444060d5f`, *"avfilter/vf_lut3d: compute size2 after the 3DLUTSIZE directive"*, `libavfilter/vf_lut3d.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin computes the stride after the directive. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    lut3d->lutsize = size = 33;
    size2 = size * size;

    NEXT_LINE(skip_line(line));
    if (!strncmp(line, "3DLUTSIZE ", 10)) {
        size = strtol(line + 10, NULL, 0);
```

## The fix

```c
    }
    size2 = size * size;

    ret = allocate_3dlut(ctx, size, 0);
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no .dat file and no LUT parsing. The two sizes are the directive's and the default's, and the write is reduced to the labelled probe at the first element past.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
