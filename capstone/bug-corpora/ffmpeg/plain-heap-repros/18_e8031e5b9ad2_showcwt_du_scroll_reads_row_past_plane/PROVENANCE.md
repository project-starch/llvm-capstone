# e8031e5b9ad2 — avfilter/avf_showcwt: fix out-of-bounds read in du scroll

## The defect

The DU scroll copies each row from the row below it: `memmove(dst, dst + linesize, s->w)` for `y` in `[0, sono_size)`. With `bar_ratio = 0` the filter sets `bar_size = 0` and `sono_size = s->h`, so the final iteration reads row `s->h` — one whole `linesize` past a plane holding exactly `h` rows.

## Upstream defect

- **Fix:** `e8031e5b9ad2`, *"avfilter/avf_showcwt: fix out-of-bounds read in du scroll"*, `libavfilter/avf_showcwt.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin stops one row earlier. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
                for (int y = 0; y < s->sono_size; y++) {
                    uint8_t *dst = s->outpicref->data[p] + y * linesize;

                    memmove(dst, dst + linesize, s->w);
                }
```

## The fix

```c
                for (int y = 0; y < s->sono_size - 1; y++) {
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no CWT, no sonogram, no frames. The plane is a bare allocation of h rows and the scroll is upstream's loop, with the crossing source row probed.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
