# 8af6c71d96f4 — avfilter/vf_vif: Fix out of array access with small dimensions

## The defect

The filter mirrors out-of-range sample positions with a hand-rolled expression that reflects **once**: `jj >= w ? 2*w - jj - 1 : jj`. The loop produces indices up to `w - 1 + filt_w/2`, so whenever `w < filt_w/2` the reflected value is negative and `temp[jj]` reads **below** the buffer's base. The fix calls `avpriv_mirror`, which folds repeatedly.

## Upstream defect

- **Fix:** `8af6c71d96f4`, *"avfilter/vf_vif: Fix out of array access with small dimensions"*, `libavfilter/vf_vif.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin uses avpriv_mirror. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
                    ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 1 : ii);

                    img_coeff = src[ii * src_stride + j];
```

## The fix

```c
                    ii = avpriv_mirror(ii, h - 1);
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no VIF metric, no filter taps, no frames. The mirror expression is upstream's, evaluated at the largest index the loop produces, and the read is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
