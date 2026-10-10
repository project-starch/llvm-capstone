# 905a4324030e — libavfilter/showcwt: fix OOB write for DU/RL position init

## The defect

`s->pos` is the row (DU) or column (RL) currently being written, and was initialised and wrapped to `s->sono_size`. Since `sono_size = s->h - s->bar_size` equals `s->h` whenever `bar_ratio == 0`, the valid indices are `0..sono_size-1` and `pos == sono_size` writes one row or column outside the plane.

## Upstream defect

- **Fix:** `905a4324030e`, *"libavfilter/showcwt: fix OOB write for DU/RL position init"*, `libavfilter/avf_showcwt.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin initialises it to the last valid row. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    case DIRECTION_RL:
    case DIRECTION_DU:
        s->pos = s->sono_size;
        break;
```

## The fix

```c
    case DIRECTION_RL:
    case DIRECTION_DU:
        s->pos = FFMAX(s->sono_size - 1, 0);
        break;
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no CWT and no frames. The plane is a bare allocation of h rows and the cursor is each arm's own expression, with the write reduced to the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
