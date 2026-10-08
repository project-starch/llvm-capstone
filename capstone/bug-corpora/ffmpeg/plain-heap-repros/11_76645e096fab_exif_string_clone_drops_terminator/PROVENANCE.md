# 76645e096fab — avcodec/exif: also copy zero termination for AV_TIFF_STRING

## The defect

`exif_read_values` allocates a string with `av_mallocz(entry->count + 1)` — the characters plus a terminator. `exif_clone_entry` cloned it through `EXIF_COPY`, whose length is `src->count * sizeof(*fname)`, i.e. `count`, so the clone is an **unterminated** `count`-byte allocation and the first `strlen` or `%s` on it reads past the end.

## Upstream defect

- **Fix:** `76645e096fab`, *"avcodec/exif: also copy zero termination for AV_TIFF_STRING"*, `libavcodec/exif.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin copies count+1. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
        case AV_TIFF_STRING:
            EXIF_COPY(dst->value.str, src->value.str);
            break;
```

## The fix

```c
        case AV_TIFF_STRING:
            dst->value.str = av_memdup(src->value.str, src->count+1);
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no EXIF entries and no TIFF parsing. The source carries its terminator as upstream's does, the clone is each arm's own length, and the consumer's scan is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
