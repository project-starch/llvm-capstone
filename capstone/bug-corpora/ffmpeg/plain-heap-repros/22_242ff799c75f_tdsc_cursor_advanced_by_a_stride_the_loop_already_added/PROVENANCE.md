# 242ff799c75f — avcodec/tdsc: remove double stride adjustment

## The defect

In the `CUR_FMT_MONO` paths the inner loops already advance `dst` by exactly `FFALIGN(cursor_w, 32) * 4 == cursor_stride` per row — they step 32 pixels at a time, four bytes each — and a second `dst += ctx->cursor_stride - ctx->cursor_w * 4` was then applied. Each row therefore advances by a stride plus up to 124 surplus bytes, so the later rows are written past the end of the allocation.

## Upstream defect

- **Fix:** `242ff799c75f`, *"avcodec/tdsc: remove double stride adjustment"*, `libavcodec/tdsc.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The MONO paths carry no trailing advance at the pin. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
            dst += ctx->cursor_stride - ctx->cursor_w * 4;
        }
```

## The fix

```c
(both occurrences inside the CUR_FMT_MONO case deleted; the loop body alone advances a full stride)
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no TDSC stream and no cursor bitmap. The buffer is sized stride * height as upstream's is, the two advances are upstream's, and the first row that leaves the allocation is probed.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
