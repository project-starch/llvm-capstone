# ad956ff076ea — avfilter/vf_drawtext: Account for bbox text seperator

## The defect

`s->text` is sized as `LABEL_MAX_SIZE * (NUM_CLASSIFY + 1)` — room for the labels themselves. The assembly joins them with a two-character `", "` separator per classify label, so the worst case needs two more bytes per label plus the terminator, and the final `strcat` writes past the end.

## Upstream defect

- **Fix:** `ad956ff076ea`, *"avfilter/vf_drawtext: Account for bbox text seperator"*, `libavfilter/vf_drawtext.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin widens each slot. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
        s->text = av_mallocz(AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE *
                             (AV_NUM_DETECTION_BBOX_CLASSIFY + 1));
```

## The fix

```c
        s->text = av_mallocz((AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE + 1) *
                             (AV_NUM_DETECTION_BBOX_CLASSIFY + 1));
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no detection bboxes and no text rendering. The buffer is each arm's own expression and the assembly is a byte loop, so the crossing is attributable to the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
