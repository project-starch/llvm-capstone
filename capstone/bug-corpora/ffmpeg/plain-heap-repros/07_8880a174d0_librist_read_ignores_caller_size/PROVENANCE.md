# 8880a174d0 — honor the caller buffer size in librist_read

## The defect

`librist_read` received the caller's buffer size in `size`, then **overwrote** it with the incoming payload's length before copying. A payload larger than the caller's buffer was copied in full, past the end of the destination.

## Upstream defect

- **Fix:** `8880a174d0`, *"avformat/librist: honor the caller buffer size in librist_read"*, `libavformat/librist.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin carries the `FFMIN` form. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```
size = data_block->payload_len;
```

## The fix

```
size = FFMIN(data_block->payload_len, size);
```

## What is real here, and what is reduced

**Real:** the arithmetic and the shape — which allocation is crossed and by how much.

**Reduced:** no demuxer, muxer or codec context; the buffers are plain allocations and the consumer
is reduced to the access that crosses, so the crossing is attributable to the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm crosses a
direct allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions. Nor
upstream reachability of the specific sizes chosen here.
