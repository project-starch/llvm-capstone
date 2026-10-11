# b2df2f4f22 — pass buffer size into put_system_header()

## The defect

`put_system_header` initialised its bit writer with a **constant** capacity of 128 bytes, whatever the cursor. Called with less than 128 bytes left in the muxer's buffer, it wrote past the end. The defect is in the capacity the callee was given, not in its writes.

## Upstream defect

- **Fix:** `b2df2f4f22`, *"avformat/mpegenc: pass buffer size into put_system_header()"*, `libavformat/mpegenc.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin carries the `buf_size` parameter and the `buf_end - buf_ptr` argument. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```
static int put_system_header(AVFormatContext *ctx, uint8_t *buf, int id)
    ...
    init_put_bits(&pb, buf, 128);
```

## The fix

```
static int put_system_header(AVFormatContext *ctx, uint8_t *buf, int buf_size, int id)
    ...
    init_put_bits(&pb, buf, buf_size);
    ...
    size = put_system_header(ctx, buf_ptr, buf_end - buf_ptr, id);
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
