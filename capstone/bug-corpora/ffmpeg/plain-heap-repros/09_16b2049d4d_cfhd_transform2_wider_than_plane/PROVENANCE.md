# 16b2049d4d — reject transform-2 output wider than the plane

## The defect

The transform-2 path writes **twice** the declared lowpass width into the plane buffer, while the guard checked only that the lowpass width itself was at least 3. A stream could therefore declare a width that passed the guard and whose doubled output did not fit the plane.

## Upstream defect

- **Fix:** `16b2049d4d`, *"avcodec/cfhd: reject transform-2 output wider than the plane"*, `libavcodec/cfhd.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin carries the added term in both transform-2 guards. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```
lowpass_width < 3 || lowpass_height < 3) {
```

## The fix

```
lowpass_width < 3 || lowpass_height < 3 || lowpass_width * 2 > s->plane[plane].width) {
```

## Why the doubling matters

The crossing length is a **multiple** of the stream-declared value, so a bound on the declared value alone does not bound the write. That is the whole content of the fix, and it is what the case asserts before writing.

## What is real here, and what is reduced

**Real:** the arithmetic and the shape — which allocation is crossed and by how much.

**Reduced:** no demuxer, muxer or codec context; the buffers are plain allocations and the consumer
is reduced to the access that crosses, so the crossing is attributable to the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm crosses a
direct allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions. Nor
upstream reachability of the specific sizes chosen here.
