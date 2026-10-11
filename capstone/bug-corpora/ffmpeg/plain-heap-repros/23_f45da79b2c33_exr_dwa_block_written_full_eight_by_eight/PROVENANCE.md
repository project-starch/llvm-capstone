# f45da79b2c33 — avcodec/exr: Dont access outside xsize/ysize

## The defect

The DWA block loop steps `x` and `y` in 8s and unconditionally writes a full 8x8 block through `bo`/`go`/`ro`, which point into `td->uncompressed_data`. When `xsize`/`ysize` are not multiples of 8 the last block writes past the row and, on the last row, past the buffer.

## Upstream defect

- **Fix:** `f45da79b2c33`, *"avcodec/exr: Dont access outside xsize/ysize"*, `libavcodec/exr.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin carries the clamps. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
                for (int yy = 0; yy < 8; yy++) {
                    for (int xx = 0; xx < 8; xx++) {
                        const int idx = xx + yy * 8;
```

## The fix

```c
            int bw = FFMIN(8, td->xsize - x);
            int bh = FFMIN(8, td->ysize - y);
...
                for (int yy = 0; yy < bh; yy++) {
                    for (int xx = 0; xx < bw; xx++) {
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no EXR file, no DWA codec, no channels. The destination is a bare allocation of one tile and the block loop is upstream's, with the first byte outside it probed.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
