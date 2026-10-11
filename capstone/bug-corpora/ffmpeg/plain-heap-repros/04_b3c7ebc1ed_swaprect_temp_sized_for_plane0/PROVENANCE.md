# b3c7ebc1ed — a scratch row buffer sized for the luma plane, overrun by a chroma row

## The defect

`vf_swaprect` keeps one scratch buffer, `s->temp`, and swaps two rectangles through it a row at a
time. `config_input` sized that buffer from **plane 0 alone**, while `filter_frame` copies a row of
**every** plane through it. For a semi-planar 4:2:0 format the two disagree at odd widths.

## Upstream defect

- **Fix:** `b3c7ebc1ed`, *"avfilter/vf_swaprect: size the temp row buffer for the widest plane"*,
  `libavfilter/vf_swaprect.c`. Its message records an out-of-array access and names a reproducer
  `odd17_nv12`, which is where the width 17 used here comes from.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** `n9.0.1:libavfilter/vf_swaprect.c:214-223` carries the fix
  verbatim. A fix-reversal.

**Citation constraint applied:** this commit's body carries `Found-by:` and `Signed-off-by:`
trailers naming people with email addresses. It is cited here by **hash, subject and path only**,
and the body is not quoted. The subject itself names no one.

## The vulnerable code, quoted from the fix's parent

    b3c7ebc1ed^:libavfilter/vf_swaprect.c:213
        s->temp = av_malloc_array(inlink->w, s->pixsteps[0]);

and the use, unchanged at the pin:

    n9.0.1:libavfilter/vf_swaprect.c:187
        memcpy(s->temp, src, pw[p] * s->pixsteps[p]);

## The fix

    for (int p = 0; p < s->nb_planes; p++) {
        int shift = p == 1 || p == 2 ? s->desc->log2_chroma_w : 0;
        int width = AV_CEIL_RSHIFT(inlink->w, shift);
        ...
        size = FFMAX(size, width * s->pixsteps[p]);
    }
    s->temp = av_malloc(size);

## Why it only shows at odd widths

NV12 has `pixsteps {1, 2}` and `log2_chroma_w 1`, so for width `w`:

| plane | bytes in a row |
|---|---|
| 0, luma | `w * 1` |
| 1, chroma | `AV_CEIL_RSHIFT(w, 1) * 2` = `2 * ceil(w / 2)` |

Equal for even `w`; the chroma row is **one byte longer** for odd `w`. At `w = 17` the buffer is 17
bytes and the row is 18. That is why the upstream reproducer is `odd17`, and the case asserts the
one-byte difference rather than assuming it.

## What is real here, and what is reduced

**Real:** the arithmetic — NV12's actual pixsteps and chroma shift, at upstream's own width — and
the fact that one direct allocation is sized by one consumer and used by another.

**Reduced:** no filter graph, no frame, no pixel-format descriptor. The two row lengths are computed
directly and the copy is written byte-wise so the crossing lands on the labelled probe rather than
inside libc's `memcpy`.

## What the run establishes, and what it does not

**Establishes:** the defect reproduces from the upstream fix differential — the buggy arm's copy
crosses a direct allocation and the fixed arm's does not, differing by exactly the sizing term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions. The
CheriBSD one is explicitly conditional: a 17-byte request may be bounded to a 32-byte size class, in
which case the crossing stays inside the capability and the arm completes. That must be measured
in-guest for this request size, not carried over from a sibling corpus — a size class is a step
function, which is what refuted one of this corpus's earlier predictions.
