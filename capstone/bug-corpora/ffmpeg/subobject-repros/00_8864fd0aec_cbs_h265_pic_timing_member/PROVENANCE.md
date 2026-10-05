# 8864fd0aec — pic_timing writes one element past its array into the next member

## The defect

The H.265 `pic_timing` SEI reader bounds the decoding-unit count **inclusively** and then loops to
it **inclusively** as well, so the index reaches `HEVC_MAX_SLICE_SEGMENTS` — one past the array it
writes.

## Upstream defect

- **Fix:** `8864fd0aec`. Trailers: `Fixes: out of array write`. It bounds the count by
  `FFMIN(pic_width_in_ctbs_y * pic_height_in_ctbs_y, HEVC_MAX_SLICE_SEGMENTS) - 1`.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: YES.**

## The vulnerable code, quoted from the PIN

The two adjacent members, `n9.0.1:libavcodec/cbs_h265.h:627-628`:

```c
uint16_t num_nalus_in_du_minus1[HEVC_MAX_SLICE_SEGMENTS];
uint32_t du_cpb_removal_delay_increment_minus1[HEVC_MAX_SLICE_SEGMENTS];
```

with `HEVC_MAX_SLICE_SEGMENTS = 600` at `n9.0.1:libavcodec/hevc/hevc.h:150`. The reader,
`n9.0.1:libavcodec/cbs_h265_syntax_template.c`:

```c
ue(num_decoding_units_minus1, 0, HEVC_MAX_SLICE_SEGMENTS);          /* :1997 -- INCLUSIVE */
...
for (i = 0; i <= current->num_decoding_units_minus1; i++) {         /* :2004 */
    ues(num_nalus_in_du_minus1[i], 0, HEVC_MAX_SLICE_SEGMENTS, 1, i);   /* :2005 */
```

**Liveness:** the inclusive bound is still at `:1997` and the fix's `cbs_h265_pic_size_in_ctbs()`
call does not appear in this function at the pin.

**Why this is a sub-object crossing and not an ordinary overflow.** Index 600 of a `uint16_t[600]`
sits at byte offset **1200**, which is 4-aligned and is exactly where
`du_cpb_removal_delay_increment_minus1` begins — so the two-byte write lands on that member's
**low half**, and its high half survives. The case asserts both. The whole struct is **one**
allocation: `libavcodec/cbs_sei.c:257` calls `av_refstruct_alloc_ext(desc->size, …)`.

**Reachability note:** `cbs_h265` is not the HEVC decoder; the entry points are the `h265_metadata`
and `trace_headers` BSFs and the VAAPI/Vulkan H.265 encoders.

## What is real here, and what is reduced

**Real:** the allocator. `av_refstruct_allocz` over the pinned `libavutil/refstruct.c`, where
`refstruct.c:109` is `av_malloc(size + REFCOUNT_OFFSET)` — header and payload in **one**
allocation. So the fact the case turns on, that the two members share an allocation, is a property
of FFmpeg's allocator and this struct's layout, not of the driver.

**Reduced:** the struct is cut down to the members the crossing involves, **at their real declared
widths**, and the surrounding decode is cut to the index arithmetic that is the defect. The case
asserts the layout rather than assuming it: it checks that one past the first member IS the second.

## What the run establishes, and what it does not

**Establishes:** the defect is real, reproduces, and **no configuration we have catches it** — the
crossing is inside one allocation, so a per-allocation bound is in bounds for it. The oracle is the
upstream fix, which is the only one available for this class.

**Does not establish** reachability upstream, anything about silicon, or the Capstone and CheriBSD
readings — those arms are declared completions and are **not** measured; a domain build for this
corpus does not exist.

**ASan is blind here, and that is measured rather than asserted**, with a positive control that
fires: see the corpus README and `results/20261005-native-subobject/`.
