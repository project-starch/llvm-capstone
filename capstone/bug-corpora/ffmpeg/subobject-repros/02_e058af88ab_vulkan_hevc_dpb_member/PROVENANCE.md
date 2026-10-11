# e058af88ab — the DPB walk writes ref_src[16], which is the next member

## The defect

The decoded-picture-buffer walk runs over `DPB[32]` while its targets are `HEVC_MAX_REFS = 16`
wide, so the index can reach 31. The **first** out-of-bounds write is `ref_src[16]`, which is
`h265_refs[0]` — the immediately following member of the same allocation.

## Upstream defect

- **Fix:** `e058af88ab`. Trailers: `Fixes: out of array write`. It adds
  `if (nb_refs >= HEVC_MAX_REFS) return AVERROR_INVALIDDATA;`.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: YES.**

## The vulnerable code, quoted from the PIN

`n9.0.1:libavcodec/vulkan_hevc.c`:

```c
for (int i = 0; i < FF_ARRAY_ELEMS(l->DPB); i++) {
    ...
    int idx = nb_refs;
    err = vk_hevc_fill_pict(avctx, &hp->ref_src[idx], ...);      /* :760 -- unguarded */
```

The targets are declared `HEVC_MAX_REFS` wide at `:123-125`, while `l->DPB` is `HEVCFrame DPB[32]`
(`n9.0.1:libavcodec/hevc/hevcdec.h:453`). One allocation, `libavcodec/decode.c:2352`.

**Liveness:** the fix's `nb_refs >= HEVC_MAX_REFS` guard is absent and the call sits unguarded
at `:760`.

## Containment holds for the FIRST crossing only, and that is why the case is reduced to it

Writing all sixteen extra entries the `DPB[32]` walk permits would run 128 bytes into a 64-byte
member and **leave the allocation** — a different claim. The first version of this case did exactly
that, and the containment check refused it, which is recorded here rather than quietly fixed. The
case now writes **one** entry past the array, the first crossing, and asserts that the far end of
the neighbouring member survives.

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
