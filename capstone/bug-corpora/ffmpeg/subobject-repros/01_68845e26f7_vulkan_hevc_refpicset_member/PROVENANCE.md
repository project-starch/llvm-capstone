# 68845e26f7 — the HEVC reference-set fill spills eight bytes into the next array

## The defect

Three eight-entry reference-set arrays are filled from a reference list that holds up to
`HEVC_MAX_REFS = 16`, so the fill can run eight bytes past an array — **wholly into its
neighbour**, which is the next member of the same struct.

## Upstream defect

- **Fix:** `68845e26f7`. Trailer: `Fixes: out of array write`. It adds a three-way guard against
  `FF_ARRAY_ELEMS` of each array.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: YES.**

## The vulnerable code, quoted from the PIN

`n9.0.1:libavcodec/vulkan_hevc.c`:

```c
memset(hp->h265pic.RefPicSetStCurrBefore, 0xff, 8);          /* :768 -- the width, upstream's own */
for (int i = 0; i < h->rps[ST_CURR_BEF].nb_refs; i++)
    hp->h265pic.RefPicSetStCurrBefore[i] = j;                /* :775 -- unguarded */
```

The array's width of **8** is proved by the pin's own `memset`. The loop bound comes from a
`RefPicList` holding up to `HEVC_MAX_REFS = 16`
(`n9.0.1:libavcodec/hevc/hevc.h:120-122`). The struct is **one** allocation:
`libavcodec/decode.c:2352` calls `av_refstruct_alloc_ext(hwaccel->frame_priv_data_size, …)` with
`frame_priv_data_size = sizeof(HEVCVulkanDecodePicture)`.

**Liveness:** the fix's three-way `FF_ARRAY_ELEMS` guard is absent; the pin runs straight from the
`nb_refs` loop at `:766-767` into the `memset` at `:768` and the unguarded fill.

**The most tightly contained row in the corpus.** The spill is exactly 8 bytes into an 8-byte
neighbour, so there is **no magnitude at which it leaves the allocation**. The case asserts the
field after the three arrays is untouched.

**Reachability note:** Vulkan-hwaccel builds only, and `nb_refs > 8` rests on the fix's own trailer
rather than on anything reproduced here.

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
