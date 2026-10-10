# d29ff88422 — writes `tile_sizes[MAX_TILES]` because its guard is both `>` and outside the loop

## The defect

The Vulkan AV1 hwaccel checks the tile count **once, before** the loop that increments it,
and with `>` where the array's last valid index needs `>=`. A tile group arriving with
`tileCount == MAX_TILES - 1` therefore writes index `MAX_TILES` on its second iteration.

## Upstream defect

- **Fix:** `d29ff88422`. It moves the check **inside** the loop and makes it `>=`.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: YES.**

## The vulnerable code, quoted from upstream

The fix's own diff, `libavcodec/vulkan_av1.c`:

```diff
-    /* Too many tiles, exceeding all defined levels in the AV1 spec */
-    if (ap->av1_pic_info.tileCount > MAX_TILES)
-        return AVERROR(ENOSYS);
-
     for (int i = s->tg_start; i <= s->tg_end; i++) {
+        /* Too many tiles, exceeding all defined levels in the AV1 spec */
+        if (ap->av1_pic_info.tileCount >= MAX_TILES)
+            return AVERROR(ENOSYS);
+
         ap->tile_sizes[ap->av1_pic_info.tileCount] = s->tile_group_info[i].tile_size;
```

The struct and the constant, `n9.0.1:libavcodec/vulkan_av1.c:24` and `:37-46`:

```c
#define MAX_TILES 256
...
typedef struct AV1VulkanDecodePicture {
    FFVulkanDecodePicture           vp;
    FFVulkanDecodeContext          *dec;
    uint32_t tile_sizes[MAX_TILES];
    /* Current picture */
    StdVideoDecodeAV1ReferenceInfo     std_ref;
```

**Liveness: LIVE AT THE `n9.0.1` PIN**, read from the pinned source rather than from
ancestry — `git merge-base --is-ancestor` has called backported fixes live before, so it is not
used here. Two-sided, so the probe is known to fire:

- `n9.0.1:libavcodec/vulkan_av1.c` contains the pre-fix `tileCount > MAX_TILES` — **1 occurrence**
- the same file contains the fixed `tileCount >= MAX_TILES` — **0 occurrences**

**Why this is a sub-object crossing.** Index 256 of a `uint32_t[256]` sits at byte offset
**1024**, which is exactly where `std_ref` begins — so the four-byte write lands on that member's
first four bytes and the rest of it survives. The case asserts both, and the surviving tail is what
distinguishes a sub-object crossing from a wild write. The struct is **one** allocation: it is the
current frame's `hwaccel_picture_private`.

**Two errors compound here, and the arms differ by both**, because either alone leaves the defect
reachable: the relation is wrong *and* the check is hoisted out of the loop it guards.

**Reachability note:** the Vulkan AV1 hardware decode path, for a stream whose tile count
reaches the spec's ceiling.

## What is real here, and what is reduced

**Real:** the allocator. `av_refstruct_allocz` over the pinned `libavutil/refstruct.c`, where
`refstruct.c:109` is `av_malloc(size + REFCOUNT_OFFSET)` — header and payload in **one**
allocation. So the fact the case turns on, that the crossed region shares an allocation with its
neighbour, is a property of FFmpeg's allocator and this struct's layout, not of the driver.

The case **asserts** the layout rather than assuming it, so a compiler or platform that laid the
members out differently would fail the case rather than quietly measure something else.

**Reduced:** `StdVideoDecodeAV1ReferenceInfo` is a Vulkan header type not available here
and stands in as a byte array — nothing about this defect depends on its contents, only on its being
the member that follows `tile_sizes`. `MAX_TILES` is **not** reduced, and the preceding members are
omitted because the crossing is at the array's far end.

## What the run establishes, and what it does not

**Establishes:** the defect is real and reproduces from the upstream fix differential, measured
two-sided with the control (fixed) arm run first.

**Does not establish** reachability upstream, anything about silicon, or the Capstone, PoisonCap,
CheriBSD and ASan readings. Those arms are **declared predictions** in `case.json` and are *not*
measured for this case. Measuring the Capstone arm needs a probe case in
`ports/ffmpeg/buffer-pool/security-tests` — the seam cases 0-2 use — and measuring ASan needs
`results/20261005-native-subobject/asan-probe.c` extended, with its positive control still firing.

**An arena caveat that bears on the Capstone and CheriBSD predictions.** This corpus's driver hands
the port's `av_malloc` **one** arena (`shared/driver.c` calls `ff2_memory_init`), and
`ports/ffmpeg/buffer-pool/src/shared/metadata-allocator.c:26` carves it by bumping a cursor,
returning a raw interior pointer and rounding every request up to 64 bytes;
`__builtin_capstone_cap_shrink` appears only in `src/capstone-domain/payload-capabilities.c:57`, on
pool *payload* blocks. So on this harness there is **no per-allocation bound on the struct at all**,
and a completion would be weaker evidence than "the bound is the whole allocation". The sub-object
claim does not rest on that — the crossing is interior by construction, which the case asserts on
the offsets — but the arm must not be read as a measured per-allocation bound.

**N = 1 per cell.**
