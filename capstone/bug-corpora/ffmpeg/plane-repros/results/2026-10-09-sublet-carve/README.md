# plane-repros: the Sublet port of the frame carve (`sublet-carve`), 2026-10-09

**Question.** Column 3 for the plane case: with av_frame_get_buffer's carve ported to Sublet --
the frame buffer lent LINEAR by the Sublet heap, one Sublet region per plane, each plane pointer an
alias bounded to its plane -- is the alphablend read past the alpha plane caught?

**Build.** `shared/sublet-frame.c`, linked with `-Wl,--wrap=av_frame_get_buffer`, so the case is
unchanged. It asks the real av_frame_get_buffer for the layout on a scratch frame, then issues the
same layout as Sublet regions: a plane's region is its allocated extent,
`av_image_fill_plane_sizes` at `FFALIGN(height, 32)`, exactly what get_video_buffer reserves.

**Result: NOT CAUGHT, as pre-registered (03c471feabcc).** The port issued four regions, alpha
`[2144, 3168)` = 32 rows of 32 bytes; the defective read is at offset 160 of that plane, inside the
region, and the run reproduces the defect. The fixed arm stays FIXED.

**The port is live, shown in the same boot.** `plane-bound-control` (one byte past the alpha
plane's extent) faults with cause 5 at the probe; `plane-free-control` (an alpha read after
av_frame_free, which is the heap's revoke of the block) faults with cause 24 at the probe. Both
fixed arms complete.

**What this means.** The bound that would catch this read is `linesize * height`, tighter than
FFmpeg's own allocation of the plane -- and FFmpeg's SIMD tails and av_image_copy_to_buffer read
over padded_height on purpose, so that bound faults correct code. No allocator-side mechanism,
Sublet's included, catches a read that stays inside what the allocator handed out (README, "Arms
not measured here"); the plane is the one carved case the Sublet carve does not fix.
