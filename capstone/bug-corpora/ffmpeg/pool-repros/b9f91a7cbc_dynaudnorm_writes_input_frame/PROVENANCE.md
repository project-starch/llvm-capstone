# Provenance

**Tier: LITERAL-traceable allocator and predicate, reduced consumer.**

- **Fix:** `b9f91a7cbc` — *"avfilter/af_dynaudnorm: make frame writable if it may be changed"*, 2022-02-27. `analyze_frame` gains a return type and calls `av_frame_make_writable(frame)` before the correction paths.
- **File:** `libavfilter/af_dynaudnorm.c`.
- **CVE:** `NO VERIFIED CVE`.
- **Live at the pin:** no.

## One candidate from this group was rejected

`5db1e07a62` (*"avfilter/af_speechnorm: check return value of
av_frame_make_writable()"*) is **not** in this corpus. The call was already
there; the fix only checks its result. That is error handling, not the defect.
