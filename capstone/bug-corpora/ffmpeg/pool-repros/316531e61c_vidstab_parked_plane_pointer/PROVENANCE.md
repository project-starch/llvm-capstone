# Provenance

**Tier: allocator literal, consumer and library both reduced.** The pool is the
real `libavutil/buffer.c`. `libvidstab` is not linked: its state is modelled as
the one field that matters, the retained `src` pointer, which is what the
upstream commit message describes in its own words.

- **Fix:** `316531e61c` — *"avfilter/vidstabtransform: always use in-place transform path"*, 2026-04-01. The filter stops choosing between paths so the library never keeps a shallow copy.
- **File:** `libavfilter/vf_vidstabtransform.c`.
- **CVE:** `NO VERIFIED CVE`.
- **Live at the pin:** no. Present at n8.0 through n8.1.2, fixed before 9.0.1.
- **Build dependency:** the real filter needs `--enable-libvidstab`. The fixture does not, because the retained pointer, not the stabiliser, is the defect.

## Scope

Reducing the library to one pointer is the largest reduction in this corpus and
is stated here rather than implied. What it preserves is the property under
test: a raw pointer into pooled storage, held across a frame boundary by code
that never took a reference. What it drops is every other reason
`vsTransformPrepare` might touch that storage.
