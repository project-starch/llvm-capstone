# 1a00ea51cb — indexes `control_url` at `strlen()-1` and underflows out of the member

## The defect

For a relative SDP control URL the parser tests the last character of `control_url`
without first checking that the string is non-empty. `strlen` returns `size_t`, so `strlen("") - 1`
is `SIZE_MAX`.

## Upstream defect

- **Fix:** `1a00ea51cb`. It takes the length once and short-circuits: `if (len == 0 || rtsp_st->control_url[len - 1] != '/')`.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: NO — the fix is already in at the pin.**

## The vulnerable code, quoted from upstream

The fix's own diff, `libavformat/rtsp.c`:

```diff
                 if (proto[0] == '\0') {
                     /* relative control URL */
-                    if (rtsp_st->control_url[strlen(rtsp_st->control_url)-1]!='/')
+                    size_t len = strlen(rtsp_st->control_url);
+                    if (len == 0 || rtsp_st->control_url[len - 1] != '/')
                         av_strlcat(rtsp_st->control_url, "/",
                                    sizeof(rtsp_st->control_url));
```

The struct, `n9.0.1:libavformat/rtsp.h`, with `control_url` **fifth**, which is the fact the case
turns on:

```c
typedef struct RTSPStream {
    URLContext *rtp_handle;
    void *transport_priv;
    int stream_index;
    int interleaved_min, interleaved_max;
    char control_url[MAX_URL_SIZE];
```

`MAX_URL_SIZE` is 4096 at `n9.0.1:libavformat/internal.h:30`, and the allocation is
`av_mallocz(sizeof(RTSPStream))` at `libavformat/rtsp.c:286` and `:525`.

**Liveness: a FIX-REVERSAL.** The fix is already in at the pin, read from the pinned
source rather than from ancestry — `git merge-base --is-ancestor` has called backported fixes live
before, so it is not used here. Two-sided, so the probe is known to fire:

- `n9.0.1:libavformat/rtsp.c` contains the fixed `len == 0 || rtsp_st->control_url` — **1 occurrence**
- the same file contains the pre-fix `control_url[strlen(rtsp_st->control_url)-1]` — **0 occurrences**

The case reconstructs the **pre-fix consumer shape against the shipped allocator**, which is this
tree's convention and not a weaker kind of case: it is stated at
`../../memcached/allocator-repros/README.md:132-135`, and most cases in this tree are fix-reversals.
Liveness is **recorded, never required**; requiring it is what left this cell nearly empty, and that
inference was retracted on `dev`.

**Why the access COMPLETES on a capability machine, which is easy to get wrong.**
`control_url + SIZE_MAX` is, in the modular arithmetic a byte pointer obeys, `control_url - 1`.
`control_url` does **not** start at offset 0 — it sits after two pointers and three ints, at offset
**28**, which the case asserts and prints — so `control_url - 1` is the last byte of
`interleaved_max` and is comfortably **inside** the allocation. The address is representable and in
bounds; nothing faults. What makes the byte reachable is the member's non-zero offset, *not* the
index wrapping. A reading of "completes" here must not be credited to the wrap being harmless in
general: had `control_url` been the first member, the same wrap would have left the allocation.

**Reachability note:** the RTSP demuxer's SDP parser, for a stream whose `a=control:` line
is empty or absent.

## What is real here, and what is reduced

**Real:** the allocator. `av_refstruct_allocz` over the pinned `libavutil/refstruct.c`, where
`refstruct.c:109` is `av_malloc(size + REFCOUNT_OFFSET)` — header and payload in **one**
allocation. So the fact the case turns on, that the crossed region shares an allocation with its
neighbour, is a property of FFmpeg's allocator and this struct's layout, not of the driver.

The case **asserts** the layout rather than assuming it, so a compiler or platform that laid the
members out differently would fail the case rather than quietly measure something else.

**Reduced:** the struct is cut to the members up to and including `control_url`, at their
real widths — which is what fixes the offset — because members after it cannot be reached by this
defect. `MAX_URL_SIZE` is **not** reduced.

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
