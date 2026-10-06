# 275e217b10 — passes `av_strlcpy` a source length where its contract takes the destination size

## The defect

`parse_playlist` copies the `URI="…"` field of an `#EXT-X-KEY` line into `vs->key_uri`
using `av_strlcpy(vs->key_uri, ptr, end - ptr)`. `av_strlcpy`'s third parameter is the size of the
**destination**; `end - ptr` is the length of the **source**, which the playlist chooses.

## Upstream defect

- **Fix:** `275e217b10`. It passes `FFMIN(end - ptr + 1, sizeof(vs->key_uri))`, restoring the contract, and separately switches the delimiter search from `","` to `'"'`.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: NO — the fix is already in at the pin.**

## The vulnerable code, quoted from upstream

The fix's own diff, `libavformat/hlsenc.c`:

```diff
                 ptr += strlen("URI=\"");
-                end = av_stristr(ptr, ",");
+                end = strchr(ptr, '"');
                 if (end) {
-                    av_strlcpy(vs->key_uri, ptr, end - ptr);
+                    av_strlcpy(vs->key_uri, ptr,
+                               FFMIN(end - ptr + 1, sizeof(vs->key_uri)));
                 } else {
-                    av_strlcpy(vs->key_uri, ptr, sizeof(vs->key_uri));
+                    ret = AVERROR_INVALIDDATA;
+                    goto fail;
                 }
```

The adjacent members, `n9.0.1:libavformat/hlsenc.c:174-178`:

```c
    char key_file[LINE_BUFFER_SIZE + 1];
    char key_uri[LINE_BUFFER_SIZE + 1];
    char key_string[KEYSIZE*2 + 1];
    char iv_string[KEYSIZE*2 + 1];
```

with `LINE_BUFFER_SIZE` = `MAX_URL_SIZE` = 4096 (`hlsenc.c:72`, `internal.h:30`) and `KEYSIZE` = 16
(`hlsenc.c:71`). And `av_strlcpy`'s contract, `n9.0.1:libavutil/avstring.h`: it writes at most
`size - 1` bytes of `src` plus a terminating NUL.

**Liveness: a FIX-REVERSAL.** The fix is already in at the pin, read from the pinned
source rather than from ancestry — `git merge-base --is-ancestor` has called backported fixes live
before, so it is not used here. Two-sided, so the probe is known to fire:

- `n9.0.1:libavformat/hlsenc.c` contains the fixed `FFMIN(end - ptr + 1, sizeof(vs->key_uri))` — **1 occurrence**
- the same file contains the pre-fix `av_strlcpy(vs->key_uri, ptr, end - ptr)` — **0 occurrences**

The case reconstructs the **pre-fix consumer shape against the shipped allocator**, which is this
tree's convention and not a weaker kind of case: it is stated at
`../../memcached/allocator-repros/README.md:132-135`, and most cases in this tree are fix-reversals.
Liveness is **recorded, never required**; requiring it is what left this cell nearly empty, and that
inference was retracted on `dev`.

**Why this is a sub-object crossing.** `key_uri` is followed directly by `key_string` and
then `iv_string` in one allocation, so a copy that overruns `key_uri` lands on its neighbours. The
case asserts the adjacency.

**The magnitude is attacker-chosen**, which separates this row from cases 3, 5 and 8 where the
overrun is fixed by a type's length: the playlist supplies `end - ptr`.

**One thing this row is NOT credited with.** The same hunk switches the delimiter from `","` to
`'"'`, which fixes a *truncation* of URIs containing a comma. That is a separate correctness fix and
is not part of this crossing; the arms differ only by the size argument.

**Reachability note:** the HLS muxer re-reading an existing playlist (`parse_playlist`) that
carries an `#EXT-X-KEY` line — i.e. append mode.

## What is real here, and what is reduced

**Real:** the allocator. `av_refstruct_allocz` over the pinned `libavutil/refstruct.c`, where
`refstruct.c:109` is `av_malloc(size + REFCOUNT_OFFSET)` — header and payload in **one**
allocation. So the fact the case turns on, that the crossed region shares an allocation with its
neighbour, is a property of FFmpeg's allocator and this struct's layout, not of the driver.

The case **asserts** the layout rather than assuming it, so a compiler or platform that laid the
members out differently would fail the case rather than quietly measure something else.

**Reduced:** the buffers are cut from `LINE_BUFFER_SIZE + 1` (4097) to **64**. The defect
is that a destination-size parameter receives a source length, so what matters is that the length can
exceed the destination, not the destination's absolute size. `av_strlcpy` itself is **not** reduced —
it is the function whose contract is misused, so reducing it would hide the bug; it is reproduced as
exactly what C requires of it.

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
