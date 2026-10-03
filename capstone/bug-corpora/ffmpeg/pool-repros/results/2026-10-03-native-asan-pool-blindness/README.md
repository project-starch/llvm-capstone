# ASan is blind to a use-after-return-to-pool at our FFmpeg pin — measured, with upstream's own patch as the control (2026-10-03)

**Question.** Every bundle in this tree that reports ASan clean on a pooled reuse has rested on *our*
reading of why: a return to an `AVRefStructPool` never reaches `free()`, so a free-keyed tool has
nothing to key on. That is an argument, not a measurement, and CLAUDE.md is explicit that a clean
result is not evidence until the check is known to be able to produce the opposite.

**What makes it measurable.** Upstream commit **`e6255fb822`** (2026-09-13), *"avutil/{buffer,refstruct}:
annotate pooled memory for ASan and MSan"*, adds exactly the missing annotation — `FF_ASAN_POISON` on
`pool_return_entry`, `FF_ASAN_UNPOISON` on `refstruct_pool_get_ext`. It is **not in our pin**
(`git cat-file -t` → `commit`; `git merge-base --is-ancestor e6255fb822 n9.0.1` → rc=1; cherry-pick
probe empty while the `4b9c4b9cfb` control returns `716d2a47c5`). So it is a positive control we can
apply *to the pin* to make the blindness fire.

**Pre-registration.** The prediction — clean unpatched, `use-after-poison` patched — was written into
[`../../../../docs/ref/ffmpeg-live-defect-triage.md`](../../../../docs/ref/ffmpeg-live-defect-triage.md)
and pushed in **`6bfcabcb9316`** *before* this run.

## Verdict

**Confirmed, and three-sided.**

| arm | variant | rc | ASan verdict | flagged |
|---|---|---:|---|---|
| control | plain `malloc`/`free`, no pool | 1 | **`heap-use-after-free`** | `ctl_uaf.c:12` |
| **armA** | the pin, `refstruct.c` unpatched | **0** | **NO REPORT** | — |
| **armB** | the pin + `e6255fb822` | 1 | **`use-after-poison`** | `pool_stale.c:32` |

All three were built with the same compiler and the same flags
(`clang -fsanitize=address -fno-omit-frame-pointer -g -O1`), against the same `libavutil`; armA and
armB differ **only** in whether `refstruct.c` carries the annotation.

The two arms diverge at exactly the stale read:

    armA   live_read=0xA0  released=1  stale_read=0xA0  reuse_same_address=1  after_new_owner_stale_read=0xCC  DONE
    armB   live_read=0xA0  released=1  (abort: use-after-poison at pool_stale.c:32)

## Why each arm is load-bearing

- **The control exists because armA's zero is otherwise worthless.** On the first attempt the control
  failed to *compile* (a missing `<string.h>`) and the harness printed "INSTRUMENT DEAD" — which is a
  build error wearing the costume of a reading. Fixed, it reports `heap-use-after-free` with rc=1, so
  ASan is demonstrably active in this exact toolchain and these exact flags.
- **armB proves the check fires on this very access**, not merely on *some* access. Same address, same
  one-byte read, same line. That is stronger than the control alone: a detector can be live in general
  and still structurally unable to see a particular class.
- **armA is therefore a measured blindness.** ASan stays silent while the program reads an entry that is
  resting on `pool->available_entries`, watches the pool hand the same storage back
  (`reuse_same_address=1`), and then reads the *new owner's* bytes through the stale pointer
  (`after_new_owner_stale_read=0xCC`). No `free()` ever happens, which is the whole mechanism.

**ASan names the pool entry, not a program allocation.** armB's region attribution is
`av_refstruct_alloc_ext_c` (`refstruct.c:110`) ← `refstruct_pool_get_ext` (`:287`) ←
`av_refstruct_pool_get` (`:318`), confirming the poisoned region is the pooled entry itself.

## Two limits, taken from upstream's diff rather than inferred

1. **The poisoning is gated on `if (!pool->free_entry_cb)`.** Upstream's reason, in its own words:
   entries with an entry free callback *"own allocations while they rest in the pool"* and
   LeakSanitizer *"does not follow pointers stored in poisoned memory"*, so those stay addressable. So
   **even on master today the blindness is narrowed, not closed** — a pool with a free callback is
   still invisible. Our probe pool is created with flags `0` and has no callback, so it *is* in the
   covered set; the uncovered set is strictly larger.
2. **`buffer.c` covers only `av_buffer_default_free` buffers**, *"since custom pool allocators may not
   be compatible"*. `AVBufferPool`s with a custom allocator — which is what a ported allocator is —
   are not annotated.

Both limits matter for the paper: they answer a "but ASan catches this now" objection using the gate in
upstream's own patch, not our framing.

### Arm C — PRE-REGISTERED, NOT YET RUN: does the `free_entry_cb` gate leave a real pool uncovered?

Limit 1 above is read from the diff, not measured. **Arm C measures it**: the same `pool_stale.c`
sequence, the same armB-patched `refstruct.c`, but the pool created through
`av_refstruct_pool_alloc_ext` with a no-op `free_entry_cb` instead of `av_refstruct_pool_alloc`.

**Prediction, registered before the run: armC is CLEAN — no `use-after-poison` — even though armB
reports one on the identical access.** If armC instead reports, then the gate does not do what the diff
says and limit 1 must be withdrawn.

**Which real pools this bears on, checked at the pin rather than assumed:**

| pool | created by | covered by `e6255fb822`? |
|---|---|---|
| h264 `qscale_table` / `mb_type` / `motion_val` / `ref_index` | `av_refstruct_pool_alloc(…, 0)` (`h264_slice.c:169-174`) | **yes** — the simple form passes all-NULL callbacks (`refstruct.c:337`) |
| VVC `rpl_tab` / `tab_dmvr_mvf` (**case 3's storage**) | `av_refstruct_pool_alloc(…, 0)` (`vvc/dec.c:387,394`) | **yes** |
| **MPVPicture** — the picture pool of the whole MPEG-1/2/4, H.263, VC-1, RV30/34/40, WMV2, MSMPEG4 family | `ff_mpv_alloc_pic_pool` → `av_refstruct_pool_alloc_ext(…, mpv_pic_init, mpv_pic_reset, **mpv_pic_free**, NULL)` (`mpegpicture.c:90-96`) | **NO** |
| `AVContainerFifo` slots | `av_refstruct_pool_alloc_ext(…, container_fifo_free_entry, NULL)` (`container_fifo.c:87`) | **NO** |

So the covered/uncovered split is not hypothetical: **FFmpeg's most widely used picture pool is in the
uncovered set.** Recorded here with its call site because an earlier draft of this reasoning asserted the
opposite for h264 — that its pools carried callbacks — and the pin says they do not.


## What this does and does not say

- **It does** establish, by measurement with a working positive control, that a use-after-return-to-pool
  on `AVRefStructPool` is invisible to ASan at the version this project compiles.
- **It does** replace our own reasoning about *why* with a primary source: upstream added the
  annotation, so the gap it repairs was real.
- **It does not** measure any capability arm. No Capstone, no QEMU, no board — this is native x86-64 and
  is about ASan only.
- **It does not** come from a corpus case. `pool_stale.c` is a direct probe of the allocator, not a
  reduction of an upstream defect, and it is deliberately minimal so that nothing but the pool is in
  play. The corpus cases remain the place where real defects are measured.
- **N = 1 per arm**, which is adequate here only because the result is deterministic by construction
  (no timing, no concurrency, no allocator nondeterminism in play) — re-running reproduces it exactly.

## Files

`result-lines.txt` — every line quoted above.
`pool_stale.c` — the pooled-reuse probe (armA and armB, same source).
`ctl_uaf.c` — the instrument's positive control.
`refstruct-e6255fb822.patch` — the exact patch applied for armB, so the arm is reproducible.

Reproduce: compile `pool_stale.c` against a `-fsanitize=address` build of the pin's `libavutil` for
armA; for armB apply `refstruct-e6255fb822.patch` (plus `libavutil/sanitizer.h` from that commit) to
`refstruct.c`, compile it separately, and link that object **ahead of** the archive.
