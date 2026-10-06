# FFmpeg's eight cases in the four-arm design

The arm design the study settled on is **cumulative**: each step adds exactly one thing, so
the difference between two adjacent arms is that thing and nothing else.

| the design's arm | what it protects | FFmpeg's knob |
|---|---|---|
| base, unprotected | nothing | `FFAPP_HEAP=level0` (`-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0`) |
| `sysalloc-bounds` | system allocator, spatial | `FFAPP_HEAP=shrink` |
| **`sysalloc-sublet` — the floor** | system allocator, spatial **and** temporal | `FFAPP_HEAP=sublet` **+ `FFAPP_POOL=stock`** |
| **+ nested** | that, plus FFmpeg's own `AVBufferPool` | `FFAPP_HEAP=sublet` **+ `FFAPP_POOL=sublet`** |
| `cheribsd-revocation` | CheriBSD, revocation as it comes | the purecap build |

The knobs already existed; this file is the mapping, not a new mechanism. `FFAPP_POOL=stock`
is the floor because it is *upstream's pools on the Sublet heap* — the system allocator fully
protected, the nested allocator not — which is exactly what makes a catch on `FFAPP_POOL=sublet`
attributable to the nested allocator and nothing else.

## What is measured, by corpus

### `pool-repros` — 4 temporal cases: the nested allocator catches all four

Floor vs +nested, 16 of 16 cells as pre-registered
(`../../ports/ffmpeg/app/results/20261004-qemu-pool-corpus-40-47`):

| case | floor (`poolstock`) | + nested (`poolsublet`) |
|---|---|---|
| 0 `af_join_dedup_bound` | DEFECT-REPRODUCED | **FAULT cause 24**, `case.c:85` |
| 1 `h264_refs_partial_clear` | DEFECT-REPRODUCED | **FAULT cause 24**, `case.c:48` |
| 2 `vidstab_parked_plane_pointer` | DEFECT-REPRODUCED | **FAULT cause 24**, `case.c:33` |
| 3 `vvc_nonref_output_releases_tabs` | DEFECT-REPRODUCED | **FAULT cause 24**, `case.c:104` |

**+4 attributable to the nested allocator**, with the floor already carrying a revoking system
allocator. Each fault reports a non-zero `value_hi`, so it is a capability that lost its
authority and not an integer used as an address. `cheribsd-revocation` is measured and silent
for these, with the reason recorded: the storage never reached `malloc`, so it never entered
the quarantine the revoker sweeps — the same shape as the Perl corpus's `17535c984a`.

### `subobject-repros` — 3 spatial cases: **nothing** catches them, and that is the row's purpose

The crossing is between two members of **one** allocation, so every per-allocation bound is in
bounds for it: `shrink`, the floor, the nested arm and CHERI alike. The authority that would
have to be narrowed is per struct member, which is the compiler or the allocation site, not an
allocator.

`cheribsd-revocation` was a **prediction** until 2026-10-06 and is now measured
(`subobject-repros/results/20261006-cheribsd-subobject`): with revocation off the control holds
and cases 01 and 02 reproduce their defect on purecap, so CHERI does not catch them. Case 00
does not reproduce on this ABI and is outside the arm's denominator. The revocation-on cells are
not scoreable — the fixture's own control arm faults, which an attribution probe pins on the
fixture rather than the platform.

### `plane-repros` — 1 spatial case

Measured natively (`plane-repros/results/20261006-native-plane`). Its purecap cell is **not**
measured: the fixture needs `av_frame_alloc`, so it does not link against the buffer-pool port's
purecap library the way the subobject fixtures do, and it would need more of libavutil built for
the target.

## The two cells this design still wants

1. **`sysalloc-bounds` and the unprotected base, app-level, for the pool cases.** The floor and
   the nested arm are measured app-level; the bounds and unprotected arms for these four cases
   exist only as the buffer-pool port's fixture probe cases, which run that port's *substitute*
   allocator rather than FFmpeg's own `buffer.c`. Monotonicity would suggest both are silent — the
   floor has strictly more protection and is already silent — but that is a derivation, not a
   measurement, and layout changes between heaps can move a verdict.
2. **`plane-repros` on purecap**, which needs libavutil for the target.

Neither gap weakens the `+4`: that number is a floor-versus-nested difference, and both of those
arms are measured.

## A caveat that crosses bundles

The pool-corpus bundle was built with the toolchain `7d01722aab88`, the only one on this host that
passes the application SDK's ABI gate; every other measured FFmpeg arm used `b7b31421e9fa`. A cell
from one bundle is therefore not built by the same compiler as a cell from another, so the ladder
above should be read per bundle, not as five cells of one run.
