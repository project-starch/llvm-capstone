# All ten sub-object defects complete on CheriBSD: 0 of 10 caught, measured with controls

Stock CheriBSD purecap, **libc revocation ON**, `guest_default: preserved`. **22 of 22 arms passed,
runner exit 0.** Every case printed its own `VERDICT`, so each completion is a reading rather than a
silent pass.

| control | reading |
|---|---|
| `cheribsd-abi` | `CHERI_ABI pointer_bytes=16 runtime_revocation=1` — **PASS** |
| `cheribsd-bounds` | `CHERI_BOUNDARY_READY` — **PASS** |

**0 of 10 caught, and that is the result this corpus exists to measure.** Every crossing here is
interior to one allocation, so a per-allocation bound is in bounds for it — CHERI included. Combined
with the sibling `../../plain-heap-repros/results/20261006-cheribsd/`, where 3 of 4 *were* caught,
this is the contrast stated as two measurements on one platform in one day rather than as an
argument.

## The reason is WEAKER than "the bound is the whole allocation", and the arms say so

This corpus's driver hands the port's `av_malloc` **one** arena
(`ports/ffmpeg/buffer-pool/src/shared/metadata-allocator.c:26` bumps a cursor and returns a raw
interior pointer, rounding every request to 64 bytes). So on this harness there is **no
per-allocation bound on the struct at all**, and the completion does not demonstrate that CHERI
bounds the struct and the crossing stays inside it — only that nothing bounded anything at that
granularity. The sub-object *claim* does not rest on the arm: each case asserts its own offsets, so
the crossing is interior by construction.

This is the same over-claim that was corrected on the `wmem` arms, written down here before it could
be made again.

## `-O0` is load-bearing, and the proof is two-sided

A first run of this suite was built `-O1` and **`so-00-buggy` returned `VERDICT INCONCLUSIVE`** while
the other nineteen arms passed and both controls fired. That is not a platform reading. Case 0's
index is a compile-time constant one past its own member, and clang at `-O1` is entitled to fold the
store away; the other cases index through a variable the compiler cannot resolve, which is why only
case 0 was affected.

Rebuilt at `-O0`, nothing else changed, the same binary source on the same platform:

| build | `so-00-buggy` |
|---|---|
| `-O1` | `VERDICT INCONCLUSIVE` — the store was folded away |
| `-O0` | `VERDICT DEFECT-REPRODUCED` |

So the reading flips on the optimisation level alone. Had the `-O1` suite been reported, case 0 would
have entered the inventory as an inconclusive *CheriBSD* row — a compiler artifact recorded as a
platform fact. The native arms use gcc and were unaffected, which is why this surfaced only here.

It is the third instance today of the same class: the ASan probe needed `-O0` for its control to
fire, and the `-O1` build of the sub-object ASan probe had silenced both arms back on 2026-10-05.

## What this does and does not establish

- **Does:** all ten crossings complete under CHERI with revocation on, with the platform's own
  controls firing in the same boot, and with each case printing a verdict that distinguishes
  "completed" from "did not run".
- **Does NOT** establish that a per-allocation bound was in force and the crossing stayed inside it.
  See the arena caveat above.
- **Does not** measure PoisonCap, which is unavailable on this host, nor the Capstone arms for cases
  3-9.
- **N = 1 per cell.**

Files: `result-lines.txt`, `inputs.json`.
