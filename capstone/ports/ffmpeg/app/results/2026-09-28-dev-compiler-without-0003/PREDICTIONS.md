# Can patch 0003 be dropped? Pre-registration, written and pushed before the run (2026-09-28)

**Question.** The compiler lane's Arm B recompiled the unpatched `mpegpicture.c` on a post-C-50
compiler and got 0 hits where the pre-fix compiler gets 1. That SUPPORTS dropping patch 0003 and
does not close it: it covers one translation unit, and `scan-addi-sp.py` has blind spots. The
remaining step is to build the image **without 0003 on a post-fix compiler and RUN it**, which is
this port's to do.

## The arms, and why this is one variable

| | compiler | patch 0003 |
|---|---|---|
| already run (2026-09-28) | `3979abd8` pre-fix | absent → gate **1 hit**, `addi a2, sp, 0x8` → `sd zero, 0(a2)` |
| already run (2026-09-25) | `3979abd8` pre-fix | present → gate **0 hits**, boots, decodes, oracle MATCH |
| **this run** | `0a458bdb` post-fix | **absent** |

The post-fix compiler is the compiler lane's `fixbuild`. Verified rather than assumed:
- it contains `428b1aa2ce86` ("C-50: give a byval local copy a capability frame index");
- `git diff 4c407f9456d4 0a458bdb802c -- llvm/lib/Target/Capstone/CapstoneISelLowering.cpp` is
  **empty**, so its codegen for this fix equals the commit that landed on dev;
- `git diff 0a458bdb802c origin/dev -- llvm clang lld compiler-rt` is **empty**, so its compiler
  sources ARE dev's.

The delta from `3979abd8` is the C-50 fix plus `#93`'s `-Wcapstone-capability-alignment`, which is a
Sema diagnostic and does not change codegen. (Note an ancestry test is the wrong question here:
`0a458bdb` is on the fix's own branch and does **not** descend from the dev-landed `4c407f9`, yet
carries identical content. Asking "is it an ancestor" answered PRE-fix and was wrong.)

## Predictions

- **P1 — the gate.** The C-50 scan reports **0 hits**. This is the compiler lane's Arm B result
  reproduced through this port's own build rather than a single recompiled TU.
- **P2 — the stages.** M1–M5 REACHED, on the level0 arm and the default minimal configure.
- **P3 — the run, which is the point.** 30 frames, **0 hash mismatches** against the native
  reference, and the flipped-input control FIRES in the same boot. A scan is not a run: 0003 is
  being dropped on the strength of this cell, so it must be the executed image that agrees.
- **P4 — noise, not failure.** New `-Wcapstone-capability-alignment` warnings may appear from `#93`.
  They are recorded, and are not a failure unless a gate treats them as one.

## What each outcome means, decided in advance

- **P1 and P3 both hold** → dropping 0003 is supported *by a run*, for this workload and the TUs it
  executes. Not "0003 is unnecessary in general": nothing here exercises code paths this build does
  not reach.
- **P1 holds and P3 fails** → the scanner missed something it is documented to be able to miss, and
  **0003 stays**. This is the outcome the blind-spot caveat exists for.
- **P1 fails (any hit)** → the fix does not cover this instance; 0003 stays and the compiler lane
  needs the disassembly.

## Order, and what is deliberately not built yet

Only the WITHOUT-0003 arm runs now. A WITH-0003 arm on the same compiler is the control that would
separate "0003 was needed" from "this compiler broke the port", and it is **only needed if this run
fails** — the port is already known to build, boot and decode correctly with 0003 on `3979abd8`.
Building both costs two full FFmpeg rebuilds, because the toolchain and the patch set each key the
cache. If this run fails, that control is the next step and its result is not predicted here.

**Standing risk, recorded:** `fixbuild` lives in another session's scratchpad and is not mine. If it
disappears mid-run the arm is void rather than negative.
