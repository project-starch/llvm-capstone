# Patch 0003 without the C-50 workaround, on dev's compiler — the run (2026-09-28)

**Every pre-registered prediction held.** Predictions were written and pushed in `d92a360e5da0`
**before** this build started; nothing below was decided after the fact.

| | predicted | got |
|---|---|---|
| **P1** the C-50 gate | 0 hits | **0 hits in 361,966 instructions** |
| **P2** stages | M1–M5 REACHED | all REACHED, plus the flip control |
| **P3** the run, the deciding cell | 30 frames, 0 hash mismatches, control FIRES | **30 frames, 0 mismatches → MATCH**; control **FIRES** on 10 hashes |
| **P4** new alignment warnings | noise, not failure | no gate treated anything as a failure; 21 pointer round-trip sites flagged |

## What this establishes, at exactly the width agreed in advance

**Dropping patch 0003 is supported by a RUN, for this workload and the translation units it
executes.** That is stronger than the compiler lane's Arm B, which recompiled one TU and read the
scanner: here the image was built, booted, and its decode compared against the native reference,
with the flipped-input control firing in the same boot to show the comparison can report a
difference.

It is **not** "0003 is unnecessary in general". Nothing here exercises code paths this build does
not reach, and `scan-addi-sp.py` has documented blind spots — which is precisely why the run, and
not the scan, is what this rests on.

## The four arms, and why this is one variable

| compiler | patch 0003 | gate | run |
|---|---|---|---|
| `3979abd8` pre-fix | present | 0 hits | boots, decodes, MATCH (2026-09-25) |
| `3979abd8` pre-fix | **absent** | **1 hit** — `addi a2, sp, 0x8` → `sd zero, 0(a2)` | not run; the build gate stops it |
| post-fix, one TU (compiler lane) | absent | 0 hits | not run |
| **`0a458bdb` post-fix** | **absent** | **0 hits** | **boots, decodes, MATCH** |

The bottom two rows differ from the top two by the compiler only. The delta from `3979abd8` is the
C-50 fix plus `#93`'s `-Wcapstone-capability-alignment`, a Sema diagnostic that does not change
codegen.

**The post-fix compiler was verified by content, not by ancestry**, and that distinction mattered:
`0a458bdb` sits on the fix's own branch and does **not** descend from the dev-landed `4c407f9`, so
`git merge-base --is-ancestor` reports it as pre-fix and is wrong. What is true of it: it carries
`428b1aa2ce86`; `git diff 4c407f9456d4 0a458bdb802c -- llvm/lib/Target/Capstone/CapstoneISelLowering.cpp`
is empty; and `git diff 0a458bdb802c origin/dev -- llvm clang lld compiler-rt` is empty, so its
compiler sources **are** dev's.

## A second finding, not the one that was asked for

This is also the first evidence that **the FFmpeg port builds, boots and decodes correctly on dev's
compiler** — every committed FFmpeg result until now was built with `3979abd8`. It is established
for the without-0003 configuration only; with-0003 on dev's compiler has not been run, and needs no
run unless someone wants that specific combination.

## Why patch 0003 is NOT being dropped in this commit

Dropping it would make the port **require** a compiler at or past `4c407f9`. The shared main-clone
build on this host is still `b7b31421`, which predates it, so a drop today would hand anyone
building there an image carrying the C-50 fault. The sequencing therefore matters: **0003 comes out
together with the port's move to dev's compiler, as one deliberate change, not before it.**

The gate protects that ordering rather than relying on anyone remembering it: on a pre-fix compiler
without 0003 the C-50 scan **fires and fails the build** (row two above, measured), so a premature
drop cannot ship silently.

## What this does not establish

1. **QEMU only** (**Q-11**, **M6** unchanged).
2. **N = 1** — one build, one boot, six sections.
3. **One workload**, the 320x180 mpeg4 clip; one arm, level0; the default minimal configure.
4. **`fixbuild` is not mine.** It lives in another session's scratchpad, so this arm is reproducible
   only while that build survives. `SHA256SUMS` records what it was.

## Files

- `PREDICTIONS.md` — pre-registered, pushed before the build (`d92a360e5da0`).
- `result-lines.txt` — the build gates and the runner's verdicts.
- `SHA256SUMS` — images, loader, compiler and emulator identities.
