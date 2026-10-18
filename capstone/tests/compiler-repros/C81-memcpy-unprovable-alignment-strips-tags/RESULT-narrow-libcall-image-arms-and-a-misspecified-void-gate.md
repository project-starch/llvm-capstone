# Result: the image arms run, the prototype converts one site per image — and my void gate is mis-specified

The image arms of `PREREGISTRATION-narrow-libcall.md` finally ran. Two findings: what the prototype
does to a real image, and a defect in the gate the pre-registration said would bind the verdict.

## Getting them to run at all cost two errors, both mine

1. **A missing include path from my own commit.** `99aa4350bcfb` added
   `#include <capstone/representable.h>` to `repro322/repro322_common.h` unconditionally, but
   `build-sqlite-silicon.sh` carries `-I$REPO_ROOT/capstone/runtime/include` only inside
   `SUBLET_INC`, which is empty unless `SQLITE_SUBLET_PATCH` is set. So every non-Sublet build of
   that header failed with `'capstone/representable.h' file not found`. **Lane-only** — not on
   `origin/dev` — so no other lane was affected. Fixed by putting that `-I` on `COMMON`
   unconditionally.
2. **`CAPSTONE_BUILDROOT_DIR` pointed at the repo root** rather than the buildroot checkout, so the
   `domdata-budget` gate could not read `configs/kernel.config` and refused — correctly, rather
   than guessing a ceiling.

Both were invisible in the top-level log because `build-corpus-images.sh` redirects the inner build
to `$OUT/<tag>/build.log`. The top-level log simply ended after the case line with no error, which
I twice misread as the staleness gate. **The error was in a nested log the whole time.**

## What the prototype does to a real image

Both cells, with the scanner's own self-test **passing** (it fires on `copy32_unknown` and
`copy32_unknown_builtin`, stays clean on `copy32_known`, and splits runs correctly):

| cell | plain | narrow | A-only runs | the removed run |
|---|---|---|---|---|
| `backupattach` | `6fe3099ceb322b33` | `5a2bfc0b8f39263f` | 1 | `20B unit=1 (12 insns)` |
| `detachtrig` | `384c363aa4e8e726` | `d02cf7194f876ec4` | 1 | `20B unit=1 (12 insns)` |

So the flag removes **exactly one byte-granular 20-byte expansion per image**, and nothing appears
only in the narrow build. That is the defect's own shape — a tag-stripping unit-1 run — and the
prototype removes it.

## The void gate reads VOID, and the gate is wrong

`PREREGISTRATION-narrow-libcall.md` says the converted-site count must be measured as
**calls-gained** — `mem*` call references the narrow build gains — and that **0 converted sites
makes the image's result VOID**.

Measured: **calls-gained = 0**, checked three ways (symbolised `<memcpy>`, any bare `mem*` token,
and `jal`/`jalr` to a `mem*` symbol): 93 in both builds, for both cells.

But the narrow image is **6 instructions shorter** (236,967 → 236,961; the 24-byte shift shows up
as the `-0x18` offset churn that dominates the 63k-line disassembly diff). So the 12-instruction
byte expansion was replaced by roughly 6 instructions that are **not a call**.

**The gate assumed that declining inline expansion yields a libcall. Here it does not.** So the
gate, applied literally, would void a run in which the prototype demonstrably did what it was
built to do — the scanner shows the conversion and the scanner's self-test passes.

**This is recorded as a pre-registration defect, not reinterpreted into a pass.** The rule against
weakening a gate to make something pass applies to a gate I wrote myself, and the honest statement
is: *the verdict on the image arms is not yet established, because the measure the pre-registration
named is the wrong one.* The correct measure is already available and already self-tested — the
scanner's A-only run count, which is 1 per image and whose instrument has a passing positive
control.

## What is still open

What those ~6 replacement instructions actually are. The disassembly diff is dominated by offset
churn, so the site has to be located by address from the scanner rather than read out of the diff.
Until that is known, "the prototype converts the site without a libcall" is a measurement, and
*why* it costs fewer instructions than the expansion it replaces is not explained. A 20-byte copy
emitted as `ldc`/`stc` plus a 4-byte remainder would fit the instruction count — and would mean the
decline is landing on the capability path rather than on a libcall, which is a better outcome than
the pre-registration anticipated and a different claim from the one it was written to test.
