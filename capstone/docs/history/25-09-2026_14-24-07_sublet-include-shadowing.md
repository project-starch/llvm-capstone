# The Sublet SQLite port stopped compiling on 2026-09-18, and nothing noticed for a week

Found 2026-09-25, while trying to validate a change (D-prime) against the Sublet patch.

## What was broken

The PROTECTED arm of the SQLite allocator measurements could not be built from dev, by either
script that claims to build it, and had not been buildable since 2026-09-18 13:10.

`build-sqlite-capstone.sh`, run unmodified, exactly as `sublet/README.md` documents:

    OUT_DIR=<tmp> SQLITE_SUBLET_PATCH=capstone/ports/sqlite/sublet/sublet-3530300.patch \
      bash capstone/ports/sqlite/build-sqlite-capstone.sh
    -> rc=1
    sqlite3-capstone.c:188237:3: error: use of undeclared identifier 'capstone_cap_slot'
    (+19 cascading errors)

`build-sqlite-silicon.sh`, the script that builds board domains, was broken too, with a DIFFERENT
error. Its `COMMON` carries `SUBLET_INC` plus six roots and NEITHER `capstone` NOR
`capstone/runtime/include` is among them, so on a one-line TU containing only the patch's own
include:

    fatal error: 'sublet/sublet.h' file not found

Two scripts, two different failures: these are two instances, not one claim generalised from the
other.

## Root cause: two headers answer to one include name

    capstone/sublet/sublet.h                  struct sublet_cap        13 fns, raw .insn asm
    capstone/runtime/include/sublet/sublet.h  capstone_cap_slot         7 fns, calls capstone_cap_*

The patch adds exactly one include, `#include <sublet/sublet.h>`, and uses `capstone_cap_slot`
nine times. `capstone/sublet/sublet.h` has NEVER contained that identifier -- `git log -S` over its
entire history returns nothing. In `build-sqlite-capstone.sh` the roots were ordered

    -I capstone/sublet   -I capstone   -I capstone/runtime/include   -I <patchdir>

and `-I capstone` resolves `<sublet/sublet.h>` to `capstone/sublet/sublet.h` first. `clang -H`
prints exactly one line for that include: `. capstone/sublet/sublet.h`.

## It was a merge resolution, which is why the fix is a restore

    4fd699dd3525  patch uses sublet_cap                                    builds (shell header)
    e973ec5996c6  patch uses sublet_cap                                    builds
    dac22bcaeca4  patch rewritten to capstone_cap_slot; runtime header added;
                  SUBLET_FLAGS is exactly ONE root, -I capstone/runtime/include   COHERENT
    06a31271f200  PR #48 merge: adds -I capstone/sublet and -I capstone AHEAD of it,
                  and drops runtime/include/sublet/sublet.h from the tree entirely
    51429e856e59  "Restore the shared Sublet header..." re-adds the FILE, not the ORDER
    HEAD          file present, order still shadows it                     DOES NOT BUILD

`dac22bcaeca4` deliberately paired the rewritten patch with the runtime header and put only that
root on the path. The merge added two roots in front of it and lost the file; the restore five days
later brought back the file but not the order, and was never validated by running the build.

## Why a week passed

Three separate things each made the failure invisible:

1. **The README asserted the opposite.** `sublet/README.md:14-17` said the build "adds that root
   too, so an `#include <sublet/sublet.h>` and an `#include "sublet.h"` both resolve to the one
   header." There is no one header; the two differ by 202 lines after normalising the type name.
   Anyone checking the include setup against the documentation would conclude it was fine.
2. **Nothing runs the protected arm.** No nightly, no lit test, no gate compiles a Sublet-patched
   amalgamation. The unprotected arm builds and runs constantly, so dev looked healthy.
3. **The last measurements predate the break.** Arm C (image 2b9e4d0d3ed523c9, booted 09-15 02:46)
   and E2 cell 6 (ceeded25, built 09-14) both used the SHELL header with the pre-rewrite patch. So
   the post-`dac22bcaeca4` patch has been measured under NEITHER header, and the numbers on record
   are not reproducible from today's tree even after the include order is fixed.

## The fix

Put `-I capstone/runtime/include` ahead of `-I capstone` in `build-sqlite-capstone.sh`, and add it
to `build-sqlite-silicon.sh`'s `SUBLET_INC`. Verified by recompiling the real build's own
`sqlite3-capstone.c` at -O2 with the reordering: rc=0, a 1,836,816-byte object. Positive control for
both scripts: the same one-line probe compiles once that root is present and fails without it.

The README's claim is corrected in the same change.

## What this cost, and the general lesson

A week in which the protected arm of the paper's allocator comparison could not be built, discovered
only because something else needed it. The specific trap is worth naming: **a header restored by
path is not a header restored to the include PATH.** `51429e856e59` checked that the file existed
and that configure-time detection saw it; neither is the question the compiler asks, which is which
of two same-named headers a given `-I` order reaches first.

The cheap check that would have caught it at any point in the week is the one the project already
requires everywhere else: build the thing the change is supposed to fix. A file-presence assertion
is not a build.

## A second trap, met while validating this: worktree submodules

The gates here resolve tool paths under `$CAPSTONE_REPO_ROOT`. In a `git worktree`, submodules are
NOT checked out, so

    CAPSTONE_BUILDROOT_DIR -> <worktree>/capstone/caplifive-buildroot   (empty)
    CAPSTONE_QEMU_BINARY   -> <worktree>/capstone/capstone-qemu/build/  (empty)

and a run fails on infrastructure rather than on its subject. It cost two runs here, the first
AFTER a complete successful build, and the board lane reports the same on its own worktree gates.
Both are overridable; point them at the main checkout.

Worth recording because the failure is loud in one case and quiet in the other. `domdata-budget`
printed "Refusing to guess the allocation ceiling" and exited non-zero, which made the cause
obvious in one read -- a tool that had defaulted to some plausible ceiling instead would have
produced a wrong number that looked like a result. That is the same principle as the include-order
bug above: the value of an assertion is that it refuses rather than guesses.

## How it was validated, two-sided

Neither script's fix is asserted from a single passing run. Each was run on the unfixed tree and on
the restored tree, with the same command and the same toolchain:

    build-sqlite-capstone.sh + SQLITE_SUBLET_PATCH
      unfixed  rc=1  "unknown type name 'capstone_cap_slot'"
      restored rc=0  sqlite_memory_capstone.dom, 3,209,208 bytes

The negative arm is the point: it shows the build was failing for the reason claimed, and that the
include order is what changes the answer.

## The protected arm runs again -- confirmed end to end

`run-sqlite-speedtest1.sh` with `SPEEDTEST1_SUBLET=1` (the real driver: it builds
`speedtest1_domain.c`, which calls `sqlite3_sublet_grant`, through `build-sqlite-capstone.sh`):

    rc=0, "QEMU smoke passed", __CAPSTONE_SPEEDTEST1_DONE__ rc=0
    sublet: split=5481 mrev=37884 delin=32575 revoke=37884 init=5309
    Successful lookasides: 25015, Lookaside Slots Used: 1 (max 154)

The Sublet counters are the proof that the port was ACTIVE rather than merely present: a build with
the patch applied but the allocator unused would report zeros. This is the first successful run of
the post-`dac22bcaeca4` Sublet port.

For reference, the last recorded Sublet numbers (arm C, 2026-09-15) were
`5568/37966/32565/37966/5401`. They are close but NOT equal, as expected: arm C was built from the
pre-rewrite patch against the SHELL header. The two are different programs, so arm C is not a
baseline this build can be compared against.

## One thing this does NOT mean, recorded to prevent a wrong inference

`build-sqlite-silicon.sh` + `SQLITE_SUBLET_PATCH` is NOT how the protected arm is built, and a run
of that combination faults at `capstone_cap_base` with cause 24 even with the include order fixed.
That is not a port defect: `run-sqlite-silicon.sh` builds the smoke domain
(`sqlite_capstone_domain.c`), which never calls `sqlite3_sublet_grant`, so `mem5.aCap[]` slots are
null by construction and the first `sublet_take_linear` does `lcc` on an untagged slot. The include
fix to that script is defensive -- the code path exists and was broken -- not a path any measurement
uses. It was nearly written up as a new runtime defect; what settled it was grepping for the callers
of `sqlite3_sublet_grant` and finding only the two speedtest1 domains.
