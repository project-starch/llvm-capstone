# A partial `ninja` target list looks exactly like a regression in whatever you just merged

**2026-09-10, board lane.** Recorded because it nearly produced a false regression report against a
peer lane's merge, and because the gate that exists to catch it **passed**.

## What happened

After merging `compiler-validation-plan` into `dev`, `toolchain-fresh.py` correctly reported the
checkout binary as stale. I rebuilt — but only the four targets I could name:

```
ninja -j90 -C llvm/cmake-build-debug llc clang lld llvm-mc
```

The full Capstone lit suite then failed one test, `MC/Capstone/obj-relocs-cap-constant.ll`, with

```
Machine: 103        (expected a name matching "Capstone"; EM_CAPSTONE is 259)
```

103 is `EM_CR`, a National Semiconductor part — a *different machine entirely*, which is what made it
look like a real object-writer defect in the merged code.

**It was not.** The build is a **shared-library** build. `llc` was fresh; the libraries it loads at
runtime, and the `llvm-readelf` the test pipes into, were still at their 2026-09-04 build. Rebuilding
`llvm-readelf llvm-readobj FileCheck llvm-objdump` made the same test pass and the object report
`Machine: Capstone`. **Full suite: 102/102.** Nothing was wrong with the merge.

## The two lessons, and the second is the expensive one

**1. A PARTIAL rebuild is worse than no rebuild.** No rebuild gives a consistently old toolchain whose
failures are attributable. A partial one gives an inconsistent toolchain, and its failures point at
whatever you most recently changed — which in a merge is someone else's work. Rebuild the whole target
set the suite touches, or do not rebuild at all.

**2. `toolchain-fresh.py` REPORTED FRESH on the inconsistent toolchain.** It checks
`libLLVMCapstoneCodeGen.so`'s build time and the commit `clang` embeds. Both were current. It has no
view of the *other* tools a suite invokes, so **it cannot fire for this failure mode at all** — a gate
that passes on exactly the state it exists to catch. Not changed here: it is a gate, changing one is
the lead's call, and the compiler lane owns that path. Flagged for them.

## What would have caught it sooner

Suspecting the instrument before the subject. The tell was in the number: the object claimed a machine
that is not merely *unnamed* but a **different, specific, unrelated architecture**. A backend that got
its own `e_machine` wrong would emit a wrong Capstone-ish value or zero, not another vendor's. That
should have pointed at a stale reader or a stale constant within a second, and it took several minutes.
