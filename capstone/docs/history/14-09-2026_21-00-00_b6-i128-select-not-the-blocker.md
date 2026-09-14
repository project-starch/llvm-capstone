# B6: the named blocker does not exist — SQLite builds AND runs at -O1/-O2

**Date:** 2026-09-14. **Toolchain:** `libLLVMCapstoneCodeGen.so 73af835a4de22fa0`, clang embeds
`7f466cfe1991`; the freshness gate was green before every artifact below.

## The hand-off

> Compiler lane: B6 — lower the i128 `select_cc` (two i64 halves) so the SQLite domain builds at
> `-O1`/`-O2` (C-17); P1's O2 arms wait on it; the proof is the size-1 QEMU pair at the oracle.

**No lowering was written, and none should be.** The premise was already false on this tree. This
note records the reproduction, because the registry's own C-17 entry says a recorded blocker is a
claim with a date on it and must be re-run before it is built on — which is exactly what went wrong
the last time this entry was acted on.

## Result: builds clean, runs correctly, at both levels

`build-sqlite-silicon.sh` at `-O1` and `-O2`: `BUILD_EXIT=0` each, budget gate included, with
**zero** occurrences of `Cannot materialize`, `Cannot select`, `LLVM ERROR` or `fatal error` in
either log.

speedtest1 `--testset main --size 1 --verify` in the silicon-config capability domain, under QEMU
(`-icount shift=0`), one boot per arm, `SPEEDTEST1_STACK=385024`:

| level | image sha256 | Verification Hash | cycles | vs -O0 |
|---|---|---|---|---|
| -O0 | `6cf8edf637f72063…` | `112006 38bb59fd…3925d8518` | 692,983,497 | 1.000x |
| -O1 | `50ca86aa70b3e425…` | `112006 38bb59fd…3925d8518` | 342,161,746 | 2.025x |
| -O2 | `ec061577fb008e18…` | `112006 38bb59fd…3925d8518` | 336,067,501 | 2.062x |

`.text` for the plain (non-measurement) domain, all three built on this toolchain on this day:
1,311,240 / 1,034,048 / 1,041,144 bytes.

**The control is valid on two independent axes.** The `-O0` arm reproduces the off/off oracle hash
AND `692,983,497` cycles — bit-identical to the archived ④ run recorded at
`docs/ref/fpga-silicon-measurements-for-paper.md:1032`. A harness that reproduces an archived run to
the cycle is measuring what it claims to measure.

Artifacts and run logs: `~/capstone-artifacts/b6-2026-09-14/` with `SHA256SUMS`. Cite by hash.

## Why the named lowering must NOT be written

`lowerCapabilitySelect` bails at `CapstoneISelLowering.cpp:10194` when a constant arm's
`getActiveBits() > XLen`. Its own comment states the invariant: *"Only integer-valued constants
(fitting in XLen, no bounds/perms/tag bits) arise this way."* A constant needing more than 64 bits is
therefore carrying bounds/permission/tag bits. Materialising it "as two i64 halves" would manufacture
a capability out of arbitrary bits — precisely the unforgeability property the ISA exists to enforce.
There is no ALU write to a capability's upper half for that reason; `SCC` asserts a tagged `rs1` and
`CIncOffset` raises on an untagged one.

So B6 as worded is not merely unnecessary, it is a change that must not be made.

## REGISTRY CORRECTION for C-17: the reproducer changed its failure mode

ISSUES.md records the `wide_arm` shape failing as:

    LLVM ERROR: Cannot select: t18: i128 = CapstoneISD::SELECT_CC ... Constant:i128<...>

On this tree it fails with the **forge diagnostic** instead:

    error: Capstone PureCap: Cannot materialize arbitrary >64-bit constants as capabilities;
           capabilities are unforgeable (value 0x10000000000000009)

Mechanism, verified at source: `MVT::i128` is no longer registered as a legal type —
`CapstoneISelLowering.cpp:201` registers `MVT::c128` and there is no `addRegisterClass` for i128 — so
the i128 `SELECT_CC` node can no longer FORM. What remains is the unforgeability guard, firing
correctly. C-17's stakes drop from "backend crash" to "correct refusal". "Reachable from C" stays
UNRESOLVED and is no longer worth spending on.

That diagnostic was also the POSITIVE CONTROL for the clean builds above: it still fires on this
exact toolchain, both under bare `llc` and under the build's own flags
(`-capstone-gp-captable -capstone-shrink-stack=false -capstone-shrink-globals=false`). The clean
builds are therefore a real negative, not a silent instrument.

## The -O0 measurement image sits 552 bytes under the ceiling

Measured from the pass-3 budget blocks (one region: `code_len + 8 KiB + declared dom_data`, order
ceiling 10 = 4,194,304):

| level | code_len | total | margin |
|---|---|---|---|
| -O0 | 1,483,656 | 4,193,752 (`pages=1024`, the exact order-10 max) | **552 bytes** |
| -O1 | 1,223,632 | 3,934,224 | 260,080 bytes |

Code size is not the binding constraint at -O1/-O2. At -O0 it very nearly is: any growth in the
measurement domain fails at domain CREATION, before entry, with no compiler error anywhere. Worth
knowing before anyone adds instrumentation to the -O0 arm.

## Two harness traps hit on the way, both already documented in-tree

Recorded because both produced a confident-looking failure that carried no verdict:

1. **The 1 MiB declared stack.** `speedtest1-geometry.sh` supersedes the run script's default and
   says so: under the one-region rule the static-heap carve plus 1 MiB is order 11 and the domain is
   refused at creation; set `SPEEDTEST1_STACK<=385576`. All three arms failed the budget gate until
   this was set.
2. **Empty worktree submodules.** `caplifive-buildroot` and `capstone-qemu` are empty directories in
   a worktree; `capstone-test-env.sh` derives both the header include path and
   `CAPSTONE_QEMU_BINARY` from the repo root. Fix = symlink the former, export the latter. The
   existing note names BOTH exports; only one was applied, and the second cost a cycle.

**One process slip worth naming:** an interim conclusion ("the opt levels are what makes the image
fit") was drawn from ONE arm of a three-arm run while the other two were still executing. When they
landed, all three had failed the same gate and -O1's code was 0.89x the fitting limit — the opposite
of the inference. One arm cannot separate "this level is special" from "every level fails this way".
The existing rule covers it: conclude when the run concludes.

## What this means for P1

P1's O2 arms were recorded as waiting on B6. They are not blocked: an O2 image exists, is licensed
by a QEMU pass keyed to `ec061577fb008e18…`, and computes the oracle. The remaining P1 dependency is
M1 and the lead's scoping decision, not the compiler.
