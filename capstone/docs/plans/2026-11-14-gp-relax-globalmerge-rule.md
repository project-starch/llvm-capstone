# gp relaxation: the GlobalMerge policy is chosen by riscv-base, and applied to both arms (pre-registration)

Written 2026-11-14 13:45, before any riscv-base gp-relaxation result exists. The rule below decides one policy for
both arms and every row. It reads riscv-base figures only.

## What was already known when this was written

This is the Capstone side, and it is stated so that the rule's independence from it can be checked.

Native-twin shift-1 user instructions against the control GPC (llvm-c10 81193d42edb0, flags off, gp SDK). Perl is
makerand, window 0; perl test.pl does not run in these stages.

| Capstone arm | gcc train01 | mcf test | perl makerand |
|---|---|---|---|
| GPR (mark, small-data 16, lld --relax-gp, GlobalMerge as today) | −12.9 M (−0.11 %) | −0.14 M | −391.7 M (−12.7 %) |
| GPR2 (GPR + -global-merge-min-data-size=17) | +82.0 M (+0.67 %) | not yet run | −391.7 M (the same count) |

GlobalMerge is LLVM's IR pass that packs a TU's globals into one container. On the Capstone triple clang emits no
SmallDataLimit module flag, so GlobalMerge merges small globals out of small data unless told otherwise.

## The two policies

- **LLVM default.** Under `-msmall-data-limit=N`, LLVM's GlobalMerge leaves globals of at most N bytes alone (its
  minimum size is N + 1, from the module flag). riscv-base gets it from `-msmall-data-limit=8`. Capstone's equivalent
  is `-global-merge-min-data-size=17` with `-capstone-small-data-limit=16`.
- **Merge unchanged.** GlobalMerge behaves as it does without small data, and small data takes whatever it leaves:
  `-global-merge-min-data-size=0` on both.

Builds: riscv-base RVGP (perf/fair/plain-cc-gp, label plain-rv64gp) and RVGPM (plain-cc-gpm, label plain-rv64gpm),
both on top of riscv-base board 3's build (plain-cc-fairr-round-evict, llvm-c4 codegen) with llvm-c10 lld --relax-gp.

## The rule

On riscv-base, compare RVGPM with RVGP. Use native-twin shift-1 user instructions, with each arm's own validation
complete (N/N): gcc train01 whole run, mcf test whole run, perl makerand window.

- If RVGPM is no worse than RVGP on all three rows, both arms use **merge unchanged**. "No worse" means RVGPM ≤
  RVGP × 1.0005 on each.
- Otherwise both arms use the **LLVM default**.

The chosen policy then applies to all 8 rows on both arms. There is no per-row or per-arm choice. The Capstone
figures above do not enter the decision.

## Outcome (2026-11-14 14:20)

The riscv-base figures: native-twin shift-1 user instructions, each arm's validation complete; RVF3 = board 3's
images.

| row | RVF3 | RVGP (LLVM default) | RVGPM (merge unchanged) | RVGPM / RVGP |
|---|---|---|---|---|
| gcc train01 | 10,860,282,131 | 10,814,311,062 | 10,845,295,645 | 1.00287 |
| perl makerand | 2,758,076,175 | 2,518,504,925 | 2,518,504,925 | 1.00000 |
| mcf test | 27,364,791,454 | (pending) | 27,364,108,101 | (not needed) |

RVGPM exceeds RVGP × 1.0005 on gcc, so by the rule **both arms use the LLVM default**:
- Capstone: GPR2's flags (`-capstone-small-data-limit=16 -global-merge-min-data-size=17`);
- riscv-base: RVGP's (`-msmall-data-limit=8`, whose module flag sets GlobalMerge's minimum).
mcf could not change the outcome.

## Adoption, and the deferred landing gate (2026-11-14 15:30)

Adoption is decided on silicon, as a whole: one policy, both arms, all 8 rows. First the board pairs on perl makerand
and gcc train01:
- riscv-base: RVF3 against RVGP on board 3;
- Capstone: GPC against GPR2 on board 1, on one bitstream.
If those cycles improve the problem-row ratios net, the lever is adopted, and both columns are rerun in full.
Otherwise it is recorded as measured and not adopted.

Gating:
- **Board images.** The native-route directed test is sufficient. SPEC images have exactly one context there: the
  kernel refuses caught handlers and further contexts.
- **Landing runtime-gp-cursor on dev** (local branch runtime-gp-cursor-dev = origin/dev d07a38aedfaf + the runtime
  commits + the pthread-contract rewrite bd6cf22e49b0, not yet run) is **deferred, and is triggered only by
  adoption**. It then requires:
  1. a virtual kit from a fresh clone at dev's pins (capstone-qemu, caplifive-buildroot firmware/kernel/rootfs,
     adapter by runtime/virtual/build-adapter.sh), built the way bug-corpora/juliet/build-kit.sh builds one;
  2. dev's unchanged pthread contract passing on it, as the control;
  3. then the rewritten contract and its four mutants (PC_MUTANT_SAME_IDENTITY, _GP, _HANDLER, _SINGLE), each
     turning off its own check, and the directed gp-cursor test on the virtual route.

Why no existing kit serves:
- spec/kit-prof fails dev's own contract ("capstone-vexec: step: Device or resource busy").
- A kit from QEMU at dev's pin and the 15 Sep build-qemu images dies in OpenSBI. That firmware predates dev's
  caplifive-buildroot pin.
