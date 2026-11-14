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
