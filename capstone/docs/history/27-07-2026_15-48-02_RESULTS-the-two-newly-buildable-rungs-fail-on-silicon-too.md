# Board: the two newly-buildable rungs FAIL on silicon too — the measured set stays at 3

**Date:** 2026-07-27
**Lane:** C (primary)
**Cost:** one board session, two rungs, two attempts each. Board powered off + unlocked.
**Corrects:** the expectation in
`27-07-2026_12-59-35_three-codegen-fixes-unblock-two-ladder-rungs-and-rv8-at-O1.md`
that these two rungs would take the perf table from 3 to 5.

---

## Result

Both rungs built at −O1, QEMU-correct through the identical controller, transferred cleanly
(sha-verified), and then failed on hardware:

| rung | opt | board | oracle | verdict |
|---|---|---:|---:|---|
| `beebs_crc32` | −O1 | **no result** — no END marker in 120 s, 2 attempts | 1703161001 | **HANGS** |
| `beebs_insertsort` | −O1 | **957879052** (1824 cyc, 560 instret) | 271779359 | **WRONG VALUE** |

Resident bitstream verified as `working-caplifive-captype-fixed.bit` before measuring.

`beebs_insertsort`'s 560 retired instructions are far too few for the sort that produces the
oracle — the computation did not really run. That is the same signature as the other failures
(a bound or index corrupted so the loop exits almost immediately), not an arithmetic slip.

## What this means

**The compiler fixes were necessary but not sufficient.** They removed the *build* blocker; the
underlying **silicon** blocker was sitting behind it and is what actually stops these rungs. Both
were already wrong on silicon at −O0 in the 2026-07-25 sweep (`beebs_crc32` 1568735421 vs
1703161001; `beebs_insertsort` 255001740 vs 271779359), so this is not a new fault and not a
regression from the fixes — it is the same unexplained divergence, now reachable at more
optimisation levels.

**The measured set stays at 3 rungs.** The "3 rungs + §5 caveats" plan is no longer a fallback,
it is the plan. `ref/fpga-silicon-measurements-for-paper.md` is unchanged in its numbers.

**The split is now 3 pass / 4 fail, and the failures are one family:** every rung that is not in
the measured set either hangs or returns a value whose instruction count shows the compute never
ran. `matmult_int`, `coremark_matrix`, `beebs_crc32`, `beebs_insertsort`. No compiler-side
property has yet separated the two groups — instruction mix, code size, global count, `.bss`
size, loop-exit condition, capability round-tripping and narrow accesses have all been tested and
none discriminates.

## Value retained from the compiler work

The three fixes stand on their own merits regardless of this outcome:

- The **CodeGenPrepare negative-offset zero-extension** was generating a *wrong address* on any
  wide-pointer target and was caught only by a backend fatal guard. That is a real defect fixed.
- The **`beebs_crc32` optimizer/large-RO interaction** is a trap that will recur — any hand-rolled
  table meant to dodge the large-RO limit can be constant-folded back into a private constant at
  −O1+. SQLite will hit this.
- The **`i128 = and` fall-through** removes an unlowerable-node bug.

All three are corpus-validated (Capstone lit 41/41, BEEBS 82/82, CoreMark, authority 32/32, full
X86 + RISCV lit with 6 verified-pre-existing failures).

## Recommendation

Do not spend further board time trying to add these two rungs. Five hypotheses have now died
against controls, the remaining board-side question is a shared mechanism nobody has isolated,
and the paper does not depend on it: 3 measured rungs with the §5 caveat list is a complete,
honest result. If board time is spent at all, the highest-information probe left is the
`coremark_matrix` fault-2 bisect (one boot, splits the seeding loop from the rest) — because it
is the only failure localized to a single ~10-line region.
