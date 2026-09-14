# Proposal: bring the paper's evaluation up to the measurements doc (2026-09-14) — NO EDIT MADE; one decision for the project lead

The paper is not edited by this lane (ask-first rule; Overleaf owns the remote). This is the
proposal that rule asks for, built from a `paper-numbers-checker` audit of the LIVE section
(`capstone/paper/parts/evaluation.tex`; `old-parts/` and `proposals/` excluded) against
`docs/ref/fpga-silicon-measurements-for-paper.md` and `docs/ref/ISSUES.md`, with every finding
below re-read at its cited line by this lane before it was written here. The two unmerged remote
branches of the paper repo are older than `main` and leave `evaluation.tex` byte-identical, so
nothing here depends on them.

## The decision: one question, three parts

**(a) Put the measured whole-SQLite overhead beside the estimate.** The paper's only SQLite
performance claim is an estimate — borrows × 171 cycles over instructions × CPI, "~1%" for the whole
benchmark and "≤6%" for the boundary-densest phase (`evaluation.tex:649-650`). The quantity that
estimate approximates is now measured directly on silicon and absent from the paper:

| measured whole-benchmark ratio | value | source |
|---|---|---|
| `main --size 20`, silicon, boot sw68 (2026-09-13) | **1.1940** (64,732,455,367 / 54,214,856,567 cycles) | measurements §7o |
| `main --size 1`, silicon pair, post-flash bridge (sw63) | 1.2195 | §7m |
| seven testsets, silicon (`main` 1.2195, `star` 1.2514, `parsenumber` 1.2677, `orm` 1.1843, `fp` 1.2435, `cte` 1.1654, `rtree` 1.1879) | 1.17–1.27 | §7k/§7m |
| allocator matrix, QEMU `-icount`, memsys5 | 1.2703 | §4g |
| `main --size 100`, silicon, boot sw73 (2026-09-14; pre-registered ≈1.18, band 1.14–1.24) | **1.1819** (337,235,381,252 → 398,572,346,349 cycles) | §7p |
| `main --size 20`, silicon, **lookaside pool ON on both arms**, boot sw77 (2026-09-14; pre-registered 1.19) | **1.1937** (53,142,976,993 → 63,437,512,052 cycles); the OFF pair of the same size and image family 1.1940 | §7q |

These are not the same quantity as the estimate: the estimate prices the boundary borrows only; the
ratio is the whole pure-capability domain's cost, spatial safety and ABI included. The proposal is
to say exactly that, with the size-100 row (the default size, 1.18) as the headline and the size-20
and size-1 rows beside it — not to
present 1.19 as "the real answer where 1% was estimated". Framing the doc leaves to the paper:
user-work-vs-user-work denominators (`main` 1.294 tick-adjusted) versus whole-machine (§7 preamble).

**(b) Replace the CPI = 1 sentence.** `evaluation.tex:658-661` keeps CPI = 1 in the estimate
"because the measured CPI comes from those kernels rather than from SQLite itself" and states the
kernel range as "1.20 on a tight register loop to 6.44 on deep recursion — never near 1". Both
halves are stale: SQLite's own native CPI is measured on silicon (`main` 3.769, §7 `:962`; baseline
3.76 / domain 3.81, `:1158`; sw68's size-20 pair native 3.831 / domain 3.646), and the kernel floor
is 1.13 (`beebs_aha_mont64`, 290,071 / 256,699 cycles per instruction, `:187-188`), which the
measurements doc's glossary corrected on 2026-09-10 (`:46-53`) in exactly the words the paper still
uses. Proposed: quote the SQLite CPI in the SQLite estimate, and quote the kernel row you mean rather
than a range.

**(c) Carry the two caveats the measurements doc binds to every silicon number.** Neither is in
the live paper (the four "timing" mentions in `evaluation.tex` are about cycle-accuracy, not closure):
* the bitstream does not meet timing — WNS −12.425 ns, 102,508 of 174,960 endpoints failing, on every
  §7 row including sw68's (`:2489-2522`); the doc's rule is to prefer instruction counts where a
  claim can be carried by either (`:2589`);
* every §7 speedtest1 row up to sw73 was measured with SQLite's lookaside pool OFF, a configuration
  SQLite does not ship (`:2593-2632`; both arms, so the ratios stand; sw68's image is built by the same
  path with no lookaside override, so OFF by construction — the transcript carries no lookaside line to
  read it from). **Measured since (boot sw77, 2026-09-14, §7q): the same `main --size 20` pair with the
  pool ON on both arms reads 1.1937 against the OFF pair's 1.1940; the pool takes 1.98 % off the native
  arm's cycles and 2.00 % off the domain's, and the native arm's `--stats` reports 25,122 lookasides on
  silicon.** The caveat is therefore a measured non-effect on the ratio, which is the stronger sentence:
  state the OFF configuration of the rows the paper quotes and cite the ON pair as showing the ratio does
  not depend on it; and every §7f–§7k row is `--size 1` against speedtest1's default of 100 (`:2635`) — sw68 is
  size 20 and sw73 is size 100 (speedtest1's default), at 1.18.

## Corrections the audit found in the live text (each re-read at its line; no edit made)

1. **Revocation-node pool: 1024 in the paper, 65,536 on the RTL.** `evaluation.tex:253` ("a fixed
   1024-entry bump allocator") and `:261-263` ("a hardware array of 1024 nodes (16 KiB); our
   measurements peak at 144 live nodes (~14% of capacity)") against ISSUES R-12 (`:3126`, `:3170-3174`:
   `capstone_rev_node.anvil:74` gates on a 16-bit head, `ariane_pkg.sv:587` "65536 nodes * 16
   bytes/node"). On the real pool 144 live nodes is ~0.22% of capacity; the safety argument's
   direction holds, every number in the sentence is wrong.
2. **"On three of seven kernels, cycles grow faster than instruction count"** (`:548-549`). The
   doc's own decomposition (`:264-271`) has a CPI ratio above 1 on six of seven (`prime` 1.052,
   `rv8_primes` 1.118, `cnt` 1.026, `bs` 1.452, `recursion` 1.342, `aha_mont64` 1.023; only `cover`
   below 1). If "three" means the kernels the doc marks as stall-dominated or balanced (`bs`,
   `recursion`, `rv8_primes`), the sentence should say so; read literally it is wrong.
3. **CPI floor "1.20 … never near 1"** (`:658-659`) — see (b): the floor is 1.13.
4. **2,850 vs 2,863 for the same quantity.** `:626` says "once per ≈2,850 instructions of in-engine
   work"; the table at `:650` says 2,863 for the same in-domain scan. Neither rounds to the other;
   neither appears in the measurements doc (the doc says the borrow and instruction counts behind
   these rows "live in the paper draft", `:539-540`).
5. **The 2,863 density is from the non-silicon ABI build.** The doc's silicon-ABI measurement of the
   same workload is `rows=200 borrows=400 scan_instrs=790,003` → 1,975 instructions per borrow
   (`:585`), and it says explicitly not to present 1,975 as a correction of 2,863 — different
   configurations (`:594-599`). The paper's row pairs the silicon per-borrow cost (171 cycles) with
   the non-silicon density; at 1,975 the "≤6%" bound would read ~8.7% at CPI = 1 (and ~2.3% at the
   measured SQLite CPI). A framing question, not a wrong number.
6. **The compatibility sentence carries no vehicle** (`:672-673`: SQLite "alongside CoreMark, the RV8
   suite, and 82 BEEBS kernels"), placed after silicon-only results. The doc's own note on this
   sentence says QEMU-backed (`:834-842`); since then SQLite's SLT correctness (10,807 records, seven
   files, identical to native) HAS run on silicon (§7c, `:1769-1774`), but CoreMark and the 82 BEEBS
   kernels have not (eight rungs on silicon; `coremark_matrix` hangs there, `:242-243`). Proposed:
   split the sentence by vehicle.

Also noted, not proposed for change: the paper's `tab:perfcompare` numbers (3,760 / 23,977 / 14.0 M;
7 / 65) and its "16 times … all 16 agree" (`:577-578`; the doc records 15/15 "on nearly every rung",
`:139`) are not in either authority doc — unsourced here, not asserted wrong.

Verified as matching, for the record: `tab:primcost-rtl` (load 2, shrink 1, mrev 50, delin+revoke
121, reclaim 171, borrow ≈173), `tab:borrowcost-rtl` (8 / 182 / 902 / 3,611; borrow(N) ≈ 75 + 3N/2),
all eight rows of `tab:spatialcost`, the `cnt`/`bs` decompositions, `beebs_prime` 1.054 vs 1.683,
the 0–96 % spread, and the `beebs_prime` −O0 correction already in the paper (`:497,552`).

## If the lead says yes

The edit is small and local to `parts/evaluation.tex`; this lane makes it only on an explicit
go-ahead naming which of (a)/(b)/(c) and which corrections, never pushes `capstone/paper`, and leaves
the parent's submodule pointer unbumped. The measurements doc already carries every number above.
