# The unified stack without the exit digest: gcc train and perlbench on board 2 (pre-registration)

Written and pushed before the board calls (7 Nov, 18:0x). The unified stack's outside-time regression is the
launcher's SHA-256 of the whole image at every context exit, switched on by the batch harness's
CAPSTONE_EXEC_DIAGNOSTICS=1 (results/2026-11-06-unify, last section). The lead chose this over the second perlbench
THP run on board 2 (results/2026-11-07-thp2: that run is withdrawn before any board call; board 3's stays).

**Stages.** The two unify board stages (perf/unify/bstages: gcc -rgz, perlbench -rg), changed by one line before the
arm: `unset CAPSTONE_EXEC_DIAGNOSTICS; echo "@@NODIAG ${CAPSTONE_EXEC_DIAGNOSTICS-unset}"`. Same images, kit
bkit30-unify, parameters. QEMU (gcc): launcher instructions in the run window 1.352 G → 0.002 G, specdiff PASS,
outputs equal the earlier twin's. A fault in these runs prints no cause line (the switch gates it).

**Runs:** board 2 (perf/unify/un-run.sh 2: flash C.5a, the list, flash bitstream 3 back), after the fill run there
ends with DONE: gcc train -rgz-nd, then perlbench -rg-nd.

**References** (board 3): gcc unify 4,611.0 s (inside 3,432.1, outside 1,179.0, ucyc 19.26 G cycles); c3f 4,253.3 s
(outside 739.5 s). perlbench unify 37,131.8 s (inside 21,901.2, outside 15,230.5); c3g 33,667.7 s (outside 9,799.8 s).

| # | Prediction |
|---|---|
| N0 | rc 0, specdiff PASS (perlbench 2/2, test.out 0 FAILED), control passes, outputs equal the twins'; @@NODIAG reads unset; the .err files hold no `exit image` line |
| N1 | gcc: inside within ±1 % of 3,432.1 s; outside ≤ 900 s; run ≤ 4,253.3 s (c3f); point 4,100-4,250 s (1.20× Plain is 4,218.6 s) |
| N2 | perlbench: inside within ±1.5 % of 21,901.2 s; run ≤ 33,667.7 s (c3g); point 30,500-32,500 s (1.31-1.39× Plain) |

**What would refute what.** N1 holding with N2 failing: perlbench's excess outside time has a second cause beyond the
digest (its exits differ from gcc's: 316 contexts, delegated system calls). N1's outside bound failing: the digest
was not the whole of gcc's 440 s.

## Result, gcc (7 Nov 19:25): 4,518.6 s, N0 holds, N1 FAILS

- **N0 holds:** rc 0, specdiff 1/1 PASS, outputs equal the twin's, control passes; @@NODIAG reads unset; no `exit image` line.
- **N1 FAILS:** inside 3,435.1 s (13.426 G, +0.1 % on unify's 3,432.1 s: holds); outside 1,083.5 s against ≤ 900 s;
  run 4,518.6 s against ≤ 4,253.3 s. 1.287× Plain on board 3's 3,510.0 s.
- Removing the digest saved 92 s of unify's 4,611.0 s. ucyc is unchanged (19.18 G against 19.26 G cycles; c3f 9.84 G),
  so the digest was not the per-step excess (results/2026-11-06-unify, RETRACTION).
- perlbench -rg-nd runs next on board 2. There the digest ran at each of 318 context exits; N2 is still open.
