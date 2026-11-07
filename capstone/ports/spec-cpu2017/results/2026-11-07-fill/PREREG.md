# Three board runs with no build (7 Nov, 16:3x): same-board Plain for gcc, gcc c3f N=2, Plain perlbench on C.5a

Written and pushed before the board calls. All three boards were idle; these runs need no new image or kit. Driver
perf/b3/gt-run-ga2.sh on each board: flash C.5a (sealed sha256), read it back, the board's list, flash bitstream 3
back. Kit bkit27-gtq (c3f's: module 65ede944a87ea742, quiet-grant launcher 1d8b4f0829bef0e7).

| board | stage | why |
|---|---|---|
| 3 | gcc train Plain (`perf/gt/stages/502.gcc_r-train01-plain-rv64`) | c3f's 1.210× divides board 3's 4,253.3 s by board 1's Plain 3,515.5 s; this is the same-board denominator |
| 2 | gcc train c3f again (`perf/ga2/stages/...-devmk-c2ac7`, the stage that read 4,253.3 s) | N = 2 for the headline gcc number |
| 1 | perlbench test Plain (`spec/stages-test-plain/500.perlbench_r-test-plain-rv64`) | perlbench's 23,311 s Plain is the 3 Nov reading on the old bitstream and kit |

| # | Prediction |
|---|---|
| F0 | each run rc 0, control passes; Plain gcc specdiff 1/1, Plain perlbench specdiff 2/2; c3f specdiff 1/1 |
| F1 | Plain gcc on board 3 within 3,515.5 s × [1.000, 1.012] (board 3 read Plain mcf 0.6 % slower than board 2). Same-board ratio for c3f = 4,253.3 / that |
| F2 | gcc c3f on board 2 within ±1.5 % of 4,253.3 s (4,189-4,317 s) |
| F3 | Plain perlbench within ±2 % of 23,311 s |

**Limits.** N = 1 per arm; board-to-board spread is the open variable, which is why F1 and F2 are read together.

## Result, the two gcc runs (7 Nov 17:52)

| run | board | time | reading |
|---|---|---|---|
| Plain gcc train | 3 | 3,510.0 s (rc 0, specdiff 1/1) | 0.9984× board 1's 3,515.5 s |
| gcc train c3f, N = 2 | 2 | 4,255.8 s (inside 3,515.4 s, outside 740.4 s; rc 0, specdiff 1/1) | 1.0006× board 3's 4,253.3 s |

- **F1 FAILS by 0.16 %:** board 3 reads Plain gcc 0.16 % faster than board 1, below the band's lower edge (it assumed
  board 3 is the slower board, from Plain mcf). The board-to-board spread on Plain gcc is 0.16 %.
- **F2 holds:** c3f reproduces within 0.06 % on another board.
- **gcc train c3f is 1.212× Plain on the same board** (4,253.3 / 3,510.0), and 1.211× across boards 2 and 1. The 1.20×
  bound on board 3 is 4,212.0 s: 41 s short.
- F3 (Plain perlbench, board 1) is still running.
