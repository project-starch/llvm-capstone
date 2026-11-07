# Plain without compressed instructions: is today's Plain baseline at its best? (pre-registration)

Written and pushed before the board calls (7 Nov, 20:0x). x264 reads 0.90× Plain (Capstone faster). On QEMU both arms
execute the same user code (Plain 511.4 G, Capstone 512.5 G instructions), but on the board Plain runs at IPC ≈ 0.45
against Capstone's 0.53 in-context. The prime suspect is RVC: Plain is built `-march=rv64gc`, about half its
instructions 2-byte (gcc 983,086 of 1,877,437; x264 62,504 of 131,106); Capstone has no compressed encodings. If RVC
slows Plain on this core, the fair Plain (Plain at its best) is the one without it.

**Images** (label plain-rv64nc): the same tree-plain, config and flags, `-march=rv64imafd` and perf/T0b's rv64imafd musl
(plain-cc-nc); 0 two-byte instructions in gcc, x264 and perlbench (the counter reads ~50 % on the rv64gc builds).
Stages are copies of the Plain stages with only the binaries replaced (perlbench `--strip-all`, as its stage).
QEMU: gcc train rc 0, specdiff 1/1 PASS, 11.88 G hart instructions (rv64gc reuse 11.31 G); x264 test rc 0, every
output hash equal to Plain rv64gc's except `log.dec`, the input-generation decoder's log (not a validated output; it
also differs between QEMU and board for the same binary).

**Runs, board 1** (perf/b3/gt-run-ga2.sh 1, C.5a), after the Plain perlbench run there ends with DONE; one list:
1. gcc train plain-rv64nc;
2. gcc train plain-rv64r, 3. perlbench test plain-rv64r (results/2026-11-07-plainreuse, moved behind 1);
4. x264 test plain-rv64nc.

| # | Prediction |
|---|---|
| C0 | rc 0; gcc specdiff 1/1; x264 outputs equal the rv64gc Plain's (except log.dec); control passes |
| C1 | gcc nc against Plain gcc rv64gc (3,510.0 s board 3, 3,515.5 s board 1): change within −8 %…+3 % (more instructions, no fetch splits) |
| C2 | x264 nc against 50,320.6 s: change within −20 %…+2 %. A drop below 45,265 s (Capstone c3f) restores Capstone ≥ 1.0× |

**What it decides.** If either nc run is faster than its rv64gc Plain, the nc Plain becomes the denominator for every
benchmark (rebuilt the same way), and the ratios are restated.

## Result, gcc (8 Nov 00:2x): Plain without RVC is 6.0 % SLOWER; C1 FAILS on the upper side

| Plain gcc train | time | machine-wide IPC | per 1k instructions: L1I miss / mispredict / scoreboard full |
|---|---|---|---|
| rv64gc (board 3, fill) | 3,510.0 s | 0.162 | 52.48 / 36.69 / 366.71 |
| rv64imafd (board 1) | 3,725.4 s (rc 0, specdiff 1/1) | 0.154 | 67.86 / 42.14 / 316.70 |

- Compressed instructions make Plain gcc FASTER on this core: denser code, 23 % fewer L1I misses. Plain rv64gc is
  Plain at its best, so today's gcc denominator is the fair one. C1's band (−8 %…+3 %) assumed RVC could only cost.
- For x264 the RVC hypothesis is weakened, not settled: its hot loops may fit the I-cache either way. Its nc run
  stays in board 1's list.
- The other reading: Capstone has no compressed encodings, and on gcc that alone is worth about 6 % to Plain;
  compressed capability encodings (plan direction 3) would recover part of it.
