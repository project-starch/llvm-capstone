# Every optimisation together, on board 1: result

Board 1, 5 Nov 03:18-11:51. Script `all-b1.sh` (pid 4077108); C.5a (caplifive_vp2c_1fc2cd5aa) flashed and read back
by name, bitstream 3 flashed back after. Kit bkit20-all-b1: the compiler evaluation's devmk images, C.5a and the
fault-ahead module (`fault_ahead_max=64`). The grade is `grade.txt` (all-grade.py, rc 0). N = 1 per arm.

## Verdicts

| # | Prediction | Reading | Verdict |
|---|---|---|---|
| AL1 | control, rc, specdiff, twin, fresh, instret within 0.1 % | every arm | holds |
| AL2 | run time below the devmk-on-bitstream-3 run, every arm | every arm; gcc 41 vs 43 s, xz 9,398 vs 11,626 s, mcf 12,073 vs 14,738 s (dev step) | holds |

## Share of the gap to Plain closed

The gap is the baseline run time minus Plain's (baseline 28-29 Oct, bitstream 3; Plain test, out-hfl). Temporal
arm (`capstone-lifetimes`) unless named.

| Stage | Baseline / Plain | All together / Plain | Gap closed | Sum of the single levers (slide 11) |
|---|---|---|---|---|
| gcc | 6.82× | 5.07× | 30 % | 31 % |
| imagick, MapLife | 1.69× | 1.41× | 41 % | |
| imagick, Temporal | 1.67× | 1.27× | 60 % | 40 % |
| nab | 1.36× | 1.19× | 47 % | 46 % |
| lbm | 1.11× | 1.00× | 101 % (MapLife 99 %) | 105 % |
| xz | 1.51× | 1.17× | 68 % | 69 % |
| mcf | 1.78× | 1.33× | 58 % | 42 % (compiler: dev step only) |

- **The levers add on gcc, nab, xz and lbm**: the combined share is within two points of the single levers' sum.
- **mcf closes more than the sum.** Its compiler column is the dev compiler only; the devmk images add the movc flag,
  whose own share on mcf was never measured (its devmk run was stopped). That is the likely difference, not measured.
- **lbm reaches Plain.** 1,893.7 s against Plain's 1,895.2 s. The difference is inside one run's noise; read it as
  parity, not as faster than Plain.
- **imagick's Temporal arm is 2.8 s faster than its MapLife arm in the same boot**, with the same inside time
  (11.99 against 12.04 s) and the same launch (5.36 + 1.30 s against 5.38 + 1.28 s). The difference is outside time
  that `launch_ns` does not hold, the same unexplained Temporal residue the outside-levers RESULT records. The
  MapLife 41 % is the reading to quote.

## Limits

- N = 1, board 1 only, test inputs. These are the evaluation images; the paper stays on the 3 Nov images.
- The baseline ran on 28-29 Oct; later sessions on the same board read 1.2-1.5 % faster (fault-ahead RESULT). On lbm
  that drift alone is 25-30 s, 12-15 % of its 214 s gap.
- Board 1 ran hpm-8 (the L1I counter control) first, on bitstream 3: AS EXPECTED, 22.1 cycles per L1I miss (sim 46.8).

## Next

all2 (results/2026-11-05-all2) adds the five outside-time switches to this kit, against out-all-b1. It started at
11:56 with the gcc canary.
