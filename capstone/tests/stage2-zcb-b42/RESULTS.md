# Zcb with a capability base on b42c8c644, LSHOFFL against ZCBB in six alternating boots: results (2026-11-26)

Pre-registration: PREREG.md, 92c7299ea337, and Amendment 1, 43a0ca92798e. The amendment was pushed at 16:44:15Z,
while ZCB-ON1's JTAG load was still running and before any ON reading existed.

**The ZCBB image FAILS the one-sided gate on perlbench and nab, so ZCB does not enter the campaign.**
- perlbench: +1.15 s (+0.41 %), against an allowance of 0.72 s.
- nab: +1.00 s (+0.14 %), against an allowance of 0.46 s.
- gcc and imagick hold. No stage reads a gain.
- The drift control holds. Every stage passes specdiff in all six boots, and every output hash field is equal across
  them, so neither failing stage is UNDECIDED.

**Attribution (Amendment 1):** the failure is the image's as a whole, not Zcb's own.
- ZCBB moves 4,713 of perlbench's 4,761 functions and resizes 720.
- LSRH's image, with a twin difference of -34 instructions, read +1.27 % on the same stage.
- Whether this reading exceeds code placement's own spread is for tests/stage2-placement-b42 to read. Its use is
  fixed there in advance: the effect is attributed to Zcb only if |mean(ON) - mean(OFF)| > P_s + range(OFF) + 0.02 s.
- "Does not enter" is this pre-registration's verdict and stands either way.

## The boots

Board 2, b42c8c644, read back before and after every boot. There was no reflash and no other boot inside the six. The
board lane ran the boots. This lane read every number below from the uart captures, after checking each capture's
sha256.

| boot | image | payload sha256 | uart sha256 | RUN END |
|---|---|---|---|---|
| OFF1 | S-LSHOFFL | 21d18193b86359a8 | 3261f3f2664843b0 | rc=0 16:41:48Z |
| ON1 | S-ZCBB | bf5c9db1c13256b7 | 531098b9d765681b | rc=0 17:09:34Z |
| OFF2 | S-LSHOFFL | 21d18193b86359a8 | 1e67d2f255b532cb | rc=0 17:37:21Z |
| ON2 | S-ZCBB | bf5c9db1c13256b7 | ff203f68e338d4dd | rc=0 18:05:11Z |
| OFF3 | S-LSHOFFL | 21d18193b86359a8 | 1ec164cf399754e9 | rc=0 18:33:03Z |
| ON3 | S-ZCBB | bf5c9db1c13256b7 | 0b9c986b24514bd4 | rc=0 19:00:49Z |

- **Payloads:** each equals the pre-registration's sha. smoke.dom read rc=7, as predicted, in every boot.
- **Preflight:** each boot's run log carries "exit 1 BLOCKED ... by C10's 20 MB working budget only ... proceeded by
  runtime's ruling; size proven on this seal by the S bracket's first boot (S-LSHOFF, fb1c99ffd26a26c9)".
- **Graders:** v2 passes all four stages in every boot, and v1 agrees.
- **Outputs:** the six boots' result.json hash fields were compared field by field, 575 in all: perlbench 557, gcc 5,
  imagick 8, nab 5. None differs.

## Per stage

Times are up1 - up0 on the stage's @@SPEC-END run line, in seconds. The allowance is
0.0005 × mean(OFF) + range(OFF) + 0.02 s.

| stage | OFF1, OFF2, OFF3 | ON1, ON2, ON3 | mean(ON) - mean(OFF) | range(OFF) | allowance | gate | gain |
|---|---|---|---:|---:|---:|---|---|
| gcc | 5.56, 5.56, 5.56 | 5.55, 5.55, 5.54 | -0.013 (-0.24 %) | 0.00 | 0.023 | holds | no |
| imagick | 11.23, 11.23, 11.23 | 11.21, 11.23, 11.22 | -0.010 (-0.09 %) | 0.00 | 0.026 | holds | no |
| perlbench | 278.72, 278.26, 278.16 | 278.94, 280.53, 279.13 | **+1.153 (+0.41 %)** | 0.56 | 0.719 | **FAILS** | no |
| nab | 727.49, 727.47, 727.55 | 728.79, 728.22, 728.49 | **+0.997 (+0.14 %)** | 0.08 | 0.464 | **FAILS** | no |

- **The drift control holds.** imagick's val reads 5.44, 5.43, 5.42 (OFF) and 5.44, 5.43, 5.43 (ON). |Δmean| is
  0.003 s, against a limit of 0.040 s.
- **The gain rule, max(ON) < min(OFF) - 0.02 s:**
  - gcc's max(ON) is 5.55 against 5.54 = 5.56 - 0.02. That is not below it, so it is not a gain, though all three ON
    readings lie below all three OFF readings, by one or two ticks.
  - imagick's ON readings overlap its OFF readings.
- **perlbench and nab:** every ON reading lies above every OFF reading on both. perlbench's ON readings spread over
  1.59 s, against OFF's 0.56 s.

## Predictions

- **perlbench, "gain or flat, at most -0.33 %": REFUTED.** It read +0.41 %.
- **nab, "flat, at most -0.08 %": REFUTED.** It read +0.14 %.
- **gcc, flat: HOLDS,** at -0.013 s, below the gain rule.
- **imagick, flat: HOLDS.**
- **"No stage is predicted to fail the gate": REFUTED,** on perlbench and nab.

## What it decides

- Per "What decides", **ZCB does not enter.** The failing stages are perlbench (+0.41 %) and nab (+0.14 %).
- Under Amendment 1, the reading is the image's as a whole. The placement control decides whether it is
  distinguishable from code placement.
- **For stage 2:** ZCB's board form gives no gain at the image level. The step-4 twin fetch reading for ZCBB
  (tests/stage2-lineb-step4) is still to be graded, and this result enters stage 2's cycle-band pre-registration
  beside it.

## What this does not show

- **Why the ZCBB image is slower.** Placement is a candidate, not a finding.
- x264, xz, mcf, lbm. P1.
