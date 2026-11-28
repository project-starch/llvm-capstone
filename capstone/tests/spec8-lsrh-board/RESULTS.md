# LSRH on silicon, the S bracket on b42c8c644: results (2026-11-26)

Pre-registration: PREREG.md, d0a80e9e5550, and Amendments 1 to 3 (0fb71e6a53d2, 0eb1ab9eaed5, 6381521d2f1f), all
pushed before the first boot.

**LSRH FAILS the one-sided gate on perlbench and nab, so it does not enter the campaign.**
- perlbench: +3.55 s (+1.27 %), against an allowance of 0.60 s.
- nab: +2.24 s (+0.31 %), against an allowance of 0.43 s.
- gcc and imagick hold.
- The drift control holds, every stage passes specdiff in all three arms, and every output hash is equal across them.
  So neither failing stage is UNDECIDED.
- By Amendment 3, the X bracket does not run, and x264 is recorded as not booted.

The twin gate held on both failing stages (tests/spec8-lsrh): perlbench -34 instructions, and nab -2,984,712
(-0.042 %). **The board time moved the other way, by more than the band, on both.** What makes up the difference is
not identified here (see "What this does not show").

## The boots

Board 2, b42c8c644 (sha256 6a09f34d50842e01), read back before and after every boot. There was no reflash and no
other boot inside the bracket. The board lane ran the boots. Every number below was read by this lane from the uart
captures, whose sha256 it checked.

| boot | payload sha256 | uart sha256 | booked | RUN END |
|---|---|---|---|---|
| S-LSHOFF | fb1c99ffd26a26c9 | edd55ee5596fe5b6 | 14:40:24Z | rc=0 15:08:03Z |
| S-LSRH | 4bc5be53046df016 | bb822394a04566c9 | 15:17:43Z | rc=0 15:45:28Z |
| S-LSHOFF2 | fb1c99ffd26a26c9 | 226b7a1b5c45707b | 15:45:35Z | rc=0 16:13:16Z |

- **Payloads:** each equals its Amendment 1 sha, gated before the boot. smoke.dom read rc=7, as predicted, in all three.
- **Preflight:** its exit line is in each run log.
  - S-LSHOFF's reads "exit 1 BLOCKED ... by C10's 20 MB working budget only: C10 BLOCK at 25856520 B (25.86 MB),
    proceeded by runtime's ruling; size not previously proven on b42c8c644".
  - That boot uploaded, loaded and unpacked, so it is the size's proof (Amendment 2).
  - The other two read "size proven on this seal by the S bracket's first boot".

**An aborted start of S-LSRH, with no board contact.**
- At 15:08:04Z, the second S-LSHOFF unbooked. The board lane's orchestrator started S-LSRH in the same second.
- Its busy check saw S-LSHOFF's own holder process (pid 4186614), not yet exited. It logged "STOP: board 2 busy"
  before any holder, booking, readback or load.
- The orchestrator classified the STOP as infra and stopped the bracket.
- **This is not Amendment 2's stop:** that stop is for an S boot whose upload, JTAG load or unpack fails. No boot
  happened here.
- The bracket was resumed at 15:17:42Z, with an orchestrator that waits for the board to be idle (orch-lsrh2.sh,
  sha256 46c146fa924a7b9e). Its diff against orch-lsrh.sh (c1a8d9737e63251a) was read here. It changes only:
  - the rows;
  - a readback gate on S-LSHOFF's log, in place of the reflash wait;
  - the C10 wording;
  - the idle wait, at most 600 s;
  - an exit-status file.
- Nothing ran on board 2 between S-LSHOFF and S-LSRH, so the three boots stay one bracket.

## Per stage

Run-phase time is up1 - up0 on the stage's @@SPEC-END run line. spread = |LSHOFF2 - LSHOFF|. The allowance is
0.0005 × LSHOFF + spread + 0.02 s.

| stage | LSHOFF | LSRH | LSHOFF2 | LSRH - LSHOFF | spread | allowance | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| gcc | 5.56 | 5.56 | 5.55 | 0.00 | 0.01 | 0.033 | holds |
| imagick | 11.22 | 11.17 | 11.22 | -0.05 (-0.45 %) | 0.00 | 0.026 | holds |
| perlbench (makerand) | 278.73 | 282.28 | 278.29 | **+3.55 (+1.27 %)** | 0.44 | 0.599 | **FAILS** |
| nab | 727.47 | 729.71 | 727.52 | **+2.24 (+0.31 %)** | 0.05 | 0.434 | **FAILS** |

- **The drift control holds.** imagick's val time is 5.42 s in all three boots. Its window is [5.397, 5.443].
- **Outputs:** the three runs' result.json hash fields were compared field by field: perlbench 557, gcc 5, imagick 8,
  nab 5. None differs. specdiff passes 1/1 on every stage of every boot.
- **Graders:** the v2 grader passes all four stages in every boot, and v1 agrees. perlbench's in-band reader removed
  one interleaved line per boot. No "sublet: fault" appears.
- **perlbench's two LSHOFF readings differ by 0.44 s.** LSRH's lies 3.55 s above the first and 3.99 s above the
  second.

## Predictions

- **imagick, the positive control:** a reading between -0.30 and -0.12 s was predicted. **REFUTED in value:** it read
  -0.05 s, a reduction of five ticks, less than predicted.
  - So the instruction-count reduction (-1.77 %) does reach time, but at about a quarter of the CPI-converted size.
  - The pre-registration's harder clause, "no reduction at all", does not apply.
- **gcc and perlbench: "no regression beyond the spread".** gcc holds. **perlbench is REFUTED:** +3.55 s against a
  spread of 0.44 s.
- **nab: -0.042 % of its time** was predicted, about -0.31 s at 727 s. **REFUTED:** it read +2.24 s.
- x264 is not booted.

## What it decides

- By "What decides", **LSRH does not enter the campaign.** The failing stages are perlbench (+1.27 %) and nab
  (+0.31 %).
- By Amendment 3, the X bracket is not booked.
- The twin gate (tests/spec8-lsrh) stands as a twin reading. **It is not sufficient for the board at this band.**
  Both stages it passed, perlbench by 34 instructions and nab by -0.042 %, read slower on silicon by more than the
  board's allowance.

## What this does not show

- **Why LSRH is slower on perlbench and nab.** No mechanism is claimed. One fact bears on it:
  - LSRH changes the size of 18 of perlbench's 4,761 text functions, and of 5 of nab's 839;
  - but it moves 4,717 and 778 of them respectively, since everything after the first changed function shifts.
  - So the two images differ in code placement almost everywhere, which an instruction count cannot see.
  - Whether placement or the changed functions' own cycles carry the 3.55 s is not read.
- **The image-to-image spread of an unchanged program.** The bracket's spread is boot to boot on one image. No pair of
  images differing only in placement has been timed on this board.
- x264, mcf, xz and lbm. P1. Train inputs.
