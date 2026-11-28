# b5b2 board graders: the readers the b5b2 SPEC kits cite

The board lane ran these six scripts to grade every b5b2 SPEC boot on silicon:
- the b5b2-spectest kit (runtime/b5b2-linux, landed on dev);
- the b5b2-campaign (Full CF on b42c8c644, +Spatial CS on 14f2ef81a);
- the compiler lane's stage-2, placement, LSRH and Zcb sessions.

The kits cite them by the sha256 of the copy that ran. This directory holds those copies, so a cited reader can be
re-run from dev. The Plain arm's reader, grade-spec-chain-v2-plain.py, is in its own kit,
b5b2-plain-campaign-2026-11-28.

## The files, and how they differ from the copies that ran

| file | ran as (sha256, first 16) | here | the only change |
|---|---|---|---|
| grade-spec-chain-v2.py | b976d2884c8c42d5 | dd666675f18230e9 | `H` (the HDD root) read from `CAPSTONE_HDD` |
| grade-spec-chain.py (v1) | 4fa95eafa23e63b7 | df8310d64c96cf03 | `H` read from `CAPSTONE_HDD` |
| mapfaults.py | 715e0cf173d0cc1b | 17ef547ff9051b68 | `OD` (llvm-objdump) from `CAPSTONE_LLVM_OBJDUMP`, else `CAPSTONE_REPO_ROOT`'s build; `ROOT` from `CAPSTONE_HDD` |
| scan-c25ldc.py | 203b6edd65275166 | the same | none |
| grade-mcffault.py | c028de3b1ae2a07d | the same | none |
| mcf-exposure.py | 17c806a6c2806ca0 | the same | none |

The copies that ran had those values as local path literals. No other line differs: `diff` against the run copies
shows 2, 2 and 4 changed lines, and 0 for the other three.

- v2 is v1 plus INBAND, the console-splice repair: a stage's step text is read with interleaved kernel lines taken out
  (b5b2-inband-interleave-2026-11-26 on board/meas-pw). The campaign's binding grade is v2, with v1 reported beside it.
- **x264's PASS is reproducibility, not correctness (RETRACTED 2026-11-28, dev 59f229e2579d).** X264_REF's
  BuckBunny_New.264 key (`0e453c44…`) is the malformed Capstone stream, so a match is reproducibility, not correctness.
  Every Capstone x264 writes a malformed H.264 stream: x264's WORD_SIZE is sizeof(void*), which is 16 on capstone64.
  The reconstructed frames, which SPEC's own validation compares, are still a correct-output check. The grader is
  unchanged, because running sessions use it.
- scan-c25ldc.py and grade-mcffault.py call mapfaults.py from their own directory.

## Equivalence with the copies that ran (2026-11-28)

Each grader was run from both places on the same captures. Stdout, stderr, the exit code and every file an OUTDIR
grader writes were compared, with only the OUTDIR path normalised. All 11 cases were identical:
- v2 and v1 on board 3's CF-B5TL1-g1b and board 1's CS-B5TL1-x264b: C=ok and every stage PASS, as graded at the time;
- scan-c25ldc.py:
  - NONE on a spectest B5TL1-g2 capture with 4 fault lines, none of cause 25;
  - NONE on CF-g1b;
  - exit 3 NO READING on a nested-corpus capture whose 150 cause-25 faults map to no SPEC image;
- mapfaults.py mapping that B5TL1-g2 capture's faults to instructions;
- grade-mcffault.py: MCF COMPLETED on CF-g1b, and exit 2 "no mcf entry" on a capture without one;
- mcf-exposure.py: 6,365.7 s of mcf, 0 wedges, on CF-g1b.

## Running them

`source capstone/tests/capstone-test-env.sh` for CAPSTONE_REPO_ROOT, and set CAPSTONE_HDD to the HDD root. The stage
references are under `$CAPSTONE_HDD/c4/b5b2spec/board-stages/<ARM>/<bench>` (SPEC_STAGE_ROOT overrides them), and the
SPEC tree and run-stage.py are under `$CAPSTONE_HDD/spec` and `$CAPSTONE_HDD/wt-spec`.

    python3 -I grade-spec-chain-v2.py <uart.txt> <outdir> <ARM-bench,...>
    python3 -I scan-c25ldc.py <uart.txt> <ARM>
    python3 -I grade-mcffault.py <uart.txt> <ARM> [<seal>]
    python3 -I mcf-exposure.py <label>:<board>:<seal>:<uart.txt> ...

Each script's docstring is its specification: what it reads, what each verdict means, and the controls it was
checked against.
