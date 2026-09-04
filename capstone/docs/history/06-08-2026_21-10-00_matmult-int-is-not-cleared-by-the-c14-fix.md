# matmult_int is NOT cleared by the C-14 fix, and stays off the controls list

Date: 2026-08-06
Status: correction. Two numbers in commit `8b10e3be3327`'s message are wrong; one inference
drawn from a board run does not hold. The C-14 fix itself is unaffected and stands.

## What happened

Boot of 2026-08-06 (bitstream asserted as `caplifive_65536_nodes.bit`, control `k800` = 4
first) ran `mmfix` -- `matmult_int_app.c` built at `8b10e3be3327`, `-O1`, `DOMAIN_GLUE=interp`,
`DOMAIN_BASE_VA=0xa0000` -- and it returned **774662735**, the correct oracle. `matmult_int` is
recorded as an open silicon divergence and sits on the NOT-controls list.

That single result was nearly read as "C-14 was R-1's cause". It does not support that, for
four independent reasons, each verified against primary sources.

## 1. The fix is a NO-OP at -O0, where the documented miscompute lives

    matmult_int -O0 fix=on   movc=14  md5=71975c8cafd8
    matmult_int -O0 fix=off  movc=14  md5=71975c8cafd8    <- byte-identical
    matmult_int -O1 fix=on   movc=13  md5=6af5cd3c2eab
    matmult_int -O1 fix=off  movc=17  md5=3a5463e843f1

Every `movc` in the -O0 build is either `movc rd, zero` (source is x0, which
`ariane_regfile_ff.sv` forces to zero every cycle, so the destructive write cannot take) or
`movc s0, sp` (sp is NONLIN after the glue's `delin`, which takes MOVC's non-destructive arm).
There is no destructive-live-scalar `movc` at -O0 at all.

So the recorded -O0 miscompute -- 1166210317 against the correct 774662735, commit
`03ca1ea85873` -- **cannot** be a C-14 instance and is **not** fixed. It stands as a separate
open divergence, or must be retracted on its own evidence. It has not been.

## 2. The before/after pair differs by TWO variables

The artifact that failed (`overlay/test-domains/matmult_int.dom`, 08-05) and the artifact that
passed (`mmfix.dom`) differ by three instructions AND by entry VA -- 0x10000 versus 0xa0000.
`build-ladder-domain.sh` documents `DOMAIN_BASE_VA` as existing specifically to probe R-3, an
address-keyed hang, and this project has a documented layout sensitivity where four added
instructions flipped a passing rung. "We moved the code 576 KiB and it started working" is not
an attribution.

The matched control -- same VA, fix off -- was physically inside the booted image and was not
run. The boot used 4 of its available domains. That experiment cost nothing and was skipped.

## 3. The "before" boot carries no verdict either

Re-scoping `/tmp/capstone/mtv/rungs-raw.txt`: one boot banner, and the first post-banner test is
`matmult_int` at position 1 of 5 with no control ahead of it, on a freshly built image. By this
repo's own rules that is void twice over -- `known-good-controls.md` says a fresh image cannot
separate "this image failed" from "the boot failed", and the control's own failure rate is ~1 in
5. One uncontrolled failure against one controlled pass is not a bisection.

## 4. "R-1 = C-14" is a category error as stated

`ISSUES.md` defines R-1 as an RTL LSU hazard with its own reproducer
(`tests/fpga-repros/R01-lsu-hazard/`, `rawhazard{5,6,7}_fpga_app.c`), its own minimal failing
case, and scored predictive controls. `matmult_int` appears there only under **Impact**. Fixing
`matmult_int` in the compiler says nothing about R-1 unless the R-1 reproducer is retested.

The fix does change those binaries (`rawhazard7`: `movc a7,a5` / `movc a6,a4` -> `mv`), so the
question is answerable in one boot -- it just has not been asked.

## Corrections to the record

* `8b10e3be3327`'s message says "matmult_int 16 -> 13". Measured: **17 -> 13**, in every
  configuration checked (interp glue, generated glue, bare object). The delta of 4 is right;
  both endpoints were wrong.
* `d0b1200d5e58`'s message presents the `mmfix` pass as evidence about `matmult_int`. It is
  evidence that a fixed binary completed the workload once on current silicon -- which is not
  nothing, since the oracle is an FNV-1a hash over the whole 8x8 product matrix and cannot
  arise from an early exit -- but it is not evidence about the cause.

## matmult_int stays on the NOT-controls list

A control requires a track record; one boot cannot provide one. Unchanged.

## What would settle it

1. One boot: `k800` control -> `mmfix` (@0xa0000, fix on) -> the same source at the SAME VA with
   `-capstone-fix-destructive-copies=false`, placed LAST as the one expected to wedge.
2. Repeat `mmfix` across separate boots; N=1 on this board is not attribution.
3. `rawhazard5/6/7` at HEAD -- the only actual test of "R-1 is an instance of C-14".
4. `matmult_int` at -O0 on current silicon: the fix cannot touch it, so if it still returns
   1166210317 the rung is still divergent.
5. Save driver stdout per run, so the resident bitstream is evidenced rather than asserted.

## Tooling hazard found while auditing this

`grep` here is `ugrep`, and it silently produced NO output on a UART transcript containing
control bytes -- which reads exactly like "the string is absent". A Python byte-level search
found 8 occurrences of the same string. Same family as the earlier `awk strtonum` and
decimal-vs-hex incidents. **Scan UART transcripts with `python3` byte search, not `grep`.**
Verdicts taken from a driver's own stdout summary are unaffected: those are produced by the
runners' Python regex over run-scoped text.
