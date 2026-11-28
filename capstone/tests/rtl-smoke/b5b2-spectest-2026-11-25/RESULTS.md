# b5b2-spectest-2026-11-25: results on the final seals

The SPEC CPU2017 **test** rows read on the two final bitstreams: b42c8c644 (Full) and 14f2ef81a (spatial-only). They
are **qualification** rows: pinned kernel `trapk/fpga-v1` (HZ_PERIODIC, TRAP_TEST on), keep-tags firmware with
FW_FAST_RDTIME off. The campaign's timed rows run on the final kernels and are not these
(`docs/plans/2026-11-25-line-b-final-campaign.md`). Earlier seals' rows (904cd9c6b, 88db40027, 28fcbb646) are kept in
the PREREG amendments that read them. Only the 28fcbb646 half of the matched pair is repeated here.

**How each row was checked, by the runtime lane, from the board lane's captures:**
- the image's fw_payload sha256 (from the run's `run.log`) against this kit's `SHA256SUMS.*`;
- the resident bitstream's name, read back before the boot and, where the chain did so, after it. The console exposes
  no sha256 (rc=3: the name matches, the sha256 is not shown);
- the verdict from `grade-spec-chain.py` (v1, 4fa95eaf). For a row first graded after Amendment 14 (pushed
  2026-11-26 14:18 +08), it comes from `grade-spec-chain-v2.py` (b976d288), with v1 beside it. B5T-g1b on board 2 was
  regraded by this lane from its capture under both, and reproduces the board lane's grade file byte for byte;
- Amendment 9's cause-25-at-LDC rule. Every capture below has **0** occurrences of the substring `cause=` (`grep -a`).
  In board 2's nested-corpus capture the same grep finds 456 lines with `cause=25`, so the check fires. Those 456 lines
  are 150 fault events: each prints as a `sublet: fault` line, its `TRAPTEST: dmesg` echo and the launcher's
  `capstone-exec: domain fault` line. Two more events are kernel faults. The board lane's `scan-c25ldc.py` (28dca1e4)
  anchored at the start of a line and found 140 of the 150 events; the 10 it missed occur only mid-line. A first
  version of this file said "140 of those 456 lines", which counted lines, not events. The board lane fixed it at
  203b6edd:
  - an unanchored match, de-duplicated by (timestamp, pid, pc);
  - NO READING for a fault that mapfaults.py cannot map. Before, it said "not at an LDC";
  - mapfaults blocks keyed by pid and pc and bounded. Before, an unmapped fault could read the next fault's
    instruction.

  Its controls are spliced into a copy of a real capture. The fixed scanner's verdict line equals the recorded one on
  all 15 rows scanned.

Times are the stage's `@@SPEC-END` uptime difference, summed over a stage's commands, in seconds. /proc/uptime has
0.01 s resolution.

## Full, b42c8c644

| Board | Row | Image | Stages: verdict, time (s) | Grader |
|---|---|---|---|---|
| 3 | B5T-g1b | 45e4fbdd | lbm PASS 1,051.91; nab PASS 754.92; xz 12/12 PASS 4,873.51; mcf PASS 6,828.57 | v1 |
| 3 | B5TL0-g1b | 17513d66 | lbm PASS 1,053.41; nab PASS 754.00; xz 12/12 PASS 4,865.42 (no mcf, Amendment 6) | v1 |
| 3 | B5T-g2b | ca1a3e42 | imagick PASS 11.85; x264 NO READING, 14,400 s bound (Amendment 11) | v1 |
| 3 | B5TL0-g2b | cba7e1f5 | imagick PASS 11.84; x264 stopped by ruling (Amendment 11) | v1 |
| 3 | B5TL1-g2b | 6f8c83c0 | imagick PASS 11.63; x264 stopped by ruling | v1 |
| 3 | B5T-g3 | c40b8ce9 | gcc PASS 7.15 | v1 |
| 3 | B5TL0-g3 | ab7fe832 | gcc PASS 7.07 | v1 |
| 3 | B5TL1-g3 | ce0ca633 | gcc PASS 5.76 | v1 |
| 3 | B5TL1-g1b | 539b2ef4 | lbm PASS 1,052.26; nab PASS 752.86; xz 12/12 PASS 4,863.14; mcf PASS 6,601.15 | **v2**, v1 equal; no INBAND line |
| 3 | B5T-g4mk | 62795fa9 | perlbench PASS 290.45 | **v2**, v1 equal; INBAND 2 |
| 3 | B5TL0-g4mk | a96705ba | perlbench PASS 291.48 | **v2**, v1 equal; INBAND 2 |
| 3 | B5TL1-g4mk | 192a5a75 | perlbench PASS 282.94 | **v2**, v1 equal; INBAND 2 |
| 2 | B5T-g1b | 45e4fbdd | lbm PASS 1,052.62; nab PASS 754.91; xz 12/12 PASS 4,875.33; mcf PASS 6,826.53 | **v2**, v1 equal; no INBAND line |
| 2 | B5T-g2b | ca1a3e42 | imagick PASS 11.84; x264 stopped by ruling | v1 |
| 2 | B5TL0-g2b | cba7e1f5 | imagick PASS 11.83; x264 stopped by ruling | v1 |
| 2 | B5TL1-g2b | 6f8c83c0 | imagick PASS 11.63; x264 stopped by ruling | v1 |

Control `smoke.dom rc=7` read AS PREDICTED in every row.

**INBAND 2** in the three g4mk rows: v2 removed two whole `TRAPTEST: tick` lines (about 399 s and 520 s) from inside
perlbench's 557-line hash block. v1's filter drops the same lines, and the verdicts are equal.

**mcf (Amendment 8).** Both B5T-g1b rows completed with specdiff 2/2 PASS and no fault line. The exposure is 6,828.2 s
on board 3 and 6,826.2 s on board 2, from `mcf-exposure.py` 17c806a6. Its predecessor, aa1d6557, took the stage's end
as the latest time anywhere after it, validation and the dmesg dump included, and gave 6,844.3 and 6,842.2. Those two
numbers are withdrawn. The `@@SPEC-END` times are 6,828.57 and 6,826.53. `grade-mcffault.py` (c028de3b, unanchored)
reads both rows byte-identically to its predecessor. B5TL1-g1b (board 3) also completed, specdiff 2/2 PASS, no fault
line: exposure 6,600.8 s from 17c806a6, `@@SPEC-END` time 6,601.15 s. This is CONSISTENT with R-80's fix, and
not evidence for it, by Amendment 8. Amendment 7's refutation condition (a silent wedge) did not occur.

**No cause-25 fault at an LDC** in any row, as Amendment 9 predicted (see the check above).

**Board constancy, same image (45e4fbdd), same seal:** board 3 against board 2, lbm 1,051.91 / 1,052.62, nab 754.92 /
754.91, xz 4,873.51 / 4,875.33, mcf 6,828.57 / 6,826.53. Each pair is within 0.07 %. For the g2b images, imagick is
11.85 / 11.84, 11.84 / 11.83 and 11.63 / 11.63. That is N=1 per board, not an interval.

## Spatial-only, 14f2ef81a (Amendment 12, board 1)

| Row | Image | Stage: verdict, time (s) | Grader |
|---|---|---|---|
| SP-B5TL1-g2i | 5064e3d5 | imagick PASS 11.57 | v1 |
| SP-B5TL1-g3 | 82df0a83 | gcc PASS 5.70 | v1 |
| SP-B5TL1-g4mk | b799fdff | perlbench PASS 293.96 | v1 |
| SP-B5TL1-g1b | d5653c64 | lbm PASS 1,051.55; nab PASS 748.94; xz 12/12 PASS 4,853.66; mcf PASS 6,247.42 | **v2**, v1 equal; no INBAND line |

The name read back before and after each boot is 14f2ef81a. SP-B5TL1-g1b's data hashes equal the spatial-only twin's
on all four stages (4, 5, 26 and 5 hashes); its mcf completed with no fault line, exposure 6,247.0 s. The first three
rows were graded at 13:41-14:00 +08, before
Amendment 14, so v1 is their binding grader.

Against B5TL1 on b42c8c644 (the same userland, a different overlay image): imagick 11.57 against 11.63 on both boards,
gcc 5.70 against 5.76. Each is one reading per arm, inside the 0.01 s uptime resolution plus run-to-run spread, which
is not measured yet. No +Spatial cost or saving is read from these rows. That is the campaign's three-run job.

The g1b and g4mk pairs, same B5TL1 userland, Full (board 3) / +Spatial (board 1), N=1 each: lbm 1,052.26 / 1,051.55,
nab 752.86 / 748.94, xz 4,863.14 / 4,853.66, mcf 6,601.15 / 6,247.42, perlbench 282.94 / 293.96. Perl is the one
stage where the spatial-only seal is slower (by 11.02 s). It is open and not read as a seal property. The twin does the same work on both: capmix3 on
the campaign g4mk images (CF on the trap-route QEMU, CS on the spatial-only QEMU, same initrd, 962 MiB) counts
3,094,621,929 perl image instructions on each, 4,075,990,306 / 4,075,990,211 user instructions in all, and
1,062,083,667 / 1,061,576,239 kernel instructions (`trapk/perlcount/RESULT.txt`). The gap is therefore cost per
instruction, the board, or noise. The twin cannot separate these. The campaign's same-board pairs can.

## The matched pair (Amendment 9): 28fcbb646 against b42c8c644, board 2

| Stage | 28fcbb646 (b2diag, 45e4fbdd) | b42c8c644 (r77b2, 45e4fbdd) |
|---|---|---|
| lbm | 1,051.99 | 1,052.62 |
| nab | 754.93 | 754.91 |
| xz (12) | 4,873.94 | 4,875.33 |
| mcf | 6,808.70 | 6,826.53 |

The image is the same and so is the board. mcf is 17.83 s (0.26 %) slower on b42c8c644, at N=1 on each side, so it is
not attributed. The imagick half is **not comparable**. On 28fcbb646 it ran in the g2 images (d837e9fa, ad19c4e7,
3891f8e1: 20.86, 20.90, 20.64 s), and on b42c8c644 in the g2b images (11.84, 11.83, 11.63 s). The images differ in
`/init`, stage order and bounds (Amendment 10), and x264 ran before imagick in g2. The 9 s gap is recorded and not read
as a property of either seal.

**ldcexc2 (Amendments 7 and 9)** is graded in its own kit, `../b5b2-ldcfix-bare-2026-11-25/`. On b42c8c644
(board 3, after B5TL1-g4mk) its fixed arm reads SR 2 0x18 0x1 and its control SR 2 0x0 0x0, `LDCKIT fixed AS
PREDICTED`. The unfixed 28fcbb646 control that must wedge is still unread, so this shows the fixed arm completing,
not the wedge gone.

## x264b rows (Amendments 11-14)

Graded under v2 with v1 beside it (Amendment 14), by x264's seven data outputs (Amendment 13); times are up1 − up0 of
each command, read from the capture by this lane.

| Board | Seal | Row | Image | gen (s) | run (s) | v2 / v1 | uart |
|---|---|---|---|---|---|---|---|
| 3 | b42c8c644 | B5T-x264b | 783d3276 | 2,127.35 | 33,839.95 | PASS 7/7 / PASS | e2ad11b2 |
| 3 | b42c8c644 | B5TL0-x264b | 0b483e78 | 2,126.51 | 33,865.01 | PASS 7/7 / PASS | c42c2150 |
| 3 | b42c8c644 | B5TL1-x264b | b3251258 | 2,127.40 | 33,881.05 | PASS 7/7 / PASS | c90303dd |
| 1 | 14f2ef81a | SP-B5TL1-x264b | 62ea4930 | 2,126.10 | 33,795.06 | PASS 7/7 / PASS | 955503a9 |

- Every command rc 0; no `cause=` substring in any capture; specdiff 0/3 NO-OUTPUT on each, as Amendment 13 predicts.
- **The pair** is B5TL1-x264b against SP-B5TL1-x264b, the same userland: +Spatial reads −0.25 % on run. It is
  cross-board (board 1 against board 3); the boards agree within 0.07 % on the campaign's CF rows.
- B5TL0 and B5T read −0.05 % and −0.12 % against B5TL1 on the same board (run).
- **UNGATED:** board 3's three rows ran in session r77b3x, armed before the rule that every boot runs
  preflight-board-run.sh first, so none of them has a preflight exit line. SP-B5TL1-x264b did run it.
- The board lane first compared SP-B5TL1-x264b with B5T-x264b, a different userland; that is not a pair.

## Still owed

Nothing on this kit's boards. The campaign's timed rows (b5b2-campaign-2026-11-26) supersede these as the Line B
readings; these remain the qualification rows.
