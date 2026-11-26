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
- Amendment 9's cause-25-at-LDC rule. Every capture below has **0** occurrences of the substring `cause=` (`grep -a`;
  the same grep counts 456 lines with `cause=25` in board 2's nested-corpus capture, so the check fires). The board
  lane's `scan-c25ldc.py` prints NONE on each, but it anchors at the start of a line and parses only 140 of those 456
  spliced lines. Its NONE is therefore not the evidence; the substring count is. It has been reported for a fix.

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
| 2 | B5T-g1b | 45e4fbdd | lbm PASS 1,052.62; nab PASS 754.91; xz 12/12 PASS 4,875.33; mcf PASS 6,826.53 | **v2**, v1 equal; no INBAND line |
| 2 | B5T-g2b | ca1a3e42 | imagick PASS 11.84; x264 stopped by ruling | v1 |
| 2 | B5TL0-g2b | cba7e1f5 | imagick PASS 11.83; x264 stopped by ruling | v1 |
| 2 | B5TL1-g2b | 6f8c83c0 | imagick PASS 11.63; x264 stopped by ruling | v1 |

Control `smoke.dom rc=7` read AS PREDICTED in every row.

**mcf (Amendment 8).** Both B5T-g1b rows completed with specdiff 2/2 PASS and no fault line: board 3 exposure
6,844.3 s, board 2 6,842.2 s. Their `@@SPEC-END` times are 6,828.57 and 6,826.53. This is CONSISTENT with R-80's fix, and
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

The name read back before and after each boot is 14f2ef81a. These three were graded at 13:41-14:00 +08, before
Amendment 14, so v1 is their binding grader.

Against B5TL1 on b42c8c644 (the same userland, a different overlay image): imagick 11.57 against 11.63 on both boards,
gcc 5.70 against 5.76. Each is one reading per arm, inside the 0.01 s uptime resolution plus run-to-run spread, which
is not measured yet. No +Spatial cost or saving is read from these rows. That is the campaign's three-run job.

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

## Still owed

- Board 3: B5TL1-g1b (running), B5T/B5TL0/B5TL1-x264b, the g4mk rows, ldcexc2.
- Board 1: SP-B5TL1-g1b (running), then SP-B5TL1-x264b.
- Every one of these grades under v2, with v1 beside it (Amendment 14). x264 is graded by its data hashes
  (Amendment 13).
