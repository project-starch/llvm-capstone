# b5b2-plain-campaign-2026-11-28: the Plain arm's timed SPEC test rows on rvbase (pre-registration)

Runtime lane, 2026-11-28, written before any boot of these images. It implements decision 2 of the Line B plan's
28 Nov audit (runtime/plan-audit-2026-11-26 8880ecd564ea): the Plain arm is the baseline every Line B overhead row
needs, and it has never booted on rvbase. The campaign it pairs with is b5b2-campaign-2026-11-26 (Full CF on
b42c8c644, +Spatial CS on 14f2ef81a).

## Why new images

The Plain images of 25 Nov (tests/line-b-plain, dev 22145821659d) do not meet the plan's matching rules:
- their firmware is stock OpenSBI without FW_FAST_RDTIME, where the Full arm's has it;
- x264 sits in one image with imagick, where the campaign gives x264 an image of its own;
- their board-init bounds every stage at 3,600 s, which would stop mcf (about 6,000 s on Full) and x264 (about
  33,000 s).
Their native images are kept: the compiler lane confirmed on 28 Nov that Plain-matched is still B5TL1's pair, and
its 16 images match MANIFEST.json (0 mismatches).

## The arm

| | Plain (this kit) | Full CF (the campaign) |
|---|---|---|
| Bitstream | rvbase 8db8f9cda, `caplifive_rvbase-matched-ic32k-dtlb64_8db8f9cda.bit`, sha256 efc3ab8abd5aa194… | b42c8c644 |
| L1D / I-cache | 32 KiB 8-way, as Full (plan, "the matching rules") | 32 KiB 8-way |
| Firmware | sbi-keep 25b14da6, keep-tags off, **FW_FAST_RDTIME=y** (fw_base.S:932's stock path; the built ELF carries `fw_fast_mtime_addr` and the mtval test, the 25 Nov images carry neither) | sbi-keep 25b14da6, keep-tags on, FW_FAST_RDTIME=y |
| Kernel | trapk/plain: Linux 6.1, the board config edits, NO_HZ_IDLE at HZ=100, no Capstone series | the final Full kernel: NO_HZ_IDLE at HZ=100, TRAP_TEST off |
| DTB | caplifive-plain.dtb; its memory node equals the Full DTB's | caplifive-b5b2.dtb |
| Userland | Plain-matched natives (MANIFEST.json, dev 22145821659d) | B5TL1 |
| Overlay | the campaign's own overlay per group, made by make-plain-overlays.py (below) | SP-B5TL1-<group>/ovl |

**The overlays** (make-plain-overlays.py): each group's campaign overlay with exactly these changes, everything else
asserted byte-identical:
- each stage B5TL1-<bench> is renamed PLM-<bench> (stages.order and stages.bound follow, checked line for line), and
  its one native is replaced by the Plain-matched image, checked by sha256 against MANIFEST.json;
- /smoke.dom, a capability domain Plain cannot run, is removed and /plain-smoke (sha256 42e1f6d1b289fa93…, the 25
  Nov images' control) added; /init's control line runs it and prints `TRAPTEST: control plain-smoke rc=`.
- Groups and bounds are the campaign's: g1b (lbm 3,600, nab 3,600, xz 7,200, mcf 30,000 s), g2i (imagick 3,600),
  g3 (gcc), g4mk (perlbench makerand), x264b (43,200 s).

**The payloads** (trapk/plain/make-payload-plain.sh, FW_EXTRA="FW_FAST_RDTIME=y"; each overlay's file list equals the
build's overlay-files.sha256):

| image | fw_payload.bin sha256 | bytes |
|---|---|---|
| PLM-g1b | 35cbcd318cba587e09e6495202be3af5cf3c2d777c491ee0786b5f0c9d54cdf2 | 19,564,552 |
| PLM-g2i | d8eebda2e4e6b9764ea19a27d186734e3067f37a4fe72f6583c497480d6e90d3 | 17,467,400 |
| PLM-g3 | ba1bdceeaf504b689029e23ec434e76c963fb4fe15e9ed7a497bf410fa91f2e0 | 21,661,704 |
| PLM-g4mk | 64ed3f3a12884d5ac74a4403c0cd845b08cd38815fd202cd78807087bc53b4f8 | 19,564,552 |
| PLM-x264b | 98b5d8ceff1bdc96246b1451251840442c5ca1a8a15bf0753eee2ba4f2f436e1 | 21,661,704 |

## The reader

- **grade-spec-chain-v2-plain.py:** grade-spec-chain-v2.py b976d288 (the campaign's grader) with three lines changed
  (`diff` shows exactly these): the control regex `control smoke.dom` → `control plain-smoke`, allowing plain-smoke's
  newline after `ok` (below); its docstring line; and the HDD root read from CAPSTONE_HDD instead of a literal path. Negative control, run before this pre-registration:
  on the campaign's CF-g3 capture it reads `C FAILED: no control line` (the stock grader reads PASS there). Stages are
  read with SPEC_STAGE_ROOT = the kit's stage-root (PLM/<bench> → the grader's B5TL1 references; perlbench's →
  g4mkroot, as the campaign graded g4mk).
- Time per stage: the sum of up1 − up0 over the stage's @@SPEC-END lines, as in the campaign.
- Hashes: each stage's result.json against the twin replay below, and against the campaign's CF twin
  (`twin-camp.json`) for the data outputs; x264's `log.dec` is reported, never counted.

## The twin replay (before this pre-registration is pushed)

twin-plain.py on qemu-crv as a plain riscv64 (-cpu rv64,sstc=false, no Capstone property), 962M, stock generic
OpenSBI (sbi-keep-run fw-n), each group's Plain Image: all five boots ran to `TRAPTEST: done` and exited, every
command rc 0, plain-smoke `ok` rc 7 (g3 2 s, g2i 3 s, g4mk 29 s, g1b 567 s, x264b 2,221 s).

- **The Plain grader, both controls.** Negative: the campaign's CF-g3 capture reads `C FAILED: no control line`, and
  a Plain capture with its `ok` line removed reads `C FAILED`. Positive: the five twin captures read C=ok and every
  stage PASS except x264 (next bullet). The positive control first caught a defect in the copy: plain-smoke prints
  `ok\n` where smoke.dom prints `ok` with no newline, so the first regex read C FAILED on a correct capture. The
  regex now allows that one newline; it is the copy's only other change.
- **Hashes against the campaign's CF twin** (`twin-camp.json`, file by file, a perturbed hash reads `1 differ`):
  gcc 5, imagick 8, perlbench 557, mcf 5, nab 5, xz 26, lbm 4 files, **all equal**.
- **x264 differs from B5TL1 in its encoded stream.** Of the 15 files, `BuckBunny_New.264` (eddbb3318627f3b0 against
  0e453c440e686acc) and the run's stderr differ, beside the timing file log.dec. Every decoded frame, BuckBunny.yuv
  and every decision percentage in stderr are equal; only the frame sizes move: I average 90,388 B against 90,394, P
  30,333 against 30,334, 2,713.67 kb/s against 2,713.74. So the Plain encode writes a few tens of bytes less than
  B5TL1's. **Not attributed** (a header string and a float-path difference are both open). It is reported as a
  finding, and x264's Plain reference is Plain's own twin: the board's x264 outputs must equal this replay's.

**Binding hash references for the board rows:** this replay (every file but log.dec), which equals the campaign's
CF twin on every stage except x264's encoded stream and stderr.

## The board session

- **Board:** the first free board after the compiler's x264 bracket on N, or board 1 or 3 when the campaign's
  Phase B ends.
- **Flash:** rvbase 8db8f9cda by the board lane's procedure: sealed bitstream, sha256 checked locally, readback
  names it, a fallback registered (the seal the board ran before), logged.
- **Boot line:** PLM-g3, PLM-g2i, PLM-g4mk, PLM-g3, PLM-g2i, PLM-g4mk, PLM-g3, PLM-g2i, PLM-g4mk, PLM-g1b, PLM-x264b,
  PLM-g1b; then the flash back to the fallback seal.
- **The first boot (PLM-g3, 21,661,704 B) is the Plain kernel's Tier 1 and the size proof on rvbase:** it must reach
  /init and print `ok` with plain-smoke rc 7. If it does not, after one rerun, the session stops before any other
  boot, the flash back runs, and the stop is recorded as infra or as a Plain-kernel finding.
- **C10:** payloads above 20 MB proceed by the size rule only, citing the first boot; no DECLARED bake.
- Every boot runs preflight-board-run.sh first and keeps its exit line.

## Predictions (written before any of these boots)

- **P1.** Every stage PASS under the Plain grader, with every hash but log.dec equal to the twin replay's, and to the
  campaign's CF twin on every stage except x264's encoded stream and stderr (the twin replay's finding above).
- **P2.** On every stage, Plain's mean time is below Full CF's mean (the campaign, b42c8c644): Plain runs no lifetime
  or capability operations. Reported with the ratio CF / Plain and both ranges. No band: no earlier Line B row
  predicts a ratio, and P1-era ratios are on another seal.
- **P3.** The Plain kernel boots on rvbase: `Run /init`, plain-smoke `ok` rc 7, no oops.
- **No band stops a run.**

## Disclosures

- Plain-matched differs from B5TL1 in the options README.md (tests/line-b-plain) lists, each with its reason; it is
  B5TL1's pair by the lead's train rule.
- /init differs from the campaign's by the control line only.
- The cross-board offset (about 0.06 % between boards on CF) applies when Plain and CF run on different boards.

## Amendment 1 (2026-11-28, before any boot of this kit): the board and the runner bounds

- **Board 2.** It stood idle while the stage-2 x264 bracket's pre-registration was not yet pushed, so by the Line B
  plan's rule (an idle board goes to the first pushed pre-registration) this session runs on board 2, b42c8c644
  resident before the flash and restored after it. The choice of board predicts nothing here.
- **Runner bounds** are the Full arm's (4,500 s short, 45,300 s g1b, 44,100 s x264b), not the "about 1 h / 4 h /
  10 h" the runtime lane's message to the board lane suggested: each must exceed the image's own stage bounds (3,600; g1b's sum 44,400;
  x264's 43,200) or a slow but legitimate Plain stage would be cut by the runner instead of read, which P2 tests. The
  wedge guard is the runner's 1,200 s silence limit.
- **rvbase 8db8f9cda on silicon before:** the P1 train campaign's Plain rows ran on it (boards 2 and 3, kit bkit95;
  origin/spec-silicon results/2026-11-08-train PREREG:5354, :5846). Its seal is a manual one (finish.verdict
  COLLECT-FAILED 127: no collector in the riscv-base tree; SEAL.txt with the .bit sha256 efc3ab8a…, 0 routing
  errors, WNS +0.177). This kit's Plain kernel has not booted on it before.

## Amendment 2 (2026-11-28, before any boot it adds): the long rows to N = 3 on a second board

The session on board 2 gives Plain's long groups N = 2 (g1b) and N = 1 (x264b). The final campaign wants three runs
per arm. Of boards 1 and 3, the one the stage-2 x264 bracket does not take goes idle when its Phase B ends.
- **Added session, on that board:** FLASH-RVB (8db8f9cda, by the same procedure and gates as board 2's session),
  PLM-g2i as the control and that board's Tier 1 on rvbase (short, expected PASS), PLM-x264b, PLM-g1b, PLM-x264b, then
  the flash back to the board's own seal.
- **The images, grader, C10 rule, bounds and predictions are this file's**, unchanged: the same five payloads by
  sha256, the Plain grader, P1-P3.
- **Reading:** Plain's g1b (N = 3) and x264b (N = 3) come from two boards. The cross-board offset on the same CF image
  is within 0.06 % (b5b2-campaign RESULTS); it is reported beside each mean and not corrected.
