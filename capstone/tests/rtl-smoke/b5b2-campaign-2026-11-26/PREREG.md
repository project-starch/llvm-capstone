# b5b2-campaign-2026-11-26: the timed SPEC test rows of the Line B campaign (pre-registration)

Written 2026-11-26 by the runtime lane, before any boot it covers. The plan is
`docs/plans/2026-11-25-line-b-final-campaign.md` ("The arms of the final campaign", "Last: three runs per arm"). The
qualification rows these follow are in `../b5b2-spectest-2026-11-25/RESULTS.md`. The Full + ports arm's first readings
are in nested-workloads (PREREG c4732c42a52b, Amendment 1 1ab5179e06d3, RESULT a162a8aefb60). The board lane runs the
boots; the runtime lane grades them. Nothing here changes an armed session.

## Images

Twelve payloads, `SHA256SUMS.camp` (full sha256 and byte size; built by `trapk/build-campaign.sh` and
`build-campaign-ports.sh`, records `build-campaign.done` and `build-campaign-ports.done`):

| Arm | Seal | Groups | Kernel tree |
|---|---|---|---|
| CF (Full) | b42c8c644 | g2i, g3, g4mk, g1b, x264b | `trapk/fpga-final` |
| CS (+Spatial) | 14f2ef81a | g2i, g3, g4mk, g1b, x264b | `trapk/fpga-final-spatial` |
| CFP (Full + ports) | b42c8c644 | g3 (gcc port v1, 6616eecf), g4mk (432e6ae2) | `trapk/fpga-final` |

- **Kernels.** Linux 6.1 with the trap-route series at 6d4504f8b073: `build/build-kernel-final.sh` and
  `build-kernel-final-spatial.sh` (dev d93ebefc8188). This identifies them, not the header comment in the tree's
  `build-kernel.sh`. Both are NO_HZ_IDLE at HZ=100, TRAP_TEST off, no SMP, KSM, THP or modules. The two `.config`s,
  INITRAMFS_SOURCE aside, differ only in `CONFIG_RISCV_CAPSTONE_SPATIAL_ONLY=y` (diff read 2026-11-26). The Image
  is relinked per overlay, so an Image hash identifies one payload, not the kernel.
- **Firmware.** caplifive-opensbi trap-route-keep-tags, `FW_FAST_RDTIME=y` (two fast-path references in each image's
  firmware, `fast-path-refs=2` in the build records). DTB `caplifive-b5b2.dtb`.
- **Userland.** B5TL1. Each group's overlay equals the qualification image's file for file: the build manifest
  `out/overlay-files.sha256` of every CF and CS image was compared with the kit's `overlay-files.sp.sha256`
  (SP-B5TL1-<group>) on 2026-11-26, and all ten are EQUAL (13, 8, 561, 29 and 12 files).
- **CFP-B5TL1-g3** carries gcc port v1. Port v2 (CFP-B5TL1-g3v2, 724f0027) replaces it in this campaign only by a
  later amendment, after nested's board pair (nested-workloads Addendum 1 c986a843f704) reads.

## Twin reference

All ten CF and CS images ran on the twin at 962 MiB (CF on the trap-route QEMU, CS on the spatial-only QEMU, `trapk/
camp-twin.sh`) and read PASS under grade-spec-chain-v2.py b976d288. g4mk was graded with
`SPEC_STAGE_ROOT=b5b2linux/g4mkroot`; my first grade used the default root and read MISS, a grader-invocation error.
`twin-camp.json` holds each image's verdict and per-stage data hashes.
- Every hash of every stage equals the board's qualification rows of the same group, on both seals (b42c8c644
  board 3 or board 2, 14f2ef81a board 1). The comparison's positive control, one changed hash, reads `1 differ`.
- CF equals CS on every file but x264's `log.dec`, which carries the decoder's timing and is not one of Amendment
  13's seven data outputs.
- Twin times are not predictions of board times.

Perl makerand's instruction count on the twin (`trapk/perlcount/RESULT.txt`, capmix3) is the same on CF and CS:
3,094,621,929 image instructions on each, 4,075,990,306 / 4,075,990,211 user, 1,062,083,667 / 1,061,576,239 kernel.

## What each boot is, and how it is read

- One image per boot. Control `smoke.dom rc=7` first, as built into each image's `/init`.
- `preflight-board-run.sh` at dev 6daffca3824c with `PREFLIGHT_DTS_NODES=65536`, before the boot; its exit line
  goes into the boot's chain.log.
- **C10.** 17.47 MB and 19.57 MB payloads are under the 20 MB budget. The 21.66 MB payloads (CF-g3, CS-g3,
  CFP-g3, CF-x264b, CS-x264b) BLOCK, and are proceeded past only by this ruling:
  - on b42c8c644, "C10 BLOCK at 21662216 B (21.66 MB), proceeded by runtime's ruling; this size ran on this seal as
    CF-B5TL1-g3 (997e2fd6fdbdecdf) and CFP-B5TL1-g3 (6616eecf0c797071)";
  - on 14f2ef81a, the largest payload proven is 21,657,608 B (SP-B5TL1-g3 82df0a83, PASS, board 1), and the CS
    21.66 MB images are 4,608 B larger. So the **first CS-B5TL1-g3 boot is the size proof**, and it runs before any
    CS-B5TL1-x264b. Its wording: "C10 BLOCK at 21662216 B (21.66 MB), proceeded by runtime's ruling; first boot of
    this size on 14f2ef81a, the size proof". If that boot fails to load or to reach `/init`, no further 21.66 MB CS
    image boots, and it is reported.
- The seal's name read back before and after each boot. A reflash follows the board-run procedure (sealed
  bitstream, sha256 checked before the flash, name read back after it, fallback registered) and is recorded in the
  session log.
- Bounds as the qualification rows: g2i, g3 and g4mk 4,500 s; g1b 45,300 s; x264b 44,100 s.
- **Grading.** grade-spec-chain-v2.py b976d288 is binding; v1 4fa95eaf is printed beside it.
  - `SPEC_STAGE_ROOT` is set per image: `c4/b5b2spec/board-stages` for g2i, g3, g1b and x264b, and
    `b5b2linux/g4mkroot` for g4mk. The CFP images use nested's roots, as their own pre-registration names them.
  - Stage lists: imagick for g2i; gcc for g3; perlbench for g4mk; lbm, nab, xz and mcf for g1b; x264 for x264b
    (all `B5TL1-…`).
  - x264 is graded by Amendment 13 (seven data outputs by hash).
  - Every other stage must also equal `twin-camp.json`'s hashes, file for file.
  - `scan-c25ldc.py` 203b6edd and, for g1b, `mcf-exposure.py` 17c806a6 and `grade-mcffault.py` c028de3b run on each
    capture.
- **Time** is the stage's `@@SPEC-END` (up1 - up0), summed over its commands. The host's clock from the runner's log
  (`continue` to `TRAPTEST: done seen`), taken against the boot's last uptime, is recorded for each boot as an
  independent check of the uptime clock. On the qualification perl pair, both clocks moved together (10.55 s by
  uptime, 11 s by host).
- A boot with no result, or a failed control, is VOID. It is re-run once, and both are recorded. A stage that does
  not return is a reading for that run, never a time.

## Design

**Three runs per image per arm**, in addition to (and not counting) the earlier N=1 campaign readings on board 2:
CF-g3 5.55 s, CF-g4mk 275.70 s, CFP-g3 5.83 s, CFP-g4mk 279.75 s (nested's RESULT a162a8aefb60; CF-g3 and CFP-g3
booted before their preflight).

**Phase A, board 1, same board for both seals (short groups g2i, g3, g4mk).** In order:
1. CS ×3 images (14f2ef81a is resident);
2. flash b42c8c644;
3. CF ×3 images, twice; then CF-g1b once;
4. flash 14f2ef81a;
5. CS ×3 images, twice;
6. flash b42c8c644;
7. CF ×3 images;
8. flash 14f2ef81a.

That is S F F S S F: three runs per seal on one board, balanced against drift. CF-g1b on board 1 measures board 1's
offset for the long stages. CS-g3's first boot in step 1 is the C10 size proof.

**Phase B, long groups g1b and x264b.**
- CS: g1b, x264b, g1b, x264b, g1b, x264b on board 1, after Phase A.
- CF: g1b, x264b, g1b, x264b on board 3, and g1b, x264b on board 2, after their armed sessions end.

**Phase C, CFP on board 2, paired with CF on the same board:** CF-g3, CFP-g3, CFP-g3, CF-g3, CF-g3, CFP-g3, then
the same order for g4mk. They give board 2's CF short rows. Board 3 needs no CF short rows: board
constancy between boards 2 and 3 is already read (within 0.07 % on four stages, `RESULTS.md`).

Slots: board 1 after SP-B5TL1-x264b (spb1e) ends; board 3 after its x264b session (r77b3x) ends; board 2 after the
CFP g3v2 pair. The board lane may reorder whole phases between boards if a board is lost, and records why. It may
not reorder within Phase A's S F F S S F.

## Predictions (written before any of these boots)

- **P1.** Every stage PASS under v2 and v1, with data hashes equal to `twin-camp.json`. No cause-25 fault at an LDC.
  mcf completes (consistent with R-80's fix, not evidence for it).
- **P2.** On b42c8c644, each CF stage is no slower than its qualification B5TL1 row on the same seal. The final
  kernel only removes work: TRAP_TEST off, fast rdtime, NO_HZ_IDLE. The two campaign readings already point this way
  (perl 275.70 against 282.94 s; gcc 5.55 against 5.76 s). A CF stage more than 1 % slower than its qualification row
  is reported as a finding.
- **P3 (rtl's, by source).** On one board, CS ≤ CF in time for every stage. Every change from b42c8c644 to 14f2ef81a
  removes work: the lifetime unit is a one-cycle stub with its memory channel tied off, verdict lookups always hit,
  and nothing in the FPGA shell changed. The twin shows equal instructions. On board 1, if CS perl is slower than CF
  perl by more than the larger of the two arms' ranges, the cause is outside the core RTL. The candidates, in rtl's
  order, are the kernel option, the board, and the clock. An hpm row is then pre-registered on its own, with rtl's
  events: 0x17 must read > 0 on Full and 0 on CS, then 0x02, 0x01, 0x04, 0x16 and 0x0f, with mcycle and minstret.
- **P4.** For every stage, board 1's CF time is within 0.2 % of the same image's CF time on boards 2 and 3. P3 and P4
  together decide the qualification perl gap (282.94 / 293.96 s):
  - CS slower than CF on board 1 means a seal or kernel cause;
  - board 1's CF and CS equal, and both slower than board 2's CF, means the board;
  - neither of these means the gap was not reproduced.
- **No band stops a run.** These predictions classify readings. They never abort a boot.

## Reported per arm

For each stage: the three times, their mean and range, the seal, the board and the image hash. For each pair:
- the Full/+Spatial ratio from board 1's same-board runs (Phase A);
- for the long stages, the cross-board ratio, with board 1's CF-g1b offset stated beside it;
- the ports ratio from board 2's pairs (Phase C).
No interval beyond the range is claimed at N=3.

## Amendment 1 (2026-11-26, before any boot of this campaign): P4 has no reading for g2i

The design runs CF-B5TL1-g2i only on board 1 (Phase A). Boards 2 and 3 get no CF imagick, so P4 cannot be read for
imagick. P4 is read for gcc and perlbench from Phase C on board 2, and for lbm, nab, xz and mcf from board 1's CF-g1b
against Phase B's CF-g1b on boards 2 and 3. For imagick it reads NO READING. The decision it exists for, perl's
qualification gap, is unaffected. The board plans armed from fec8d4357796 are unchanged (camp-b1.plan d84532ac46260a2c,
camp-b2.plan 14035a3ab9821615, camp-b3.plan 827186facd1d7022; each read here against the design, with the order
S F F S S F on board 1).
