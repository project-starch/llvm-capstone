# CREVOKE run lengths in the B5b-2 SPEC images: results (2026-11-26)

Pre-registration: PREREG.md, c1445d0e878f, and amendment 1, 24b297690265, both pushed before the boot.

**Every prediction holds. No CREVOKE or CRENEW in any stage of B5TL1 or B5T walks a single record, so no row is
exposed to the long-walk timeout.**

## The boot

- One boot on the trap twin, 2026-11-25 22:53 to 2026-11-26 00:17 (5,013 s after the lock).
  - QEMU: capstone-qemu fa6ce7554d, pinned binary 1da6e8b57877ce56.
  - Kernel: trapk 749fed2606e3ebca (pinned copy). Firmware: sbi-keep-run fw-y. -m 963M.
- Stage images: b5b2spec/board-stages, B5TL1 then B5T; images.sha beside the run.
- crevoke-ctl.dom: 4f223793ff8dfcd3.

An earlier boot of the same initramfs ended at the control check (22:47, after 37 s). The driver's reader treated the
snapshot taken before the first CREVOKE as unreadable, because the counter opens its files lazily and no file existed
yet. The counter itself was fine: that boot's BIG lines read exactly 1500 and 25000. The reader now takes a missing
BEGIN snapshot as an empty histogram, and still treats a missing END snapshot as an error. Re-checked on that boot's
own snapshots, the control then passes; a synthetic END-missing case still fails. That boot's files are kept apart
(abort1/) and nothing from it is read below.

## Prediction 1, the control: PASS

ctl1 minus ctl0 is exactly one CREVOKE each with runs of 0, 10, 1500 and 25000, nothing missing, and CRENEW is equal.
The BIG lines are the control's two (n=1500 and n=25000, pc 0x3ff7fe1a54, U), and only those.

## Predictions 2 to 4, the stages

| stage | B5TL1 CREVOKE | B5TL1 CRENEW | B5T CREVOKE | B5T CRENEW |
|---|---:|---:|---:|---:|
| 519.lbm_r | 6 | 2 | 8 | 0 |
| 505.mcf_r | 59,994 | 276 | 60,240 | 0 |
| 544.nab_r | 4,997 | 117 | 5,114 | 0 |
| 557.xz_r | 420 | 168 | 555 (cut) | 0 |
| 500.perlbench_r | 4,452,088 (cut) | 18,648 (cut) | 3,353,419 (cut) | 0 |
| 502.gcc_r | 2,880 | 237 | 3,117 | 0 |
| 538.imagick_r | 738 | 47 | 785 | 0 |
| 525.x264_r | 5 (cut) | 0 (cut) | 5 (cut) | 0 |

- **Every execution in every row has a run of 0.** Max 0, p99 0, none over 1,000, none over 20,000.
- The stages marked (cut) were ended at 600 s: B5TL1 perlbench and x264, and B5T xz, perlbench and x264, each rc 137.
  Their numbers cover the first 600 s. Their max read 0 at 300 s, at 590 s and at the end, so nothing was growing.
- CRENEW: 0 executions on every B5T stage, as predicted. B5TL1 has them (the mapping cache), every one with a run of 0.
- Privilege: U 7,944,295 and S 19,579 executions, M 0, as predicted. No CREVOKE walked nothing: nowalk 0.
- The boot log has one kernel soft-lockup warning (B5T lbm at 161 s, a 57 s stall of the single TCG vCPU) and no
  panic.
- The guest's "watchdog killed" echo appears for two of the five cut stages. The rc 137 at each @@SC-END is the
  evidence for all five.

## Reading, against rtl's mapping (amendment 1)

- A run of 0 is the operand's record plus the stop record.
  - 904cd9c6b: the check pass reads n + 1 = 1 record and the commit pass none.
  - 4(b): the forward scan reads 1 and the release reads none.
- This is far below the ~40,000-member warm bound and any cold factor. No B5b-2 SPEC row needs the long-walk fix to
  run.

What the count does not cover:
- workloads other than these SPEC test stages;
- the first 600 s limit on the five cut stages;
- the walk length on a bitstream whose CREVOKE_S visits records outside the run (amendment 1: it does not, on either
  family).

## Instrument identity

- capstone-qemu sublet/crevoke-scan-count fa6ce7554d, on 741be49bb0, on 3a9fc39167.
- The run's scripts (init, make.sh, drive.py, hist.py) stay with the run's local files, because they carry host paths.
