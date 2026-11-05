# all2 without the exit tag sweep, boards 1 and 2: pre-registration

Written and pushed before any board call (5 Nov, 14:10). Driver `ns-run.sh`, grader `ns-grade.py`, probe verdict
`tl-verdict.py`, lists `probe.list`, `b1-rest.list`, `b2-rest.list`, the probe's source and stage script in `tagleak/`,
the dry run in `dryrun.txt`.

## Why

After all three levers (results/2026-11-04-all) most of the remaining gap to Plain is time outside the program. On
xz the exit teardown alone is 1.63 M pages a run at 0.50 ms (sweep 3), of which the sweep is 0.32 ms: about 530 s of
xz's 1,338 s remaining gap. outside-levers measured the teardown with no sweep (mode 2) at 0.178 ms per page.

## The claim this run tests: the exit sweep is redundant

A frame a context releases at exit goes back to Linux, and every route by which a frame becomes readable in a context
again overwrites the whole page with plain stores first. A plain store clears its granule's tag.
- **RTL** (rtl-oracle, read from the 1fc2cd5aa tree; the store path is identical to e7af48457): `sb/sh/sw/sd` and FP
  stores write ctag 0 for their 16-byte granule, in every privilege mode and inside and outside a context
  (store_unit.sv:389-393, wt_dcache_wbuffer.sv:754-756, wt_axi_adapter.sv:156, :416-418, wt_dcache_mem.sv:507-512).
- **Kernel** (vanilla 6.4.14 on the board):
  - anonymous faults go through clear_page; COW goes through memcpy; tmpfs/rootfs pages are cleared or zero-filled;
  - clear_page takes cbo.zero only with zicboz in the device tree, and the board's says `rv64imafdc`, so it runs
    `__memset`, an sd loop;
  - THP, KSM and init_on_free are off.
- **Module**:
  - arenas are anonymous with mapcount 1 (capstone_vm.c:1123); the image text is never writable;
  - nothing checks that freed frames are tag-free at exit;
  - the design doc already relies on next-use zeroing for unpinned resident pages (runtime.md:154-156: "Linux's
    ordinary zeroing clears residual tags when untracked backing is reused").
- **Exceptions:**
  - AMO and `sc` writes leave a tag untouched (a known RTL residual). They cannot set one, so a page already zeroed
    stays clean.
  - Bus masters other than the CPU do not clear tags. This board has no DMA into user memory, and the JTAG load
    happens before boot.
- **Gate weakness.** Mode 2 skips the sweep when the releasing task has PF_EXITING, not when the context's mm is
  dead. Today the launcher guarantees the two coincide: it opens the device O_CLOEXEC after its only fork, and the
  owner's exit_mm runs before exit_files. A production version should gate on `mm_users == 0`. Non-exiting releases
  and CV_RETIRE keep their sweep in mode 2 (mode 0 there).

## The probe (board 1, first, gates everything)

Stage 999.tagleak, one boot on the kit below, before any SPEC stage. For sweep 3 and then sweep 2 (set through sysfs;
the kit value is restored afterwards):
- `tagleak fill 128` stores a capability in every granule of 128 MB and records the block's PFNs;
- `tagleak scan 128` takes a fresh 128 MB block and counts tagged granules;
- it counts how many of its frames the fill held, then stores one capability of its own and reads its tag back.

| # | Prediction |
|---|---|
| N1 | In both pairs: the fill reads a tag in every sampled granule; at least half of the scan's frames are the fill's (the test is loaded); the scan reads 0 tagged granules, 0 in recycled frames; its own tag reads 1. **Gates both boards' SPEC stages.** |
| N2 | ETXTBSY (image_rootfs): while the image runs, opening it for writing fails; once it has exited, the same open succeeds. Reported; gates nothing. |

QEMU (kit-egq, this stage, 5 Nov 13:58, `tagleak/qemu-q3.txt`):
- sweep 2: 32,422 of 32,768 frames recycled, 0 tagged, ctl 1;
- sweep 3: the same;
- N2 cannot be read there: the stage directory is a read-only 9p mount.

## The SPEC run

- **Kit bkit24-ns** is bkit23-all2-b1 (results/2026-11-05-all2) with one params line changed: `retire_exit_sweep=2`.
  - Module 2f81e8a5842010b3, launcher 0bde5fd8e0c9c374, firmware quantum 8,000,000: all unchanged.
  - The devmk images, on C.5a.
  - The SHA256SUMS differ in that one line.
- **Reference: out-all2-b1** (board 1). Same bitstream, images, module, launcher and firmware.
- **Board 1**, after all2-b1.sh (pid 548710) ends with DONE: the probe, then lbm, mcf, gcc. These are the small
  effects, so they run on the reference's own board.
- **Board 2**, after dt-b2.sh ends with DONE and once N1 holds: imagick, nab, xz.
  - xz is cross-board; board 2 against board 1 read 1.004 on a 53 s stage (outside-levers).
  - That is far below xz's predicted effect.
- Each board flashes C.5a before and bitstream 3 after.

| # | Prediction (every arm against the same arm of out-all2-b1) | Point estimate |
|---|---|---|
| Z0 | rc 0, specdiff PASS, outputs = twin, control, fresh; the four params read back (retire_exit_sweep=2); QNT0 007A1200; instret within 0.1 %; every teardown line of the arm reads sweep=2 | |
| Z1 | teardown ≤ 0.25 ms per page, every arm with ≥ 10,000 pages | 0.18 (reference about 0.50) |
| Z2 | xz run ≥ 300 s below | −528 s: 1.63 M pages × 0.324 ms |
| Z3 | lbm ≥ 15 s below; mcf ≥ 25 s below | lbm −34 s (105 k pages), mcf −56 s (174 k pages) |
| Z4 | gcc, imagick, nab: run within 2 % | about 0 (≤ 2,250 pages) |

**Reported:** the share of the gap to Plain closed (baseline 28-29 Oct, Plain test), beside all2's.

**What would refute what.**
- **N1 fails on a tagged recycled frame.** A frame reaches a context still tagged, so the sweep is needed. No SPEC
  stage runs.
- **N1 fails on loading or ctl.** The probe says nothing, and no SPEC stage runs either.
- **Z1 holds and Z2 fails.** The teardown time is not where xz's outside time goes.

**Grader controls** (`dryrun.txt`):
- the mock built from the predictions reads rc 0;
- out-all2-b1's own runs given as the run read Z0 FAILS, rc 1;
- an empty folder reads rc 2;
- xz at −200 s reads Z2 FAILS;
- tl-verdict.py fires both ways.

**Limits.** N = 1. Test inputs. The probe exercises the anonymous-fault route only: the one the module's arenas use,
and the one 1.63 M pages a run take on xz.
