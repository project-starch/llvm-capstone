# Outside-time levers on board 2: result

Board 2, bitstream 3 (caplifive_vp2c_e7af48457, read back before the first chain), 5 Nov 04:33-07:37. Script
`xp-b2.sh` (pid 183714), pre-registered at ecdbae8e8187, Addendum 1 at a3b04f285666, Addendum 2 at c6d1bcaba1f0.
Eight chains, all rc 0. Temporal arm (`capstone-lifetimes`), N = 1 each. The reference is out-fa-b1 (board 1,
bitstream 3, fault-ahead 64, module 6e854c46c654c843). It is cross-board: ctl, the same configuration on board 2,
reads 53.5 s against its 53.3 s (1.004).

## Verdicts

| # | Prediction | Reading | Verdict |
|---|---|---|---|
| X0 | rc, specdiff, twin, control, params, QNT0 | all eight runs; QNT0 001E8480, q8 007A1200 | holds (after the grader fix below) |
| X1 | ctl teardown 0.4-1.2 ms/page | 0.722 ms/page | holds |
| X2 | store-only sweep ≤ 0.35 × ctl | 0.502 ms/page, 0.695 × | FAILS |
| X3 | no sweep ≤ 0.15 × ctl (report) | 0.178 ms/page, 0.246 × | FAILS |
| X4 | rootfs, gcc launch ≤ 5 s | 6.18 s (ctl 21.88 s) | FAILS |
| X4b | rootfs, imagick launch ≤ 2.9 s, both commands | 4.60 s (ref 6.71 s) | FAILS |
| X5 | quantum ×4, nab outside (report) | 165.9 s against 198.4 s | partial |
| X6 | all module levers, lbm run ≤ 0.985 × ref | 0.9855 × | FAILS by 0.05 % |
| X6b | lbm faults ≤ 2,500 | 2,206 (ref 6,211) | holds |
| X7 | exact grant, gcc: peak ≤ 2,700, launch ≥ 3 s and run ≥ 5 s below ctl | peak 2,155; launch −4.45 s; run −10.7 s | holds |

## What each lever buys (run time, Temporal)

| Lever | Stage | Run | Against | Where the time went |
|---|---|---|---|---|
| `image_rootfs=1` | gcc | 36.7 s | 53.5 s (ctl), **−31 %** | launch 21.88 → 6.18 s; outside 45.6 → 28.7 s |
| `image_rootfs=1` | imagick | 26.3 s | 32.2 s (ref), **−18 %** | MapLife arm, same boot: run 32.6 → 27.8 s (−15 %), launch −4.5 s of it |
| exact grant (launcher 0bde5fd8e0c9c374) | gcc | 42.8 s | 53.5 s (ctl), **−20 %** | pinned 6,118 → 2,125 pages; launch −4.44 s |
| quantum 8,000,000 (firmware) | nab | 1,138.8 s | 1,183.8 s (ref), **−3.8 %** | steps 12,826 → 3,692; outside −32.5 s, inside −12.6 s |
| `retire_exit_sweep=3` | gcc | 52.1 s | 53.5 s (ctl), −2.6 % | teardown 0.722 → 0.502 ms/page |
| sweep 3 + rootfs + `step_budget_ns=80000000` | lbm | 1,934.6 s | 1,963.0 s (ref), −1.4 % | faults 6,211 → 2,206; steps 24,000 → 20,005 |
| `retire_exit_sweep=2` (measurement only) | gcc | 48.0 s | 53.5 s (ctl) | teardown 0.178 ms/page |

Inside time is unchanged in every run (gcc 7.9-8.0 s, imagick 15.6 s, lbm 1,453.0 against 1,453.5 s), except nab's
−1.3 %, which is about 9,100 fewer resumes.

## Readings behind the failed predictions

- **X2, the store-only sweep.** The sweep is 0.722 − 0.178 = 0.544 ms of the teardown's 0.722 ms per page. Store-only
  costs 0.502 − 0.178 = 0.324 ms: 256 stores, about 32 cycles each. The prediction took the stores as posted. On this
  write-through D-cache each store costs about as much as a memory write. The store-only sweep removes 40 % of the
  sweep, not two thirds.
- **X3.** What is left without a sweep costs 0.178 ms, about 4,450 cycles per page. It covers unpin, dirty and free,
  plus each arena's retire op and kvfree. The sweep is 75 % of the teardown, if the two costs add.
- **X4.** The 6.18 s left is about the pin of the padded image: the exact grant's 3,993 fewer pages saved 4.45 s,
  1.1 ms per page. The prediction counted the memfd copy and left out the pin. rootfs and the exact grant together
  should leave about 1.7 s.
- **X4b.** It fails in the Temporal arm the grader reads, but the lever works on imagick.
  - Temporal arm: command 1's launch fell 5.43 → 2.24 s, and command 2's rose 1.28 → 2.36 s, which is unexplained. The
    run fell 5.9 s while the launch fell 2.1 s, so 3.8 s lies outside `launch_ns`.
  - MapLife arm of the same boot (audit): launch 1.22 + 1.01 s against 5.46 + 1.31 s, so −4.5 s. The run fell 4.8 s,
    32.6 → 27.8 s, leaving 0.3 s.
  - So rootfs's imagick gain is the launch, about −15 %. The Temporal arm's 3.8 s residue and command 2's rise belong to
    that arm in this boot, and their cause is UNRESOLVED.
  - There is no same-bitstream control on board 2. Board-1 Temporal imagick runs span 32.2-33.3 s.
- **X5.** A two-point fit over the ref and q8 gives a cost per quantum exit and a part that does not scale with exits.
  - Temporal: 3.56 ms per exit plus 152.7 s, 13 % of the run. MapLife, the same two boots: 3.10 ms plus 153.7 s.
    So the slope is known to about ±13 %, from two cross-board runs.
  - Inside fell 12.6 s for 9,134 fewer exits, about 34 k cycles each. That they are resume costs is a reading, not a
    measurement.
  - The longer quantum is the fix for the per-exit part. The fixed part may be mostly Plain's background, by the rate
    below. That is an inference.
  - Plain nab's own times stage (out-hfl, `times` around the command): real 867.3 s, the command's user + sys 788.0 s.
    So 79 s, 9.1 %, is spent outside the process.
  - Every long Plain times stage sits at 8.7-9.3 % (mcf 8.86 %, lbm 9.09 %, xz's 12 commands 8.7-9.3 %). Short
    stages read higher: gcc 16.6 %, imagick 20.5 %.
  - At that rate Capstone's 1,138.8 s would carry about 104 s, leaving about 49 s, 4 %, as Capstone's own.
  - Applying Plain's rate is untested. In Capstone, interrupts are masked for a whole quantum. A consumer driven by
    timers whose missed periods are not replayed would scale with exits instead.
  - What would settle it: per-task runtime (`/proc/<pid>/schedstat`) and `/proc/stat` idle against other-task time,
    read in a Plain boot and in a Capstone boot.
  - The board kernel is HZ_PERIODIC at HZ=100, without HIGH_RES_TIMERS, on a oneshot-only RISC-V timer. After each
    masked quantum, `tick_handle_periodic` replays every missed period, one `tick_periodic` each. The tick count per
    second is therefore Plain's, and the replay is not a Capstone excess by count.
- **X6.** The prediction took the sweep at about 0.5 ms per page saved. X2 measured 0.22 ms.

## Instrument note: X0's quantum read (grader fixed after the first grade)

The first grade (grade-v1.txt) read X0 FAILS for sweep only, with every parameter correct. Its QNT0 line was split across two
driver.log chunks (`'...QNT0:'`, then `'001E8480\r\n'`), so the regex read nothing. The grader now reads QNT0 from
`board-raw.txt`, the reassembled console, taking the last occurrence. Controls:
- the same check with the q8 and stock values swapped reads X0 FAILS on all eight runs;
- q8 reads 007A1200 and the seven others 001E8480;
- the diff from grade-v1.txt to grade.txt is that one line;
- each run's board-raw.txt has exactly one QNT0, after that boot's OpenSBI banner (audit). The console replay of the
  previous boot carries none.

## Safety of each lever

- **`image_rootfs`.** It is consistent with the module's own criterion, from the source; no negative test has run.
  - The criterion: `one_inode_per_file` admits a filesystem that keeps one inode per file, so `deny_write_access` sees
    every writer. It already admits tmpfs.
  - On this board rootfs is tmpfs code. CONFIG_TMPFS=y, and there is no `root=` in the command line, so
    `init_rootfs` sets `is_tmpfs` (init/do_mounts.c).
  - What still holds (audit):
    - hard links share the inode;
    - overlayfs is not built;
    - truncate, O_TRUNC, writable mmaps and fallocate all need write access, which the denial excludes;
    - `capstone_application_fixed` (application-image.c) demands seals only on unlinked files, and a named rootfs
      file is linked;
    - sweep 3 cannot touch the image's page cache, because text is not pinned and `pin_remaining` refuses non-anonymous
      pages.
  - Not yet tested: opening the image for writing while it runs should fail with ETXTBSY. It needs one board probe.
  - Before this switch the list missed rootfs, and the launcher fell back to the sealed memfd copy at about 1 s per MB.
- **Exact grant.** It grants the image at its smallest exactly representable length, page-rounded, on a granule-aligned
  base. QEMU gate kit-egq: six of six PASS.
- **Quantum 8,000,000.** That is 320 ms with interrupts masked, against 80 ms today. Neither drains the UART RX FIFO
  while the context runs. The driver waits for each prompt, so neither bit in these runs.
- **`retire_exit_sweep=3`.** It clears every tag, as the default sweep does, with stores only. QEMU gate kit-rsq3: six
  of six PASS.
- **`retire_exit_sweep=2`** leaves tags in returned frames. It is a measurement, not a candidate.

## What follows

1. All the safe levers together run on board 1 next: rootfs, exact grant, sweep 3, step budget 80 ms and quantum
   8,000,000, with fault-ahead 64. They run on the devmk images on C.5a, against out-all-b1: the same board, bitstream
   and images. Pre-registered separately.
2. `image_rootfs` belongs in the default list once the ETXTBSY probe passes. By the source it changes no safety
   property.
3. Two costs are open:
   - nab's 153 s of outside time that does not scale with exits;
   - about 1.1 ms to pin a page and 0.18 ms to unpin it, before any sweep. xz pins about 130 k pages per command.
