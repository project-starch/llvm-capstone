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
| `image_rootfs=1` | imagick | 26.3 s | 32.2 s (ref), **−18 %** | outside 16.6 → 10.7 s; launch only −2.1 s of it |
| exact grant (launcher 0bde5fd8e0c9c374) | gcc | 42.8 s | 53.5 s (ctl), **−20 %** | pinned 6,118 → 2,125 pages; launch −4.45 s |
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
- **X3.** Unpin, dirty and free cost 0.178 ms, about 4,450 cycles per page. The sweep is 75 % of the teardown.
- **X4.** The 6.18 s left is about the pin of the padded image: the exact grant's 3,993 fewer pages saved 4.45 s,
  1.1 ms per page. The prediction counted the memfd copy and left out the pin. rootfs and the exact grant together
  should leave about 1.7 s.
- **X4b.** Command 1's launch fell 5.43 → 2.24 s. Command 2's rose 1.28 → 2.36 s; that is unexplained. The run fell
  5.9 s while the launch fell only 2.1 s. The other 3.8 s is outside `launch_ns`, and imagick has no same-board control.
  Its attribution is UNRESOLVED.
- **X5.** A two-point fit over the ref and q8 gives 3.56 ms per quantum exit, plus 152.7 s that does not scale with
  exits (13 % of the run). The longer quantum is the fix for the first part. The second is the next target. Plain
  loses about 9 % of its run time outside its own process too (outside RESULT, 4 Nov).
- **X6.** The prediction took the sweep at about 0.5 ms per page saved. X2 measured 0.22 ms.

## Instrument note: X0's quantum read (grader fixed before grading)

The first grade read X0 FAILS for sweep only, with every parameter correct. Its QNT0 line was split across two
driver.log chunks (`'...QNT0:'`, then `'001E8480\r\n'`), so the regex read nothing. The grader now reads QNT0 from
`board-raw.txt`, the reassembled console, taking the last occurrence. Controls:
- the same check with the q8 and stock values swapped reads X0 FAILS on all eight runs;
- q8 reads 007A1200 and the seven others 001E8480.

## Safety of each lever

- **`image_rootfs`.** It is safe by the criterion the module already applies. `one_inode_per_file` admits a filesystem
  that keeps one inode per file, so that `deny_write_access` sees every writer, and it already admits tmpfs. rootfs and
  ramfs keep one in-memory inode per file like tmpfs. The list missed them, and the launcher fell back to the sealed
  memfd copy at about 1 s per MB.
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
2. `image_rootfs` belongs in the default list: it changes no safety property.
3. Two costs are open:
   - nab's 153 s of outside time that does not scale with exits;
   - about 1.1 ms to pin a page and 0.18 ms to unpin it, before any sweep. xz pins about 130 k pages per command.
