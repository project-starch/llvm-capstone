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

## Addendum 1 (5 Nov 16:30, before any board call; both drivers stopped while still waiting, relaunched after this push)

Three changes, to get results sooner. The claim, the probe and Z0-Z4 are unchanged except where named.
1. **The probe runs on board 2.** Board 2 frees first: stage A's last stage, mcf, ends about 18:35. all2's mcf on
   board 1 ends about 19:25. Board 2 runs the probe, xz and gcc. Board 1 waits for the verdict, then runs lbm, imagick,
   nab and nab-b10s.
   - The roles are a parameter, `NS_PROBE_BOARD`, default 2.
   - xz now compares on the same board as its reference, which is better. lbm stays on board 1.
2. **mcf is dropped.** Its predicted −56 s sits inside its ±36 s board noise, for 3 h of board time. Z3 now covers lbm
   only.
3. **Kit bkit25-nset = bkit24-ns with the exit-timing module.**
   - Module 1ae19c5e337fbeb2 = wt-et perf/exit-timing-vp2 68373bbdbcc2 (0 two-byte instructions). It is 2188cab26d40
     plus two commits:
     - 0a7f0e8aecaa: counters in the BENCH_COUNTERS block, measurement only;
     - 68373bbdbcc2: `resume_quantum`, a 0644 parameter, default off. Off, the step loop is unchanged.
   - The control build from 2188cab26d40 by the same recipe has a `.text` byte-identical to 2f81e8a5842010b3's; only
     the source-path string differs.
   - QEMU gate of 0a7f0e8aecaa, kit-etq (kit-egq with its QEMU build 3eb6f86f37a5a1ed): gcc, imagick, lbm, nab, xz and
     mcf all rc 0 and specdiff PASS, with one `capstone_vm: exits` line per context.
   - QEMU gate of 68373bbdbcc2, kit-etq2 (66ef877000d232db): see Addendum 2.
   - The counters add about ten CSR reads per exit, against an exit of about 5 ms. Z0's instret check still holds the
     program identical.
   - The probe stage runs first on this module, so it is also the module's first silicon contexts. A module that broke
     contexts would fail N1, and nothing else would run.
   - **Reported per arm:** cycles per CSRUNV exit outside the context, split into the CV_STEP kernel time (fault
     resolution part of it), the time between CV_STEPs (launcher, its calls for the context, interrupts, other tasks)
     and other ioctls.

**A second hypothesis in the same session: B1, quantum exits that do not return to the launcher.**
- Today a quantum exit (event kind 1) always returns to the launcher, and the launcher answers it with `sched_yield()`
  and the next CV_STEP (exec.c virtual_service_loop).
- A 10 s step budget alone does not change that. Under QEMU, nab took 1,746 CV_STEPs with and without it (perf/ns/q-b10s),
  because the loop resumes only fault events (capstone_vm.c `if (t->event != 4) break;`). This was found before any
  board call, and that variant is withdrawn.
- `resume_quantum=1` resumes event 1 inside CV_STEP. The stop check before every entry still returns it to the
  launcher on a resched, a signal or the budget.
- The stage: 544.nab_r-test-devmk-rq, the same images. Its run.sh sets `resume_quantum=1` and
  `step_budget_ns=10000000000` through sysfs and restores both at the end. It runs on board 1 right after nab.
- B1: CV_STEP ioctls ≤ 0.6 × nab's, and quantum exits within 10 % of nab's. The run-time difference is reported, not
  graded.
- What B1 decides: whether the return to the launcher, per quantum exit, is worth removing by default.

## Addendum 2 (5 Nov 16:45, before any board call)

**QEMU gate of module 68373bbdbcc2** (kit-etq2, 66ef877000d232db):
- **resume_quantum on** (the rq stage): CV_STEPs 589 against 1,746 for nab in the same kit with it off (0.34 ×),
  1,369 against 1,762 in the Temporal arm. The quantum exits are the same (1,380 and 1,369). rc 0 and specdiff PASS
  in both arms, and the outputs match across the arms. The sysfs values read back and are restored.
- **resume_quantum off:** gcc and lbm read the same exit counts as module 0a7f0e8aecaa, with specdiff PASS.
- **B1 grader:**
  - on these QEMU runs: B1 holds 2/2;
  - given nab as its own rq run: B1 FAILS 2/2.

**Seen on board 1 since the pre-registration, and relevant to the xz stage here.**
- all2's xz (results/2026-11-05-all2, kit bkit23) stopped after 9 of its 12 commands.
  - The console went quiet after heartbeat 182, and the driver dumped the vbatch wedge apertures (sw=225 0x80,
    sw=228 0x80). The stage read NO DATA (completed=False).
  - The same images on C.5a without the five switches (all-b1) finished all 12 commands.
  - Nine teardowns read 0.497 ms per page, sweep 3.
- Cause not yet read. N = 1.
- This run's xz, on board 2 with the same switches and no sweep, is a second sample of the same configuration. If it
  wedges too, that is reported as a finding about the switches together on C.5a, not as a Z2 reading.

## Addendum 3 (5 Nov 17:50, before any board call of this run)

**The drivers were stopped and the run follows gcc train.** The lead made gcc train (results/2026-11-05-gcctrain)
the top priority. The waiting drivers had already stopped when the sessions they waited on were killed. They are
relaunched to wait on gcc train's drivers (gt-run.sh), one per board.

**The module is 18-bit.**
- Lead rule: no 16-bit module on bitstream 3 or C.5a. Module 645749816c02751e is the exit-timing and resume_quantum
  source (perf/exit-timing-vp2 3fbed8df1fb8; hpm_snapshot compiled out) built with `-DCAPSTONE_VM_INDEX_BITS=18`.
  That is a 262,144-record flat table, against the 16-bit 65,536 of every reference run here.
- QEMU (kit-etq2b18, 10672b9695c39db9):
  - gcc test: rc 0, specdiff PASS, `node_capacity=262144`;
  - the rq stage: both arms PASS, CV_STEPs 588 and 617 against nab's 1,746 and 1,762.
- The 4 MiB table is zeroed at each context's creation. Against the 16-bit reference that is a launch-time
  difference, read from `launch_ns`.
- Z0's instret check still holds the program identical.

**The probe board is board 1** (`NS_PROBE_BOARD=1`). Board 1 frees first: its gcc train run is Plain, about 0.5 h.
- Board 1: the probe, then xz. xz is the large predicted effect, about 2.5 h.
- Board 2, after its gcc train run (about 19:15): lbm, imagick, nab, nab-rq, gcc.
- The reference for every stage stays out-all2-b1. all2's xz has no data (Addendum 2), so Z2 compares xz against
  all-b1 (out-all-b1, the same images on C.5a without the five switches). The switches are then a second difference
  on xz, so Z2 reads as "no sweep plus the switches against neither".

## Addendum 4 (5 Nov 19:05, after the probe's first board run)

**The probe's first run on board 1 (18:18-18:33, kit bkit25-nset, 18-bit module) gave N2 and no N1.**
Transcript lines in `tagleak/board1-1831.txt`.
- **N2 holds on silicon.** While the image ran, the open for writing failed ("Text file busy"); after it exited, the
  same open succeeded. That was so in both pairs. `image_rootfs` keeps Linux's write denial on this board.
- **N1 read NO DATA** (verdict rc 2), so both boards' no-sweep stages stopped, as designed. The board kernel has no
  `/proc/self/pagemap` (`# CONFIG_PROC_PAGE_MONITOR is not set`), while the QEMU kernel has it. The probe then exited
  before printing its tag count.
  - This was a probe design slip. Its QEMU check ran on a kernel configured differently from the board's.
- **The probe is reworked.** It always prints the tag count.
  - Without pagemap, the loading proof is by counting. The block is 60 % of MemFree, so the fill and the scan
    together exceed the free memory. That forces at least fill + scan − MemFree before the scan of the scan's frames
    to have been the fill's.
  - `tl-verdict.py` takes that bound as the load, and requires it to be at least a quarter of the scan's pages.
  - The PFN check stays where pagemap exists.
- **QEMU in the forced counting mode** (`tagleak/qemu-q4-nopm.txt`): 529 MB blocks, forced reuse ≥ 45,317 of 135,424
  pages, 0 tags in the scan, ctl 1, the fill tagged in every sampled granule, both pairs: N1 HOLDS.

**The boards.**
- **Board 2** runs the probe first, then xz, after its gcc train Temporal run (`NS_PROBE_BOARD=2`).
- **Board 1** runs mcf on the counter kit (results/2026-11-05-gcctrain Addendum 4), then lbm, imagick, nab, nab-rq
  and gcc once the verdict holds.
- The failed run's verdict file and output are kept, with a `.nopagemap-1833` suffix.

## Addendum 5 (5 Nov 20:10): N1 holds on silicon

**The probe on board 2** (19:37-20:06, kit bkit25-nset, 18-bit module, C.5a). Transcript lines in
`tagleak/board2-2006.txt`, verdict in `tagleak/verdict-board2-2006.txt`.

| Pair | Fill | Forced reuse in the scan | Tags in the scan | ctl | N2 |
|---|---|---|---|---|---|
| sweep 3 | 371,596 / 371,596 sampled granules tagged | ≥ 47,231 of 140,800 pages | 0 | 1 | DENIED / WRITABLE |
| sweep 2 | 371,596 / 371,596 | ≥ 47,129 of 140,800 pages | 0 | 1 | DENIED / WRITABLE |

**N1 HOLDS and N2 HOLDS.**
- With no exit sweep, a context that takes at least 47 k frames which a previous context left full of capabilities
  reads not one tagged granule in them.
- The kernel's next-use overwrite clears the tags on silicon, as the RTL and source reading said.

**Two driver defects, fixed.**
- The driver read the verdict from `999.tagleak-test`, the first probe's name, and so wrote `N1 FAILS rc=1` before
  stopping board 2. The verdict now names the stage from probe.list. The real verdict, read from the probe's own
  transcript, is the one in `verdict-N1.txt`; the wrong file is kept as `.wrongpath-2006`.
- Earlier at 19:29-19:34, a console-lock timeout on the C.5a flash stopped the first try. The drivers now retry a
  failed flash once; the retry was dry-run. The same no-wait DONE check as gt-run's was also fixed in ns-run.

**Board 2 now runs xz** behind the verdict, with output in out-ns-b2x. Board 1 runs its list after its mcf.
