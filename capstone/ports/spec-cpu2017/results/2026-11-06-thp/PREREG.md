# mcf with 2 MiB transparent huge pages on a private kernel: pre-registration

Written and pushed before the board call (6 Nov, 11:00). The lead chose a separate kernel (6 Nov). The shared
buildroot, its kernel and the other lanes' kits are untouched.

## What changes

**Kernel.** A copy of each build tree, board 6.4.14 and QEMU 6.1, in perf/thp/linux-{board,qemu}, with
TRANSPARENT_HUGEPAGE in madvise mode (config-{board,qemu}.diff; the board tree's selection also enables COMPACTION).
- The board kit bkit29-thp relinks this kernel on each bake's own rootfs.cpio after the shared bake and before the
  firmware. A standalone relink test regenerated usr/initramfs_data.o.
- The control process-ABI module is rebuilt against the THP tree, 5fe2b197eac95237. The same sources against the
  stock tree reproduce the shipped f2a2645cc570dcc6.

**Module** (wt-et perf/zero-read → perf/thp: b23b38ce15b4, 37806776d76b), parameter `thp`, default off, set in the kit:
- a PMD-mapped page counts as resident at retire, so every 4 KiB page of a huge page is pinned, swept and unpinned;
- an arena whose VMA allows huge pages is admitted. Today a THP kernel requires VM_NOHUGEPAGE.
- Built: board 5973baaff48a17db against the THP tree (no 2-byte instructions), QEMU 9717b10d084a21b0.

**Launcher** (wt-fa perf/huge-heap 19bafe4ac4fe, on the quiet launcher): `madvise(MADV_HUGEPAGE)` on heap arenas of
4 MiB or more. Board build 959892e722a73f21, QEMU 75d219af742d1616; with the option off the build is byte-identical to
the quiet launcher 1d8b4f08.

**Safety argument.**
- Pinning one 4 KiB page of a PMD-mapped folio holds the folio: split and khugepaged collapse refuse a pinned folio.
- The context reaches only frames of folios with at least one pinned page.
- A retire pins and sweeps every resident page (mode 3).
- Teardown sweeps only the pinned pages. The rest are left to Linux's zeroing before reuse, as the no-sweep run showed
  for 4 KiB frames (results/2026-11-05-nosweep N1). huge-page zeroing (clear_huge_page) also stores; untested on
  silicon.
- The RTL supports 2 MiB leaves end to end (rtl-oracle, 6 Nov): PTW, TLB, physical address, CPMP and lifetime table.
- **Owed:** a performance-only RTL anomaly. The DTLB's `lsu_dtlb_ppn_o` splice for superpages may make the
  non-idempotent check stall some loads (UNRESOLVED; it would show as higher inside cycles).

**QEMU** (kit-proft, perf/xprof mcfH: the mcf test, Temporal, all2 parameters; qemu-mcfH.txt):
- rc 0, specdiff 2/2 PASS; thp_fault_alloc 743;
- context exits 32,510 → 7,716 and fault exits 26,499 → 1,663;
- peak pinned pages 174,291 → 59,789;
- kernel instructions in the window 2,443 M → 796 M (Plain's 738 M); module 356 → 366 M.

## The runs (board 2, after x264 Plain: driver pid 1655525)

1. **mcf, Temporal, THP** (`505.mcf_r-test-capstone-lifetimes-devmk-thp`): the all2 configuration, sweep 3,
   machine-wide snapshots. References:
   - 11,573.79 s, out-gtm-b1 (4 KiB pages);
   - 11,534.8 s, out-mx-b1;
   - Plain 9,094.9 s, out-mx-b2.
2. **mcf Plain on the THP kernel** (the perf/mx Plain stage). Plain does not madvise, so it stays on 4 KiB pages. This
   reading checks that the kernel change does not move Plain.

| # | Prediction | Point estimate |
|---|---|---|
| H0 | rc 0, specdiff PASS, control passes; @@THP reads `[madvise]` and thp=Y; thp_fault_alloc > 500 | |
| H1 | exits ≤ 15,000 (all2: 56,746), faults ≤ 5,000 | |
| H2 | outside time ≤ 1,500 s (all2: 2,258 s) | about 1,100-1,300 s: the resolve and pin cost (649 s) mostly goes, plus the shorter run's background |
| H3 | run ≤ 10,900 s = 1.20 × Plain | 9,800-10,700 s. The outside saving, plus DTLB misses (17/1k, 463 M) falling where the walk now hits 2 MiB leaves. The NI-splice stall could eat part of the inside saving |
| H4 | Plain on the THP kernel within ±1.5 % of 9,094.9 s | |

**What would refute what.**
- H1 holding with H2 failing: the outside cost was not the faults themselves.
- H3 failing with H2 holding: the inside time rose. Read the DTLB and issue-stall counters; this is the NI-splice
  question.
- H0 failing on thp_fault_alloc: the madvise did not reach the arena.

**Limits.** N = 1. A private kernel; Plain is re-measured on it. The board's huge-page path has not booted before:
the control rungs run first in the boot.

## Addendum 1 (6 Nov 15:15, before any board call of these runs): the order

- **What came first.** The driver (perf/thp/gt-run.sh, pid 2024109) was queued behind gcc train's
  arena-cache run, perf/ga2. That run, and the images it carries, changed after this pre-registration (arena
  Addenda 3-7). Nothing here depends on them.
- **What changes.** The waiting driver is killed before any board call. It now runs, unchanged and with the same
  stages, kit and grader, inside `perf/best/b2-seq.sh`. The new order on board 2: gcc train (ga2), then the rtl
  lane's R-32 SHRINKTO bare boot (one short boot on C.5a), then these runs, then perlbench in the best
  configuration (rest Addendum 2).
- **The predictions are unchanged.**

## Result, Temporal (6 Nov 23:24): 11,230.8 s, H0 holds; H1, H2 and H3 FAIL

The run moved to board 3 before its boot (arena Addendum 10); it ran there 20:04-23:22 on C.5a, kit bkit29-thp.
Graded with `thp-grade.py` (this folder). The reference out-gtm-b1 is board 1's: **this is a cross-board comparison**,
and Plain on the THP kernel (H4) runs next on board 3, which gives the same-board ratio.

| | 4 KiB (out-gtm-b1) | THP (out-thp-b3) | change |
|---|---|---|---|
| run | 11,573.8 s | 11,230.8 s | −343.0 s (−3.0 %) |
| inside | 9,315.8 s, 27.044 G instructions | 9,171.1 s, 27.042 G | −144.8 s |
| outside | 2,258.0 s | 2,059.7 s | −198.2 s |
| exits by kind [1, 3, 4] | 27,481 / 1,099 / 28,166 | 27,686 / 1,099 / 1,683 | fault exits −94 % |
| faults; peak pinned pages | 28,162; 174,293 | 1,679; 59,530 | |
| in-context DTLB / L1D misses per 1k | 17.12 / 140.35 | 16.71 / 138.60 | −2.4 % / −1.2 % |

- **H0 holds:** rc 0, specdiff 2/2 PASS, control passes, outputs equal the twin's; `@@THP always [madvise] never thp=Y`,
  thp_fault_alloc 742 (QEMU 743).
- **H1 FAILS:** faults 1,679 hold (≤ 5,000), but exits are 30,468. The bound left out the quantum exits (kind 1), 27,686,
  which this configuration's 80 ms step budget sets by run time; QEMU's 7,716 counts them in instructions.
- **H2 FAILS:** outside 2,059.7 s. Removing 26,483 fault exits saved 198 s, 7.5 ms each; per remaining exit the outside
  time rose from 39.8 to 67.6 ms. As with arena_pin on gcc (arena Result), outside time follows run time, not exits.
- **H3 FAILS:** 11,230.8 s = 1.235× Plain's 9,094.9 s (board 2), against the 1.20× bound of 10,900 s.
- **Inside fell 145 s, not the 400-650 s the plan expected.** The in-context DTLB miss count fell only 2.4 %: on this
  core 2 MiB pages barely change the L1 DTLB miss count, so the inside saving is not a DTLB walk saving at the
  predicted scale. Issue stalls per 1k fell 0.7 %. What the 145 s is made of is UNRESOLVED; the NI-splice stall the
  pre-registration names was not separated.
- Gap closed against the 4 KiB reference: 343 of 2,478.9 s, 14 %.
