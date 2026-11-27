# b5b2-ticklog-2026-11-27: where silicon places mcf's timer ticks (pre-registration)

Runtime lane, 2026-11-27, written before any boot of this kit's payload. The decision it serves is rtl's (the mcf
lifetime-buffer lever LB 1024x8, record-cache-model README); the rule below is rtl's, pasted with its agreement.

## The question

The lever's twin estimate removed 54.6 M misses, a population set by where the twin's ticks fall: every tick ends the
verdict cache for protected U, and the twin placed about 131 k of them by the host clock. Silicon's mcf stage takes
about 660 k ticks (HZ=100 periodic, 6,601.15 s), placed in cycle time. Re-placing the twin trace's flushes uniformly
at silicon's count removes about 17.9 M (NO BUILD). BUILD needs the instructions where the lever helps to run at no
more than about 0.55-0.6x the mean CPI. Only silicon's tick positions decide it.

## The instrument

- **Kernel:** a copy of trapk/fpga-v1 (the 165.03 G row's tree trapk/fpga; the two .configs differ only in
  CONFIG_INITRAMFS_SOURCE; HZ=100, HZ_PERIODIC, no NO_HZ). Source identity before the patch, sha256:
  Image c4853d98389bb598…, .config 76a23aa7189dfcc2…. Patch: apply-ticklog.py. At every timer interrupt the handler
  reads cycle, instret and hpmcounter3 and stores the three deltas from the previous interrupt (u32 each, 2^20
  records). Writing 0 to /proc/ticklog freezes it; reading returns a header and the records. No per-interrupt time
  CSR read: on this firmware it traps to M.
- **Firmware:** a clone of sbi-keep 25b14da6 (the commit and clean tree make-payload.sh builds) with
  sbi-tick-apply.py: mhpmevent3 = 2 (L1D miss, perf_counters.sv:115 at b42c8c644: one count per allocating cacheable
  miss on any D$ port, PTW and the lifetime unit's port included) and mcountinhibit[3] cleared, once, on the cold boot
  path; one console line `TICKFW mhpmevent3=… mcountinhibit=… trap=…` reads both back.
- **Overlay:** the g1b overlay's init, probes.list and smoke.dom unchanged, and three stages in this order:
  - TICK-ctl: 10 s idle (sleep), then a 60 s `awk 'BEGIN { for (;;) ; }'` busy loop;
  - B5TL1-505.mcf_r: the row's stage directory byte for byte (`diff -r` clean), mcf_r_base.native a1daec395a9e7260…;
  - TICK-dump: freeze, print the table's sha256 and size, then the table gzip'ed in base64.
- **Payload:** built by this tree's make-payload.sh. Its sha256 goes in an amendment before the boot.

## The reader (read-ticklog.py)

- The table is decoded from the capture with every foreign line removed (printk lines, /init's tick lines, whole or
  spliced). Its sha256 must equal the printed one; otherwise the boot is UNREADABLE, never a reading.
- Interrupt times come from cycles, calibrated against the header's two boot-clock snapshots.
- The window and F_k are as in the rule below; `--keep-list` writes the F_k for rtl's re-placement, one float per
  line in [0, 1): the window's total minstret is taken to the first interrupt after up1, so the last window tick's
  fraction is below 1.

## The rule (rtl, verbatim with the two agreed edits)

RULE (LB 1024x8, 505.mcf_r test, the 165.03 G row's mcf_r_base.native a1daec395a9e7260…, its byte-identical stage
directory and its kernel config (trapk/fpga, HZ=100 periodic), on a new payload whose sha256 is recorded in an
amendment before the boot):
- Window: the ticks whose boottime lies in [up0, up1] of the stage's own SPEC-BEGIN/SPEC-END lines. Silicon's N_s is
  the count of ticks in the window. F_k = (cumulative minstret at tick k, from the window's first tick) / (the
  window's total minstret).
- Re-placement onto the nested lane's ic7 trace (spec-mcf-ic7.1876342; 1,001,357 ticks at uniform instruction
  spacing, with the window's ticks taken between its own SPEC markers):
  - For each silicon tick k, keep ic7 tick floor(F_k × N_w), N_w = the ic7 ticks inside the trace's protected window
    (1,001,346 of 1,001,357). (rtl's revision of 2026-11-27, before this kit's boot: nearest-by-interpolated position
    gave 7,453 false collisions on evenly spaced targets, since the trace's kind-8 markers lie up to 2.2e8 instructions
    apart; under -icount ic7's ticks sit at constant instruction spacing, so a tick's rank is its instruction fraction.)
  - Clear the flush window (MTI through the MRET after the next cause-9 edge) of every ic7 tick not kept.
  - Run rcm-lb-dc --lb 1024x8, unchanged.
  - n = LB0_misses - LB misses.
- Collisions: silicon ticks mapping to an already-kept ic7 tick are counted and reported as C. Each one loses a
  flush, which biases n UP (toward BUILD), so the rule is conservative for NO BUILD.
- Verdict:
  - NO BUILD if n < 40,849,010 (1.6503 G / 40.4, all-cold).
  - Otherwise CANDIDATE BUILD, decided by the residency of the re-placed run's removed misses at warm 15.6 / cold
    40.4 against 1.6503 G.
  - VOID if any instrument control fails, or if N_s is outside THIS boot's (up1 - up0) x 100 ± 1 %.
  - The threshold 40,849,010 stays tied to the 165.03 G row's denominator. This boot's stage time is reported
    against 6,601.15 s and flagged above 2 %, without VOID; if flagged, n is reported against both 1 % of 165.03 G
    and 1 % of this boot's own cycles, and the verdict stays on the 165.03 G line.
- Controls:
  - the awk loop: ≥ 95 % of its ticks within ±3 % of its median;
  - mcycle delta: ≥ 99 % of the window's ticks within ±1 % of 250,000 (25 MHz / HZ);
  - idle ticks resolved: the idle window's median delta below 0.1x the loop's median;
  - a synthetic log with 10 % of its ticks at 2x the others, which the reader must find (1,000 ticks, 18.18 % of
    instructions in the 1.8x tail);
  - and the reader's own: the table's sha256 matches, nothing lost, the cycle clock within 0.1 % of 25 MHz.
- A model control, run before this pre-registration: thin_keep.py (capstone/tools/record-cache-model/mcf-tick-thin,
  dev c531aea8dbd0) on 667,564 evenly spaced F_k reproduces n = 17,917,437 exactly with 0 collisions; a clustered
  negative (1,000 F_k in a 0.1 % band) reports 355 collisions, as random placement predicts (results.txt).
- Reported, not graded: the histogram of delta / mean delta in 0.1 bins up to 4x plus overflow; the fraction of
  window instructions executed in ticks with delta ≥ 1.8x mean; C; per-tick CPI (mcycle / minstret) per bin; the
  window's hpmcounter3 total beside nested's model totals (ic6 3,699,876,396-3,699,882,854 misses over 9,558,342,792
  reads; ic7 3,799,654,954-3,799,676,759 over 10,183,136,384; runs/spec-mcf-ic{6,7}/window.out), whose difference
  estimates the PTW's and is not a bound.
- Limits: the mapping is by fraction of ALL instructions. Silicon's kernel and tick handler differ from the twin's, so
  position errors scale with the kernel's share, about 8 % on the twin (32.18 G all against 26.64 G protected on
  ic7). A mapping by user instructions is not available from minstret; stated, not run. N = 1 on both sides.

## Predictions, written before the boot

- If silicon places ticks as uniformly in instructions as the uniform model (the current best hypothesis): N_s ≈
  660,115, mean delta ≈ 44 k instructions, the ≥ 1.8x tail near 0 %, n ≈ 17.9 M: NO BUILD.
- The busy loop: one narrow peak. The firmware line reads mhpmevent3=2 with trap=0,0.
- The window's hpmcounter3 total above nested's model totals (it adds PTW and lifetime-unit misses).

## Disclosures

- mcf runs first here; in the row it ran after lbm, nab and xz. The boot history differs.
- The kernel differs from the row's by the patch only; the payload, firmware and initramfs differ.
- The dump takes about 15 min at 57,600 baud.

## The payload booted (identity, recorded before the boot)

- fw_payload.bin 247d7ea4a8a018d871c0ac7c174abc45b36cda6295608279401323772a3c8977 (17,463,304 B); Image
  26b6347715d49497…, initramfs.cpio ca3bf6d1c31a0ded…, overlay.cpio bda4b1d1b5fbf03f… (overlay-files.sha256 here).
- The tree's recipes differ from trapk/fpga-v1's by their paths only: W (both scripts) and the firmware tree
  (make-payload.sh:22). The .config differs only in CONFIG_INITRAMFS_SOURCE.
- The patched handler reads the three counters, by disassembly of timer-riscv.o: `rdcycle s3`, `rdinstret s2`,
  `csrr s1,hpmcounter3`.
- The overlay's stage scripts and order are in ovl-extra/. The mcf stage directory is `diff -r` identical to
  trapk/fpga/spec/B5TL1-g1b/ovl's; init is `cmp` identical.

## The twin dry run (before this pre-registration is pushed)

twin-dry.py boots this Image on qemu-crv with the stock fw-y firmware, as the campaign twins boot:
- rc 0 in 510 s; every stage ended; the kernel printed `ticklog: 1048576 records of 12 bytes` and, at the freeze,
  `n=50642 lost=0` (507 s at HZ=100);
- the reader decoded the table (sha256 02023428346df7d7… matched), cut the mcf window ([72.41, 506.35], N_s 43,353
  against 43,394 expected) and wrote 43,353 F_k in [0, 1);
- **expected to FAIL on the twin, and did:** the clock calibration, the mcycle control and the idle control. QEMU's
  cycle and instret both follow the host clock (2.6 GHz here), so idle and loop ticks read the same and nothing runs
  at 25 MHz. The awk-loop, N_s, table and synthetic controls PASS;
- **the splice controls:** four foreign lines inserted into the base64 block (a printk mid-line, a tick line
  mid-line, a whole printk line, a tick line at a line start) give a reading identical to the clean one, byte for
  byte. One flipped base64 character reads UNREADABLE (gzip error), and an unknown foreign token (`@@HB`) reads
  UNREADABLE (characters outside base64). Neither yields a reading;
- **not exercised:** the firmware change (the twin runs fw-y; fpga/ariane does not boot on QEMU). On the board its
  TICKFW line is read first. A trap there is recorded and does not stop the boot (csr_*_allowed); a zero or missing
  hpmcounter3 removes only the RC side reading, not the LB verdict.
