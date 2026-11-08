# QEMU runs for the NO_HZ_IDLE lever (8 Nov): function on the variant's tick mode, and the mechanism at HZ=1000

All under the shared qemu.lock, domain slot prof-slot-76, one after another. Kits: copies of spec/kit-profq (module
1c29ae65c8d3c57b, launcher 9251005fcc197743, firmware and rootfs unchanged; link count 1, SHA256SUMS in each) with only
the kernel Image replaced and the wrapper's `-icount` shift set. Kernels: private copies of the QEMU twin's 6.1 tree
(kit Image e58613598c897103), config diffs beside this file.

**Why the module needs no rebuild.** c3f's source built with kit-profq's recipe against the stock QEMU tree reproduces
1c29ae65's .text/.rodata/.data; built against the qb, qn and qb1k trees it differs from 1c29ae65 only in
`.rodata.str1.8` (the source path) and the build-id. The same holds on the board (PREREG-nohz.md).

**What QEMU cannot show.** QEMU's quantum is counted on the cycle CSR, which follows the icount clock, so a masked window
stays under ~10 ms of guest time at any shift (shift 5: 84,679 quantum exits against 1,330 at shift 0). At HZ=100 no
QEMU run replays more than one period, so the board's saving cannot be measured on QEMU at the board's HZ.

## Stages

perf/exitk/stages: `gtr-c3f-tk` (the board's c3f gcc train stage, perf/ga2, with its resume_quantum / 10 s budget /
sweep 2 block) and `perl-c3g-tk` (perf/best's perlbench test c3g stage), each with `@@TL` (/proc/timer_list's
event_handler) and `@@TK0/1` (/proc/stat cpu, intr, ctxt, uptime) around each command. Read with `tick-read.py`.
References: perf/best out/gtr-c3f and q-perl-c3g (kit-profq, stock QEMU kernel).

## Results

| run | kernel (Image) | shift | rc / specdiff | outputs vs reference | @@TL | cpu ticks / wall |
|---|---|---|---|---|---|---|
| qb-gtr | 6.1 HZ=100 HZ_PERIODIC, no HRES (402399ae) | 5 | 0 / 1/1 PASS | .s .out equal | tick_handle_periodic | 100.0 % |
| qn-gtr | 6.1 HZ=100 NO_HZ_IDLE, no HRES (4110e405) | 5 | 0 / 1/1 PASS | .s .out equal | nohz_mode 1, tick_nohz_handler | 99.9 % |
| qn-perl | the same | 5 | 0 / 2/2 PASS | all 560 hashes but .err equal to q-perl-c3g | nohz_mode 1, tick_nohz_handler | 99.9 % |
| qb1k-gtr | 6.1 HZ=1000 HZ_PERIODIC, no HRES (94d68047) | 0 | 0 / 1/1 PASS | equal | tick_handle_periodic | 100.0 % |
| qn1k-gtr | 6.1 HZ=1000 NO_HZ_IDLE, no HRES (df3d8e6e) | 0 | 0 / 1/1 PASS | equal | nohz_mode 1, tick_nohz_handler | **43.2 %** |

(/proc/stat counts in USER_HZ = 100 at either HZ.) At shift 5 every period is its own interrupt (qb-gtr: intr +47,271
for 47,163 ticks), so the qb/qn pair tests function and the readback, not the replay.

**The mechanism, HZ=1000, shift 0** (a ~10 ms quantum then spans ~10 periods):
- qb1k accounts every period (1,455 cs over 14.55 s); qn1k accounts 628 cs: about 8,270 periods were caught up in one
  step instead of replayed.
- Kernel window `kcyc - rcyc`: 46.25 M → 31.21 M instructions (−15.04 M, −33 %), about 1.8 k instructions per period
  not replayed (6.1 with debug spinlocks; the board estimate is ~2.2 k at the window's IPC, ACCOUNTING.md).
- Machine-wide (GT-SNAP cycle = instructions): 14.515 G → 14.507 G (−8.3 M). The kernel window's 15.0 M is not all
  net: qn1k shows more context switches (ctxt +3,293 against +1,689 over the command). ucyc 116.98 → 115.54 M.
- So on QEMU the variant removes the replay from the kernel window, and about half of it from the machine total. On the
  board the replay costs 13.8 k cycles per period against an unknown amount of displaced work; PREREG-nohz N1 (kernel
  window) and N4 (run) read the two separately.

**A cost of the nohz path, recorded.** At shift 5 (no replay), qn's kernel window is +7.5 % against qb's (6.781 against
6.311 G icount units), about 310 more instructions per tick interrupt on this 6.1 kernel. On the board's Plain at
100 interrupts a second that would be ~0.75 % if it transferred at ~6 cycles an instruction: inside PREREG-nohz P1's
±1.5 %, but it is the reason P1 exists.
