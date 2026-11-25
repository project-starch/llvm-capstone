# b5b2-linux-2026-11-25: results

Linux boots on the B5b-2 bitstream (capstone-ariane 904cd9c6b, board 2, no flash), on the trap-route stack:
- the integer-M firmware: caplifive-opensbi trap-route-keep-tags eb85e21c;
- the collaborator's trap-route kernel: series at 6d4504f8b073;
- the board DTS with the `capstone,unpinned-v1` node.

The trap-route test's registers keep their references across machine traps on silicon. With the same image under the
stock firmware they do not. The board lane's records, with the result lines and graders, are board/meas-pw 819ab44207c6
(boot 2) and b319f11ec296 (boots 3 and 4), under `capstone/tests/rtl-smoke/b5b2-linux-board-2026-11-25/`.

| Boot | Image | Firmware | Readings | Against the PREREG |
|---|---|---|---|---|
| 1 | `0e36edbb…` | keep-tags | the banner line; MEDELEG `0x7f00b109`; Linux to `/init` at 19.47 s; then nothing (no `/dev/console` in the initramfs) | P1-P3 as predicted; P4 NO READING, a defect of the image (Amendment 1) |
| 2 | `54d11b89…` | keep-tags | P1-P3 plus `.dom loader registered`; the test loader protected pids 26 and 27; `regs status 0`; `enter status 200`; `done` | P1-P3 as predicted; P4 missed its twin-based prediction, because the twin was wrong (Amendment 2) |
| 3 | `77de789f…` | keep-tags | `regs status 0`; `regs-strip23 status 23`; `enter status 200` | P1-P4 as predicted; the checker's control fires |
| 4 | `c1698787…` | stock (flag off) | no banner line; MEDELEG `0xb109`; `regs status 139` and `regs-strip23 status 139` (SIGSEGV); kernel: `sublet: fault cause=24 pc=0x10218` for both, the first capability op on a1 after the timer-tick spin; `enter status 200` | P1-P4 as predicted; the firmware's control |

**What this establishes.**
1. Linux boots on 904cd9c6b with this stack. P2's MEDELEG equals OpenSBI's set plus causes 24-30 under the RTL's
   mask.
2. regs.S passes on silicon: 29 registers each still hold a reference of A's lifetime after two 40M-iteration
   timer-tick spins, a getpid call, a page fault from U and the test hook's call. Its check can fire: the x23 strip
   reads 23.
3. The keep-tags firmware is necessary for it. The same kernel and initramfs under the stock firmware fault with
   cause 24 at the first capability use after the spin. That the tag was lost at an M trap is the mechanism the
   amendment names, consistent with the fault site. It is not itself observed.

**Retraction** (Amendment 2, 90f2f5bc5e83): the twin's EPERM was not "a kernel or test issue". The twin's initramfs
had been repacked as uid 1000, which made the setuid busybox run /init's shell with euid 1000. The twin's own `id`
line, `uid=0(root) gid=0(root) euid=1000`, confirms it. The twin harness now appends a root-owned overlay to the
unchanged rootfs, as the board images always did.

**Not covered:** SMP (the kernel is !SMP); Sstc (absent); a fault from M; the stock firmware with protected
programs other than regs.S.

## Boots 7 and 8: the cost of an M round trip from U (Amendment 4)

Board 2, resident 88db40027, no flash. The kernel and initramfs are the same; only the firmware differs. Cycles per op,
best of 5 x 20,000, three runs per boot:

| | rdtime (one M trip) | rdcycle (no trap) | getppid (U-S syscall) |
|---|---|---|---|
| boot 7, keep-tags | 1929.87 / 1932.33 / 1943.72 | 5.00 x 3 | 1287.47 / 1287.71 / 1288.09 |
| boot 8, stock | 1137.84 / 1136.99 / 1149.14 | 5.00 x 3 | 1260.01 / 1258.63 / 1259.53 |

Against the predictions (medians, chosen by the board lane before the run):
- rdcycle < 5: MISS. It reads exactly 5.00, the boundary; the prediction was too tight.
- getppid within ±3 %: PASS (+2.24 %).
- rdtime > 100: PASS.
- keep - stock in [60, 600]: MISS, +794 cycles.
So the keep-tags save path costs about 794 cycles per M trip on silicon, 1.7 x stock's 1,138.

The probable reason, not yet shown: this core's D$ is write-through, so each of the 31 STC is a DRAM write behind an
8-entry write buffer. The rule said the difference would be "read for what it is, not stretched to fit". The next step
is a firmware fast path for the rdtime emulation that touches two registers instead of 31. It goes into BOTH arms'
firmware, so no divergence favours either. On Line B the verdict cache's refill after every M trip comes on top of the
794 (rtl's sim: about 470 cycles over 33 ids).

## Boots 9-12: the firmware's rdtime fast path (Amendment 5)

Board 2, resident 28fcbb646, no flash, 2026-11-25 16:13-16:33Z. The kernel and initramfs are the mtrip images.
The firmware is caplifive-opensbi trap-route-keep-tags 25b14da6, built four ways: keep-tags or stock, each with
FW_FAST_RDTIME off or on. Before the boots the fast build passed the keep-tags QEMU suite: keep-timer 0, keep-call 0,
stock 152/152, the strip control 53. Read from the UART captures (`b2mtrip2-20261125T1613-boot{9..12}`). Boot 11's
image is out-fy, 3cabb79f42a9e6d7.

| Boot | Firmware | regs / strip23 / enter | rdtime (3 runs) | getppid | cycles_per_tick, backwards |
|---|---|---|---|---|---|
| 9 | keep, full path | 0 / 23 / 200 | 1940.84, 1942.97, 1950.14 | 1309.0-1310.0 | 1.99, 0 |
| 10 | stock, full path | 139 / 139 / 200 | 1151.55, 1155.39, 1139.37 | 1278.4-1279.9 | 1.99, 0 |
| 11 | keep, fast path | 0 / 23 / 200 | 119.29, 119.19, 119.31 | 1259.1-1260.0 | 1.99, 0 |
| 12 | stock, fast path | 139 / 139 / 200 | 119.66, 119.77, 119.77 | 1252.4-1252.9 | 1.99, 0 |

What this establishes:
1. **The fast path cuts an emulated rdtime from about 1,150 cycles (stock) and 1,940 (keep-tags) to about 119, on
   both firmwares.** keep minus stock on the fast path is -0.48 cycles, so keep-tags adds nothing there. Its full
   cost (about +790 cycles a trip, boots 7/8) remains on every other M trap.
2. **The values are right.** cycles_per_tick, read against the independent rdcycle counter, is 1.99 on every run of
   both paths. No read goes backwards.
3. **The trap-route contract holds under the fast path:** boot 11 reads like boot 9.
4. **Stock still loses register tags at M traps.** Boots 10 and 12 fault with cause 24 at pc 0x1021c (`sd zero,0(a0)`),
   one instruction later than boot 4's 0x10218. On 28fcbb646 the CINCOFFSET of an untagged a1 yields an integer (the
   FLU ungate), and the store through it then faults. The PREREG's "every boot: regs 0" over-applied to the stock
   arms; boot 4 had already shown this reading.
5. getppid's spread over the four boots, 1.0452, exceeds the 1.03 predicted, with the keep-full boot highest, as on
   boot 7. Recorded as read.

**Consequence for the campaign:** both arms' firmware carries FW_FAST_RDTIME=y, so neither pays the M trip for
rdtime. Plain's firmware stays stock apart from that flag.

## The final campaign Full kernel (built, not yet booted)

`build/build-kernel-final.sh`, tree `trapk/fpga-final` (a copy of the board tree, so it carries the series and the
msghdr loader; the script's fresh-tree branch applies the loader patch but has not been run from scratch). Against
the bring-up kernel it adds NO_HZ_IDLE at HZ=100 and turns RISCV_CAPSTONE_TRAP_TEST off. `config-keys-final.txt`
holds its keys. Checks:
- Every .config key compared against the Plain kernel's (`trapk/plain/linux-6.1/.config`): they differ only in the
  Capstone options and the initramfs path. NO_HZ_IDLE, HZ=100, HZ_PERIODIC off and SMP off agree.
- The test-only symbols `sublet_test_enter` and `sublet_test_fault` are absent from its vmlinux (0) and present in
  the bring-up kernel's (2). `sublet_copy_msghdr_from_user` is present (1).
- Not yet run: a boot of this kernel (twin or board) with a .dom program. Without TRAP_TEST the regs.S test cannot
  run; its first boot needs a .dom smoke program as the control.

The +Spatial twin of that kernel is `build/build-kernel-final-spatial.sh` (tree `trapk/fpga-final-spatial`): its .config
differs from the Full kernel's only in CONFIG_RISCV_CAPSTONE_SPATIAL_ONLY=y (and the initramfs path). Its smoke image,
the same overlay as boot 13, is `trapk/fpga-final-spatial/smoke-out/fw_payload.bin` b2e2c905521d2e68…, for rtl's
spatial-only seal once it exists. Not yet booted.
