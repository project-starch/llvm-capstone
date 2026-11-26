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
3. Keep-tags is needed for regs.S on 904cd9c6b. Under the flag-off firmware, the same payload faulted: byte-identical
   from offset 0x200000, so the kernel and initramfs are the same, and the 82,425 differing bytes all lie in the
   firmware region. a1 lost its tag (tval = its cursor 0x3f8b9eb000, non-zero: ex_stage.sv:555-563 puts the rs1 cursor
   in tval) at the first capability use after the timer spin. The stock arm is N=1 boot (two programs); keep-tags is
   N=3 boots (boots 2 and 3 here, boot 6 on 88db40027). Which M trap stripped the tag is inferred.
   The firmware also changes MEDELEG, but delegation decides who handles a cause; it cannot untag a register with its
   cursor intact, which only the stock save path predicts. (Correction 2026-11-25, after an adversarial audit: this
   item first read "is necessary for it", without the counts.)
   On disk, `linux-6.1/arch/riscv/boot/Image` has since been relinked for later images, so SHA256SUMS.boot34's Image
   line no longer matches the file. The two payloads still verify.

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

## Boot 13: the final campaign Full kernel on b42c8c644 (board 2, 2026-11-25 22:52-22:56Z), 4 of 4 as predicted

Pre-registered as Amendment 6 (8caa259c163c). Image 798c53e0…, which the board lane checked against its arming file.
The seal read back by name as `…_b42c8c644.bit` before and after the boot. The result lines are on board/meas-pw at
f47b2a686ce2 (`b5b2-seals-board-2026-11-25/b42c8c644-board2.result-lines.txt`, section boot13).

| Prediction | Reading |
|---|---|
| 1. control `ok`, smoke.dom rc=7 | `okTRAPTEST: control smoke.dom rc=7` |
| 2. mtrip x3: rdtime 100-140, rdcycle 5.00-5.2, getppid 1,221-1,297, cycles_per_tick 1.99, backwards 0 | rdtime 119.70-119.72, rdcycle 5.00, getppid 1,266.94-1,267.95, cycles_per_tick 1.99, backwards 0, rc 0 on all three |
| 3. no `sublet: fault` or `call` line | none |
| 4. `TRAPTEST: done` | present |

- rdtime matches boot 11's keep + fast reading (119.2-119.3): the fast path is in the firmware.
- getppid is 0.6 % above boot 11's 1,259.3 and inside the band. Prediction 2 expected TRAP_TEST off to remove a few
  cycles, and this reading does not show it. One reading, so the difference is not resolved.
- The kernel prints neither CONFIG_HZ nor NO_HZ_IDLE, so the boot does not show the tick configuration. That comes
  from the build's .config (`config-keys-final.txt`).

**Consequence:** the Full kernel stands for the campaign (Amendment 6's void condition did not fire). Boot 14 (the
+Spatial kernel on 14f2ef81a, board 1) attempt 1 was NO READING for infrastructure reasons (a GDB timeout after a
completed load, booking 1566), and it is still owed.

## Boot 14: the final +Spatial kernel on the spatial-only seal 14f2ef81a (board 1, 2026-11-26 05:32-05:36Z), as predicted

Pre-registered as Amendment 7 (4447a1b8cd1b). Boot 13, which had been unread when this boot was registered, read 4
of 4 first. Image c975128b…, the smoke2 build with FW_FAST_RDTIME.
- The seal: 14f2ef81a was flashed by the wapflu kit at 05:28-05:30Z. Board 1 had P1 resident, left by nested.
  - The kit read WFKIT 1/1 AS PREDICTED.
  - The bitstream read back by name as `…-spatial_14f2ef81a.bit` after the flash, after the kit's loads, and before
    and after this boot.
- The capture: UART sha16 a438f4e24665e144, 9,235 characters, bookings 1577 (flash and kit) and 1578 (boot). The
  readings below are the board lane's report; its result lines will be on board/meas-pw.
- Attempt 1 (booking 1566) was NO READING: a GDB timeout after a completed load.

| Prediction | Reading |
|---|---|
| 1. control `ok`, smoke.dom rc=7 (binding) | `okTRAPTEST: control smoke.dom rc=7`, no R-reads-back line |
| 2. mtrip in Amendment 6's bands | rdtime 119.71-119.76, rdcycle 5.00, getppid 1,269.06-1,269.66, cycles_per_tick 1.99, backwards 0, all three runs |
| 3. msgt native and protected rc 0, fails=0, the five lines (binding) | both rc 0, fails=0, all five MSGT lines equal to the twin's |
| 4. no `sublet: fault`, `call` or "R reads back" line | none |
| 5. `TRAPTEST: done` (binding) | present |

- The spatial-only path is used. Prediction 3's protected msgt is the step whose root check accepts R without its
  record. On the spatial-only twin the Full kernel refuses R (the twin pair of e247b4fdd995).
  That refusal has not been read on silicon. Here the step passes with fails=0.
- rdtime and getppid match boot 13's Full kernel on b42c8c644 within 0.2 % (119.70-119.72, 1,266.9-1,268.0).

**Consequence:** the +Spatial final kernel stands for the campaign's +Spatial arm. Both final kernels are now read
on silicon, on the seals they pair with.
