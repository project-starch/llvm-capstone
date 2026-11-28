# b5b2-linux-2026-11-25: the first Linux boot on the B5b-2 bitstream (pre-registration)

Status: pre-registered before any board run. The one boot reads whether the trap-route stack boots on silicon:
the integer-M firmware, the collaborator's trap-route kernel, and the board device tree carrying the Sublet node.

## What runs

| Part | Identity |
|---|---|
| bitstream | `caplifive_…-r75d-b5b2n_904cd9c6b.bit`, sha256 `36e70855a55de3692a359817b5043cc6f6f2cc92d4f41f0317df7861223a1b08` (B5b-2, CONFIRMED N=2), resident on boards 2 and 3 |
| firmware | caplifive-opensbi `trap-route-keep-tags` eb85e21c (OpenSBI 1.3.1 + `FW_CAPSTONE_KEEP_TAGS=y`), platform `fpga/ariane`, payload = the kernel below, FDT = `build/caplifive-b5b2.dts` |
| kernel | Linux 6.1 + the series at `origin/native-kernel-new-qemu` 6d4504f8b073 (`capstone/kernel/apply.sh`), the board's `fpgakernel.config` plus `RISCV_CAPSTONE_TASK`, `_TRAP`, `_TRAP_TEST`, with SMP, KSM, THP and modules off (`build/build-kernel.sh`); buildroot 2023.08 GCC 12.3 |
| device tree | the board's `caplifive.dts` plus `/capstone-unpinned` (`capstone,unpinned-v1`, id-bits 63, exact-bounds 0, page 4096, granule 16), as capstone-qemu `hw/riscv/virt.c` emits it |
| initramfs | the board's buildroot `rootfs.cpio` plus `/regs` (the collaborator's `kernel/tests/trap-route/regs.S`, unchanged), `/enter` (`build/enter.S`, the test-loader call alone, exit = its -errno), and `/init` = `build/init-diag` (their trap-route `/init` plus the `/enter` line) |

Artifact hashes: `SHA256SUMS` (the board image `fw_payload.bin` is `0e36edbb343c6703…`). Built artifacts stay on
`/mnt/hdd`; only the recipe is committed.

## Checks before the boot (done)

- **Encodings**, by dev's `capstone/tests/b5b2-illegal-scan/scan_illegal.py` (937208e0991f):
  - firmware code without its payload: ILLEGAL 0 (LDC 33, STC 63, which are legal in M);
  - `vmlinux`: only CLDRAW/CSTRAW/CPTAG (custom-0, 24) and CSMINT (1), which are legal in S; no other illegal class;
  - `/regs`, `/enter`: ILLEGAL 0.
- **Firmware contract**: the QEMU suite on the branch (keep 0/0, stock 152/152, strip control 53), and the M rules on the RTL
  (capstone-ariane b5b2-mtrap.S, efeff0dee / 1afde2023).
- **The same stack on the QEMU twin** (capstone-qemu build ≥ 9289114e06, the generic platform, the QEMU-config kernel built
  from the same series): it boots to `/init`, then:
  - `TRAPTEST: regs status 91` and `TRAPTEST: enter status 1` (the test loader returns EPERM);
  - identical with the stock firmware, so it is not the firmware's. UNRESOLVED, under investigation.
- **R-73**: `reset halt` resets the core through ndmreset, so the JTAG load is inside the covered case (dev 1e4c200c4c9b).

## Predictions, one boot

| # | Reading | Predicted | From |
|---|---|---|---|
| P1 | OpenSBI banner, then `Capstone: register tags kept across M traps, causes 24-30 delegated` | present | the firmware source |
| P2 | `Boot HART MEDELEG` | `0x000000007f00b109`: bits 0, 3, 8, 12, 13 and 15 (OpenSBI `delegate_traps`, no H) plus 24-30, under the RTL's mask `0x7f00b1ff` (rtl's reading of medeleg written -1) | `lib/sbi/sbi_hart.c`; b5b2-mtrap rule 4 |
| P3 | Linux reaches `Run /init as init process`, with `sublet: 63-bit Sublet format, exact bounds 0` and `captag: bulk=0` before it | present | the QEMU twin's lines |
| P4 | `TRAPTEST: up`, then `regs status 91`, `enter status 1`, `loop alive`, `done` | as on the twin | the twin's run; a known failure, not a pass |

**Classification.** P1-P3 are the boot's purpose. P4 must match the twin. A different P4 reading is a finding about silicon
against the twin, attributed to nothing until it is read.

**Void conditions:**
- no OpenSBI output after the load: the load or firmware, so the boot is VOID for P2-P4;
- a banner but no Linux banner: the payload, FDT or kernel;
- a kernel oops or trap before `/init`: read its cause and pc, and P4 is VOID.

After `done`, `/init` calls `poweroff -f`, which panics ("Attempted to kill init") on the twin. On the board it is the
end of the run, not a reading.

**Budget:** one boot on board 2 or 3, no flash. The upload is about 17 MB (about 135 s at about 130 KiB/s); the run is
bounded at 900 s after the load.

## Amendment 1 (2026-11-25, after boot 1, before boot 2)

**Boot 1** (board 2, booking 1501, image `0e36edbb…`):
- P1, P2 and P3 read as predicted: the banner line; `Boot HART MEDELEG 0x000000007f00b109`; Linux to
  `Run /init as init process` at 19.47 s, with the sublet and captag lines.
- P4: NO READING. Just before `/init` the kernel printed `Warning: unable to open an initial console`.
  - The initramfs had no `/dev/console`: the recipe extracted the board's rootfs.cpio as a non-root user, and that
    cannot create the node. So `/init` ran with no fds 0/1/2.
  - The twin had a console because its kernel's INITRAMFS_SOURCE is empty: the default list creates the node, and the
    initrd is unpacked over it.
  - This is a defect of the image, not a reading.

**Boot 2's image:**
- The initramfs is the board's rootfs.cpio, copied unchanged as `base-rootfs.cpio` (it carries `dev/console` c 5,1,
  root), concatenated with `overlay.cpio`. The overlay holds `/init`, `/regs` and `/enter`, owned by root (`cpio -R 0:0`);
  the kernel unpacks both archives in order.
- The embedded copy equals the source: `usr/initramfs_inc_data` gunzips to `initramfs.cpio`.
- bootargs (DTS chosen) = `earlycon console=ttyS0,57600 rdinit=/init sublet.dom_binfmt=1`, aligned with the twin.
  The twin's `panic=-1` is left out, so the board doesn't reboot.
- New hashes in `SHA256SUMS`: the board image `fw_payload.bin` is `54d11b896b2a59e3…`.

**Predictions:**
- P1 and P2 are unchanged.
- P3 is unchanged, plus `sublet: .dom loader registered`, which boot 1 printed as "off" without the parameter.
- P4 is unchanged: as on the twin, `regs status 91` and `enter status 1`.

## Amendment 2 (2026-11-25, after boot 2, before boots 3 and 4)

**Boot 2** (board 2, booking 1502, image `54d11b89…`; board/meas-pw 819ab44207c6):
- P1, P2 and P3 read as predicted, including `sublet: .dom loader registered`; `/init` started at 19.49 s.
- P4 MISSED its twin-based prediction. On silicon: `regs status 0` (every register kept its reference), with the
  kernel's `sublet: test loader: pid 26 protected`, then `enter status 200`, `loop alive` and `done`.

**The twin, not the silicon, was wrong.** The twin's image repacked the rootfs as uid 1000, which made the setuid
`bin/busybox` owned by uid 1000 (`-rwsr-xr-x <user> <user>`, against `root root` in the original). So /init's shell
ran with euid 1000, and `capable(CAP_SYS_ADMIN)` refused the test loader with EPERM. Two things follow:
- **Retracted:** the claim in this PREREG's pre-boot checks that the EPERM is "a kernel or test issue, not the
  firmware's". It was the harness's.
- **Withdrawn:** "a known failure" as P4's basis. It was a defective twin image.
The board images were never repacked; they are the rootfs unchanged, with a root-owned overlay (Amendment 1).

**What boot 2 cannot say:** whether regs.S's lost-register checks can fire on silicon. Neither platform has
exercised that failing path, so boots 3 and 4 add the controls. Both use one kernel, Image `686d255fb841…`, whose
initramfs adds `/regs-strip23`: the collaborator's regs.S unchanged except one line, `xori x23, x23, 0`, before
its dump, which makes s7 an integer (`build/regs-strip23.S`, ILLEGAL 0 by the scan). `/init` = `build/init-diag3`.
Hashes are in `SHA256SUMS.boot34`.

| Boot | Firmware | Image | Predicted |
|---|---|---|---|
| 3 | keep-tags (eb85e21c, `FW_CAPSTONE_KEEP_TAGS=y`) | `77de789fdd00…` | P1-P3 as boot 2; `regs status 0`; `regs-strip23 status 23` (the check fires on x23); `enter status 200`; `loop alive`; `done` |
| 4 | the same tree built with the flag OFF (byte-identical to stock OpenSBI 1.3.1, checked on the generic platform) | `c1698787d831…` | no `Capstone:` banner line; `Boot HART MEDELEG 0x000000000000b109` (no 24-30); Linux to `/init`; `regs status` NOT 0 |

For boot 4, any of the following counts as the firmware's contract mattering on silicon:
- a lost-register code 1..31;
- 100 + c (the test hook's reference faulted);
- a signal-kill status;
- a kernel failure after `/init`.
`regs status 0` under the stock firmware refutes the claim that keep-tags is needed for this test on silicon, or
shows that no M trap landed while references were held. The 2 x 40M-iteration spins span many machine-timer ticks, so
the second reading would itself need explaining. Boot 3 runs first. If its strip control does not read 23, boot 4 is
not run.

## Amendment 3 (2026-11-25): boot 6, the same stack on the 88db40027 seal

The 88db40027 seal (capstone-ariane: 4(b) interruptible CREVOKE, the committed-store-queue drain and the CAP_WB guard,
on 904cd9c6b; sha256 bdff45d0…) is resident on board 2. rtl's crevoke4b kit read 16/16 on it (board lane, 09:33Z).

**Boot 6:** board 2, no flash, image `fw3y` `77de789fdd00…`, byte-identical to boot 3's.
**Predicted:** exactly boot 3's readings: P1-P3; `regs status 0`; `regs-strip23 status 23`; `enter status 200`; `done`.
**Classification:**
- any other reading is a finding about 88db40027 against 904cd9c6b;
- the board differs from boot 3 (board 2, as boot 3), so the pair is matched in board and image and differs only in
  the bitstream.

## Amendment 4 (2026-11-25): boots 7 and 8, the cost of an M round trip from U

**The question.** The board has no time CSR. Its OpenSBI banner reads `Boot HART ISA Extensions : none`, where the twin
reads `time`. So every rdtime, in the kernel's sched_clock and in user code, traps to M for emulation. On Line B each
such trap also runs the keep-tags save path (31 STC, a compare and 31 LDC) and empties the verdict cache (rtl,
b5b2-srvcost). This pair reads the firmware's part on silicon. The verdict refill is rtl's sim reading.

**Program.** `/mtrip` (`build/mtrip.c`, static RISC-V, buildroot GCC 12.3, -O2, ILLEGAL 0). It reports cycles per op,
best of 5 repeats of 20,000 ops each, for:
- rdtime: one M trip each;
- rdcycle: no trap, the control;
- getppid: a U-to-S syscall, no M, the second control.
`/init` (`build/init-mtrip`) runs it three times.

**Images** (`SHA256SUMS.mtrip`): the same kernel and initramfs; only the firmware differs.
- boot 7: keep-tags, `870ca3f2…`;
- boot 8: the same tree with the flag off, `248c932a…`.

**Board:** whichever of 2/3 is free, both boots on that one board, no flash. The record names the resident bitstream.

**Predictions:**
- rdcycle under 5 cycles on both;
- getppid equal on both firmwares within ±3 % (no M on that path);
- rdtime above 100 cycles on both: a full trap, OpenSBI's C emulation and mret;
- rdtime(keep) - rdtime(stock) between 60 and 600 cycles: the keep-tags entry and exit, 62 capability memory ops plus
  31 compares, against stock's 31 sd and 31 ld.
- A difference outside that band is read for what it is, not stretched to fit.

## Amendment 5 (2026-11-25): boots 9-12, the rdtime fast path (caplifive-opensbi 25b14da6)

**Change.** `FW_FAST_RDTIME`: the trap entry answers `csrrs rd, time, x0` from mtime without the C part. It touches
t0-t2 and sp, and the flag is independent of keep-tags, so both arms get it. With both flags off the builds are
byte-identical to the base and to eb85e21c (generic fw_jump.bin hashes).

**Program.** `/mtrip` now also prints `MTRIP-TIME backwards <n> cycles_per_tick <x> ticks <t>`: rdtime answers that go
backwards, and the core cycles per 1 MHz mtime tick over the last loop. `/init` = `build/init-mtrip2`: regs, its strip
control and the loader first, then mtrip three times.

**Images** (`SHA256SUMS.mtrip2`): the same kernel and initramfs; only the firmware differs.
- boot 9: keep-tags `4dd09b4e…`
- boot 10: stock `269bd80a…`
- boot 11: keep-tags + fast `3cabb79f…`
- boot 12: stock + fast `d217c7e3…`
They run on one board, in the order 9, 10, 11, 12, no flash; the record names the resident.

**Predictions:**
- every boot: `regs status 0`, `regs-strip23 status 23`, `enter status 200`. The fast path must keep the trap-route
  contract (boot 11 is the one that tests it);
- `backwards 0` on every run, and cycles_per_tick equal across the four boots within ±1 %. The fast path's answers are
  mtime itself;
- boots 9/10 rdtime within ±3 % of boots 7/8's 1,932 / 1,138 (the same firmware);
- boots 11/12: rdtime below 600 cycles on both; keep-fast minus stock-fast below 200 (three STC/LDC pairs against
  three sd/ld);
- getppid within ±3 % across all four. The fast path is not on its path.
A reading outside these is reported as read.

**Note:** the QEMU keep-tags suite with FW_FAST_RDTIME=y is queued on the lock and not yet read. The twin has a time CSR, so it cannot exercise the fast path anyway; on silicon the fast path is checked by these boots' own regs/strip/enter controls and the MTRIP-TIME check.

## Amendment 6 (2026-11-26, before the boot): boot 13, the first boot of the final campaign Full kernel

**Image** `trapk/fpga-final/smoke-out/fw_payload.bin`, sha256 798c53e0086b6fa3037165ae420cf821ec52549a296fa4f21fecf71388b488ae,
17,467,912 B. Its parts:
- kernel: `build/build-kernel-final.sh` (NO_HZ_IDLE at HZ=100, TRAP_TEST off, the msghdr loader in), Image 1ab7b4c6…;
- firmware: keep-tags with FW_FAST_RDTIME=y, caplifive-opensbi 25b14da6. The fast path is present: two
  `fw_fast_mtime_addr` references in fw_payload.elf, as in boot 11's out-fy, against 0 in out2-y;
- DTB caplifive-b5b2.dtb;
- overlay: `build/init-final-smoke`, smoke.dom (the SPEC kits' control, byte-identical to g2b's), mtrip (boots 9-12's).
The quiesced-state shell commands of the earlier inits are not repeated here.

**Board and seal:** b42c8c644, on whichever board runs it first (board 2's session, as a conditional step after
nested's boots, or board 3's late branch). Bound 900 s.

**Predictions:**
1. The control prints `ok` and `TRAPTEST: control smoke.dom rc=7`, as on every SPEC kit boot.
2. mtrip, three runs:
   - rdtime 100-140 cycles per op. Boot 11 (keep + fast) read 119.2-119.3, and neither the tick nor TRAP_TEST is on
     rdtime's path;
   - rdcycle 5.00-5.2;
   - getppid within ±3 % of boot 11's 1,259.3 (1,221-1,297). TRAP_TEST off removes one compare and branch from the
     native syscall path, a few cycles;
   - MTRIP-TIME cycles_per_tick 1.99 and backwards 0 on every run.
3. No `sublet: fault` or `sublet: call` line.
4. `TRAPTEST: done`.
A failure of 1 or 4 voids the kernel for the campaign until it is explained; a reading outside 2 is reported as read.

## Amendment 7 (2026-11-26, before the boot): boot 14, the +Spatial final kernel on the spatial-only seal

**Image** `trapk/fpga-final-spatial/smoke2-out/fw_payload.bin`, sha256
c975128ba1f22f570f1418c8453bff828bde78d9a3d8039207353c120a390be1, 17,467,912 B. Its parts:
- kernel: `build/build-kernel-final-spatial.sh` (the Full final kernel of Amendment 6 plus
  CONFIG_RISCV_CAPSTONE_SPATIAL_ONLY=y; the two configs differ only in that key and the initramfs path),
  Image 2d48cd2d358014c5…;
- firmware: keep-tags with FW_FAST_RDTIME=y, caplifive-opensbi 25b14da6; two `fw_fast_mtime_addr` references in
  fw_payload.elf, as in boot 13's image. (A first build of this image without `FW_EXTRA=FW_FAST_RDTIME=y` had 0 and
  was discarded before any use; the payload script now prints the count and warns on 0.)
- DTB caplifive-b5b2.dtb;
- overlay: `build/init-final-spatial-smoke` (sha256 eaf6d6d7…), smoke.dom and mtrip (byte-identical to boot 13's),
  msgt.dom 2c6c30f7… and msgt.native cba6b00a… (the msghdr-loader directed test, byte-identical to the twin pair's).
  Order: control, mtrip three times, msgt native, msgt protected last.
- Not b2e2c905 (`smoke-out`, f9a0c520ac32): same kernel config, no msgt.

**Board and seal:** board 1, rtl's spatial-only seal 14f2ef81a (b42c8c644 plus 5ff493d0a and a lint baseline;
b42c8c644 is an ancestor), after rtl's kit boot in the same session. Bound 900 s.

**Twin check before the boot** (capstone-qemu sublet/spatial-only with x-capstone-spatial-only=on, firmware keep-tags
fw-y, the same /init and files; `trapk/run-sfin-*`), a matched pair that differs only in the kernel:
- +Spatial final Image: `ok`, control rc=7; mtrip three runs rc 0, backwards 0; msgt native and protected both rc 0,
  `fails=0`, the five MSGT lines identical; no `sublet: fault`/`call` line; `TRAPTEST: done`.
- Full final Image (boot 13's): `sublet: R reads back as type 1, id 0x3 (no record), and CSMINT answered 0x3`, then
  `load failed after the point of no return: -95` and control rc=139; the same for msgt protected (rc 139); native
  msgt rc 0. So the control itself reaches the spatial-only root check: the option is what lets any protected
  program start on this processor, and msgt adds the sendmsg path on top.
The twin's mtrip cycle counts are QEMU's and are not predictions.

**Predictions:**
1. The control prints `ok` and `TRAPTEST: control smoke.dom rc=7`, with no `R reads back` line.
2. mtrip, three runs, as Amendment 6's bands (rdtime 100-140, rdcycle 5.00-5.2, getppid 1,221-1,297, MTRIP-TIME
   cycles_per_tick 1.99 and backwards 0). Nothing on these paths touches the lifetime unit, so the spatial-only core
   is not predicted to move them; a reading outside is reported as read.
3. msgt native and protected: rc 0, `MSGT done fails=0`, and the five MSGT lines identical to the twin's.
4. No `sublet: fault`, `sublet: call` or `R reads back` line.
5. `TRAPTEST: done`.

**Reading.** A failure of 1, 3 or 5 voids the +Spatial kernel for the campaign until explained. Boot 13 (the Full
final kernel, board 2) may not yet be graded when this boot runs. If it is not and this boot fails 1 or 5, the cause
is UNDECIDED between the shared kernel base and the spatial-only option until boot 13 reads. A pass here is
evidence about the shared base too, but it does not replace boot 13 for the Full arm.
