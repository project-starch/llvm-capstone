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
