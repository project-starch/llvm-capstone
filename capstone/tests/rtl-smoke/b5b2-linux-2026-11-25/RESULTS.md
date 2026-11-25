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
