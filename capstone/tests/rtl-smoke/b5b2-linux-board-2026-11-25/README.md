# b5b2-linux-2026-11-25 on board 2: the board lane's run record

This folder records the first Linux boot on the B5b-2 bitstream. The runtime lane's pre-registration is
`capstone/tests/rtl-smoke/b5b2-linux-2026-11-25/PREREG.md` on branch runtime/b5b2-linux, at c3a45f1e9904.

The run, 2026-11-25:
- Board 2, hub booking 1501, from 03:45 to 04:03 UTC.
- The resident `caplifive_…-r75d-b5b2n_904cd9c6b.bit` was read back by name before and after the boot. No flash.
- The image was `fw_payload.bin`, sha256 `0e36edbb343c6703…`, the PREREG's `SHA256SUMS`. The board DTB is embedded in it
  at 0x14000.

## Procedure

`run_sup_bare-linux.py` (sha256 prefix dd45d0a3a270f431) follows the cold boot of `run_ladder_perf_fpga.py`, without
the login:
1. it checks the image sha and the resident's name;
2. it uploads the image under a content-hashed name;
3. it power-cycles the board;
4. it runs `monitor reset halt`, which is R-73's covered case (dev 1e4c200c4c9b);
5. it runs `load_image` at 0x80000000 (149 s), then sets `$pc = 0x80000000` and `$a0 = 0`, then `continue`.

The capture covers only this run's own `continue`. It stops at "TRAPTEST: done" plus 20 s, or 900 s after the load.

`grade-linux.py` (a62b09041e25dfcf) grades P1-P4 on the first boot only.

Both scripts were dry-run first, and those runs are kept beside the run on `<mnt-hdd>`:
- the runner against a fake console, in four modes: done, a hang to the bound, a wrong resident, a wrong image sha;
- the grader against the QEMU twin's log. P2 MISSes there, as it must, because QEMU reads 0x7ff0b509.

The grader's negative controls also fired:
- no OpenSBI output;
- no Linux banner;
- an oops before /init;
- a changed P4 line.

## Result lines

```
P1 AS PREDICTED  OpenSBI v1.3, then "Capstone: register tags kept across M traps, causes 24-30 delegated"
P2 AS PREDICTED  Boot HART MEDELEG = 0x000000007f00b109   (MIDELEG 0x222, Platform Name ARIANE RISC-V)
P3 AS PREDICTED  "captag: bulk=0 (requires firmware S-mode CBZE permission)", "sublet: 63-bit Sublet format,
                 exact bounds 0", then "Run /init as init process" at 19.47 s; no oops or trap before /init
P4 NO READING    no TRAPTEST line and no panic line in 900 s after the load; the kernel printed
                 "Warning: unable to open an initial console" just before "Run /init"
```

The grader's own output is `result-lines.txt`.

## Why P4 has no reading (by source, not by a board reading)

`init/main.c` `console_on_rootfs()` opens `/dev/console` for fd 0, 1 and 2. When it fails, it prints the warning above,
and /init starts with no stdout, so every `echo` in it is lost.

The board kernel's `CONFIG_INITRAMFS_SOURCE` is the kit's `initramfs.cpio`. `cpio -itv` shows that archive has `dev/`
and no `dev/console` node.

The QEMU twin's kernel has `CONFIG_INITRAMFS_SOURCE=""`, so it carries the kernel's default list
(`usr/default_cpio_list`: `nod /dev/console 0600 0 0 c 5 1`), and the initrd is unpacked over it. That is why the twin
has a console and the board image does not.

So this boot says nothing about P4. Whether /init ran its tests, hung, or powered off without the panic is not
determined.

A second difference from the twin: the board's command line is its DTS bootargs, `earlycon console=ttyS0,57600`, so
the kernel prints `sublet: .dom loader off`. The twin passes `rdinit=/init panic=-1 sublet.dom_binfmt=1`, and its
kernel prints `registered`.

## Boot 2 (amendment 1, runtime/b5b2-linux 05ac6828a092)

The run:
- Board 2, hub booking 1502, from 04:34 to 04:39 UTC.
- The resident 904cd9c6b was read back by name before and after the boot. No flash.
- The image was `fw_payload.bin` 54d11b896b2a59e3. Its initramfs is `base-rootfs.cpio` followed by `overlay.cpio`,
  byte-exact. The base carries `dev/console` (c 5,1, root), and the embedded DTB passes `sublet.dom_binfmt=1`.
- The procedure was the same, with the load taking 149 s. "TRAPTEST: done" arrived 58 s after `continue`.

The grader was `grade-linux2.py` (160ae1381bac95cf), which adds the amended P3 line. Before the run it read AS
PREDICTED on the twin's log and MISS on boot 1's capture, which printed "off".

```
P1 AS PREDICTED  OpenSBI v1.3, then the keep-tags line
P2 AS PREDICTED  Boot HART MEDELEG = 0x000000007f00b109
P3 AS PREDICTED  captag bulk=0, sublet 63-bit exact bounds 0, ".dom loader registered", then Run /init at 19.49 s
P4 DIFFERS FROM THE TWIN:
                 TRAPTEST: up, kernel 6.1.0
                 [21.96] sublet: test loader: pid 26 protected, table 0x81c5e000
                 TRAPTEST: regs status 0          (twin: 91)
                 [43.57] sublet: test loader: pid 27 protected, table 0x81c58000
                 TRAPTEST: enter status 200       (twin: 1)
                 TRAPTEST: loop alive
                 TRAPTEST: done
                 [45.88] reboot: System halted    (twin: "Attempted to kill init" panic)
```

The grader's output is `result-lines-boot2.txt`.

What the exit codes mean, as the test programs' own headers define them:
- `regs.S` exits 0 when every register kept its reference, n when register xn lost its tag or lifetime, 100 + c on a
  hook fault with cause c, 90-95 on a setup failure, and 91 (`not_protected`) when the test loader refused.
- `enter.S` exits with the -errno of SUBLET_NR_TEST_ENTER, 200 when the call returned 0 with a reference in a0
  (LCC type 1), and 201 when it returned 0 or more without one.

So on silicon the test loader protected both processes, where the twin refused them with EPERM.

This is a reading, attributed to nothing. The twin's failure is listed as unresolved in the PREREG. No positive control
in this boot shows that regs.S's lost-register checks can fire on silicon: the twin exits before they run, so neither
platform has yet exercised the failing path of the check that read 0.

## Boots 3 and 4 (amendment 2, runtime/b5b2-linux 90f2f5bc5e83)

The run:
- Board 2, one hub booking, 1503, from 04:46 to 04:56 UTC.
- The resident 904cd9c6b was read back by name before each boot and after the last. No flash.
- Both boots use the same kernel, Image `686d255f…`, and the same initramfs. `/init` runs `/regs`, then `/regs-strip23`,
  then `/enter`.
- `/regs-strip23` is regs.S with one added `xori s7, s7, 0`, found once in the binary at 0x1024e (`000bcb93`).
- Boot 4 was run only because boot 3's grade printed the exact line `STRIP CONTROL 23`.

The scripts:
- `run-linux34.sh` (a9e97e38c97e1b49, kept on `<mnt-hdd>`) drove both boots and the gate.
- `grade-linux3.py` (59eef26ef8a5a82d) is the grader. Before the run it was dry-run on synthetic captures:
  - boot 3 good, and boot 3 with the strip missed;
  - boot 4 with a lost x7, with regs 0 (REFUTES), with a setup failure 92 (MISS), with signal 139, and with a kernel
    oops after /init.

  Both gate branches were also dry-run.

```
boot 3, fw3y 77de789fdd00 (keep-tags on), all AS PREDICTED:
  P1 "Capstone: register tags kept across M traps, causes 24-30 delegated"   P2 MEDELEG 0x000000007f00b109
  P3 Linux to Run /init, with the captag, sublet and loader-registered lines
  P4 regs status 0 | regs-strip23 status 23 | enter status 200 | loop alive | done   (test loader: pids 26-28 protected)
boot 4, fw3n c1698787d831 (the same tree, flag off), all AS PREDICTED:
  P1 no "Capstone:" line   P2 MEDELEG 0x000000000000b109   P3 Linux to Run /init
  P4 regs status 139 | regs-strip23 status 139 | enter status 200 | loop alive | done
     sublet: fault cause=24 pc=0x10218 ... pid=26 comm=regs          (and pid=27 comm=regs-strip23, same pc)
```

The grader's outputs are `result-lines-boot3.txt` and `result-lines-boot4.txt`.

What the readings show:
- **Boot 3.** The strip control's 23 shows that regs.S's lost-register check fires on silicon. Under keep-tags, the same
  check reads 0 for the unmodified program.
- **Boot 4, status.** Under the flag-off firmware, both programs exit 139: 128 + 11, so SIGSEGV.
- **Boot 4, the fault.** The kernel reports cause 24 at pc 0x10218 in both. In the 904cd9c6b RTL's `riscv_pkg.sv`, 24 is
  `UNEXPECTED_OPERAND_TYPE`.
  - The disassembly puts pc 0x10218 at `INC(a0, a1, a7)`: the first capability operation on a1, a reference to mapping
    B. It comes right after the first 40M-iteration spin, which runs while timer interrupts trap to M.
  - Each program's fault came about 10.6 s after its "protected" line, against about 21.6 s per program under keep-tags.

This is a reading of when and where it failed. That a1 lost its tag at an M-mode trap during the spin is the
mechanism amendment 2 predicts, and the faulting instruction is consistent with it. The board did not observe the
strip itself.

## Boots 6, 7 and 8 (Amendments 3 and 4), board 2, resident 88db40027, no flash

- **Boot 6** is fw3y 77de789fdd00, byte-identical to boot 3's image. It reads exactly as boot 3: P1-P4 AS PREDICTED;
  regs 0, regs-strip23 23, enter 200. The matched pair (boot 3 on 904cd9c6b, boot 6 on 88db40027) is identical on the
  trap-route test. `result-lines-boot6.txt`.
- **Boots 7 and 8** are the mtrip pair: keep-tags 870ca3f2 and stock 248c932a, the same kernel and initramfs.
  - Each boot printed three MTRIP lines and done, and the firmware identity checked: the keep-tags banner present on 7,
    absent on 8.
  - **2/4 as predicted.** getppid equal within ±3 % (+2.24 %); rdtime > 100 on every run.
  - **MISS:** rdcycle reads exactly 5.00 on all six runs, against a prediction of < 5.
  - **MISS:** median rdtime keep - stock = +794.49 cycles (1932.33 - 1137.84), against a band of [60, 600].
  - The medians were chosen by the board lane before the run; the PREREG names no statistic.
  - `result-lines-boot7-8.txt` holds all three raw values per op.
