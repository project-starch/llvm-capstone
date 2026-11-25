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
